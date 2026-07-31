"""Content-addressed authorization certificates for executable DSS rulebooks.

The certificate implemented here is deliberately deterministic and unkeyed.  It
detects missing evidence, accidental corruption, stale-policy use, and edits made
after a trusted build produced the certificate.  It establishes *integrity*, not
authorship: an adversary able to replace a rulebook and run this code can also
generate a new certificate.  Authenticity therefore remains the responsibility
of a signed release manifest or another external trust anchor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import hmac
import json
import math
import re
from typing import Any


CERTIFICATE_FIELD = "eligibility_certificate"
EVIDENCE_FIELD = "eligibility_evidence"
CERTIFICATE_VERSION = "tm-ecg-dss-eligibility-v1"
CERTIFICATE_ALGORITHM = "sha256"
CERTIFICATE_THREAT_MODEL = "content_integrity_not_authorship"

_REQUIRED_PROVENANCE_HASHES = frozenset(
    {
        "split_manifest_sha256",
        "model_sha256",
        "metrics_sha256",
    }
)
_CERTIFICATE_FIELDS = frozenset(
    {
        "version",
        "algorithm",
        "threat_model",
        "ontology_version",
        "policy_config_sha256",
        "provenance_hashes",
        "rulebook_content_sha256",
        "certificate_sha256",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class EligibilityCertificateError(ValueError):
    """Raised when an eligible DSS artifact lacks valid authorization evidence."""


def _json_value(value: Any, *, location: str) -> Any:
    """Normalize JSON-compatible values while rejecting ambiguous encodings."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EligibilityCertificateError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise EligibilityCertificateError(
                    f"{location} contains a non-string or empty object key"
                )
            normalized[raw_key] = _json_value(
                raw_value,
                location=f"{location}.{raw_key}",
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise EligibilityCertificateError(
        f"{location} contains unsupported value type {type(value).__name__}"
    )


def _canonical_bytes(value: Any, *, location: str) -> bytes:
    normalized = _json_value(value, location=location)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any, *, location: str) -> str:
    return hashlib.sha256(_canonical_bytes(value, location=location)).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise EligibilityCertificateError(
            f"{field} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _provenance_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    evidence = payload.get(EVIDENCE_FIELD)
    if not isinstance(evidence, Mapping):
        raise EligibilityCertificateError(
            "eligible rulebook lacks an eligibility_evidence object"
        )
    if set(evidence) != _REQUIRED_PROVENANCE_HASHES:
        missing = sorted(_REQUIRED_PROVENANCE_HASHES - set(evidence))
        unexpected = sorted(set(evidence) - _REQUIRED_PROVENANCE_HASHES)
        raise EligibilityCertificateError(
            "eligibility_evidence must contain exactly the required provenance hashes "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return {
        field: _require_sha256(evidence[field], field=f"eligibility_evidence.{field}")
        for field in sorted(_REQUIRED_PROVENANCE_HASHES)
    }


def dss_policy_config_sha256(config: Any) -> str:
    """Hash the active ontology and complete DSS policy mapping.

    Hashing the complete ``dss`` mapping binds both inference settings and strict
    selection thresholds.  Adding or changing a policy key intentionally revokes
    certificates created under the earlier configuration.
    """

    ontology_version = getattr(config, "ontology_version", None)
    if not isinstance(ontology_version, str) or not ontology_version:
        raise EligibilityCertificateError("active config lacks ontology_version")
    policy = getattr(config, "dss", None)
    if not isinstance(policy, Mapping):
        raise EligibilityCertificateError("active config.dss must be a mapping")
    return _sha256(
        {
            "ontology_version": ontology_version,
            "dss": dict(policy),
        },
        location="active_dss_policy_config",
    )


def rulebook_content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash every rulebook field except the certificate itself."""

    unsigned = dict(payload)
    unsigned.pop(CERTIFICATE_FIELD, None)
    return _sha256(unsigned, location="rulebook")


def issue_rulebook_eligibility_certificate(
    payload: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Create a deterministic integrity certificate for a rulebook plus evidence.

    Issuance only binds bytes and provenance; it does not determine clinical
    eligibility.  The runtime independently enforces executable-shape and target
    constraints after verifying the certificate.
    """

    ontology_version = payload.get("ontology_version")
    if not isinstance(ontology_version, str) or not ontology_version:
        raise EligibilityCertificateError("rulebook lacks ontology_version")
    active_ontology = getattr(config, "ontology_version", None)
    if ontology_version != active_ontology:
        raise EligibilityCertificateError(
            "rulebook ontology_version does not match active config"
        )
    certificate: dict[str, Any] = {
        "version": CERTIFICATE_VERSION,
        "algorithm": CERTIFICATE_ALGORITHM,
        "threat_model": CERTIFICATE_THREAT_MODEL,
        "ontology_version": ontology_version,
        "policy_config_sha256": dss_policy_config_sha256(config),
        "provenance_hashes": _provenance_hashes(payload),
        "rulebook_content_sha256": rulebook_content_sha256(payload),
    }
    certificate["certificate_sha256"] = _sha256(
        certificate,
        location="eligibility_certificate",
    )
    return certificate


def attach_rulebook_eligibility_certificate(
    payload: Mapping[str, Any],
    config: Any,
    *,
    split_manifest_sha256: str,
    model_sha256: str,
    metrics_sha256: str,
) -> dict[str, Any]:
    """Return a certified copy without mutating the caller's rulebook."""

    certified = deepcopy(dict(payload))
    certified[EVIDENCE_FIELD] = {
        "split_manifest_sha256": split_manifest_sha256,
        "model_sha256": model_sha256,
        "metrics_sha256": metrics_sha256,
    }
    certified[CERTIFICATE_FIELD] = issue_rulebook_eligibility_certificate(
        certified,
        config,
    )
    return certified


def verify_rulebook_eligibility_certificate(
    payload: Mapping[str, Any],
    config: Any,
) -> None:
    """Verify certificate, content, policy, ontology, and provenance bindings."""

    raw_certificate = payload.get(CERTIFICATE_FIELD)
    if not isinstance(raw_certificate, Mapping):
        raise EligibilityCertificateError(
            "eligible rulebook lacks an eligibility_certificate object"
        )
    certificate = dict(raw_certificate)
    if set(certificate) != _CERTIFICATE_FIELDS:
        missing = sorted(_CERTIFICATE_FIELDS - set(certificate))
        unexpected = sorted(set(certificate) - _CERTIFICATE_FIELDS)
        raise EligibilityCertificateError(
            "eligibility_certificate has an invalid field set "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if certificate["version"] != CERTIFICATE_VERSION:
        raise EligibilityCertificateError("unsupported eligibility certificate version")
    if certificate["algorithm"] != CERTIFICATE_ALGORITHM:
        raise EligibilityCertificateError("unsupported eligibility certificate algorithm")
    if certificate["threat_model"] != CERTIFICATE_THREAT_MODEL:
        raise EligibilityCertificateError("eligibility certificate threat model mismatch")

    supplied_certificate_hash = _require_sha256(
        certificate["certificate_sha256"],
        field="eligibility_certificate.certificate_sha256",
    )
    unsigned_certificate = dict(certificate)
    unsigned_certificate.pop("certificate_sha256")
    expected_certificate_hash = _sha256(
        unsigned_certificate,
        location="eligibility_certificate",
    )
    if not hmac.compare_digest(
        supplied_certificate_hash,
        expected_certificate_hash,
    ):
        raise EligibilityCertificateError("eligibility certificate self-hash mismatch")

    ontology_version = payload.get("ontology_version")
    active_ontology = getattr(config, "ontology_version", None)
    if (
        not isinstance(ontology_version, str)
        or certificate["ontology_version"] != ontology_version
        or ontology_version != active_ontology
    ):
        raise EligibilityCertificateError(
            "eligibility certificate ontology binding mismatch"
        )

    supplied_policy_hash = _require_sha256(
        certificate["policy_config_sha256"],
        field="eligibility_certificate.policy_config_sha256",
    )
    if not hmac.compare_digest(
        supplied_policy_hash,
        dss_policy_config_sha256(config),
    ):
        raise EligibilityCertificateError(
            "eligibility certificate does not match the active DSS policy/config"
        )

    evidence_hashes = _provenance_hashes(payload)
    certificate_provenance = certificate["provenance_hashes"]
    if not isinstance(certificate_provenance, Mapping):
        raise EligibilityCertificateError(
            "eligibility certificate provenance_hashes must be an object"
        )
    normalized_certificate_provenance = {
        str(field): _require_sha256(
            value,
            field=f"eligibility_certificate.provenance_hashes.{field}",
        )
        for field, value in certificate_provenance.items()
    }
    if normalized_certificate_provenance != evidence_hashes:
        raise EligibilityCertificateError(
            "eligibility certificate provenance binding mismatch"
        )

    supplied_rulebook_hash = _require_sha256(
        certificate["rulebook_content_sha256"],
        field="eligibility_certificate.rulebook_content_sha256",
    )
    if not hmac.compare_digest(
        supplied_rulebook_hash,
        rulebook_content_sha256(payload),
    ):
        raise EligibilityCertificateError("rulebook content hash mismatch")
