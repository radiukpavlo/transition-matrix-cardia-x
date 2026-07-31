from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tm_ecg.dss.eligibility import attach_rulebook_eligibility_certificate
from tm_ecg.inference import (
    _MEASURED_DSS_QUALITY_FEATURES,
    _infer_with_rulebook,
    _measure_dss_quality_features,
    _validate_transition_artifacts,
)
from tm_ecg.io.common import sha256_file


def _config(version: str = "test-v3") -> SimpleNamespace:
    return SimpleNamespace(
        ontology_version=version,
        dss={
            "soft_match_min_fraction": 0.60,
            "top_k": 5,
            "close_vote_ratio": 1.10,
            "low_strength_threshold": 0.20,
            "contradiction_penalty": 0.25,
            "hard_veto_criticality": 1.5,
        },
    )


def _eligible_rulebook() -> dict[str, object]:
    return {
        "status": "eligible",
        "rules_eligible": True,
        "ontology_version": "test-v3",
        "selected_features": ["qrs_dur_med_ms"],
        "discretization_plan": {
            "feature_domains": {
                "qrs_dur_med_ms": {
                    "name": "qrs_dur_med_ms",
                    "value_type": "continuous",
                    "unit": "ms",
                    "family": "qrs",
                    "bins": [
                        {
                            "code": 0,
                            "label": "not_wide",
                            "lower": None,
                            "upper": 120.0,
                            "include_lower": True,
                            "include_upper": False,
                        },
                        {
                            "code": 1,
                            "label": "wide_qrs",
                            "lower": 120.0,
                            "upper": None,
                            "include_lower": True,
                            "include_upper": False,
                        },
                    ],
                    "allowed_states": [],
                    "missing_state": "missing",
                    "clinical_priority": 1.0,
                    "provenance": "test",
                }
            },
            "thresholds": [],
            "class_labels": ["RBBB spectrum"],
            "alpha": 0.1,
            "min_support": 1,
            "max_depth": 1,
            "orientation": "B_hat = A @ T",
        },
        "medical_predicates": {
            "RBBB spectrum": {
                "label": "RBBB spectrum",
                "required": [
                    {"feature": "qrs_dur_med_ms", "state": "wide_qrs"}
                ],
                "supportive": [],
                "contraindications": [],
            }
        },
        "production_rules": [
            {
                "rule_id": "R1",
                "target_label": "RBBB spectrum",
                "antecedents": [
                    {"feature": "qrs_dur_med_ms", "state": "wide_qrs"}
                ],
                "confidence": 1.0,
                "support_count": 20,
                "support_fraction": 0.2,
                "covered_object_ids": ["r1"],
                "class_distribution": {"RBBB spectrum": 20},
                "oof_precision": 0.95,
                "oof_recall": 0.90,
                "class_prior": 0.1,
                "calibration_uncertainty": 0.01,
                "fold_stability": 0.95,
            }
        ],
    }


def _certify(
    rulebook: dict[str, object],
    config: SimpleNamespace | None = None,
) -> dict[str, object]:
    return attach_rulebook_eligibility_certificate(
        rulebook,
        config or _config(),
        split_manifest_sha256="1" * 64,
        model_sha256="2" * 64,
        metrics_sha256="3" * 64,
    )


def test_ineligible_rulebook_fails_closed() -> None:
    result = _infer_with_rulebook(
        {"feature": 10.0},
        {
            "status": "ineligible_no_rulebook",
            "rules_eligible": False,
            "ontology_version": "test-v3",
        },
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "dss_rulebook_ineligible"
    assert result["activated_rules"] == []


def test_ontology_mismatch_fails_closed() -> None:
    result = _infer_with_rulebook(
        {},
        {
            "status": "eligible",
            "rules_eligible": True,
            "ontology_version": "old",
        },
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "dss_ontology_mismatch"


def test_forged_eligibility_flags_without_certificate_fail_closed() -> None:
    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        _eligible_rulebook(),
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "dss_eligibility_certificate_invalid"
    assert "lacks an eligibility_certificate" in result["explanation"]


def test_tampered_certified_rulebook_content_fails_closed() -> None:
    rulebook = _certify(_eligible_rulebook())
    production_rules = rulebook["production_rules"]
    assert isinstance(production_rules, list)
    first_rule = production_rules[0]
    assert isinstance(first_rule, dict)
    first_rule["confidence"] = 0.01

    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        rulebook,
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "dss_eligibility_certificate_invalid"
    assert "content hash mismatch" in result["explanation"]


def test_tampered_certificate_self_hash_fails_closed() -> None:
    rulebook = _certify(_eligible_rulebook())
    certificate = rulebook["eligibility_certificate"]
    assert isinstance(certificate, dict)
    certificate["certificate_sha256"] = "0" * 64

    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        rulebook,
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "dss_eligibility_certificate_invalid"
    assert "self-hash mismatch" in result["explanation"]


def test_certificate_fails_closed_under_different_dss_policy() -> None:
    authoring_config = _config()
    rulebook = _certify(_eligible_rulebook(), authoring_config)
    runtime_config = _config()
    runtime_config.dss["close_vote_ratio"] = 1.25

    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        rulebook,
        runtime_config,
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "dss_eligibility_certificate_invalid"
    assert "active DSS policy/config" in result["explanation"]


def test_tampered_provenance_hash_fails_closed() -> None:
    rulebook = _certify(_eligible_rulebook())
    evidence = rulebook["eligibility_evidence"]
    assert isinstance(evidence, dict)
    evidence["metrics_sha256"] = "4" * 64

    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        rulebook,
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "dss_eligibility_certificate_invalid"


def test_eligible_rulebook_quantizes_numeric_features_before_matching() -> None:
    rulebook = _certify(_eligible_rulebook())

    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        rulebook,
        _config(),
    )

    assert result["predicted_label"] == "RBBB spectrum"
    assert result["activated_rules"][0]["rule_id"] == "R1"


@pytest.mark.parametrize(
    ("field", "empty_value"),
    [
        ("selected_features", []),
        ("medical_predicates", {}),
        ("production_rules", []),
    ],
)
def test_certified_rulebook_with_empty_executable_component_fails_closed(
    field: str,
    empty_value: object,
) -> None:
    rulebook = _eligible_rulebook()
    rulebook[field] = empty_value
    certified = _certify(rulebook)

    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        certified,
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "invalid_dss_rulebook"


def test_certified_rulebook_with_unknown_target_fails_closed() -> None:
    rulebook = deepcopy(_eligible_rulebook())
    plan = rulebook["discretization_plan"]
    assert isinstance(plan, dict)
    plan["class_labels"] = ["Invented rhythm"]
    predicates = rulebook["medical_predicates"]
    assert isinstance(predicates, dict)
    predicates["Invented rhythm"] = predicates.pop("RBBB spectrum")
    invented_predicate = predicates["Invented rhythm"]
    assert isinstance(invented_predicate, dict)
    invented_predicate["label"] = "Invented rhythm"
    rules = rulebook["production_rules"]
    assert isinstance(rules, list)
    first_rule = rules[0]
    assert isinstance(first_rule, dict)
    first_rule["target_label"] = "Invented rhythm"
    certified = _certify(rulebook)

    result = _infer_with_rulebook(
        {"qrs_dur_med_ms": 145.0},
        certified,
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "invalid_dss_rulebook"
    assert "unknown targets" in result["explanation"]


def test_malformed_eligible_rulebook_fails_closed() -> None:
    rulebook = _certify(
        {
            "status": "eligible",
            "rules_eligible": True,
            "ontology_version": "test-v3",
            "discretization_plan": {},
            "production_rules": [],
        }
    )
    result = _infer_with_rulebook(
        {},
        rulebook,
        _config(),
    )

    assert result["predicted_label"] is None
    assert result["abstention"]["reason"] == "invalid_dss_rulebook"


def test_rulebook_schema_requires_certificate_only_for_eligible_artifacts() -> None:
    from jsonschema import ValidationError, validate

    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "dss_rulebook_schema.json").read_text(
            encoding="utf-8"
        )
    )
    rulebook = _eligible_rulebook()
    rulebook.update(
        {
            "dataset": "b1",
            "orientation": "B_hat = A @ T",
        }
    )
    certified = _certify(rulebook)
    validate(certified, schema)

    forged = deepcopy(certified)
    forged.pop("eligibility_certificate")
    with pytest.raises(ValidationError):
        validate(forged, schema)


def test_waveform_quality_measurement_failure_clears_all_quality_guards(
    monkeypatch,
) -> None:
    import tm_ecg.real_data

    def fail_measurement(*_args, **_kwargs):
        raise ValueError("synthetic delineation failure")

    monkeypatch.setattr(
        tm_ecg.real_data,
        "_one_record_measurements",
        fail_measurement,
    )
    quality, provenance = _measure_dss_quality_features(
        signal=object(),
        fs=500.0,
        sig_names=["II"],
        config=SimpleNamespace(thresholds={}),
    )

    assert set(quality) == set(_MEASURED_DSS_QUALITY_FEATURES)
    assert all(value is None for value in quality.values())
    assert provenance["status"] == "unavailable_fail_closed"


def test_transition_inference_artifacts_require_matching_hashes(tmp_path) -> None:
    operator_path = tmp_path / "B1_T_ridge.json"
    bundle_path = tmp_path / "B1_transform_bundle.json"
    a_bundle_path = tmp_path / "B1_A_preprocess_bundle.json"
    operator_path.write_text("{}", encoding="utf-8")
    bundle_path.write_text("{}", encoding="utf-8")
    a_bundle_path.write_text("{}", encoding="utf-8")
    metadata_path = tmp_path / "B1_operator_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_version": 2,
                "ontology_version": "test-v3",
                "legacy_operator_sha256": sha256_file(operator_path),
                "transform_bundle_sha256": sha256_file(bundle_path),
                "a_preprocess_bundle": str(a_bundle_path),
                "a_preprocess_bundle_sha256": sha256_file(a_bundle_path),
            }
        ),
        encoding="utf-8",
    )

    metadata = _validate_transition_artifacts(
        _config(), operator_path, bundle_path
    )
    assert metadata["artifact_version"] == 2

    bundle_path.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="transform-bundle hash mismatch"):
        _validate_transition_artifacts(_config(), operator_path, bundle_path)
