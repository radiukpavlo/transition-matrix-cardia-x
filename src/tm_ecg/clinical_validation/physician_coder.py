"""Deterministic, benchmark-blind coding of existing unassisted responses."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

from tm_ecg.clinical_validation.field_policy import PhysicianView
from tm_ecg.clinical_validation.models import (
    ClinicalAssertion,
    ClinicalFindingSet,
    EvidenceSpan,
)
from tm_ecg.clinical_validation.ontology import load_json_yaml


NEGATIONS = (
    "no evidence of",
    "not seen",
    "without",
    "absent",
    "none",
    "no",
    "not",
)
UNCERTAINTY = (
    "cannot exclude",
    "may represent",
    "question of",
    "possible",
    "possibly",
    "probable",
    "probably",
    "suspected",
)


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: object, default: Iterable[str] = ()) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in default]
    return [str(item) for item in value]


def normalize_clinical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace(chr(0x2013), "-").replace(chr(0x2014), "-")
    text = text.replace(chr(0x2212), "-")
    text = text.replace(chr(0xE2) + chr(0x20AC) + chr(0x201C), "-")
    text = text.replace(chr(0xE2) + chr(0x20AC) + chr(0x201D), "-")
    text = text.replace(chr(0xE2) + chr(0x20AC) + chr(0x2122), "'")
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(normalize_clinical_text(term)).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def _context(text: str, start: int, width: int = 72) -> str:
    """Return only the local clause before a match.

    Negation and uncertainty must not leak across punctuation or adversative
    conjunctions.  This deliberately favours a short, auditable context over a
    language model or a benchmark-informed interpretation.
    """

    prefix = text[:start]
    boundaries = [
        prefix.rfind(separator)
        for separator in (";", ",", ".", "\n", " but ", " however ", " although ")
    ]
    clause_start = max(boundaries)
    return text[max(clause_start + 1, start - width) : start]


def _polarity_and_certainty(text: str, start: int) -> tuple[str, str]:
    prefix = _context(text, start)
    polarity = (
        "negative"
        if any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", prefix) for term in NEGATIONS)
        else "positive"
    )
    certainty = "uncertain" if any(term in prefix for term in UNCERTAINTY) else "definite"
    return polarity, certainty


def _source_cell(view: PhysicianView, field: str) -> str:
    return dict(view.response.source_cells).get(field, "")


def _extract_text_rules(
    view: PhysicianView,
    rules: Iterable[Mapping[str, object]],
) -> tuple[dict[str, set[str]], list[EvidenceSpan], set[str]]:
    axes: dict[str, set[str]] = {
        "rhythm": set(),
        "ectopy": set(),
        "conduction": set(),
        "repolarization": set(),
    }
    evidence: list[EvidenceSpan] = []
    uncertain: set[str] = set()
    fields = {
        "primary_diagnosis": normalize_clinical_text(view.response.primary_diagnosis),
        "rationale": normalize_clinical_text(view.response.rationale),
    }
    for rule in rules:
        rule_id = str(rule["id"])
        axis = str(rule["axis"])
        finding = str(rule["finding"])
        allowed_fields = tuple(_string_list(rule.get("fields"), fields))
        for field in allowed_fields:
            text = fields.get(field, "")
            for term in _string_list(rule.get("terms")):
                match = _term_pattern(str(term)).search(text)
                if not match:
                    continue
                polarity, certainty = _polarity_and_certainty(text, match.start())
                span = EvidenceSpan(
                    source_field=field,
                    source_cell=_source_cell(view, field),
                    normalized_text_span=match.group(0),
                    matched_rule_id=rule_id,
                    polarity=polarity,
                    temporality="current",
                    certainty=certainty,
                    axis=axis,
                    finding=finding,
                )
                evidence.append(span)
                if polarity == "positive" and certainty == "definite":
                    axes.setdefault(axis, set()).add(finding)
                elif polarity == "positive":
                    uncertain.add(f"{axis}:{finding}")
                break
    return axes, evidence, uncertain


def _apply_specificity_precedence(axes: dict[str, set[str]]) -> None:
    if axes["rhythm"] & {"af", "afl"}:
        axes["rhythm"].discard("other_arrhythmia")
    if axes["ectopy"] & {"atrial_premature", "ventricular_premature", "mixed_ectopy"}:
        axes["ectopy"].discard("unspecified_ectopy")
    if axes["conduction"] & {"rbbb_spectrum", "lbbb_spectrum"}:
        axes["conduction"].discard("other_conduction")
    if axes["conduction"] & {
        "rbbb_spectrum",
        "lbbb_spectrum",
        "other_conduction",
    }:
        axes["conduction"].discard("unspecified_conduction")
    if axes["repolarization"] & {
        "st_elevation",
        "st_depression",
        "t_wave_abnormality",
        "qtc_abnormality",
    }:
        axes["repolarization"].discard("nonspecific_st_t")
        axes["repolarization"].discard("unspecified_repolarization")


def _response_text_fields(view: PhysicianView) -> dict[str, str]:
    return {
        "primary_diagnosis": normalize_clinical_text(
            view.response.primary_diagnosis
        ),
        "rationale": normalize_clinical_text(view.response.rationale),
    }


def _assertion_status(polarity: str, certainty: str) -> str:
    if polarity == "negative":
        return "negated"
    if certainty == "uncertain":
        return "uncertain"
    return "definite"


def _clinical_assertion(
    *,
    view: PhysicianView,
    source_field: str,
    raw_value: object,
    normalized_span: str,
    rule_id: str,
    rule_version: str,
    assertion_status: str,
    derivation_status: str,
    axis: str,
    finding: str,
) -> ClinicalAssertion:
    return ClinicalAssertion(
        source_field=source_field,
        source_cell=_source_cell(view, source_field),
        raw_text_or_value=str(raw_value or ""),
        normalized_span=normalized_span,
        rule_id=rule_id,
        rule_version=rule_version,
        assertion_status=assertion_status,
        derivation_status=derivation_status,
        axis=axis,
        finding=finding,
    )


def _legacy_evidence(assertion: ClinicalAssertion) -> EvidenceSpan:
    polarity = "negative" if assertion.assertion_status == "negated" else "positive"
    certainty = (
        "uncertain" if assertion.assertion_status == "uncertain" else "definite"
    )
    return EvidenceSpan(
        source_field=assertion.source_field,
        source_cell=assertion.source_cell,
        normalized_text_span=assertion.normalized_span,
        matched_rule_id=assertion.rule_id,
        polarity=polarity,
        temporality="current",
        certainty=certainty,
        axis=assertion.axis,
        finding=assertion.finding,
    )


def _code_physician_response_v2(
    view: PhysicianView,
    rules_config: Mapping[str, object],
) -> ClinicalFindingSet:
    """Preserve the immutable v2 audit coding contract."""

    axes, evidence, uncertain = _extract_text_rules(
        view, _mapping_list(rules_config.get("text_rules", []))
    )
    evidence_values = {
        key: normalize_clinical_text(value) for key, value in view.response.evidence
    }
    for rule in _mapping_list(rules_config.get("evidence_rules", [])):
        field = str(rule["field"])
        if evidence_values.get(field) not in {"yes", "true", "1"}:
            continue
        axis = str(rule["axis"])
        finding = str(rule["finding"])
        axes.setdefault(axis, set()).add(finding)
        evidence.append(
            EvidenceSpan(
                source_field=field,
                source_cell=_source_cell(view, field),
                normalized_text_span=evidence_values[field],
                matched_rule_id=str(rule["id"]),
                polarity="positive",
                temporality="current",
                certainty=str(rule.get("certainty", "definite")),
                axis=axis,
                finding=finding,
            )
        )

    _apply_specificity_precedence(axes)
    combined_text = normalize_clinical_text(
        f"{view.response.primary_diagnosis} {view.response.rationale}"
    )
    # "Wandering/migrating atrial pacemaker" describes a supraventricular
    # rhythm, not an implanted electronic pacemaker or paced complex.
    if any(
        phrase in combined_text
        for phrase in ("pacemaker migration", "wandering atrial pacemaker")
    ):
        axes.get("pacing", set()).discard("present")
        axes["rhythm"].add("other_arrhythmia")
    pacing = "present" if "present" in axes.get("pacing", set()) else "absent"
    abnormal = bool(
        axes["rhythm"] & {"af", "afl", "other_arrhythmia"}
        or axes["ectopy"]
        or axes["conduction"]
        or axes["repolarization"]
        or pacing == "present"
    )
    normality = (
        "abnormal"
        if abnormal
        else "normal"
        if "sinus" in axes["rhythm"]
        else "indeterminate"
    )

    quality_score = view.response.ecg_quality
    lead_noise = evidence_values.get("evidence_lead_noise") in {
        "yes",
        "true",
        "1",
    }
    quality = (
        "noisy_artifact"
        if lead_noise or (quality_score is not None and quality_score <= 2)
        else "adequate"
    )
    uninterpretable_terms = tuple(
        _string_list(rules_config.get("uninterpretable_terms", []))
    )
    if any(term in combined_text for term in uninterpretable_terms) or quality_score == 1:
        interpretability = "uninterpretable"
    elif (
        normalize_clinical_text(view.response.ambiguous) == "yes"
        or normalize_clinical_text(view.response.requires_additional_information)
        == "yes"
        or (
            view.response.diagnostic_confidence is not None
            and view.response.diagnostic_confidence <= 2
        )
    ):
        interpretability = "limited"
    else:
        interpretability = "interpretable"

    return ClinicalFindingSet(
        case_id=view.response.case_id,
        rhythm=tuple(sorted(axes["rhythm"])),
        ectopy=tuple(sorted(axes["ectopy"])),
        conduction=tuple(sorted(axes["conduction"])),
        repolarization=tuple(sorted(axes["repolarization"])),
        pacing=pacing,
        normality=normality,
        quality=quality,
        interpretability=interpretability,
        residual_abnormal="other_abnormal" in axes.get("residual", set()),
        uncertain_findings=tuple(sorted(uncertain)),
        evidence=tuple(evidence),
        coding_rule_version="physician_coding_v2",
    )


def _text_assertions_v3(
    view: PhysicianView,
    rules: Iterable[Mapping[str, object]],
    *,
    rule_version: str,
    derivation_status: str,
) -> list[ClinicalAssertion]:
    fields = _response_text_fields(view)
    raw_fields = {
        "primary_diagnosis": view.response.primary_diagnosis,
        "rationale": view.response.rationale,
    }
    assertions: list[ClinicalAssertion] = []
    for rule in rules:
        rule_id = str(rule["id"])
        axis = str(rule.get("axis", "observation"))
        finding = str(rule.get("finding", rule.get("observation", "")))
        allowed_fields = tuple(_string_list(rule.get("fields"), fields))
        for field in allowed_fields:
            text = fields.get(field, "")
            for term in _string_list(rule.get("terms")):
                match = _term_pattern(term).search(text)
                if match is None:
                    continue
                polarity, certainty = _polarity_and_certainty(text, match.start())
                assertions.append(
                    _clinical_assertion(
                        view=view,
                        source_field=field,
                        raw_value=raw_fields.get(field, ""),
                        normalized_span=match.group(0),
                        rule_id=rule_id,
                        rule_version=rule_version,
                        assertion_status=_assertion_status(polarity, certainty),
                        derivation_status=derivation_status,
                        axis=axis,
                        finding=finding,
                    )
                )
                break
    return assertions


def _quality_and_interpretability(
    view: PhysicianView,
    rules_config: Mapping[str, object],
    evidence_values: Mapping[str, str],
) -> tuple[str, str]:
    quality_score = view.response.ecg_quality
    lead_noise = evidence_values.get("evidence_lead_noise") in {
        "yes",
        "true",
        "1",
    }
    quality = (
        "noisy_artifact"
        if lead_noise or (quality_score is not None and quality_score <= 2)
        else "adequate"
    )
    combined_text = normalize_clinical_text(
        f"{view.response.primary_diagnosis} {view.response.rationale}"
    )
    uninterpretable_terms = tuple(
        _string_list(rules_config.get("uninterpretable_terms", []))
    )
    if any(term in combined_text for term in uninterpretable_terms) or quality_score == 1:
        interpretability = "uninterpretable"
    elif (
        normalize_clinical_text(view.response.ambiguous) == "yes"
        or normalize_clinical_text(view.response.requires_additional_information)
        == "yes"
        or (
            view.response.diagnostic_confidence is not None
            and view.response.diagnostic_confidence <= 2
        )
    ):
        interpretability = "limited"
    else:
        interpretability = "interpretable"
    return quality, interpretability


def _code_physician_response_v3(
    view: PhysicianView,
    rules_config: Mapping[str, object],
) -> ClinicalFindingSet:
    """Code physician text into diagnoses and observations without a benchmark."""

    rule_version = str(
        rules_config.get("rule_version", "physician_coding_v3.0.0")
    )
    diagnoses = _text_assertions_v3(
        view,
        _mapping_list(rules_config.get("text_diagnosis_rules", [])),
        rule_version=rule_version,
        derivation_status="explicit",
    )
    observations = _text_assertions_v3(
        view,
        _mapping_list(rules_config.get("text_observation_rules", [])),
        rule_version=rule_version,
        derivation_status="observation",
    )

    evidence_values = {
        key: normalize_clinical_text(value) for key, value in view.response.evidence
    }
    raw_evidence_values = dict(view.response.evidence)
    for rule in _mapping_list(rules_config.get("checkbox_observation_rules", [])):
        field = str(rule["field"])
        if evidence_values.get(field) not in {"yes", "true", "1"}:
            continue
        observation = str(rule["observation"])
        observations.append(
            _clinical_assertion(
                view=view,
                source_field=field,
                raw_value=raw_evidence_values.get(field, ""),
                normalized_span=evidence_values[field],
                rule_id=str(rule["id"]),
                rule_version=rule_version,
                assertion_status="definite",
                derivation_status="observation",
                axis="observation",
                finding=observation,
            )
        )

    axes: dict[str, set[str]] = {
        "rhythm": set(),
        "ectopy": set(),
        "conduction": set(),
        "repolarization": set(),
        "pacing": set(),
        "residual": set(),
    }
    uncertain: set[str] = set()
    for assertion in diagnoses:
        if assertion.assertion_status == "definite":
            axes.setdefault(assertion.axis, set()).add(assertion.finding)
        elif assertion.assertion_status == "uncertain":
            uncertain.add(f"{assertion.axis}:{assertion.finding}")

    present_observations = {
        item.finding
        for item in observations
        if item.assertion_status == "definite"
    }
    derived_assertions: list[ClinicalAssertion] = []
    for rule in _mapping_list(rules_config.get("axis_derivation_rules", [])):
        requires_all = set(_string_list(rule.get("requires_observations")))
        requires_any = set(_string_list(rule.get("requires_any_observation")))
        if requires_all and not requires_all.issubset(present_observations):
            continue
        if requires_any and not requires_any.intersection(present_observations):
            continue
        if not requires_all and not requires_any:
            continue
        axis = str(rule["axis"])
        finding = str(rule["finding"])
        axes.setdefault(axis, set()).add(finding)
        source_observations = sorted(
            (requires_all | requires_any).intersection(present_observations)
        )
        derived_assertions.append(
            _clinical_assertion(
                view=view,
                source_field="derived_from_observations",
                raw_value=" | ".join(source_observations),
                normalized_span=" & ".join(source_observations),
                rule_id=str(rule["id"]),
                rule_version=rule_version,
                assertion_status="definite",
                derivation_status="evidence-supported",
                axis=axis,
                finding=finding,
            )
        )

    _apply_specificity_precedence(axes)
    # The v3 terms deliberately distinguish a wandering atrial pacemaker from
    # electronic pacing.  Retain a hard fail-safe in case later rules regress.
    combined_text = normalize_clinical_text(
        f"{view.response.primary_diagnosis} {view.response.rationale}"
    )
    conflict_trace: list[str] = []
    if any(
        phrase in combined_text
        for phrase in (
            "pacemaker migration",
            "wandering atrial pacemaker",
            "migrating atrial pacemaker",
        )
    ) and "present" in axes["pacing"]:
        axes["pacing"].discard("present")
        conflict_trace.append(
            "PHYS-V3-CONFLICT-WANDERING-PACEMAKER:removed_electronic_pacing"
        )

    normality_config = rules_config.get("normality_resolution", {})
    normality_rules = (
        normality_config if isinstance(normality_config, Mapping) else {}
    )
    normal_assertions: list[ClinicalAssertion] = []
    for term in _string_list(normality_rules.get("explicit_normal_terms")):
        for field, text in _response_text_fields(view).items():
            match = _term_pattern(term).search(text)
            if match is None:
                continue
            polarity, certainty = _polarity_and_certainty(text, match.start())
            normal_assertions.append(
                _clinical_assertion(
                    view=view,
                    source_field=field,
                    raw_value=getattr(view.response, field),
                    normalized_span=match.group(0),
                    rule_id="PHYS-V3-NORMAL-EXPLICIT",
                    rule_version=rule_version,
                    assertion_status=_assertion_status(polarity, certainty),
                    derivation_status="explicit",
                    axis="normality",
                    finding="normal",
                )
            )
            break
    diagnoses.extend(normal_assertions)
    explicit_normal = any(
        item.assertion_status == "definite" for item in normal_assertions
    )
    abnormal = bool(
        axes["rhythm"]
        & {
            "af",
            "afl",
            "other_arrhythmia",
            "sinus_bradycardia",
            "sinus_tachycardia",
        }
        or axes["ectopy"]
        or axes["conduction"]
        or axes["repolarization"]
        or "present" in axes["pacing"]
        or "other_abnormal" in axes["residual"]
    )
    if abnormal:
        normality = "abnormal"
        if explicit_normal:
            conflict_trace.append(
                "PHYS-V3-NORMALITY-ABNORMAL-PRECEDENCE:explicit_normal_overridden"
            )
    elif explicit_normal:
        normality = "normal"
    else:
        normality = str(normality_rules.get("no_assertion", "indeterminate"))

    pacing = "present" if "present" in axes["pacing"] else "absent"
    quality, interpretability = _quality_and_interpretability(
        view, rules_config, evidence_values
    )
    all_assertions = [*diagnoses, *observations, *derived_assertions]
    return ClinicalFindingSet(
        case_id=view.response.case_id,
        rhythm=tuple(sorted(axes["rhythm"])),
        ectopy=tuple(sorted(axes["ectopy"])),
        conduction=tuple(sorted(axes["conduction"])),
        repolarization=tuple(sorted(axes["repolarization"])),
        pacing=pacing,
        normality=normality,
        quality=quality,
        interpretability=interpretability,
        residual_abnormal="other_abnormal" in axes["residual"],
        uncertain_findings=tuple(sorted(uncertain)),
        evidence=tuple(_legacy_evidence(item) for item in all_assertions),
        explicit_diagnoses=tuple(diagnoses),
        observations=tuple(observations),
        axis_derivations=tuple(derived_assertions),
        projection_labels=(),
        conflict_resolution_trace=tuple(conflict_trace),
        coding_rule_version=rule_version,
    )


def code_physician_response(
    view: PhysicianView,
    rules_config: Mapping[str, object],
) -> ClinicalFindingSet:
    version = int(str(rules_config.get("version", 0)))
    if version == 2:
        return _code_physician_response_v2(view, rules_config)
    if version == 3:
        return _code_physician_response_v3(view, rules_config)
    raise ValueError(f"Unsupported physician coding rules version: {version}")


def load_physician_rules(path: str | Path) -> dict[str, object]:
    payload = load_json_yaml(path)
    if payload.get("version") not in {2, 3}:
        raise ValueError("Physician coding rules must declare version 2 or 3")
    return payload
