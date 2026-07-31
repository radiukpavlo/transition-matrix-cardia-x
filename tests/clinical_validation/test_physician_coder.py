from __future__ import annotations

from pathlib import Path

from tm_ecg.clinical_validation.field_policy import build_physician_view
from tm_ecg.clinical_validation.physician_coder import (
    code_physician_response,
    load_physician_rules,
    normalize_clinical_text,
)


ROOT = Path(__file__).resolve().parents[2]


def _code(diagnosis: str, rationale: str = ""):
    rules = load_physician_rules(ROOT / "clinical_validation/config/physician_coding_rules_v2.yaml")
    view = build_physician_view(
        {
            "case_id": "P001",
            "primary_diagnosis": diagnosis,
            "rationale": rationale,
            "diagnostic_confidence": 4,
            "ecg_quality": 4,
            "ambiguous": "No",
            "requires_additional_information": "No",
        }
    )
    return code_physician_response(view, rules)


def _code_v3(diagnosis: str, rationale: str = "", **evidence: str):
    rules = load_physician_rules(
        ROOT / "clinical_validation/config/physician_coding_rules_v3.yaml"
    )
    view = build_physician_view(
        {
            "case_id": "P001",
            "primary_diagnosis": diagnosis,
            "rationale": rationale,
            "diagnostic_confidence": 4,
            "ecg_quality": 4,
            "ambiguous": "No",
            "requires_additional_information": "No",
            **evidence,
        }
    )
    return code_physician_response(view, rules)


def test_negation_and_uncertainty_are_not_definite_findings() -> None:
    finding = _code("No atrial fibrillation; possible PVC")
    assert "af" not in finding.rhythm
    assert "ventricular_premature" not in finding.ectopy
    assert "ectopy:ventricular_premature" in finding.uncertain_findings
    assert {item.polarity for item in finding.evidence} >= {"negative", "positive"}


def test_specificity_precedence_removes_unspecified_ectopy() -> None:
    finding = _code("PVC with premature beats")
    assert finding.ectopy == ("ventricular_premature",)


def test_specific_rhythm_removes_generic_irregular_rhythm_evidence() -> None:
    rules = load_physician_rules(ROOT / "clinical_validation/config/physician_coding_rules_v2.yaml")
    view = build_physician_view(
        {
            "case_id": "P001",
            "primary_diagnosis": "Atrial fibrillation",
            "rationale": "Irregular rhythm without visible P waves",
            "diagnostic_confidence": 5,
            "ecg_quality": 4,
            "ambiguous": "No",
            "requires_additional_information": "No",
            "evidence_rr_irregularity": "Yes",
        }
    )
    finding = code_physician_response(view, rules)
    assert finding.rhythm == ("af",)


def test_wandering_atrial_pacemaker_is_not_electronic_pacing() -> None:
    finding = _code("Sinus rhythm, pacemaker migration through the atrium")
    assert finding.pacing == "absent"
    assert "other_arrhythmia" in finding.rhythm


def test_v3_rr_irregularity_is_an_observation_not_arrhythmia() -> None:
    finding = _code_v3(
        "Sinus rhythm",
        evidence_rr_irregularity="Yes",
    )
    assert finding.rhythm == ("sinus",)
    assert finding.normality == "indeterminate"
    assert {item.finding for item in finding.observations} == {"rr_irregularity"}
    assert finding.projection_labels == ()


def test_v3_wide_qrs_alone_does_not_create_conduction_diagnosis() -> None:
    finding = _code_v3("Sinus rhythm", evidence_wide_qrs="Yes")
    assert finding.conduction == ()
    assert {item.finding for item in finding.observations} == {"wide_qrs"}


def test_v3_bundle_pattern_derives_only_unspecified_conduction() -> None:
    finding = _code_v3(
        "Sinus rhythm",
        evidence_bundle_branch_pattern="Yes",
    )
    assert finding.conduction == ("unspecified_conduction",)
    assert not {"rbbb_spectrum", "lbbb_spectrum"} & set(finding.conduction)
    assert finding.normality == "abnormal"


def test_v3_checkbox_derivations_remain_generic() -> None:
    finding = _code_v3(
        "",
        evidence_premature_beat="Yes",
        evidence_st_deviation="Yes",
    )
    assert finding.ectopy == ("unspecified_ectopy",)
    assert finding.repolarization == ("unspecified_repolarization",)
    assert finding.normality == "abnormal"
    assert finding.explicit_diagnoses == ()


def test_v3_other_checkbox_has_no_diagnostic_semantics() -> None:
    finding = _code_v3("", evidence_other="Yes")
    assert finding.residual_abnormal is False
    assert finding.normality == "indeterminate"
    assert {item.finding for item in finding.observations} == {
        "other_observation_unspecified"
    }


def test_v3_explicit_normality_and_abnormal_precedence() -> None:
    normal = _code_v3("Normal ECG")
    assert normal.normality == "normal"
    abnormal = _code_v3("Normal ECG with PVC")
    assert abnormal.normality == "abnormal"
    assert abnormal.ectopy == ("ventricular_premature",)
    assert abnormal.conflict_resolution_trace == (
        "PHYS-V3-NORMALITY-ABNORMAL-PRECEDENCE:explicit_normal_overridden",
    )


def test_v3_clause_local_negation_and_uncertainty() -> None:
    finding = _code_v3(
        "No evidence of atrial fibrillation, sinus rhythm; probable PVC"
    )
    assert finding.rhythm == ("sinus",)
    assert finding.ectopy == ()
    assert "ectopy:ventricular_premature" in finding.uncertain_findings
    statuses = {
        (item.finding, item.assertion_status)
        for item in finding.explicit_diagnoses
    }
    assert ("af", "negated") in statuses
    assert ("ventricular_premature", "uncertain") in statuses


def test_v3_wandering_atrial_pacemaker_is_not_electronic_pacing() -> None:
    finding = _code_v3("Wandering atrial pacemaker")
    assert finding.pacing == "absent"
    assert finding.rhythm == ("other_arrhythmia",)


def test_unicode_normalization_repairs_known_mojibake() -> None:
    assert normalize_clinical_text("STâ€“T CHANGE") == "st-t change"
