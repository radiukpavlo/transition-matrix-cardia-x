from __future__ import annotations

from tm_ecg.clinical_validation.models import ClinicalFindingSet
from tm_ecg.clinical_validation.ontology import (
    normalize_axis_values,
    project_binary,
    project_dominant,
    project_exact,
    project_five_family,
)


def test_multilabel_projections_are_deterministic() -> None:
    finding = ClinicalFindingSet(
        case_id="P001",
        rhythm=("af",),
        ectopy=("ventricular_premature",),
        pacing="absent",
        normality="abnormal",
    )
    assert project_exact(finding) == "AF | PVC"
    assert project_five_family(finding) == "Ectopy | Rhythm"
    assert project_binary(finding) == "Abnormal"
    assert project_dominant(finding, ("PVC", "AF")) == "PVC"


def test_uninterpretable_is_not_forced_to_normal_or_abnormal() -> None:
    finding = ClinicalFindingSet(case_id="P001", interpretability="uninterpretable")
    assert project_binary(finding) == "Uninterpretable"


def test_sinus_is_not_projected_as_global_normal_when_abnormality_exists() -> None:
    finding = ClinicalFindingSet(
        case_id="P001",
        rhythm=("sinus",),
        ectopy=("ventricular_premature",),
        normality="abnormal",
    )
    assert project_exact(finding) == "PVC"
    assert project_five_family(finding) == "Ectopy"
    assert project_binary(finding) == "Abnormal"


def test_shared_index_axis_aliases_are_normalized_at_validation_boundary() -> None:
    normalized = normalize_axis_values(
        {
            "ectopy": ["pvc", "apb"],
            "repolarization": ["t_abnormality", "other_st_t"],
        }
    )
    assert normalized["ectopy"] == ("atrial_premature", "ventricular_premature")
    assert normalized["repolarization"] == (
        "nonspecific_st_t",
        "t_wave_abnormality",
    )
