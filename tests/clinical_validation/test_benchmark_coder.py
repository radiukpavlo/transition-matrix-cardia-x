from __future__ import annotations

from pathlib import Path

import pytest

from tm_ecg.clinical_validation.benchmark_coder import (
    code_benchmark_case,
    load_benchmark_mapping,
)
from tm_ecg.clinical_validation.models import CaseIdentity


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = CaseIdentity("P001", "P001", "ptbxl", "1", "exact_match", None, 2)
DEVELOPMENT_MAPPING = (
    ROOT
    / "clinical_validation/config/benchmark_mapping_semantic_development_2e1a59f5.yaml"
)
V3_MAPPING = ROOT / "clinical_validation/config/benchmark_mapping_v3.yaml"
PTBXL_STATEMENTS = (
    ROOT
    / "data"
    / "raw"
    / "ptbxl"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    / "scp_statements.csv"
)
requires_ptbxl_metadata = pytest.mark.skipif(
    not PTBXL_STATEMENTS.is_file(),
    reason="PTB-XL metadata is available after the public dataset is installed",
)


def test_known_multilabel_benchmark_is_mapped_without_other_fallback() -> None:
    mapping = load_benchmark_mapping(ROOT / "clinical_validation/config/benchmark_mapping_v2.yaml")
    result = code_benchmark_case(
        IDENTITY,
        {"hidden_benchmark_label": "AF|PVC|Other / unmapped"},
        mapping,
    )
    assert result.rhythm == ("af",)
    assert result.ectopy == ("ventricular_premature",)
    assert not result.residual_abnormal
    assert result.pacing == "absent"


def test_unknown_benchmark_code_is_retained_as_residual_abnormality() -> None:
    mapping = load_benchmark_mapping(ROOT / "clinical_validation/config/benchmark_mapping_v2.yaml")
    result = code_benchmark_case(
        IDENTITY, {"hidden_benchmark_label": "UNREGISTERED_CODE"}, mapping
    )
    assert result.residual_abnormal
    assert "BENCH-UNKNOWN-RESIDUAL" in result.mapping_rule_ids


def test_ptbxl_source_confidence_policy_overrides_legacy_projection() -> None:
    mapping = load_benchmark_mapping(DEVELOPMENT_MAPPING)
    result = code_benchmark_case(
        IDENTITY,
        {"hidden_benchmark_label": "Normal|PVC"},
        mapping,
        {
            "source_likelihoods_json": '{"AFIB": 0, "PVC": 0, "RBBB": 20}',
            "axis_targets_json": '{"rhythm": ["af"], "ectopy": ["pvc"], "conduction": []}',
        },
    )
    assert result.rhythm == ("af",)
    assert result.ectopy == ("ventricular_premature",)
    assert result.conduction == ()
    assert result.normality == "abnormal"
    assert result.source_labels == ("AFIB", "PVC", "RBBB")
    assert result.accepted_source_labels == ("AFIB", "PVC")
    assert result.ignored_source_labels == ("RBBB",)
    assert result.uncertain_findings == ()


def test_diagnostic_likelihood_bands_remain_accepted_uncertain_and_ignored() -> None:
    mapping = load_benchmark_mapping(DEVELOPMENT_MAPPING)
    result = code_benchmark_case(
        IDENTITY,
        {"hidden_benchmark_label": "Other / unmapped"},
        mapping,
        {
            "source_likelihoods_json": '{"RBBB": 90, "LBBB": 60, "IVCD": 20}',
        },
    )
    assert result.conduction == ("rbbb_spectrum",)
    assert result.source_labels == ("IVCD", "LBBB", "RBBB")
    assert result.accepted_source_labels == ("RBBB",)
    assert result.uncertain_source_labels == ("LBBB",)
    assert result.ignored_source_labels == ("IVCD",)
    assert result.uncertain_findings == ("conduction:lbbb_spectrum",)


def test_invalid_confidence_bands_fail_closed() -> None:
    mapping = load_benchmark_mapping(ROOT / "clinical_validation/config/benchmark_mapping_v2.yaml")
    mapping["confidence_bands"] = {"accepted_min": 40, "uncertain_min": 50}
    import pytest

    with pytest.raises(ValueError, match="confidence bands"):
        code_benchmark_case(
            IDENTITY,
            {"hidden_benchmark_label": "Normal"},
            mapping,
            {"source_likelihoods_json": '{"NORM": 100}'},
        )


@requires_ptbxl_metadata
def test_v3_sr_presence_does_not_imply_normality() -> None:
    mapping = load_benchmark_mapping(V3_MAPPING)
    result = code_benchmark_case(
        IDENTITY,
        {"hidden_benchmark_label": "Other / unmapped"},
        mapping,
        {"source_likelihoods_json": '{"SR": 0}'},
    )
    assert result.rhythm == ("sinus",)
    assert result.normality == "indeterminate"
    assert result.accepted_source_labels == ("SR",)
    assert result.source_statement_trace[0].category == "rhythm_statement"
    assert result.source_statement_trace[0].state == "present"


@requires_ptbxl_metadata
def test_v3_presence_coded_af_and_pac_are_retained_at_zero() -> None:
    mapping = load_benchmark_mapping(V3_MAPPING)
    result = code_benchmark_case(
        IDENTITY,
        {"hidden_benchmark_label": "Other / unmapped"},
        mapping,
        {"source_likelihoods_json": '{"AFIB": 0, "PAC": 0}'},
    )
    assert result.rhythm == ("af",)
    assert result.ectopy == ("atrial_premature",)
    assert result.normality == "abnormal"


@requires_ptbxl_metadata
def test_v3_norm_with_abnormal_finding_resolves_to_abnormal() -> None:
    mapping = load_benchmark_mapping(V3_MAPPING)
    result = code_benchmark_case(
        IDENTITY,
        {"hidden_benchmark_label": "Normal|RBBB spectrum"},
        mapping,
        {"source_likelihoods_json": '{"NORM": 100, "RBBB": 90}'},
    )
    assert result.conduction == ("rbbb_spectrum",)
    assert result.normality == "abnormal"
