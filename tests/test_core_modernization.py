from __future__ import annotations

import pytest

from tm_ecg.dss.discretization import fit_wedd_discretization
from tm_ecg.dss.models import ProductionRule, RuleCondition
from tm_ecg.dss.predicates import default_medical_predicates
from tm_ecg.dss.rules import infer_with_rules
from tm_ecg.dss.selection import select_b_features, validate_matrix_alignment
from tm_ecg.stages.fit_transition import (
    _bootstrap_transition_stability,
    _reconcile_missing_latents,
)
from tm_ecg.features.formulas import (
    BeatMeasurement,
    RecordMeasurements,
    compute_feature_quality_states,
    compute_record_features,
)
from tm_ecg.features.signatures import apply_signature_scores, fit_signature_artifact
from tm_ecg.ontology import compatibility_projection, map_ptbxl_axes
from tm_ecg.stages.splits import _split_audit
from tm_ecg.types import RecordIndexRow, SplitManifest


THRESHOLDS = {
    "minimum_valid_beats": 2,
    "minimum_analyzable_fraction": 0.5,
    "feature_quality_min_db": 5.0,
    "detector_agreement_minimum": 0.5,
    "minimum_rhythm_lead_coverage": 1.0,
    "minimum_p_wave_lead_coverage": 0.67,
    "minimum_qrs_lead_coverage": 0.6,
    "minimum_st_t_lead_coverage": 0.6,
    "qrs_wide_ms": 120.0,
    "t_inverted_threshold_mv": -0.1,
    "t_inverted_duration_ms": 80.0,
}


def test_ptbxl_multiaxial_mapping_preserves_concurrent_findings() -> None:
    target = map_ptbxl_axes(
        {"scp_codes": "{'NORM': 100, 'SR': 0, 'PVC': 90, 'STD_': 80, 'RBBB': 90}"}
    )
    assert target.rhythm == ("sinus",)
    assert target.normality == "normal"
    assert target.ectopy == ("pvc",)
    assert target.conduction == ("rbbb_spectrum",)
    assert target.repolarization == ("st_depression",)
    assert compatibility_projection(target) == [
        "PVC",
        "RBBB spectrum",
    ]


def test_ptbxl_zero_likelihood_presence_statements_are_not_discarded() -> None:
    target = map_ptbxl_axes(
        {"scp_codes": "{'AFIB': 0, 'PAC': 0, 'PVC': 0, 'NORM': 15}"}
    )

    assert target.rhythm == ("af",)
    assert target.ectopy == ("apb", "pvc")
    assert "sinus" not in target.rhythm


def test_split_audit_hard_fails_patient_leakage() -> None:
    base = {
        "dataset": "ptbxl",
        "labels": ["Normal"],
        "source_path": "x",
        "preprocessing_hash": "p",
        "ontology_version": "v",
    }
    rows = [
        RecordIndexRow("1", "1", "same", split="train", **base),
        RecordIndexRow("2", "2", "same", split="test", **base),
    ]
    manifest = SplitManifest("ptbxl", "now", 17, rows)
    with pytest.raises(RuntimeError, match="Patient leakage"):
        _split_audit(manifest)


def test_quality_gating_uses_nn_intervals_and_never_zero_fills_unavailable_qrs() -> None:
    beats = [
        BeatMeasurement("b1", rr_s=1.0, qrs_valid=False, detector_agreement=0.9),
        BeatMeasurement(
            "b2",
            rr_s=0.5,
            is_ectopic=True,
            qrs_valid=False,
            detector_agreement=0.9,
        ),
        BeatMeasurement("b3", rr_s=1.0, qrs_valid=False, detector_agreement=0.9),
    ]
    record = RecordMeasurements(
        "r1",
        beats=beats,
        lead_quality_by_lead_db={"II": 12.0, "I": 12.0, "V1": 12.0},
        analyzable_duration_s=12.0,
    )
    features = compute_record_features(record, THRESHOLDS)
    states = compute_feature_quality_states(record, THRESHOLDS)
    assert features["af_irregularity_cv"] == 0.0
    assert features["qrs_wide_any"] is None
    assert states["qrs"].state == "unavailable"


def test_signature_artifact_is_train_only_deterministic_and_schema_checked() -> None:
    train = []
    validation = []
    labels = ["AF", "PVC", "RBBB spectrum", "LBBB spectrum", "Paced", "Normal"]
    for index in range(60):
        label = labels[index % len(labels)]
        row = {
            "record_id": f"t{index}",
            "labels": label,
            "af_irregularity_cv": 0.8 if label == "AF" else 0.05,
            "pvc_like_beat_count": 4 if label == "PVC" else 0,
            "r_prime_v1_any": 1 if label == "RBBB spectrum" else 0,
            "broad_r_v6_any": 1 if label == "LBBB spectrum" else 0,
            "paced_like_beat_count": 4 if label == "Paced" else 0,
        }
        (train if index < 48 else validation).append(row)
    first = fit_signature_artifact(train, validation, random_seed=17)
    second = fit_signature_artifact(train, validation, random_seed=17)
    assert first == second
    scores, states = apply_signature_scores(validation[0], first)
    assert all(scores[name] is not None for name in scores)
    assert set(states.values()) == {"observed"}
    broken = dict(first)
    broken["signatures"] = {
        **dict(first["signatures"]),
        "af_signature_score": {
            **dict(dict(first["signatures"])["af_signature_score"]),
            "coefficients": [],
        },
    }
    scores, states = apply_signature_scores(validation[0], broken)
    assert scores["af_signature_score"] is None
    assert states["af_signature_score"] == "schema_incompatible"


def test_strict_selection_has_no_silent_all_feature_fallback() -> None:
    assert select_b_features(set(), ["qrs_dur_med_ms"]) == []
    assert select_b_features(
        set(),
        ["qrs_dur_med_ms"],
        operator=[[1.0]],
        operator_b_columns=["qrs_dur_med_ms"],
        top_transition_features=1,
        stable_transition_features={"qrs_dur_med_ms"},
    ) == []
    assert select_b_features(
        set(), ["qrs_dur_med_ms"], allow_research_fallback=True
    ) == ["qrs_dur_med_ms"]


def test_transition_stability_is_training_only_and_deterministic() -> None:
    a_rows = [[float(index)] for index in range(1, 31)]
    b_rows = [[float(index), 0.0] for index in range(1, 31)]
    kwargs = {
        "lambda_value": 0.1,
        "rank_cap": 1,
        "replicates": 20,
        "seed": 17,
        "top_features": 1,
        "minimum_frequency": 0.8,
    }
    first = _bootstrap_transition_stability(
        a_rows, b_rows, ["signal", "null"], **kwargs
    )
    second = _bootstrap_transition_stability(
        a_rows, b_rows, ["signal", "null"], **kwargs
    )
    assert first == second
    assert first["stable_features"] == ["signal"]


def test_missing_latent_exclusion_is_bounded_and_audited() -> None:
    a_rows = [{"record_id": "a"}, {"record_id": "b"}]
    b_rows = [
        {"record_id": "a", "value": 1.0},
        {"record_id": "b", "value": 2.0},
        {"record_id": "c", "value": 3.0},
    ]
    reconciled, audit = _reconcile_missing_latents(
        a_rows, b_rows, maximum_missing_latent_fraction=0.34
    )
    assert [row["record_id"] for row in reconciled] == ["a", "b"]
    assert audit["missing_latent_record_ids"] == ["c"]
    with pytest.raises(RuntimeError, match="exceeds policy"):
        _reconcile_missing_latents(
            a_rows, b_rows, maximum_missing_latent_fraction=0.10
        )


def test_conduction_inference_abstains_when_required_leads_are_poor() -> None:
    rows = [
        {"qrs_dur_med_ms": 140.0, "r_prime_v1_any": 1, "qrs_lead_coverage": 1.0},
        {"qrs_dur_med_ms": 90.0, "r_prime_v1_any": 0, "qrs_lead_coverage": 1.0},
    ]
    plan = fit_wedd_discretization(
        rows,
        ["RBBB spectrum", "Normal"],
        features=["qrs_dur_med_ms", "r_prime_v1_any", "qrs_lead_coverage"],
        min_support=1,
    )
    rule = ProductionRule(
        "r1",
        "RBBB spectrum",
        [
            RuleCondition("qrs_dur_med_ms", state="wide_qrs"),
            RuleCondition("r_prime_v1_any", state="present"),
        ],
        1.0,
        10,
        0.5,
        ["r1"],
        {"RBBB spectrum": 10},
    )
    result = infer_with_rules(
        {"qrs_dur_med_ms": 145.0, "r_prime_v1_any": 1, "qrs_lead_coverage": 0.2},
        [rule],
        plan,
        predicates=default_medical_predicates(),
    )
    assert result.predicted_label is None
    assert result.abstention is not None
    assert result.abstention.reason == "insufficient_required_leads"
    assert result.score_decomposition


def test_transition_alignment_rejects_permuted_row_id_sets() -> None:
    report = validate_matrix_alignment(
        2, 2, [[1.0]], ["a", "b"], ["a", "c"]
    )
    assert not report.valid
    assert "row_id_sets_are_not_identical" in report.notes
