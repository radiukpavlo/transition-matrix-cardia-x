from tm_ecg.dss.discretization import (
    bootstrap_threshold_stability,
    fit_wedd_discretization,
    quantize_row,
    threshold_perturbation_flip_rate,
    wedd_candidate_scores,
)
import pytest


def test_wedd_prefers_low_density_class_split():
    values = [1.0, 1.1, 1.2, 8.8, 9.0, 9.2]
    labels = ["A", "A", "A", "B", "B", "B"]
    scores = wedd_candidate_scores(values, labels, alpha=0.65, min_leaf=2)
    assert scores
    assert 1.2 < scores[0].value < 8.8
    assert scores[0].entropy == 0.0


def test_clinical_quantization_uses_anchor_bins():
    rows = [
        {"qrs_dur_med_ms": 90.0, "af_irregularity_cv": 0.05, "record_id": "a"},
        {"qrs_dur_med_ms": 130.0, "af_irregularity_cv": 0.30, "record_id": "b"},
    ]
    labels = ["Normal", "RBBB spectrum"]
    plan = fit_wedd_discretization(rows, labels, features=["qrs_dur_med_ms", "af_irregularity_cv"])
    assert quantize_row(rows[0], plan)["qrs_dur_med_ms"] == "narrow_qrs"
    assert quantize_row(rows[1], plan)["qrs_dur_med_ms"] == "wide_qrs"
    assert quantize_row(rows[1], plan)["af_irregularity_cv"] == "irregular"
    assert plan.feature_domains["qrs_dur_med_ms"].provenance == "clinical_anchor"


def test_patient_bootstrap_and_candidate_ledger_are_deterministic():
    values = [1.0, 1.1, 1.2, 8.8, 9.0, 9.2]
    labels = ["A", "A", "A", "B", "B", "B"]
    patient_ids = [f"p{index}" for index in range(len(values))]
    first = bootstrap_threshold_stability(
        values,
        labels,
        patient_ids,
        min_leaf=1,
        n_bootstrap=20,
        seed=7,
    )
    second = bootstrap_threshold_stability(
        values,
        labels,
        patient_ids,
        min_leaf=1,
        n_bootstrap=20,
        seed=7,
    )
    assert first == second
    rows = [{"custom_score": value} for value in values]
    plan = fit_wedd_discretization(
        rows,
        labels,
        features=["custom_score"],
        prefer_clinical_bins=False,
        min_support=1,
        patient_ids=patient_ids,
        fit_partition="oof",
        n_bootstrap=20,
        stability_tolerance=4.0,
        min_selection_frequency=0.2,
        random_seed=7,
    )
    assert plan.candidate_thresholds
    assert plan.fit_partition == "oof"
    assert all(item.version == "wedd_v2" for item in plan.thresholds)
    assert all(item.selection_frequency is not None for item in plan.thresholds)


def test_wedd_rejects_fragile_thresholds_and_non_development_partition():
    rows = [{"custom_score": value} for value in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]
    labels = ["A", "A", "A", "B", "B", "B"]
    plan = fit_wedd_discretization(
        rows,
        labels,
        features=["custom_score"],
        prefer_clinical_bins=False,
        min_support=1,
        perturbation_tolerance=10.0,
        max_perturbation_flip_rate=0.2,
    )
    assert plan.thresholds == []
    assert threshold_perturbation_flip_rate(
        [1.0, 2.0, 3.0],
        threshold=2.0,
        tolerance=0.1,
    ) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="training or OOF"):
        fit_wedd_discretization(
            rows,
            labels,
            features=["custom_score"],
            fit_partition="confirmatory",
        )
