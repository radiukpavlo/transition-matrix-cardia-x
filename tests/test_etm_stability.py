from __future__ import annotations

import numpy as np
import pytest

from tm_ecg.dss.etm import (
    ecg_safe_waveform_perturbations,
    semantic_perturbation_audit,
)


def test_safe_perturbations_are_deterministic_and_explicitly_scoped() -> None:
    signal = np.linspace(-1.0, 1.0, 400)[:, None] * np.ones((1, 3))
    first = ecg_safe_waveform_perturbations(
        signal,
        sampling_rate_hz=500.0,
        seed=4,
        redundant_lead_index=2,
        beat_indices=[100, 250],
    )
    second = ecg_safe_waveform_perturbations(
        signal,
        sampling_rate_hz=500.0,
        seed=4,
        redundant_lead_index=2,
        beat_indices=[100, 250],
    )
    assert set(first) == set(second)
    assert "beat_aligned_jitter" in first
    assert np.isnan(first["single_declared_redundant_lead_dropout"][:, 2]).all()
    assert np.allclose(
        first["low_amplitude_additive_noise"],
        second["low_amplitude_additive_noise"],
    )
    with pytest.raises(ValueError, match="outside"):
        ecg_safe_waveform_perturbations(
            signal,
            sampling_rate_hz=500.0,
            redundant_lead_index=9,
        )


def test_semantic_audit_reports_all_required_stability_metrics() -> None:
    operator = [[1.0, 0.0], [0.0, 1.0]]
    original = [[0.2, 0.8], [0.7, 0.3]]
    perturbed = {"noise": [[0.21, 0.79], [0.69, 0.31]]}
    audit = semantic_perturbation_audit(
        operator,
        original,
        perturbed,
        semantic_thresholds=[0.5, 0.5],
        original_routes=["exact_match", "soft_match"],
        perturbed_routes={"noise": ["exact_match", "soft_match"]},
        original_truth=original,
        perturbed_truth={"noise": perturbed["noise"]},
    )
    row = audit["perturbations"]["noise"]
    assert set(row) == {
        "semantic_stability_error",
        "predicate_flip_rate",
        "rule_route_flip_rate",
        "transition_fidelity_delta",
    }
    assert row["predicate_flip_rate"] == 0.0
    assert row["rule_route_flip_rate"] == 0.0
