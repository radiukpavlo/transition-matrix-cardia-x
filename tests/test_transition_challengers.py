from __future__ import annotations

import numpy as np
import pytest

from tm_ecg.stages.fit_transition import _transition_validation_diagnostics
from tm_ecg.transition.ridge import (
    apply_transition_package,
    fit_reduced_rank_transition,
    fit_robust_transition,
    fit_masked_robust_transition,
    fit_sign_constrained_transition,
    fit_standardized_ridge_transition,
    transition_metrics,
    load_operator_package,
    save_operator_package,
)


def _linear_fixture() -> tuple[list[list[float]], list[list[float]]]:
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(80, 4))
    operator = np.asarray(
        [[1.2, 0.0], [0.0, -0.8], [0.5, 0.4], [-0.2, 0.3]]
    )
    targets = inputs @ operator
    return inputs.tolist(), targets.tolist()


def test_standardized_and_reduced_rank_packages_apply_consistently() -> None:
    inputs, targets = _linear_fixture()
    package = fit_standardized_ridge_transition(inputs, targets, 1e-6)
    predictions = apply_transition_package(inputs, package)
    metrics = transition_metrics(
        targets,
        predictions,
        feature_names=["rhythm_score", "conduction_score"],
        thresholds={"rhythm_score": 0.0, "conduction_score": 0.0},
    )
    assert metrics["mean_absolute_error"] < 1e-5
    assert (
        metrics["per_feature"]["rhythm_score"]["threshold_direction_accuracy"]
        == 1.0
    )
    reduced = fit_reduced_rank_transition(
        inputs,
        targets,
        1e-6,
        output_rank=1,
    )
    assert reduced["output_rank"] == 1
    assert np.asarray(reduced["operator"]).shape == (4, 2)


@pytest.mark.parametrize("method", ["elastic_net", "huber"])
def test_robust_transition_challengers_are_finite(method: str) -> None:
    inputs, targets = _linear_fixture()
    package = fit_robust_transition(
        inputs,
        targets,
        method=method,
        alpha=0.0001,
    )
    predictions = np.asarray(apply_transition_package(inputs, package))
    assert predictions.shape == (80, 2)
    assert np.isfinite(predictions).all()


def test_sign_constraints_are_enforced() -> None:
    inputs, targets = _linear_fixture()
    package = fit_sign_constrained_transition(
        inputs,
        targets,
        lambda_value=1e-6,
        sign_constraints={
            0: [1, 0, 1, -1],
            1: [0, -1, 1, 1],
        },
    )
    operator = np.asarray(package["operator"])
    assert operator[0, 0] >= 0
    assert operator[2, 0] >= 0
    assert operator[3, 0] <= 0
    assert operator[1, 1] <= 0


def test_masked_robust_transition_preserves_intercept_through_npz(tmp_path) -> None:
    inputs, targets = _linear_fixture()
    masked = [list(row) for row in targets]
    masked[0][0] = None
    package = fit_masked_robust_transition(
        inputs,
        masked,
        method="ridge",
        alpha=1e-6,
        minimum_target_rows=10,
    )
    predictions = np.asarray(apply_transition_package(inputs, package))
    assert predictions.shape == (80, 2)
    assert np.isfinite(predictions).all()
    path = save_operator_package(tmp_path / "operator.npz", package)
    restored = load_operator_package(path)
    assert restored["method"] == package["method"]
    assert np.allclose(restored["intercept"], package["intercept"])
    assert np.allclose(
        apply_transition_package(inputs, restored),
        predictions,
    )


def test_transition_validation_diagnostics_report_feature_fidelity() -> None:
    training = [
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ]
    truth = [[0.5, 1.5], [2.5, 3.5], [None, 4.0]]
    predicted = [[0.6, 1.4], [2.4, 3.6], [0.0, 4.1]]

    diagnostics = _transition_validation_diagnostics(
        truth,
        predicted,
        ["rhythm_score", "qrs_duration"],
        training,
    )

    assert diagnostics["threshold_source"] == "training_target_median_diagnostic"
    assert 0.0 <= diagnostics["exact_semantic_state_accuracy"] <= 1.0
    assert (
        diagnostics["per_semantic_feature"]["rhythm_score"]["observations"]
        == 2
    )
    assert (
        diagnostics["per_semantic_feature"]["qrs_duration"]["mae"]
        == pytest.approx(0.1)
    )
