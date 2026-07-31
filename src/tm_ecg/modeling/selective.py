"""Secondary selective-risk and split-conformal compatibility audits."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def risk_coverage_curve(
    truth: object,
    predictions: object,
    confidence: Sequence[float],
    *,
    coverages: Sequence[float] = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50),
) -> dict[str, object]:
    """Report exact-set risk at fixed coverages without changing all-row accuracy."""

    import numpy as np  # type: ignore

    y = np.asarray(truth, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    scores = np.asarray(confidence, dtype=float).reshape(-1)
    if y.shape != predicted.shape or y.ndim != 2 or len(y) != len(scores):
        raise ValueError("Selective inputs must be aligned")
    if not np.isfinite(scores).all():
        raise ValueError("Selective confidence must be finite")
    exact = np.all(y == predicted, axis=1)
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows: list[dict[str, object]] = []
    for requested in coverages:
        if not 0.0 < requested <= 1.0:
            raise ValueError("Coverage values must be in (0, 1]")
        retained = max(1, int(np.ceil(requested * len(y))))
        selected = order[:retained]
        accuracy = float(exact[selected].mean())
        rows.append(
            {
                "requested_coverage": float(requested),
                "retained_rows": retained,
                "realized_coverage": retained / len(y),
                "exact_subset_accuracy": accuracy,
                "exact_set_risk": 1.0 - accuracy,
                "minimum_retained_confidence": float(scores[selected].min()),
            }
        )
    return {
        "all_row_records": len(y),
        "all_row_exact_successes": int(exact.sum()),
        "all_row_exact_subset_accuracy": float(exact.mean()),
        "all_row_metric_remains_primary": True,
        "abstentions_count_as_errors_in_primary_metric": True,
        "curve": rows,
    }


def calibrate_abstention_threshold(
    truth: object,
    predictions: object,
    confidence: Sequence[float],
    *,
    maximum_selective_risk: float,
    minimum_coverage: float,
) -> dict[str, object]:
    """Choose the highest-coverage threshold satisfying a development risk cap."""

    import numpy as np  # type: ignore

    y = np.asarray(truth, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    scores = np.asarray(confidence, dtype=float).reshape(-1)
    if y.shape != predicted.shape or len(y) != len(scores):
        raise ValueError("Abstention calibration inputs must align")
    exact = np.all(y == predicted, axis=1)
    candidates = np.unique(scores)
    eligible: list[tuple[float, float, float, int]] = []
    for threshold in candidates:
        retained = scores >= threshold
        coverage = float(retained.mean())
        if not retained.any() or coverage < minimum_coverage:
            continue
        risk = 1.0 - float(exact[retained].mean())
        if risk <= maximum_selective_risk:
            eligible.append((coverage, -risk, -float(threshold), int(retained.sum())))
    if not eligible:
        return {
            "status": "no_threshold_satisfies_constraints",
            "threshold": None,
            "maximum_selective_risk": maximum_selective_risk,
            "minimum_coverage": minimum_coverage,
        }
    coverage, negative_risk, negative_threshold, retained_rows = max(eligible)
    return {
        "status": "ok",
        "threshold": -negative_threshold,
        "coverage": coverage,
        "selective_risk": -negative_risk,
        "retained_rows": retained_rows,
        "maximum_selective_risk": maximum_selective_risk,
        "minimum_coverage": minimum_coverage,
        "fit_partition": "development_validation_only",
    }


def split_conformal_candidate_sets(
    calibration_candidate_probabilities: object,
    calibration_true_candidate_indices: Sequence[int],
    evaluation_candidate_probabilities: object,
    *,
    alpha: float = 0.10,
    calibration_patient_groups: Sequence[object] | None = None,
) -> dict[str, object]:
    """Construct candidate-label-set prediction sets with grouped calibration."""

    import numpy as np  # type: ignore

    calibration = np.asarray(calibration_candidate_probabilities, dtype=float)
    evaluation = np.asarray(evaluation_candidate_probabilities, dtype=float)
    true_indices = np.asarray(calibration_true_candidate_indices, dtype=int)
    if (
        calibration.ndim != 2
        or evaluation.ndim != 2
        or calibration.shape[1] != evaluation.shape[1]
        or len(calibration) != len(true_indices)
    ):
        raise ValueError("Conformal candidate probabilities must align")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if (true_indices < 0).any() or (true_indices >= calibration.shape[1]).any():
        raise ValueError("True candidate indices are out of range")
    row_scores = 1.0 - calibration[np.arange(len(calibration)), true_indices]
    if calibration_patient_groups is None:
        grouped_scores = row_scores
        group_count = len(row_scores)
    else:
        if len(calibration_patient_groups) != len(row_scores):
            raise ValueError("Calibration patient groups must align")
        scores_by_group: dict[str, list[float]] = defaultdict(list)
        for group, score in zip(
            calibration_patient_groups,
            row_scores,
            strict=True,
        ):
            scores_by_group[str(group)].append(float(score))
        grouped_scores = np.asarray(
            [max(values) for values in scores_by_group.values()],
            dtype=float,
        )
        group_count = len(scores_by_group)
    rank = min(
        int(np.ceil((len(grouped_scores) + 1) * (1.0 - alpha))),
        len(grouped_scores),
    )
    quantile = float(np.partition(grouped_scores, rank - 1)[rank - 1])
    included = (1.0 - evaluation) <= quantile
    # Always retain the most likely candidate if finite-sample calibration
    # otherwise creates an empty set through extreme numeric rounding.
    empty = ~included.any(axis=1)
    included[empty, evaluation[empty].argmax(axis=1)] = True
    return {
        "alpha": alpha,
        "coverage_target": 1.0 - alpha,
        "calibration_rows": len(calibration),
        "calibration_patient_groups": group_count,
        "grouped_patient_max_nonconformity": (
            calibration_patient_groups is not None
        ),
        "nonconformity_quantile": quantile,
        "candidate_membership": included.astype(int),
        "candidate_set_sizes": included.sum(axis=1),
        "extra_candidates_do_not_count_as_exact": True,
    }

