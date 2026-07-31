"""Optional symmetry-consistency diagnostics for transition matrices.

These helpers do not impose clinically invalid transformations. They only audit whether a
transition operator is consistent with supplied feature-space generators, following the
row-vector convention B_hat = A @ T.
"""

from __future__ import annotations

import math
from typing import Sequence


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    if not left or not right:
        return []
    rows = len(left)
    inner = len(left[0])
    if any(len(row) != inner for row in left):
        raise ValueError("left matrix is ragged")
    if any(len(row) != len(right[0]) for row in right):
        raise ValueError("right matrix is ragged")
    if len(right) != inner:
        raise ValueError("matrix shapes are incompatible")
    cols = len(right[0])
    return [[sum(float(left[i][k]) * float(right[k][j]) for k in range(inner)) for j in range(cols)] for i in range(rows)]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix, strict=False)]


def _frobenius(matrix: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(float(value) ** 2 for row in matrix for value in row))


def _subtract(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(left) != len(right) or (left and len(left[0]) != len(right[0])):
        raise ValueError("matrix shapes must match")
    return [[float(a) - float(b) for a, b in zip(row_l, row_r, strict=False)] for row_l, row_r in zip(left, right, strict=False)]


def symmetry_defect(
    operator: Sequence[Sequence[float]],
    generator_a: Sequence[Sequence[float]],
    generator_b: Sequence[Sequence[float]],
) -> float:
    """Return normalized row-vector intertwining defect ||G_A T - T G_B||_F / ||T||_F."""

    left = _matmul(generator_a, operator)
    right = _matmul(operator, generator_b)
    denom = _frobenius(operator) or 1.0
    return _frobenius(_subtract(left, right)) / denom


def estimate_linear_generator(
    original: Sequence[Sequence[float]],
    transformed: Sequence[Sequence[float]],
    epsilon: float = 1.0,
    ridge: float = 1e-8,
) -> list[list[float]]:
    """Estimate a first-order feature generator from paired rows using normal equations.

    The returned generator G solves approximately (transformed-original)/epsilon = original @ G.
    This compact implementation uses NumPy when available through the project dependency stack.
    """

    import numpy as np

    x = np.asarray(original, dtype=float)
    y = (np.asarray(transformed, dtype=float) - x) / float(epsilon)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("original and transformed feature matrices must have the same 2D shape")
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    rhs = x.T @ y
    generator = np.linalg.solve(gram, rhs)
    return generator.tolist()


def transformation_consistency(
    operator: Sequence[Sequence[float]],
    a_original: Sequence[Sequence[float]],
    a_transformed: Sequence[Sequence[float]],
    b_original: Sequence[Sequence[float]] | None = None,
    b_transformed: Sequence[Sequence[float]] | None = None,
) -> dict[str, float]:
    """Audit explanation stability under paired feature-space transformations."""

    pred_original = _matmul(a_original, operator)
    pred_transformed = _matmul(a_transformed, operator)
    delta_pred = _subtract(pred_transformed, pred_original)
    pred_delta_norm = _frobenius(delta_pred)
    result = {"predicted_explanation_delta_norm": pred_delta_norm}
    if b_original is not None and b_transformed is not None:
        true_delta = _subtract(b_transformed, b_original)
        err = _subtract(delta_pred, true_delta)
        result["observed_interpretable_delta_norm"] = _frobenius(true_delta)
        result["transformation_delta_error"] = _frobenius(err)
    return result


def ecg_safe_waveform_perturbations(
    signal: object,
    *,
    sampling_rate_hz: float,
    seed: int = 17,
    amplitude_scale: float = 1.01,
    baseline_shift: float = 0.005,
    translation_ms: float = 4.0,
    noise_scale: float = 0.002,
    redundant_lead_index: int | None = None,
    beat_indices: Sequence[int] | None = None,
    beat_jitter_ms: float = 2.0,
) -> dict[str, object]:
    """Generate only small transformations expected to preserve interpretation."""

    import numpy as np  # type: ignore

    matrix = np.asarray(signal, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("ECG perturbations require samples-by-leads data")
    rng = np.random.default_rng(seed)
    shift = max(int(round(translation_ms * sampling_rate_hz / 1000.0)), 1)
    translated = np.roll(matrix, shift=shift, axis=0)
    translated[:shift] = matrix[:shift]
    noisy = matrix + rng.normal(
        0.0,
        noise_scale * (np.nanstd(matrix, axis=0, keepdims=True) + 1e-9),
        size=matrix.shape,
    )
    result: dict[str, object] = {
        "small_baseline_shift": matrix + baseline_shift,
        "small_amplitude_scaling": matrix * amplitude_scale,
        "minor_temporal_translation": translated,
        "low_amplitude_additive_noise": noisy,
    }
    if redundant_lead_index is not None:
        if not 0 <= redundant_lead_index < matrix.shape[1]:
            raise ValueError("redundant_lead_index is outside the lead matrix")
        lead_dropout = matrix.copy()
        lead_dropout[:, redundant_lead_index] = np.nan
        result["single_declared_redundant_lead_dropout"] = lead_dropout
    if beat_indices:
        beat_jitter = matrix.copy()
        jitter_samples = max(
            1,
            int(round(beat_jitter_ms * sampling_rate_hz / 1000.0)),
        )
        radius = max(2, int(round(0.06 * sampling_rate_hz)))
        for beat_number, center in enumerate(beat_indices):
            direction = -1 if beat_number % 2 else 1
            start = max(0, int(center) - radius)
            stop = min(len(matrix), int(center) + radius + 1)
            segment = matrix[start:stop].copy()
            shifted = np.roll(segment, direction * jitter_samples, axis=0)
            if direction > 0:
                shifted[:jitter_samples] = segment[:jitter_samples]
            else:
                shifted[-jitter_samples:] = segment[-jitter_samples:]
            beat_jitter[start:stop] = shifted
        result["beat_aligned_jitter"] = beat_jitter
    return result


def semantic_perturbation_audit(
    operator: Sequence[Sequence[float]],
    original_features: Sequence[Sequence[float]],
    perturbed_features: dict[str, Sequence[Sequence[float]]],
    *,
    semantic_thresholds: Sequence[float] | None = None,
    original_routes: Sequence[str] | None = None,
    perturbed_routes: dict[str, Sequence[str]] | None = None,
    original_truth: Sequence[Sequence[float]] | None = None,
    perturbed_truth: dict[str, Sequence[Sequence[float]]] | None = None,
) -> dict[str, object]:
    """Measure semantic, predicate, route, and fidelity stability."""

    import numpy as np  # type: ignore

    original_prediction = np.asarray(
        _matmul(original_features, operator),
        dtype=float,
    )
    thresholds = (
        np.asarray(semantic_thresholds, dtype=float)
        if semantic_thresholds is not None
        else None
    )
    if thresholds is not None and len(thresholds) != original_prediction.shape[1]:
        raise ValueError("semantic_thresholds must align with transition outputs")
    rows: dict[str, dict[str, float | None]] = {}
    for name, features in sorted(perturbed_features.items()):
        prediction = np.asarray(_matmul(features, operator), dtype=float)
        if prediction.shape != original_prediction.shape:
            raise ValueError(f"Perturbation {name} changed the row/output shape")
        difference = prediction - original_prediction
        semantic_error = float(np.sqrt(np.mean(difference**2)))
        predicate_flip = (
            float(
                (
                    (prediction >= thresholds)
                    != (original_prediction >= thresholds)
                ).mean()
            )
            if thresholds is not None
            else None
        )
        route_flip = None
        if original_routes is not None and perturbed_routes is not None:
            routes = perturbed_routes.get(name)
            if routes is None or len(routes) != len(original_routes):
                raise ValueError(f"Perturbation {name} routes are missing/misaligned")
            route_flip = sum(
                left != right
                for left, right in zip(original_routes, routes, strict=True)
            ) / len(original_routes)
        fidelity_delta = None
        if original_truth is not None and perturbed_truth is not None:
            base_truth = np.asarray(original_truth, dtype=float)
            transformed_truth = np.asarray(perturbed_truth[name], dtype=float)
            base_mae = float(np.mean(np.abs(original_prediction - base_truth)))
            transformed_mae = float(np.mean(np.abs(prediction - transformed_truth)))
            fidelity_delta = transformed_mae - base_mae
        rows[name] = {
            "semantic_stability_error": semantic_error,
            "predicate_flip_rate": predicate_flip,
            "rule_route_flip_rate": route_flip,
            "transition_fidelity_delta": fidelity_delta,
        }
    return {
        "perturbations": rows,
        "maximum_semantic_stability_error": max(
            (
                float(row["semantic_stability_error"])
                for row in rows.values()
            ),
            default=0.0,
        ),
        "prohibited_transformations": [
            "arbitrary_time_warping",
            "lead_permutation",
            "diagnosis_changing_amplitude_or_rate_modification",
        ],
    }
