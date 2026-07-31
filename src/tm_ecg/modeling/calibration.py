"""Leakage-safe validation-fold calibration for multilabel model heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _sigmoid(values):  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _binary_nll(targets, probabilities, eps: float = 1e-7) -> float:  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    probability = np.clip(probabilities, eps, 1.0 - eps)
    return float(
        -np.mean(targets * np.log(probability) + (1.0 - targets) * np.log(1.0 - probability))
    )


def _ece(targets, probabilities, bins: int = 10) -> tuple[float, list[dict[str, float | int]]]:  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    y = np.asarray(targets, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    curve: list[dict[str, float | int]] = []
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        error += count / len(p) * abs(confidence - observed)
        curve.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": confidence,
                "observed_frequency": observed,
            }
        )
    return error, curve


def _metrics(targets, probabilities) -> dict[str, object]:  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    calibration_error, curve = _ece(targets, probabilities)
    return {
        "negative_log_likelihood": _binary_nll(targets, probabilities),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "expected_calibration_error": calibration_error,
        "reliability_curve": curve,
    }


def _threshold_performance(targets, probabilities, labels: Sequence[str]) -> dict[str, object]:  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    output: dict[str, object] = {}
    candidates = np.linspace(0.05, 0.95, 19)
    for column, label in enumerate(labels):
        y = targets[:, column].astype(int)
        p = probabilities[:, column]
        if len(set(y.tolist())) < 2:
            output[str(label)] = {
                "status": "not_estimable_single_class",
                "threshold": 0.5,
            }
            continue
        scored = []
        for threshold in candidates:
            prediction = p >= threshold
            tp = int(((prediction == 1) & (y == 1)).sum())
            tn = int(((prediction == 0) & (y == 0)).sum())
            fp = int(((prediction == 1) & (y == 0)).sum())
            fn = int(((prediction == 0) & (y == 1)).sum())
            sensitivity = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            scored.append((min(sensitivity, specificity), sensitivity + specificity, -threshold, threshold, sensitivity, specificity, tp, tn, fp, fn))
        best = max(scored)
        output[str(label)] = {
            "status": "ok",
            "selection_objective": "maximize minimum(sensitivity, specificity), then balanced accuracy",
            "threshold": float(best[3]),
            "sensitivity": float(best[4]),
            "specificity": float(best[5]),
            "tp": int(best[6]),
            "tn": int(best[7]),
            "fp": int(best[8]),
            "fn": int(best[9]),
        }
    return output


def fit_temperature_calibration(
    logits: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    labels: Sequence[str],
) -> dict[str, object]:
    """Fit a scalar temperature using validation rows only.

    Identity calibration is retained unless a temperature improves binary
    negative log-likelihood, implementing the simplest-effective-method rule.
    """

    import numpy as np  # type: ignore

    z = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=float)
    if z.ndim != 2 or y.ndim != 2 or z.shape != y.shape or z.shape[1] != len(labels):
        raise ValueError("Calibration logits, targets, and labels must have aligned 2D shapes")
    if len(z) < 2:
        return {
            "status": "not_estimable_insufficient_validation_rows",
            "validation_rows": len(z),
            "temperature": 1.0,
        }
    temperatures = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
    identity_probabilities = _sigmoid(z)
    identity_nll = _binary_nll(y, identity_probabilities)
    candidates = [
        (_binary_nll(y, _sigmoid(z / temperature)), float(temperature))
        for temperature in temperatures
    ]
    best_nll, best_temperature = min(candidates)
    if best_nll >= identity_nll - 1e-8:
        best_temperature = 1.0
    calibrated = _sigmoid(z / best_temperature)
    return {
        "status": "ok",
        "method": "temperature_scaling" if best_temperature != 1.0 else "identity",
        "temperature": best_temperature,
        "fit_partition": "validation_only",
        "validation_rows": len(z),
        "labels": list(labels),
        "before": _metrics(y, identity_probabilities),
        "after": _metrics(y, calibrated),
        "class_thresholds": _threshold_performance(y, calibrated, labels),
    }


def apply_temperature(logits, calibration: dict[str, object]):  # type: ignore[no-untyped-def]
    temperature = float(calibration.get("temperature", 1.0) or 1.0)
    return logits / temperature


def compatibility_from_calibrated_axes(
    axis_scores: dict[str, dict[str, float]],
    calibration: dict[str, object],
) -> tuple[str | None, dict[str, object]]:
    """Produce the historical label only after calibrated axis inference."""

    axis_calibration = dict(calibration.get("axes", {}))

    def active(axis: str, label: str) -> tuple[bool, float, float]:
        score = float(axis_scores.get(axis, {}).get(label, 0.0))
        head = dict(axis_calibration.get(axis, {}))
        threshold_row = dict(dict(head.get("class_thresholds", {})).get(label, {}))
        threshold = float(threshold_row.get("threshold", 0.5))
        return score >= threshold, score, threshold

    candidates = [
        ("Paced", *active("pacing", "present")),
        ("AF", *active("rhythm", "af")),
        ("AFL", *active("rhythm", "afl")),
        ("PVC", *active("ectopy", "pvc")),
        ("APB", *active("ectopy", "apb")),
        ("RBBB spectrum", *active("conduction", "rbbb_spectrum")),
        ("LBBB spectrum", *active("conduction", "lbbb_spectrum")),
    ]
    active_candidates = [item for item in candidates if item[1]]
    if active_candidates:
        selected = max(active_candidates, key=lambda item: (item[2], item[0]))
        return selected[0], {
            "source": "calibrated_axis_projection",
            "axis_candidates": candidates,
        }
    repolarization = axis_scores.get("repolarization", {})
    if any(
        label != "none" and active("repolarization", label)[0]
        for label in repolarization
    ):
        return "Other / unmapped", {
            "source": "calibrated_axis_projection",
            "reason": "repolarization finding has no legacy compatibility class",
        }
    sinus_active, sinus_score, sinus_threshold = active("rhythm", "sinus")
    if sinus_active:
        return "Normal", {
            "source": "calibrated_axis_projection",
            "sinus_score": sinus_score,
            "sinus_threshold": sinus_threshold,
        }
    return None, {
        "source": "calibrated_axis_projection",
        "reason": "no axis probability crossed its validation-fold threshold",
    }


@dataclass(slots=True)
class SelectedBinaryCalibrator:
    """A fitted, support-aware binary probability calibrator."""

    method: str
    model: object | None
    selection_metrics: dict[str, dict[str, float]]
    support: int
    positive_support: int
    negative_support: int

    def predict(self, probabilities):  # type: ignore[no-untyped-def]
        return _apply_binary_calibrator(
            self.method,
            self.model,
            probabilities,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "method": self.method,
            "selection_metrics": self.selection_metrics,
            "support": self.support,
            "positive_support": self.positive_support,
            "negative_support": self.negative_support,
            "selection_partition": "nested_development_calibration",
        }


def _calibration_features(method: str, probabilities):  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p))
    if method == "platt":
        return logit.reshape(-1, 1)
    if method == "beta":
        return np.column_stack((np.log(p), -np.log1p(-p)))
    return p


def _fit_binary_calibrator(
    method: str,
    probabilities,
    targets,
):  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    y = np.asarray(targets, dtype=int).reshape(-1)
    if method == "identity":
        return None
    if method in {"platt", "beta"}:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        model = LogisticRegression(
            C=1.0,
            class_weight=None,
            max_iter=1000,
            solver="lbfgs",
            random_state=17,
        )
        model.fit(_calibration_features(method, probabilities), y)
        return model
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        model = IsotonicRegression(
            y_min=1e-6,
            y_max=1.0 - 1e-6,
            out_of_bounds="clip",
        )
        model.fit(_calibration_features(method, probabilities), y)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def _apply_binary_calibrator(
    method: str,
    model: object | None,
    probabilities,
):  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    if method == "identity" or model is None:
        return p
    if method in {"platt", "beta"}:
        return np.clip(
            model.predict_proba(_calibration_features(method, p))[:, 1],  # type: ignore[attr-defined]
            1e-6,
            1.0 - 1e-6,
        )
    if method == "isotonic":
        return np.clip(
            model.predict(_calibration_features(method, p)),  # type: ignore[attr-defined]
            1e-6,
            1.0 - 1e-6,
        )
    raise ValueError(f"Unknown calibration method: {method}")


def _selection_score(targets, probabilities) -> dict[str, float]:  # type: ignore[no-untyped-def]
    import numpy as np  # type: ignore

    y = np.asarray(targets, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    log_loss = _binary_nll(y, p)
    ece = float(_ece(y, p)[0])
    return {
        "brier_score": brier,
        "negative_log_likelihood": log_loss,
        "expected_calibration_error": ece,
        "combined_selection_score": brier + 0.25 * log_loss + ece,
    }


def fit_best_binary_calibrator(
    probabilities,
    targets,
    *,
    random_state: int = 17,
) -> SelectedBinaryCalibrator:  # type: ignore[no-untyped-def]
    """Select identity, Platt, isotonic, or beta calibration by nested CV.

    Isotonic is considered only with at least 200 rows and 30 examples of each
    class.  Sparse labels fall back to identity rather than fitting an unstable
    calibration curve.
    """

    import numpy as np  # type: ignore
    from sklearn.model_selection import StratifiedKFold  # type: ignore

    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    y = np.asarray(targets, dtype=int).reshape(-1)
    if len(p) != len(y) or not len(p):
        raise ValueError("Calibration probabilities and targets must be aligned")
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive < 8 or negative < 8:
        return SelectedBinaryCalibrator(
            method="identity",
            model=None,
            selection_metrics={"identity": _selection_score(y, p)},
            support=len(y),
            positive_support=positive,
            negative_support=negative,
        )

    methods = ["identity", "platt", "beta"]
    if len(y) >= 200 and positive >= 30 and negative >= 30:
        methods.append("isotonic")
    folds = min(5, positive, negative)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    predictions = {
        method: np.zeros(len(y), dtype=float) for method in methods
    }
    for train_indices, validation_indices in splitter.split(p, y):
        for method in methods:
            model = _fit_binary_calibrator(
                method,
                p[train_indices],
                y[train_indices],
            )
            predictions[method][validation_indices] = _apply_binary_calibrator(
                method,
                model,
                p[validation_indices],
            )
    selection_metrics = {
        method: _selection_score(y, predictions[method]) for method in methods
    }
    simplicity = {"identity": 0, "platt": 1, "beta": 2, "isotonic": 3}
    selected = min(
        methods,
        key=lambda method: (
            selection_metrics[method]["combined_selection_score"],
            simplicity[method],
        ),
    )
    identity_score = selection_metrics["identity"]["combined_selection_score"]
    selected_score = selection_metrics[selected]["combined_selection_score"]
    if selected_score >= identity_score - 1e-4:
        selected = "identity"
    final_model = _fit_binary_calibrator(selected, p, y)
    return SelectedBinaryCalibrator(
        method=selected,
        model=final_model,
        selection_metrics=selection_metrics,
        support=len(y),
        positive_support=positive,
        negative_support=negative,
    )
