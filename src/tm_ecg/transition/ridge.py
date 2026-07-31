"""Reduced-rank ridge transition operator utilities."""

from __future__ import annotations

import json
from pathlib import Path

from tm_ecg.io.common import ensure_parent


def singular_value_keep_mask(values: list[float], m: int, r: int, eps_machine: float | None = None) -> list[bool]:
    if not values:
        return []
    eps = eps_machine if eps_machine is not None else 2.220446049250313e-16
    threshold = max(m, r) * eps * values[0]
    return [value > threshold for value in values]


def fit_ridge_transition(
    a_rows: list[list[float]],
    b_rows: list[list[float]],
    lambda_value: float,
    rank_cap: int | None = None,
) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("fit_ridge_transition requires numpy") from exc

    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(b_rows, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have the same row count")

    u, singular_values, vt = np.linalg.svd(a, full_matrices=False)
    keep = singular_value_keep_mask(singular_values.tolist(), a.shape[0], a.shape[1])
    if rank_cap is not None:
        keep = [flag and idx < rank_cap for idx, flag in enumerate(keep)]
    kept_idx = [idx for idx, flag in enumerate(keep) if flag]
    if not kept_idx:
        raise RuntimeError("No singular values passed the truncation threshold")

    u_r = u[:, kept_idx]
    s_r = singular_values[kept_idx]
    vt_r = vt[kept_idx, :]
    diagonal = np.diag(s_r / (s_r**2 + lambda_value))
    operator = vt_r.T @ diagonal @ u_r.T @ b
    return {
        "operator": operator.tolist(),
        "singular_values": singular_values.tolist(),
        "retained_rank": len(kept_idx),
        "lambda_value": lambda_value,
    }


def fit_masked_ridge_transition(
    a_rows: list[list[float]],
    b_rows: list[list[float | None]],
    lambda_value: float,
    *,
    rank_cap: int | None = None,
    minimum_target_rows: int = 10,
) -> dict[str, object]:
    """Fit each B output on its observed rows without complete-case attrition.

    Outputs sharing an observation mask are fitted together so the SVD is
    reused.  Unavailable outputs receive a zero column (the transformed-space
    training mean) and an explicit ``not_estimable`` status in the package.
    """

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("fit_masked_ridge_transition requires numpy") from exc

    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(
        [
            [float("nan") if value is None else float(value) for value in row]
            for row in b_rows
        ],
        dtype=float,
    )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have the same row count")
    if not np.isfinite(a).all():
        raise ValueError("A must contain only finite values")
    if minimum_target_rows < 2:
        raise ValueError("minimum_target_rows must be at least two")

    operator = np.zeros((a.shape[1], b.shape[1]), dtype=float)
    observation_masks = np.isfinite(b)
    mask_groups: dict[bytes, list[int]] = {}
    mask_lookup: dict[bytes, object] = {}
    for column in range(b.shape[1]):
        mask = observation_masks[:, column]
        key = mask.tobytes()
        mask_groups.setdefault(key, []).append(column)
        mask_lookup[key] = mask

    target_support = [int(observation_masks[:, column].sum()) for column in range(b.shape[1])]
    target_retained_rank = [0] * b.shape[1]
    target_status = ["not_estimable_insufficient_observations"] * b.shape[1]
    for key, columns in mask_groups.items():
        mask = np.asarray(mask_lookup[key], dtype=bool)
        if int(mask.sum()) < minimum_target_rows:
            continue
        result = fit_ridge_transition(
            a[mask].tolist(),
            b[mask][:, columns].tolist(),
            lambda_value,
            rank_cap=rank_cap,
        )
        fitted = np.asarray(result["operator"], dtype=float)
        operator[:, columns] = fitted
        for column in columns:
            target_retained_rank[column] = int(result["retained_rank"])
            target_status[column] = "ok"

    singular_values = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    return {
        "method": "masked_svd_ridge",
        "operator": operator.tolist(),
        "singular_values": singular_values.tolist(),
        "retained_rank": max(target_retained_rank, default=0),
        "lambda_value": lambda_value,
        "minimum_target_rows": minimum_target_rows,
        "target_support": target_support,
        "target_retained_rank": target_retained_rank,
        "target_status": target_status,
        "observation_mask_groups": len(mask_groups),
        "missing_target_policy": "featurewise_observed_rows_only",
    }


def fit_masked_robust_transition(
    a_rows: list[list[float]],
    b_rows: list[list[float | None]],
    *,
    method: str,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    minimum_target_rows: int = 10,
) -> dict[str, object]:
    """Fit finite-row featurewise Ridge, ElasticNet, or Huber challengers."""

    import numpy as np  # type: ignore
    from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(
        [
            [float("nan") if value is None else float(value) for value in row]
            for row in b_rows
        ],
        dtype=float,
    )
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError("A and B must be aligned 2D matrices")
    if not np.isfinite(a).all():
        raise ValueError("A must contain only finite values")
    operator = np.zeros((a.shape[1], b.shape[1]), dtype=float)
    intercept = np.zeros(b.shape[1], dtype=float)
    support: list[int] = []
    status: list[str] = []
    iterations: list[int] = []
    for column in range(b.shape[1]):
        observed = np.isfinite(b[:, column])
        count = int(observed.sum())
        support.append(count)
        if count < minimum_target_rows:
            status.append("not_estimable_insufficient_observations")
            iterations.append(0)
            continue
        x_scaler = StandardScaler().fit(a[observed])
        y_values = b[observed, column]
        y_mean = float(y_values.mean())
        y_scale = float(y_values.std())
        if y_scale <= 1e-12:
            intercept[column] = y_mean
            status.append("constant_target")
            iterations.append(0)
            continue
        x = x_scaler.transform(a[observed])
        y = (y_values - y_mean) / y_scale
        if method == "ridge":
            model = Ridge(alpha=alpha)
        elif method == "elastic_net":
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=5000,
                random_state=17,
            )
        elif method == "huber":
            model = HuberRegressor(alpha=alpha, max_iter=500)
        else:
            raise ValueError("method must be ridge, elastic_net, or huber")
        model.fit(x, y)
        standardized_coefficient = np.asarray(model.coef_, dtype=float)
        original_coefficient = (
            standardized_coefficient / x_scaler.scale_ * y_scale
        )
        operator[:, column] = original_coefficient
        intercept[column] = (
            y_mean
            + y_scale * float(model.intercept_)
            - float(x_scaler.mean_ @ original_coefficient)
        )
        status.append("ok")
        iterations.append(int(getattr(model, "n_iter_", 0) or 0))
    return {
        "method": f"masked_standardized_{method}",
        "operator": operator.tolist(),
        "intercept": intercept.tolist(),
        "alpha": alpha,
        "lambda_value": alpha,
        "l1_ratio": l1_ratio if method == "elastic_net" else None,
        "minimum_target_rows": minimum_target_rows,
        "target_support": support,
        "target_status": status,
        "target_retained_rank": [
            int(np.count_nonzero(operator[:, column]))
            if status[column] == "ok"
            else 0
            for column in range(b.shape[1])
        ],
        "iterations": iterations,
        "retained_rank": int(np.linalg.matrix_rank(operator)),
        "missing_target_policy": "featurewise_observed_rows_only",
    }


def reduce_transition_output_rank(
    package: dict[str, object],
    *,
    output_rank: int,
) -> dict[str, object]:
    """Apply a low-rank output projection to an already fitted package."""

    import numpy as np  # type: ignore

    operator = np.asarray(package["operator"], dtype=float)
    if output_rank < 1:
        raise ValueError("output_rank must be positive")
    _, singular_values, vt = np.linalg.svd(operator, full_matrices=False)
    rank = min(output_rank, len(singular_values))
    projection = vt[:rank].T @ vt[:rank]
    reduced = dict(package)
    reduced["operator"] = (operator @ projection).tolist()
    if "intercept" in package:
        reduced["intercept"] = (
            np.asarray(package["intercept"], dtype=float) @ projection
        ).tolist()
    reduced.update(
        {
            "method": f"{package.get('method', 'transition')}::reduced_rank",
            "output_rank": rank,
            "output_singular_values": singular_values.tolist(),
            "retained_rank": int(np.linalg.matrix_rank(operator @ projection)),
        }
    )
    return reduced


def apply_transition(a_rows: list[list[float]], operator: list[list[float]]) -> list[list[float]]:
    output = []
    for row in a_rows:
        output_row = []
        for column_idx in range(len(operator[0])):
            output_row.append(sum(row[k] * operator[k][column_idx] for k in range(len(row))))
        output.append(output_row)
    return output


def fit_standardized_ridge_transition(
    a_rows: list[list[float]],
    b_rows: list[list[float]],
    lambda_value: float,
    *,
    rank_cap: int | None = None,
) -> dict[str, object]:
    """Fit ridge after train-only standardization and retain the transform."""

    import numpy as np  # type: ignore

    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(b_rows, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError("A and B must be aligned 2D matrices")
    a_mean = a.mean(axis=0)
    a_scale = a.std(axis=0)
    a_scale[a_scale <= 1e-12] = 1.0
    b_mean = b.mean(axis=0)
    b_scale = b.std(axis=0)
    b_scale[b_scale <= 1e-12] = 1.0
    standardized = fit_ridge_transition(
        ((a - a_mean) / a_scale).tolist(),
        ((b - b_mean) / b_scale).tolist(),
        lambda_value,
        rank_cap=rank_cap,
    )
    standardized.update(
        {
            "method": "standardized_ridge",
            "a_mean": a_mean.tolist(),
            "a_scale": a_scale.tolist(),
            "b_mean": b_mean.tolist(),
            "b_scale": b_scale.tolist(),
        }
    )
    return standardized


def fit_reduced_rank_transition(
    a_rows: list[list[float]],
    b_rows: list[list[float]],
    lambda_value: float,
    *,
    output_rank: int,
    input_rank_cap: int | None = None,
) -> dict[str, object]:
    """Fit ridge then reduce the predicted semantic output subspace."""

    import numpy as np  # type: ignore

    if output_rank < 1:
        raise ValueError("output_rank must be positive")
    ridge = fit_standardized_ridge_transition(
        a_rows,
        b_rows,
        lambda_value,
        rank_cap=input_rank_cap,
    )
    standardized_a = (
        np.asarray(a_rows, dtype=float) - np.asarray(ridge["a_mean"])
    ) / np.asarray(ridge["a_scale"])
    fitted = standardized_a @ np.asarray(ridge["operator"], dtype=float)
    _, singular_values, vt = np.linalg.svd(fitted, full_matrices=False)
    rank = min(output_rank, len(singular_values))
    projection = vt[:rank].T @ vt[:rank]
    operator = np.asarray(ridge["operator"], dtype=float) @ projection
    ridge.update(
        {
            "method": "standardized_reduced_rank_ridge",
            "operator": operator.tolist(),
            "output_rank": rank,
            "output_singular_values": singular_values.tolist(),
        }
    )
    return ridge


def fit_robust_transition(
    a_rows: list[list[float]],
    b_rows: list[list[float]],
    *,
    method: str,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
) -> dict[str, object]:
    """Fit featurewise ElasticNet or Huber robust transition challengers."""

    import numpy as np  # type: ignore
    from sklearn.linear_model import ElasticNet, HuberRegressor  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(b_rows, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError("A and B must be aligned 2D matrices")
    x_scaler = StandardScaler().fit(a)
    y_scaler = StandardScaler().fit(b)
    x = x_scaler.transform(a)
    y = y_scaler.transform(b)
    coefficients = np.zeros((a.shape[1], b.shape[1]), dtype=float)
    intercepts = np.zeros(b.shape[1], dtype=float)
    iterations: list[int] = []
    for column in range(b.shape[1]):
        if method == "elastic_net":
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=5000,
                random_state=17,
            )
        elif method == "huber":
            model = HuberRegressor(
                alpha=alpha,
                max_iter=500,
            )
        else:
            raise ValueError("method must be elastic_net or huber")
        model.fit(x, y[:, column])
        coefficients[:, column] = model.coef_
        intercepts[column] = float(model.intercept_)
        iterations.append(int(getattr(model, "n_iter_", 0) or 0))
    return {
        "method": f"standardized_{method}",
        "operator": coefficients.tolist(),
        "standardized_intercept": intercepts.tolist(),
        "a_mean": x_scaler.mean_.tolist(),
        "a_scale": x_scaler.scale_.tolist(),
        "b_mean": y_scaler.mean_.tolist(),
        "b_scale": y_scaler.scale_.tolist(),
        "alpha": alpha,
        "l1_ratio": l1_ratio if method == "elastic_net" else None,
        "iterations": iterations,
        "retained_rank": int(np.linalg.matrix_rank(coefficients)),
    }


def fit_sign_constrained_transition(
    a_rows: list[list[float]],
    b_rows: list[list[float]],
    *,
    lambda_value: float,
    sign_constraints: dict[int, list[int]],
) -> dict[str, object]:
    """Fit ridge-like least squares with per-output coefficient signs."""

    import numpy as np  # type: ignore
    from scipy.optimize import lsq_linear  # type: ignore

    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(b_rows, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError("A and B must be aligned 2D matrices")
    augmented_a = np.vstack(
        (a, np.sqrt(lambda_value) * np.eye(a.shape[1]))
    )
    operator = np.zeros((a.shape[1], b.shape[1]), dtype=float)
    statuses: list[int] = []
    for column in range(b.shape[1]):
        signs = sign_constraints.get(column, [0] * a.shape[1])
        if len(signs) != a.shape[1] or any(sign not in {-1, 0, 1} for sign in signs):
            raise ValueError("Sign constraints must provide -1/0/1 per input")
        lower = np.asarray(
            [0.0 if sign == 1 else -np.inf for sign in signs],
            dtype=float,
        )
        upper = np.asarray(
            [0.0 if sign == -1 else np.inf for sign in signs],
            dtype=float,
        )
        augmented_b = np.concatenate((b[:, column], np.zeros(a.shape[1])))
        result = lsq_linear(
            augmented_a,
            augmented_b,
            bounds=(lower, upper),
            lsmr_tol="auto",
            max_iter=500,
        )
        if not result.success:
            raise RuntimeError(
                f"Sign-constrained transition failed for target {column}: "
                f"{result.message}"
            )
        operator[:, column] = result.x
        statuses.append(int(result.status))
    return {
        "method": "sign_constrained_ridge",
        "operator": operator.tolist(),
        "lambda_value": lambda_value,
        "sign_constraints": sign_constraints,
        "solver_statuses": statuses,
        "retained_rank": int(np.linalg.matrix_rank(operator)),
    }


def apply_transition_package(
    a_rows: list[list[float]],
    package: dict[str, object],
) -> list[list[float]]:
    """Apply plain or standardized transition packages."""

    import numpy as np  # type: ignore

    a = np.asarray(a_rows, dtype=float)
    operator = np.asarray(package["operator"], dtype=float)
    if "a_mean" in package and "a_scale" in package:
        a = (a - np.asarray(package["a_mean"])) / np.asarray(package["a_scale"])
    output = a @ operator
    if "intercept" in package:
        output += np.asarray(package["intercept"], dtype=float)
    if "standardized_intercept" in package:
        output += np.asarray(package["standardized_intercept"])
    if "b_mean" in package and "b_scale" in package:
        output = output * np.asarray(package["b_scale"]) + np.asarray(
            package["b_mean"]
        )
    return output.tolist()


def transition_metrics(
    truth_rows: list[list[float]],
    predicted_rows: list[list[float]],
    *,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Report clinician-facing fidelity metrics for every semantic output."""

    import numpy as np  # type: ignore
    from scipy.stats import spearmanr  # type: ignore

    truth = np.asarray(truth_rows, dtype=float)
    predicted = np.asarray(predicted_rows, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 2:
        raise ValueError("Transition truth and prediction matrices must align")
    names = feature_names or [f"target_{index}" for index in range(truth.shape[1])]
    if len(names) != truth.shape[1]:
        raise ValueError("feature_names must align with transition targets")
    per_feature: dict[str, dict[str, float | None]] = {}
    for column, name in enumerate(names):
        target = truth[:, column]
        output = predicted[:, column]
        mae = float(np.mean(np.abs(target - output)))
        scale = float(np.std(target))
        residual = float(np.sum((target - output) ** 2))
        total = float(np.sum((target - target.mean()) ** 2))
        rank = spearmanr(target, output).statistic
        threshold = (thresholds or {}).get(name)
        per_feature[name] = {
            "mae": mae,
            "normalized_mae": mae / scale if scale > 1e-12 else None,
            "r_squared": 1.0 - residual / total if total > 1e-12 else None,
            "spearman_rank_correlation": (
                float(rank) if np.isfinite(rank) else None
            ),
            "threshold_direction_accuracy": (
                float(((target >= threshold) == (output >= threshold)).mean())
                if threshold is not None
                else None
            ),
        }
    return {
        "rows": len(truth),
        "features": names,
        "mean_absolute_error": float(np.mean(np.abs(truth - predicted))),
        "exact_semantic_state_accuracy": (
            float(
                np.all(
                    np.isclose(truth, predicted, rtol=0.0, atol=1e-12),
                    axis=1,
                ).mean()
            )
        ),
        "per_feature": per_feature,
    }


def save_operator_package(path: str | Path, payload: dict[str, object]) -> Path:
    destination = ensure_parent(path)
    if destination.suffix == ".npz":
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Saving NPZ transition operators requires numpy") from exc
        arrays = {
            "operator": np.asarray(payload["operator"], dtype=float),
            "singular_values": np.asarray(
                payload.get("singular_values", []), dtype=float
            ),
            "lambda_value": np.asarray(
                float(payload.get("lambda_value", payload.get("alpha", 0.0)))
            ),
            "retained_rank": np.asarray(
                int(payload.get("retained_rank", 0))
            ),
            "method": np.asarray(str(payload.get("method", "svd_ridge"))),
        }
        if "intercept" in payload:
            arrays["intercept"] = np.asarray(
                payload["intercept"], dtype=float
            )
        np.savez_compressed(destination, **arrays)
        return destination
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def load_operator_package(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.suffix == ".npz":
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Loading NPZ transition operators requires numpy") from exc
        payload = np.load(source)
        result = {
            "operator": payload["operator"].tolist(),
            "singular_values": payload["singular_values"].tolist(),
            "lambda_value": float(payload["lambda_value"]),
            "retained_rank": int(payload["retained_rank"]),
            "method": (
                str(payload["method"])
                if "method" in payload.files
                else "svd_ridge"
            ),
        }
        if "intercept" in payload.files:
            result["intercept"] = payload["intercept"].tolist()
        return result
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)
