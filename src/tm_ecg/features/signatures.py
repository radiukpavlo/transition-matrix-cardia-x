"""Training-only calibration and application of interpretable ECG signatures."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from tm_ecg.constants import B_COLUMNS, SIGNATURE_FEATURES


SIGNATURE_TARGETS = {
    "rbbb_signature_score": "RBBB spectrum",
    "lbbb_signature_score": "LBBB spectrum",
    "pvc_signature_score": "PVC",
    "af_signature_score": "AF",
    "paced_signature_score": "Paced",
}


def robust_semantic_aggregates(
    rows: Sequence[Mapping[str, object]],
    *,
    feature_names: Sequence[str],
) -> dict[str, float | None]:
    """Aggregate beat/lead specialist evidence with explicit missingness."""

    import numpy as np  # type: ignore

    output: dict[str, float | None] = {}
    forbidden = (
        "label",
        "target",
        "scp",
        "fold",
        "benchmark",
        "outcome",
    )
    for feature in feature_names:
        if any(token in feature.lower() for token in forbidden):
            raise ValueError(
                f"Target- or partition-derived semantic aggregate is forbidden: "
                f"{feature}"
            )
        values: list[float] = []
        for row in rows:
            try:
                value = float(row[feature])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        array = np.asarray(values, dtype=float)
        output[f"{feature}_available_fraction"] = (
            len(values) / len(rows) if rows else 0.0
        )
        output[f"{feature}_count"] = float(len(values))
        output[f"{feature}_median"] = (
            float(np.median(array)) if len(array) else None
        )
        output[f"{feature}_iqr"] = (
            float(np.quantile(array, 0.75) - np.quantile(array, 0.25))
            if len(array) >= 4
            else None
        )
        output[f"{feature}_maximum"] = (
            float(np.max(array)) if len(array) else None
        )
    return output


def _labels(value: object) -> set[str]:
    text = str(value or "")
    delimiter = "|" if "|" in text else ","
    return {item.strip() for item in text.split(delimiter) if item.strip()}


def _ece(y_true: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            idx
            for idx, value in enumerate(probabilities)
            if lower <= value < upper or (index == bins - 1 and value == upper)
        ]
        if not members:
            continue
        confidence = sum(probabilities[idx] for idx in members) / len(members)
        accuracy = sum(y_true[idx] for idx in members) / len(members)
        error += (len(members) / len(y_true)) * abs(confidence - accuracy)
    return error


def _threshold(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    candidates = sorted({0.50, *[float(value) for value in probabilities]})
    best = (float("-inf"), 0.50)
    for candidate in candidates:
        tp = sum(y == 1 and p >= candidate for y, p in zip(y_true, probabilities, strict=True))
        fn = sum(y == 1 and p < candidate for y, p in zip(y_true, probabilities, strict=True))
        tn = sum(y == 0 and p < candidate for y, p in zip(y_true, probabilities, strict=True))
        fp = sum(y == 0 and p >= candidate for y, p in zip(y_true, probabilities, strict=True))
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        objective = min(sensitivity, specificity)
        key = (objective, -abs(candidate - 0.5))
        if key > (best[0], -abs(best[1] - 0.5)):
            best = (objective, candidate)
    return min(max(best[1], 0.05), 0.95)


def _logit(probability: float) -> float:
    bounded = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _split_hash(rows: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {"record_id": str(row.get("record_id", "")), "labels": str(row.get("labels", ""))}
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def fit_signature_artifact(
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    *,
    random_seed: int = 17,
) -> dict[str, object]:
    try:
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Signature calibration requires the declared scikit-learn dependency") from exc

    candidate_features = [feature for feature in B_COLUMNS if feature not in SIGNATURE_FEATURES]
    input_features = [
        feature
        for feature in candidate_features
        if any(row.get(feature) not in {None, ""} for row in train_rows)
    ]
    if not input_features:
        raise ValueError("No numeric B features are available for signature fitting")

    def matrix(rows: list[dict[str, object]]):
        return np.asarray(
            [
                [float(row[feature]) if row.get(feature) not in {None, ""} else np.nan for feature in input_features]
                for row in rows
            ],
            dtype=float,
        )

    x_train = matrix(train_rows)
    x_validation = matrix(validation_rows) if validation_rows else x_train
    signatures: dict[str, object] = {}
    for signature, target in SIGNATURE_TARGETS.items():
        y_train = np.asarray([int(target in _labels(row.get("labels"))) for row in train_rows])
        eval_rows = validation_rows or train_rows
        y_validation = np.asarray([int(target in _labels(row.get("labels"))) for row in eval_rows])
        if min(int(y_train.sum()), int(len(y_train) - y_train.sum())) < 2:
            signatures[signature] = {
                "status": "unavailable",
                "target_label": target,
                "reason": "training split has fewer than two positive or negative examples",
            }
            continue
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_validation)[:, 1]
        decision_threshold = _threshold(y_validation.tolist(), probability.tolist())
        # Reserve a deterministic validation-defined uncertainty band around
        # the operating point.  A complementary threshold (1 - positive) can
        # reverse the band whenever the selected operating point is below 0.5.
        uncertainty_half_width = 0.05
        negative_threshold = max(0.01, decision_threshold - uncertainty_half_width)
        positive_threshold = min(0.99, decision_threshold + uncertainty_half_width)
        if not negative_threshold < positive_threshold:
            raise RuntimeError("Signature threshold band is not strictly ordered")
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["classifier"]
        auc = (
            float(roc_auc_score(y_validation, probability))
            if len(set(y_validation.tolist())) == 2
            else None
        )
        signatures[signature] = {
            "status": "available",
            "target_label": target,
            "input_features": input_features,
            "imputer_medians": [float(value) for value in imputer.statistics_],
            "normalization_mean": [float(value) for value in scaler.mean_],
            "normalization_scale": [float(value) for value in scaler.scale_],
            "coefficients": [float(value) for value in classifier.coef_[0]],
            "intercept": float(classifier.intercept_[0]),
            "calibration_method": "balanced_platt_logistic",
            "class_prevalence": float(y_train.mean()),
            "training_positive_count": int(y_train.sum()),
            "training_negative_count": int(len(y_train) - y_train.sum()),
            "validation_metrics": {
                "sample_size": len(eval_rows),
                "roc_auc": auc,
                "brier_score": float(brier_score_loss(y_validation, probability)),
                "expected_calibration_error": _ece(y_validation.tolist(), probability.tolist()),
            },
            "threshold_bands": {
                "validation_decision_probability": decision_threshold,
                "uncertainty_half_width": uncertainty_half_width,
                "negative_max_probability": negative_threshold,
                "positive_min_probability": positive_threshold,
                "negative_max_logodds": _logit(negative_threshold),
                "positive_min_logodds": _logit(positive_threshold),
            },
        }
    return {
        "version": 1,
        "training_only": True,
        "training_split_hash": _split_hash(train_rows),
        "input_feature_schema": input_features,
        "random_seed": random_seed,
        "signatures": signatures,
    }


def apply_signature_scores(
    row: Mapping[str, object], artifact: Mapping[str, object]
) -> tuple[dict[str, float | None], dict[str, str]]:
    scores: dict[str, float | None] = {}
    states: dict[str, str] = {}
    for signature in SIGNATURE_FEATURES:
        model = dict(dict(artifact.get("signatures", {})).get(signature, {}))
        if model.get("status") != "available":
            scores[signature] = None
            states[signature] = "unavailable_artifact"
            continue
        features = [str(item) for item in model.get("input_features", [])]
        arrays = [
            model.get("imputer_medians", []),
            model.get("normalization_mean", []),
            model.get("normalization_scale", []),
            model.get("coefficients", []),
        ]
        if not features or any(len(values) != len(features) for values in arrays):
            scores[signature] = None
            states[signature] = "schema_incompatible"
            continue
        values: list[float] = []
        for index, feature in enumerate(features):
            raw = row.get(feature)
            try:
                numeric = float(raw) if raw not in {None, ""} else float(arrays[0][index])
            except (TypeError, ValueError):
                numeric = float(arrays[0][index])
            scale = float(arrays[2][index]) or 1.0
            values.append((numeric - float(arrays[1][index])) / scale)
        score = float(model.get("intercept", 0.0)) + sum(
            value * float(coefficient)
            for value, coefficient in zip(values, arrays[3], strict=True)
        )
        scores[signature] = score
        states[signature] = "observed"
    return scores, states


def load_signature_artifact(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not payload.get("training_only"):
        raise ValueError("Unsupported or non-training signature artifact")
    for feature, raw_model in dict(payload.get("signatures", {})).items():
        model = dict(raw_model)
        if model.get("status") != "available":
            continue
        bands = dict(model.get("threshold_bands", {}))
        required = {
            "negative_max_probability",
            "positive_min_probability",
            "negative_max_logodds",
            "positive_min_logodds",
        }
        if not required.issubset(bands):
            raise ValueError(f"Signature {feature} lacks complete threshold bands")
        negative_probability = float(bands["negative_max_probability"])
        positive_probability = float(bands["positive_min_probability"])
        negative_logodds = float(bands["negative_max_logodds"])
        positive_logodds = float(bands["positive_min_logodds"])
        if not 0.0 < negative_probability < positive_probability < 1.0:
            raise ValueError(f"Signature {feature} has reversed probability bands")
        if not negative_logodds < positive_logodds:
            raise ValueError(f"Signature {feature} has reversed log-odds bands")
    return payload
