"""Leakage-safe PTB-XL compatibility-head training on ECG-only PTB-XL+ features.

The waveform CNN remains the source of transition-matrix latents.  This module
provides a separate, versioned compatibility head for the strict DSS gate.  It
uses only PTB-XL+ 12SL morphology measurements, fits models on the locked train
partition, fits probability calibration and decision thresholds on validation,
and touches the held-out test partition only during final evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tm_ecg.config import ProjectConfig
from tm_ecg.constants import B_COLUMNS, LEADS_12, PROJECT_LABELS
from tm_ecg.io.common import sha256_file, stable_hash, write_json
from tm_ecg.io.readers import find_table, read_table_frame
from tm_ecg.io.tabular import write_records_table
from tm_ecg.modeling.label_contract import DEFAULT_COMPATIBILITY_CONTRACT_V4


@dataclass(frozen=True, slots=True)
class CompatibilityTrainingSpec:
    """Frozen validation-selected estimator specification."""

    feature_source: str = "ptbxl_plus_12sl_plus_adaptive_waveform_b_v3"
    estimator: str = "sklearn_hist_gradient_boosting"
    max_iter: int = 120
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    class_weight: str = "balanced"
    random_state: int = 17
    calibration_method: str = "platt_logistic_validation_only"
    threshold_objective: str = (
        "per_class_f1_initialization_then_joint_exact_match_coordinate_descent"
    )
    normal_target_contract: str = "normal_only_without_cooccurring_abnormality"


def _validation_lock_path(config: ProjectConfig) -> Path:
    return config.paths.reports / "metrics" / "b1_validation_model_lock.json"


def _canonical_12sl_feature_allowlist(description_path: Path) -> set[str]:
    """Expand the PTB-XL+ canonical feature dictionary into concrete columns.

    ``feature_description.csv`` uses ``_X`` as its documented placeholder for
    a lead-specific morphology measurement.  Only canonical IDs from that
    dictionary, expanded over the locked 12-lead set, are eligible inputs.
    """

    import pandas as pd  # type: ignore

    description = pd.read_csv(description_path, low_memory=False)
    if "id" not in description.columns:
        raise ValueError("PTB-XL+ feature description lacks the canonical id column")
    canonical_ids = [
        str(value).strip()
        for value in description["id"].dropna().tolist()
        if str(value).strip()
    ]
    if not canonical_ids:
        raise RuntimeError("PTB-XL+ feature description has no canonical feature IDs")
    duplicates = sorted(
        identifier
        for identifier in set(canonical_ids)
        if canonical_ids.count(identifier) > 1
    )
    if duplicates:
        raise RuntimeError(
            f"PTB-XL+ feature description contains duplicate canonical IDs: {duplicates}"
        )
    unsupported_placeholders = sorted(
        identifier
        for identifier in canonical_ids
        if "_X" in identifier and not identifier.endswith("_X")
    )
    if unsupported_placeholders:
        raise RuntimeError(
            "PTB-XL+ canonical feature IDs contain unsupported lead placeholders: "
            f"{unsupported_placeholders}"
        )

    allowed: set[str] = set()
    for identifier in canonical_ids:
        if identifier.endswith("_X"):
            stem = identifier[:-2]
            allowed.update(f"{stem}_{lead}" for lead in LEADS_12)
        else:
            allowed.add(identifier)
    return allowed


def _validate_12sl_feature_schema(
    feature_columns: Sequence[object],
    description_path: Path,
) -> dict[str, object]:
    """Fail closed unless every 12SL input is a documented ECG measurement."""

    observed = [str(column) for column in feature_columns if str(column) != "ecg_id"]
    if len(observed) != len(set(observed)):
        raise RuntimeError("PTB-XL+ 12SL table contains duplicate feature columns")
    allowed = _canonical_12sl_feature_allowlist(description_path)
    undocumented = sorted(set(observed) - allowed)
    if undocumented:
        raise RuntimeError(
            "PTB-XL+ 12SL columns are outside the canonical feature-description "
            f"allowlist: {undocumented}"
        )
    return {
        "feature_allowlist_source": str(description_path),
        "feature_allowlist_policy": (
            "canonical feature_description.csv id values with _X expanded over LEADS_12"
        ),
        "feature_allowlist_expanded_count": len(allowed),
        "observed_12sl_schema_sha256": stable_hash(observed),
        "feature_allowlist_sha256": stable_hash(sorted(allowed)),
    }


def _require_artifact_hash(
    path: Path,
    expected_sha256: object,
    *,
    artifact_name: str,
) -> None:
    """Require an existing regular file to match its previously sealed hash."""

    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError(f"{artifact_name} has no valid locked SHA-256")
    if not path.is_file():
        raise RuntimeError(f"{artifact_name} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{artifact_name} hash mismatch: observed={actual} expected={expected}"
        )


def normalize_compatibility_labels(
    raw: object,
    *,
    empty_policy: str = "residual",
) -> list[str]:
    """Delegate legacy inputs to the residual-exclusive v4 contract."""

    return list(
        DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
            raw,
            empty_policy=empty_policy,
        )
    )


def select_f1_threshold(
    targets: Any, probabilities: Any
) -> dict[str, float | int | str]:
    """Select a deterministic threshold on validation rows only."""

    import numpy as np  # type: ignore

    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or not len(y):
        raise ValueError("Threshold targets and probabilities must be non-empty and aligned")
    candidates = np.linspace(0.05, 0.95, 181)
    scored: list[tuple[float, float, float, float, int, int, int, int]] = []
    for threshold in candidates:
        prediction = p >= threshold
        tp = int(((prediction == 1) & (y == 1)).sum())
        tn = int(((prediction == 0) & (y == 0)).sum())
        fp = int(((prediction == 1) & (y == 0)).sum())
        fn = int(((prediction == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        balanced_accuracy = 0.5 * (recall + specificity)
        scored.append(
            (
                f1,
                balanced_accuracy,
                -abs(float(threshold) - 0.5),
                float(threshold),
                tp,
                tn,
                fp,
                fn,
            )
        )
    best = max(scored)
    return {
        "status": "ok",
        "threshold": best[3],
        "validation_f1": best[0],
        "validation_balanced_accuracy": best[1],
        "tp": best[4],
        "tn": best[5],
        "fp": best[6],
        "fn": best[7],
        "threshold_partition": "validation_only",
        "selection_objective": "max_f1_then_balanced_accuracy_then_nearest_0.5",
    }


def project_compatibility_predictions(
    probabilities: Any,
    thresholds: Any,
) -> Any:
    """Apply thresholds and deterministic ontology constraints.

    The project ontology defines ``Normal`` as an exclusive label.  Every
    indexed record also has at least one compatibility label, so an empty
    prediction is projected to the highest calibrated probability.  The
    projection uses probabilities only and therefore does not inspect truth.
    """

    import numpy as np  # type: ignore

    probability_matrix = np.asarray(probabilities, dtype=float)
    threshold_vector = np.asarray(thresholds, dtype=float)
    if probability_matrix.ndim != 2:
        raise ValueError("Compatibility probabilities must be a two-dimensional matrix")
    if probability_matrix.shape[1] != len(PROJECT_LABELS):
        raise ValueError(
            "Compatibility probability columns must follow the complete project ontology"
        )
    if threshold_vector.shape != (len(PROJECT_LABELS),):
        raise ValueError("Compatibility thresholds must have one value per project label")
    if not np.isfinite(probability_matrix).all() or not np.isfinite(threshold_vector).all():
        raise ValueError("Compatibility probabilities and thresholds must be finite")

    prediction = probability_matrix >= threshold_vector
    contract = DEFAULT_COMPATIBILITY_CONTRACT_V4
    normal_index = PROJECT_LABELS.index(contract.normal_label)
    residual_index = PROJECT_LABELS.index(contract.residual_label)
    specific_indices = np.asarray(
        [PROJECT_LABELS.index(label) for label in contract.specific_labels]
    )
    abnormal_indices = np.asarray(
        [index for index in range(len(PROJECT_LABELS)) if index != normal_index]
    )
    conflicts = prediction[:, normal_index] & prediction[:, abnormal_indices].any(axis=1)
    for row_index in np.flatnonzero(conflicts):
        predicted_abnormal = abnormal_indices[prediction[row_index, abnormal_indices]]
        strongest_abnormal = predicted_abnormal[
            int(np.argmax(probability_matrix[row_index, predicted_abnormal]))
        ]
        if (
            probability_matrix[row_index, normal_index]
            >= probability_matrix[row_index, strongest_abnormal]
        ):
            prediction[row_index, :] = False
            prediction[row_index, normal_index] = True
        else:
            prediction[row_index, normal_index] = False

    # The residual class is a fallback state, not an independent abnormality.
    residual_conflicts = (
        prediction[:, residual_index] & prediction[:, specific_indices].any(axis=1)
    )
    prediction[residual_conflicts, residual_index] = False

    empty_rows = ~prediction.any(axis=1)
    if empty_rows.any():
        row_indices = np.flatnonzero(empty_rows)
        strongest = np.argmax(probability_matrix[row_indices], axis=1)
        prediction[row_indices, strongest] = True
    contract.validate_prediction_matrix(prediction)
    return prediction


def _prediction_coverage(predictions: Any) -> dict[str, float | str]:
    """Describe operational coverage from rows receiving a model output.

    This is deliberately not presented as an independently validated signal
    quality measure.  The compatibility head currently enforces a non-empty
    label set for every row and therefore does not abstain.
    """

    import numpy as np  # type: ignore

    prediction_matrix = np.asarray(predictions)
    if prediction_matrix.ndim != 2 or len(prediction_matrix) == 0:
        raise ValueError("Coverage predictions must be a non-empty two-dimensional matrix")
    non_abstained = prediction_matrix.astype(bool).any(axis=1)
    analyzable_coverage = float(non_abstained.mean())
    return {
        "analyzable_coverage": analyzable_coverage,
        "abstention_rate": float(1.0 - analyzable_coverage),
        "analyzable_coverage_status": (
            "operational_prediction_coverage_not_independent_signal_quality"
        ),
        "analyzable_coverage_definition": (
            "fraction of evaluation rows with at least one emitted compatibility label"
        ),
    }


def _require_reproduced_validation_evidence(
    evidence: Any,
    record_ids: Sequence[str],
    truth: Any,
    probabilities: Any,
    predictions: Any,
    *,
    split_manifest_sha256: str,
) -> None:
    """Verify the refit exactly reproduces sealed validation-only evidence.

    This check runs before held-out scoring.  It prevents a changed estimator,
    calibration implementation, or environment from passing merely because it
    happens to select the same thresholds and thresholded validation scores.
    """

    import numpy as np  # type: ignore

    required_metadata = {
        "record_id",
        "evaluation_partition",
        "threshold_partition",
        "split_manifest_sha256",
    }
    missing_metadata = sorted(required_metadata - set(evidence.columns))
    if missing_metadata:
        raise RuntimeError(
            f"Validation prediction evidence lacks metadata columns: {missing_metadata}"
        )
    frame = evidence.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    expected_ids = [str(record_id) for record_id in record_ids]
    if (
        frame["record_id"].duplicated().any()
        or len(frame) != len(expected_ids)
        or set(frame["record_id"]) != set(expected_ids)
    ):
        raise RuntimeError(
            "Validation prediction evidence does not exactly match the locked validation split"
        )
    frame = frame.set_index("record_id").loc[expected_ids]
    if not frame["evaluation_partition"].astype(str).eq(
        "validation_model_selection"
    ).all():
        raise RuntimeError("Validation evidence has an invalid evaluation partition")
    if not frame["threshold_partition"].astype(str).eq("validation_only").all():
        raise RuntimeError("Validation evidence has an invalid threshold partition")
    if not frame["split_manifest_sha256"].astype(str).eq(
        split_manifest_sha256
    ).all():
        raise RuntimeError("Validation evidence is bound to a different split manifest")

    truth_matrix = np.asarray(truth, dtype=int)
    probability_matrix = np.asarray(probabilities, dtype=float)
    prediction_matrix = np.asarray(predictions, dtype=int)
    expected_shape = (len(expected_ids), len(PROJECT_LABELS))
    if (
        truth_matrix.shape != expected_shape
        or probability_matrix.shape != expected_shape
        or prediction_matrix.shape != expected_shape
    ):
        raise RuntimeError("Reproduced validation matrices do not match the locked ontology")

    for column, label in enumerate(PROJECT_LABELS):
        key = label.lower().replace(" / ", "_").replace(" ", "_")
        required_columns = {
            f"true::{key}",
            f"probability::{key}",
            f"predicted::{key}",
        }
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise RuntimeError(
                f"Validation prediction evidence lacks columns for {label}: {missing}"
            )
        locked_truth = frame[f"true::{key}"].to_numpy(dtype=int)
        locked_probabilities = frame[f"probability::{key}"].to_numpy(dtype=float)
        locked_predictions = frame[f"predicted::{key}"].to_numpy(dtype=int)
        if not np.array_equal(locked_truth, truth_matrix[:, column]):
            raise RuntimeError(f"Validation truth evidence is not reproducible for {label}")
        if not np.isfinite(locked_probabilities).all() or not np.allclose(
            locked_probabilities,
            probability_matrix[:, column],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Validation probability evidence is not reproducible for {label}"
            )
        if not np.array_equal(locked_predictions, prediction_matrix[:, column]):
            raise RuntimeError(
                f"Validation prediction evidence is not reproducible for {label}"
            )


def _multilabel_objective(truth: Any, prediction: Any) -> dict[str, float]:
    """Return exact-match and secondary multilabel validation objectives."""

    import numpy as np  # type: ignore

    truth_matrix = np.asarray(truth, dtype=int)
    prediction_matrix = np.asarray(prediction, dtype=int)
    if truth_matrix.shape != prediction_matrix.shape or truth_matrix.ndim != 2:
        raise ValueError("Multilabel truth and predictions must be aligned matrices")
    exact_match = float(np.mean(np.all(prediction_matrix == truth_matrix, axis=1)))
    bitwise_accuracy = float(np.mean(prediction_matrix == truth_matrix))
    tp = int(((prediction_matrix == 1) & (truth_matrix == 1)).sum())
    fp = int(((prediction_matrix == 1) & (truth_matrix == 0)).sum())
    fn = int(((prediction_matrix == 0) & (truth_matrix == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "compatibility_subset_exact_match": exact_match,
        "micro_f1": float(micro_f1),
        "per_label_bitwise_accuracy": bitwise_accuracy,
    }


def _multilabel_cluster_bootstrap(
    truth: Any,
    prediction: Any,
    patient_groups: Sequence[str],
    *,
    seed: int,
    replicates: int = 1000,
) -> dict[str, object]:
    """Estimate global multilabel intervals by resampling patient clusters."""

    import numpy as np  # type: ignore

    truth_matrix = np.asarray(truth, dtype=int)
    prediction_matrix = np.asarray(prediction, dtype=int)
    groups = np.asarray(list(patient_groups), dtype=str)
    if truth_matrix.shape != prediction_matrix.shape or truth_matrix.ndim != 2:
        raise ValueError("Bootstrap truth and predictions must be aligned matrices")
    if len(groups) != len(truth_matrix):
        raise ValueError("Bootstrap patient groups must align with metric rows")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "compatibility_subset_exact_match": [],
        "micro_f1": [],
        "per_label_bitwise_accuracy": [],
    }
    for _ in range(replicates):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        objective = _multilabel_objective(
            truth_matrix[indices],
            prediction_matrix[indices],
        )
        for metric_name in samples:
            samples[metric_name].append(objective[metric_name])
    return {
        "confidence_level": 0.95,
        "confidence_interval_method": (
            f"patient_cluster_bootstrap_percentile_{replicates}"
        ),
        "bootstrap_cluster_count": int(len(unique_groups)),
        "bootstrap_replicates": replicates,
        "intervals": {
            metric_name: [float(value) for value in np.quantile(values, [0.025, 0.975])]
            for metric_name, values in samples.items()
        },
    }


def select_joint_thresholds(
    truth: Any,
    probabilities: Any,
    initial_thresholds: Any,
    *,
    maximum_passes: int = 6,
) -> tuple[Any, dict[str, object]]:
    """Select ontology-aware thresholds on validation rows only.

    Coordinate descent starts from independently selected F1 thresholds.  Its
    lexicographic objective is exact label-set agreement, micro-F1, then
    bitwise accuracy.  The deterministic distance tie-break prevents arbitrary
    drift when several thresholds induce identical predictions.
    """

    import numpy as np  # type: ignore

    truth_matrix = np.asarray(truth, dtype=int)
    probability_matrix = np.asarray(probabilities, dtype=float)
    initial = np.asarray(initial_thresholds, dtype=float)
    expected_shape = (len(probability_matrix), len(PROJECT_LABELS))
    if truth_matrix.shape != expected_shape or probability_matrix.shape != expected_shape:
        raise ValueError(
            "Joint threshold truth and probabilities must be aligned to the project ontology"
        )
    if initial.shape != (len(PROJECT_LABELS),):
        raise ValueError("Initial thresholds must have one value per project label")
    if maximum_passes < 1:
        raise ValueError("maximum_passes must be positive")

    candidates = np.linspace(0.05, 0.95, 181)
    thresholds = initial.copy()
    initial_prediction = project_compatibility_predictions(
        probability_matrix, thresholds
    )
    initial_objective = _multilabel_objective(truth_matrix, initial_prediction)
    passes_completed = 0
    for pass_index in range(maximum_passes):
        changed = False
        for column in range(len(PROJECT_LABELS)):
            current = float(thresholds[column])
            best_threshold = current
            best_score: tuple[float, float, float, float, float] | None = None
            for candidate in np.unique(np.append(candidates, current)):
                trial = thresholds.copy()
                trial[column] = float(candidate)
                objective = _multilabel_objective(
                    truth_matrix,
                    project_compatibility_predictions(probability_matrix, trial),
                )
                score = (
                    objective["compatibility_subset_exact_match"],
                    objective["micro_f1"],
                    objective["per_label_bitwise_accuracy"],
                    -abs(float(candidate) - float(initial[column])),
                    -abs(float(candidate) - 0.5),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_threshold = float(candidate)
            if abs(best_threshold - current) > 1e-12:
                thresholds[column] = best_threshold
                changed = True
        passes_completed = pass_index + 1
        if not changed:
            break

    final_prediction = project_compatibility_predictions(
        probability_matrix, thresholds
    )
    final_objective = _multilabel_objective(truth_matrix, final_prediction)
    if (
        final_objective["compatibility_subset_exact_match"]
        < initial_objective["compatibility_subset_exact_match"]
    ):
        raise RuntimeError("Joint threshold selection degraded its primary objective")
    return thresholds, {
        "status": "ok",
        "partition": "validation_only",
        "selection_method": "deterministic_coordinate_descent",
        "projection_contract": "normal_exclusive_and_nonempty_argmax",
        "candidate_grid": {"minimum": 0.05, "maximum": 0.95, "step": 0.005},
        "maximum_passes": maximum_passes,
        "passes_completed": passes_completed,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
    }


def expected_calibration_error(
    targets: Any, probabilities: Any, bins: int = 15
) -> float:
    """Return equal-width expected calibration error."""

    import numpy as np  # type: ignore

    y = np.asarray(targets, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or not len(y):
        raise ValueError("Calibration targets and probabilities must be non-empty and aligned")
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return error


def _current_12sl_paths(config: ProjectConfig) -> tuple[Path, Path]:
    root = config.paths.raw / config.datasets["ptbxl_plus"].extract_dir
    candidates = sorted(
        path
        for path in root.rglob("12sl_features.csv")
        if path.parent.name == "features" and path.parent.parent.name != "old"
    )
    descriptions = sorted(
        path
        for path in root.rglob("feature_description.csv")
        if path.parent.name == "features" and path.parent.parent.name != "old"
    )
    if len(candidates) != 1 or len(descriptions) != 1:
        raise FileNotFoundError(
            "Expected exactly one current PTB-XL+ features/12sl_features.csv and "
            f"feature_description.csv; found {candidates=} {descriptions=}"
        )
    return candidates[0], descriptions[0]


def _manifest_path(config: ProjectConfig) -> Path:
    parquet = config.paths.manifests / "ptbxl_split_index.parquet"
    csv_path = config.paths.manifests / "ptbxl_split_index.csv"
    if parquet.exists():
        return parquet
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError("Missing PTB-XL split index; run the index/splits stages first")


def _patient_key(patient_id: object, record_id: object) -> str:
    normalized = str(patient_id).strip().lower()
    return f"record:{record_id}" if normalized in {"", "none", "nan"} else normalized


def _build_compatibility_target_frame(
    frame: Any,
    target_splits: Sequence[str],
) -> tuple[Any, dict[str, dict[str, int]]]:
    """Construct targets only for explicitly authorized partitions."""

    import pandas as pd  # type: ignore

    target_frame = pd.DataFrame(
        pd.NA,
        index=frame.index,
        columns=[f"target::{label}" for label in PROJECT_LABELS],
        dtype="Int8",
    )
    target_support_by_split: dict[str, dict[str, int]] = {}
    for split in target_splits:
        split_mask = frame["split"].astype(str).eq(split)
        split_labels = [
            normalize_compatibility_labels(value, empty_policy="error")
            for value in frame.loc[split_mask, "labels"]
        ]
        support: dict[str, int] = {}
        for label in PROJECT_LABELS:
            values = [int(label in labels) for labels in split_labels]
            target_frame.loc[split_mask, f"target::{label}"] = values
            support[label] = int(sum(values))
        target_support_by_split[split] = support
    return target_frame, target_support_by_split


def _load_waveform_b_features(
    config: ProjectConfig,
    manifest: Any,
) -> tuple[Any, list[str], dict[str, object]]:
    """Load ECG-only adaptive waveform features with exact split-ID binding."""

    import pandas as pd  # type: ignore

    frames: list[Any] = []
    artifact_hashes: dict[str, str] = {}
    feature_columns: list[str] | None = None
    for split in ("train", "val", "test"):
        path = find_table(config.paths.features, f"B1_raw_{split}", required=True)
        assert path is not None
        frame = read_table_frame(path).copy()
        if "record_id" not in frame.columns:
            raise ValueError(f"Waveform B artifact lacks record_id: {path}")
        frame["record_id"] = frame["record_id"].astype(str)
        if frame["record_id"].duplicated().any():
            raise RuntimeError(f"Waveform B artifact has duplicate record IDs: {path}")
        expected_ids = set(
            manifest.loc[
                manifest["split"].astype(str).eq(split),
                "record_id",
            ].astype(str)
        )
        observed_ids = set(frame["record_id"])
        if observed_ids != expected_ids:
            raise RuntimeError(
                f"Waveform B {split} IDs do not match the locked split: "
                f"missing={len(expected_ids - observed_ids)} "
                f"unexpected={len(observed_ids - expected_ids)}"
            )
        numeric = [
            column
            for column in B_COLUMNS
            if column in frame.columns
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if feature_columns is None:
            feature_columns = numeric
        elif numeric != feature_columns:
            raise RuntimeError(
                "Waveform B feature columns are inconsistent across partitions"
            )
        renamed = {column: f"waveform_b::{column}" for column in numeric}
        frames.append(frame[["record_id", *numeric]].rename(columns=renamed))
        artifact_hashes[split] = sha256_file(path)
    columns = [f"waveform_b::{column}" for column in (feature_columns or [])]
    if not columns:
        raise RuntimeError("No numeric adaptive waveform B features are available")
    return (
        pd.concat(frames, ignore_index=True),
        columns,
        {
            "waveform_b_partition_sha256": artifact_hashes,
            "waveform_b_artifact_hash": stable_hash(artifact_hashes),
            "waveform_b_feature_count": len(columns),
            "waveform_b_feature_policy": "numeric columns from locked B_COLUMNS only",
        },
    )


def _split_audit(manifest: Any) -> dict[str, object]:
    patients: dict[str, set[str]] = {}
    record_ids: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        rows = manifest[manifest["split"].astype(str) == split]
        record_ids[split] = [str(value) for value in rows["record_id"].tolist()]
        patients[split] = {
            _patient_key(row.patient_id, row.record_id)
            for row in rows.itertuples(index=False)
        }
    overlaps = {
        "train_val": sorted(patients["train"] & patients["val"]),
        "train_test": sorted(patients["train"] & patients["test"]),
        "val_test": sorted(patients["val"] & patients["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Patient leakage detected: {overlaps}")
    if any(len(ids) != len(set(ids)) for ids in record_ids.values()):
        raise RuntimeError("Duplicate record IDs detected inside a PTB-XL split")
    return {
        "patient_disjoint": True,
        "patient_overlap_count": 0,
        "record_counts": {split: len(ids) for split, ids in record_ids.items()},
        "record_id_hashes": {
            split: stable_hash(sorted(ids)) for split, ids in record_ids.items()
        },
    }


def _load_design(
    config: ProjectConfig,
    *,
    target_splits: Sequence[str] = ("train", "val", "test"),
) -> tuple[Any, list[str], dict[str, object]]:
    import pandas as pd  # type: ignore

    normalized_target_splits = tuple(dict.fromkeys(str(split) for split in target_splits))
    allowed_splits = {"train", "val", "test"}
    invalid_target_splits = sorted(set(normalized_target_splits) - allowed_splits)
    if not normalized_target_splits or invalid_target_splits:
        raise ValueError(
            "Compatibility target_splits must be a non-empty subset of train/val/test; "
            f"invalid={invalid_target_splits}"
        )
    manifest_path = _manifest_path(config)
    manifest = read_table_frame(manifest_path).copy()
    required = {"record_id", "patient_id", "split", "labels"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Split index lacks columns {sorted(required - set(manifest.columns))}")
    manifest["record_id"] = manifest["record_id"].astype(str)
    if manifest["record_id"].duplicated().any():
        raise RuntimeError("The locked PTB-XL split index contains duplicate record IDs")
    if "ontology_version" not in manifest.columns:
        raise RuntimeError("The locked PTB-XL split index lacks ontology_version")
    ontology_versions = {
        str(value) for value in manifest["ontology_version"].dropna().unique().tolist()
    }
    if ontology_versions != {config.ontology_version}:
        raise RuntimeError(
            "The locked PTB-XL split ontology does not match the active configuration: "
            f"manifest={sorted(ontology_versions)} active={config.ontology_version}"
        )
    audit = _split_audit(manifest)
    # A validation-only development run must not parse or expose held-out test
    # labels.  Mask labels outside the explicitly requested target partitions
    # before any normalization or target construction occurs.
    target_partition_mask = manifest["split"].astype(str).isin(
        normalized_target_splits
    )
    manifest.loc[~target_partition_mask, "labels"] = ""
    if "raw_source_labels" in manifest.columns:
        manifest.loc[~target_partition_mask, "raw_source_labels"] = ""

    feature_path, description_path = _current_12sl_paths(config)
    features = pd.read_csv(feature_path, low_memory=False)
    if "ecg_id" not in features.columns:
        raise ValueError("PTB-XL+ 12SL table lacks ecg_id")
    features["ecg_id"] = features["ecg_id"].astype(str)
    if features["ecg_id"].duplicated().any():
        raise RuntimeError("PTB-XL+ 12SL table contains duplicate ecg_id values")
    feature_allowlist_audit = _validate_12sl_feature_schema(
        features.columns,
        description_path,
    )
    # Consolidate mixed CSV blocks before the wide merge; this avoids quadratic
    # fragmentation overhead without changing values or column order.
    features = features.copy()

    joined = manifest.merge(
        features,
        how="left",
        left_on="record_id",
        right_on="ecg_id",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"] != "both", "record_id"].tolist()
        raise RuntimeError(f"PTB-XL+ features missing for locked records: {missing[:10]}")

    waveform_b, waveform_columns, waveform_provenance = _load_waveform_b_features(
        config,
        manifest,
    )
    joined = joined.merge(
        waveform_b,
        how="left",
        on="record_id",
        validate="one_to_one",
    )
    if joined[waveform_columns].isna().all(axis=1).any():
        raise RuntimeError("Adaptive waveform B features are missing for locked records")

    excluded = set(manifest.columns) | {"ecg_id", "_merge"}
    feature_columns_12sl = [
        column
        for column in features.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(features[column])
        and joined.loc[joined["split"].eq("train"), column].notna().any()
    ]
    if not feature_columns_12sl:
        raise RuntimeError("No train-observed numeric 12SL features remain")
    feature_columns = [*feature_columns_12sl, *waveform_columns]
    target_frame, target_support_by_split = _build_compatibility_target_frame(
        joined,
        normalized_target_splits,
    )
    joined = pd.concat([joined.copy(), target_frame], axis=1)
    provenance = {
        **audit,
        "ontology_version": config.ontology_version,
        "split_manifest_path": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "feature_description_path": str(description_path),
        "feature_description_sha256": sha256_file(description_path),
        "feature_count": len(feature_columns),
        "feature_count_12sl": len(feature_columns_12sl),
        "excluded_identifiers": ["ecg_id", "record_id", "patient_id"],
        "excluded_source_labels": True,
        "excluded_diagnostic_metadata": True,
        "feature_description_role": "executable_canonical_feature_allowlist",
        "feature_selection_method": (
            "numeric train-observed 12SL columns constrained to canonical feature IDs"
        ),
        "target_splits_loaded": list(normalized_target_splits),
        "target_support_by_split": target_support_by_split,
        **feature_allowlist_audit,
        **waveform_provenance,
    }
    return joined, feature_columns, provenance


def _fit_platt(raw_probabilities: Sequence[float], targets: Sequence[int]) -> Any | None:
    import numpy as np  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore

    y = np.asarray(targets, dtype=int)
    if min(int(y.sum()), int((1 - y).sum())) < 20:
        return None
    p = np.clip(np.asarray(raw_probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=17).fit(
        logits, y
    )


def _apply_platt(calibrator: Any | None, raw_probabilities: Sequence[float]) -> Any:
    import numpy as np  # type: ignore

    p = np.clip(np.asarray(raw_probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    if calibrator is None:
        return p
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _class_metrics(
    targets: Any,
    probabilities: Any,
    threshold: float,
    *,
    seed: int,
    patient_groups: Sequence[str] | None = None,
    replicates: int = 1000,
    predictions: Any | None = None,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prediction = (
        np.asarray(predictions, dtype=bool)
        if predictions is not None
        else p >= threshold
    )
    if prediction.shape != y.shape:
        raise ValueError("Supplied class predictions and targets must be aligned")
    tp = int(((prediction == 1) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    groups = np.asarray(
        list(patient_groups) if patient_groups is not None else [f"record:{i}" for i in range(len(y))]
    )
    if len(groups) != len(y):
        raise ValueError("Patient groups and metric rows must be aligned")
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    bootstrap_f1: list[float] = []
    for _ in range(replicates):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        yt = y[indices]
        yp = prediction[indices]
        btp = int(((yp == 1) & (yt == 1)).sum())
        bfp = int(((yp == 1) & (yt == 0)).sum())
        bfn = int(((yp == 0) & (yt == 1)).sum())
        bp = btp / (btp + bfp) if btp + bfp else 0.0
        br = btp / (btp + bfn) if btp + bfn else 0.0
        bootstrap_f1.append(2.0 * bp * br / (bp + br) if bp + br else 0.0)
    lower, upper = np.quantile(bootstrap_f1, [0.025, 0.975])
    stability = sum(abs(value - f1) <= 0.05 for value in bootstrap_f1) / replicates
    positive_confidence = p[prediction]
    return {
        "status": "ok" if len(set(y.tolist())) == 2 else "not_estimable_single_class",
        "support": int(y.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "binary_accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_confidence": (
            float(positive_confidence.mean()) if len(positive_confidence) else None
        ),
        "metric_lower_bound": float(lower),
        "f1_ci_95": [float(lower), float(upper)],
        "confidence_level": 0.95,
        "confidence_interval_method": f"patient_cluster_bootstrap_percentile_{replicates}",
        "bootstrap_cluster_count": int(len(unique_groups)),
        "bootstrap_stability": float(stability),
        "bootstrap_stability_definition": (
            "fraction of fixed-test patient-cluster bootstrap F1 within 0.05 of point F1"
        ),
        "calibration_error": float(expected_calibration_error(y, p)),
    }


def build_validation_model_lock(
    config: ProjectConfig,
    *,
    spec: CompatibilityTrainingSpec | None = None,
) -> dict[str, str]:
    """Fit and audit the frozen estimator using train/validation rows only."""

    import numpy as np  # type: ignore
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

    chosen = spec or CompatibilityTrainingSpec(random_state=config.seed)
    frame, feature_columns, provenance = _load_design(
        config,
        target_splits=("train", "val"),
    )
    train_mask = frame["split"].astype(str).eq("train").to_numpy()
    val_mask = frame["split"].astype(str).eq("val").to_numpy()
    train_matrix = frame.loc[train_mask, feature_columns].to_numpy(np.float32)
    val_matrix = frame.loc[val_mask, feature_columns].to_numpy(np.float32)
    val_patient_groups = [
        _patient_key(row.patient_id, row.record_id)
        for row in frame.loc[val_mask].itertuples(index=False)
    ]
    initial_threshold_rows: dict[str, dict[str, object]] = {}
    calibration_status: dict[str, bool] = {}
    truth_columns: list[Any] = []
    probability_columns: list[Any] = []
    for column, label in enumerate(PROJECT_LABELS):
        print(
            f"Validation lock class {column + 1}/{len(PROJECT_LABELS)}: {label}",
            flush=True,
        )
        target_column = f"target::{label}"
        train_y = frame.loc[train_mask, target_column].to_numpy(dtype=int)
        val_y = frame.loc[val_mask, target_column].to_numpy(dtype=int)
        if len(set(train_y.tolist())) < 2:
            val_probability = np.full(
                len(val_y), float(train_y[0]) if len(train_y) else 0.0
            )
            calibrator = None
        else:
            model = HistGradientBoostingClassifier(
                max_iter=chosen.max_iter,
                learning_rate=chosen.learning_rate,
                max_leaf_nodes=chosen.max_leaf_nodes,
                l2_regularization=chosen.l2_regularization,
                class_weight=chosen.class_weight,
                random_state=chosen.random_state + column,
            ).fit(train_matrix, train_y)
            raw_val = model.predict_proba(val_matrix)[:, 1]
            calibrator = _fit_platt(raw_val, val_y)
            val_probability = _apply_platt(calibrator, raw_val)
        threshold_row = (
            select_f1_threshold(val_y, val_probability)
            if len(set(val_y.tolist())) == 2
            else {
                "status": "not_estimable_single_class",
                "threshold": 0.5,
                "threshold_partition": "validation_only",
            }
        )
        initial_threshold_rows[label] = dict(threshold_row)
        calibration_status[label] = calibrator is not None
        truth_columns.append(val_y)
        probability_columns.append(val_probability)

    truth_matrix = np.column_stack(truth_columns)
    probability_matrix = np.column_stack(probability_columns)
    initial_vector = np.asarray(
        [float(initial_threshold_rows[label]["threshold"]) for label in PROJECT_LABELS]
    )
    initial_independent_prediction = probability_matrix >= initial_vector
    joint_vector, joint_selection = select_joint_thresholds(
        truth_matrix,
        probability_matrix,
        initial_vector,
    )
    prediction_matrix = project_compatibility_predictions(
        probability_matrix,
        joint_vector,
    )
    global_metrics = _multilabel_objective(truth_matrix, prediction_matrix)
    coverage_metrics = _prediction_coverage(prediction_matrix)

    thresholds: dict[str, dict[str, object]] = {}
    per_class: dict[str, dict[str, object]] = {}
    for column, label in enumerate(PROJECT_LABELS):
        threshold_row = dict(initial_threshold_rows[label])
        f1_threshold = float(threshold_row["threshold"])
        joint_threshold = float(joint_vector[column])
        threshold_row.update(
            {
                "per_class_f1_threshold": f1_threshold,
                "threshold": joint_threshold,
                "joint_threshold_changed": abs(joint_threshold - f1_threshold) > 1e-12,
                "threshold_selection_stage": "validation_joint_coordinate_descent",
                "projection_contract": "normal_exclusive_and_nonempty_argmax",
            }
        )
        thresholds[label] = threshold_row
        metrics = _class_metrics(
            truth_matrix[:, column],
            probability_matrix[:, column],
            joint_threshold,
            seed=chosen.random_state + column,
            patient_groups=val_patient_groups,
            predictions=prediction_matrix[:, column],
        )
        metrics.update(
            {
                "threshold": joint_threshold,
                "per_class_f1_threshold": f1_threshold,
                "probabilities_calibrated": calibration_status[label],
                "evaluation_partition": "validation_model_selection",
                "threshold_partition": "validation_only",
                "prediction_projection_applied": True,
                **coverage_metrics,
            }
        )
        per_class[label] = metrics

    validation_record_ids = frame.loc[val_mask, "record_id"].astype(str).tolist()
    prediction_rows: list[Mapping[str, object]] = []
    for row_index, record_id in enumerate(validation_record_ids):
        row: dict[str, object] = {
            "record_id": record_id,
            "evaluation_partition": "validation_model_selection",
            "threshold_partition": "validation_only",
            "split_manifest_sha256": provenance["split_manifest_sha256"],
        }
        for column, label in enumerate(PROJECT_LABELS):
            key = label.lower().replace(" / ", "_").replace(" ", "_")
            row[f"true::{key}"] = int(truth_matrix[row_index, column])
            row[f"probability::{key}"] = float(probability_matrix[row_index, column])
            row[f"predicted::{key}"] = int(prediction_matrix[row_index, column])
        prediction_rows.append(row)
    validation_predictions_path = (
        config.paths.reports / "metrics" / "ptbxl_compatibility_validation_predictions.parquet"
    )
    actual_validation_predictions = write_records_table(
        validation_predictions_path,
        prediction_rows,
    )

    lock_path = _validation_lock_path(config)
    payload: dict[str, object] = {
        "artifact_version": 2,
        "status": "frozen_validation_only",
        "ontology_version": config.ontology_version,
        "training_spec": asdict(chosen),
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "feature_sha256": provenance["feature_sha256"],
        "feature_description_sha256": provenance["feature_description_sha256"],
        "waveform_b_artifact_hash": provenance["waveform_b_artifact_hash"],
        "train_record_id_hash": provenance["record_id_hashes"]["train"],
        "validation_record_id_hash": provenance["record_id_hashes"]["val"],
        "train_records": int(train_mask.sum()),
        "validation_records": int(val_mask.sum()),
        "validation_patient_clusters": len(set(val_patient_groups)),
        "validation_thresholds": thresholds,
        "validation_joint_threshold_selection": joint_selection,
        "validation_per_class_metrics": per_class,
        "validation_initial_independent_objective": _multilabel_objective(
            truth_matrix,
            initial_independent_prediction,
        ),
        "validation_compatibility_subset_exact_match": global_metrics[
            "compatibility_subset_exact_match"
        ],
        "validation_per_label_bitwise_accuracy": global_metrics[
            "per_label_bitwise_accuracy"
        ],
        "validation_micro_f1": global_metrics["micro_f1"],
        "validation_analyzable_coverage": coverage_metrics["analyzable_coverage"],
        "validation_abstention_rate": coverage_metrics["abstention_rate"],
        "validation_analyzable_coverage_definition": coverage_metrics[
            "analyzable_coverage_definition"
        ],
        "validation_global_metric_intervals": _multilabel_cluster_bootstrap(
            truth_matrix,
            prediction_matrix,
            val_patient_groups,
            seed=chosen.random_state,
        ),
        "best_validation_compatibility_f1": max(
            float(metric["f1"]) for metric in per_class.values()
        ),
        "validation_predictions_path": str(actual_validation_predictions),
        "validation_predictions_sha256": sha256_file(actual_validation_predictions),
        "test_partition_evaluated": False,
        "selection_note": (
            "Estimator hyperparameters, calibration, ontology projection, and joint "
            "thresholds were frozen using train/validation rows before held-out test "
            "evaluation. This artifact contains no held-out test result."
        ),
    }
    write_json(lock_path, payload)
    return {"validation_lock": str(lock_path)}


def train_and_evaluate_ptbxl_plus_compatibility(
    config: ProjectConfig,
    *,
    spec: CompatibilityTrainingSpec | None = None,
) -> dict[str, str]:
    """Train, calibrate, and evaluate the frozen ECG-only compatibility head."""

    import joblib  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

    chosen = spec or CompatibilityTrainingSpec(random_state=config.seed)
    frame, feature_columns, provenance = _load_design(config)
    validation_lock_path = _validation_lock_path(config)
    if not validation_lock_path.exists():
        raise FileNotFoundError(
            "Missing validation-only model lock; run compatibility.py --validation-audit first"
        )
    validation_lock = json.loads(validation_lock_path.read_text(encoding="utf-8"))
    lock_requirements = {
        "artifact_version": 2,
        "status": "frozen_validation_only",
        "ontology_version": config.ontology_version,
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "feature_sha256": provenance["feature_sha256"],
        "feature_description_sha256": provenance["feature_description_sha256"],
        "waveform_b_artifact_hash": provenance["waveform_b_artifact_hash"],
        "train_record_id_hash": provenance["record_id_hashes"]["train"],
        "validation_record_id_hash": provenance["record_id_hashes"]["val"],
    }
    for key, expected in lock_requirements.items():
        if validation_lock.get(key) != expected:
            raise RuntimeError(
                f"Validation model lock mismatch for {key}: "
                f"lock={validation_lock.get(key)!r} expected={expected!r}"
            )
    if validation_lock.get("training_spec") != asdict(chosen):
        raise RuntimeError("Validation model lock training specification is stale")
    validation_predictions_path = Path(
        str(validation_lock.get("validation_predictions_path", ""))
    )
    expected_validation_predictions_hash = str(
        validation_lock.get("validation_predictions_sha256", "")
    )
    _require_artifact_hash(
        validation_predictions_path,
        expected_validation_predictions_hash,
        artifact_name="Validation prediction evidence",
    )
    locked_thresholds = dict(validation_lock.get("validation_thresholds", {}))
    split_masks = {
        split: frame["split"].astype(str).eq(split).to_numpy()
        for split in ("train", "val", "test")
    }
    matrices = {
        split: frame.loc[mask, feature_columns].to_numpy(np.float32)
        for split, mask in split_masks.items()
    }
    record_ids = {
        split: frame.loc[mask, "record_id"].astype(str).tolist()
        for split, mask in split_masks.items()
    }
    test_patient_groups = [
        _patient_key(row.patient_id, row.record_id)
        for row in frame.loc[split_masks["test"]].itertuples(index=False)
    ]
    validation_prediction_evidence = read_table_frame(validation_predictions_path).copy()
    validation_prediction_evidence["record_id"] = validation_prediction_evidence[
        "record_id"
    ].astype(str)
    if (
        validation_prediction_evidence["record_id"].duplicated().any()
        or set(validation_prediction_evidence["record_id"]) != set(record_ids["val"])
        or len(validation_prediction_evidence) != len(record_ids["val"])
    ):
        raise RuntimeError(
            "Validation prediction evidence does not exactly match the locked validation split"
        )

    models: dict[str, object | None] = {}
    calibrators: dict[str, object | None] = {}
    thresholds: dict[str, dict[str, object]] = {}
    constant_probabilities: dict[str, float] = {}
    validation_probabilities: dict[str, Any] = {}
    validation_targets: dict[str, Any] = {}
    test_probabilities: dict[str, Any] = {}
    test_targets: dict[str, Any] = {}
    per_class: dict[str, dict[str, object]] = {}

    for column, label in enumerate(PROJECT_LABELS):
        print(
            f"Reproducing validation lock class {column + 1}/{len(PROJECT_LABELS)}: {label}",
            flush=True,
        )
        target_column = f"target::{label}"
        train_y = frame.loc[split_masks["train"], target_column].to_numpy(dtype=int)
        val_y = frame.loc[split_masks["val"], target_column].to_numpy(dtype=int)
        validation_targets[label] = val_y
        if len(set(train_y.tolist())) < 2:
            constant_probability = float(train_y[0]) if len(train_y) else 0.0
            val_probability = np.full(len(val_y), constant_probability)
            model = None
            calibrator = None
            constant_probabilities[label] = constant_probability
        else:
            model = HistGradientBoostingClassifier(
                max_iter=chosen.max_iter,
                learning_rate=chosen.learning_rate,
                max_leaf_nodes=chosen.max_leaf_nodes,
                l2_regularization=chosen.l2_regularization,
                class_weight=chosen.class_weight,
                random_state=chosen.random_state + column,
            ).fit(matrices["train"], train_y)
            raw_val = model.predict_proba(matrices["val"])[:, 1]
            calibrator = _fit_platt(raw_val, val_y)
            val_probability = _apply_platt(calibrator, raw_val)
        models[label] = model
        calibrators[label] = calibrator
        validation_probabilities[label] = val_probability
        if len(set(val_y.tolist())) < 2:
            threshold_row: dict[str, float | int | str] = {
                "status": "not_estimable_single_class",
                "threshold": 0.5,
                "threshold_partition": "validation_only",
            }
        else:
            threshold_row = select_f1_threshold(val_y, val_probability)
        locked_threshold = dict(locked_thresholds.get(label, {}))
        locked_initial_threshold = locked_threshold.get("per_class_f1_threshold")
        if not locked_threshold or abs(
            float(locked_initial_threshold if locked_initial_threshold is not None else -1.0)
            - float(threshold_row["threshold"])
        ) > 1e-12:
            raise RuntimeError(f"Validation F1-threshold lock mismatch for {label}")
        thresholds[label] = locked_threshold
        print(
            f"  locked joint threshold={float(locked_threshold['threshold']):.3f} "
            f"status={locked_threshold['status']}",
            flush=True,
        )

    validation_truth_matrix = np.column_stack(
        [validation_targets[label] for label in PROJECT_LABELS]
    )
    validation_probability_matrix = np.column_stack(
        [validation_probabilities[label] for label in PROJECT_LABELS]
    )
    reproduced_initial_vector = np.asarray(
        [
            float(thresholds[label]["per_class_f1_threshold"])
            for label in PROJECT_LABELS
        ]
    )
    reproduced_joint_vector, reproduced_joint_selection = select_joint_thresholds(
        validation_truth_matrix,
        validation_probability_matrix,
        reproduced_initial_vector,
    )
    locked_joint_vector = np.asarray(
        [float(thresholds[label]["threshold"]) for label in PROJECT_LABELS]
    )
    if not np.allclose(reproduced_joint_vector, locked_joint_vector, atol=1e-12, rtol=0.0):
        raise RuntimeError("Joint validation threshold lock is not reproducible")
    if reproduced_joint_selection != validation_lock.get(
        "validation_joint_threshold_selection"
    ):
        raise RuntimeError("Joint validation selection evidence is stale")

    reproduced_validation_predictions = project_compatibility_predictions(
        validation_probability_matrix,
        reproduced_joint_vector,
    )
    _require_reproduced_validation_evidence(
        validation_prediction_evidence,
        record_ids["val"],
        validation_truth_matrix,
        validation_probability_matrix,
        reproduced_validation_predictions,
        split_manifest_sha256=str(provenance["split_manifest_sha256"]),
    )

    # The held-out test is scored only after all train/validation evidence is reproduced.
    for label in PROJECT_LABELS:
        target_column = f"target::{label}"
        test_y = frame.loc[split_masks["test"], target_column].to_numpy(dtype=int)
        model = models[label]
        calibrator = calibrators[label]
        if model is None:
            test_probability = np.full(
                len(test_y), constant_probabilities[label], dtype=float
            )
        else:
            raw_test = model.predict_proba(matrices["test"])[:, 1]
            test_probability = _apply_platt(calibrator, raw_test)
        test_targets[label] = test_y
        test_probabilities[label] = test_probability

    truth_matrix = np.column_stack([test_targets[label] for label in PROJECT_LABELS])
    probability_matrix = np.column_stack(
        [test_probabilities[label] for label in PROJECT_LABELS]
    )
    threshold_vector = np.asarray(
        [float(thresholds[label]["threshold"]) for label in PROJECT_LABELS]
    )
    prediction_matrix = project_compatibility_predictions(
        probability_matrix,
        threshold_vector,
    )
    global_metrics = _multilabel_objective(truth_matrix, prediction_matrix)
    coverage_metrics = _prediction_coverage(prediction_matrix)
    subset_accuracy = global_metrics["compatibility_subset_exact_match"]
    bitwise_accuracy = global_metrics["per_label_bitwise_accuracy"]
    micro_f1 = global_metrics["micro_f1"]

    for column, label in enumerate(PROJECT_LABELS):
        threshold = float(thresholds[label]["threshold"])
        calibrator = calibrators[label]
        metric = _class_metrics(
            truth_matrix[:, column],
            probability_matrix[:, column],
            threshold,
            seed=chosen.random_state + column,
            patient_groups=test_patient_groups,
            predictions=prediction_matrix[:, column],
        )
        metric.update(
            {
                "threshold": threshold,
                "per_class_f1_threshold": float(
                    thresholds[label]["per_class_f1_threshold"]
                ),
                "threshold_partition": "validation_only",
                "evaluation_partition": "held_out_test",
                "patient_disjoint": True,
                "patient_overlap_count": 0,
                "probabilities_calibrated": calibrator is not None,
                "calibration_method": (
                    chosen.calibration_method if calibrator is not None else "not_estimable"
                ),
                "prediction_projection_applied": True,
                "evaluation_coverage": 1.0,
                **coverage_metrics,
                "evaluation_records": len(truth_matrix),
                "test_records": len(truth_matrix),
                "split_manifest_sha256": provenance["split_manifest_sha256"],
                "metric_provenance": "held_out_compatibility_head",
            }
        )
        per_class[label] = metric
    for metric in per_class.values():
        metric["global_accuracy"] = subset_accuracy
        metric["global_accuracy_definition"] = "compatibility_subset_exact_match"

    prediction_rows: list[Mapping[str, object]] = []
    for index, record_id in enumerate(record_ids["test"]):
        row: dict[str, object] = {
            "record_id": record_id,
            "evaluation_partition": "held_out_test",
            "threshold_partition": "validation_only",
            "patient_disjoint": True,
            "split_manifest_sha256": provenance["split_manifest_sha256"],
        }
        for column, label in enumerate(PROJECT_LABELS):
            key = label.lower().replace(" / ", "_").replace(" ", "_")
            row[f"true::{key}"] = int(truth_matrix[index, column])
            row[f"probability::{key}"] = float(probability_matrix[index, column])
            row[f"predicted::{key}"] = int(prediction_matrix[index, column])
        prediction_rows.append(row)

    metrics_dir = config.paths.reports / "metrics"
    model_path = config.paths.models / "ptbxl_12sl_compatibility.joblib"
    metrics_path = metrics_dir / "b1_classification_metrics.json"
    predictions_path = metrics_dir / "ptbxl_12sl_test_predictions.parquet"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "calibrators": calibrators,
            "thresholds": thresholds,
            "prediction_projection": "normal_exclusive_and_nonempty_argmax",
            "feature_columns": feature_columns,
            "training_spec": asdict(chosen),
            "provenance": provenance,
        },
        model_path,
        compress=3,
    )
    actual_predictions = write_records_table(predictions_path, prediction_rows)
    payload: dict[str, object] = {
        "artifact_version": 2,
        "ontology_version": config.ontology_version,
        "model_family": "PTB-XL+ 12SL plus adaptive waveform-B compatibility head",
        "training_spec": asdict(chosen),
        "feature_provenance": provenance,
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "train_records": len(record_ids["train"]),
        "val_records": len(record_ids["val"]),
        "test_records": len(record_ids["test"]),
        "evaluation_records": len(record_ids["test"]),
        "patient_disjoint": True,
        "patient_overlap_count": 0,
        "evaluation_partition": "held_out_test",
        "threshold_partition": "validation_only",
        "prediction_projection": "normal_exclusive_and_nonempty_argmax",
        "metric_provenance": "held_out_compatibility_head",
        "validation_lock_path": str(validation_lock_path),
        "validation_lock_sha256": sha256_file(validation_lock_path),
        "global_accuracy": subset_accuracy,
        "global_accuracy_definition": "compatibility_subset_exact_match",
        "compatibility_subset_exact_match": subset_accuracy,
        "per_label_bitwise_accuracy": bitwise_accuracy,
        "micro_f1": float(micro_f1),
        **coverage_metrics,
        "global_metric_intervals": _multilabel_cluster_bootstrap(
            truth_matrix,
            prediction_matrix,
            test_patient_groups,
            seed=chosen.random_state,
        ),
        "accuracy_endpoint_gate_0_90_passed": subset_accuracy >= 0.90,
        "per_class_metrics": per_class,
        "target_support_by_split": provenance["target_support_by_split"],
        "validation_thresholds": thresholds,
        "predictions_path": str(actual_predictions),
        "predictions_sha256": sha256_file(actual_predictions),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
    }
    write_json(metrics_path, payload)
    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "predictions": str(actual_predictions),
    }


def refresh_clustered_metrics(config: ProjectConfig) -> dict[str, str]:
    """Refresh CIs/provenance for frozen predictions without refitting or rescoring."""

    import numpy as np  # type: ignore

    metrics_path = config.paths.reports / "metrics" / "b1_classification_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing frozen compatibility metrics: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions_path = Path(str(payload["predictions_path"]))
    _require_artifact_hash(
        predictions_path,
        payload.get("predictions_sha256"),
        artifact_name="Frozen compatibility predictions",
    )

    manifest_path = _manifest_path(config)
    _require_artifact_hash(
        manifest_path,
        payload.get("split_manifest_sha256"),
        artifact_name="Frozen compatibility split manifest",
    )
    predictions = read_table_frame(predictions_path).copy()
    predictions["record_id"] = predictions["record_id"].astype(str)

    manifest = read_table_frame(manifest_path).copy()
    manifest["record_id"] = manifest["record_id"].astype(str)
    test_manifest = manifest[manifest["split"].astype(str).eq("test")]
    expected_ids = set(test_manifest["record_id"].tolist())
    observed_ids = set(predictions["record_id"].tolist())
    if len(predictions) != len(observed_ids) or observed_ids != expected_ids:
        raise RuntimeError("Frozen prediction IDs do not exactly match the locked test split")
    patient_by_record = {
        str(row.record_id): _patient_key(row.patient_id, row.record_id)
        for row in test_manifest.itertuples(index=False)
    }
    patient_groups = [patient_by_record[record_id] for record_id in predictions["record_id"]]

    per_class = dict(payload["per_class_metrics"])
    truth_matrix = np.column_stack(
        [
            predictions[
                f"true::{label.lower().replace(' / ', '_').replace(' ', '_')}"
            ].to_numpy(dtype=int)
            for label in PROJECT_LABELS
        ]
    )
    probability_matrix = np.column_stack(
        [
            predictions[
                f"probability::{label.lower().replace(' / ', '_').replace(' ', '_')}"
            ].to_numpy(dtype=float)
            for label in PROJECT_LABELS
        ]
    )
    prediction_matrix = np.column_stack(
        [
            predictions[
                f"predicted::{label.lower().replace(' / ', '_').replace(' ', '_')}"
            ].to_numpy(dtype=int)
            for label in PROJECT_LABELS
        ]
    )
    coverage_metrics = _prediction_coverage(prediction_matrix)
    threshold_vector = np.asarray(
        [float(per_class[label]["threshold"]) for label in PROJECT_LABELS]
    )
    expected_prediction_matrix = project_compatibility_predictions(
        probability_matrix,
        threshold_vector,
    )
    if not np.array_equal(prediction_matrix, expected_prediction_matrix.astype(int)):
        raise RuntimeError(
            "Frozen predictions are inconsistent with the locked ontology projection"
        )

    for column, label in enumerate(PROJECT_LABELS):
        truth = truth_matrix[:, column]
        probability = probability_matrix[:, column]
        predicted = prediction_matrix[:, column]
        threshold = float(per_class[label]["threshold"])
        updated = dict(per_class[label])
        # Clustered values and fail-closed clinical coverage supersede legacy fields.
        clustered = _class_metrics(
            truth,
            probability,
            threshold,
            seed=config.seed + column,
            patient_groups=patient_groups,
            predictions=predicted,
        )
        updated.update(clustered)
        updated.update(
            {
                "evaluation_coverage": 1.0,
                **coverage_metrics,
                "evaluation_records": len(predictions),
                "test_records": len(predictions),
                "split_manifest_sha256": sha256_file(manifest_path),
            }
        )
        per_class[label] = updated

    global_metrics = _multilabel_objective(truth_matrix, prediction_matrix)
    subset_accuracy = global_metrics["compatibility_subset_exact_match"]
    bitwise_accuracy = global_metrics["per_label_bitwise_accuracy"]
    for metric in per_class.values():
        metric["global_accuracy"] = subset_accuracy
        metric["global_accuracy_definition"] = "compatibility_subset_exact_match"

    feature_provenance = dict(payload.get("feature_provenance", {}))
    # Preserve the feature-selection claim made when the frozen artifact was
    # created; a metrics-only refresh must not upgrade legacy provenance.
    feature_provenance["split_manifest_sha256"] = sha256_file(manifest_path)
    payload.update(
        {
            "feature_provenance": feature_provenance,
            "split_manifest_sha256": sha256_file(manifest_path),
            "test_records": len(predictions),
            "evaluation_records": len(predictions),
            "test_patient_clusters": len(set(patient_groups)),
            "global_accuracy": subset_accuracy,
            "global_accuracy_definition": "compatibility_subset_exact_match",
            "compatibility_subset_exact_match": subset_accuracy,
            "per_label_bitwise_accuracy": bitwise_accuracy,
            "micro_f1": global_metrics["micro_f1"],
            **coverage_metrics,
            "global_metric_intervals": _multilabel_cluster_bootstrap(
                truth_matrix,
                prediction_matrix,
                patient_groups,
                seed=config.seed,
            ),
            "accuracy_endpoint_gate_0_90_passed": subset_accuracy >= 0.90,
            "per_class_metrics": per_class,
            "predictions_sha256": sha256_file(predictions_path),
        }
    )
    write_json(metrics_path, payload)
    return {"metrics": str(metrics_path), "predictions": str(predictions_path)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/defaults.toml")
    parser.add_argument("--refresh-clustered-metrics", action="store_true")
    parser.add_argument("--validation-audit", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ProjectConfig.load(args.config)
    config.ensure_directories()
    if args.validation_audit:
        result = build_validation_model_lock(config)
    elif args.refresh_clustered_metrics:
        result = refresh_clustered_metrics(config)
    else:
        result = train_and_evaluate_ptbxl_plus_compatibility(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
