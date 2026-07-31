"""Development-only experiments for the v4 structured compatibility head."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from tm_ecg.config import ProjectConfig
from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.io.readers import read_table_frame
from tm_ecg.modeling.calibration import fit_best_binary_calibrator
from tm_ecg.modeling.compatibility import (
    _canonical_12sl_feature_allowlist,
    _current_12sl_paths,
    _validate_12sl_feature_schema,
    project_compatibility_predictions,
    select_joint_thresholds,
)
from tm_ecg.modeling.confirmatory_split import (
    SPLIT_NAMES,
    verify_sealed_confirmatory_split,
)
from tm_ecg.modeling.crossfit import (
    generate_crossfit_probabilities,
    write_crossfit_artifacts,
)
from tm_ecg.modeling.label_contract import (
    DEFAULT_COMPATIBILITY_CONTRACT_V4,
)
from tm_ecg.modeling.label_set_decoder import StructuredLabelSetDecoder
from tm_ecg.reproducibility import write_artifact_manifest
from tm_ecg.modeling.selective import risk_coverage_curve
from tm_ecg.modeling.hierarchical import (
    crossfit_hierarchical_probabilities,
)
from tm_ecg.features.registry import governed_project_feature_specs


LABELS = tuple(PROJECT_LABELS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _patient_key(patient_id: object, record_id: object) -> str:
    text = str(patient_id).strip()
    return text if text and text.lower() != "nan" else f"record::{record_id}"


def _read_physically_restricted_development_index(
    index_path: Path,
    split_by_patient: Mapping[str, str],
) -> tuple[object, dict[str, object]]:
    """Materialize labels only for development IDs, never sealed IDs."""

    import pandas as pd  # type: ignore
    import pyarrow.dataset as ds  # type: ignore

    if index_path.suffix.lower() != ".parquet":
        raise ValueError(
            "Sealed development experiments require a Parquet index so label "
            "rows can be filtered before materialization"
        )
    safe_columns = ["record_id", "patient_id", "strat_fold"]
    metadata = pd.read_parquet(index_path, columns=safe_columns)
    metadata["record_id"] = metadata["record_id"].astype(str)
    metadata["patient_key"] = [
        _patient_key(patient_id, record_id)
        for patient_id, record_id in zip(
            metadata["patient_id"],
            metadata["record_id"],
            strict=True,
        )
    ]
    metadata["sealed_split"] = metadata["patient_key"].map(split_by_patient)
    folds_one_to_eight = metadata["strat_fold"].astype(str).isin(
        {"1", "2", "3", "4", "5", "6", "7", "8"}
    )
    if metadata.loc[folds_one_to_eight, "sealed_split"].isna().any():
        raise RuntimeError("Sealed manifest omits a folds 1-8 patient")
    if metadata.loc[~folds_one_to_eight, "sealed_split"].notna().any():
        raise RuntimeError("Sealed manifest includes a historical fold 9/10 patient")

    development_metadata = metadata[
        metadata["sealed_split"].isin(
            {"development_train", "development_validation"}
        )
    ].copy()
    development_ids = development_metadata["record_id"].tolist()
    dataset = ds.dataset(index_path, format="parquet")
    label_table = dataset.to_table(
        columns=["record_id", "labels"],
        filter=ds.field("record_id").isin(development_ids),
    )
    labels = label_table.to_pandas()
    labels["record_id"] = labels["record_id"].astype(str)
    if labels["record_id"].duplicated().any():
        raise RuntimeError("Development label projection contains duplicate record IDs")
    development = development_metadata.merge(
        labels,
        on="record_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not development["_merge"].eq("both").all():
        missing = development.loc[
            ~development["_merge"].eq("both"),
            "record_id",
        ].tolist()
        raise RuntimeError(
            f"Physically filtered labels are missing for development rows: {missing[:10]}"
        )
    development = development.drop(columns=["_merge"])
    if development["labels"].eq("").any():
        raise RuntimeError("A development label was unexpectedly empty")
    return development, {
        "mode": "physical_parquet_row_filter_before_label_materialization",
        "initial_columns_loaded": safe_columns,
        "label_columns_loaded": ["record_id", "labels"],
        "label_filter_record_count": len(development_ids),
        "development_label_rows_materialized": len(labels),
        "confirmatory_label_rows_materialized": 0,
        "historical_label_rows_materialized": 0,
    }


def _label_matrix(values: Sequence[object]) -> object:
    import numpy as np  # type: ignore

    matrix = np.zeros((len(values), len(LABELS)), dtype=int)
    for row, value in enumerate(values):
        labels = DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
            value,
            empty_policy="error",
        )
        for label in labels:
            matrix[row, LABELS.index(label)] = 1
    return matrix


def _load_development_design(
    config: ProjectConfig,
    *,
    index_path: Path,
    sealed_manifest_path: Path,
    project_feature_paths: Sequence[Path] = (),
) -> tuple[object, object, list[str], dict[str, object]]:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    verification = verify_sealed_confirmatory_split(sealed_manifest_path)
    if not verification["passed"]:
        raise RuntimeError(f"Sealed split verification failed: {verification}")
    sealed = json.loads(sealed_manifest_path.read_text(encoding="utf-8"))
    patient_ids = sealed["patient_ids"]
    split_by_patient = {
        str(patient_id): split
        for split in SPLIT_NAMES
        for patient_id in patient_ids[split]
    }
    development, label_access_audit = (
        _read_physically_restricted_development_index(
            index_path,
            split_by_patient,
        )
    )

    feature_path, description_path = _current_12sl_paths(config)
    feature_frame = pd.read_csv(feature_path, low_memory=False)
    if "ecg_id" not in feature_frame.columns:
        raise ValueError("PTB-XL+ 12SL table lacks ecg_id")
    feature_frame["ecg_id"] = feature_frame["ecg_id"].astype(str)
    if feature_frame["ecg_id"].duplicated().any():
        raise RuntimeError("PTB-XL+ 12SL table contains duplicate ecg_id")
    schema_audit = _validate_12sl_feature_schema(
        feature_frame.columns,
        description_path,
    )
    allowed = _canonical_12sl_feature_allowlist(description_path)
    numeric = [
        column
        for column in feature_frame.columns
        if column != "ecg_id"
        and column in allowed
        and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if not numeric:
        raise RuntimeError("No permitted numeric 12SL measurements found")
    feature_frame = feature_frame.copy()
    joined = development.merge(
        feature_frame[["ecg_id", *numeric]],
        left_on="record_id",
        right_on="ecg_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[~joined["_merge"].eq("both"), "record_id"].tolist()
        raise RuntimeError(f"12SL features are missing for records: {missing[:10]}")
    project_feature_audit: dict[str, object] = {
        "paths": [],
        "feature_count": 0,
        "matched_rows": 0,
    }
    if project_feature_paths:
        project_frames = []
        project_hashes = {}
        for project_path in project_feature_paths:
            frame = read_table_frame(project_path).copy()
            if "record_id" not in frame.columns:
                raise ValueError(
                    f"Project feature table lacks record_id: {project_path}"
                )
            frame["record_id"] = frame["record_id"].astype(str)
            project_frames.append(frame)
            project_hashes[str(project_path)] = _sha256_file(project_path)
        project_frame = pd.concat(project_frames, ignore_index=True)
        if project_frame["record_id"].duplicated().any():
            raise RuntimeError(
                "Project feature tables contain duplicate record identities"
            )
        governed = governed_project_feature_specs()
        permitted_project_features = [
            feature
            for feature, spec in governed.items()
            if feature in project_frame.columns
            and spec.inference_safe
            and spec.target_leakage_risk == "low_waveform_measurement_only"
            and pd.api.types.is_numeric_dtype(project_frame[feature])
        ]
        renamed = {
            feature: f"project::{feature}"
            for feature in permitted_project_features
        }
        project_frame = project_frame[
            ["record_id", *permitted_project_features]
        ].rename(columns=renamed)
        joined = joined.merge(
            project_frame,
            on="record_id",
            how="left",
            validate="one_to_one",
            indicator="project_feature_merge",
        )
        joined["project::feature_view_available"] = joined[
            "project_feature_merge"
        ].eq("both").astype(float)
        numeric.extend(
            [renamed[feature] for feature in permitted_project_features]
        )
        numeric.append("project::feature_view_available")
        project_feature_audit = {
            "paths": [str(path) for path in project_feature_paths],
            "sha256": project_hashes,
            "feature_count": len(permitted_project_features),
            "feature_names_hash": _canonical_hash(
                permitted_project_features
            ),
            "matched_rows": int(
                joined["project_feature_merge"].eq("both").sum()
            ),
            "missing_rows": int(
                joined["project_feature_merge"].ne("both").sum()
            ),
            "policy": (
                "inference-safe waveform measurements only; signature scores, "
                "labels, source statements, split fields, and outcomes excluded"
            ),
        }
    train_observed = joined["sealed_split"].eq("development_train")
    numeric = [
        column for column in numeric if joined.loc[train_observed, column].notna().any()
    ]
    features = (
        joined[numeric]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )
    metadata = {
        "index_path": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "sealed_manifest_path": str(sealed_manifest_path),
        "sealed_manifest_sha256": _sha256_file(sealed_manifest_path),
        "confirmatory_labels_opened": False,
        "label_access_audit": label_access_audit,
        "development_rows": len(joined),
        "development_train_rows": int(train_observed.sum()),
        "development_validation_rows": int((~train_observed).sum()),
        "development_patient_count": int(joined["patient_key"].nunique()),
        "feature_path": str(feature_path),
        "feature_sha256": _sha256_file(feature_path),
        "feature_description_path": str(description_path),
        "feature_description_sha256": _sha256_file(description_path),
        "feature_count": len(numeric),
        "feature_names_hash": _canonical_hash(numeric),
        "feature_policy": (
            "canonical numeric PTB-XL+ 12SL measurements only; identifiers, "
            "diagnostic statements, source codes, folds, and outcomes excluded"
        ),
        "project_feature_view": project_feature_audit,
        **schema_audit,
    }
    return joined, features, numeric, metadata


def _feature_indices_by_label(
    feature_names: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    patterns = {
        "AF": re.compile(r"(?:^|_)(?:p|pr|rr|hr|atrial)", re.IGNORECASE),
        "AFL": re.compile(r"(?:^|_)(?:p|pr|rr|hr|atrial)", re.IGNORECASE),
        "APB": re.compile(r"(?:p_|pr|rr|hr|qrs)", re.IGNORECASE),
        "PVC": re.compile(r"(?:rr|hr|qrs|r_|s_|axis)", re.IGNORECASE),
        "RBBB spectrum": re.compile(r"(?:qrs|r_|s_|axis)", re.IGNORECASE),
        "LBBB spectrum": re.compile(r"(?:qrs|r_|s_|axis)", re.IGNORECASE),
        "Paced": re.compile(r"(?:qrs|r_|s_|spike|axis)", re.IGNORECASE),
    }
    result: dict[str, tuple[int, ...]] = {}
    all_indices = tuple(range(len(feature_names)))
    for label in LABELS:
        pattern = patterns.get(label)
        selected = (
            tuple(
                index
                for index, feature in enumerate(feature_names)
                if pattern is not None and pattern.search(feature)
            )
            if pattern is not None
            else all_indices
        )
        result[label] = selected if len(selected) >= 20 else all_indices
    return result


def _estimator_factory(name: str) -> Callable[[str, int], object]:
    def factory(_label: str, seed: int) -> object:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier  # type: ignore
        from sklearn.impute import SimpleImputer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        base_name = name.removeprefix("project_")
        if base_name == "hgb":
            return make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=80,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        if base_name == "logistic":
            return make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=seed,
                    solver="liblinear",
                ),
            )
        if base_name == "extra_trees":
            return make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                ExtraTreesClassifier(
                    n_estimators=160,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=seed,
                ),
            )
        raise ValueError(f"Unknown compatibility base estimator: {name}")

    return factory


def _positive_probability(model: object, features: object) -> object:
    import numpy as np  # type: ignore

    probabilities = np.asarray(model.predict_proba(features), dtype=float)  # type: ignore[attr-defined]
    classes = np.asarray(model.classes_)  # type: ignore[attr-defined]
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def _fit_full_base_view(
    *,
    estimator_name: str,
    train_features: object,
    train_targets: object,
    validation_features: object,
    labels: Sequence[str],
    oof_probabilities: object,
    feature_indices: Mapping[str, Sequence[int]],
    seed: int,
) -> tuple[object, dict[str, object]]:
    import numpy as np  # type: ignore

    x_train = np.asarray(train_features)
    x_validation = np.asarray(validation_features)
    y_train = np.asarray(train_targets, dtype=int)
    oof = np.asarray(oof_probabilities, dtype=float)
    probabilities = np.zeros((len(x_validation), len(labels)), dtype=float)
    metadata: dict[str, object] = {}
    factory = _estimator_factory(estimator_name)
    for column, label in enumerate(labels):
        indices = tuple(int(index) for index in feature_indices[label])
        model = factory(label, seed + column)
        model.fit(x_train[:, indices], y_train[:, column])  # type: ignore[attr-defined]
        raw_validation = _positive_probability(
            model,
            x_validation[:, indices],
        )
        calibrator = fit_best_binary_calibrator(
            oof[:, column],
            y_train[:, column],
            random_state=seed + column,
        )
        probabilities[:, column] = calibrator.predict(raw_validation)
        metadata[label] = {
            "estimator": estimator_name,
            "feature_count": len(indices),
            "calibration": calibrator.to_metadata(),
        }
    return np.clip(probabilities, 1e-6, 1 - 1e-6), metadata


def _patient_cluster_exact_interval(
    exact_rows: object,
    groups: Sequence[str],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    exact = np.asarray(exact_rows, dtype=float)
    group_array = np.asarray(groups, dtype=str)
    if len(exact) != len(group_array):
        raise ValueError("Patient groups must align with exact-match rows")
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    successes = np.bincount(inverse, weights=exact)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(unique_groups),
        size=(replicates, len(unique_groups)),
    )
    estimates = successes[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {
        "method": "patient_cluster_bootstrap",
        "seed": seed,
        "replicates": replicates,
        "patient_groups": len(unique_groups),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def _prediction_contract_audit(prediction: object) -> dict[str, object]:
    import numpy as np  # type: ignore

    predicted = np.asarray(prediction, dtype=int)
    normal = predicted[:, LABELS.index("Normal")].astype(bool)
    residual = predicted[:, LABELS.index("Other / unmapped")].astype(bool)
    specific_indices = [
        index
        for index, label in enumerate(LABELS)
        if label not in {"Normal", "Other / unmapped"}
    ]
    specific = predicted[:, specific_indices].any(axis=1)
    empty = predicted.sum(axis=1) == 0
    invalid = empty | (normal & (residual | specific)) | (residual & specific)
    return {
        "invalid_sets": int(invalid.sum()),
        "invalid_set_rate": float(invalid.mean()),
        "empty_sets": int(empty.sum()),
        "normal_abnormal_conflicts": int((normal & (residual | specific)).sum()),
        "residual_specific_conflicts": int((residual & specific).sum()),
    }


def _metrics(
    truth: object,
    prediction: object,
    *,
    groups: Sequence[str] | None = None,
    bootstrap_seed: int = 1701,
) -> dict[str, object]:
    import numpy as np  # type: ignore
    from sklearn.metrics import (  # type: ignore
        f1_score,
        hamming_loss,
        jaccard_score,
        precision_recall_fscore_support,
    )

    y = np.asarray(truth, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    exact_rows = np.all(y == predicted, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predicted,
        average=None,
        zero_division=0,
    )
    per_label = {
        label: {
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(LABELS)
    }
    metrics: dict[str, object] = {
        "records": len(y),
        "exact_successes": int(exact_rows.sum()),
        "exact_errors": int((~exact_rows).sum()),
        "compatibility_subset_exact_match": float(exact_rows.mean()),
        "per_label_bitwise_accuracy": float((y == predicted).mean()),
        "micro_f1": float(f1_score(y, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y, predicted, average="weighted", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y, predicted)),
        "sample_jaccard": float(
            jaccard_score(y, predicted, average="samples", zero_division=0)
        ),
        "best_class_f1": max(float(value) for value in f1),
        "best_class": LABELS[int(f1.argmax())],
        "per_class_metrics": per_label,
        "prediction_contract": _prediction_contract_audit(predicted),
    }
    if groups is not None:
        metrics["patient_cluster_bootstrap_95"] = (
            _patient_cluster_exact_interval(
                exact_rows,
                groups,
                seed=bootstrap_seed,
            )
        )
    return metrics


def _label_set_token(row: object) -> str:
    import numpy as np  # type: ignore

    values = np.asarray(row, dtype=int)
    return " | ".join(
        label for index, label in enumerate(LABELS) if values[index]
    )


def _error_budget(
    truth: object,
    prediction: object,
    *,
    historical_remaining_gap: int = 296,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    y = np.asarray(truth, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    errors: Counter[tuple[str, str]] = Counter()
    for target, output in zip(y, predicted, strict=True):
        if np.array_equal(target, output):
            continue
        errors[(_label_set_token(target), _label_set_token(output))] += 1
    return {
        "development_validation_error_count": int(
            (~np.all(y == predicted, axis=1)).sum()
        ),
        "historical_v4_remaining_success_gap": historical_remaining_gap,
        "historical_maximum_errors_for_strict_gate": 219,
        "largest_error_transitions": [
            {
                "true_label_set": truth_set,
                "predicted_label_set": prediction_set,
                "count": count,
            }
            for (truth_set, prediction_set), count in errors.most_common(50)
        ],
    }


def _rare_label_gates(metrics: Mapping[str, object]) -> dict[str, object]:
    per_class = metrics["per_class_metrics"]
    if not isinstance(per_class, Mapping):
        raise TypeError("Per-class metrics must be an object")
    floors = {"AFL": 0.60, "APB": 0.60, "PVC": 0.65, "Paced": 0.75}
    checks = {
        label: {
            "recall": float(per_class[label]["recall"]),  # type: ignore[index]
            "floor": floor,
            "passed": float(per_class[label]["recall"]) >= floor,  # type: ignore[index]
        }
        for label, floor in floors.items()
    }
    return {
        "checks": checks,
        "passed": all(bool(item["passed"]) for item in checks.values()),
    }


def run_structured_development_experiment(
    config: ProjectConfig,
    *,
    index_path: Path,
    sealed_manifest_path: Path,
    output_dir: Path,
    estimators: Sequence[str],
    crossfit_folds: int = 5,
    seed: int = 1701,
    project_feature_paths: Sequence[Path] = (),
) -> dict[str, object]:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    frame, features, feature_names, design_metadata = _load_development_design(
        config,
        index_path=index_path,
        sealed_manifest_path=sealed_manifest_path,
        project_feature_paths=project_feature_paths,
    )
    train_mask = frame["sealed_split"].eq("development_train").to_numpy()
    validation_mask = frame["sealed_split"].eq(
        "development_validation"
    ).to_numpy()
    x_train = features[train_mask]
    x_validation = features[validation_mask]
    y_all = _label_matrix(frame["labels"].tolist())
    y_train = y_all[train_mask]
    y_validation = y_all[validation_mask]
    groups_train = frame.loc[train_mask, "patient_key"].astype(str).tolist()
    groups_validation = (
        frame.loc[validation_mask, "patient_key"].astype(str).tolist()
    )
    default_feature_indices = _feature_indices_by_label(feature_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_views: dict[str, object] = {}
    validation_views: dict[str, object] = {}
    crossfit_metadata: dict[str, object] = {}
    full_model_metadata: dict[str, object] = {}
    for view_index, estimator_name in enumerate(estimators):
        if estimator_name.startswith("project_"):
            project_indices = tuple(
                index
                for index, feature in enumerate(feature_names)
                if feature.startswith("project::")
            )
            if len(project_indices) < 10:
                raise RuntimeError(
                    "A project_* estimator requires at least ten governed "
                    "project waveform features"
                )
            feature_indices = {
                label: tuple(
                    index
                    for index in default_feature_indices[label]
                    if index in project_indices
                )
                or project_indices
                for label in LABELS
            }
        else:
            feature_indices = default_feature_indices
        crossfit = generate_crossfit_probabilities(
            x_train,
            y_train,
            groups_train,
            LABELS,
            _estimator_factory(estimator_name),
            feature_names=feature_names,
            feature_indices_by_label=feature_indices,
            n_splits=crossfit_folds,
            seed=seed + view_index * 10000,
        )
        write_crossfit_artifacts(
            crossfit,
            output_dir / f"crossfit_{estimator_name}",
        )
        validation_probabilities, model_metadata = _fit_full_base_view(
            estimator_name=estimator_name,
            train_features=x_train,
            train_targets=y_train,
            validation_features=x_validation,
            labels=LABELS,
            oof_probabilities=crossfit.probabilities,
            feature_indices=feature_indices,
            seed=seed + view_index * 10000,
        )
        oof_views[estimator_name] = np.asarray(crossfit.probabilities)
        validation_views[estimator_name] = validation_probabilities
        crossfit_metadata[estimator_name] = crossfit.metadata
        full_model_metadata[estimator_name] = model_metadata

    oof_ensemble = np.mean(
        np.stack([oof_views[name] for name in estimators]),
        axis=0,
    )
    validation_ensemble = np.mean(
        np.stack([validation_views[name] for name in estimators]),
        axis=0,
    )
    hierarchical_oof, hierarchical_validation, hierarchical_audit = (
        crossfit_hierarchical_probabilities(
            oof_ensemble,
            y_train,
            groups_train,
            validation_ensemble,
            n_splits=crossfit_folds,
            seed=seed + 75000,
        )
    )
    initial_thresholds = np.full(len(LABELS), 0.5, dtype=float)
    thresholds, threshold_audit = select_joint_thresholds(
        y_train,
        oof_ensemble,
        initial_thresholds,
        maximum_passes=4,
    )
    independent_prediction = project_compatibility_predictions(
        validation_ensemble,
        thresholds,
    ).astype(int)
    experiment_metrics: dict[str, dict[str, object]] = {
        "independent_ensemble_joint_thresholds": _metrics(
            y_validation,
            independent_prediction,
            groups=groups_validation,
            bootstrap_seed=seed + 81001,
        )
    }
    predictions: dict[str, object] = {
        "independent_ensemble_joint_thresholds": independent_prediction
    }
    hierarchical_thresholds, hierarchical_threshold_audit = (
        select_joint_thresholds(
            y_train,
            hierarchical_oof,
            initial_thresholds,
            maximum_passes=4,
        )
    )
    hierarchical_independent_prediction = project_compatibility_predictions(
        hierarchical_validation,
        hierarchical_thresholds,
    ).astype(int)
    experiment_metrics[
        "hierarchical_independent_joint_thresholds"
    ] = _metrics(
        y_validation,
        hierarchical_independent_prediction,
        groups=groups_validation,
        bootstrap_seed=seed + 81002,
    )
    predictions[
        "hierarchical_independent_joint_thresholds"
    ] = hierarchical_independent_prediction

    decoder_candidates: list[
        tuple[
            float,
            float,
            StructuredLabelSetDecoder,
            object,
            dict[str, object],
            str,
            object,
        ]
    ] = []
    for variant, train_probabilities, validation_probabilities in (
        ("flat", oof_ensemble, validation_ensemble),
        (
            "hierarchical",
            hierarchical_oof,
            hierarchical_validation,
        ),
    ):
        for regularization in (0.05, 0.2, 1.0, 5.0):
            decoder = StructuredLabelSetDecoder(
                regularization=regularization,
                pairwise_regularization=regularization * 10.0,
                source_permits_af_afl=True,
            ).fit(train_probabilities, y_train)
            structured_prediction = decoder.predict(
                validation_probabilities
            )
            metrics = _metrics(
                y_validation,
                structured_prediction,
                groups=groups_validation,
                bootstrap_seed=seed + 82000 + len(decoder_candidates),
            )
            name = (
                f"{variant}_structured_log_linear_l2_"
                f"{regularization:g}"
            )
            experiment_metrics[name] = metrics
            predictions[name] = structured_prediction
            decoder_candidates.append(
                (
                    float(metrics["compatibility_subset_exact_match"]),
                    float(metrics["micro_f1"]),
                    decoder,
                    structured_prediction,
                    metrics,
                    name,
                    validation_probabilities,
                )
            )
    (
        best_exact,
        _,
        best_decoder,
        best_prediction,
        best_metrics,
        best_name,
        best_decoder_probabilities,
    ) = max(
        decoder_candidates,
        key=lambda item: (item[0], item[1], -item[2].regularization),
    )
    best_decoder.write_artifact(output_dir / "best_label_set_decoder.json")
    DEFAULT_COMPATIBILITY_CONTRACT_V4.validate_prediction_matrix(
        best_prediction
    )
    rare_gates = _rare_label_gates(best_metrics)
    exact_gate = best_exact >= 0.92
    best_f1_gate = float(best_metrics["best_class_f1"]) > 0.80
    calibration_checks: dict[str, dict[str, object]] = {}
    for estimator_name, per_label in full_model_metadata.items():
        if not isinstance(per_label, Mapping):
            continue
        for label, model_row in per_label.items():
            if not isinstance(model_row, Mapping):
                continue
            calibration = model_row.get("calibration", {})
            if not isinstance(calibration, Mapping):
                continue
            method = str(calibration.get("method", ""))
            selection_metrics = calibration.get("selection_metrics", {})
            selected_metrics = (
                selection_metrics.get(method, {})
                if isinstance(selection_metrics, Mapping)
                else {}
            )
            ece = (
                selected_metrics.get("expected_calibration_error")
                if isinstance(selected_metrics, Mapping)
                else None
            )
            passed = (
                ece is not None
                and math.isfinite(float(str(ece)))
                and float(str(ece)) <= 0.05
                and calibration.get("selection_partition")
                == "nested_development_calibration"
            )
            calibration_checks[f"{estimator_name}::{label}"] = {
                "method": method,
                "expected_calibration_error": ece,
                "passed": passed,
            }
    calibration_gate_passed = bool(calibration_checks) and all(
        bool(item["passed"]) for item in calibration_checks.values()
    )

    baseline = experiment_metrics["independent_ensemble_joint_thresholds"]
    baseline_per_class = baseline["per_class_metrics"]
    best_per_class = best_metrics["per_class_metrics"]
    class_regressions = {
        label: {
            "support": int(best_per_class[label]["support"]),  # type: ignore[index]
            "baseline_f1": float(baseline_per_class[label]["f1"]),  # type: ignore[index]
            "best_f1": float(best_per_class[label]["f1"]),  # type: ignore[index]
            "delta": float(best_per_class[label]["f1"])  # type: ignore[index]
            - float(baseline_per_class[label]["f1"]),  # type: ignore[index]
        }
        for label in LABELS
        if int(best_per_class[label]["support"]) >= 20  # type: ignore[index]
        and float(best_per_class[label]["f1"])  # type: ignore[index]
        < float(baseline_per_class[label]["f1"]) - 0.05  # type: ignore[index]
    }
    from scipy.special import softmax  # type: ignore

    candidate_probabilities = softmax(
        best_decoder.candidate_scores(best_decoder_probabilities),
        axis=1,
    )
    selective_audit = risk_coverage_curve(
        y_validation,
        best_prediction,
        candidate_probabilities.max(axis=1),
    )

    prediction_rows: list[dict[str, object]] = []
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    for row_index, row in enumerate(validation_frame.itertuples(index=False)):
        output: dict[str, object] = {
            "record_id": str(row.record_id),
            "patient_id": str(row.patient_key),
            "evaluation_partition": "development_validation",
            "confirmatory_label": False,
            "true_label_set": _label_set_token(y_validation[row_index]),
            "predicted_label_set": _label_set_token(best_prediction[row_index]),
            "exact_match": bool(
                np.array_equal(
                    y_validation[row_index],
                    best_prediction[row_index],
                )
            ),
        }
        for column, label in enumerate(LABELS):
            key = label.lower().replace(" / ", "_").replace(" ", "_")
            output[f"true::{key}"] = int(y_validation[row_index, column])
            output[f"probability::{key}"] = float(
                validation_ensemble[row_index, column]
            )
            output[f"predicted::{key}"] = int(
                best_prediction[row_index, column]
            )
        prediction_rows.append(output)
    prediction_path = output_dir / "development_validation_predictions.parquet"
    pd.DataFrame(prediction_rows).to_parquet(prediction_path, index=False)

    report = {
        "version": 1,
        "experiment_id": "compatibility_structured_v4_development",
        "evaluation_partition": "development_validation",
        "confirmatory_labels_opened": False,
        "design": design_metadata,
        "estimators": list(estimators),
        "crossfit_folds": crossfit_folds,
        "seed": seed,
        "label_order": list(LABELS),
        "feature_subsets": {
            label: {
                "feature_count": len(default_feature_indices[label]),
                "feature_names_hash": _canonical_hash(
                    [
                        feature_names[index]
                        for index in default_feature_indices[label]
                    ]
                ),
            }
            for label in LABELS
        },
        "crossfit": crossfit_metadata,
        "full_models": full_model_metadata,
        "threshold_selection": threshold_audit,
        "hierarchical_gate": hierarchical_audit,
        "hierarchical_threshold_selection": hierarchical_threshold_audit,
        "experiments": experiment_metrics,
        "selected_experiment": best_name,
        "selected_metrics": best_metrics,
        "selection_constraints": {
            "development_exact_at_least_0_92": {
                "value": best_exact,
                "threshold": 0.92,
                "passed": exact_gate,
            },
            "best_class_f1_above_0_80": {
                "value": best_metrics["best_class_f1"],
                "threshold": 0.80,
                "passed": best_f1_gate,
            },
            "rare_label_recall_floors": rare_gates,
            "nested_calibration_ece_at_most_0_05": {
                "passed": calibration_gate_passed,
                "checks": calibration_checks,
            },
            "support_ge_20_f1_regressions_gt_0_05": class_regressions,
            "all_predictions_contract_valid": True,
        },
        "confirmatory_opening_authorized": (
            exact_gate
            and best_f1_gate
            and rare_gates["passed"]
            and calibration_gate_passed
            and not class_regressions
        ),
        "calibration_gate_passed": calibration_gate_passed,
        "selective_risk_audit": selective_audit,
        "error_budget": _error_budget(y_validation, best_prediction),
        "predictions_path": str(prediction_path),
        "predictions_sha256": _sha256_file(prediction_path),
        "decoder_path": str(output_dir / "best_label_set_decoder.json"),
        "decoder_sha256": _sha256_file(
            output_dir / "best_label_set_decoder.json"
        ),
    }
    report_path = output_dir / "structured_compatibility_development_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(
        output_dir,
        producer_command="tm-ecg experiment-structured-compatibility",
        input_hashes={
            "index": _sha256_file(index_path),
            "sealed_manifest": _sha256_file(sealed_manifest_path),
        },
        code_root=output_dir.parents[2] if len(output_dir.parents) > 2 else None,
    )
    return report


def run(config: ProjectConfig, args: object) -> int:
    root = config.paths.root

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    estimators = tuple(
        item.strip()
        for item in str(getattr(args, "estimators", "hgb,logistic")).split(",")
        if item.strip()
    )
    report = run_structured_development_experiment(
        config,
        index_path=resolve(str(getattr(args, "index"))),
        sealed_manifest_path=resolve(str(getattr(args, "sealed_manifest"))),
        output_dir=resolve(str(getattr(args, "output_dir"))),
        estimators=estimators,
        crossfit_folds=int(getattr(args, "crossfit_folds", 5)),
        seed=int(getattr(args, "seed", 1701)),
        project_feature_paths=tuple(
            resolve(item.strip())
            for item in str(
                getattr(args, "project_feature_paths", "")
            ).split(",")
            if item.strip()
        ),
    )
    print(
        json.dumps(
            {
                "selected_experiment": report["selected_experiment"],
                "selected_metrics": report["selected_metrics"],
                "confirmatory_opening_authorized": report[
                    "confirmatory_opening_authorized"
                ],
                "output": str(
                    resolve(str(getattr(args, "output_dir")))
                    / "structured_compatibility_development_metrics.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
