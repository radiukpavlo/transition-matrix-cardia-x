"""Patient-grouped, nested-calibrated out-of-fold base predictions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tm_ecg.modeling.calibration import (
    fit_best_binary_calibrator,
)


EstimatorFactory = Callable[[str, int], object]


def _array_hash(array: object) -> str:
    import numpy as np  # type: ignore

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
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


def _rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def patient_grouped_fold_assignments(
    targets: object,
    groups: Sequence[object],
    *,
    n_splits: int = 5,
    seed: int = 17,
) -> object:
    """Greedily balance multilabel support while keeping patients intact."""

    import numpy as np  # type: ignore

    y = np.asarray(targets, dtype=int)
    group_values = np.asarray([str(group) for group in groups], dtype=object)
    if y.ndim != 2 or len(y) != len(group_values):
        raise ValueError("Crossfit targets and groups must have aligned rows")
    if n_splits < 2:
        raise ValueError("Crossfit requires at least two folds")
    group_indices: dict[str, object] = {}
    for group in sorted(set(group_values.tolist())):
        group_indices[group] = np.flatnonzero(group_values == group)
    if len(group_indices) < n_splits:
        raise ValueError("Crossfit has fewer patient groups than folds")

    total_label_counts = y.sum(axis=0).astype(float)
    group_support = {
        group: y[indices].sum(axis=0)
        for group, indices in group_indices.items()
    }
    label_group_support = np.zeros(y.shape[1], dtype=int)
    for support in group_support.values():
        label_group_support += support > 0

    def rarity(group: str) -> float:
        support = group_support[group]
        present = np.flatnonzero(support > 0)
        if not len(present):
            return 0.0
        return max(1.0 / max(label_group_support[index], 1) for index in present)

    ordered = sorted(
        group_indices,
        key=lambda group: (
            -rarity(group),
            -len(group_indices[group]),
            _rank(group, seed),
        ),
    )
    fold_rows = np.zeros(n_splits, dtype=int)
    fold_groups = np.zeros(n_splits, dtype=int)
    fold_labels = np.zeros((n_splits, y.shape[1]), dtype=float)
    assignments: dict[str, int] = {}
    target_rows = len(y) / n_splits
    target_groups = len(group_indices) / n_splits
    target_labels = total_label_counts / n_splits

    for group in ordered:
        indices = group_indices[group]
        support = group_support[group]

        def score(fold: int) -> tuple[float, str]:
            row_delta = (
                ((fold_rows[fold] + len(indices) - target_rows) / max(target_rows, 1)) ** 2
                - ((fold_rows[fold] - target_rows) / max(target_rows, 1)) ** 2
            )
            group_delta = (
                ((fold_groups[fold] + 1 - target_groups) / max(target_groups, 1)) ** 2
                - ((fold_groups[fold] - target_groups) / max(target_groups, 1)) ** 2
            )
            label_delta = 0.0
            for column, addition in enumerate(support):
                if addition <= 0:
                    continue
                target = max(target_labels[column], 1.0)
                current = fold_labels[fold, column]
                label_delta += (
                    ((current + addition - target) / target) ** 2
                    - ((current - target) / target) ** 2
                )
            return (
                3.0 * row_delta + group_delta + 2.0 * label_delta,
                _rank(f"{group}\0{fold}", seed),
            )

        selected = min(range(n_splits), key=score)
        assignments[group] = selected
        fold_rows[selected] += len(indices)
        fold_groups[selected] += 1
        fold_labels[selected] += support

    row_assignments = np.asarray(
        [assignments[str(group)] for group in group_values],
        dtype=int,
    )
    if set(row_assignments.tolist()) != set(range(n_splits)):
        raise RuntimeError("Crossfit fold construction produced an empty fold")
    for group, indices in group_indices.items():
        if len(set(row_assignments[indices].tolist())) != 1:
            raise RuntimeError(f"Patient group was split across folds: {group}")
    return row_assignments


@dataclass(slots=True)
class CrossfitResult:
    probabilities: object
    fold_assignments: object
    calibration_metadata: dict[str, list[dict[str, object]]]
    fold_metadata: list[dict[str, object]]
    metadata: dict[str, object]


def _positive_probability(model: object, features: object) -> object:
    import numpy as np  # type: ignore

    probabilities = model.predict_proba(features)  # type: ignore[attr-defined]
    matrix = np.asarray(probabilities, dtype=float)
    classes = np.asarray(model.classes_)  # type: ignore[attr-defined]
    if matrix.ndim != 2 or 1 not in classes:
        raise RuntimeError("Binary estimator does not expose a positive class")
    return matrix[:, int(np.flatnonzero(classes == 1)[0])]


def _inner_calibration_mask(
    groups: object,
    *,
    label: str,
    seed: int,
    fraction_denominator: int = 5,
) -> object:
    import numpy as np  # type: ignore

    return np.asarray(
        [
            int(_rank(f"{label}\0{group}", seed), 16) % fraction_denominator == 0
            for group in groups
        ],
        dtype=bool,
    )


def generate_crossfit_probabilities(
    features: object,
    targets: object,
    groups: Sequence[object],
    labels: Sequence[str],
    estimator_factory: EstimatorFactory,
    *,
    feature_names: Sequence[str] = (),
    feature_indices_by_label: Mapping[str, Sequence[int]] | None = None,
    n_splits: int = 5,
    seed: int = 17,
) -> CrossfitResult:
    """Generate complete OOF probabilities with nested calibration."""

    import numpy as np  # type: ignore

    x = np.asarray(features)
    y = np.asarray(targets, dtype=int)
    group_values = np.asarray([str(group) for group in groups], dtype=object)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Crossfit features and targets must be aligned matrices")
    if y.shape[1] != len(labels) or len(group_values) != len(y):
        raise ValueError("Crossfit labels/groups do not align with target matrix")
    assignments = patient_grouped_fold_assignments(
        y,
        group_values,
        n_splits=n_splits,
        seed=seed,
    )
    probabilities = np.full(y.shape, np.nan, dtype=float)
    calibration_metadata: dict[str, list[dict[str, object]]] = {
        str(label): [] for label in labels
    }
    fold_metadata: list[dict[str, object]] = []

    for fold in range(n_splits):
        outer_validation = assignments == fold
        outer_training = ~outer_validation
        training_groups = group_values[outer_training]
        fold_row = {
            "fold": fold,
            "training_rows": int(outer_training.sum()),
            "validation_rows": int(outer_validation.sum()),
            "training_patient_hash": _canonical_hash(
                sorted(set(training_groups.tolist()))
            ),
            "validation_patient_hash": _canonical_hash(
                sorted(set(group_values[outer_validation].tolist()))
            ),
        }
        fold_metadata.append(fold_row)
        for column, label in enumerate(labels):
            selected_indices = tuple(
                int(index)
                for index in (
                    feature_indices_by_label.get(str(label), ())
                    if feature_indices_by_label is not None
                    else ()
                )
            )
            label_features = (
                x[:, selected_indices] if selected_indices else x
            )
            outer_y = y[outer_training, column]
            validation_y = y[outer_validation, column]
            if len(set(outer_y.tolist())) < 2:
                prevalence = float(outer_y.mean()) if len(outer_y) else 0.0
                probabilities[outer_validation, column] = prevalence
                calibration_metadata[str(label)].append(
                    {
                        "fold": fold,
                        "method": "constant_training_prevalence",
                        "training_support": int(outer_y.sum()),
                        "validation_support": int(validation_y.sum()),
                    }
                )
                continue

            calibration_mask_local = _inner_calibration_mask(
                training_groups,
                label=str(label),
                seed=seed + fold * 101,
            )
            fitting_mask_local = ~calibration_mask_local
            can_calibrate = (
                calibration_mask_local.sum() >= 20
                and fitting_mask_local.sum() >= 20
                and len(set(outer_y[calibration_mask_local].tolist())) == 2
                and len(set(outer_y[fitting_mask_local].tolist())) == 2
            )
            training_indices = np.flatnonzero(outer_training)
            if can_calibrate:
                fit_indices = training_indices[fitting_mask_local]
                calibration_indices = training_indices[calibration_mask_local]
                estimator = estimator_factory(
                    str(label),
                    seed + fold * 1009 + column,
                )
                estimator.fit(
                    label_features[fit_indices],
                    y[fit_indices, column],
                )  # type: ignore[attr-defined]
                calibration_raw = _positive_probability(
                    estimator,
                    label_features[calibration_indices],
                )
                calibrator = fit_best_binary_calibrator(
                    calibration_raw,
                    y[calibration_indices, column],
                    random_state=seed + fold * 1009 + column,
                )
                raw_outer = _positive_probability(
                    estimator,
                    label_features[outer_validation],
                )
                probabilities[outer_validation, column] = calibrator.predict(
                    raw_outer
                )
                calibration_row = calibrator.to_metadata()
                calibration_row.update(
                    {
                        "fold": fold,
                        "base_fit_rows": int(len(fit_indices)),
                        "calibration_rows": int(len(calibration_indices)),
                        "validation_support": int(validation_y.sum()),
                    }
                )
            else:
                estimator = estimator_factory(
                    str(label),
                    seed + fold * 1009 + column,
                )
                estimator.fit(
                    label_features[outer_training],
                    outer_y,
                )  # type: ignore[attr-defined]
                probabilities[outer_validation, column] = _positive_probability(
                    estimator,
                    label_features[outer_validation],
                )
                calibration_row = {
                    "fold": fold,
                    "method": "identity_insufficient_nested_support",
                    "base_fit_rows": int(outer_training.sum()),
                    "calibration_rows": 0,
                    "positive_support": int(outer_y.sum()),
                    "negative_support": int(len(outer_y) - outer_y.sum()),
                    "validation_support": int(validation_y.sum()),
                }
            calibration_metadata[str(label)].append(calibration_row)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Crossfit probability matrix contains missing/non-finite values")
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    metadata = {
        "version": 1,
        "rows": len(y),
        "labels": list(labels),
        "n_splits": n_splits,
        "seed": seed,
        "patient_group_count": len(set(group_values.tolist())),
        "patient_disjoint_within_folds": True,
        "feature_names_hash": _canonical_hash(list(feature_names)),
        "feature_indices_by_label_hash": _canonical_hash(
            {
                label: list(
                    feature_indices_by_label.get(label, ())
                    if feature_indices_by_label is not None
                    else ()
                )
                for label in labels
            }
        ),
        "features_hash": _array_hash(x),
        "targets_hash": _array_hash(y),
        "fold_assignments_hash": _array_hash(assignments),
        "probability_checksum": _array_hash(probabilities),
    }
    return CrossfitResult(
        probabilities=probabilities,
        fold_assignments=assignments,
        calibration_metadata=calibration_metadata,
        fold_metadata=fold_metadata,
        metadata=metadata,
    )


def write_crossfit_artifacts(
    result: CrossfitResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    import numpy as np  # type: ignore

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "crossfit_probabilities.npz"
    np.savez_compressed(
        matrix_path,
        probabilities=result.probabilities,
        fold_assignments=result.fold_assignments,
    )
    metadata_path = destination / "crossfit_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                **result.metadata,
                "folds": result.fold_metadata,
                "calibration": result.calibration_metadata,
                "matrix_sha256": _sha256_file(matrix_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"matrix": matrix_path, "metadata": metadata_path}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
