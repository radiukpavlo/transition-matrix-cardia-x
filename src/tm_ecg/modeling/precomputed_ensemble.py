"""Development-only weighted ensembles from governed cross-fitted views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from tm_ecg.config import ProjectConfig
from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.modeling.hierarchical import crossfit_hierarchical_probabilities
from tm_ecg.modeling.label_set_decoder import StructuredLabelSetDecoder
from tm_ecg.modeling.structured_experiment import (
    _canonical_hash,
    _error_budget,
    _label_matrix,
    _metrics,
    _read_physically_restricted_development_index,
    _rare_label_gates,
    _sha256_file,
)
from tm_ecg.reproducibility import write_artifact_manifest


LABELS = tuple(PROJECT_LABELS)


def _probability_column(label: str) -> str:
    key = label.lower().replace(" / ", "_").replace(" ", "_")
    return f"probability::{key}"


def _load_precomputed_view(
    view_dir: Path,
    *,
    training_rows: int,
    validation_record_ids: Sequence[str],
) -> tuple[object, object, dict[str, object]]:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    report_path = view_dir / "structured_compatibility_development_metrics.json"
    prediction_path = view_dir / "development_validation_predictions.parquet"
    if not report_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"Incomplete structured view directory: {view_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("confirmatory_labels_opened") is not False:
        raise RuntimeError(f"View is not sealed-development-only: {view_dir}")
    if tuple(report.get("label_order", ())) != LABELS:
        raise RuntimeError(f"View label order differs from the frozen contract: {view_dir}")
    label_access_audit = report.get("design", {}).get("label_access_audit")
    if (
        not isinstance(label_access_audit, Mapping)
        or label_access_audit.get("mode")
        != "physical_parquet_row_filter_before_label_materialization"
        or label_access_audit.get("confirmatory_label_rows_materialized") != 0
    ):
        raise RuntimeError(
            f"View does not prove physically restricted label access: {view_dir}"
        )

    crossfit_paths = sorted(view_dir.glob("crossfit_*/crossfit_probabilities.npz"))
    if not crossfit_paths:
        raise RuntimeError(f"View has no cross-fitted probability artifacts: {view_dir}")
    matrices = []
    for path in crossfit_paths:
        with np.load(path, allow_pickle=False) as payload:
            matrix = np.asarray(payload["probabilities"], dtype=float)
        if matrix.shape != (training_rows, len(LABELS)):
            raise RuntimeError(
                f"Crossfit matrix has unexpected shape {matrix.shape}: {path}"
            )
        matrices.append(matrix)
    oof = np.mean(np.stack(matrices), axis=0)

    validation = pd.read_parquet(prediction_path)
    required = {"record_id", *(_probability_column(label) for label in LABELS)}
    if not required <= set(validation.columns):
        raise RuntimeError(
            f"View predictions lack columns: {sorted(required - set(validation.columns))}"
        )
    validation["record_id"] = validation["record_id"].astype(str)
    if validation["record_id"].duplicated().any():
        raise RuntimeError(f"View has duplicate validation record IDs: {view_dir}")
    validation = validation.set_index("record_id").reindex(validation_record_ids)
    if validation.isna().all(axis=1).any():
        raise RuntimeError(f"View omits governed validation records: {view_dir}")
    validation_matrix = validation[
        [_probability_column(label) for label in LABELS]
    ].to_numpy(dtype=float)
    return oof, validation_matrix, {
        "directory": str(view_dir),
        "report_sha256": _sha256_file(report_path),
        "predictions_sha256": _sha256_file(prediction_path),
        "crossfit_paths": [str(path) for path in crossfit_paths],
        "crossfit_sha256": [_sha256_file(path) for path in crossfit_paths],
        "base_estimators": report.get("estimators", []),
        "label_access_audit": dict(label_access_audit),
        "calibration_gate_passed": bool(
            report.get("calibration_gate_passed", False)
        ),
    }


def run_precomputed_ensemble_experiment(
    config: ProjectConfig,
    *,
    index_path: Path,
    sealed_manifest_path: Path,
    view_dirs: Sequence[Path],
    output_dir: Path,
    weights: Sequence[float],
    crossfit_folds: int = 5,
    seed: int = 2901,
) -> dict[str, object]:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    if len(view_dirs) != 2:
        raise ValueError("The governed weighted experiment currently requires two views")
    sealed = json.loads(sealed_manifest_path.read_text(encoding="utf-8"))
    split_by_patient = {
        str(patient_id): split
        for split, patient_ids in sealed["patient_ids"].items()
        for patient_id in patient_ids
    }
    frame, label_access_audit = _read_physically_restricted_development_index(
        index_path,
        split_by_patient,
    )
    train_mask = frame["sealed_split"].eq("development_train").to_numpy()
    validation_mask = frame["sealed_split"].eq(
        "development_validation"
    ).to_numpy()
    y_all = _label_matrix(frame["labels"].tolist())
    y_train = y_all[train_mask]
    y_validation = y_all[validation_mask]
    groups_train = frame.loc[train_mask, "patient_key"].astype(str).tolist()
    groups_validation = (
        frame.loc[validation_mask, "patient_key"].astype(str).tolist()
    )
    validation_record_ids = (
        frame.loc[validation_mask, "record_id"].astype(str).tolist()
    )

    loaded = [
        _load_precomputed_view(
            view_dir,
            training_rows=len(y_train),
            validation_record_ids=validation_record_ids,
        )
        for view_dir in view_dirs
    ]
    oof_views = [item[0] for item in loaded]
    validation_views = [item[1] for item in loaded]
    view_audits = [item[2] for item in loaded]
    candidates: list[
        tuple[float, float, str, object, object, dict[str, object], dict[str, object]]
    ] = []
    experiment_metrics: dict[str, dict[str, object]] = {}

    for weight_index, first_weight in enumerate(weights):
        weight = float(first_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Ensemble weights must be within [0, 1]")
        oof = weight * oof_views[0] + (1.0 - weight) * oof_views[1]
        validation = (
            weight * validation_views[0]
            + (1.0 - weight) * validation_views[1]
        )
        hierarchical_oof, hierarchical_validation, hierarchy_audit = (
            crossfit_hierarchical_probabilities(
                oof,
                y_train,
                groups_train,
                validation,
                n_splits=crossfit_folds,
                seed=seed + weight_index * 100,
            )
        )
        for hierarchy_name, fit_probabilities, validation_probabilities in (
            ("flat", oof, validation),
            ("hierarchical", hierarchical_oof, hierarchical_validation),
        ):
            for regularization in (0.05, 0.2, 1.0, 5.0):
                decoder = StructuredLabelSetDecoder(
                    regularization=regularization,
                    pairwise_regularization=regularization * 10.0,
                    source_permits_af_afl=True,
                ).fit(fit_probabilities, y_train)
                prediction = decoder.predict(validation_probabilities)
                name = (
                    f"weight_{weight:g}_{hierarchy_name}_decoder_l2_"
                    f"{regularization:g}"
                )
                metrics = _metrics(
                    y_validation,
                    prediction,
                    groups=groups_validation,
                    bootstrap_seed=seed + weight_index * 100 + len(candidates),
                )
                experiment_metrics[name] = metrics
                candidates.append(
                    (
                        float(metrics["compatibility_subset_exact_match"]),
                        float(metrics["micro_f1"]),
                        name,
                        decoder,
                        prediction,
                        metrics,
                        {
                            "first_view_weight": weight,
                            "second_view_weight": 1.0 - weight,
                            "hierarchy": hierarchy_name,
                            "hierarchy_audit": hierarchy_audit,
                            "regularization": regularization,
                        },
                    )
                )
    (
        best_exact,
        _,
        best_name,
        best_decoder,
        best_prediction,
        best_metrics,
        best_parameters,
    ) = max(candidates, key=lambda item: (item[0], item[1], item[2]))

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "record_id": validation_record_ids,
            "patient_id": groups_validation,
            "exact_match": np.all(y_validation == best_prediction, axis=1),
        }
    )
    prediction_frame.to_parquet(
        output_dir / "development_validation_predictions.parquet",
        index=False,
    )
    decoder_path = output_dir / "best_label_set_decoder.json"
    decoder_path.write_text(
        json.dumps(best_decoder.to_artifact(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "version": 1,
        "experiment_id": "compatibility_precomputed_crossfit_ensemble_v1",
        "evaluation_partition": "development_validation",
        "confirmatory_labels_opened": False,
        "label_order": list(LABELS),
        "label_access_audit": label_access_audit,
        "views": view_audits,
        "weight_grid": [float(weight) for weight in weights],
        "crossfit_folds": crossfit_folds,
        "seed": seed,
        "experiments": experiment_metrics,
        "selected_experiment": best_name,
        "selected_parameters": best_parameters,
        "selected_metrics": best_metrics,
        "rare_label_gates": _rare_label_gates(best_metrics),
        "calibration_gate_passed": all(
            bool(audit["calibration_gate_passed"]) for audit in view_audits
        ),
        "error_budget": _error_budget(y_validation, best_prediction),
        "confirmatory_opening_authorized": bool(
            best_exact >= 0.92
            and float(best_metrics["best_class_f1"]) > 0.80
            and _rare_label_gates(best_metrics)["passed"]
            and all(
                bool(audit["calibration_gate_passed"])
                for audit in view_audits
            )
        ),
        "selection_note": (
            "Weights and decoder were selected on development validation only; "
            "the sealed confirmatory set remains unopened."
        ),
        "target_matrix_hash": _canonical_hash(y_train.tolist()),
    }
    report_path = output_dir / "precomputed_ensemble_development_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(
        output_dir,
        producer_command="tm-ecg experiment-precomputed-ensemble",
        input_hashes={
            "index": _sha256_file(index_path),
            "sealed_manifest": _sha256_file(sealed_manifest_path),
            **{
                f"view_{index}": str(audit["report_sha256"])
                for index, audit in enumerate(view_audits)
            },
        },
        code_root=config.paths.root,
    )
    return report


def run(config: ProjectConfig, args: object) -> int:
    root = config.paths.root

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    report = run_precomputed_ensemble_experiment(
        config,
        index_path=resolve(str(getattr(args, "index"))),
        sealed_manifest_path=resolve(str(getattr(args, "sealed_manifest"))),
        view_dirs=[
            resolve(value.strip())
            for value in str(getattr(args, "view_dirs")).split(",")
            if value.strip()
        ],
        output_dir=resolve(str(getattr(args, "output_dir"))),
        weights=[
            float(value.strip())
            for value in str(getattr(args, "weights")).split(",")
            if value.strip()
        ],
        crossfit_folds=int(getattr(args, "crossfit_folds", 5)),
        seed=int(getattr(args, "seed", 2901)),
    )
    print(
        json.dumps(
            {
                "selected_experiment": report["selected_experiment"],
                "selected_metrics": report["selected_metrics"],
                "confirmatory_opening_authorized": report[
                    "confirmatory_opening_authorized"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
