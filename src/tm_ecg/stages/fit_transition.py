"""Typed B-space transform fitting and transition-operator estimation."""

from __future__ import annotations

import random
from math import isfinite, sqrt
from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.features.registry import fit_columns
from tm_ecg.features.registry import FEATURE_SPECS
from tm_ecg.io.common import sha256_file, stable_hash, write_json
from tm_ecg.io.readers import dataset_a_path, find_table, read_table_rows
from tm_ecg.io.tabular import write_records_table
from tm_ecg.stages.shared import write_stage_manifest
from tm_ecg.transition.a_preprocess import (
    apply_a_preprocess_bundle,
    fit_a_preprocess_bundle,
    write_a_preprocess_bundle,
)
from tm_ecg.transition.ridge import (
    apply_transition_package,
    fit_masked_ridge_transition,
    fit_masked_robust_transition,
    reduce_transition_output_rank,
    save_operator_package,
)
from tm_ecg.transition.typed_transforms import fit_transform_bundle, transform_rows
def _artifact_evidence(
    path: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError(f"Duplicate record IDs in transition input artifact: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "records": len(record_ids),
        "record_id_hash": stable_hash(sorted(record_ids)),
    }


def _align(
    a_rows: list[dict[str, object]],
    b_rows: list[dict[str, object]],
    b_columns: list[str],
) -> tuple[list[list[float]], list[list[float | None]], list[str]]:
    a_map = {str(row["record_id"]): row for row in a_rows}
    b_map = {str(row["record_id"]): row for row in b_rows}
    if len(a_map) != len(a_rows) or len(b_map) != len(b_rows):
        raise RuntimeError("Duplicate record IDs detected before transition alignment")
    if set(a_map) != set(b_map):
        missing_a = sorted(set(b_map) - set(a_map))[:10]
        missing_b = sorted(set(a_map) - set(b_map))[:10]
        raise RuntimeError(
            f"Transition row-ID mismatch; missing_from_A={missing_a}, missing_from_B={missing_b}"
        )
    common = sorted(a_map)
    if not common:
        return [], [], []
    a_columns = [column for column in a_map[common[0]].keys() if column not in {"record_id", "split"}]
    a_matrix = []
    b_matrix = []
    retained_ids: list[str] = []
    for record_id in common:
        a_row = a_map[record_id]
        b_row = b_map[record_id]
        if any(a_row.get(column) is None for column in a_columns):
            continue
        a_matrix.append([float(a_row[column]) for column in a_columns])
        b_matrix.append(
            [
                None if b_row.get(column) is None else float(b_row[column])
                for column in b_columns
            ]
        )
        retained_ids.append(record_id)
    return a_matrix, b_matrix, retained_ids


def _reconcile_missing_latents(
    a_rows: list[dict[str, object]],
    b_rows: list[dict[str, object]],
    *,
    maximum_missing_latent_fraction: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Exclude only B rows with a documented unavailable latent, within policy."""

    a_ids = [str(row["record_id"]) for row in a_rows]
    b_ids = [str(row["record_id"]) for row in b_rows]
    if len(set(a_ids)) != len(a_ids) or len(set(b_ids)) != len(b_ids):
        raise RuntimeError("Duplicate record IDs detected during latent-availability audit")
    missing_features = sorted(set(a_ids) - set(b_ids))
    missing_latents = sorted(set(b_ids) - set(a_ids))
    if missing_features:
        raise RuntimeError(
            f"Transition rows have latents but no B features: {missing_features[:10]}"
        )
    exclusion_fraction = len(missing_latents) / max(len(b_ids), 1)
    if exclusion_fraction > maximum_missing_latent_fraction:
        raise RuntimeError(
            "Missing-latent exclusion fraction exceeds policy: "
            f"{len(missing_latents)}/{len(b_ids)}={exclusion_fraction:.6f} > "
            f"{maximum_missing_latent_fraction:.6f}"
        )
    retained = set(a_ids)
    filtered = [row for row in b_rows if str(row["record_id"]) in retained]
    return filtered, {
        "a_row_count": len(a_rows),
        "b_row_count_before_exclusion": len(b_rows),
        "aligned_row_count": len(filtered),
        "missing_latent_count": len(missing_latents),
        "missing_latent_record_ids": missing_latents,
        "missing_latent_fraction": exclusion_fraction,
        "maximum_missing_latent_fraction": maximum_missing_latent_fraction,
        "status": "aligned_after_documented_latent_exclusion" if missing_latents else "aligned",
    }


def _mae(
    y_true: list[list[float | None]],
    y_pred: list[list[float]],
) -> float:
    errors = []
    for true_row, pred_row in zip(y_true, y_pred, strict=False):
        for true_value, pred_value in zip(true_row, pred_row, strict=False):
            if true_value is not None and isfinite(float(true_value)):
                errors.append(abs(float(true_value) - pred_value))
    return sum(errors) / len(errors) if errors else float("inf")


def _mae_by_feature_family(
    y_true: list[list[float | None]],
    y_pred: list[list[float]],
    columns: list[str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for true_row, pred_row in zip(y_true, y_pred, strict=False):
        for column, true_value, pred_value in zip(columns, true_row, pred_row, strict=False):
            if true_value is None or not isfinite(float(true_value)):
                continue
            family = FEATURE_SPECS.get(column, ("unknown", "", "", ""))[0]
            grouped.setdefault(family, []).append(abs(float(true_value) - pred_value))
    return {
        family: sum(values) / len(values)
        for family, values in sorted(grouped.items())
        if values
    }


def _transition_validation_diagnostics(
    y_true: list[list[float | None]],
    y_pred: list[list[float]],
    columns: list[str],
    training_targets: list[list[float | None]],
) -> dict[str, object]:
    """Report feature-level fidelity and state agreement without row deletion."""

    import numpy as np  # type: ignore
    from scipy.stats import spearmanr  # type: ignore

    per_feature: dict[str, dict[str, object]] = {}
    thresholds: list[float | None] = []
    observed_state_rows: list[list[bool]] = [[] for _ in y_true]
    predicted_state_rows: list[list[bool]] = [[] for _ in y_true]
    for column_index, column in enumerate(columns):
        train_values = np.asarray(
            [
                float(row[column_index])
                for row in training_targets
                if row[column_index] is not None
                and isfinite(float(row[column_index]))
            ],
            dtype=float,
        )
        threshold = (
            float(np.median(train_values)) if len(train_values) else None
        )
        thresholds.append(threshold)
        observed_indices = [
            row_index
            for row_index, row in enumerate(y_true)
            if row[column_index] is not None
            and isfinite(float(row[column_index]))
        ]
        if not observed_indices:
            per_feature[column] = {
                "status": "not_estimable",
                "observations": 0,
            }
            continue
        observed = np.asarray(
            [float(y_true[index][column_index]) for index in observed_indices],
            dtype=float,
        )
        predicted = np.asarray(
            [float(y_pred[index][column_index]) for index in observed_indices],
            dtype=float,
        )
        errors = np.abs(observed - predicted)
        scale = float(np.quantile(train_values, 0.75) - np.quantile(train_values, 0.25))
        if not isfinite(scale) or scale <= 1e-12:
            scale = float(np.ptp(train_values)) if len(train_values) else 0.0
        residual = float(np.sum((observed - predicted) ** 2))
        centred = float(np.sum((observed - observed.mean()) ** 2))
        correlation = (
            float(spearmanr(observed, predicted).statistic)
            if len(observed) >= 3
            and float(np.ptp(observed)) > 1e-12
            and float(np.ptp(predicted)) > 1e-12
            else None
        )
        if correlation is not None and not isfinite(correlation):
            correlation = None
        state_agreement = None
        if threshold is not None:
            observed_state = observed >= threshold
            predicted_state = predicted >= threshold
            state_agreement = float(np.mean(observed_state == predicted_state))
            for local_index, row_index in enumerate(observed_indices):
                observed_state_rows[row_index].append(bool(observed_state[local_index]))
                predicted_state_rows[row_index].append(bool(predicted_state[local_index]))
        per_feature[column] = {
            "status": "ok",
            "observations": len(observed),
            "mae": float(errors.mean()),
            "normalized_mae": (
                float(errors.mean() / scale) if scale > 1e-12 else None
            ),
            "r_squared": 1.0 - residual / centred if centred > 1e-12 else None,
            "spearman_rank_correlation": correlation,
            "training_median_predicate_threshold": threshold,
            "threshold_predicate_agreement": state_agreement,
        }
    exact_rows = [
        all(left == right for left, right in zip(observed, predicted, strict=True))
        for observed, predicted in zip(
            observed_state_rows,
            predicted_state_rows,
            strict=True,
        )
        if observed
    ]
    predicate_agreements = [
        float(item["threshold_predicate_agreement"])
        for item in per_feature.values()
        if item.get("threshold_predicate_agreement") is not None
    ]
    return {
        "per_semantic_feature": per_feature,
        "threshold_source": "training_target_median_diagnostic",
        "mean_threshold_predicate_agreement": (
            sum(predicate_agreements) / len(predicate_agreements)
            if predicate_agreements
            else None
        ),
        "exact_semantic_state_accuracy": (
            sum(exact_rows) / len(exact_rows) if exact_rows else None
        ),
        "calibration_status": (
            "not_applicable_to_unbounded_transformed_targets"
        ),
        "training_only_thresholds": dict(zip(columns, thresholds, strict=True)),
    }


def _mean_baseline(
    train: list[list[float | None]],
    rows: int,
) -> list[list[float]]:
    if not train:
        return []
    means = []
    for index in range(len(train[0])):
        observed = [
            float(row[index])
            for row in train
            if row[index] is not None and isfinite(float(row[index]))
        ]
        means.append(sum(observed) / len(observed) if observed else 0.0)
    return [list(means) for _ in range(rows)]


def _bootstrap_transition_stability(
    a_train: list[list[float]],
    b_train: list[list[float | None]],
    columns: list[str],
    *,
    lambda_value: float,
    rank_cap: int,
    replicates: int,
    seed: int,
    top_features: int,
    minimum_frequency: float,
    minimum_target_rows: int = 10,
) -> dict[str, object]:
    """Estimate B-feature salience stability from training-only bootstraps."""

    if not a_train or not b_train or replicates <= 0:
        return {
            "status": "not_estimable",
            "replicates_requested": replicates,
            "replicates_completed": 0,
            "stable_features": [],
        }
    rng = random.Random(seed)
    counts = {column: 0 for column in columns}
    completed = 0
    failed = 0
    keep_count = max(1, min(top_features, len(columns)))
    for _ in range(replicates):
        indices = [rng.randrange(len(a_train)) for _ in range(len(a_train))]
        try:
            payload = fit_masked_ridge_transition(
                [a_train[index] for index in indices],
                [b_train[index] for index in indices],
                lambda_value,
                rank_cap=rank_cap,
                minimum_target_rows=minimum_target_rows,
            )
        except (RuntimeError, ValueError):
            failed += 1
            continue
        operator = payload["operator"]
        scores = {
            column: sqrt(
                sum(float(row[column_index]) ** 2 for row in operator)  # type: ignore[union-attr]
            )
            for column_index, column in enumerate(columns)
        }
        for column, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:keep_count]:
            counts[column] += 1
        completed += 1
    frequencies = {
        column: (count / completed if completed else 0.0)
        for column, count in sorted(counts.items())
    }
    stable = sorted(
        column for column, frequency in frequencies.items()
        if frequency >= minimum_frequency
    )
    return {
        "status": "ok" if completed else "not_estimable",
        "replicates_requested": replicates,
        "replicates_completed": completed,
        "replicates_failed": failed,
        "seed": seed,
        "top_features_per_replicate": keep_count,
        "minimum_selection_frequency": minimum_frequency,
        "selection_frequency": frequencies,
        "stable_features": stable,
    }


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    manifest_stem = (
        "ptbxl_split_index" if dataset == "b1" else "ludb_split_index_repeat_1"
    )
    manifest_path = find_table(config.paths.manifests, manifest_stem)
    if manifest_path is None:
        raise FileNotFoundError(
            f"Missing locked split manifest for transition dataset {dataset}: {manifest_stem}"
        )
    manifest_rows = read_table_rows(manifest_path)
    ontology_versions = {
        str(row.get("ontology_version"))
        for row in manifest_rows
        if row.get("ontology_version") is not None
    }
    if ontology_versions != {config.ontology_version}:
        raise RuntimeError(
            "Transition split-manifest ontology mismatch: "
            f"manifest={sorted(ontology_versions)} active={config.ontology_version}"
        )
    raw_train_path = find_table(config.paths.features, f"{dataset.upper()}_raw_train")
    if raw_train_path is None:
        write_stage_manifest(config, f"fit_transition_{dataset}", {"dataset": dataset, "status": "waiting_for_raw_features"})
        print(f"No raw training features found for {dataset}")
        return 0

    train_rows = read_table_rows(raw_train_path)
    columns = fit_columns(train_rows)
    bundle = fit_transform_bundle(train_rows, columns, eps=float(config.thresholds["eps"]))
    bundle.dataset = dataset

    fit_outputs: dict[str, str] = {}
    fit_output_evidence: dict[str, dict[str, object]] = {}
    raw_input_evidence: dict[str, dict[str, object]] = {}
    transformed_rows_by_split: dict[str, list[dict[str, object]]] = {}
    for path in sorted(config.paths.features.glob(f"{dataset.upper()}_raw_*.parquet")):
        split = path.stem.replace(f"{dataset.upper()}_raw_", "")
        rows = read_table_rows(path)
        raw_input_evidence[split] = _artifact_evidence(path, rows)
        transformed = transform_rows(rows, bundle)
        transformed_rows_by_split[split] = transformed
        fit_path = write_records_table(config.paths.features / f"{dataset.upper()}_fit_{split}.parquet", transformed)
        fit_outputs[split] = str(fit_path)
        fit_output_evidence[split] = _artifact_evidence(fit_path, transformed)

    bundle_path = config.paths.transition / f"{dataset.upper()}_transform_bundle.json"
    write_json(bundle_path, bundle.to_dict())

    a_train_path = dataset_a_path(config, dataset, "train")
    if a_train_path is None:
        write_stage_manifest(
            config,
            f"fit_transition_{dataset}",
            {
                "dataset": dataset,
                "status": "waiting_for_a_train",
                "transform_bundle": str(bundle_path),
                "fit_outputs": fit_outputs,
            },
        )
        print(f"Typed transforms written for {dataset}; no A_train found yet.")
        return 0

    a_train_rows = read_table_rows(a_train_path)
    latent_input_evidence: dict[str, dict[str, object]] = {
        "train": _artifact_evidence(a_train_path, a_train_rows)
    }
    a_bundle, a_train_red_rows = fit_a_preprocess_bundle(
        a_train_rows,
        dataset=dataset,
        variance_retained=float(config.latents["variance_retained"]),
        rank_cap=int(config.transition["rank_cap"]),
    )
    a_bundle_path = config.paths.transition / f"{dataset.upper()}_A_preprocess_bundle.json"
    write_a_preprocess_bundle(a_bundle_path, a_bundle)
    a_red_outputs = {
        "train": str(write_records_table(config.paths.latents / f"A_{dataset}_train_red.parquet", a_train_red_rows))
    }
    a_red_output_evidence: dict[str, dict[str, object]] = {
        "train": _artifact_evidence(Path(a_red_outputs["train"]), a_train_red_rows)
    }
    a_rows_by_split = {"train": a_train_red_rows}
    for split in ("val", "test"):
        path = dataset_a_path(config, dataset, split)
        if path is None:
            continue
        source_rows = read_table_rows(path)
        latent_input_evidence[split] = _artifact_evidence(path, source_rows)
        reduced = apply_a_preprocess_bundle(source_rows, a_bundle)
        a_rows_by_split[split] = reduced
        a_red_outputs[split] = str(write_records_table(config.paths.latents / f"A_{dataset}_{split}_red.parquet", reduced))
        a_red_output_evidence[split] = _artifact_evidence(
            Path(a_red_outputs[split]), reduced
        )

    alignment_audit: dict[str, object] = {}
    for split, a_rows in a_rows_by_split.items():
        if split not in transformed_rows_by_split:
            continue
        reconciled, audit = _reconcile_missing_latents(
            a_rows,
            transformed_rows_by_split[split],
            maximum_missing_latent_fraction=float(
                config.transition.get("maximum_missing_latent_fraction", 0.01)
            ),
        )
        transformed_rows_by_split[split] = reconciled
        alignment_audit[split] = audit

    a_train, b_train, train_ids = _align(a_rows_by_split["train"], transformed_rows_by_split["train"], bundle.fit_columns)
    if not a_train or not b_train:
        raise RuntimeError("No aligned rows between A_train and B_fit_train")

    best = None
    candidate_audit: list[dict[str, object]] = []
    minimum_target_rows = int(
        config.transition.get("minimum_target_observations", 10)
    )
    validation_payload: (
        tuple[list[list[float]], list[list[float | None]], list[str]] | None
    ) = None
    if "val" in a_rows_by_split and "val" in transformed_rows_by_split:
        validation_payload = _align(
            a_rows_by_split["val"], transformed_rows_by_split["val"], bundle.fit_columns
        )
    rank_grid = [
        int(value)
        for value in config.transition.get("rank_grid", [config.transition["rank_cap"]])
    ]

    def consider_candidate(
        candidate: dict[str, object],
        *,
        candidate_id: str,
        rank_cap: int,
    ) -> None:
        nonlocal best
        score = 0.0
        if validation_payload is not None:
            a_val, b_val, _val_ids = validation_payload
            if a_val and b_val:
                predicted = apply_transition_package(a_val, candidate)
                score = _mae(b_val, predicted)
        candidate_audit.append(
            {
                "candidate_id": candidate_id,
                "method": candidate.get("method", "masked_svd_ridge"),
                "validation_mae": score,
                "retained_rank": candidate.get("retained_rank"),
                "rank_cap": rank_cap,
                "lambda_or_alpha": candidate.get(
                    "lambda_value",
                    candidate.get("alpha"),
                ),
            }
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "payload": candidate,
                "rank_cap": rank_cap,
                "candidate_id": candidate_id,
            }

    for lambda_value in config.transition["lambda_grid"]:
        for rank_cap in rank_grid:
            candidate = fit_masked_ridge_transition(
                a_train,
                b_train,
                float(lambda_value),
                rank_cap=min(rank_cap, int(config.transition["rank_cap"])),
                minimum_target_rows=minimum_target_rows,
            )
            consider_candidate(
                candidate,
                candidate_id=f"masked_svd_ridge::{lambda_value}::{rank_cap}",
                rank_cap=rank_cap,
            )
            for output_rank in config.transition.get(
                "output_rank_grid",
                [4, 8, 16],
            ):
                reduced = reduce_transition_output_rank(
                    candidate,
                    output_rank=int(output_rank),
                )
                consider_candidate(
                    reduced,
                    candidate_id=(
                        f"masked_reduced_rank::{lambda_value}::{rank_cap}"
                        f"::{output_rank}"
                    ),
                    rank_cap=rank_cap,
                )
    for method in config.transition.get(
        "robust_challengers",
        ["ridge", "elastic_net", "huber"],
    ):
        for alpha in config.transition.get(
            "robust_alpha_grid",
            [0.001],
        ):
            candidate = fit_masked_robust_transition(
                a_train,
                b_train,
                method=str(method),
                alpha=float(alpha),
                l1_ratio=float(
                    config.transition.get("elastic_net_l1_ratio", 0.5)
                ),
                minimum_target_rows=minimum_target_rows,
            )
            consider_candidate(
                candidate,
                candidate_id=f"masked_standardized_{method}::{alpha}",
                rank_cap=int(candidate.get("retained_rank", 0)),
            )

    if best is None:
        raise RuntimeError("Transition candidate grid is empty")
    validation_family_mae: dict[str, float] = {}
    validation_diagnostics: dict[str, object] = {}
    baseline_score = float("inf")
    if validation_payload is not None:
        a_val, b_val, _val_ids = validation_payload
        predicted = apply_transition_package(a_val, best["payload"])
        validation_family_mae = _mae_by_feature_family(
            b_val, predicted, bundle.fit_columns
        )
        validation_diagnostics = _transition_validation_diagnostics(
            b_val,
            predicted,
            bundle.fit_columns,
            b_train,
        )
        baseline_score = _mae(b_val, _mean_baseline(b_train, len(b_val)))
    selected_representation = (
        "transition" if best["score"] < baseline_score else "direct_b_feature_baseline"
    )
    stability = _bootstrap_transition_stability(
        a_train,
        b_train,
        bundle.fit_columns,
        lambda_value=float(
            best["payload"].get(
                "lambda_value",
                best["payload"].get("alpha", 0.001),
            )
        ),
        rank_cap=int(best["rank_cap"]),
        replicates=int(config.transition.get("stability_bootstrap_replicates", 100)),
        seed=int(config.transition.get("stability_seed", config.seed)),
        top_features=int(config.transition.get("stability_top_features", 20)),
        minimum_frequency=float(
            config.transition.get("stability_minimum_frequency", 0.60)
        ),
        minimum_target_rows=minimum_target_rows,
    )

    operator_path = save_operator_package(config.paths.transition / f"{dataset.upper()}_T_ridge.npz", best["payload"])
    legacy_operator_path = save_operator_package(config.paths.transition / f"{dataset.upper()}_T_ridge.json", best["payload"])
    metadata_path = config.paths.transition / f"{dataset.upper()}_operator_metadata.json"
    operator_sha256 = sha256_file(operator_path)
    legacy_operator_sha256 = sha256_file(legacy_operator_path)
    transform_bundle_sha256 = sha256_file(bundle_path)
    a_preprocess_bundle_sha256 = sha256_file(a_bundle_path)
    write_json(
        metadata_path,
        {
            "artifact_version": 2,
            "dataset": dataset,
            "ontology_version": config.ontology_version,
            "split_manifest_path": str(manifest_path.resolve()),
            "split_manifest_sha256": sha256_file(manifest_path),
            "raw_input_artifacts": raw_input_evidence,
            "fit_output_artifacts": fit_output_evidence,
            "latent_input_artifacts": latent_input_evidence,
            "a_red_output_artifacts": a_red_output_evidence,
            "orientation": "B_hat = A @ T; A is m x k, B is m x l, T is k x l",
            "operator_path": str(operator_path),
            "operator_sha256": operator_sha256,
            "legacy_operator_path": str(legacy_operator_path),
            "legacy_operator_sha256": legacy_operator_sha256,
            "transform_bundle": str(bundle_path),
            "transform_bundle_sha256": transform_bundle_sha256,
            "a_preprocess_bundle": str(a_bundle_path),
            "a_preprocess_bundle_sha256": a_preprocess_bundle_sha256,
            "a_red_outputs": a_red_outputs,
            "b_fit_columns": bundle.fit_columns,
            "b_fit_columns_hash": stable_hash(bundle.fit_columns),
            "a_original_columns": a_bundle.original_columns,
            "a_original_columns_hash": stable_hash(a_bundle.original_columns),
            "a_kept_columns": a_bundle.kept_columns,
            "a_kept_columns_hash": stable_hash(a_bundle.kept_columns),
            "a_dropped_zero_variance_columns": a_bundle.dropped_zero_variance_columns,
            "selected_lambda": best["payload"].get(
                "lambda_value",
                best["payload"].get("alpha"),
            ),
            "selected_candidate_id": best["candidate_id"],
            "selected_method": best["payload"].get("method"),
            "challenger_candidates": candidate_audit,
            "retained_rank": best["payload"]["retained_rank"],
            "singular_values": best["payload"].get("singular_values", []),
            "validation_mae": best["score"],
            "validation_mae_by_feature_family": validation_family_mae,
            "validation_semantic_diagnostics": validation_diagnostics,
            "direct_b_mean_baseline_mae": baseline_score,
            "selected_representation": selected_representation,
            "aligned_training_row_count": len(train_ids),
            "aligned_training_record_id_hash": stable_hash(sorted(train_ids)),
            "row_id_alignment_validated": True,
            "row_id_alignment_audit": alignment_audit,
            "selected_rank_cap": best["rank_cap"],
            "minimum_target_observations": minimum_target_rows,
            "target_training_support": dict(
                zip(
                    bundle.fit_columns,
                    best["payload"].get("target_support", []),
                    strict=True,
                )
            ),
            "target_training_status": dict(
                zip(
                    bundle.fit_columns,
                    best["payload"].get("target_status", []),
                    strict=True,
                )
            ),
            "target_retained_rank": dict(
                zip(
                    bundle.fit_columns,
                    best["payload"].get("target_retained_rank", []),
                    strict=True,
                )
            ),
            "missing_target_policy": best["payload"].get(
                "missing_target_policy"
            ),
            "observation_mask_groups": best["payload"].get(
                "observation_mask_groups"
            ),
            "transition_feature_stability": stability,
            "training_only_fits": ["B typed transform", "A standardizer", "A PCA", "ridge operator"],
        },
    )
    write_stage_manifest(
        config,
        f"fit_transition_{dataset}",
        {
            "dataset": dataset,
            "status": "complete",
            "transform_bundle": str(bundle_path),
            "fit_outputs": fit_outputs,
            "operator_path": str(operator_path),
            "operator_metadata": str(metadata_path),
            "operator_metadata_sha256": sha256_file(metadata_path),
            "operator_sha256": operator_sha256,
            "ontology_version": config.ontology_version,
            "split_manifest_sha256": sha256_file(manifest_path),
            "a_preprocess_bundle": str(a_bundle_path),
            "a_red_outputs": a_red_outputs,
            "selected_lambda": best["payload"].get(
                "lambda_value",
                best["payload"].get("alpha"),
            ),
            "selected_method": best["payload"].get("method"),
        },
    )
    print(f"Transition operator written for {dataset}")
    return 0
