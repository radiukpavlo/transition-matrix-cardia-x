"""Render experiment-level report artifacts."""

from __future__ import annotations

import json
import random

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import sha256_file, write_json
from tm_ecg.io.readers import read_table_rows
from tm_ecg.reporting.reports import write_bootstrap_report, write_metrics_markdown
from tm_ecg.stages.shared import write_stage_manifest


def _table(config: ProjectConfig, directory_name: str, stem: str):
    directory = getattr(config.paths, directory_name)
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _row_count(config: ProjectConfig, directory_name: str, stem: str) -> int | None:
    path = _table(config, directory_name, stem)
    if path is None:
        return None
    return len(read_table_rows(path))


def _null_rate(rows: list[dict[str, object]], columns: list[str]) -> float:
    if not rows or not columns:
        return 0.0
    total = len(rows) * len(columns)
    missing = sum(1 for row in rows for column in columns if row.get(column) in {None, ""})
    return missing / total


def _record_error_clusters(
    true_rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
    *,
    allowed_missing_prediction_ids: set[str] | None = None,
) -> tuple[dict[str, list[float]], int]:
    true_ids = [str(row.get("record_id")) for row in true_rows]
    pred_ids = [str(row.get("record_id")) for row in pred_rows]
    if len(true_ids) != len(set(true_ids)) or len(pred_ids) != len(set(pred_ids)):
        raise RuntimeError("Duplicate record IDs in transition evaluation artifacts")
    missing_predictions = set(true_ids) - set(pred_ids)
    unexpected_predictions = set(pred_ids) - set(true_ids)
    allowed_missing = allowed_missing_prediction_ids or set()
    if unexpected_predictions or missing_predictions != allowed_missing:
        raise RuntimeError(
            "Transition truth and prediction record IDs do not match the documented alignment"
        )
    pred_by_id = {str(row.get("record_id")): row for row in pred_rows}
    clusters: dict[str, list[float]] = {}
    opportunities = 0
    for row in true_rows:
        record_id = str(row.get("record_id"))
        if record_id in allowed_missing:
            continue
        pred = pred_by_id[record_id]
        errors: list[float] = []
        for column, value in row.items():
            if column in {"record_id", "split", "qtc_formula_code"}:
                continue
            opportunities += 1
            try:
                errors.append(abs(float(value) - float(pred[column])))
            except (KeyError, TypeError, ValueError):
                continue
        clusters[record_id] = errors
    return clusters, opportunities


def _cluster_bootstrap_mean_ci(
    clusters: dict[str, list[float]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = [value for errors in clusters.values() for value in errors]
    if not values:
        return float("nan"), float("nan"), float("nan")
    point = sum(values) / len(values)
    record_ids = sorted(clusters)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = [rng.choice(record_ids) for _ in record_ids]
        sampled_values = [
            value for record_id in sampled for value in clusters[record_id]
        ]
        if sampled_values:
            estimates.append(sum(sampled_values) / len(sampled_values))
    estimates.sort()
    lower_index = int(0.025 * (len(estimates) - 1))
    upper_index = int(0.975 * (len(estimates) - 1))
    return point, estimates[lower_index], estimates[upper_index]


def run(config: ProjectConfig, args: object) -> int:
    experiment = getattr(args, "experiment")
    sections: dict[str, list[str]] = {"Artifact Completeness": [], "Explanation Quality": [], "Known Limitations": []}
    bootstrap_rows = []
    transition_metrics: dict[str, object] = {
        "artifact_version": 1,
        "experiment": experiment,
        "ontology_version": config.ontology_version,
        "confidence_interval_method": (
            f"record_cluster_bootstrap_percentile_{int(config.reporting['bootstrap_replicates'])}"
        ),
        "datasets": {},
    }
    for dataset in ("B1", "B2"):
        dataset_metrics: dict[str, object] = {}
        operator_metadata_path = (
            config.paths.transition / f"{dataset}_operator_metadata.json"
        )
        if not operator_metadata_path.exists():
            raise FileNotFoundError(
                f"Missing transition metadata for report: {operator_metadata_path}"
            )
        operator_metadata = json.loads(
            operator_metadata_path.read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test"):
            count = _row_count(config, "features", f"{dataset}_raw_{split}")
            sections["Artifact Completeness"].append(f"{dataset}_raw_{split}: {count if count is not None else 'missing'} rows.")
        train_path = _table(config, "features", f"{dataset}_raw_train")
        if train_path is not None:
            rows = read_table_rows(train_path)
            columns = [column for column in rows[0] if column not in {"record_id", "split", "qtc_formula_code"}] if rows else []
            sections["Artifact Completeness"].append(f"{dataset}_raw_train null rate across feature columns: {_null_rate(rows, columns):.4f}.")
        for split in ("val", "test"):
            true_path = _table(config, "features", f"{dataset}_fit_{split}")
            pred_path = _table(config, "transition", f"{dataset}_B_hat_fit_{split}")
            if true_path is None or pred_path is None:
                sections["Explanation Quality"].append(f"{dataset} {split}: transition prediction artifacts missing.")
                continue
            true_rows = read_table_rows(true_path)
            pred_rows = read_table_rows(pred_path)
            alignment = dict(
                dict(operator_metadata.get("row_id_alignment_audit", {})).get(
                    split, {}
                )
            )
            documented_missing = {
                str(value)
                for value in alignment.get("missing_latent_record_ids", [])
            }
            clusters, opportunities = _record_error_clusters(
                true_rows,
                pred_rows,
                allowed_missing_prediction_ids=documented_missing,
            )
            point, lower, upper = _cluster_bootstrap_mean_ci(
                clusters,
                replicates=int(config.reporting["bootstrap_replicates"]),
                seed=config.seed,
            )
            observed_comparisons = sum(len(values) for values in clusters.values())
            observation_coverage = (
                observed_comparisons / opportunities if opportunities else 0.0
            )
            sections["Explanation Quality"].append(
                f"{dataset} {split}: B_fit absolute-error mean={point:.6f}, "
                f"95% record-cluster bootstrap CI [{lower:.6f}, {upper:.6f}], "
                f"observed-target coverage={observation_coverage:.4f}."
            )
            dataset_metrics[split] = {
                "status": "ok",
                "record_count": len(clusters),
                "source_record_count": len(true_rows),
                "record_coverage": len(clusters) / len(true_rows) if true_rows else 0.0,
                "record_id_alignment": (
                    "exact_after_documented_latent_exclusion"
                    if documented_missing
                    else "exact"
                ),
                "documented_missing_latent_count": len(documented_missing),
                "documented_missing_latent_record_ids": sorted(documented_missing),
                "maximum_missing_latent_fraction": alignment.get(
                    "maximum_missing_latent_fraction"
                ),
                "observed_target_comparisons": observed_comparisons,
                "target_opportunities": opportunities,
                "observed_target_coverage": observation_coverage,
                "b_fit_mae": point,
                "b_fit_mae_ci_95": [lower, upper],
                "true_artifact_path": str(true_path),
                "true_artifact_sha256": sha256_file(true_path),
                "prediction_artifact_path": str(pred_path),
                "prediction_artifact_sha256": sha256_file(pred_path),
            }
            bootstrap_rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset.lower(),
                    "split": split,
                    "metric": "b_fit_absolute_error_mean",
                    "point_estimate": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "notes": (
                        "Computed from exact-ID-aligned B_fit and B_hat_fit rows; "
                        "95% interval resamples record clusters."
                    ),
                }
            )
        transition_metrics["datasets"][dataset.lower()] = dataset_metrics
    sections["Known Limitations"].extend(
        [
            "Classifier performance metrics are reported only when row-level prediction artifacts are present.",
            "DSS outputs remain research decision-support artifacts and are not standalone diagnostic outputs.",
        ]
    )
    metrics_path = write_metrics_markdown(config.paths.reports / "metrics" / "metrics_report.md", sections)
    bootstrap_path = write_bootstrap_report(
        config.paths.reports / "metrics" / "bootstrap_ci_report.csv",
        bootstrap_rows
        or [
            {
                "experiment": experiment,
                "dataset": "",
                "split": "",
                "metric": "artifact_missing",
                "point_estimate": "",
                "ci_lower": "",
                "ci_upper": "",
                "notes": "No aligned transition evaluation artifacts were found.",
            }
        ],
    )
    transition_metrics_path = config.paths.reports / "metrics" / "transition_validation_metrics.json"
    write_json(transition_metrics_path, transition_metrics)
    write_stage_manifest(
        config,
        f"report_{experiment}",
        {
            "experiment": experiment,
            "status": "report_templates_written",
            "metrics_report": str(metrics_path),
            "bootstrap_report": str(bootstrap_path),
            "transition_metrics": str(transition_metrics_path),
            "transition_metrics_sha256": sha256_file(transition_metrics_path),
        },
    )
    print(f"Report templates written for {experiment}")
    return 0
