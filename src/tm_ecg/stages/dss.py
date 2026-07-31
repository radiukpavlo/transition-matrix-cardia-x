"""Build production-rule DSS artifacts from matrix B and transition outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import sha256_file, stable_hash
from tm_ecg.io.readers import find_table, read_table_frame
from tm_ecg.reporting.dss_report import write_dss_markdown_report
from tm_ecg.reporting.dss_report import _condition_text
from tm_ecg.dss.discretization import fit_wedd_discretization
from tm_ecg.constants import B_COLUMNS, PROJECT_LABELS
from tm_ecg.dss.eligibility import (
    EligibilityCertificateError,
    attach_rulebook_eligibility_certificate,
    verify_rulebook_eligibility_certificate,
)
from tm_ecg.features.signatures import load_signature_artifact
from tm_ecg.dss.predicates import default_medical_predicates, predicates_to_dict
from tm_ecg.dss.rules import (
    build_decision_system,
    build_predicate_template_rules,
    induce_rules,
    infer_with_rules,
    postprocess_rules,
)
from tm_ecg.dss.selection import (
    ClassMetric,
    SelectionPolicy,
    select_b_features,
    select_well_classified,
)

if TYPE_CHECKING:
    import pandas as pd




def _load_labels(config: ProjectConfig, dataset: str, frame: "pd.DataFrame") -> list[str]:
    if "labels" in frame.columns:
        return [
            next(
                (
                    item.strip()
                    for item in str(value or "").replace(",", "|").split("|")
                    if item.strip() in PROJECT_LABELS
                ),
                "Other / unmapped",
            )
            for value in frame["labels"].tolist()
        ]

    manifest_stem = "ptbxl_split_index" if dataset == "b1" else "ludb_split_index_repeat_1"
    manifest_path = find_table(config.paths.manifests, manifest_stem)
    if manifest_path is None:
        return ["Other / unmapped"] * len(frame)
    manifest = read_table_frame(manifest_path)
    if dataset == "b1":
        manifest = manifest[manifest["split"].astype(str) == "train"]
    else:
        manifest = manifest[manifest["split"].astype(str).str.contains("train", na=False)]
    label_by_record = {
        str(row.record_id): str(row.labels).split(",")[0].strip()
        for row in manifest.itertuples(index=False)
        if str(row.labels).strip()
    }
    return [label_by_record.get(str(record_id), "Other / unmapped") for record_id in frame["record_id"]]


def _wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    """Return a two-sided 95% Wilson lower confidence bound."""

    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = proportion + (z * z / (2.0 * total))
    radius = z * sqrt(
        (proportion * (1.0 - proportion) / total)
        + (z * z / (4.0 * total * total))
    )
    return max(0.0, (centre - radius) / denominator)


def _conservative_metric_lower_bound(tp: int, fp: int, tn: int, fn: int) -> float:
    """Conservatively bound one-vs-rest precision, recall, and specificity."""

    return min(
        _wilson_lower_bound(tp, tp + fp),
        _wilson_lower_bound(tp, tp + fn),
        _wilson_lower_bound(tn, tn + fp),
    )


def _evidence_value(
    payload: Mapping[str, Any],
    context: Mapping[str, Any],
    *keys: str,
) -> Any:
    """Read an evidence field from class-level or artifact-level metadata."""

    sources: list[Mapping[str, Any]] = [payload]
    for container in ("eligibility_evidence", "evaluation", "calibration", "split_audit"):
        nested = payload.get(container)
        if isinstance(nested, Mapping):
            sources.append(nested)
    sources.append(context)
    for container in ("eligibility_evidence", "evaluation", "calibration", "split_audit"):
        nested = context.get(container)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _compatibility_set(value: object) -> frozenset[str]:
    """Parse a compatibility label cell into the locked project vocabulary."""

    return frozenset(
        token.strip()
        for token in str(value or "")
        .replace(",", "|")
        .replace(";", "|")
        .split("|")
        if token.strip() in PROJECT_LABELS
    )


def _canonical_compatibility_labels(value: object) -> str:
    labels = _compatibility_set(value)
    return "|".join(label for label in PROJECT_LABELS if label in labels) or "Other / unmapped"


def _constant_column(frame: "pd.DataFrame", candidates: list[str]) -> tuple[Any, bool]:
    """Return a sole non-null value and whether the column was internally consistent."""

    column = _first_existing_column(frame, candidates)
    if column is None:
        return None, True
    values = frame[column].dropna().unique().tolist()
    if not values:
        return None, True
    return (values[0], len(values) == 1)


def _local_split_evidence(config: ProjectConfig, dataset: str) -> dict[str, Any]:
    """Load independently verifiable split facts for eligibility auditing."""

    manifest_stem = (
        "ptbxl_split_index" if dataset == "b1" else "ludb_split_index_repeat_1"
    )
    manifests_path = getattr(config.paths, "manifests", None)
    if manifests_path is None:
        return {"available": False}
    path = find_table(manifests_path, manifest_stem)
    if path is None:
        return {"available": False}
    frame = read_table_frame(path)
    if dataset != "b1":
        # LUDB uses repeated nested folds. A fold identifier must be added to the
        # model evidence before a single train/evaluation partition can be proven.
        return {
            "available": True,
            "verifiable": False,
            "manifest_path": str(path),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    train_rows = frame[frame["split"].astype(str) == "train"]
    evaluation_rows = frame[frame["split"].astype(str) == "test"]
    evaluation_ids = [str(value) for value in evaluation_rows["record_id"].tolist()]
    label_by_record = {
        str(row.record_id): _canonical_compatibility_labels(row.labels)
        for row in evaluation_rows.itertuples(index=False)
    }
    class_support = {
        label: sum(
            label in _compatibility_set(value)
            for value in evaluation_rows["labels"].tolist()
        )
        for label in PROJECT_LABELS
    }
    ontology_versions = (
        sorted(
            {
                str(value)
                for value in frame["ontology_version"].dropna().unique().tolist()
            }
        )
        if "ontology_version" in frame.columns
        else []
    )

    def patient_keys(partition: "pd.DataFrame") -> set[str]:
        keys: set[str] = set()
        for row in partition.itertuples(index=False):
            patient_id = getattr(row, "patient_id", None)
            normalized = str(patient_id).strip().lower()
            if normalized in {"", "none", "nan"}:
                normalized = f"record:{getattr(row, 'record_id')}"
            keys.add(normalized)
        return keys

    overlap = patient_keys(train_rows) & patient_keys(evaluation_rows)
    return {
        "available": True,
        "verifiable": True,
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evaluation_record_ids": set(evaluation_ids),
        "evaluation_record_ids_unique": len(evaluation_ids) == len(set(evaluation_ids)),
        "evaluation_label_by_record": label_by_record,
        "evaluation_class_support": class_support,
        "ontology_versions": ontology_versions,
        "evaluation_records": len(evaluation_rows),
        "patient_disjoint": not overlap,
        "patient_overlap_count": len(overlap),
    }




def _metric_from_mapping(
    label: str,
    payload: dict[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    evidence_source: str | None = None,
) -> ClassMetric:
    artifact = context or {}
    support = int(payload.get("support", payload.get("n", payload.get("count", 0))) or 0)
    precision = float(payload.get("precision", payload.get("ppv", 0.0)) or 0.0)
    recall = float(payload.get("recall", payload.get("sensitivity", payload.get("se", 0.0))) or 0.0)
    specificity = float(payload.get("specificity", payload.get("sp", 0.0)) or 0.0)
    f1 = float(payload.get("f1", payload.get("f1_score", payload.get("f1-score", 0.0))) or 0.0)
    tp = int(payload.get("tp", payload.get("true_positive", 0)) or 0)
    fp = int(payload.get("fp", payload.get("false_positive", 0)) or 0)
    tn = int(payload.get("tn", payload.get("true_negative", 0)) or 0)
    fn = int(payload.get("fn", payload.get("false_negative", 0)) or 0)
    mean_confidence = _evidence_value(payload, artifact, "mean_confidence", "confidence")
    metric_lower_bound = _evidence_value(
        payload,
        artifact,
        "metric_lower_bound",
        "f1_lower_bound",
        "ci_lower_bound",
    )
    confidence_level = _evidence_value(payload, artifact, "confidence_level")
    confidence_interval_method = _evidence_value(
        payload,
        artifact,
        "confidence_interval_method",
        "ci_method",
    )
    if metric_lower_bound is None:
        metric_lower_bound = _conservative_metric_lower_bound(tp, fp, tn, fn)
        confidence_level = 0.95
        confidence_interval_method = "minimum_one_vs_rest_wilson_score"
    analyzable_coverage = _evidence_value(
        payload,
        artifact,
        "analyzable_coverage",
        "coverage",
    )
    abstention_rate = _evidence_value(payload, artifact, "abstention_rate")
    fold_stability = _evidence_value(
        payload,
        artifact,
        "fold_stability",
        "bootstrap_stability",
    )
    executable_predicate = _evidence_value(payload, artifact, "executable_predicate")
    calibration_error = _evidence_value(
        payload,
        artifact,
        "calibration_error",
        "expected_calibration_error",
        "ece",
    )
    probabilities_calibrated = _optional_bool(
        _evidence_value(
            payload,
            artifact,
            "probabilities_calibrated",
            "calibrated_probabilities",
            "calibrated",
        )
    )
    evaluation_partition = _evidence_value(
        payload,
        artifact,
        "evaluation_partition",
        "evaluation_split",
    )
    threshold_partition = _evidence_value(
        payload,
        artifact,
        "threshold_partition",
        "threshold_selection_partition",
    )
    patient_disjoint = _optional_bool(
        _evidence_value(
            payload,
            artifact,
            "patient_disjoint",
            "patient_disjoint_split",
        )
    )
    overlap_count = _evidence_value(
        payload,
        artifact,
        "patient_overlap_count",
        "train_evaluation_patient_overlap",
    )
    if patient_disjoint is None and overlap_count is not None:
        patient_disjoint = int(overlap_count) == 0
    split_integrity_verified = _optional_bool(
        _evidence_value(payload, artifact, "split_integrity_verified")
    )
    split_manifest_sha256 = _evidence_value(
        payload,
        artifact,
        "split_manifest_sha256",
        "manifest_sha256",
    )
    evaluation_records = _evidence_value(
        payload,
        artifact,
        "evaluation_records",
        "test_records",
    )
    global_accuracy = _evidence_value(
        payload,
        artifact,
        "global_accuracy",
        "compatibility_accuracy",
        "classification_accuracy",
        "accuracy",
    )
    global_accuracy_definition = _evidence_value(
        payload,
        artifact,
        "global_accuracy_definition",
        "accuracy_definition",
    )
    metric_provenance = _evidence_value(
        payload,
        artifact,
        "metric_provenance",
        "evaluation_task",
        "prediction_provenance",
    )
    evidence_issues = [
        str(issue) for issue in artifact.get("_split_evidence_issues", [])
    ]
    expected_manifest_sha256 = artifact.get("_expected_split_manifest_sha256")
    expected_evaluation_records = artifact.get("_expected_evaluation_records")
    verified_patient_disjoint = artifact.get("_verified_patient_disjoint")
    expected_class_support = artifact.get("_expected_class_support")
    if expected_manifest_sha256 is not None:
        if split_manifest_sha256 is None:
            evidence_issues.append("missing_split_manifest_sha256")
        elif str(split_manifest_sha256) != str(expected_manifest_sha256):
            evidence_issues.append("split_manifest_sha256_mismatch")
        if evaluation_records is None:
            evidence_issues.append("missing_evaluation_record_count")
        elif int(evaluation_records) != int(expected_evaluation_records):
            evidence_issues.append("evaluation_record_count_manifest_mismatch")
        patient_disjoint = bool(verified_patient_disjoint)
        split_integrity_verified = not any(
            issue
            in {
                "missing_split_manifest_sha256",
                "split_manifest_sha256_mismatch",
                "missing_evaluation_record_count",
                "evaluation_record_count_manifest_mismatch",
            }
            for issue in evidence_issues
        ) and patient_disjoint
    if isinstance(expected_class_support, Mapping) and label in expected_class_support:
        expected_support = int(expected_class_support[label])
        if support != expected_support or tp + fn != expected_support:
            evidence_issues.append("class_support_manifest_mismatch")
    if str(payload.get("status", "ok")).lower() not in {"ok", "valid", "complete"}:
        evidence_issues.append(f"class_status={payload.get('status')}")
    return ClassMetric(
        label=label,
        support=support,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        mean_confidence=float(mean_confidence) if mean_confidence is not None else None,
        metric_lower_bound=(
            float(metric_lower_bound) if metric_lower_bound is not None else None
        ),
        analyzable_coverage=(
            float(analyzable_coverage) if analyzable_coverage is not None else None
        ),
        abstention_rate=(float(abstention_rate) if abstention_rate is not None else None),
        fold_stability=float(fold_stability) if fold_stability is not None else None,
        executable_predicate=_optional_bool(executable_predicate),
        calibration_error=(
            float(calibration_error) if calibration_error is not None else None
        ),
        probabilities_calibrated=probabilities_calibrated,
        evaluation_partition=(
            str(evaluation_partition) if evaluation_partition is not None else None
        ),
        threshold_partition=(
            str(threshold_partition) if threshold_partition is not None else None
        ),
        patient_disjoint=patient_disjoint,
        split_integrity_verified=split_integrity_verified,
        split_manifest_sha256=(
            str(split_manifest_sha256) if split_manifest_sha256 is not None else None
        ),
        evaluation_records=(
            int(evaluation_records) if evaluation_records is not None else None
        ),
        global_accuracy=(
            float(global_accuracy) if global_accuracy is not None else None
        ),
        global_accuracy_definition=(
            str(global_accuracy_definition)
            if global_accuracy_definition is not None
            else None
        ),
        metric_provenance=(
            str(metric_provenance) if metric_provenance is not None else None
        ),
        confidence_level=(
            float(confidence_level) if confidence_level is not None else None
        ),
        confidence_interval_method=(
            str(confidence_interval_method)
            if confidence_interval_method is not None
            else None
        ),
        evidence_source=evidence_source,
        evidence_valid=not evidence_issues,
        evidence_issues=evidence_issues,
    )


def _load_per_class_metrics(config: ProjectConfig, dataset: str) -> dict[str, ClassMetric] | None:
    split_evidence = _local_split_evidence(config, dataset)
    candidate_names = [
        f"{dataset}_classification_metrics.json",
        f"{dataset}_per_class_metrics.json",
        "ptbxl_training_metrics.json" if dataset == "b1" else "ludb_training_metrics.json",
    ]
    for name in candidate_names:
        for path in (config.paths.reports / name, config.paths.reports / "metrics" / name):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            context = dict(payload)
            context_issues = [
                str(issue) for issue in context.get("_split_evidence_issues", [])
            ]
            if split_evidence.get("verifiable"):
                context["_expected_split_manifest_sha256"] = split_evidence[
                    "manifest_sha256"
                ]
                context["_expected_evaluation_records"] = split_evidence[
                    "evaluation_records"
                ]
                context["_verified_patient_disjoint"] = split_evidence[
                    "patient_disjoint"
                ]
                context["_expected_class_support"] = split_evidence[
                    "evaluation_class_support"
                ]
                if not split_evidence.get("evaluation_record_ids_unique", False):
                    context_issues.append("duplicate_evaluation_record_id_in_manifest")
            elif split_evidence.get("available"):
                context_issues.append("nested_fold_partition_not_identified")
            active_ontology = getattr(config, "ontology_version", None)
            if active_ontology is not None:
                artifact_ontology = payload.get("ontology_version")
                if artifact_ontology is None:
                    context_issues.append("missing_ontology_version")
                elif str(artifact_ontology) != str(active_ontology):
                    context_issues.append("ontology_version_mismatch")
                manifest_versions = split_evidence.get("ontology_versions", [])
                if manifest_versions != [str(active_ontology)]:
                    context_issues.append("manifest_ontology_version_mismatch")
            if payload.get("metric_provenance") == "held_out_compatibility_head":
                for artifact_key, hash_key in (
                    ("model_path", "model_sha256"),
                    ("predictions_path", "predictions_sha256"),
                    ("validation_lock_path", "validation_lock_sha256"),
                ):
                    artifact_path = payload.get(artifact_key)
                    artifact_hash = payload.get(hash_key)
                    if not artifact_path or not artifact_hash:
                        context_issues.append(f"missing_{artifact_key}_integrity_evidence")
                        continue
                    resolved = Path(str(artifact_path))
                    if not resolved.exists():
                        context_issues.append(f"missing_{artifact_key}")
                    elif hashlib.sha256(resolved.read_bytes()).hexdigest() != str(artifact_hash):
                        context_issues.append(f"{artifact_key}_sha256_mismatch")
                provenance = payload.get("feature_provenance")
                record_hashes = (
                    provenance.get("record_id_hashes")
                    if isinstance(provenance, Mapping)
                    else None
                )
                if not isinstance(record_hashes, Mapping) or not all(
                    split in record_hashes for split in ("train", "val", "test")
                ):
                    context_issues.append("missing_partition_record_id_hashes")
                supports = payload.get("target_support_by_split")
                if not isinstance(supports, Mapping) or "test" not in supports:
                    context_issues.append("missing_target_support_by_split")
            context["_split_evidence_issues"] = list(dict.fromkeys(context_issues))
            per_class = (
                payload.get("per_class_metrics")
                or payload.get("class_metrics")
                or payload.get("per_label")
                or payload.get("labels")
            )
            if isinstance(per_class, dict) and per_class:
                return {
                    str(label): _metric_from_mapping(
                        str(label),
                        dict(values),
                        context=context,
                        evidence_source=str(path),
                    )
                    for label, values in per_class.items()
                }
            if isinstance(per_class, list) and per_class:
                metrics: dict[str, ClassMetric] = {}
                for item in per_class:
                    if isinstance(item, dict) and "label" in item:
                        label = str(item["label"])
                        metrics[label] = _metric_from_mapping(
                            label,
                            item,
                            context=context,
                            evidence_source=str(path),
                        )
                if metrics:
                    return metrics
    return None


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _load_prediction_metrics(
    path: Path,
    *,
    true_label_col: str = "true_label",
    pred_label_col: str = "pred_label",
    confidence_col: str = "confidence",
    prediction_id_col: str = "record_id",
    label_by_record: dict[str, str] | None = None,
    expected_evaluation_ids: set[str] | None = None,
    verified_patient_disjoint: bool | None = None,
    split_manifest_sha256: str | None = None,
) -> dict[str, ClassMetric]:
    """Load row-level deep-model predictions and compute Step-1 class metrics.

    Strict eligibility metadata are intentionally read from the audit table rather
    than inferred from its filename.  Missing or internally inconsistent provenance
    is retained as an evidence issue and therefore fails closed during selection.
    """

    frame = read_table_frame(path)
    true_col = _first_existing_column(frame, [true_label_col, "true_label", "y_true", "label", "target", "diagnosis"])
    pred_col = _first_existing_column(
        frame, [pred_label_col, "predicted_label", "y_pred", "prediction", "pred_label", "pred"]
    )
    conf_col = _first_existing_column(
        frame, [confidence_col, "confidence", "probability", "proba", "max_probability", "score"]
    )
    id_col = _first_existing_column(frame, [prediction_id_col, "record_id", "row_id", "object_id"])
    if pred_col is None:
        raise ValueError(
            f"Prediction audit table {path} must contain a predicted-label column; "
            f"found columns={list(frame.columns)}"
        )
    evidence_issues: list[str] = []
    if id_col is not None and label_by_record:
        record_ids = frame[id_col].astype(str).tolist()
        missing_truth_ids = [value for value in record_ids if value not in label_by_record]
        if missing_truth_ids:
            evidence_issues.append("prediction_ids_missing_locked_manifest_truth")
        y_true = [
            label_by_record.get(value, "Other / unmapped") for value in record_ids
        ]
        if true_col is not None:
            supplied = [_compatibility_set(value) for value in frame[true_col].tolist()]
            authoritative = [_compatibility_set(value) for value in y_true]
            if supplied != authoritative:
                evidence_issues.append("prediction_true_label_manifest_mismatch")
    elif true_col is not None:
        y_true = [str(value) for value in frame[true_col].tolist()]
    else:
        raise ValueError(
            f"Prediction audit table {path} must contain a true-label column or an ID column "
            "that aligns with known labels"
        )
    abstention_tokens = {
        "",
        "abstain",
        "abstained",
        "defer",
        "deferred",
        "no decision",
        "none",
        "nan",
        "__abstain__",
    }
    y_pred = [
        "__ABSTAIN__"
        if str(value).strip().lower() in abstention_tokens
        else str(value)
        for value in frame[pred_col].tolist()
    ]
    confidences = None
    if conf_col is not None:
        confidences = [float(value) for value in frame[conf_col].fillna(0.0).tolist()]
    else:
        evidence_issues.append("missing_row_confidence")
    if id_col is None:
        evidence_issues.append("missing_prediction_record_id")
    elif frame[id_col].astype(str).duplicated().any():
        evidence_issues.append("duplicate_prediction_record_id")
    observed_ids = (
        set(frame[id_col].astype(str).tolist()) if id_col is not None else set()
    )
    split_integrity_verified: bool | None = None
    if expected_evaluation_ids is not None:
        split_integrity_verified = observed_ids == expected_evaluation_ids
        if observed_ids - expected_evaluation_ids:
            evidence_issues.append("prediction_ids_outside_evaluation_split")
        if expected_evaluation_ids - observed_ids:
            evidence_issues.append("evaluation_split_predictions_missing")

    evaluation_partition, evaluation_consistent = _constant_column(
        frame,
        ["evaluation_partition", "evaluation_split", "partition"],
    )
    threshold_partition, threshold_consistent = _constant_column(
        frame,
        ["threshold_partition", "threshold_selection_partition"],
    )
    patient_disjoint_value, patient_disjoint_consistent = _constant_column(
        frame,
        ["patient_disjoint", "patient_disjoint_split"],
    )
    calibrated_value, calibrated_consistent = _constant_column(
        frame,
        ["probabilities_calibrated", "calibrated_probabilities", "calibrated"],
    )
    calibration_error, calibration_error_consistent = _constant_column(
        frame,
        ["calibration_error", "expected_calibration_error", "ece"],
    )
    fold_stability, fold_stability_consistent = _constant_column(
        frame,
        ["fold_stability", "bootstrap_stability"],
    )
    for consistent, issue in (
        (evaluation_consistent, "mixed_evaluation_partitions"),
        (threshold_consistent, "mixed_threshold_partitions"),
        (patient_disjoint_consistent, "inconsistent_patient_disjointness"),
        (calibrated_consistent, "inconsistent_calibration_status"),
        (calibration_error_consistent, "inconsistent_calibration_error"),
        (fold_stability_consistent, "inconsistent_fold_stability"),
    ):
        if not consistent:
            evidence_issues.append(issue)

    true_sets = [_compatibility_set(value) for value in y_true]
    predicted_sets = [
        frozenset() if value == "__ABSTAIN__" else _compatibility_set(value)
        for value in y_pred
    ]
    analyzable = sum(bool(prediction) for prediction in predicted_sets)
    analyzable_coverage = analyzable / len(y_pred) if y_pred else 0.0
    abstention_rate = 1.0 - analyzable_coverage
    global_accuracy = (
        sum(
            truth == prediction
            for truth, prediction in zip(true_sets, predicted_sets, strict=True)
        )
        / len(y_true)
        if y_true
        else 0.0
    )
    labels = sorted(
        set(PROJECT_LABELS)
        | {label for values in true_sets for label in values}
        | {label for values in predicted_sets for label in values}
    )
    metrics: dict[str, ClassMetric] = {}
    for label in labels:
        tp = sum(
            label in truth and label in prediction
            for truth, prediction in zip(true_sets, predicted_sets, strict=True)
        )
        fp = sum(
            label not in truth and label in prediction
            for truth, prediction in zip(true_sets, predicted_sets, strict=True)
        )
        fn = sum(
            label in truth and label not in prediction
            for truth, prediction in zip(true_sets, predicted_sets, strict=True)
        )
        tn = len(true_sets) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_confidences = (
            [
                confidence
                for confidence, truth in zip(confidences, true_sets, strict=True)
                if label in truth
            ]
            if confidences is not None
            else []
        )
        metrics[label] = ClassMetric(
            label=label,
            support=tp + fn,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1=f1,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            mean_confidence=(
                sum(class_confidences) / len(class_confidences)
                if class_confidences
                else None
            ),
        )
    for metric in metrics.values():
        metric.metric_lower_bound = _conservative_metric_lower_bound(
            metric.tp,
            metric.fp,
            metric.tn,
            metric.fn,
        )
        metric.confidence_level = 0.95
        metric.confidence_interval_method = "minimum_one_vs_rest_wilson_score"
        metric.analyzable_coverage = analyzable_coverage
        metric.abstention_rate = abstention_rate
        metric.fold_stability = (
            float(fold_stability) if fold_stability is not None else None
        )
        metric.calibration_error = (
            float(calibration_error) if calibration_error is not None else None
        )
        metric.probabilities_calibrated = _optional_bool(calibrated_value)
        metric.evaluation_partition = (
            str(evaluation_partition) if evaluation_partition is not None else None
        )
        metric.threshold_partition = (
            str(threshold_partition) if threshold_partition is not None else None
        )
        metric.patient_disjoint = _optional_bool(patient_disjoint_value)
        if verified_patient_disjoint is not None:
            metric.patient_disjoint = verified_patient_disjoint
        metric.split_integrity_verified = split_integrity_verified
        metric.split_manifest_sha256 = split_manifest_sha256
        metric.evaluation_records = len(frame)
        metric.global_accuracy = global_accuracy
        metric.global_accuracy_definition = "compatibility_subset_exact_match"
        metric.metric_provenance = "row_level_compatibility_predictions"
        metric.evidence_source = str(path)
        metric.evidence_valid = not evidence_issues
        metric.evidence_issues = list(evidence_issues)
    return metrics


_QUALITY_GUARD_FEATURES = {
    "lead_quality_min_db",
    "delineation_confidence",
    "analyzable_duration_s",
}


def _predicate_executability(
    predicates: Mapping[str, Any],
    available_features: set[str],
) -> dict[str, bool]:
    """Verify predicates against the actual B table; never trust metric metadata."""

    executable: dict[str, bool] = {}
    for label, predicate in predicates.items():
        required_features = {condition.feature for condition in predicate.required}
        executable[str(label)] = bool(
            predicate.required
            and not predicate.weak_signature
            and required_features <= available_features
            and _QUALITY_GUARD_FEATURES <= required_features
        )
    return executable


def _selection_audit(
    config: ProjectConfig,
    dataset: str,
    labels: list[str],
    predicate_executability: Mapping[str, bool],
    policy: SelectionPolicy,
    predictions_path: Path | None = None,
    prediction_columns: dict[str, str] | None = None,
    label_by_record: dict[str, str] | None = None,
    allow_research_fallback: bool = False,
) -> tuple[set[str], dict[str, Any]]:
    """Implement Step 1 audit for the DSS selection logic.

    The preferred path uses row-level deep-model predictions supplied by
    ``--predictions``.  If they are absent, the function attempts to read
    per-class metrics from ``artifacts/reports/``. Missing model evidence always fails
    closed. A legacy fallback request is recorded but can never create an
    eligible production class.
    """

    metrics: dict[str, ClassMetric] | None = None
    provenance = ""
    if predictions_path is not None:
        cols = prediction_columns or {}
        split_evidence = _local_split_evidence(config, dataset)
        authoritative_labels = label_by_record or split_evidence.get(
            "evaluation_label_by_record"
        )
        metrics = _load_prediction_metrics(
            predictions_path,
            true_label_col=cols.get("true_label_col", "true_label"),
            pred_label_col=cols.get("pred_label_col", "pred_label"),
            confidence_col=cols.get("confidence_col", "confidence"),
            prediction_id_col=cols.get("prediction_id_col", "record_id"),
            label_by_record=authoritative_labels,
            expected_evaluation_ids=(
                set(split_evidence["evaluation_record_ids"])
                if split_evidence.get("verifiable")
                else None
            ),
            verified_patient_disjoint=(
                bool(split_evidence["patient_disjoint"])
                if split_evidence.get("verifiable")
                else None
            ),
            split_manifest_sha256=(
                str(split_evidence["manifest_sha256"])
                if split_evidence.get("verifiable")
                else None
            ),
        )
        provenance = f"row_level_prediction_audit:{predictions_path}"
    else:
        metrics = _load_per_class_metrics(config, dataset)
        if metrics:
            provenance = "per_class_dl_metrics"

    if metrics:
        for label, metric in metrics.items():
            metric.executable_predicate = bool(
                predicate_executability.get(label, False)
            )
        selected, audited = select_well_classified(metrics, policy)
        return selected, {
            "provenance": provenance,
            "policy": policy.to_dict(),
            "well_classified_labels": sorted(selected),
            "predicate_executability": dict(
                sorted(predicate_executability.items())
            ),
            "research_fallback_requested": allow_research_fallback,
            "research_fallback_applied": False,
            "class_metrics": {label: metric.to_dict() for label, metric in audited.items()},
            "caution": (
                "Eligibility requires complete held-out, patient-disjoint, calibrated, "
                "confidence-bounded, coverage, abstention, stability, and executable-predicate evidence."
            ),
        }

    distribution = Counter(labels)
    return set(), {
        "status": "failed_no_model_metrics",
        "provenance": "none",
        "policy": policy.to_dict(),
        "well_classified_labels": [],
        "predicate_executability": dict(sorted(predicate_executability.items())),
        "label_support": dict(sorted(distribution.items())),
        "research_fallback_requested": allow_research_fallback,
        "research_fallback_applied": False,
        "caution": (
            "Strict selection refused to infer model quality from label support. "
            "No label-support or all-feature fallback is permitted; supply a complete "
            "held-out eligibility artifact."
        ),
    }


def _source_distribution(rules: list[Any]) -> dict[str, int]:
    counts = Counter(str(rule.source) for rule in rules)
    return dict(sorted(counts.items()))


def _load_transition_context(
    config: ProjectConfig, dataset: str
) -> tuple[list[list[float]] | None, list[str], set[str]]:
    """Load transition operator and its B-column order when available."""

    metadata_path = config.paths.transition / f"{dataset.upper()}_operator_metadata.json"
    json_operator_path = config.paths.transition / f"{dataset.upper()}_T_ridge.json"
    transform_path = config.paths.transition / f"{dataset.upper()}_transform_bundle.json"
    operator = None
    fit_columns: list[str] = []
    stable_features: set[str] = set()
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("artifact_version") != 2:
            raise RuntimeError("Transition metadata is missing the strict provenance contract")
        if payload.get("ontology_version") != config.ontology_version:
            raise RuntimeError("Transition metadata ontology does not match the active ontology")
        manifest_stem = (
            "ptbxl_split_index" if dataset == "b1" else "ludb_split_index_repeat_1"
        )
        active_manifest = find_table(config.paths.manifests, manifest_stem)
        if (
            active_manifest is None
            or payload.get("split_manifest_sha256") != sha256_file(active_manifest)
        ):
            raise RuntimeError("Transition split-manifest provenance is stale")
        for artifact_group in ("raw_input_artifacts", "latent_input_artifacts"):
            evidence = dict(payload.get(artifact_group, {}))
            if not evidence:
                raise RuntimeError(f"Transition metadata lacks {artifact_group}")
            for split, raw_evidence in evidence.items():
                artifact_evidence = dict(raw_evidence)
                artifact_path = Path(str(artifact_evidence.get("path", "")))
                if (
                    not artifact_path.exists()
                    or artifact_evidence.get("sha256") != sha256_file(artifact_path)
                ):
                    raise RuntimeError(
                        f"Transition {artifact_group} hash mismatch for {split}"
                    )
        if payload.get("selected_representation") == "direct_b_feature_baseline":
            return None, [], set()
        fit_columns = [str(column) for column in payload.get("b_fit_columns", [])]
        if payload.get("b_fit_columns_hash") != stable_hash(fit_columns):
            raise RuntimeError("Transition B-column schema hash is stale")
        stability = dict(payload.get("transition_feature_stability", {}))
        if stability.get("status") == "ok":
            stable_features = {
                str(column) for column in stability.get("stable_features", [])
            }
        operator_ref = Path(str(payload.get("legacy_operator_path") or json_operator_path))
        if (
            not operator_ref.exists()
            or payload.get("legacy_operator_sha256") != sha256_file(operator_ref)
        ):
            raise RuntimeError("Transition operator hash does not match its metadata")
        transform_ref = Path(str(payload.get("transform_bundle", transform_path)))
        if (
            not transform_ref.exists()
            or payload.get("transform_bundle_sha256") != sha256_file(transform_ref)
        ):
            raise RuntimeError("Transition transform-bundle hash does not match metadata")
        if operator_ref.suffix == ".json":
            operator = json.loads(operator_ref.read_text(encoding="utf-8")).get("operator")
    elif json_operator_path.exists() or transform_path.exists():
        raise RuntimeError("Transition artifacts exist without strict provenance metadata")
    return operator, fit_columns, stable_features


def _rows_from_frame(frame: "pd.DataFrame", selected_features: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in frame[selected_features].to_dict(orient="records"):
        rows.append({str(key): value for key, value in item.items()})
    return rows


def _patient_groups_for_records(
    config: ProjectConfig,
    dataset: str,
    object_ids: list[str],
) -> list[str]:
    """Resolve development-fold groups from the locked split manifest."""

    manifest_stem = (
        "ptbxl_split_index" if dataset == "b1" else "ludb_split_index_repeat_1"
    )
    manifest_path = find_table(config.paths.manifests, manifest_stem)
    if manifest_path is None:
        return [f"record:{record_id}" for record_id in object_ids]
    manifest = read_table_frame(manifest_path)
    patient_by_record: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        record_id = str(row.record_id)
        patient_id = getattr(row, "patient_id", None)
        normalized = str(patient_id).strip().lower()
        patient_by_record[record_id] = (
            f"record:{record_id}"
            if normalized in {"", "none", "nan"}
            else normalized
        )
    return [
        patient_by_record.get(record_id, f"record:{record_id}")
        for record_id in object_ids
    ]


def _cross_validated_rule_metrics(
    rows: list[dict[str, object]],
    labels: list[str],
    object_ids: list[str],
    *,
    features: list[str],
    selected_labels: set[str],
    predicates: dict[str, Any],
    signature_thresholds: dict[str, dict[str, float]],
    config: ProjectConfig,
    args: argparse.Namespace,
    patient_groups: list[str] | None = None,
    folds: int = 5,
) -> dict[str, dict[str, float | int | None]]:
    """Estimate label-level rule precision/recall on held-out development folds."""

    groups = patient_groups or [f"record:{object_id}" for object_id in object_ids]
    if len(groups) != len(object_ids):
        raise ValueError("Patient groups must align with DSS development rows")
    assignments = [
        int(hashlib.sha256(f"{config.seed}:{group}".encode()).hexdigest()[:8], 16)
        % folds
        for group in groups
    ]
    predictions: list[str | None] = [None] * len(rows)
    fold_recalls: dict[str, list[float]] = {label: [] for label in selected_labels}
    for fold in range(folds):
        train_indices = [index for index, value in enumerate(assignments) if value != fold]
        validation_indices = [index for index, value in enumerate(assignments) if value == fold]
        if not train_indices or not validation_indices:
            continue
        plan = fit_wedd_discretization(
            [rows[index] for index in train_indices],
            [labels[index] for index in train_indices],
            features=features,
            alpha=float(args.alpha),
            min_support=int(args.min_support),
            max_depth=int(args.max_depth),
            prefer_clinical_bins=not args.no_clinical_bins,
            signature_thresholds=signature_thresholds,
        )
        system = build_decision_system(
            [rows[index] for index in train_indices],
            [labels[index] for index in train_indices],
            plan,
            object_ids=[object_ids[index] for index in train_indices],
        )
        rules = induce_rules(
            system,
            plan,
            min_support=int(args.min_support),
            use_reducts=not args.no_reducts,
            allowed_labels=selected_labels,
        )
        if not getattr(args, "no_predicate_rules", False):
            rules.extend(
                build_predicate_template_rules(
                    predicates,
                    system,
                    allowed_labels=selected_labels,
                    include_weak_signature=False,
                )
            )
        rules = postprocess_rules(
            rules,
            predicates=predicates,
            max_rules_per_label=(
                int(args.max_rules_per_label) if args.max_rules_per_label else None
            ),
        )
        for index in validation_indices:
            inference = infer_with_rules(
                rows[index],
                rules,
                plan,
                predicates=predicates,
                soft_match_min_fraction=float(args.soft_match_min_fraction),
                close_vote_ratio=float(config.dss.get("close_vote_ratio", 1.10)),
                low_strength_threshold=float(
                    config.dss.get("low_strength_threshold", 0.20)
                ),
                contradiction_penalty=float(
                    config.dss.get("contradiction_penalty", 0.25)
                ),
                hard_veto_criticality=float(
                    config.dss.get("hard_veto_criticality", 1.5)
                ),
            )
            predictions[index] = inference.predicted_label
        for label in selected_labels:
            positives = [index for index in validation_indices if labels[index] == label]
            if positives:
                fold_recalls[label].append(
                    sum(predictions[index] == label for index in positives)
                    / len(positives)
                )

    metrics: dict[str, dict[str, float | int | None]] = {}
    for label in sorted(selected_labels):
        tp = sum(
            truth == label and prediction == label
            for truth, prediction in zip(labels, predictions, strict=True)
        )
        fp = sum(
            truth != label and prediction == label
            for truth, prediction in zip(labels, predictions, strict=True)
        )
        fn = sum(
            truth == label and prediction != label
            for truth, prediction in zip(labels, predictions, strict=True)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recall_values = fold_recalls[label]
        recall_mean = sum(recall_values) / len(recall_values) if recall_values else 0.0
        recall_variance = (
            sum((value - recall_mean) ** 2 for value in recall_values)
            / len(recall_values)
            if recall_values
            else 1.0
        )
        uncertainty = sqrt(precision * (1.0 - precision) / max(tp + fp, 1))
        metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "class_prior": labels.count(label) / max(len(labels), 1),
            "calibration_uncertainty": uncertainty,
            "fold_stability": max(0.0, 1.0 - sqrt(recall_variance)),
            "folds_with_positive_support": len(recall_values),
        }
    return metrics




def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON artifact atomically within its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_evidence_file(
    config: ProjectConfig,
    raw_path: object,
    *,
    field: str,
) -> Path:
    """Resolve one evidence file without guessing between multiple locations."""

    if raw_path is None or not str(raw_path).strip():
        raise EligibilityCertificateError(f"eligibility provenance lacks {field}")
    supplied = Path(str(raw_path))
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates = [config.paths.root / supplied, config.paths.reports / supplied]
    existing = {
        candidate.resolve()
        for candidate in candidates
        if candidate.exists() and candidate.is_file()
    }
    if not existing:
        raise EligibilityCertificateError(
            f"eligibility provenance {field} does not identify an existing file"
        )
    if len(existing) != 1:
        raise EligibilityCertificateError(
            f"eligibility provenance {field} resolves ambiguously"
        )
    resolved = existing.pop()
    try:
        resolved.relative_to(config.paths.root.resolve())
    except ValueError as exc:
        raise EligibilityCertificateError(
            f"eligibility provenance {field} is outside the project root"
        ) from exc
    return resolved


def _verified_eligibility_provenance(
    config: ProjectConfig,
    dataset: str,
    rulebook: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Path]]:
    """Verify and hash the exact split, model, and metrics evidence files.

    Eligibility selection proves metric values, while this final producer gate
    binds the executable rulebook to the files that supplied those values.  The
    gate accepts only one common metrics artifact for all selected/rule targets
    and independently re-hashes every referenced file.
    """

    if rulebook.get("status") != "eligible" or rulebook.get("rules_eligible") is not True:
        raise EligibilityCertificateError(
            "only an explicitly eligible rulebook can receive a certificate"
        )
    production_rules = rulebook.get("production_rules")
    if not isinstance(production_rules, list) or not production_rules:
        raise EligibilityCertificateError(
            "an empty rulebook cannot receive an eligibility certificate"
        )
    rule_targets = {
        str(rule.get("target_label"))
        for rule in production_rules
        if isinstance(rule, Mapping) and rule.get("target_label")
    }
    if not rule_targets:
        raise EligibilityCertificateError(
            "eligible rules lack auditable target labels"
        )

    selection_audit = rulebook.get("selection_audit")
    if not isinstance(selection_audit, Mapping):
        raise EligibilityCertificateError("rulebook lacks a selection audit")
    selected_labels = selection_audit.get("well_classified_labels")
    if not isinstance(selected_labels, list) or not selected_labels:
        raise EligibilityCertificateError(
            "selection audit lacks eligible compatibility classes"
        )
    selected = {str(label) for label in selected_labels}
    if not rule_targets <= selected:
        raise EligibilityCertificateError(
            "rule targets are not a subset of the selected compatibility classes"
        )
    class_metrics = selection_audit.get("class_metrics")
    if not isinstance(class_metrics, Mapping):
        raise EligibilityCertificateError("selection audit lacks class metrics")

    evidence_sources: set[str] = set()
    selected_split_hashes: set[str] = set()
    audited_metrics: dict[str, Mapping[str, Any]] = {}
    for label in sorted(rule_targets):
        metric = class_metrics.get(label)
        if not isinstance(metric, Mapping):
            raise EligibilityCertificateError(
                f"selection audit lacks metrics for eligible rule target {label}"
            )
        if metric.get("selected") is not True or metric.get("evidence_valid") is not True:
            raise EligibilityCertificateError(
                f"eligibility evidence for {label} is not selected and valid"
            )
        issues = metric.get("evidence_issues")
        if issues not in (None, []):
            raise EligibilityCertificateError(
                f"eligibility evidence for {label} contains unresolved issues"
            )
        source = metric.get("evidence_source")
        split_hash = metric.get("split_manifest_sha256")
        if not source or not split_hash:
            raise EligibilityCertificateError(
                f"eligibility evidence for {label} lacks source or split hash"
            )
        evidence_sources.add(str(source))
        selected_split_hashes.add(str(split_hash))
        audited_metrics[label] = metric
    if len(evidence_sources) != 1:
        raise EligibilityCertificateError(
            "eligible rule targets do not share one immutable metrics artifact"
        )
    if len(selected_split_hashes) != 1:
        raise EligibilityCertificateError(
            "eligible rule targets disagree on the split-manifest hash"
        )

    metrics_path = _resolve_evidence_file(
        config,
        next(iter(evidence_sources)),
        field="metrics_path",
    )
    try:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EligibilityCertificateError(
            "eligible metrics evidence must be a readable JSON artifact"
        ) from exc
    if not isinstance(metrics_payload, Mapping):
        raise EligibilityCertificateError(
            "eligible metrics evidence must be a JSON object"
        )
    if metrics_payload.get("ontology_version") != config.ontology_version:
        raise EligibilityCertificateError(
            "metrics artifact ontology does not match the active ontology"
        )

    per_class_payload = (
        metrics_payload.get("per_class_metrics")
        or metrics_payload.get("class_metrics")
        or metrics_payload.get("per_label")
        or metrics_payload.get("labels")
    )
    source_metrics: dict[str, Mapping[str, Any]] = {}
    if isinstance(per_class_payload, Mapping):
        source_metrics = {
            str(label): values
            for label, values in per_class_payload.items()
            if isinstance(values, Mapping)
        }
    elif isinstance(per_class_payload, list):
        source_metrics = {
            str(values["label"]): values
            for values in per_class_payload
            if isinstance(values, Mapping) and values.get("label")
        }
    metric_binding_fields = (
        "support",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "mean_confidence",
        "metric_lower_bound",
        "analyzable_coverage",
        "abstention_rate",
        "fold_stability",
        "calibration_error",
        "probabilities_calibrated",
        "evaluation_partition",
        "threshold_partition",
        "patient_disjoint",
        "split_manifest_sha256",
        "evaluation_records",
        "global_accuracy",
        "global_accuracy_definition",
        "metric_provenance",
        "confidence_level",
        "confidence_interval_method",
    )
    for label, audited_metric in audited_metrics.items():
        source_metric = source_metrics.get(label)
        if source_metric is None:
            raise EligibilityCertificateError(
                f"metrics artifact no longer contains eligible rule target {label}"
            )
        replayed_metric = _metric_from_mapping(
            label,
            dict(source_metric),
            context=metrics_payload,
            evidence_source=str(metrics_path),
        ).to_dict()
        mismatches = [
            field
            for field in metric_binding_fields
            if audited_metric.get(field) != replayed_metric.get(field)
        ]
        if mismatches:
            raise EligibilityCertificateError(
                f"selection audit for {label} is inconsistent with the metrics "
                f"artifact ({', '.join(mismatches)})"
            )

    manifest_stem = (
        "ptbxl_split_index" if dataset == "b1" else "ludb_split_index_repeat_1"
    )
    split_path = find_table(config.paths.manifests, manifest_stem)
    if split_path is None or not split_path.is_file():
        raise EligibilityCertificateError(
            "the active split manifest is unavailable for certificate issuance"
        )
    split_path = split_path.resolve()
    split_hash = sha256_file(split_path)
    if selected_split_hashes != {split_hash}:
        raise EligibilityCertificateError(
            "selected metrics are not bound to the active split manifest"
        )
    if str(metrics_payload.get("split_manifest_sha256") or "") != split_hash:
        raise EligibilityCertificateError(
            "metrics artifact split-manifest hash is missing or inconsistent"
        )

    model_path = _resolve_evidence_file(
        config,
        metrics_payload.get("model_path"),
        field="model_path",
    )
    model_hash = sha256_file(model_path)
    if str(metrics_payload.get("model_sha256") or "") != model_hash:
        raise EligibilityCertificateError(
            "metrics artifact model hash is missing or inconsistent"
        )

    hashes = {
        "split_manifest_sha256": split_hash,
        "model_sha256": model_hash,
        "metrics_sha256": sha256_file(metrics_path),
    }
    paths = {
        "split_manifest_sha256": split_path,
        "model_sha256": model_path,
        "metrics_sha256": metrics_path,
    }
    return hashes, paths


def _certify_eligible_rulebook(
    config: ProjectConfig,
    dataset: str,
    rulebook: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach and immediately verify a content-addressed release certificate."""

    hashes, paths = _verified_eligibility_provenance(config, dataset, rulebook)
    certified = attach_rulebook_eligibility_certificate(
        rulebook,
        config,
        **hashes,
    )
    verify_rulebook_eligibility_certificate(certified, config)
    # Re-hash immediately before returning so concurrent/stale evidence changes
    # fail closed instead of authorizing a rulebook against bytes no longer present.
    for hash_field, path in paths.items():
        if sha256_file(path) != hashes[hash_field]:
            raise EligibilityCertificateError(
                f"{hash_field} evidence changed during certificate issuance"
            )
    return certified


def _write_ineligible_rulebook(
    config: ProjectConfig,
    dataset: str,
    raw_path: Path,
    selection_audit: Mapping[str, Any],
) -> Path:
    """Revoke stale rulebooks whenever the current eligibility gate fails."""

    report_directory = config.paths.reports / "dss"
    report_directory.mkdir(parents=True, exist_ok=True)
    rulebook_json_path = report_directory / f"dss_rulebook_{dataset}.json"
    rulebook_markdown_path = report_directory / f"dss_rulebook_{dataset}.md"
    failure_path = report_directory / f"dss_selection_failure_{dataset}.json"
    failure = {
        "dataset": dataset,
        "status": "failed_no_features",
        "selection_audit": selection_audit,
        "raw_input": str(raw_path),
        "rules_eligible": False,
        "selected_features": [],
        "production_rules": [],
        "invalidated_rulebook_json": str(rulebook_json_path),
        "invalidated_rulebook_markdown": str(rulebook_markdown_path),
    }
    _write_json_atomic(failure_path, failure)
    invalidated_rulebook = {
        "dataset": dataset,
        "status": "ineligible_no_rulebook",
        "rules_eligible": False,
        "ontology_version": config.ontology_version,
        "orientation": "B_hat = A @ T; no operator/rules are eligible in this artifact",
        "selected_features": [],
        "discretization_plan": {},
        "medical_predicates": {},
        "production_rules": [],
        "selection_failure_audit": str(failure_path),
        "selection_audit": selection_audit,
        "caution": (
            "This file intentionally contains no rules. The current strict DSS "
            "eligibility gate failed; previously generated rules are revoked."
        ),
        "limitations": [
            "No class met the complete strict DSS eligibility contract.",
            "This zero-rule artifact revokes every earlier rulebook for the dataset.",
        ],
    }
    _write_json_atomic(rulebook_json_path, invalidated_rulebook)
    rulebook_markdown_path.write_text(
        "# DSS rulebook unavailable\n\n"
        f"Dataset: `{dataset}`\n\n"
        "The strict DSS eligibility gate failed. No production rules or selected "
        "features are eligible, and any previously generated rulebook is revoked.\n\n"
        f"See `{failure_path}` for the complete class-level audit.\n",
        encoding="utf-8",
    )
    manifest_path = config.paths.manifests / f"dss_{dataset}.json"
    invalidated_manifest = {
        "dataset": dataset,
        "status": "ineligible_no_rulebook",
        "rules_eligible": False,
        "ontology_version": config.ontology_version,
        "raw_input": str(raw_path),
        "rulebook_json": str(rulebook_json_path),
        "rulebook_markdown": str(rulebook_markdown_path),
        "selection_failure_audit": str(failure_path),
        "selected_features": 0,
        "rules": 0,
        "selection_audit": selection_audit,
    }
    _write_json_atomic(manifest_path, invalidated_manifest)
    return failure_path


def run(config: ProjectConfig, args: argparse.Namespace) -> int:
    dataset = args.dataset
    signed_path = find_table(config.paths.features, f"{dataset.upper()}_signed_train")
    raw_path = signed_path or find_table(
        config.paths.features, f"{dataset.upper()}_raw_train", required=True
    )
    assert raw_path is not None
    frame = read_table_frame(raw_path)
    labels = _load_labels(config, dataset, frame)
    raw_columns = [
        column
        for column in B_COLUMNS
        if column in frame.columns and not frame[column].isna().all()
    ]
    predicates = default_medical_predicates()
    predicate_executability = _predicate_executability(predicates, set(raw_columns))
    policy = SelectionPolicy(
        min_precision=float(getattr(args, "min_precision", 0.85)),
        min_recall=float(getattr(args, "min_recall", 0.85)),
        min_f1=float(getattr(args, "min_f1", 0.85)),
        min_specificity=float(getattr(args, "min_specificity", 0.85)),
        min_support=int(getattr(args, "min_class_support", args.min_support)),
        min_calibration_confidence=max(
            float(getattr(args, "min_calibration_confidence", 0.0)),
            float(config.dss.get("minimum_calibration_confidence", 0.70)),
        ),
        min_metric_lower_bound=float(
            config.dss.get("minimum_metric_lower_bound", 0.80)
        ),
        min_analyzable_coverage=float(
            config.dss.get("minimum_analyzable_coverage", 0.80)
        ),
        max_abstention_rate=float(config.dss.get("maximum_abstention_rate", 0.20)),
        min_fold_stability=float(config.dss.get("minimum_fold_stability", 0.80)),
        max_calibration_error=float(config.dss.get("maximum_calibration_error", 0.10)),
        min_global_accuracy=float(config.dss.get("minimum_global_accuracy", 0.90)),
        required_global_accuracy_definition=str(
            config.dss.get(
                "global_accuracy_definition",
                "compatibility_subset_exact_match",
            )
        ),
        allowed_metric_provenance=tuple(
            str(value)
            for value in config.dss.get(
                "allowed_metric_provenance",
                [
                    "held_out_compatibility_head",
                    "external_compatibility_audit",
                    "row_level_compatibility_predictions",
                ],
            )
        ),
        require_executable_predicate=True,
        require_complete_evidence=True,
        require_held_out_evidence=True,
        require_patient_disjoint=True,
        require_calibrated_probabilities=True,
        require_independent_threshold_partition=True,
        allowed_evaluation_partitions=tuple(
            str(value)
            for value in config.dss.get(
                "allowed_evaluation_partitions",
                ["held_out_test", "external_test", "external_validation"],
            )
        ),
        strict_mode=True,
    )
    predictions_arg = getattr(args, "predictions", None)
    predictions_path = Path(predictions_arg) if predictions_arg else None
    research_fallback_requested = bool(
        getattr(args, "allow_research_feature_fallback", False)
        or config.dss.get("allow_research_feature_fallback", False)
    )
    selected_labels, selection_audit = _selection_audit(
        config,
        dataset,
        labels,
        predicate_executability,
        policy=policy,
        predictions_path=predictions_path,
        prediction_columns={
            "prediction_id_col": str(getattr(args, "prediction_id_col", "record_id")),
            "true_label_col": str(getattr(args, "true_label_col", "true_label")),
            "pred_label_col": str(getattr(args, "pred_label_col", "pred_label")),
            "confidence_col": str(getattr(args, "confidence_col", "confidence")),
        },
        label_by_record=None,
        allow_research_fallback=research_fallback_requested,
    )
    operator, fit_columns, stable_transition_features = _load_transition_context(
        config, dataset
    )
    b_columns_for_transition = fit_columns or raw_columns
    selected_features = select_b_features(
        well_classified_labels=selected_labels,
        b_columns=raw_columns,
        predicates=predicates,
        operator=operator if operator and fit_columns and len(fit_columns) == len(operator[0]) else None,
        operator_b_columns=fit_columns,
        top_transition_features=int(args.top_transition_features),
        stable_transition_features=stable_transition_features,
        allow_research_fallback=False,
    )
    # Keep only columns available in the raw B table.
    selected_features = [feature for feature in selected_features if feature in raw_columns]
    if not selected_features:
        failure_path = _write_ineligible_rulebook(
            config,
            dataset,
            raw_path,
            selection_audit,
        )
        print(f"DSS selection failed closed; audit written to {failure_path}")
        return 8
    signature_thresholds: dict[str, dict[str, float]] = {}
    signature_artifact_path = config.paths.transition / "signature_artifact_v1.json"
    if dataset == "b1" and signature_artifact_path.exists():
        signature_issues: list[str] = []
        try:
            signature_artifact = load_signature_artifact(signature_artifact_path)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            signature_artifact = {}
            signature_issues.append(f"invalid_signature_artifact:{exc}")
        raw_train_path = find_table(config.paths.features, "B1_raw_train")
        raw_validation_path = find_table(config.paths.features, "B1_raw_val")
        if signature_artifact.get("ontology_version") != config.ontology_version:
            signature_issues.append("signature_ontology_version_mismatch")
        for source_path, hash_key, issue in (
            (raw_train_path, "raw_train_sha256", "signature_train_hash_mismatch"),
            (
                raw_validation_path,
                "raw_validation_sha256",
                "signature_validation_hash_mismatch",
            ),
        ):
            if (
                source_path is None
                or signature_artifact.get(hash_key) != hashlib.sha256(source_path.read_bytes()).hexdigest()
            ):
                signature_issues.append(issue)
        if signature_issues:
            failed_audit = dict(selection_audit)
            failed_audit["status"] = "failed_signature_provenance"
            failed_audit["signature_evidence_issues"] = signature_issues
            failure_path = _write_ineligible_rulebook(
                config,
                dataset,
                raw_path,
                failed_audit,
            )
            print(f"DSS signature provenance failed closed; audit written to {failure_path}")
            return 8
        for feature, payload in dict(signature_artifact.get("signatures", {})).items():
            model = dict(payload)
            if model.get("status") == "available":
                signature_thresholds[str(feature)] = {
                    str(key): float(value)
                    for key, value in dict(model.get("threshold_bands", {})).items()
                }
    rows = _rows_from_frame(frame, selected_features)
    object_ids = [str(record_id) for record_id in frame["record_id"]]
    plan = fit_wedd_discretization(
        rows,
        labels,
        features=selected_features,
        alpha=float(args.alpha),
        min_support=int(args.min_support),
        max_depth=int(args.max_depth),
        prefer_clinical_bins=not args.no_clinical_bins,
        signature_thresholds=signature_thresholds,
    )
    system = build_decision_system(
        rows,
        labels,
        plan,
        object_ids=object_ids,
        metadata={"dataset": dataset, "raw_path": str(raw_path), "source": "matrix_B_raw_train"},
    )
    rules = induce_rules(
        system,
        plan,
        min_support=int(args.min_support),
        use_reducts=not args.no_reducts,
        allowed_labels=selected_labels,
    )
    predicate_template_rules = []
    if not getattr(args, "no_predicate_rules", False):
        predicate_template_rules = build_predicate_template_rules(
            predicates,
            system,
            allowed_labels=selected_labels,
            include_weak_signature=False,
        )
        existing_ids = {rule.rule_id for rule in rules}
        rules.extend(rule for rule in predicate_template_rules if rule.rule_id not in existing_ids)
    rules = postprocess_rules(
        rules,
        predicates=predicates,
        max_rules_per_label=int(args.max_rules_per_label) if args.max_rules_per_label else None,
    )
    oof_rule_metrics = _cross_validated_rule_metrics(
        rows,
        labels,
        object_ids,
        features=selected_features,
        selected_labels=selected_labels,
        predicates=predicates,
        signature_thresholds=signature_thresholds,
        config=config,
        args=args,
        patient_groups=_patient_groups_for_records(config, dataset, object_ids),
    )
    for rule in rules:
        metrics = oof_rule_metrics.get(rule.target_label, {})
        rule.oof_precision = (
            float(metrics["precision"]) if metrics.get("precision") is not None else None
        )
        rule.oof_recall = (
            float(metrics["recall"]) if metrics.get("recall") is not None else None
        )
        rule.class_prior = (
            float(metrics["class_prior"])
            if metrics.get("class_prior") is not None
            else None
        )
        rule.calibration_uncertainty = (
            float(metrics["calibration_uncertainty"])
            if metrics.get("calibration_uncertainty") is not None
            else None
        )
        rule.fold_stability = (
            float(metrics["fold_stability"])
            if metrics.get("fold_stability") is not None
            else None
        )
    # Rule-generalization limits are part of the strict release contract. A
    # local configuration may demand stronger evidence, but cannot weaken it.
    minimum_rule_precision = max(
        float(config.dss.get("minimum_rule_oof_precision", 0.80)), 0.80
    )
    minimum_rule_recall = max(
        float(config.dss.get("minimum_rule_oof_recall", 0.80)), 0.80
    )
    minimum_rule_stability = max(
        float(config.dss.get("minimum_rule_fold_stability", 0.80)), 0.80
    )
    minimum_positive_folds = max(
        int(config.dss.get("minimum_rule_positive_folds", 3)), 3
    )
    rule_gate_audit: dict[str, dict[str, object]] = {}
    eligible_rule_labels: set[str] = set()
    for label, metrics in sorted(oof_rule_metrics.items()):
        reasons: list[str] = []
        if float(metrics.get("precision") or 0.0) < minimum_rule_precision:
            reasons.append(f"oof_precision<{minimum_rule_precision:.2f}")
        if float(metrics.get("recall") or 0.0) < minimum_rule_recall:
            reasons.append(f"oof_recall<{minimum_rule_recall:.2f}")
        if float(metrics.get("fold_stability") or 0.0) < minimum_rule_stability:
            reasons.append(f"fold_stability<{minimum_rule_stability:.2f}")
        if int(metrics.get("folds_with_positive_support") or 0) < minimum_positive_folds:
            reasons.append(f"positive_folds<{minimum_positive_folds}")
        if not reasons:
            eligible_rule_labels.add(label)
        rule_gate_audit[label] = {
            **metrics,
            "eligible": not reasons,
            "reasons": reasons,
        }
    rules = [rule for rule in rules if rule.target_label in eligible_rule_labels]
    if not rules:
        failed_audit = dict(selection_audit)
        failed_audit["rule_generalization_gate"] = rule_gate_audit
        failed_audit["status"] = "failed_no_generalizable_rules"
        failure_path = _write_ineligible_rulebook(
            config,
            dataset,
            raw_path,
            failed_audit,
        )
        print(f"DSS rule generalization gate failed closed; audit written to {failure_path}")
        return 8
    rulebook = {
        "dataset": dataset,
        "status": "eligible",
        "rules_eligible": True,
        "ontology_version": config.ontology_version,
        "orientation": "B_hat = A @ T; A is m x k, B is m x l, T is k x l",
        "selected_features": selected_features,
        "stable_transition_features": sorted(stable_transition_features),
        "label_distribution": {label: labels.count(label) for label in sorted(set(labels))},
        "selection_audit": selection_audit,
        "out_of_fold_rule_metrics": oof_rule_metrics,
        "rule_generalization_gate": rule_gate_audit,
        "rule_source_distribution": _source_distribution(rules),
        "discretization_plan": plan.to_dict(),
        "decision_system": {
            "universe_size": len(system.universe),
            "conditional_attributes": system.conditional_attributes,
            "decision_attribute": system.decision_attribute,
        },
        "medical_predicates": predicates_to_dict(predicates),
        "production_rules": [rule.to_dict() for rule in rules],
        "limitations": [
            "Rules are induced from available B-matrix artifacts and labels; no new cardiologist validation was performed in this run.",
            "The artifact is for research decision support only and is not a standalone diagnostic medical device.",
        ],
    }
    try:
        rulebook = _certify_eligible_rulebook(config, dataset, rulebook)
    except EligibilityCertificateError as exc:
        failed_audit = dict(selection_audit)
        failed_audit["status"] = "failed_eligibility_certificate_provenance"
        failed_audit["eligibility_certificate_issues"] = [str(exc)]
        failure_path = _write_ineligible_rulebook(
            config,
            dataset,
            raw_path,
            failed_audit,
        )
        print(
            "DSS eligibility certificate gate failed closed; "
            f"audit written to {failure_path}"
        )
        return 8
    json_path = config.paths.reports / "dss" / f"dss_rulebook_{dataset}.json"
    _write_json_atomic(json_path, rulebook)
    md_path = config.paths.reports / "dss" / f"dss_rulebook_{dataset}.md"
    write_dss_markdown_report(md_path, dataset, selected_features, labels, rules, int(args.min_support))
    manifest_path = config.paths.manifests / f"dss_{dataset}.json"
    manifest = {
        "dataset": dataset,
        "status": "eligible",
        "rules_eligible": True,
        "ontology_version": config.ontology_version,
        "raw_input": str(raw_path),
        "rulebook_json": str(json_path),
        "rulebook_markdown": str(md_path),
        "selected_features": len(selected_features),
        "rules": len(rules),
        "rule_source_distribution": _source_distribution(rules),
        "rulebook_sha256": sha256_file(json_path),
        "eligibility_evidence": rulebook["eligibility_evidence"],
        "eligibility_certificate_sha256": rulebook["eligibility_certificate"][
            "certificate_sha256"
        ],
        "selection_audit": selection_audit,
        "b_columns_for_transition": b_columns_for_transition,
        "stable_transition_features": sorted(stable_transition_features),
    }
    _write_json_atomic(manifest_path, manifest)
    print(f"Wrote DSS rulebook to {json_path}")
    print(f"Wrote DSS report to {md_path}")
    return 0


def export_rules_csv(rules: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rule_id", "target_label", "support", "confidence", "antecedents"])
        for rule in rules:
            writer.writerow([
                rule.rule_id,
                rule.target_label,
                rule.support_count,
                f"{rule.confidence:.6f}",
                ";".join(_condition_text(condition) for condition in rule.antecedents),
            ])
