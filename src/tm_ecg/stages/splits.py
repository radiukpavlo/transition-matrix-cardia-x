"""Ontology mapping and frozen split generation."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import stable_hash, utc_now_iso, write_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.ontology import appendix_d_mapping
from tm_ecg.stages.shared import write_stage_manifest
from tm_ecg.types import RecordIndexRow, SplitManifest


def _load_index_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"Parquet index requires pyarrow: {path}") from exc
    table = pq.read_table(path)
    columns = {name: table[name].to_pylist() for name in table.column_names}
    row_count = len(next(iter(columns.values()), []))
    rows = []
    for idx in range(row_count):
        rows.append({name: columns[name][idx] for name in table.column_names})
    return rows


def _index_path(config: ProjectConfig, dataset: str) -> Path:
    parquet = config.paths.manifests / f"{dataset}_index.parquet"
    if parquet.exists():
        return parquet
    csv_path = config.paths.manifests / f"{dataset}_index.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Missing index for {dataset}. Run `tm-ecg index` first.")


def _row_id(dataset: str, split: str, record_id: str) -> str:
    return stable_hash([dataset, split, record_id])[:16]


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _record_index_row(row: dict[str, object], dataset: str, split_name: str) -> RecordIndexRow:
    labels = [item for item in str(row.get("labels", "")).split("|") if item]
    raw_labels = [item for item in str(row.get("raw_source_labels", "")).split("|") if item]
    source_path = row.get("filename_hr") or row.get("filename_lr") or row.get("header_path") or ""
    source_metadata = {
        key: row.get(key)
        for key in ("source_codes", "source_likelihoods_json", "diagnosis_text", "pacemaker")
        if row.get(key) not in {None, ""}
    }
    return RecordIndexRow(
        row_id=_row_id(dataset, split_name, str(row["record_id"])),
        record_id=str(row["record_id"]),
        patient_id=str(row["patient_id"]) if row.get("patient_id") not in {None, ""} else None,
        dataset=dataset,
        split=split_name,
        labels=labels,
        source_path=str(source_path),
        preprocessing_hash="pending",
        ontology_version="pending",
        raw_source_labels=raw_labels,
        axis_targets=_json_object(row.get("axis_targets_json")),
        source_metadata=source_metadata,
    )


def _split_audit(manifest: SplitManifest) -> dict[str, object]:
    """Report clinical prevalence and prove patient isolation per evaluation unit."""

    units: dict[str, list[RecordIndexRow]] = defaultdict(list)
    for row in manifest.split_assignments:
        unit = "locked" if manifest.dataset == "ptbxl" else row.split.rsplit("_", 1)[0]
        units[unit].append(row)
    overlap_findings: list[dict[str, object]] = []
    unit_reports: dict[str, object] = {}
    for unit, rows in sorted(units.items()):
        patients: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        prevalence: dict[str, Counter[str]] = defaultdict(Counter)
        combinations: Counter[str] = Counter()
        unknown = 0
        for row in rows:
            split_name = row.split if manifest.dataset == "ptbxl" else row.split.rsplit("_", 1)[1]
            patient = row.patient_id or f"record:{row.record_id}"
            patients[split_name].add(patient)
            target = row.axis_targets
            for axis in ("rhythm", "ectopy", "conduction", "repolarization"):
                for value in target.get(axis, []):  # type: ignore[union-attr]
                    prevalence[f"{split_name}:{axis}"][str(value)] += 1
            prevalence[f"{split_name}:pacing"][str(target.get("pacing", "unknown"))] += 1
            prevalence[f"{split_name}:quality"][str(target.get("quality", "unknown"))] += 1
            if target.get("unsupported_source_labels"):
                unknown += 1
            combinations["|".join(sorted(row.labels)) or "<none>"] += 1
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(patients[left] & patients[right])
            if overlap:
                overlap_findings.append(
                    {"unit": unit, "left": left, "right": right, "patient_ids": overlap}
                )
        unit_reports[unit] = {
            "row_count": len(rows),
            "patient_counts": {key: len(value) for key, value in patients.items()},
            "axis_prevalence": {key: dict(value) for key, value in sorted(prevalence.items())},
            "label_combinations": dict(combinations),
            "unknown_label_rows": unknown,
        }
    audit = {
        "dataset": manifest.dataset,
        "patient_overlap_count": sum(len(item["patient_ids"]) for item in overlap_findings),
        "patient_overlaps": overlap_findings,
        "evaluation_units": unit_reports,
    }
    if overlap_findings:
        raise RuntimeError(f"Patient leakage detected: {overlap_findings}")
    return audit


def _freeze_ptbxl(config: ProjectConfig) -> SplitManifest:
    rows = _load_index_rows(_index_path(config, "ptbxl"))
    train_pool = [row for row in rows if str(row.get("strat_fold", "")) in {"1", "2", "3", "4", "5", "6", "7", "8"}]
    val_rows = [row for row in rows if str(row.get("strat_fold", "")) == "9"]
    test_rows = [row for row in rows if str(row.get("strat_fold", "")) == "10"]

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in train_pool:
        patient_id = row.get("patient_id") or f"anon-{row['record_id']}"
        grouped.setdefault(str(patient_id), []).append(row)

    patients = list(grouped.keys())
    random.Random(config.seed).shuffle(patients)
    selected_train: list[dict[str, str]] = []
    total_target = int(config.splits["ptbxl_train_target_rows"])
    for patient_id in patients:
        bucket = grouped[patient_id]
        if len(selected_train) + len(bucket) > total_target:
            continue
        selected_train.extend(bucket)
        if len(selected_train) == total_target:
            break
    if len(selected_train) < total_target:
        raise RuntimeError(
            f"Could only select {len(selected_train)} PTB-XL rows for the locked 10,000-row cohort."
        )

    assignments: list[RecordIndexRow] = []
    for split_name, split_rows in (("train", selected_train), ("val", val_rows), ("test", test_rows)):
        for row in split_rows:
            indexed = _record_index_row(row, "ptbxl", split_name)
            indexed.ontology_version = config.ontology_version
            assignments.append(indexed)

    return SplitManifest(
        dataset="ptbxl",
        generated_at_utc=utc_now_iso(),
        seed=config.seed,
        split_assignments=assignments,
        notes=["Locked 10,000-row PTB-XL training cohort sampled patient-disjointly from folds 1-8."],
    )


def _freeze_ludb(config: ProjectConfig) -> list[SplitManifest]:
    rows = _load_index_rows(_index_path(config, "ludb"))
    manifests: list[SplitManifest] = []
    repeats = int(config.datasets["ludb"].repeats or 1)
    fold_count = int(config.datasets["ludb"].folds or 5)
    for repeat_idx in range(repeats):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            identity = str(row.get("patient_id") or f"record:{row['record_id']}")
            grouped[identity].append(row)
        shuffled = list(grouped.values())
        random.Random(config.seed + repeat_idx).shuffle(shuffled)
        folds = [[] for _ in range(fold_count)]
        for bucket in shuffled:
            smallest = min(range(fold_count), key=lambda idx: (len(folds[idx]), idx))
            folds[smallest].extend(bucket)

        assignments: list[RecordIndexRow] = []
        for fold_idx in range(fold_count):
            test_fold = fold_idx
            val_fold = (fold_idx + 1) % fold_count
            for idx, fold_rows in enumerate(folds):
                split_name = "train"
                if idx == test_fold:
                    split_name = "test"
                elif idx == val_fold:
                    split_name = "val"
                split_key = f"repeat_{repeat_idx+1}_fold_{fold_idx+1}_{split_name}"
                for row in fold_rows:
                    indexed = _record_index_row(row, "ludb", split_key)
                    indexed.ontology_version = config.ontology_version
                    assignments.append(indexed)
        manifests.append(
            SplitManifest(
                dataset="ludb",
                generated_at_utc=utc_now_iso(),
                seed=config.seed + repeat_idx,
                split_assignments=assignments,
                notes=[f"Repeated stratified-style 5-fold split repeat {repeat_idx + 1}."],
            )
        )
    return manifests


def _write_ontology_mapping(config: ProjectConfig) -> Path:
    rows = [item.to_dict() for item in appendix_d_mapping()]
    return write_records_table(config.paths.manifests / "ontology_mapping.csv", rows)


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    ontology_path = _write_ontology_mapping(config)
    if dataset == "ptbxl":
        manifest = _freeze_ptbxl(config)
        audit = _split_audit(manifest)
        manifest_path = config.paths.manifests / "split_manifest_ptbxl.json"
        audit_path = config.paths.manifests / "split_audit_ptbxl.json"
        write_json(manifest_path, manifest.to_dict())
        write_json(audit_path, audit)
        write_records_table(
            config.paths.manifests / "ptbxl_split_index.parquet",
            [row.to_dict() for row in manifest.split_assignments],
        )
        payload = {
            "dataset": "ptbxl",
            "ontology_mapping": str(ontology_path),
            "split_manifest": str(manifest_path),
            "split_audit": str(audit_path),
            "row_count": len(manifest.split_assignments),
        }
    else:
        manifests = _freeze_ludb(config)
        written = []
        total_rows = 0
        for repeat_idx, manifest in enumerate(manifests, start=1):
            audit = _split_audit(manifest)
            manifest_path = config.paths.manifests / f"split_manifest_ludb_repeat_{repeat_idx}.json"
            audit_path = config.paths.manifests / f"split_audit_ludb_repeat_{repeat_idx}.json"
            write_json(manifest_path, manifest.to_dict())
            write_json(audit_path, audit)
            write_records_table(
                config.paths.manifests / f"ludb_split_index_repeat_{repeat_idx}.parquet",
                [row.to_dict() for row in manifest.split_assignments],
            )
            written.append(str(manifest_path))
            total_rows += len(manifest.split_assignments)
        payload = {
            "dataset": "ludb",
            "ontology_mapping": str(ontology_path),
            "split_manifests": written,
            "row_count": total_rows,
        }
    write_stage_manifest(config, f"splits_{dataset}", payload)
    print(f"Split manifests written for {dataset}")
    return 0
