"""Build clinician-facing raw feature matrices."""

from __future__ import annotations

from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.constants import DATASET_MAP
from tm_ecg.features.formulas import BeatMeasurement, RecordMeasurements
from tm_ecg.features.registry import build_raw_feature_row, feature_dictionary_rows
from tm_ecg.io.common import read_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.stages.shared import write_stage_manifest



def _measurement_path(config: ProjectConfig, dataset_key: str) -> Path:
    return config.paths.interim / f"{dataset_key}_record_measurements.json"


def _split_lookup(config: ProjectConfig, dataset_key: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if dataset_key == "ptbxl":
        manifest_path = config.paths.manifests / "split_manifest_ptbxl.json"
        if manifest_path.exists():
            payload = read_json(manifest_path)
            for row in payload.get("split_assignments", []):
                lookup[str(row["record_id"])] = str(row["split"])
    else:
        for path in sorted(config.paths.manifests.glob("split_manifest_ludb_repeat_*.json")):
            payload = read_json(path)
            for row in payload.get("split_assignments", []):
                lookup[f"{row['record_id']}::{row['split']}"] = str(row["split"])
    return lookup


def _load_measurements(path: Path) -> list[dict[str, object]]:
    payload = read_json(path)
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    if not isinstance(records, list):
        raise ValueError(f"Unexpected measurement payload in {path}")
    return records


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    dataset_key = DATASET_MAP[dataset]
    dictionary_rows = [row.to_dict() for row in feature_dictionary_rows()]
    dictionary_path = write_records_table(
        config.paths.features / f"{dataset.upper()}_dictionary.csv",
        dictionary_rows,
    )

    measurements_path = _measurement_path(config, dataset_key)
    if not measurements_path.exists():
        write_stage_manifest(
            config,
            f"build_{dataset}",
            {
                "dataset": dataset,
                "dictionary_path": str(dictionary_path),
                "status": "waiting_for_measurements",
                "expected_measurements_path": str(measurements_path),
            },
        )
        print(f"Feature dictionary written to {dictionary_path}; no measurements found yet.")
        return 0

    split_lookup = _split_lookup(config, dataset_key)
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for item in _load_measurements(measurements_path):
        beats = [BeatMeasurement(**beat) for beat in item.get("beats", [])]
        record = RecordMeasurements(
            record_id=str(item["record_id"]),
            beats=beats,
            tq_power_ratios=list(item.get("tq_power_ratios", [])),
            sampling_rate_hz=float(item.get("sampling_rate_hz", 500.0)),
            qrs_def_threshold=float(item.get("qrs_def_threshold", 0.5)),
            lead_quality_by_lead_db={
                str(key): float(value)
                for key, value in dict(item.get("lead_quality_by_lead_db", {})).items()
            },
            analyzable_duration_s=(
                float(item["analyzable_duration_s"])
                if item.get("analyzable_duration_s") is not None
                else None
            ),
        )
        row = build_raw_feature_row(record, config.thresholds)
        row["labels"] = "|".join(str(label) for label in item.get("labels", []))
        split = str(item.get("split") or split_lookup.get(str(item["record_id"])) or "unspecified")
        row["split"] = split
        grouped_rows.setdefault(split, []).append(row)

    written = {}
    for split, rows in grouped_rows.items():
        path = write_records_table(config.paths.features / f"{dataset.upper()}_raw_{split}.parquet", rows)
        written[split] = str(path)

    write_stage_manifest(
        config,
        f"build_{dataset}",
        {
            "dataset": dataset,
            "dictionary_path": str(dictionary_path),
            "raw_outputs": written,
            "status": "complete",
        },
    )
    print(f"Raw feature matrices written for {dataset}")
    return 0
