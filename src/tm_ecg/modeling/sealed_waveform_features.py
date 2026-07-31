"""Label-blind waveform feature extraction for the sealed development split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.features.beat_extraction import (
    build_measurement_records_for_tasks,
)
from tm_ecg.features.formulas import BeatMeasurement, RecordMeasurements
from tm_ecg.features.registry import (
    build_raw_feature_row,
    governed_project_feature_specs,
)
from tm_ecg.reproducibility import write_artifact_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_sealed_development_waveform_features(
    config: ProjectConfig,
    *,
    index_path: Path,
    sealed_manifest_path: Path,
    existing_feature_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Extract only waveform measurements; never load the index label column."""

    import pandas as pd  # type: ignore

    sealed = json.loads(sealed_manifest_path.read_text(encoding="utf-8"))
    if sealed.get("confirmatory_labels_opened") is not False:
        raise RuntimeError("Sealed manifest no longer certifies unopened labels")
    patient_map = sealed.get("patient_ids", {})
    development_patients = set(
        str(item)
        for split in ("development_train", "development_validation")
        for item in patient_map.get(split, [])
    )
    # Column projection is a material leakage control: labels, source codes,
    # diagnostic text, and clinical outcomes never enter this process.
    index = pd.read_parquet(
        index_path,
        columns=[
            "record_id",
            "patient_id",
            "strat_fold",
            "filename_hr",
        ],
    )
    index["record_id"] = index["record_id"].astype(str)
    index["patient_key"] = [
        (
            str(patient_id)
            if str(patient_id).strip()
            and str(patient_id).lower() != "nan"
            else f"record::{record_id}"
        )
        for patient_id, record_id in zip(
            index["patient_id"],
            index["record_id"],
            strict=True,
        )
    ]
    selected = index[
        index["patient_key"].isin(development_patients)
        & index["strat_fold"].astype(str).isin(
            {"1", "2", "3", "4", "5", "6", "7", "8"}
        )
    ].copy()
    if len(selected) != int(
        sealed["counts"]["development_train"]["records"]
    ) + int(sealed["counts"]["development_validation"]["records"]):
        raise RuntimeError("Development record retention does not match manifest")
    governed = governed_project_feature_specs()
    feature_names = [
        name
        for name, spec in governed.items()
        if spec.inference_safe
        and spec.target_leakage_risk == "low_waveform_measurement_only"
    ]
    existing = (
        pd.read_parquet(
            existing_feature_path,
            columns=["record_id", *feature_names],
        )
        if existing_feature_path.exists()
        else pd.DataFrame(columns=["record_id", *feature_names])
    )
    existing["record_id"] = existing["record_id"].astype(str)
    existing = existing[
        existing["record_id"].isin(set(selected["record_id"]))
    ].copy()
    if existing["record_id"].duplicated().any():
        raise RuntimeError("Existing waveform features contain duplicate IDs")
    missing_ids = set(selected["record_id"]) - set(existing["record_id"])
    missing = selected[selected["record_id"].isin(missing_ids)].sort_values(
        "record_id"
    )
    tasks = [
        (
            "sealed_development",
            {
                "record_id": str(row.record_id),
                "source_path": str(row.filename_hr),
                "labels": [],
            },
        )
        for row in missing.itertuples(index=False)
    ]
    measured, _triads = build_measurement_records_for_tasks(
        config,
        "ptbxl",
        tasks,
    )
    new_rows: list[dict[str, object]] = []
    for item in measured:
        record = RecordMeasurements(
            record_id=str(item["record_id"]),
            beats=[
                BeatMeasurement(**dict(beat))
                for beat in item.get("beats", [])
            ],
            tq_power_ratios=list(item.get("tq_power_ratios", [])),
            sampling_rate_hz=float(item.get("sampling_rate_hz", 500.0)),
            qrs_def_threshold=float(item.get("qrs_def_threshold", 0.5)),
            lead_quality_by_lead_db={
                str(key): float(value)
                for key, value in dict(
                    item.get("lead_quality_by_lead_db", {})
                ).items()
            },
            analyzable_duration_s=(
                float(item["analyzable_duration_s"])
                if item.get("analyzable_duration_s") is not None
                else None
            ),
        )
        row = build_raw_feature_row(record, config.thresholds)
        new_rows.append(
            {
                "record_id": str(item["record_id"]),
                **{name: row.get(name) for name in feature_names},
            }
        )
    combined = pd.concat(
        [existing, pd.DataFrame(new_rows)],
        ignore_index=True,
    ).sort_values("record_id")
    expected_ids = set(selected["record_id"])
    if (
        combined["record_id"].duplicated().any()
        or set(combined["record_id"]) != expected_ids
    ):
        raise RuntimeError("Waveform feature extraction did not retain every row")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined[["record_id", *feature_names]].to_parquet(
        output_path,
        index=False,
    )
    manifest = {
        "version": 1,
        "artifact_id": "sealed_development_waveform_features_v1",
        "index_sha256": _sha256_file(index_path),
        "sealed_manifest_sha256": _sha256_file(sealed_manifest_path),
        "existing_feature_sha256": (
            _sha256_file(existing_feature_path)
            if existing_feature_path.exists()
            else None
        ),
        "output_sha256": _sha256_file(output_path),
        "records": len(combined),
        "patients": int(selected["patient_key"].nunique()),
        "reused_records": len(existing),
        "newly_measured_records": len(new_rows),
        "feature_count": len(feature_names),
        "confirmatory_labels_opened": False,
        "index_columns_loaded": [
            "record_id",
            "patient_id",
            "strat_fold",
            "filename_hr",
        ],
        "forbidden_columns_loaded": [],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(
        output_path.parent,
        producer_command="tm-ecg extract-sealed-waveform-features",
        input_hashes={
            "index": manifest["index_sha256"],
            "sealed_manifest": manifest["sealed_manifest_sha256"],
        },
        code_root=config.paths.root,
    )
    return manifest


def run(config: ProjectConfig, args: object) -> int:
    root = config.paths.root

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    manifest = extract_sealed_development_waveform_features(
        config,
        index_path=resolve(str(getattr(args, "index"))),
        sealed_manifest_path=resolve(
            str(getattr(args, "sealed_manifest"))
        ),
        existing_feature_path=resolve(
            str(getattr(args, "existing_features"))
        ),
        output_path=resolve(str(getattr(args, "output"))),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
