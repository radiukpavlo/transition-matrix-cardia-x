"""Fit calibrated signature scores on local train/validation folds only."""

from __future__ import annotations

import argparse
from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.features.signatures import apply_signature_scores, fit_signature_artifact
from tm_ecg.io.common import sha256_file, write_json
from tm_ecg.io.readers import find_table, read_table_frame
from tm_ecg.io.tabular import write_records_table
from tm_ecg.stages.shared import write_stage_manifest


def _rows(path: Path) -> list[dict[str, object]]:
    return read_table_frame(path).to_dict(orient="records")


def run(config: ProjectConfig, args: argparse.Namespace) -> int:
    dataset = str(getattr(args, "dataset", "b1"))
    if dataset != "b1":
        raise ValueError("Signature calibration currently requires the PTB-XL B1 development folds")
    train_path = find_table(config.paths.features, "B1_raw_train", required=True)
    validation_path = find_table(config.paths.features, "B1_raw_val", required=True)
    train_rows = _rows(train_path)
    validation_rows = _rows(validation_path)
    artifact = fit_signature_artifact(train_rows, validation_rows, random_seed=config.seed)
    artifact.update(
        {
            "ontology_version": config.ontology_version,
            "raw_train_path": str(train_path),
            "raw_train_sha256": sha256_file(train_path),
            "raw_validation_path": str(validation_path),
            "raw_validation_sha256": sha256_file(validation_path),
        }
    )
    artifact_path = config.paths.transition / "signature_artifact_v1.json"
    write_json(artifact_path, artifact)

    signed_outputs: dict[str, str] = {}
    for split in ("train", "val", "test"):
        try:
            source = find_table(config.paths.features, f"B1_raw_{split}", required=True)
        except FileNotFoundError:
            continue
        rows = _rows(source)
        for row in rows:
            scores, states = apply_signature_scores(row, artifact)
            row.update(scores)
            row["signature_states"] = "|".join(
                f"{key}:{value}" for key, value in sorted(states.items())
            )
        destination = write_records_table(
            config.paths.features / f"B1_signed_{split}.parquet", rows
        )
        signed_outputs[split] = str(destination)
    manifest = {
        "dataset": dataset,
        "artifact": str(artifact_path),
        "training_split_hash": artifact["training_split_hash"],
        "ontology_version": config.ontology_version,
        "raw_train_sha256": artifact["raw_train_sha256"],
        "raw_validation_sha256": artifact["raw_validation_sha256"],
        "signed_outputs": signed_outputs,
    }
    write_stage_manifest(config, "train_signatures_b1", manifest)
    print(f"Calibrated signature artifact written to {artifact_path}")
    return 0
