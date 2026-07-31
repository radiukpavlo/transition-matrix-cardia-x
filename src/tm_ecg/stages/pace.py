"""Pacing artifact policy stage."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.stages.readiness import raw_dataset_root, require_split_manifest
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    root = raw_dataset_root(config, dataset)
    split_path = require_split_manifest(config, dataset)
    payload = {
        "dataset": dataset,
        "raw_root": str(root),
        "split_manifest": str(split_path),
        "status": "ready_for_executable_measurement_extraction",
        "notes": [
            "Pacing checks are executed per record by the measurement extraction path.",
            "Pacing spikes must be detected before routine morphology filtering.",
            "Spike-removed waveforms must preserve adjacent physiologic morphology.",
        ],
    }
    write_stage_manifest(config, f"pace_{dataset}", payload)
    print(f"Pacing readiness manifest written for {dataset}")
    return 0
