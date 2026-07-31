"""Adaptive R-peak detection stage policy."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.stages.readiness import expected_measurement_path, raw_dataset_root, require_split_manifest
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    root = raw_dataset_root(config, dataset)
    split_path = require_split_manifest(config, dataset)
    payload = {
        "dataset": dataset,
        "raw_root": str(root),
        "split_manifest": str(split_path),
        "measurement_output": str(expected_measurement_path(config, dataset)),
        "lead_ii_snr_db": config.thresholds["lead_ii_snr_db"],
        "status": "ready_for_executable_measurement_extraction",
        "notes": [
            "R-peak detection is executed per record by the triads/measurement stage.",
            "Use adaptive RR-horizon updates instead of a fixed 260-sample window.",
            "Accepted peaks must satisfy cross-lead temporal consistency.",
        ],
    }
    write_stage_manifest(config, f"rpeaks_{dataset}", payload)
    print(f"R-peak readiness manifest written for {dataset}")
    return 0
