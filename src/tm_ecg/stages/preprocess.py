"""Waveform preprocessing policy stage."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.signal.filtering import branch_parameters
from tm_ecg.stages.readiness import raw_dataset_root, require_dataset_index, require_split_manifest
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    root = raw_dataset_root(config, dataset)
    index_path = require_dataset_index(config, dataset)
    split_path = require_split_manifest(config, dataset)
    payload = {
        "dataset": dataset,
        "raw_root": str(root),
        "index_path": str(index_path),
        "split_manifest": str(split_path),
        "detection_branch": branch_parameters(config.filters["detection"]),
        "diagnostic_branch": branch_parameters(config.filters["diagnostic"]),
        "status": "ready_for_executable_measurement_extraction",
        "notes": [
            "Filtering is executed per record by the triads/measurement stage to avoid duplicating large waveforms.",
            "Detection branch allows 0.67 Hz high-pass for anchor robustness.",
            "Diagnostic branch preserves 0.05 Hz high-pass for ST/QT fidelity.",
        ],
    }
    write_stage_manifest(config, f"preprocess_{dataset}", payload)
    print(f"Preprocessing readiness manifest written for {dataset}")
    return 0
