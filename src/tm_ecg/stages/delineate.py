"""Fiducial delineation and beat acceptance stage policy."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.signal.fiducials import DELINEATION_METHOD, SECONDARY_R_DETECTOR
from tm_ecg.stages.readiness import (
    expected_measurement_path,
    raw_dataset_root,
    require_split_manifest,
)
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
        "minimum_valid_beats": config.thresholds["minimum_valid_beats"],
        "minimum_analyzable_fraction": config.thresholds["minimum_analyzable_fraction"],
        "fiducial_method": DELINEATION_METHOD,
        "delineation_confidence_minimum": float(
            config.thresholds.get("delineation_confidence_minimum", 0.5)
        ),
        "secondary_r_detector": SECONDARY_R_DETECTOR,
        "r_peak_match_tolerance_ms": float(
            config.thresholds.get("r_peak_match_tolerance_ms", 60.0)
        ),
        "detector_agreement_metric": "one_to_one_temporal_match_f1",
        "analyzable_duration_definition": (
            "sum_of_adjacent_RR_intervals_bounded_by_accepted_quality_valid_beats"
        ),
        "status": "ready_for_executable_measurement_extraction",
        "notes": [
            "The triads/measurement stage derives boundaries from each waveform; fixed-offset "
            "P/QRS/T surrogates are prohibited.",
            "QRS/T delineation failure, low confidence, low lead quality, or unmatched independent "
            "R detections reject the affected beat.",
            "An absent P delineation disables atrial measurements without inventing a P wave.",
            "LUDB and PTB-XL+ annotations remain external validation resources and are not silently "
            "substituted for algorithmic outputs.",
        ],
    }
    write_stage_manifest(config, f"delineate_{dataset}", payload)
    print(f"Delineation readiness manifest written for {dataset}")
    return 0
