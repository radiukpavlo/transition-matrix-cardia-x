"""Triad construction and latent extraction stages."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import write_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.real_data import build_measurement_records, build_samples_for_dataset, save_latent_rows
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    records, triads = build_measurement_records(config, dataset)
    measurements_path = config.paths.interim / f"{dataset}_record_measurements.json"
    write_json(measurements_path, {"records": records})
    triads_path = write_records_table(config.paths.interim / f"{dataset}_triad_membership.parquet", triads)
    payload = {
        "dataset": dataset,
        "pooling": config.latents["pooling"],
        "status": "complete",
        "measurement_records": len(records),
        "measurements_path": str(measurements_path),
        "triads_path": str(triads_path),
        "notes": [
            "Triads are [previous, current, next] accepted beats aligned on the R peak.",
            "Central beats without both neighbors are excluded.",
        ],
    }
    write_stage_manifest(config, f"triads_{dataset}", payload)
    print(f"Triad and measurement artifacts written for {dataset}")
    return 0


def extract_latents(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    latent_rows_by_split = build_samples_for_dataset(config, dataset, include_targets=False, checkpoint_dataset="ptbxl")
    written = save_latent_rows(config, dataset, latent_rows_by_split)
    payload = {
        "dataset": dataset,
        "penultimate_dim": config.latents["penultimate_dim"],
        "variance_retained": config.latents["variance_retained"],
        "status": "complete",
        "outputs": written,
        "notes": [
            "Extract preactivation penultimate vectors only.",
            "Apply zero-variance masking, training-only standardization, and PCA after pooling.",
        ],
    }
    write_stage_manifest(config, f"extract_a_{dataset}", payload)
    print(f"Latent extraction outputs written for {dataset}")
    return 0
