"""Classifier training stage."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.real_data import train_ptbxl_classifier
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    outputs = train_ptbxl_classifier(config)
    payload = {
        "dataset": dataset,
        "latent_dim": config.latents["penultimate_dim"],
        "status": "complete",
        "outputs": outputs,
        "notes": [
            "Use focal loss or dynamically weighted cross-entropy.",
            "Track macro-F1, class-wise recall, and calibration on validation only.",
        ],
    }
    write_stage_manifest(config, f"train_classifier_{dataset}", payload)
    print(f"Classifier checkpoint written for {dataset}")
    return 0
