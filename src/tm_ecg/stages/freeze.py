"""Freeze reproducibility manifests and artifact hashes."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import sha256_file, write_json
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    experiment = getattr(args, "experiment")
    tracked = []
    for root in (
        config.paths.features,
        config.paths.latents,
        config.paths.transition,
        config.paths.reports,
        config.paths.manifests,
    ):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                tracked.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                )
    manifest_path = config.paths.manifests / f"freeze_manifest_{experiment}.json"
    write_json(
        manifest_path,
        {
            "experiment": experiment,
            "artifacts": tracked,
        },
    )
    write_stage_manifest(
        config,
        f"freeze_{experiment}",
        {
            "experiment": experiment,
            "status": "complete",
            "freeze_manifest": str(manifest_path),
            "artifact_count": len(tracked),
        },
    )
    print(f"Freeze manifest written for {experiment}")
    return 0
