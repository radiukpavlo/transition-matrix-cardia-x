"""Shared helpers for stage entrypoints."""

from __future__ import annotations

from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import utc_now_iso, write_json


def dataset_root(config: ProjectConfig, dataset_key: str) -> Path:
    return config.paths.raw / config.datasets[dataset_key].extract_dir


def find_single_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    matches.sort()
    return matches[0]


def write_stage_manifest(config: ProjectConfig, stage_name: str, payload: dict[str, object]) -> Path:
    destination = config.paths.manifests / f"{stage_name}.json"
    body = {"stage": stage_name, "generated_at_utc": utc_now_iso(), **payload}
    write_json(destination, body)
    return destination
