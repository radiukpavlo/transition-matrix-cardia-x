"""Prerequisite checks shared by lightweight stage entrypoints."""

from __future__ import annotations

from pathlib import Path

from tm_ecg.config import ProjectConfig


def require_dataset_index(config: ProjectConfig, dataset: str) -> Path:
    path = config.paths.manifests / f"{dataset}_index.parquet"
    if not path.exists():
        path = config.paths.manifests / f"{dataset}_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset index for {dataset}. Run `tm-ecg index` first.")
    return path


def require_split_manifest(config: ProjectConfig, dataset: str) -> Path:
    if dataset == "ptbxl":
        path = config.paths.manifests / "split_manifest_ptbxl.json"
    else:
        path = config.paths.manifests / "split_manifest_ludb_repeat_1.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest for {dataset}. Run `tm-ecg splits --dataset {dataset}` first.")
    return path


def raw_dataset_root(config: ProjectConfig, dataset: str) -> Path:
    root = config.paths.raw / config.datasets[dataset].extract_dir
    if not root.exists():
        raise FileNotFoundError(f"Missing raw dataset directory for {dataset}. Run `tm-ecg ingest` first.")
    return root


def expected_measurement_path(config: ProjectConfig, dataset: str) -> Path:
    return config.paths.interim / f"{dataset}_record_measurements.json"
