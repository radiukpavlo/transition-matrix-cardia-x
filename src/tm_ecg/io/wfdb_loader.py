"""Record loading via wfdb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import read_json


def _runtime():
    import numpy as np  # type: ignore
    import wfdb  # type: ignore
    from scipy import signal as sp_signal  # type: ignore
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore

    return np, torch, wfdb, sp_signal


def _parse_labels(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    text = str(raw or "")
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text] if text else []


def split_entries(config: ProjectConfig, dataset: str) -> dict[str, list[dict[str, Any]]]:
    if dataset == "ptbxl":
        payload = read_json(config.paths.manifests / "split_manifest_ptbxl.json")
        grouped = {"train": [], "val": [], "test": []}
        for row in payload["split_assignments"]:
            grouped[str(row["split"])].append(row)
        return grouped

    payload = read_json(config.paths.manifests / "split_manifest_ludb_repeat_1.json")
    grouped = {"train": [], "val": [], "test": []}
    for row in payload["split_assignments"]:
        split = str(row["split"])
        prefix = "repeat_1_fold_1_"
        if split.startswith(prefix):
            grouped[split[len(prefix) :]].append(row)
    return grouped


def _resolved_dataset_root(config: ProjectConfig, dataset: str) -> Path:
    base = config.paths.raw / config.datasets[dataset].extract_dir
    children = sorted(child for child in base.iterdir() if child.is_dir())
    return children[0] if children else base


def _lead_index(sig_names: list[str], target: str) -> int:
    lookup = {name.lower(): idx for idx, name in enumerate(sig_names)}
    if target.lower() in lookup:
        return lookup[target.lower()]
    alias = target.lower().replace("av", "a")
    return lookup.get(alias, 1 if len(sig_names) > 1 else 0)


def _load_record(config: ProjectConfig, dataset: str, entry: dict[str, Any]):
    np, _torch, wfdb, _sp_signal = _runtime()
    if dataset == "ptbxl":
        root = _resolved_dataset_root(config, dataset)
        record_path = root / str(entry["source_path"])
    else:
        record_path = Path(str(entry["source_path"])).with_suffix("")
    record = wfdb.rdrecord(str(record_path))
    return np.asarray(record.p_signal, dtype=np.float32), float(record.fs), list(record.sig_name)
