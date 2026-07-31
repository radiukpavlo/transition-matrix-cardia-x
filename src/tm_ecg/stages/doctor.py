"""Environment and artifact readiness checks."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.constants import DATASET_MAP
from tm_ecg.reproducibility import environment_fingerprint
from tm_ecg.stages.shared import write_stage_manifest


REQUIRED_PACKAGES = ["numpy", "scipy", "pandas", "pyarrow", "sklearn", "wfdb"]
OPTIONAL_PACKAGES = ["torch", "pytest", "invoke", "ruff", "mypy"]


def _package_status(names: list[str]) -> dict[str, bool]:
    status = {}
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            status[name] = False
        else:
            status[name] = True
    return status


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run(config: ProjectConfig, args: object) -> int:
    required = _package_status(REQUIRED_PACKAGES)
    optional = _package_status(OPTIONAL_PACKAGES)
    archives = {
        key: _exists(config.paths.data_lock / dataset.archive)
        for key, dataset in config.datasets.items()
    }
    materialized = {}
    for b_name, dataset_key in DATASET_MAP.items():
        materialized[b_name] = {
            "raw_train": _exists(config.paths.features / f"{b_name.upper()}_raw_train.csv")
            or _exists(config.paths.features / f"{b_name.upper()}_raw_train.parquet"),
            "fit_train": _exists(config.paths.features / f"{b_name.upper()}_fit_train.csv")
            or _exists(config.paths.features / f"{b_name.upper()}_fit_train.parquet"),
            "a_train": _exists(config.paths.latents / f"A_{dataset_key}_train.parquet"),
            "transition": _exists(config.paths.transition / f"{b_name.upper()}_T_ridge.json")
            or _exists(config.paths.transition / f"{b_name.upper()}_T_ridge.npz"),
        }

    payload = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "required_packages": required,
        "optional_packages": optional,
        "data_lock_archives": archives,
        "materialized_artifacts": materialized,
        "ready_for_lightweight_artifact_mode": all(
            all(status.values()) for status in materialized.values()
        ),
        "ready_for_full_rebuild": all(required.values()) and archives.get("ptbxl") and archives.get("ludb"),
        "release_environment": environment_fingerprint(
            Path.cwd(),
            lock_path=Path.cwd() / "requirements.lock",
        ),
    }
    write_stage_manifest(config, "doctor", payload)
    print("Doctor check")
    print(f"  Python: {payload['python_version']} ({payload['python_executable']})")
    missing_required = [name for name, ok in required.items() if not ok]
    print(f"  Required packages: {'ok' if not missing_required else 'missing ' + ', '.join(missing_required)}")
    missing_archives = [name for name, ok in archives.items() if not ok]
    print(f"  Data archives: {'ok' if not missing_archives else 'missing ' + ', '.join(missing_archives)}")
    print(f"  Lightweight artifacts: {'ok' if payload['ready_for_lightweight_artifact_mode'] else 'incomplete'}")
    lock_valid = bool(
        payload["release_environment"]["dependency_lock"]["valid"]
    )
    print(f"  Exact dependency lock: {'ok' if lock_valid else 'mismatch'}")
    output = config.paths.reports / "metrics" / "environment_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload["release_environment"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0 if not missing_required and lock_valid else 1
