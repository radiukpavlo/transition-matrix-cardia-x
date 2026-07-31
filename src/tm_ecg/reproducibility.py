"""Release-environment and artifact-manifest controls."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import locale
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Mapping


LOCK_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact_lock(path: str | Path) -> dict[str, str]:
    """Parse an exact-pinned requirements lock and reject loose requirements."""

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = LOCK_PATTERN.fullmatch(line)
        if not match:
            raise ValueError(
                f"Lock line {line_number} is not an exact package==version pin"
            )
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in pins:
            raise ValueError(f"Duplicate locked distribution: {name}")
        pins[name] = match.group(2)
    if not pins:
        raise ValueError("Dependency lock contains no exact pins")
    return pins


def verify_installed_lock(path: str | Path) -> dict[str, object]:
    pins = load_exact_lock(path)
    installed: dict[str, str] = {}
    missing: list[str] = []
    mismatched: dict[str, dict[str, str]] = {}
    for name, expected in sorted(pins.items()):
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        installed[name] = observed
        if observed != expected:
            mismatched[name] = {"expected": expected, "observed": observed}
    return {
        "lock_path": str(Path(path)),
        "lock_sha256": sha256_file(path),
        "locked_distribution_count": len(pins),
        "installed_versions": installed,
        "missing": missing,
        "mismatched": mismatched,
        "valid": not missing and not mismatched,
    }


def _distribution_fingerprint(name: str) -> dict[str, str | None]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"version": None, "metadata_sha256": None, "record_sha256": None}
    metadata_hash = None
    record_hash = None
    for item in distribution.files or ():
        filename = str(item).replace("\\", "/")
        if filename.endswith(".dist-info/METADATA"):
            candidate = Path(distribution.locate_file(item))
            if candidate.exists():
                metadata_hash = sha256_file(candidate)
        elif filename.endswith(".dist-info/RECORD"):
            candidate = Path(distribution.locate_file(item))
            if candidate.exists():
                record_hash = sha256_file(candidate)
    return {
        "version": distribution.version,
        "metadata_sha256": metadata_hash,
        "record_sha256": record_hash,
    }


def _git_state(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip() if process.returncode == 0 else ""

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "dirty_path_count": len(status.splitlines()) if status else 0,
    }


def environment_fingerprint(
    root: str | Path,
    *,
    lock_path: str | Path = "requirements.lock",
) -> dict[str, object]:
    """Capture the deterministic CPU reference environment and lock state."""

    root_path = Path(root).resolve()
    lock = Path(lock_path)
    if not lock.is_absolute():
        lock = root_path / lock
    lock_audit = verify_installed_lock(lock)
    packages = {
        name: _distribution_fingerprint(name)
        for name in sorted(lock_audit["installed_versions"])
    }
    blas: object
    try:
        import numpy as np  # type: ignore

        blas = np.__config__.show(mode="dicts")
    except Exception as exc:  # pragma: no cover - platform dependent
        blas = {"status": "unavailable", "reason": type(exc).__name__}
    return {
        "schema_version": "cardia_x_environment_v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "container": {
            "image_digest": os.environ.get("CARDIA_X_CONTAINER_DIGEST"),
            "digest_recorded": bool(os.environ.get("CARDIA_X_CONTAINER_DIGEST")),
        },
        "determinism": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "locale": locale.getlocale(),
        "timezone": list(time.tzname),
        "blas": blas,
        "dependency_lock": lock_audit,
        "distribution_fingerprints": packages,
        "git": _git_state(root_path),
    }


def build_artifact_manifest(
    directory: str | Path,
    *,
    producer_command: str,
    input_hashes: Mapping[str, str] | None = None,
    code_root: str | Path | None = None,
    schema_version: str = "cardia_x_artifact_manifest_v1",
) -> dict[str, object]:
    """Hash every release artifact in a deterministic, self-excluding ledger."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Artifact directory does not exist: {root}")
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "artifact_manifest.json"
            or path.is_symlink()
        ):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "producer_command": producer_command,
                "input_hashes": dict(sorted((input_hashes or {}).items())),
                "schema_version": schema_version,
            }
        )
    git = _git_state(Path(code_root or Path.cwd()).resolve())
    return {
        "schema_version": schema_version,
        "artifact_root": str(root),
        "producer_command": producer_command,
        "input_hashes": dict(sorted((input_hashes or {}).items())),
        "code_commit": git["commit"],
        "dirty_state": git["dirty"],
        "artifact_count": len(files),
        "artifacts": files,
    }


def write_artifact_manifest(
    directory: str | Path,
    *,
    producer_command: str,
    input_hashes: Mapping[str, str] | None = None,
    code_root: str | Path | None = None,
) -> Path:
    root = Path(directory).resolve()
    manifest = build_artifact_manifest(
        root,
        producer_command=producer_command,
        input_hashes=input_hashes,
        code_root=code_root,
    )
    destination = root / "artifact_manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
