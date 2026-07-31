"""Release-environment and artifact-manifest controls."""

from __future__ import annotations

import argparse
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
PUBLIC_MANIFEST_PATTERN = re.compile(r"^(?P<sha256>[0-9a-f]{64})  (?P<path>.+)$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_manifest_path(root: Path, manifest_path: str | Path) -> tuple[Path, str]:
    """Resolve a public manifest and return its repository-relative POSIX path."""

    candidate = Path(manifest_path)
    resolved = candidate if candidate.is_absolute() else root / candidate
    resolved = resolved.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Public manifest must be located inside the repository root") from exc
    return resolved, relative


def _git_public_files(root: Path) -> list[str]:
    """Return tracked and non-ignored public paths using POSIX separators."""

    try:
        process = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError("Git is required to inspect the public repository manifest") from exc
    if process.returncode != 0:
        detail = process.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Could not list Git-tracked files{': ' + detail if detail else ''}")
    return sorted(
        item.decode("utf-8")
        for item in process.stdout.split(b"\0")
        if item
    )


def load_public_manifest(path: str | Path) -> dict[str, str]:
    """Load a deterministic ``sha256  relative/path`` public manifest."""

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        match = PUBLIC_MANIFEST_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"Invalid public manifest line {line_number}")
        relative_path = match.group("path").replace("\\", "/")
        if relative_path in entries:
            raise ValueError(f"Duplicate public manifest path: {relative_path}")
        entries[relative_path] = match.group("sha256")
    return entries


def write_public_manifest(
    root: str | Path,
    *,
    manifest_path: str | Path = "MANIFEST.sha256",
) -> Path:
    """Write hashes for every tracked or non-ignored public file."""

    root_path = Path(root).resolve()
    destination, relative_manifest = _resolved_manifest_path(root_path, manifest_path)
    entries: list[tuple[str, str]] = []
    for relative_path in _git_public_files(root_path):
        if relative_path == relative_manifest:
            continue
        source = root_path / Path(relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"Tracked public file does not exist: {relative_path}")
        entries.append((relative_path, sha256_file(source)))
    destination.write_text(
        "".join(f"{digest}  {relative_path}\n" for relative_path, digest in entries),
        encoding="utf-8",
    )
    return destination


def verify_public_manifest(
    root: str | Path,
    *,
    manifest_path: str | Path = "MANIFEST.sha256",
) -> dict[str, object]:
    """Verify public tracked/non-ignored files against the repository manifest."""

    root_path = Path(root).resolve()
    destination, relative_manifest = _resolved_manifest_path(root_path, manifest_path)
    entries = load_public_manifest(destination)
    public_files = {
        relative_path
        for relative_path in _git_public_files(root_path)
        if relative_path != relative_manifest
    }
    manifest_paths = set(entries)
    missing_from_manifest = sorted(public_files - manifest_paths)
    manifest_only = sorted(manifest_paths - public_files)
    hash_mismatches: list[str] = []
    for relative_path in sorted(public_files & manifest_paths):
        source = root_path / Path(relative_path)
        if not source.is_file() or sha256_file(source) != entries[relative_path]:
            hash_mismatches.append(relative_path)
    return {
        "manifest_path": relative_manifest,
        "public_file_count": len(public_files),
        "manifest_entry_count": len(entries),
        "missing_from_manifest": missing_from_manifest,
        "manifest_only": manifest_only,
        "hash_mismatches": hash_mismatches,
        "valid": not (missing_from_manifest or manifest_only or hash_mismatches),
    }


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
            candidate = Path(str(distribution.locate_file(item)))
            if candidate.exists():
                metadata_hash = sha256_file(candidate)
        elif filename.endswith(".dist-info/RECORD"):
            candidate = Path(str(distribution.locate_file(item)))
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
    installed_versions = lock_audit["installed_versions"]
    if not isinstance(installed_versions, dict):
        raise RuntimeError("Dependency-lock audit returned invalid installed-version data")
    packages = {
        name: _distribution_fingerprint(name)
        for name in sorted(str(name) for name in installed_versions)
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


def main(argv: list[str] | None = None) -> int:
    """Verify or regenerate the public repository manifest."""

    parser = argparse.ArgumentParser(prog="tm-ecg-verify-manifest")
    parser.add_argument("--root", default=".", help="Repository root (default: current directory)")
    parser.add_argument(
        "--manifest",
        default="MANIFEST.sha256",
        help="Manifest path relative to --root",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Regenerate the manifest instead of verifying it",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        if args.write_manifest:
            destination = write_public_manifest(root, manifest_path=args.manifest)
            print(f"Wrote public manifest: {destination}")
            return 0
        result = verify_public_manifest(root, manifest_path=args.manifest)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Public manifest verification failed: {exc}")
        return 1

    if result["valid"]:
        print(
            "Verified public manifest: "
            f"{result['public_file_count']} public files and "
            f"{result['manifest_entry_count']} manifest entries."
        )
        return 0

    print("Public manifest verification failed.")
    for key in ("missing_from_manifest", "manifest_only", "hash_mismatches"):
        values = result[key]
        if isinstance(values, list) and values:
            print(f"  {key}: {', '.join(str(value) for value in values)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
