"""Dataset ingestion and source manifest generation."""

from __future__ import annotations

import subprocess
from pathlib import Path
import zipfile

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import sha256_file, utc_now_iso, write_json
from tm_ecg.stages.shared import dataset_root, write_stage_manifest
from tm_ecg.types import SourceArchive, SourceManifest


DOWNLOAD_URLS = {
    "ptbxl": "https://physionet.org/files/ptb-xl/1.0.3/",
    "ptbxl_plus": "https://physionet.org/files/ptb-xl-plus/1.0.1/",
    "ludb": "https://physionet.org/files/ludb/1.0.1/",
}


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as handle:
        handle.extractall(destination)


def _download_dataset(url: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-r", "-N", "-c", "-np", url],
        cwd=destination,
        check=True,
    )


def _manifest_for_dataset(config: ProjectConfig, dataset_key: str, source: str) -> SourceArchive:
    dataset = config.datasets[dataset_key]
    archive_path = config.paths.data_lock / dataset.archive
    extracted_dir = dataset_root(config, dataset_key)

    if source == "zip":
        if not archive_path.exists():
            raise FileNotFoundError(f"Locked archive is missing: {archive_path}")
        
        # Skip extraction if the directory already exists and has files
        if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
            _extract_archive(archive_path, extracted_dir)
        archive_hash = sha256_file(archive_path)
        archive_ref = archive_path
    else:
        _download_dataset(DOWNLOAD_URLS[dataset_key], extracted_dir)
        archive_hash = "download-mode"
        archive_ref = extracted_dir

    return SourceArchive(
        dataset=dataset_key,
        archive_path=str(archive_ref),
        sha256=archive_hash,
        extracted_dir=str(extracted_dir),
        extracted_at_utc=utc_now_iso(),
        archive_version=dataset.version,
        verified_against_manifest=False,
        source_mode=source,
    )


def run(config: ProjectConfig, args: object) -> int:
    source = getattr(args, "source", "zip")
    archives = [
        _manifest_for_dataset(config, dataset_key, source)
        for dataset_key in ("ptbxl", "ptbxl_plus", "ludb")
    ]
    manifest = SourceManifest(
        generated_at_utc=utc_now_iso(),
        archives=archives,
        notes=[
            "Archive checksums were computed locally.",
            "Top-level DATA_LOCK_SHA256SUMS.txt was not bundled in this repository.",
        ],
    )
    output = config.paths.manifests / "source_manifest.json"
    write_json(output, manifest.to_dict())
    write_stage_manifest(
        config,
        "ingest",
        {
            "source": source,
            "source_manifest": str(output),
            "archives": [archive.to_dict() for archive in archives],
        },
    )
    print(f"Ingest manifest written to {output}")
    return 0
