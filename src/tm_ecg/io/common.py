"""Shared file, serialization, and hashing utilities."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(items: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_parent(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv_rows(path: str | Path, rows: list[Mapping[str, object]]) -> None:
    destination = ensure_parent(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
