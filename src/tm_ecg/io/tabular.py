"""Tabular output helpers with optional Parquet support."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from tm_ecg.io.common import ensure_parent


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> Path:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_records_table(path: str | Path, rows: list[Mapping[str, object]]) -> Path:
    destination = ensure_parent(path)
    if destination.suffix.lower() == ".parquet":
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:
            fallback = destination.with_suffix(".csv")
            return _write_csv(fallback, rows)
        table = pa.Table.from_pylist(list(rows))
        pq.write_table(table, destination)
        return destination
    return _write_csv(destination, rows)
