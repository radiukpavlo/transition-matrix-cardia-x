"""Unified table reading and path resolution for all pipeline stages.

This module consolidates table I/O that was previously duplicated across
``stages.fit_transition``, ``stages.explain``, and ``stages.dss``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from tm_ecg.config import ProjectConfig


# ---------------------------------------------------------------------------
# Row-oriented table reading (list[dict])
# ---------------------------------------------------------------------------

def read_table_rows(path: Path) -> list[dict[str, object]]:
    """Read a CSV or Parquet table as a list of row dicts.

    CSV values are returned as strings (consistent with ``csv.DictReader``).
    Parquet values retain their native types.
    """
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(path)
    columns = {name: table[name].to_pylist() for name in table.column_names}
    row_count = len(next(iter(columns.values()), []))
    return [{name: columns[name][idx] for name in table.column_names} for idx in range(row_count)]


# ---------------------------------------------------------------------------
# DataFrame-oriented table reading
# ---------------------------------------------------------------------------

def read_table_frame(path: Path) -> "pd.DataFrame":
    """Read a CSV, Parquet, Excel, or JSON table as a ``pandas.DataFrame``."""
    import json

    import pandas as pd  # type: ignore[import-untyped]

    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows", payload.get("predictions", payload)) if isinstance(payload, dict) else payload
            return pd.DataFrame(rows)
    raise ValueError(f"Unsupported table extension: {path}")


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def find_table(directory: Path, stem: str, *, required: bool = False) -> Path | None:
    """Locate a table file by stem, preferring Parquet over CSV.

    When *required* is True, raise ``FileNotFoundError`` instead of
    returning ``None``.
    """
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Cannot locate table for {stem} in {directory}")
    return None


def dataset_a_path(config: ProjectConfig, dataset: str, split: str) -> Path | None:
    """Resolve the path for matrix A for a given dataset and split."""
    from tm_ecg.constants import DATASET_MAP

    dataset_key = DATASET_MAP.get(dataset, dataset)
    preferred = find_table(config.paths.latents, f"A_{dataset_key}_{split}")
    if preferred is not None:
        return preferred
    return find_table(config.paths.latents, f"A_{split}")
