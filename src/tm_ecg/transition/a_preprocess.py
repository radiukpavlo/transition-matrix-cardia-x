"""Training-only preprocessing for matrix A before transition fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tm_ecg.io.common import read_json, write_json


@dataclass(slots=True)
class APreprocessBundle:
    dataset: str
    original_columns: list[str]
    kept_columns: list[str]
    dropped_zero_variance_columns: list[str]
    means: list[float]
    stds: list[float]
    components: list[list[float]]
    explained_variance_ratio: list[float]
    variance_retained: float
    rank_cap: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _a_columns(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []
    return [
        column
        for column in rows[0]
        if column not in {"record_id", "split", "label", "labels", "triad_count"}
    ]


def _matrix(rows: list[dict[str, object]], columns: list[str]):
    import numpy as np  # type: ignore

    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


def fit_a_preprocess_bundle(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    variance_retained: float = 0.99,
    rank_cap: int = 512,
) -> tuple[APreprocessBundle, list[dict[str, object]]]:
    """Fit zero-variance removal, train-only standardization, and PCA."""

    import numpy as np  # type: ignore

    columns = _a_columns(rows)
    if not columns:
        raise ValueError("Cannot fit A preprocessing without latent columns")
    raw = _matrix(rows, columns)
    variances = raw.var(axis=0)
    keep_mask = variances > 0.0
    kept_columns = [column for column, keep in zip(columns, keep_mask, strict=False) if bool(keep)]
    dropped = [column for column, keep in zip(columns, keep_mask, strict=False) if not bool(keep)]
    if not kept_columns:
        raise ValueError("All A columns have zero variance")
    kept = raw[:, keep_mask]
    means = kept.mean(axis=0)
    stds = kept.std(axis=0)
    stds[stds == 0.0] = 1.0
    standardized = (kept - means) / stds

    _u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    total = float(np.sum(singular_values**2))
    ratios = (singular_values**2 / total).tolist() if total > 0 else [1.0]
    cumulative = np.cumsum(ratios)
    rank = int(np.searchsorted(cumulative, variance_retained, side="left") + 1)
    rank = max(1, min(rank, rank_cap, vt.shape[0]))
    components = vt[:rank, :]
    reduced = standardized @ components.T
    bundle = APreprocessBundle(
        dataset=dataset,
        original_columns=columns,
        kept_columns=kept_columns,
        dropped_zero_variance_columns=dropped,
        means=means.astype(float).tolist(),
        stds=stds.astype(float).tolist(),
        components=components.astype(float).tolist(),
        explained_variance_ratio=[float(value) for value in ratios[:rank]],
        variance_retained=variance_retained,
        rank_cap=rank_cap,
    )
    return bundle, reduced_rows(rows, reduced)


def apply_a_preprocess_bundle(
    rows: list[dict[str, object]],
    bundle: APreprocessBundle,
) -> list[dict[str, object]]:
    import numpy as np  # type: ignore

    raw = _matrix(rows, bundle.kept_columns)
    means = np.asarray(bundle.means, dtype=float)
    stds = np.asarray(bundle.stds, dtype=float)
    components = np.asarray(bundle.components, dtype=float)
    standardized = (raw - means) / stds
    reduced = standardized @ components.T
    return reduced_rows(rows, reduced)


def reduced_rows(rows: list[dict[str, object]], reduced) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    output: list[dict[str, object]] = []
    for source, values in zip(rows, reduced, strict=False):
        row: dict[str, object] = {"record_id": source.get("record_id"), "split": source.get("split")}
        for idx, value in enumerate(values):
            row[f"a_red_{idx:04d}"] = float(value)
        output.append(row)
    return output


def write_a_preprocess_bundle(path: str | Path, bundle: APreprocessBundle) -> None:
    write_json(path, bundle.to_dict())


def read_a_preprocess_bundle(path: str | Path) -> APreprocessBundle:
    payload = read_json(path)
    return APreprocessBundle(
        dataset=str(payload["dataset"]),
        original_columns=list(payload["original_columns"]),
        kept_columns=list(payload["kept_columns"]),
        dropped_zero_variance_columns=list(payload["dropped_zero_variance_columns"]),
        means=[float(value) for value in payload["means"]],
        stds=[float(value) for value in payload["stds"]],
        components=[[float(value) for value in row] for row in payload["components"]],
        explained_variance_ratio=[float(value) for value in payload["explained_variance_ratio"]],
        variance_retained=float(payload["variance_retained"]),
        rank_cap=int(payload["rank_cap"]),
    )
