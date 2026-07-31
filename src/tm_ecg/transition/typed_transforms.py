"""Typed B-space transforms and inverse maps."""

from __future__ import annotations

import math
from statistics import mean, pstdev

from tm_ecg.features.registry import feature_types
from tm_ecg.types import TransformBundle, TransformColumnStats


def _clean(values: list[object]) -> list[float]:
    cleaned = []
    for value in values:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue
        try:
            cleaned.append(float(value))
        except (ValueError, TypeError):
            continue
    return cleaned


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _zscore(value: float, stats: TransformColumnStats) -> float:
    std = stats.std if stats.std > 0 else 1.0
    return (value - stats.mean) / std


def _inv_zscore(value: float, stats: TransformColumnStats) -> float:
    return value * (stats.std if stats.std > 0 else 1.0) + stats.mean


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def fit_transform_bundle(
    rows: list[dict[str, object]],
    fit_columns: list[str],
    eps: float = 1e-3,
) -> TransformBundle:
    stats: list[TransformColumnStats] = []
    value_types = feature_types()
    for column in fit_columns:
        family = value_types[column]
        values = _clean([row.get(column) for row in rows])
        if not values:
            continue
        lower = upper = None
        transformed = []
        if family == "continuous":
            lower = _percentile(values, 0.005)
            upper = _percentile(values, 0.995)
            transformed = [min(max(value, lower), upper) for value in values]
        elif family == "count":
            transformed = [math.log1p(value) for value in values]
        elif family in {"binary", "bounded"}:
            for value in values:
                clipped = min(max(value, eps), 1.0 - eps)
                transformed.append(_logit(clipped))
        else:
            transformed = list(values)
        stats.append(
            TransformColumnStats(
                column=column,
                family=family,
                mean=mean(transformed),
                std=pstdev(transformed) if len(transformed) >= 2 else 1.0,
                lower=lower,
                upper=upper,
                eps=eps,
            )
        )
    stat_map = {item.column: item for item in stats}
    return TransformBundle(
        dataset="unknown",
        stats=[stat_map[column] for column in fit_columns if column in stat_map],
        fit_columns=[column for column in fit_columns if column in stat_map],
        dropped_columns=[column for column in fit_columns if column not in stat_map],
    )


def transform_rows(rows: list[dict[str, object]], bundle: TransformBundle) -> list[dict[str, object]]:
    stat_map = {item.column: item for item in bundle.stats}
    transformed_rows: list[dict[str, object]] = []
    for row in rows:
        transformed: dict[str, object] = {"record_id": row.get("record_id")}
        for column in bundle.fit_columns:
            value = row.get(column)
            stats = stat_map[column]
            if value is None or (isinstance(value, str) and value.strip() == ""):
                transformed[column] = None
                continue
            try:
                numeric = float(value)
            except (ValueError, TypeError):
                transformed[column] = None
                continue

            if stats.family == "continuous":
                clipped = min(max(numeric, stats.lower if stats.lower is not None else numeric), stats.upper if stats.upper is not None else numeric)
                transformed[column] = _zscore(clipped, stats)
            elif stats.family == "count":
                transformed[column] = _zscore(math.log1p(numeric), stats)
            elif stats.family in {"binary", "bounded"}:
                clipped = min(max(numeric, stats.eps), 1.0 - stats.eps)
                transformed[column] = _zscore(_logit(clipped), stats)
            else:
                transformed[column] = _zscore(numeric, stats)
        transformed_rows.append(transformed)
    return transformed_rows


def inverse_rows(rows: list[dict[str, object]], bundle: TransformBundle) -> list[dict[str, object]]:
    stat_map = {item.column: item for item in bundle.stats}
    inversed: list[dict[str, object]] = []
    for row in rows:
        restored: dict[str, object] = {"record_id": row.get("record_id")}
        for column in bundle.fit_columns:
            value = row.get(column)
            stats = stat_map[column]
            if value is None or (isinstance(value, str) and value.strip() == ""):
                restored[column] = None
                continue
            try:
                raw = _inv_zscore(float(value), stats)
            except (ValueError, TypeError):
                restored[column] = None
                continue
            if stats.family == "count":
                restored[column] = math.exp(raw) - 1.0
            elif stats.family in {"binary", "bounded"}:
                restored[column] = _logistic(raw)
            else:
                restored[column] = raw
        inversed.append(restored)
    return inversed
