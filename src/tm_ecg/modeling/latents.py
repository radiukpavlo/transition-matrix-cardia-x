"""Latent pooling and training-only standardization helpers."""

from __future__ import annotations

from statistics import mean, pstdev


def trimmed_mean_pool(vectors: list[list[float]], trim_ratio: float = 0.1) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    result = []
    for column_idx in range(width):
        values = sorted(vector[column_idx] for vector in vectors)
        trim = int(len(values) * trim_ratio)
        trimmed = values[trim: len(values) - trim] if len(values) > 2 * trim else values
        result.append(mean(trimmed))
    return result


def max_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    return [max(vector[column_idx] for vector in vectors) for column_idx in range(len(vectors[0]))]


def zero_variance_mask(rows: list[list[float]]) -> list[bool]:
    if not rows:
        return []
    mask = []
    for column_idx in range(len(rows[0])):
        column = [row[column_idx] for row in rows]
        mask.append(len(set(column)) == 1)
    return mask


def fit_standardizer(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    means = []
    stds = []
    if not rows:
        return means, stds
    for column_idx in range(len(rows[0])):
        column = [row[column_idx] for row in rows]
        means.append(mean(column))
        stds.append(pstdev(column) if len(column) >= 2 else 1.0)
    return means, stds


def apply_standardizer(rows: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    output = []
    for row in rows:
        output.append(
            [
                (value - means[idx]) / (stds[idx] if stds[idx] > 0 else 1.0)
                for idx, value in enumerate(row)
            ]
        )
    return output
