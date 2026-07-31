"""Metrics, bootstrap, and report rendering helpers."""

from __future__ import annotations

import csv
import random
from statistics import mean
from pathlib import Path

from tm_ecg.io.common import ensure_parent


def bootstrap_mean_ci(values: list[float], replicates: int = 1000, seed: int = 17) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(mean(draw))
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return mean(values), lower, upper


def write_bootstrap_report(path: str | Path, rows: list[dict[str, object]]) -> Path:
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
    return destination


def write_metrics_markdown(path: str | Path, sections: dict[str, list[str]]) -> Path:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8") as handle:
        for heading, lines in sections.items():
            handle.write(f"## {heading}\n\n")
            for line in lines:
                handle.write(f"- {line}\n")
            handle.write("\n")
    return destination
