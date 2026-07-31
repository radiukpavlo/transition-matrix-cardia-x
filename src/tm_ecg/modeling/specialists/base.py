"""Shared specialist output and calibrated linear-head helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SpecialistOutput:
    specialist_id: str
    probabilities: dict[str, float | None]
    eligible: bool
    quality_flags: tuple[str, ...]
    features: dict[str, float | None]
    model_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def safe_probability(value: object) -> float:
    import math

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Specialist probability must be finite")
    return min(max(numeric, 0.0), 1.0)


def logistic_score(
    features: Mapping[str, float | None],
    coefficients: Mapping[str, float],
    *,
    intercept: float = 0.0,
) -> float:
    import math

    score = float(intercept)
    for name, coefficient in coefficients.items():
        value = features.get(name)
        if value is None:
            continue
        score += float(coefficient) * float(value)
    score = min(max(score, -40.0), 40.0)
    return 1.0 / (1.0 + math.exp(-score))


def robust_summary(values: Sequence[float]) -> dict[str, float | None]:
    import numpy as np  # type: ignore

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "median": None,
            "iqr": None,
            "trimmed_mean": None,
            "missing_fraction": 1.0,
        }
    lower, upper = np.quantile(array, [0.1, 0.9])
    trimmed = array[(array >= lower) & (array <= upper)]
    return {
        "median": float(np.median(array)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "trimmed_mean": float(trimmed.mean() if len(trimmed) else array.mean()),
        "missing_fraction": 0.0,
    }
