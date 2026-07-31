"""Cluster-aware bootstrap confidence intervals for agreement estimates."""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Sequence

from tm_ecg.clinical_validation.metrics import compute_cohen_kappa
from tm_ecg.clinical_validation.models import KappaResult


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_kappa(
    reference: Sequence[str],
    comparison: Sequence[str],
    case_ids: Sequence[str],
    cluster_ids: Sequence[str],
    *,
    replicates: int = 1000,
    seed: int = 17,
    threshold: float = 0.70,
) -> tuple[KappaResult, list[float]]:
    if not (len(reference) == len(comparison) == len(case_ids) == len(cluster_ids)):
        raise ValueError("Bootstrap inputs must be aligned")
    point = compute_cohen_kappa(reference, comparison, case_ids=case_ids, threshold=threshold)
    if point.status != "ok" or replicates <= 0:
        return point, []
    grouped: dict[str, list[int]] = {}
    for index, cluster in enumerate(cluster_ids):
        grouped.setdefault(str(cluster), []).append(index)
    clusters = sorted(grouped)
    rng = random.Random(seed)
    estimates: list[float] = []
    failed = 0
    for _ in range(replicates):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        indices = [index for cluster in sampled for index in grouped[cluster]]
        boot = compute_cohen_kappa(
            [reference[index] for index in indices],
            [comparison[index] for index in indices],
            case_ids=[case_ids[index] for index in indices],
            threshold=threshold,
        )
        if boot.kappa is None:
            failed += 1
        else:
            estimates.append(boot.kappa)
    interval = (
        (_percentile(estimates, 0.025), _percentile(estimates, 0.975))
        if estimates
        else (None, None)
    )
    return (
        replace(
            point,
            confidence_interval=interval,
            bootstrap_replicates=replicates,
            bootstrap_failed_replicates=failed,
        ),
        estimates,
    )

