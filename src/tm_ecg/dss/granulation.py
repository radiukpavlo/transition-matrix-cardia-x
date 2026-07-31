"""Rough-set granulation utilities for discretized ECG feature matrices."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

from tm_ecg.dss.models import InformationGranule


def signature_for_row(
    row: Mapping[str, object],
    attributes: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return the indiscernibility signature for one discretized B row."""

    selected = attributes or sorted(row.keys())
    return tuple((attribute, str(row.get(attribute, "missing"))) for attribute in selected)


def build_granules(
    information_function: Mapping[str, Mapping[str, object]],
    decisions: Mapping[str, str],
    attributes: Sequence[str] | None = None,
) -> list[InformationGranule]:
    """Group rows with identical signatures into information granules."""

    groups: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for object_id, row in information_function.items():
        groups[signature_for_row(row, attributes)].append(object_id)

    granules: list[InformationGranule] = []
    for signature, object_ids in groups.items():
        distribution = Counter(decisions[object_id] for object_id in object_ids)
        majority_label, majority_count = distribution.most_common(1)[0]
        total = sum(distribution.values()) or 1
        deterministic = len(distribution) == 1
        granules.append(
            InformationGranule(
                signature=signature,
                object_ids=sorted(object_ids),
                class_distribution=dict(sorted(distribution.items())),
                deterministic=deterministic,
                majority_label=majority_label,
                confidence=majority_count / total,
                boundary_region=not deterministic,
            )
        )
    granules.sort(key=lambda item: (-len(item.object_ids), item.majority_label, item.signature))
    return granules


def weighted_hamming_distance(
    row_a: Mapping[str, object],
    row_b: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
    attributes: Sequence[str] | None = None,
) -> float:
    """Return normalized weighted Hamming distance in [0, 1]."""

    selected = list(attributes or sorted(set(row_a) | set(row_b)))
    if not selected:
        return 0.0
    total = 0.0
    different = 0.0
    for attribute in selected:
        weight = float(weights.get(attribute, 1.0)) if weights else 1.0
        total += max(weight, 0.0)
        if str(row_a.get(attribute, "missing")) != str(row_b.get(attribute, "missing")):
            different += max(weight, 0.0)
    if total <= 0:
        return 0.0
    return min(max(different / total, 0.0), 1.0)


def weighted_hamming_similarity(
    row_a: Mapping[str, object],
    row_b: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
    attributes: Sequence[str] | None = None,
) -> float:
    """Return normalized weighted Hamming similarity in [0, 1]."""

    return 1.0 - weighted_hamming_distance(row_a, row_b, weights, attributes)


def soft_neighbors(
    query_row: Mapping[str, object],
    information_function: Mapping[str, Mapping[str, object]],
    max_distance: float = 0.25,
    weights: Mapping[str, float] | None = None,
    attributes: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    """Return near-indiscernible rows using weighted Hamming similarity."""

    neighbors: list[tuple[str, float]] = []
    for object_id, row in information_function.items():
        distance = weighted_hamming_distance(query_row, row, weights, attributes)
        if distance <= max_distance:
            neighbors.append((object_id, 1.0 - distance))
    neighbors.sort(key=lambda item: (-item[1], item[0]))
    return neighbors


def lower_approximation(
    granules: Sequence[InformationGranule],
    target_label: str,
    min_support: int = 1,
) -> list[InformationGranule]:
    """Return deterministic granules wholly contained in one decision class."""

    return [
        granule
        for granule in granules
        if granule.deterministic
        and granule.majority_label == target_label
        and len(granule.object_ids) >= min_support
    ]


def granule_signature_dict(granule: InformationGranule) -> dict[str, str]:
    """Convert a signature tuple back into a feature-state dictionary."""

    return {feature: state for feature, state in granule.signature}
