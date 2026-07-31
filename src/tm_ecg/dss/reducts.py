"""Discernibility matrices and reduct search for ECG production rules."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

from tm_ecg.dss.models import ProductionRule, RuleCondition


def discernibility_sets(
    information_function: Mapping[str, Mapping[str, object]],
    decisions: Mapping[str, str],
    attributes: Sequence[str] | None = None,
) -> list[frozenset[str]]:
    """Build discernibility sets for object pairs with different decisions."""

    object_ids = sorted(information_function)
    selected = list(attributes or sorted({key for row in information_function.values() for key in row}))
    result: list[frozenset[str]] = []
    for idx, left_id in enumerate(object_ids):
        left_row = information_function[left_id]
        for right_id in object_ids[idx + 1 :]:
            if decisions[left_id] == decisions[right_id]:
                continue
            right_row = information_function[right_id]
            differing = frozenset(
                attr
                for attr in selected
                if str(left_row.get(attr, "missing")) != str(right_row.get(attr, "missing"))
            )
            if differing:
                result.append(differing)
    # Deduplicate while preserving deterministic sort order.
    return sorted(set(result), key=lambda item: (len(item), sorted(item)))


def exact_reduct(
    attributes: Sequence[str],
    discernibility: Sequence[frozenset[str]],
    max_attributes: int = 14,
) -> set[str]:
    """Find a minimum-cardinality hitting set for small discernibility problems."""

    selected = sorted(set(attributes))
    if not discernibility:
        return set()
    if len(selected) > max_attributes:
        raise ValueError("Exact reduct search is restricted to small attribute sets")
    for size in range(1, len(selected) + 1):
        for candidate in combinations(selected, size):
            candidate_set = set(candidate)
            if all(candidate_set & set(diff) for diff in discernibility):
                return candidate_set
    return set(selected)


def greedy_reduct(
    attributes: Sequence[str],
    discernibility: Sequence[frozenset[str]],
    weights: Mapping[str, float] | None = None,
) -> set[str]:
    """Johnson-style greedy hitting-set reduct with clinical/quality weights."""

    uncovered = [set(item) for item in discernibility if item]
    remaining = set(attributes)
    reduct: set[str] = set()
    while uncovered and remaining:
        best_attr: str | None = None
        best_score = -1.0
        for attr in sorted(remaining):
            hit_count = sum(1 for diff in uncovered if attr in diff)
            if not hit_count:
                continue
            priority = float(weights.get(attr, 1.0)) if weights else 1.0
            score = hit_count * max(priority, 0.1)
            if score > best_score or (score == best_score and (best_attr is None or attr < best_attr)):
                best_score = score
                best_attr = attr
        if best_attr is None:
            break
        reduct.add(best_attr)
        remaining.remove(best_attr)
        uncovered = [diff for diff in uncovered if best_attr not in diff]
    return reduct


def _rows_matching_conditions(
    conditions: Sequence[RuleCondition],
    information_function: Mapping[str, Mapping[str, object]],
) -> list[str]:
    matching: list[str] = []
    for object_id, row in information_function.items():
        if all(condition.matches_discrete(row) for condition in conditions):
            matching.append(object_id)
    return matching


def minimize_rule_conditions(
    rule: ProductionRule,
    information_function: Mapping[str, Mapping[str, object]],
    decisions: Mapping[str, str],
    min_support: int = 1,
) -> ProductionRule:
    """Remove antecedents while preserving deterministic coverage for the target label."""

    original_count = len(rule.antecedents)
    conditions = list(rule.antecedents)
    # Try to remove the lowest-criticality conditions first.
    for condition in sorted(rule.antecedents, key=lambda item: (item.criticality, item.feature)):
        trial = [item for item in conditions if item is not condition]
        if not trial:
            continue
        covered = _rows_matching_conditions(trial, information_function)
        if len(covered) < min_support:
            continue
        if all(decisions.get(object_id) == rule.target_label for object_id in covered):
            conditions = trial
    covered = _rows_matching_conditions(conditions, information_function)
    distribution: dict[str, int] = {}
    for object_id in covered:
        label = decisions.get(object_id, "unknown")
        distribution[label] = distribution.get(label, 0) + 1
    target_count = distribution.get(rule.target_label, 0)
    confidence = target_count / max(len(covered), 1)
    return ProductionRule(
        rule_id=rule.rule_id,
        target_label=rule.target_label,
        antecedents=conditions,
        confidence=confidence,
        support_count=target_count,
        support_fraction=target_count / max(len(decisions), 1),
        covered_object_ids=covered,
        class_distribution=dict(sorted(distribution.items())),
        threshold_provenance=rule.threshold_provenance,
        source=rule.source,
        reduced_from_conditions=original_count - len(conditions),
        notes=rule.notes,
        physician_predicate_label=rule.physician_predicate_label,
        predicate_similarity=dict(rule.predicate_similarity),
    )
