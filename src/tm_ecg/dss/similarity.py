"""Predicate similarity metrics for comparing DSS rules with physician predicates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from tm_ecg.dss.models import MedicalPredicate, RuleCondition


def rule_to_medical_predicate(rule: object) -> MedicalPredicate:
    """Represent a production rule as a generated predicate.

    Rule antecedents are interpreted as required conditions because a production
    rule fires only when every antecedent is satisfied. This enables direct
    comparison with a physician predicate using the same mathematical objective.
    The function deliberately accepts ``object`` to avoid a runtime import cycle;
    it requires the production-rule attributes used below.
    """

    return MedicalPredicate(
        label=getattr(rule, "target_label"),
        source_labels={"generated_rule": [getattr(rule, "rule_id")]},
        required=list(getattr(rule, "antecedents")),
        supportive=[],
        contraindications=[],
        references=["Generated from rough-set lower approximation over matrix B."],
        explanation="Generated production-rule predicate.",
    )


def max_predicate_similarity(
    generated: MedicalPredicate,
    physician_predicates: Mapping[str, MedicalPredicate],
) -> tuple[str, dict[str, float]]:
    """Return the physician predicate with maximum similarity to ``generated``."""

    if not physician_predicates:
        return generated.label, {"score": 0.0}
    scored = [
        (label, predicate_similarity(generated, predicate))
        for label, predicate in physician_predicates.items()
    ]
    scored.sort(key=lambda item: (-item[1].get("score", 0.0), item[0]))
    return scored[0]


@dataclass(slots=True)
class SimilarityWeights:
    feature_overlap: float = 0.22
    interval: float = 0.16
    logical_structure: float = 0.14
    semantic: float = 0.16
    coverage: float = 0.12
    decision_agreement: float = 0.14
    safety_penalty: float = 0.06

    def normalized_positive_sum(self) -> float:
        return (
            self.feature_overlap
            + self.interval
            + self.logical_structure
            + self.semantic
            + self.coverage
            + self.decision_agreement
        )


def _all_conditions(predicate: MedicalPredicate) -> list[RuleCondition]:
    return predicate.required + predicate.supportive + predicate.contraindications


def _feature_weights(*predicates: MedicalPredicate) -> dict[str, float]:
    weights: dict[str, float] = {}
    for predicate in predicates:
        for condition in _all_conditions(predicate):
            weights[condition.feature] = max(weights.get(condition.feature, 0.0), condition.criticality)
    return weights


def weighted_feature_jaccard(
    left: MedicalPredicate,
    right: MedicalPredicate,
    weights: Mapping[str, float] | None = None,
) -> float:
    left_features = left.features()
    right_features = right.features()
    all_features = left_features | right_features
    if not all_features:
        return 1.0
    feature_weights = dict(weights or _feature_weights(left, right))
    intersection = sum(feature_weights.get(feature, 1.0) for feature in left_features & right_features)
    union = sum(feature_weights.get(feature, 1.0) for feature in all_features)
    return intersection / union if union else 1.0


def _condition_interval_similarity(left: RuleCondition, right: RuleCondition) -> float:
    if left.feature != right.feature:
        return 0.0
    if left.state is not None or right.state is not None:
        return 1.0 if left.state == right.state and left.negated == right.negated else 0.0
    if left.operator == right.operator and left.value == right.value and left.lower == right.lower and left.upper == right.upper:
        return 1.0
    l_low = left.lower if left.lower is not None else float("-inf")
    l_up = left.upper if left.upper is not None else float("inf")
    r_low = right.lower if right.lower is not None else float("-inf")
    r_up = right.upper if right.upper is not None else float("inf")
    if l_low == float("-inf") or r_low == float("-inf") or l_up == float("inf") or r_up == float("inf"):
        return 0.5 if left.operator == right.operator else 0.25
    overlap = max(0.0, min(l_up, r_up) - max(l_low, r_low))
    union = max(l_up, r_up) - min(l_low, r_low)
    return overlap / union if union > 0 else 0.0


def interval_similarity(left: MedicalPredicate, right: MedicalPredicate) -> float:
    left_conditions = _all_conditions(left)
    right_conditions = _all_conditions(right)
    if not left_conditions and not right_conditions:
        return 1.0
    if not left_conditions or not right_conditions:
        return 0.0
    matched: list[float] = []
    for condition in left_conditions:
        best = max((_condition_interval_similarity(condition, other) for other in right_conditions), default=0.0)
        matched.append(best)
    reverse: list[float] = []
    for condition in right_conditions:
        best = max((_condition_interval_similarity(condition, other) for other in left_conditions), default=0.0)
        reverse.append(best)
    return (sum(matched) / len(matched) + sum(reverse) / len(reverse)) / 2.0


def logical_structure_similarity(left: MedicalPredicate, right: MedicalPredicate) -> float:
    left_counts = (len(left.required), len(left.supportive), len(left.contraindications))
    right_counts = (len(right.required), len(right.supportive), len(right.contraindications))
    parts = []
    for a, b in zip(left_counts, right_counts, strict=False):
        parts.append(1.0 - abs(a - b) / max(a, b, 1))
    neg_left = {condition.feature for condition in left.contraindications}
    neg_right = {condition.feature for condition in right.contraindications}
    if neg_left or neg_right:
        negation_score = len(neg_left & neg_right) / len(neg_left | neg_right)
    else:
        negation_score = 1.0
    return max(0.0, min(1.0, 0.75 * (sum(parts) / len(parts)) + 0.25 * negation_score))


def semantic_similarity(left: MedicalPredicate, right: MedicalPredicate) -> float:
    left_conditions = _all_conditions(left)
    right_conditions = _all_conditions(right)
    if not left_conditions and not right_conditions:
        return 1.0
    feature_weights = _feature_weights(left, right)
    total = 0.0
    matched = 0.0
    right_concepts = {(condition.feature, condition.clinical_concept) for condition in right_conditions}
    right_features = {condition.feature for condition in right_conditions}
    for condition in left_conditions:
        weight = feature_weights.get(condition.feature, condition.criticality)
        total += weight
        if (condition.feature, condition.clinical_concept) in right_concepts:
            matched += weight
        elif condition.feature in right_features:
            matched += 0.70 * weight
    return matched / total if total else 1.0


def coverage_similarity(
    left_object_ids: Iterable[str] | None,
    right_object_ids: Iterable[str] | None,
) -> float:
    if left_object_ids is None or right_object_ids is None:
        return 1.0
    left_set = set(left_object_ids)
    right_set = set(right_object_ids)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def cohen_kappa(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    from tm_ecg.clinical_validation.metrics import compute_cohen_kappa

    result = compute_cohen_kappa(y_true, y_pred)
    if result.kappa is None:
        raise ValueError(f"Cohen's kappa is {result.status}: {result.reason}")
    return result.kappa


def decision_agreement(
    y_true: Sequence[str] | None,
    y_pred: Sequence[str] | None,
    same_label: bool | None = None,
) -> float:
    if y_true is not None and y_pred is not None:
        if len(y_true) != len(y_pred):
            raise ValueError("label sequences must be aligned")
        if not y_true:
            return 1.0
        accuracy = sum(1 for a, b in zip(y_true, y_pred, strict=False) if a == b) / len(y_true)
        kappa = cohen_kappa(y_true, y_pred)
        return max(0.0, min(1.0, 0.5 * accuracy + 0.5 * ((kappa + 1.0) / 2.0)))
    if same_label is None:
        return 1.0
    return 1.0 if same_label else 0.0


def safety_penalty(generated: MedicalPredicate, physician: MedicalPredicate) -> float:
    generated_features = {condition.feature for condition in generated.contraindications}
    missed = [
        condition
        for condition in physician.contraindications
        if condition.feature not in generated_features and condition.criticality >= 1.0
    ]
    if not physician.contraindications:
        return 0.0
    return min(1.0, sum(condition.criticality for condition in missed) / sum(
        max(condition.criticality, 0.1) for condition in physician.contraindications
    ))


def predicate_similarity(
    generated: MedicalPredicate,
    physician: MedicalPredicate,
    generated_coverage: Iterable[str] | None = None,
    physician_coverage: Iterable[str] | None = None,
    y_true: Sequence[str] | None = None,
    y_pred: Sequence[str] | None = None,
    weights: SimilarityWeights | None = None,
) -> dict[str, float]:
    """Compute maximum-similarity components and weighted combined score."""

    w = weights or SimilarityWeights()
    feature = weighted_feature_jaccard(generated, physician)
    intervals = interval_similarity(generated, physician)
    logic = logical_structure_similarity(generated, physician)
    semantics = semantic_similarity(generated, physician)
    coverage = coverage_similarity(generated_coverage, physician_coverage)
    agreement = decision_agreement(y_true, y_pred, generated.label == physician.label)
    penalty = safety_penalty(generated, physician)
    positive = (
        w.feature_overlap * feature
        + w.interval * intervals
        + w.logical_structure * logic
        + w.semantic * semantics
        + w.coverage * coverage
        + w.decision_agreement * agreement
    ) / max(w.normalized_positive_sum(), 1e-12)
    score = max(0.0, min(1.0, positive - w.safety_penalty * penalty))
    return {
        "score": score,
        "feature_overlap": feature,
        "interval_similarity": intervals,
        "logical_structure_similarity": logic,
        "semantic_similarity": semantics,
        "coverage_similarity": coverage,
        "decision_agreement": agreement,
        "safety_penalty": penalty,
    }
