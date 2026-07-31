"""Rule induction and weighted inference for the ECG DSS."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Mapping, Sequence

from tm_ecg.dss.discretization import build_decision_table, quantize_row
from tm_ecg.dss.granulation import build_granules, granule_signature_dict
from tm_ecg.dss.models import (
    DecisionSystem,
    DiscretizationPlan,
    InferenceResult,
    InformationGranule,
    MedicalPredicate,
    ProductionRule,
    RuleCondition,
)
from tm_ecg.types import TypedAbstention
from tm_ecg.dss.reducts import minimize_rule_conditions
from tm_ecg.dss.similarity import max_predicate_similarity, rule_to_medical_predicate


def build_decision_system(
    rows: list[Mapping[str, object]],
    labels: list[str],
    plan: DiscretizationPlan,
    object_ids: list[str] | None = None,
    decision_attribute: str = "arrhythmia_label",
    metadata: Mapping[str, object] | None = None,
) -> DecisionSystem:
    """Construct the formal DSS tuple (U, C, d, V, f)."""

    information, decisions = build_decision_table(rows, labels, plan, object_ids)
    universe = list(information)
    system = DecisionSystem(
        universe=universe,
        conditional_attributes=list(plan.feature_domains),
        decision_attribute=decision_attribute,
        attribute_domains=plan.feature_domains,
        information_function=information,
        decisions=decisions,
        metadata=dict(metadata or {}),
    )
    system.validate()
    return system


def _stable_rule_id(label: str, signature: tuple[tuple[str, str], ...]) -> str:
    digest = hashlib.sha1(repr((label, signature)).encode("utf-8")).hexdigest()[:12]
    safe = label.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return f"rule_{safe}_{digest}"


def rule_from_granule(
    granule: InformationGranule,
    plan: DiscretizationPlan,
    universe_size: int,
) -> ProductionRule:
    """Convert one deterministic granule into a production rule."""

    signature = granule_signature_dict(granule)
    antecedents: list[RuleCondition] = []
    threshold_provenance: dict[str, str] = {}
    for feature, state in sorted(signature.items()):
        if state == "missing":
            continue
        domain = plan.feature_domains.get(feature)
        priority = domain.clinical_priority if domain else 1.0
        family = domain.family if domain else ""
        antecedents.append(
            RuleCondition(
                feature=feature,
                state=state,
                family=family,
                clinical_concept=feature,
                source=domain.provenance if domain else "unknown",
                criticality=priority,
            )
        )
        if domain is not None:
            threshold_provenance[feature] = domain.provenance
    return ProductionRule(
        rule_id=_stable_rule_id(granule.majority_label, granule.signature),
        target_label=granule.majority_label,
        antecedents=antecedents,
        confidence=1.0,
        support_count=len(granule.object_ids),
        support_fraction=len(granule.object_ids) / max(universe_size, 1),
        covered_object_ids=list(granule.object_ids),
        class_distribution=dict(granule.class_distribution),
        threshold_provenance=threshold_provenance,
        source="rough_set_lower_approximation",
        notes="Deterministic information granule from discretized matrix B.",
    )




def _negated_guard(condition: RuleCondition) -> RuleCondition:
    """Return a cloned condition that must *not* be present in a predicate rule."""

    return RuleCondition(
        feature=condition.feature,
        state=condition.state,
        operator=condition.operator,
        value=condition.value,
        lower=condition.lower,
        upper=condition.upper,
        family=condition.family,
        clinical_concept=f"not_{condition.clinical_concept or condition.feature}",
        source=condition.source,
        criticality=condition.criticality,
        negated=not condition.negated,
    )


def _rows_matching_rule_conditions(
    conditions: Sequence[RuleCondition],
    information_function: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Return rows satisfying a clinical template or induced antecedent list."""

    covered: list[str] = []
    for object_id, row in information_function.items():
        if all(condition.matches_discrete(row) for condition in conditions):
            covered.append(object_id)
    return covered


def _class_distribution_for_rows(
    object_ids: Sequence[str],
    decisions: Mapping[str, str],
) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for object_id in object_ids:
        label = decisions.get(object_id, "unknown")
        distribution[label] = distribution.get(label, 0) + 1
    return dict(sorted(distribution.items()))


def predicate_template_rule(
    predicate: MedicalPredicate,
    system: DecisionSystem,
    include_supportive: bool = False,
    include_contraindication_guards: bool = True,
) -> ProductionRule:
    """Convert an executable medical predicate into an auditable production rule.

    Rough-set lower approximations may be sparse when clinically meaningful B features
    overlap across labels. A predicate-template rule preserves the physician-style rule
    even when no deterministic lower-approximation granule exists for that label.

    The default template uses required conditions plus negated contraindication guards.
    Supportive conditions are intentionally not all conjoined by default because they are
    often alternative forms of evidence (for example, P waves can be absent or
    intermittent in atrial fibrillation). The predicate evaluator still uses supportive
    conditions for score calibration during inference.
    """

    antecedents: list[RuleCondition] = list(predicate.required)
    if include_supportive:
        antecedents.extend(predicate.supportive)
    if include_contraindication_guards:
        antecedents.extend(_negated_guard(condition) for condition in predicate.contraindications)
    signature = tuple((condition.feature, condition.state or condition.operator or "observed") for condition in antecedents)
    rule_id = _stable_rule_id(f"clinical_predicate_{predicate.label}", signature)
    covered = _rows_matching_rule_conditions(antecedents, system.information_function)
    distribution = _class_distribution_for_rows(covered, system.decisions)
    target_count = distribution.get(predicate.label, 0)
    confidence = target_count / len(covered) if covered else 0.0
    return ProductionRule(
        rule_id=rule_id,
        target_label=predicate.label,
        antecedents=antecedents,
        confidence=confidence,
        support_count=target_count,
        support_fraction=target_count / max(len(system.universe), 1),
        covered_object_ids=covered,
        class_distribution=distribution,
        threshold_provenance={condition.feature: condition.source for condition in antecedents},
        source="clinical_predicate_template",
        reduced_from_conditions=0,
        notes=(
            "Physician-facing predicate template grounded in the locked medical predicate library; "
            "empirical support/confidence were computed on the discretized matrix-B decision table; "
            "supportive predicate conditions remain available for scoring but are not all mandatory."
        ),
    )


def build_predicate_template_rules(
    predicates: Mapping[str, MedicalPredicate],
    system: DecisionSystem,
    allowed_labels: set[str] | None = None,
    include_weak_signature: bool = True,
) -> list[ProductionRule]:
    """Build one physician-predicate template rule per allowed project label."""

    rules: list[ProductionRule] = []
    for label, predicate in sorted(predicates.items()):
        if allowed_labels is not None and label not in allowed_labels:
            continue
        if predicate.weak_signature and not include_weak_signature:
            continue
        rules.append(predicate_template_rule(predicate, system))
    rules.sort(key=lambda item: (item.target_label, item.rule_id))
    return rules

def induce_rules(
    system: DecisionSystem,
    plan: DiscretizationPlan,
    min_support: int = 2,
    use_reducts: bool = True,
    allowed_labels: set[str] | None = None,
    max_rules_per_label: int | None = None,
) -> list[ProductionRule]:
    """Induce deterministic lower-approximation production rules."""

    granules = build_granules(
        system.information_function,
        system.decisions,
        attributes=system.conditional_attributes,
    )
    rules_by_label: dict[str, list[ProductionRule]] = defaultdict(list)
    for granule in granules:
        if not granule.deterministic:
            continue
        if len(granule.object_ids) < min_support:
            continue
        if allowed_labels is not None and granule.majority_label not in allowed_labels:
            continue
        rule = rule_from_granule(granule, plan, universe_size=len(system.universe))
        if use_reducts:
            rule = minimize_rule_conditions(
                rule,
                system.information_function,
                system.decisions,
                min_support=min_support,
            )
        if rule.confidence >= 1.0 and rule.support_count >= min_support:
            rules_by_label[rule.target_label].append(rule)

    rules: list[ProductionRule] = []
    for label, label_rules in sorted(rules_by_label.items()):
        label_rules.sort(
            key=lambda item: (
                -item.support_count,
                len(item.antecedents),
                -item.confidence,
                item.rule_id,
            )
        )
        rules.extend(label_rules[:max_rules_per_label] if max_rules_per_label else label_rules)
    return rules


def rule_signature_key(rule: ProductionRule) -> tuple[str, tuple[tuple[str, str | None, bool], ...]]:
    """Return a stable deduplication key after reduct/minimization.

    Different deterministic granules can collapse to the same clinically visible
    antecedent set after condition minimization. Keeping all of those rows makes
    rulebooks look larger than the actual physician-facing rule set. The key uses
    only what a clinician sees: target label and sorted antecedent atoms.
    """

    atoms = tuple(
        sorted(
            (condition.feature, condition.state, condition.negated)
            for condition in rule.antecedents
        )
    )
    return (rule.target_label, atoms)


def deduplicate_rules(rules: Sequence[ProductionRule]) -> list[ProductionRule]:
    """Merge duplicate clinician-facing rules while preserving aggregate support.

    A duplicate is a rule with the same consequent and the same reduced antecedent
    conditions. The merged rule keeps the best-ranked representative and unions
    coverage sets so support reflects all B-matrix objects represented by that
    production rule.
    """

    merged: dict[tuple[str, tuple[tuple[str, str | None, bool], ...]], ProductionRule] = {}
    for rule in rules:
        key = rule_signature_key(rule)
        if key not in merged:
            merged[key] = rule
            continue
        incumbent = merged[key]
        covered = sorted(set(incumbent.covered_object_ids) | set(rule.covered_object_ids))
        distribution = dict(incumbent.class_distribution)
        for label, count in rule.class_distribution.items():
            distribution[label] = distribution.get(label, 0) + int(count)
        # Avoid double-counting coverage in the displayed support. If source
        # object IDs are available, they are the authoritative support basis.
        support_count = len(covered) if covered else max(incumbent.support_count, rule.support_count)
        notes = incumbent.notes
        if "duplicate_reduced_granule_merged" not in notes:
            notes = f"{notes} duplicate_reduced_granule_merged".strip()
        merged[key] = ProductionRule(
            rule_id=incumbent.rule_id,
            target_label=incumbent.target_label,
            antecedents=incumbent.antecedents,
            confidence=max(incumbent.confidence, rule.confidence),
            support_count=support_count,
            support_fraction=max(incumbent.support_fraction, rule.support_fraction),
            covered_object_ids=covered,
            class_distribution=dict(sorted(distribution.items())),
            threshold_provenance={**rule.threshold_provenance, **incumbent.threshold_provenance},
            source=incumbent.source,
            reduced_from_conditions=max(incumbent.reduced_from_conditions, rule.reduced_from_conditions),
            notes=notes,
            physician_predicate_label=incumbent.physician_predicate_label,
            predicate_similarity=dict(incumbent.predicate_similarity),
        )
    return list(merged.values())


def annotate_rule_predicate_similarity(
    rule: ProductionRule,
    predicates: Mapping[str, MedicalPredicate],
) -> ProductionRule:
    """Attach maximum-similarity alignment to a physician predicate library."""

    generated = rule_to_medical_predicate(rule)
    best_label, components = max_predicate_similarity(generated, predicates)
    target_components = components
    if rule.target_label in predicates:
        from tm_ecg.dss.similarity import predicate_similarity

        target_components = predicate_similarity(generated, predicates[rule.target_label])
    notes = rule.notes
    if best_label != rule.target_label:
        notes = f"{notes} best_physician_predicate={best_label}".strip()
    return ProductionRule(
        rule_id=rule.rule_id,
        target_label=rule.target_label,
        antecedents=rule.antecedents,
        confidence=rule.confidence,
        support_count=rule.support_count,
        support_fraction=rule.support_fraction,
        covered_object_ids=rule.covered_object_ids,
        class_distribution=rule.class_distribution,
        threshold_provenance=rule.threshold_provenance,
        source=rule.source,
        reduced_from_conditions=rule.reduced_from_conditions,
        notes=notes,
        physician_predicate_label=best_label,
        predicate_similarity=target_components,
    )


def postprocess_rules(
    rules: Sequence[ProductionRule],
    predicates: Mapping[str, MedicalPredicate] | None = None,
    max_rules_per_label: int | None = None,
) -> list[ProductionRule]:
    """Deduplicate, clinically annotate, and rank production rules.

    This step operationalizes the DSS objective that generated predicates should
    be selected by maximum similarity to physician predicates rather than by data
    support alone. Data support remains part of the ordering, but a clinically
    aligned rule is preferred over a larger rule that contradicts the predicate
    library.
    """

    library = predicates or {}
    unique = deduplicate_rules(rules)
    if library:
        unique = [annotate_rule_predicate_similarity(rule, library) for rule in unique]
    by_label: dict[str, list[ProductionRule]] = defaultdict(list)
    for rule in unique:
        by_label[rule.target_label].append(rule)
    result: list[ProductionRule] = []
    for label, label_rules in sorted(by_label.items()):
        label_rules.sort(
            key=lambda item: (
                -float(item.predicate_similarity.get("score", 0.0)),
                item.physician_predicate_label != item.target_label,
                -item.support_count,
                len(item.antecedents),
                item.rule_id,
            )
        )
        result.extend(label_rules[:max_rules_per_label] if max_rules_per_label else label_rules)
    return result


def _row_is_discrete(row: Mapping[str, object], plan: DiscretizationPlan) -> bool:
    for feature, domain in plan.feature_domains.items():
        value = row.get(feature)
        states = {item.label for item in domain.bins} | set(domain.allowed_states) | {domain.missing_state}
        if value is not None and str(value) not in states:
            return False
    return True


def _rule_vote_weight(rule: ProductionRule, match_fraction: float, max_support: int) -> float:
    del max_support
    support_weight = (rule.support_count + 5.0) / (rule.support_count + 10.0)
    condition_count_penalty = 1.0 / (1.0 + 0.025 * max(len(rule.antecedents) - 1, 0))
    precision = rule.oof_precision if rule.oof_precision is not None else rule.confidence
    recall = rule.oof_recall if rule.oof_recall is not None else rule.confidence
    generalization = max(precision * recall, 0.0) ** 0.5
    prior = rule.class_prior if rule.class_prior is not None else 0.5
    prior_correction = min(max((0.5 / max(prior, 0.01)) ** 0.5, 0.5), 1.5)
    calibration_factor = 1.0 - min(max(rule.calibration_uncertainty or 0.0, 0.0), 1.0)
    stability = min(
        max(rule.fold_stability if rule.fold_stability is not None else 1.0, 0.0),
        1.0,
    )
    return (
        generalization
        * support_weight
        * match_fraction
        * condition_count_penalty
        * prior_correction
        * calibration_factor
        * stability
    )


def infer_with_rules(
    row: Mapping[str, object],
    rules: Sequence[ProductionRule],
    plan: DiscretizationPlan,
    predicates: Mapping[str, MedicalPredicate] | None = None,
    soft_match_min_fraction: float = 0.60,
    top_k: int = 5,
    close_vote_ratio: float = 1.10,
    low_strength_threshold: float = 0.20,
    contradiction_penalty: float = 0.25,
    hard_veto_criticality: float = 1.5,
) -> InferenceResult:
    """Discretize a new B-row, activate rules, and return a ranked decision list."""

    discrete_row = dict(row) if _row_is_discrete(row, plan) else quantize_row(row, plan)
    max_support = max((rule.support_count for rule in rules), default=1)
    exact: list[tuple[ProductionRule, float]] = []
    soft: list[tuple[ProductionRule, float]] = []
    for rule in rules:
        match_fraction = rule.match_fraction(discrete_row)
        if rule.matches(discrete_row):
            exact.append((rule, match_fraction))
        elif match_fraction >= soft_match_min_fraction:
            soft.append((rule, match_fraction))

    candidates = exact if exact else soft
    scores: dict[str, float] = defaultdict(float)
    decomposition: dict[str, dict[str, float | bool | None]] = {}
    activated: list[dict[str, object]] = []
    for rule, match_fraction in candidates:
        weight = _rule_vote_weight(rule, match_fraction, max_support)
        predicate_score = None
        predicate_ok = None
        hard_veto = False
        quality_completeness = (
            sum(discrete_row.get(condition.feature) not in {None, "", "missing"} for condition in rule.antecedents)
            / len(rule.antecedents)
            if rule.antecedents
            else 0.0
        )
        weight *= 0.5 + 0.5 * quality_completeness
        if predicates and rule.target_label in predicates:
            predicate = predicates[rule.target_label]
            predicate_ok, predicate_score, _messages = predicate.evaluate(discrete_row)
            weight *= 0.75 + 0.25 * predicate_score
            if predicate_ok is False:
                failed_required = [
                    condition
                    for condition in predicate.required
                    if not condition.matches_discrete(discrete_row)
                ]
                contraindications = [
                    condition
                    for condition in predicate.contraindications
                    if condition.matches_discrete(discrete_row)
                ]
                hard_veto = any(
                    condition.criticality >= hard_veto_criticality
                    for condition in failed_required + contraindications
                )
                weight = 0.0 if hard_veto else weight * contradiction_penalty
        scores[rule.target_label] += weight
        label_decomposition = decomposition.setdefault(
            rule.target_label,
            {
                "total_vote": 0.0,
                "mean_quality_completeness": 0.0,
                "activated_rules": 0.0,
                "hard_veto": False,
                "predicate_score": None,
            },
        )
        label_decomposition["total_vote"] = float(label_decomposition["total_vote"] or 0.0) + weight
        label_decomposition["mean_quality_completeness"] = float(
            label_decomposition["mean_quality_completeness"] or 0.0
        ) + quality_completeness
        label_decomposition["activated_rules"] = float(
            label_decomposition["activated_rules"] or 0.0
        ) + 1.0
        label_decomposition["hard_veto"] = bool(label_decomposition["hard_veto"]) or hard_veto
        label_decomposition["predicate_score"] = predicate_score
        payload = rule.to_dict()
        payload.update(
            {
                "match_fraction": match_fraction,
                "vote_weight": weight,
                "predicate_score": predicate_score,
                "predicate_ok": predicate_ok,
                "quality_completeness": quality_completeness,
                "contradiction_penalty": contradiction_penalty if predicate_ok is False else 1.0,
                "hard_veto": hard_veto,
                "oof_precision": rule.oof_precision,
                "oof_recall": rule.oof_recall,
                "class_prior": rule.class_prior,
                "calibration_uncertainty": rule.calibration_uncertainty,
                "fold_stability": rule.fold_stability,
            }
        )
        activated.append(payload)

    ranked_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for payload in decomposition.values():
        count = float(payload["activated_rules"] or 1.0)
        payload["mean_quality_completeness"] = float(
            payload["mean_quality_completeness"] or 0.0
        ) / count
    abstention: TypedAbstention | None = None
    uncertainty_flags: list[str] = []
    if not exact:
        uncertainty_flags.append("no_exact_rule_match_soft_matching_used")
    if not candidates and abstention is None:
        uncertainty_flags.append("no_rule_match")
    if len({label for label, _score in ranked_scores[:2]}) > 1 and len(ranked_scores) > 1:
        if ranked_scores[0][1] <= ranked_scores[1][1] * close_vote_ratio:
            uncertainty_flags.append("close_conflicting_rule_votes")
    if ranked_scores and ranked_scores[0][1] < low_strength_threshold:
        uncertainty_flags.append("low_rule_vote_strength")

    predicted_label = ranked_scores[0][0] if ranked_scores else None
    plan_features = set(plan.feature_domains)
    if predicted_label in {"RBBB spectrum", "LBBB spectrum"} and "qrs_lead_coverage" in plan_features:
        if discrete_row.get("qrs_lead_coverage") in {None, "missing", "insufficient"}:
            abstention = TypedAbstention(
                "insufficient_required_leads",
                "Bundle-branch classification requires analyzable V1/V2 and lateral leads",
                ("qrs_lead_coverage", "r_prime_v1_any", "broad_r_v6_any"),
            )
    if predicted_label in {"AF", "AFL"} and abstention is None:
        if "atrial_lead_coverage" in plan_features and discrete_row.get("atrial_lead_coverage") in {None, "missing", "insufficient"}:
            abstention = TypedAbstention(
                "insufficient_signal_quality",
                "Atrial rhythm classification requires observable atrial activity",
                ("atrial_lead_coverage", "p_present_ratio", "detector_agreement"),
            )
        elif "analyzable_duration_s" in plan_features and discrete_row.get("analyzable_duration_s") in {None, "missing", "too_short"}:
            abstention = TypedAbstention(
                "insufficient_valid_beats",
                "AF/AFL classification requires the configured analyzable duration and beat count",
                ("analyzable_duration_s", "rhythm_valid_beat_fraction"),
            )
    if not candidates:
        abstention = TypedAbstention(
            "no_supported_rule",
            "No exact or sufficiently similar executable rule was activated",
            tuple(sorted(plan_features)),
        )
    elif abstention is None and "close_conflicting_rule_votes" in uncertainty_flags and ranked_scores[1][1] >= low_strength_threshold:
        abstention = TypedAbstention(
            "conflicting_high_confidence_axes",
            "The two leading rule families have materially indistinguishable evidence",
            tuple(sorted({ranked_scores[0][0], ranked_scores[1][0]})),
        )
    elif abstention is None and "low_rule_vote_strength" in uncertainty_flags:
        abstention = TypedAbstention(
            "low_calibrated_probability",
            "The strongest calibrated evidence score is below the configured threshold",
            tuple(sorted(plan_features)),
        )
    if abstention is not None:
        predicted_label = None
    if predicted_label:
        explanation = (
            f"DSS selected {predicted_label} from {len(candidates)} activated rule(s); "
            f"top vote strength={ranked_scores[0][1]:.3f}."
        )
    else:
        explanation = "No deterministic or sufficiently similar production rule was activated."
    activated.sort(key=lambda item: (-float(item["vote_weight"]), str(item["rule_id"])))
    return InferenceResult(
        predicted_label=predicted_label,
        ranked_scores=ranked_scores[:top_k],
        activated_rules=activated[:top_k],
        uncertainty_flags=uncertainty_flags,
        explanation=explanation,
        abstention=abstention,
        score_decomposition=decomposition,
    )


def rules_to_dict(rules: Sequence[ProductionRule]) -> list[dict[str, object]]:
    return [rule.to_dict() for rule in rules]
