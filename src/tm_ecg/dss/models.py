"""Typed data models for the production-rule ECG decision-support system.

The DSS uses the row-vector transition convention already used by the transition
operator in this repository:

    A is m x k, B is m x l, T is k x l, and B_hat = A @ T.

Every row identifier must therefore be aligned across A, B, predicted B_hat, model
labels, and any clinical predicate/rule artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from tm_ecg.types import TypedAbstention


Number = int | float


@dataclass(slots=True)
class IntervalBin:
    """One admissible interval/state in an attribute domain."""

    code: int
    label: str
    lower: float | None = None
    upper: float | None = None
    include_lower: bool = True
    include_upper: bool = False
    provenance: str = "data-driven"
    threshold_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Interval-bin label cannot be empty")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(
                f"Interval-bin lower bound exceeds upper bound for {self.label}"
            )

    def contains(self, value: object) -> bool:
        if value is None or value == "":
            return False
        numeric = float(value)
        if self.lower is not None:
            if self.include_lower and numeric < self.lower:
                return False
            if not self.include_lower and numeric <= self.lower:
                return False
        if self.upper is not None:
            if self.include_upper and numeric > self.upper:
                return False
            if not self.include_upper and numeric >= self.upper:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AttributeDomain:
    """Domain for one ECG clinical feature in the decision table."""

    name: str
    value_type: str
    unit: str = ""
    family: str = ""
    bins: list[IntervalBin] = field(default_factory=list)
    allowed_states: list[str] = field(default_factory=list)
    missing_state: str = "missing"
    clinical_priority: float = 1.0
    provenance: str = "unspecified"

    def state_for_value(self, value: object) -> str:
        if value is None or value == "":
            return self.missing_state
        text_value = str(value)
        known_states = {item.label for item in self.bins} | set(self.allowed_states) | {self.missing_state}
        if text_value in known_states:
            return text_value
        if self.value_type in {"binary", "categorical"}:
            try:
                numeric = float(value)
                if self.value_type == "binary":
                    return "present" if numeric >= 0.5 else "absent"
            except (TypeError, ValueError):
                pass
            return str(value)
        for interval in self.bins:
            if interval.contains(value):
                return interval.label
        return self.missing_state

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bins"] = [item.to_dict() for item in self.bins]
        return payload


@dataclass(slots=True)
class ThresholdRecord:
    """Audit record for one learned or clinically anchored threshold."""

    threshold_id: str
    feature: str
    value: float
    left_size: int
    right_size: int
    entropy: float
    density: float
    objective: float
    source: str
    alpha: float
    depth: int = 0
    accepted: bool = True
    notes: str = ""
    unit: str = ""
    version: str = "wedd_v2"
    selection_frequency: float | None = None
    bootstrap_median: float | None = None
    bootstrap_iqr: float | None = None
    perturbation_flip_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DiscretizationPlan:
    """A learned feature-wise quantization plan for matrix B."""

    feature_domains: dict[str, AttributeDomain]
    thresholds: list[ThresholdRecord]
    class_labels: list[str]
    alpha: float
    min_support: int
    max_depth: int
    orientation: str = "B_hat = A @ T"
    candidate_thresholds: list[ThresholdRecord] = field(default_factory=list)
    fit_partition: str = "training_or_oof"
    threshold_version: str = "wedd_v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_domains": {name: domain.to_dict() for name, domain in self.feature_domains.items()},
            "thresholds": [item.to_dict() for item in self.thresholds],
            "class_labels": self.class_labels,
            "alpha": self.alpha,
            "min_support": self.min_support,
            "max_depth": self.max_depth,
            "orientation": self.orientation,
            "candidate_thresholds": [item.to_dict() for item in self.candidate_thresholds],
            "fit_partition": self.fit_partition,
            "threshold_version": self.threshold_version,
        }

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


@dataclass(slots=True)
class DecisionSystem:
    """Formal DMS/DSS tuple (U, C, d, V, f) for ECG rule induction."""

    universe: list[str]
    conditional_attributes: list[str]
    decision_attribute: str
    attribute_domains: dict[str, AttributeDomain]
    information_function: dict[str, dict[str, object]]
    decisions: dict[str, str]
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        missing_rows = [object_id for object_id in self.universe if object_id not in self.information_function]
        if missing_rows:
            raise ValueError(f"Missing information rows for {len(missing_rows)} object(s)")
        missing_decisions = [object_id for object_id in self.universe if object_id not in self.decisions]
        if missing_decisions:
            raise ValueError(f"Missing decisions for {len(missing_decisions)} object(s)")
        unknown_attrs = [attr for attr in self.conditional_attributes if attr not in self.attribute_domains]
        if unknown_attrs:
            raise ValueError(f"Missing attribute domains: {unknown_attrs}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "conditional_attributes": self.conditional_attributes,
            "decision_attribute": self.decision_attribute,
            "attribute_domains": {name: domain.to_dict() for name, domain in self.attribute_domains.items()},
            "information_function": self.information_function,
            "decisions": self.decisions,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> Path:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


@dataclass(frozen=True, slots=True)
class RuleCondition:
    """One antecedent atom in a clinical predicate or production rule."""

    feature: str
    state: str | None = None
    operator: str | None = None
    value: float | str | None = None
    lower: float | None = None
    upper: float | None = None
    family: str = ""
    clinical_concept: str = ""
    source: str = ""
    criticality: float = 1.0
    negated: bool = False

    def key(self) -> tuple[str, str | None, str | None, str | None]:
        return (self.feature, self.state, self.operator, self.clinical_concept or None)

    def matches_discrete(self, row: Mapping[str, object]) -> bool:
        observed = row.get(self.feature)
        if self.state is not None:
            match = str(observed) == self.state
        elif self.operator is not None:
            match = self._compare(row.get(self.feature))
        else:
            match = observed not in {None, "", "missing"}
        return not match if self.negated else match

    def _compare(self, observed: object) -> bool:
        if observed is None or observed == "":
            return False
        try:
            left = float(observed)
            right = float(self.value) if self.value is not None else None
        except (TypeError, ValueError):
            return False
        if self.operator == ">=":
            return right is not None and left >= right
        if self.operator == ">":
            return right is not None and left > right
        if self.operator == "<=":
            return right is not None and left <= right
        if self.operator == "<":
            return right is not None and left < right
        if self.operator == "==":
            return right is not None and left == right
        if self.operator == "between":
            lower_ok = self.lower is None or left >= self.lower
            upper_ok = self.upper is None or left <= self.upper
            return lower_ok and upper_ok
        raise ValueError(f"Unsupported operator: {self.operator}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InformationGranule:
    """Equivalence class induced by a discretized feature signature."""

    signature: tuple[tuple[str, str], ...]
    object_ids: list[str]
    class_distribution: dict[str, int]
    deterministic: bool
    majority_label: str
    confidence: float
    boundary_region: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": list(self.signature),
            "object_ids": self.object_ids,
            "class_distribution": self.class_distribution,
            "deterministic": self.deterministic,
            "majority_label": self.majority_label,
            "confidence": self.confidence,
            "boundary_region": self.boundary_region,
        }


@dataclass(slots=True)
class ProductionRule:
    """Clinician-facing production rule induced from the B-matrix."""

    rule_id: str
    target_label: str
    antecedents: list[RuleCondition]
    confidence: float
    support_count: int
    support_fraction: float
    covered_object_ids: list[str]
    class_distribution: dict[str, int]
    threshold_provenance: dict[str, str] = field(default_factory=dict)
    source: str = "rough_set_lower_approximation"
    reduced_from_conditions: int = 0
    notes: str = ""
    physician_predicate_label: str | None = None
    predicate_similarity: dict[str, float] = field(default_factory=dict)
    oof_precision: float | None = None
    oof_recall: float | None = None
    class_prior: float | None = None
    calibration_uncertainty: float | None = None
    fold_stability: float | None = None

    def matches(self, discrete_row: Mapping[str, object]) -> bool:
        return all(condition.matches_discrete(discrete_row) for condition in self.antecedents)

    def match_fraction(self, discrete_row: Mapping[str, object]) -> float:
        if not self.antecedents:
            return 0.0
        weight_sum = sum(max(condition.criticality, 0.0) for condition in self.antecedents) or 1.0
        matched = sum(max(condition.criticality, 0.0) for condition in self.antecedents if condition.matches_discrete(discrete_row))
        return matched / weight_sum

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "target_label": self.target_label,
            "antecedents": [item.to_dict() for item in self.antecedents],
            "confidence": self.confidence,
            "support_count": self.support_count,
            "support_fraction": self.support_fraction,
            "covered_object_ids": self.covered_object_ids,
            "class_distribution": self.class_distribution,
            "threshold_provenance": self.threshold_provenance,
            "source": self.source,
            "reduced_from_conditions": self.reduced_from_conditions,
            "notes": self.notes,
            "physician_predicate_label": self.physician_predicate_label,
            "predicate_similarity": self.predicate_similarity,
            "oof_precision": self.oof_precision,
            "oof_recall": self.oof_recall,
            "class_prior": self.class_prior,
            "calibration_uncertainty": self.calibration_uncertainty,
            "fold_stability": self.fold_stability,
        }


@dataclass(slots=True)
class MedicalPredicate:
    """Executable and human-readable ECG diagnostic predicate."""

    label: str
    source_labels: dict[str, list[str]]
    required: list[RuleCondition]
    supportive: list[RuleCondition] = field(default_factory=list)
    contraindications: list[RuleCondition] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    explanation: str = ""
    weak_signature: bool = False

    def features(self) -> set[str]:
        return {item.feature for item in self.required + self.supportive + self.contraindications}

    def evaluate(self, row: Mapping[str, object]) -> tuple[bool, float, list[str]]:
        required_hits = [condition.matches_discrete(row) for condition in self.required]
        supportive_hits = [condition.matches_discrete(row) for condition in self.supportive]
        contraindication_hits = [condition.matches_discrete(row) for condition in self.contraindications]
        required_ok = all(required_hits) if self.required else True
        safety_ok = not any(contraindication_hits)
        total_weight = sum(c.criticality for c in self.required + self.supportive + self.contraindications) or 1.0
        hit_weight = 0.0
        for condition, hit in zip(self.required, required_hits, strict=False):
            hit_weight += condition.criticality if hit else 0.0
        for condition, hit in zip(self.supportive, supportive_hits, strict=False):
            hit_weight += 0.5 * condition.criticality if hit else 0.0
        for condition, hit in zip(self.contraindications, contraindication_hits, strict=False):
            hit_weight += condition.criticality if not hit else 0.0
        messages = []
        if required_ok and safety_ok:
            messages.append(f"{self.label}: required ECG predicate conditions are satisfied")
        elif not required_ok:
            messages.append(f"{self.label}: at least one required ECG predicate condition is not satisfied")
        if not safety_ok:
            messages.append(f"{self.label}: contraindicating or warning condition is present")
        return required_ok and safety_ok, min(max(hit_weight / total_weight, 0.0), 1.0), messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_labels": self.source_labels,
            "required": [item.to_dict() for item in self.required],
            "supportive": [item.to_dict() for item in self.supportive],
            "contraindications": [item.to_dict() for item in self.contraindications],
            "references": self.references,
            "explanation": self.explanation,
            "weak_signature": self.weak_signature,
        }


@dataclass(slots=True)
class InferenceResult:
    predicted_label: str | None
    ranked_scores: list[tuple[str, float]]
    activated_rules: list[dict[str, Any]]
    uncertainty_flags: list[str]
    explanation: str
    abstention: TypedAbstention | None = None
    score_decomposition: dict[str, dict[str, float | bool | None]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatrixAlignmentReport:
    a_rows: int
    b_rows: int
    t_rows: int
    t_cols: int
    common_row_ids: int
    orientation: str
    valid: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
