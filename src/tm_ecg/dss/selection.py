"""Well-classified arrhythmia and B-feature selection for DSS induction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from math import isfinite, sqrt
from typing import Mapping, Sequence

from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.dss.models import MatrixAlignmentReport, MedicalPredicate
from tm_ecg.dss.predicates import default_medical_predicates


STRICT_SELECTION_POLICY_VERSION = "2026-07-22-v1"
STRICT_ALLOWED_METRIC_PROVENANCE = (
    "held_out_compatibility_head",
    "external_compatibility_audit",
    "row_level_compatibility_predictions",
)
STRICT_ALLOWED_EVALUATION_PARTITIONS = (
    "held_out_test",
    "external_test",
    "external_validation",
)


@dataclass(slots=True)
class SelectionPolicy:
    min_precision: float = 0.85
    min_recall: float = 0.85
    min_f1: float = 0.85
    min_specificity: float = 0.85
    min_support: int = 20
    min_calibration_confidence: float = 0.70
    per_class_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    min_metric_lower_bound: float = 0.80
    min_analyzable_coverage: float = 0.80
    max_abstention_rate: float = 0.20
    require_executable_predicate: bool = True
    min_fold_stability: float = 0.80
    max_calibration_error: float = 0.10
    min_global_accuracy: float = 0.90
    required_global_accuracy_definition: str = "compatibility_subset_exact_match"
    allowed_metric_provenance: tuple[str, ...] = (
        "held_out_compatibility_head",
        "external_compatibility_audit",
        "row_level_compatibility_predictions",
    )
    require_complete_evidence: bool = True
    require_held_out_evidence: bool = True
    require_patient_disjoint: bool = True
    require_calibrated_probabilities: bool = True
    require_independent_threshold_partition: bool = True
    allowed_evaluation_partitions: tuple[str, ...] = (
        "held_out_test",
        "external_test",
        "external_validation",
    )
    strict_mode: bool = False
    policy_version: str = "custom"

    def __post_init__(self) -> None:
        if not self.strict_mode:
            return

        def strict_minimum(value: float, floor: float, *, field_name: str) -> float:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"strict policy {field_name} must be finite")
            return max(numeric, floor)

        def strict_maximum(value: float, ceiling: float, *, field_name: str) -> float:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"strict policy {field_name} must be finite")
            return min(numeric, ceiling)

        # These limits are release-safety invariants. Runtime/configuration
        # overrides may tighten them, but can never make a strict run easier.
        self.min_precision = strict_minimum(
            self.min_precision, 0.85, field_name="min_precision"
        )
        self.min_recall = strict_minimum(
            self.min_recall, 0.85, field_name="min_recall"
        )
        self.min_f1 = strict_minimum(self.min_f1, 0.85, field_name="min_f1")
        self.min_specificity = strict_minimum(
            self.min_specificity, 0.85, field_name="min_specificity"
        )
        self.min_support = max(int(self.min_support), 20)
        self.min_calibration_confidence = strict_minimum(
            self.min_calibration_confidence,
            0.70,
            field_name="min_calibration_confidence",
        )
        self.min_metric_lower_bound = strict_minimum(
            self.min_metric_lower_bound,
            0.80,
            field_name="min_metric_lower_bound",
        )
        self.min_analyzable_coverage = strict_minimum(
            self.min_analyzable_coverage,
            0.80,
            field_name="min_analyzable_coverage",
        )
        self.max_abstention_rate = strict_maximum(
            self.max_abstention_rate,
            0.20,
            field_name="max_abstention_rate",
        )
        self.min_fold_stability = strict_minimum(
            self.min_fold_stability,
            0.80,
            field_name="min_fold_stability",
        )
        self.max_calibration_error = strict_maximum(
            self.max_calibration_error,
            0.10,
            field_name="max_calibration_error",
        )
        self.min_global_accuracy = strict_minimum(
            self.min_global_accuracy,
            0.90,
            field_name="min_global_accuracy",
        )
        self.required_global_accuracy_definition = "compatibility_subset_exact_match"
        self.require_executable_predicate = True
        self.require_complete_evidence = True
        self.require_held_out_evidence = True
        self.require_patient_disjoint = True
        self.require_calibrated_probabilities = True
        self.require_independent_threshold_partition = True
        self.allowed_metric_provenance = tuple(
            value
            for value in self.allowed_metric_provenance
            if value in STRICT_ALLOWED_METRIC_PROVENANCE
        )
        self.allowed_evaluation_partitions = tuple(
            value
            for value in self.allowed_evaluation_partitions
            if value in STRICT_ALLOWED_EVALUATION_PARTITIONS
        )
        self.per_class_thresholds = {
            label: {
                **thresholds,
                "min_support": max(
                    int(thresholds.get("min_support", self.min_support)),
                    self.min_support,
                ),
                "min_precision": max(
                    strict_minimum(
                        thresholds.get("min_precision", self.min_precision),
                        self.min_precision,
                        field_name=f"per_class_thresholds.{label}.min_precision",
                    ),
                    self.min_precision,
                ),
                "min_recall": max(
                    strict_minimum(
                        thresholds.get("min_recall", self.min_recall),
                        self.min_recall,
                        field_name=f"per_class_thresholds.{label}.min_recall",
                    ),
                    self.min_recall,
                ),
                "min_specificity": max(
                    strict_minimum(
                        thresholds.get("min_specificity", self.min_specificity),
                        self.min_specificity,
                        field_name=f"per_class_thresholds.{label}.min_specificity",
                    ),
                    self.min_specificity,
                ),
                "min_f1": max(
                    strict_minimum(
                        thresholds.get("min_f1", self.min_f1),
                        self.min_f1,
                        field_name=f"per_class_thresholds.{label}.min_f1",
                    ),
                    self.min_f1,
                ),
            }
            for label, thresholds in self.per_class_thresholds.items()
        }
        self.policy_version = STRICT_SELECTION_POLICY_VERSION

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        payload["policy_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload


@dataclass(slots=True)
class ClassMetric:
    label: str
    support: int
    precision: float
    recall: float
    specificity: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    mean_confidence: float | None = None
    selected: bool = False
    reason: str = ""
    metric_lower_bound: float | None = None
    analyzable_coverage: float | None = None
    abstention_rate: float | None = None
    fold_stability: float | None = None
    executable_predicate: bool | None = None
    calibration_error: float | None = None
    probabilities_calibrated: bool | None = None
    evaluation_partition: str | None = None
    threshold_partition: str | None = None
    patient_disjoint: bool | None = None
    split_integrity_verified: bool | None = None
    split_manifest_sha256: str | None = None
    evaluation_records: int | None = None
    global_accuracy: float | None = None
    global_accuracy_definition: str | None = None
    metric_provenance: str | None = None
    confidence_level: float | None = None
    confidence_interval_method: str | None = None
    evidence_source: str | None = None
    evidence_valid: bool = True
    evidence_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def per_class_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, ClassMetric]:
    """Compute transparent one-vs-rest metrics for each arrhythmia label."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be aligned")
    if confidences is not None and len(confidences) != len(y_true):
        raise ValueError("confidences and labels must be aligned")
    all_labels = list(labels or sorted(set(y_true) | set(y_pred) | set(PROJECT_LABELS)))
    result: dict[str, ClassMetric] = {}
    n = len(y_true)
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p != label)
        tn = n - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        conf = None
        if confidences is not None:
            conf_values = [c for c, t in zip(confidences, y_true, strict=False) if t == label]
            conf = sum(conf_values) / len(conf_values) if conf_values else None
        result[label] = ClassMetric(
            label=label,
            support=tp + fn,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1=f1,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            mean_confidence=conf,
        )
    return result


def select_well_classified(
    metrics: Mapping[str, ClassMetric],
    policy: SelectionPolicy | None = None,
) -> tuple[set[str], dict[str, ClassMetric]]:
    """Apply conservative threshold policy to per-class metrics."""

    selected: set[str] = set()
    p = policy or SelectionPolicy()
    audited: dict[str, ClassMetric] = {}
    for label, metric in metrics.items():
        reasons: list[str] = []
        class_policy = p.per_class_thresholds.get(label, {})
        min_support = int(class_policy.get("min_support", p.min_support))
        min_precision = float(class_policy.get("min_precision", p.min_precision))
        min_recall = float(class_policy.get("min_recall", p.min_recall))
        min_specificity = float(class_policy.get("min_specificity", p.min_specificity))
        min_f1 = float(class_policy.get("min_f1", p.min_f1))
        for name in ("precision", "recall", "specificity", "f1"):
            value = float(getattr(metric, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                reasons.append(f"invalid_{name}")
        if metric.support < 0 or min(metric.tp, metric.fp, metric.tn, metric.fn) < 0:
            reasons.append("invalid_confusion_counts")
        if metric.support != metric.tp + metric.fn:
            reasons.append("support_confusion_mismatch")
        expected_precision = metric.tp / (metric.tp + metric.fp) if metric.tp + metric.fp else 0.0
        expected_recall = metric.tp / (metric.tp + metric.fn) if metric.tp + metric.fn else 0.0
        expected_specificity = metric.tn / (metric.tn + metric.fp) if metric.tn + metric.fp else 0.0
        expected_f1 = (
            2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)
            if expected_precision + expected_recall
            else 0.0
        )
        for name, expected in (
            ("precision", expected_precision),
            ("recall", expected_recall),
            ("specificity", expected_specificity),
            ("f1", expected_f1),
        ):
            if abs(float(getattr(metric, name)) - expected) > 1e-6:
                reasons.append(f"{name}_confusion_mismatch")
        if not metric.evidence_valid:
            reasons.append("invalid_evidence")
        reasons.extend(f"evidence:{issue}" for issue in metric.evidence_issues)
        if metric.support < min_support:
            reasons.append(f"support<{min_support}")
        if metric.precision < min_precision:
            reasons.append(f"precision<{min_precision:.2f}")
        if metric.recall < min_recall:
            reasons.append(f"recall<{min_recall:.2f}")
        if metric.specificity < min_specificity:
            reasons.append(f"specificity<{min_specificity:.2f}")
        if metric.f1 < min_f1:
            reasons.append(f"f1<{min_f1:.2f}")
        if metric.mean_confidence is None:
            if p.require_complete_evidence:
                reasons.append("missing_calibration_confidence")
        elif not 0.0 <= metric.mean_confidence <= 1.0:
            reasons.append("invalid_calibration_confidence")
        elif metric.mean_confidence < p.min_calibration_confidence:
            reasons.append(f"confidence<{p.min_calibration_confidence:.2f}")
        if metric.metric_lower_bound is None:
            if p.require_complete_evidence:
                reasons.append("missing_metric_lower_bound")
        elif not 0.0 <= metric.metric_lower_bound <= 1.0:
            reasons.append("invalid_metric_lower_bound")
        elif metric.metric_lower_bound < p.min_metric_lower_bound:
            reasons.append(f"metric_lower_bound<{p.min_metric_lower_bound:.2f}")
        if metric.confidence_level is None or metric.confidence_interval_method is None:
            if p.require_complete_evidence:
                reasons.append("missing_confidence_interval_provenance")
        if metric.analyzable_coverage is None:
            if p.require_complete_evidence:
                reasons.append("missing_analyzable_coverage")
        elif not 0.0 <= metric.analyzable_coverage <= 1.0:
            reasons.append("invalid_analyzable_coverage")
        elif metric.analyzable_coverage < p.min_analyzable_coverage:
            reasons.append(f"analyzable_coverage<{p.min_analyzable_coverage:.2f}")
        if metric.abstention_rate is None:
            if p.require_complete_evidence:
                reasons.append("missing_abstention_rate")
        elif not 0.0 <= metric.abstention_rate <= 1.0:
            reasons.append("invalid_abstention_rate")
        elif metric.abstention_rate > p.max_abstention_rate:
            reasons.append(f"abstention_rate>{p.max_abstention_rate:.2f}")
        if metric.fold_stability is None:
            if p.require_complete_evidence:
                reasons.append("missing_fold_stability")
        elif not 0.0 <= metric.fold_stability <= 1.0:
            reasons.append("invalid_fold_stability")
        elif metric.fold_stability < p.min_fold_stability:
            reasons.append(f"fold_stability<{p.min_fold_stability:.2f}")
        if p.require_calibrated_probabilities:
            if metric.probabilities_calibrated is not True:
                reasons.append("probabilities_not_proven_calibrated")
            if metric.calibration_error is None:
                reasons.append("missing_calibration_error")
            elif not 0.0 <= metric.calibration_error <= 1.0:
                reasons.append("invalid_calibration_error")
            elif metric.calibration_error > p.max_calibration_error:
                reasons.append(f"calibration_error>{p.max_calibration_error:.2f}")
        if p.require_held_out_evidence and metric.evaluation_partition not in p.allowed_evaluation_partitions:
            reasons.append("not_held_out_evidence")
        if p.require_independent_threshold_partition:
            if metric.threshold_partition != "validation_only":
                reasons.append("threshold_not_validation_only")
            if metric.threshold_partition == metric.evaluation_partition:
                reasons.append("threshold_evaluation_partition_overlap")
        if p.require_patient_disjoint and metric.patient_disjoint is not True:
            reasons.append("patient_disjointness_not_proven")
        if p.require_held_out_evidence and metric.split_integrity_verified is not True:
            reasons.append("split_integrity_not_verified")
        confusion_total = metric.tp + metric.fp + metric.tn + metric.fn
        if metric.evaluation_records is None:
            if p.require_complete_evidence:
                reasons.append("missing_evaluation_record_count")
        elif metric.evaluation_records != confusion_total:
            reasons.append("evaluation_record_count_mismatch")
        if metric.global_accuracy is None:
            if p.require_complete_evidence:
                reasons.append("missing_global_accuracy")
        elif not 0.0 <= metric.global_accuracy <= 1.0:
            reasons.append("invalid_global_accuracy")
        elif metric.global_accuracy < p.min_global_accuracy:
            reasons.append(f"global_accuracy<{p.min_global_accuracy:.2f}")
        if metric.global_accuracy_definition != p.required_global_accuracy_definition:
            reasons.append("ambiguous_global_accuracy_definition")
        if metric.metric_provenance not in p.allowed_metric_provenance:
            reasons.append("ambiguous_metric_provenance")
        if p.require_executable_predicate and metric.executable_predicate is not True:
            reasons.append("no_executable_quality_aware_predicate")
        # Preserve stable ordering while avoiding duplicate reasons from malformed evidence.
        reasons = list(dict.fromkeys(reasons))
        metric.selected = not reasons
        metric.reason = "selected" if metric.selected else ";".join(reasons)
        if metric.selected:
            selected.add(label)
        audited[label] = metric
    return selected, audited


def transition_column_importance(
    operator: Sequence[Sequence[float]],
    b_columns: Sequence[str],
) -> dict[str, float]:
    """Return column norms for T in the repository convention B_hat = A @ T."""

    if not operator:
        return {}
    t_cols = len(operator[0])
    if t_cols != len(b_columns):
        raise ValueError("operator column count must equal number of B columns")
    scores: dict[str, float] = {}
    for j, column in enumerate(b_columns):
        scores[column] = sqrt(sum(float(row[j]) ** 2 for row in operator))
    return scores


def select_b_features(
    well_classified_labels: set[str],
    b_columns: Sequence[str],
    predicates: Mapping[str, MedicalPredicate] | None = None,
    operator: Sequence[Sequence[float]] | None = None,
    operator_b_columns: Sequence[str] | None = None,
    top_transition_features: int = 0,
    stable_transition_features: set[str] | None = None,
    allow_research_fallback: bool = False,
) -> list[str]:
    """Step 2/3 feature selection from predicates and optional transition-matrix salience."""

    library = predicates or default_medical_predicates()
    available = set(b_columns)
    selected: set[str] = set()
    for label in well_classified_labels:
        if label in library:
            selected.update(library[label].features() & available)
    if operator is not None and top_transition_features > 0 and well_classified_labels:
        scores = transition_column_importance(operator, operator_b_columns or b_columns)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        stable = stable_transition_features or set()
        transition_only = [
            feature
            for feature, _score in ranked
            if feature in available and feature in stable
        ][:top_transition_features]
        selected.update(transition_only)
    if not selected and allow_research_fallback:
        # Explicit opt-in only; strict production selection must fail closed.
        for predicate in library.values():
            selected.update(predicate.features() & available)
    return sorted(selected)


def validate_matrix_alignment(
    a_rows: Sequence[object] | int,
    b_rows: Sequence[object] | int,
    operator: Sequence[Sequence[float]],
    a_row_ids: Sequence[str] | None = None,
    b_row_ids: Sequence[str] | None = None,
) -> MatrixAlignmentReport:
    """Validate the central transition convention B_hat = A @ T."""

    a_count = a_rows if isinstance(a_rows, int) else len(a_rows)
    b_count = b_rows if isinstance(b_rows, int) else len(b_rows)
    t_rows = len(operator)
    t_cols = len(operator[0]) if operator else 0
    notes: list[str] = []
    valid = True
    if a_count != b_count:
        notes.append(f"row_count_mismatch: A={a_count}, B={b_count}")
        valid = False
    common = 0
    if a_row_ids is not None and b_row_ids is not None:
        common = len(set(a_row_ids) & set(b_row_ids))
        if common != min(len(a_row_ids), len(b_row_ids)):
            notes.append("row_id_sets_are_not_identical")
            valid = False
    else:
        common = min(a_count, b_count)
    if t_rows <= 0 or t_cols <= 0:
        notes.append("empty_transition_operator")
        valid = False
    return MatrixAlignmentReport(
        a_rows=int(a_count),
        b_rows=int(b_count),
        t_rows=t_rows,
        t_cols=t_cols,
        common_row_ids=common,
        orientation="B_hat = A @ T; A is m x k, B is m x l, T is k x l",
        valid=valid,
        notes=notes,
    )


def three_step_dss_selection(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    b_columns: Sequence[str],
    confidences: Sequence[float] | None = None,
    operator: Sequence[Sequence[float]] | None = None,
    operator_b_columns: Sequence[str] | None = None,
    policy: SelectionPolicy | None = None,
    predicates: Mapping[str, MedicalPredicate] | None = None,
    top_transition_features: int = 0,
    stable_transition_features: set[str] | None = None,
    allow_research_fallback: bool = False,
) -> dict[str, object]:
    """Implement the required A/B three-step selection logic.

    Step 1 selects only arrhythmia labels whose deep model performance meets
    the supplied policy. Step 2 selects B-matrix features associated with those
    labels. Step 3 keeps the features that are clinically meaningful for the
    selected predicates, optionally supplemented by transition-matrix column
    salience.
    """

    metrics = per_class_classification_metrics(y_true, y_pred, confidences=confidences)
    selected_labels, audited_metrics = select_well_classified(metrics, policy)
    selected_features = select_b_features(
        selected_labels,
        b_columns,
        predicates,
        operator,
        operator_b_columns,
        top_transition_features=top_transition_features,
        stable_transition_features=stable_transition_features,
        allow_research_fallback=allow_research_fallback,
    )
    return {
        "well_classified_labels": sorted(selected_labels),
        "selected_b_features": selected_features,
        "class_metrics": {label: metric.to_dict() for label, metric in audited_metrics.items()},
        "policy": (policy or SelectionPolicy()).to_dict(),
        "top_transition_features": top_transition_features,
        "selection_status": "selected" if selected_features else "failed_no_features",
        "research_fallback_enabled": allow_research_fallback,
    }
