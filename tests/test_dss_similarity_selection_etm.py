import pytest

from tm_ecg.dss.etm import symmetry_defect
from tm_ecg.dss.predicates import default_medical_predicates
from tm_ecg.dss.selection import SelectionPolicy, per_class_classification_metrics, select_well_classified, validate_matrix_alignment
from tm_ecg.dss.similarity import cohen_kappa, predicate_similarity


def test_predicate_similarity_and_kappa():
    predicates = default_medical_predicates()
    score_same = predicate_similarity(predicates["AF"], predicates["AF"])
    score_diff = predicate_similarity(predicates["AF"], predicates["PVC"])
    assert score_same["score"] > 0.95
    assert score_diff["score"] < score_same["score"]
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "A", "B", "B"]) == 1.0


def test_well_classified_selection_and_alignment():
    y_true = ["AF"] * 4 + ["PVC"] * 4
    y_pred = ["AF"] * 4 + ["PVC", "PVC", "AF", "PVC"]
    metrics = per_class_classification_metrics(y_true, y_pred, labels=["AF", "PVC"])
    for metric in metrics.values():
        metric.mean_confidence = 0.90
        metric.metric_lower_bound = 0.50
        metric.analyzable_coverage = 1.0
        metric.abstention_rate = 0.0
        metric.fold_stability = 0.90
        metric.executable_predicate = True
        metric.calibration_error = 0.02
        metric.probabilities_calibrated = True
        metric.evaluation_partition = "held_out_test"
        metric.threshold_partition = "validation_only"
        metric.patient_disjoint = True
        metric.split_integrity_verified = True
        metric.split_manifest_sha256 = "test-fixture"
        metric.evaluation_records = 8
        metric.global_accuracy = 7 / 8
        metric.global_accuracy_definition = "compatibility_subset_exact_match"
        metric.metric_provenance = "held_out_compatibility_head"
        metric.confidence_level = 0.95
        metric.confidence_interval_method = "test_fixture"
    selected, audited = select_well_classified(
        metrics,
        SelectionPolicy(
            min_precision=0.75,
            min_recall=0.75,
            min_f1=0.75,
            min_specificity=0.75,
            min_support=4,
            min_metric_lower_bound=0.50,
            min_global_accuracy=0.80,
        ),
    )
    assert "AF" in selected
    assert audited["AF"].selected
    report = validate_matrix_alignment(3, 3, [[1.0, 0.0], [0.0, 1.0]], ["a", "b", "c"], ["a", "b", "c"])
    assert report.valid
    assert "B_hat = A @ T" in report.orientation


def test_symmetry_defect_zero_for_commuting_generators():
    operator = [[1.0, 0.0], [0.0, 1.0]]
    generator = [[0.0, -1.0], [1.0, 0.0]]
    assert symmetry_defect(operator, generator, generator) == 0.0


def test_strict_selection_policy_can_only_be_tightened() -> None:
    policy = SelectionPolicy(
        min_precision=0.10,
        min_recall=0.20,
        min_f1=0.30,
        min_specificity=0.40,
        min_support=1,
        min_calibration_confidence=0.10,
        min_metric_lower_bound=0.10,
        min_analyzable_coverage=0.10,
        max_abstention_rate=0.90,
        min_fold_stability=0.10,
        max_calibration_error=0.90,
        min_global_accuracy=0.10,
        required_global_accuracy_definition="bitwise_accuracy",
        allowed_metric_provenance=("forged", "held_out_compatibility_head"),
        allowed_evaluation_partitions=("training", "external_test"),
        require_complete_evidence=False,
        require_held_out_evidence=False,
        require_patient_disjoint=False,
        require_calibrated_probabilities=False,
        require_independent_threshold_partition=False,
        require_executable_predicate=False,
        per_class_thresholds={"AF": {"min_f1": 0.01, "min_support": 1}},
        strict_mode=True,
    )

    assert policy.min_precision == 0.85
    assert policy.min_recall == 0.85
    assert policy.min_f1 == 0.85
    assert policy.min_specificity == 0.85
    assert policy.min_support == 20
    assert policy.min_calibration_confidence == 0.70
    assert policy.min_metric_lower_bound == 0.80
    assert policy.min_analyzable_coverage == 0.80
    assert policy.max_abstention_rate == 0.20
    assert policy.min_fold_stability == 0.80
    assert policy.max_calibration_error == 0.10
    assert policy.min_global_accuracy == 0.90
    assert policy.required_global_accuracy_definition == "compatibility_subset_exact_match"
    assert policy.allowed_metric_provenance == ("held_out_compatibility_head",)
    assert policy.allowed_evaluation_partitions == ("external_test",)
    assert policy.require_complete_evidence
    assert policy.require_held_out_evidence
    assert policy.require_patient_disjoint
    assert policy.require_calibrated_probabilities
    assert policy.require_independent_threshold_partition
    assert policy.require_executable_predicate
    assert policy.per_class_thresholds["AF"]["min_f1"] == 0.85
    assert policy.per_class_thresholds["AF"]["min_support"] == 20

    first = policy.to_dict()
    second = policy.to_dict()
    assert first["policy_version"] == "2026-07-22-v1"
    assert first["policy_sha256"] == second["policy_sha256"]


@pytest.mark.parametrize(
    "field_name",
    [
        "min_precision",
        "min_calibration_confidence",
        "min_metric_lower_bound",
        "min_analyzable_coverage",
        "max_abstention_rate",
        "min_fold_stability",
        "max_calibration_error",
        "min_global_accuracy",
    ],
)
def test_strict_policy_rejects_nonfinite_thresholds(field_name: str) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        SelectionPolicy(strict_mode=True, **{field_name: float("nan")})


def test_strict_policy_rejects_nonfinite_per_class_threshold() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        SelectionPolicy(
            strict_mode=True,
            per_class_thresholds={"AF": {"min_f1": float("nan")}},
        )
