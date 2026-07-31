from tm_ecg.dss.discretization import fit_wedd_discretization
from tm_ecg.dss.granulation import build_granules, lower_approximation, weighted_hamming_similarity
from tm_ecg.dss.predicates import default_medical_predicates
from tm_ecg.dss.rules import (
    _rule_vote_weight,
    build_decision_system,
    induce_rules,
    infer_with_rules,
)


def _rows_labels():
    rows = [
        {"qrs_dur_med_ms": 92.0, "r_prime_v1_any": 0, "broad_r_v6_any": 0, "record_id": "n1"},
        {"qrs_dur_med_ms": 96.0, "r_prime_v1_any": 0, "broad_r_v6_any": 0, "record_id": "n2"},
        {"qrs_dur_med_ms": 140.0, "r_prime_v1_any": 1, "broad_r_v6_any": 0, "record_id": "r1"},
        {"qrs_dur_med_ms": 150.0, "r_prime_v1_any": 1, "broad_r_v6_any": 0, "record_id": "r2"},
    ]
    for row in rows:
        row.update(
            lead_quality_min_db=15.0,
            delineation_confidence=0.95,
            analyzable_duration_s=10.0,
        )
    labels = ["Normal", "Normal", "RBBB spectrum", "RBBB spectrum"]
    return rows, labels


def test_granules_lower_approximation_and_rules():
    rows, labels = _rows_labels()
    features = [
        "qrs_dur_med_ms",
        "r_prime_v1_any",
        "broad_r_v6_any",
        "lead_quality_min_db",
        "delineation_confidence",
        "analyzable_duration_s",
    ]
    plan = fit_wedd_discretization(rows, labels, features=features, min_support=1)
    system = build_decision_system(rows, labels, plan, object_ids=[row["record_id"] for row in rows])
    granules = build_granules(system.information_function, system.decisions)
    rbbb_lower = lower_approximation(granules, "RBBB spectrum", min_support=1)
    assert rbbb_lower
    rules = induce_rules(system, plan, min_support=1, use_reducts=True)
    assert any(rule.target_label == "RBBB spectrum" for rule in rules)
    inferred = infer_with_rules(
        {
            "qrs_dur_med_ms": 145.0,
            "r_prime_v1_any": 1,
            "broad_r_v6_any": 0,
            "lead_quality_min_db": 15.0,
            "delineation_confidence": 0.95,
            "analyzable_duration_s": 10.0,
        },
        rules,
        plan,
        predicates=default_medical_predicates(),
    )
    assert inferred.predicted_label == "RBBB spectrum"
    assert inferred.activated_rules


def test_weighted_hamming_similarity():
    assert weighted_hamming_similarity({"a": "x", "b": "y"}, {"a": "x", "b": "z"}) == 0.5


def test_zero_fold_stability_produces_zero_rule_vote():
    rows, labels = _rows_labels()
    features = ["qrs_dur_med_ms", "r_prime_v1_any"]
    plan = fit_wedd_discretization(rows, labels, features=features, min_support=1)
    system = build_decision_system(
        rows,
        labels,
        plan,
        object_ids=[row["record_id"] for row in rows],
    )
    rule = induce_rules(system, plan, min_support=1, use_reducts=False)[0]
    rule.oof_precision = 0.9
    rule.oof_recall = 0.9
    rule.fold_stability = 0.0

    assert _rule_vote_weight(rule, 1.0, rule.support_count) == 0.0
