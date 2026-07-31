from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from tm_ecg.dss.selection import ClassMetric, SelectionPolicy, select_well_classified
from tm_ecg.dss.eligibility import (
    EligibilityCertificateError,
    verify_rulebook_eligibility_certificate,
)
from tm_ecg.dss.predicates import default_medical_predicates
from tm_ecg.dss.rules import postprocess_rules
import json
from types import SimpleNamespace

from tm_ecg.stages.dss import (
    _certify_eligible_rulebook,
    _load_per_class_metrics,
    _load_prediction_metrics,
    _metric_from_mapping,
    _predicate_executability,
    _selection_audit,
    _write_ineligible_rulebook,
)
from tm_ecg.dss.discretization import fit_wedd_discretization
from tm_ecg.dss.rules import build_decision_system, induce_rules


def test_prediction_audit_table_drives_step1_selection(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text(
        "record_id,true_label,predicted_label,confidence,evaluation_partition,"
        "threshold_partition,patient_disjoint,probabilities_calibrated,"
        "calibration_error,fold_stability\n"
        "1,AF,AF,0.97,held_out_test,validation_only,true,true,0.03,0.94\n"
        "2,AF,AF,0.96,held_out_test,validation_only,true,true,0.03,0.94\n"
        "3,AF,AF,0.95,held_out_test,validation_only,true,true,0.03,0.94\n"
        "4,AF,AF,0.94,held_out_test,validation_only,true,true,0.03,0.94\n"
        "5,PVC,PVC,0.80,held_out_test,validation_only,true,true,0.03,0.94\n"
        "6,PVC,AF,0.52,held_out_test,validation_only,true,true,0.03,0.94\n",
        encoding="utf-8",
    )
    metrics = _load_prediction_metrics(
        path,
        expected_evaluation_ids={"1", "2", "3", "4", "5", "6"},
        verified_patient_disjoint=True,
        split_manifest_sha256="fixture-manifest",
    )
    metrics["AF"].executable_predicate = True
    selected, audited = select_well_classified(
        metrics,
        SelectionPolicy(
            min_precision=0.75,
            min_recall=0.75,
            min_f1=0.75,
            min_specificity=0.50,
            min_support=2,
            min_metric_lower_bound=0.0,
            min_global_accuracy=0.80,
        ),
    )
    assert "AF" in selected
    assert audited["AF"].mean_confidence > 0.9
    assert audited["PVC"].selected is False


def test_global_accuracy_is_compatibility_subset_exact_match(tmp_path):
    path = tmp_path / "multilabel_predictions.csv"
    path.write_text(
        "record_id,true_label,pred_label,confidence,evaluation_partition,"
        "threshold_partition,patient_disjoint,probabilities_calibrated,"
        "calibration_error,fold_stability\n"
        "1,AF|PVC,AF,0.95,held_out_test,validation_only,true,true,0.02,0.95\n"
        "2,Normal,Normal,0.95,held_out_test,validation_only,true,true,0.02,0.95\n",
        encoding="utf-8",
    )

    metrics = _load_prediction_metrics(
        path,
        expected_evaluation_ids={"1", "2"},
        verified_patient_disjoint=True,
        split_manifest_sha256="fixture-manifest",
    )

    assert metrics["AF"].global_accuracy == 0.5
    assert (
        metrics["AF"].global_accuracy_definition
        == "compatibility_subset_exact_match"
    )
    assert metrics["PVC"].fn == 1


def test_locked_manifest_truth_overrides_and_invalidates_supplied_truth(tmp_path):
    path = tmp_path / "tampered_truth.csv"
    path.write_text(
        "record_id,true_label,pred_label,confidence,evaluation_partition,"
        "threshold_partition,patient_disjoint,probabilities_calibrated,"
        "calibration_error,fold_stability\n"
        "1,AF,PVC,0.95,held_out_test,validation_only,true,true,0.02,0.95\n",
        encoding="utf-8",
    )

    metrics = _load_prediction_metrics(
        path,
        label_by_record={"1": "PVC"},
        expected_evaluation_ids={"1"},
        verified_patient_disjoint=True,
        split_manifest_sha256="fixture-manifest",
    )

    assert metrics["PVC"].tp == 1
    assert metrics["AF"].support == 0
    assert metrics["PVC"].evidence_valid is False
    assert "prediction_true_label_manifest_mismatch" in metrics["PVC"].evidence_issues


def test_training_metrics_are_loaded_from_metrics_subdirectory(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "ptbxl_training_metrics.json").write_text(
        json.dumps(
            {
                "per_class_metrics": {
                    "AF": {
                        "support": 25,
                        "precision": 0.9,
                        "recall": 0.8,
                        "specificity": 0.95,
                        "f1": 0.85,
                        "tp": 20,
                        "fp": 2,
                        "tn": 38,
                        "fn": 5,
                        "analyzable_coverage": 0.9,
                        "abstention_rate": 0.1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(paths=SimpleNamespace(reports=tmp_path))

    metrics = _load_per_class_metrics(config, "b1")

    assert metrics is not None
    assert metrics["AF"].support == 25
    assert metrics["AF"].analyzable_coverage == 0.9
    assert metrics["AF"].abstention_rate == 0.1


def test_per_class_metrics_are_bound_to_local_split_manifest(tmp_path):
    metrics_dir = tmp_path / "reports" / "metrics"
    manifests_dir = tmp_path / "manifests"
    metrics_dir.mkdir(parents=True)
    manifests_dir.mkdir()
    manifest_path = manifests_dir / "ptbxl_split_index.csv"
    manifest_path.write_text(
        "row_id,record_id,patient_id,dataset,split,labels,included\n"
        "train-row,1,p1,ptbxl,train,AF,true\n"
        "test-row,2,p2,ptbxl,test,AF,true\n",
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    payload = {
        "split_manifest_sha256": manifest_hash,
        "test_records": 1,
        "per_class_metrics": {
            "AF": {
                "support": 1,
                "precision": 1.0,
                "recall": 1.0,
                "specificity": 0.0,
                "f1": 1.0,
                "tp": 1,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "evaluation_partition": "held_out_test",
                "threshold_partition": "validation_only",
            }
        },
    }
    metrics_path = metrics_dir / "ptbxl_training_metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    config = SimpleNamespace(
        paths=SimpleNamespace(
            reports=tmp_path / "reports",
            manifests=manifests_dir,
        )
    )

    metrics = _load_per_class_metrics(config, "b1")

    assert metrics is not None
    assert metrics["AF"].split_integrity_verified is True
    assert metrics["AF"].patient_disjoint is True
    assert metrics["AF"].evaluation_records == 1

    payload["split_manifest_sha256"] = "wrong-hash"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    mismatched = _load_per_class_metrics(config, "b1")
    assert mismatched is not None
    assert mismatched["AF"].split_integrity_verified is False
    assert "split_manifest_sha256_mismatch" in mismatched["AF"].evidence_issues


def _complete_metric(**updates):
    metric = ClassMetric(
        label="AF",
        support=100,
        precision=90 / 95,
        recall=0.9,
        specificity=95 / 100,
        f1=2 * (90 / 95) * 0.9 / ((90 / 95) + 0.9),
        tp=90,
        fp=5,
        tn=95,
        fn=10,
        mean_confidence=0.92,
        metric_lower_bound=0.84,
        analyzable_coverage=0.90,
        abstention_rate=0.10,
        fold_stability=0.90,
        executable_predicate=True,
        calibration_error=0.03,
        probabilities_calibrated=True,
        evaluation_partition="held_out_test",
        threshold_partition="validation_only",
        patient_disjoint=True,
        split_integrity_verified=True,
        split_manifest_sha256="fixture-manifest",
        evaluation_records=200,
        global_accuracy=0.95,
        global_accuracy_definition="compatibility_subset_exact_match",
        metric_provenance="held_out_compatibility_head",
        confidence_level=0.95,
        confidence_interval_method="bootstrap_percentile",
        evidence_source="test_fixture",
    )
    for key, value in updates.items():
        setattr(metric, key, value)
    return metric


def test_complete_strict_evidence_is_eligible():
    selected, audited = select_well_classified({"AF": _complete_metric()})

    assert selected == {"AF"}
    assert audited["AF"].reason == "selected"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("mean_confidence", None, "missing_calibration_confidence"),
        ("metric_lower_bound", None, "missing_metric_lower_bound"),
        ("analyzable_coverage", None, "missing_analyzable_coverage"),
        ("abstention_rate", None, "missing_abstention_rate"),
        ("fold_stability", None, "missing_fold_stability"),
        ("calibration_error", None, "missing_calibration_error"),
        ("probabilities_calibrated", False, "probabilities_not_proven_calibrated"),
        ("evaluation_partition", "train", "not_held_out_evidence"),
        ("patient_disjoint", False, "patient_disjointness_not_proven"),
        ("split_integrity_verified", False, "split_integrity_not_verified"),
        ("evaluation_records", 199, "evaluation_record_count_mismatch"),
        ("global_accuracy", None, "missing_global_accuracy"),
        ("global_accuracy", 0.89, "global_accuracy<0.90"),
        (
            "global_accuracy_definition",
            "bitwise_label_accuracy",
            "ambiguous_global_accuracy_definition",
        ),
        ("metric_provenance", None, "ambiguous_metric_provenance"),
        ("executable_predicate", False, "no_executable_quality_aware_predicate"),
    ],
)
def test_strict_eligibility_rejects_missing_or_unsafe_evidence(field, value, reason):
    metric = _complete_metric()
    setattr(metric, field, value)

    selected, audited = select_well_classified({"AF": metric})

    assert not selected
    assert reason in audited["AF"].reason


def test_strict_eligibility_rejects_inconsistent_confusion_evidence():
    metric = replace(_complete_metric(), precision=0.99)

    selected, audited = select_well_classified({"AF": metric})

    assert not selected
    assert "precision_confusion_mismatch" in audited["AF"].reason


def test_auto_derived_ci_cannot_replace_missing_strict_evidence():
    metric = _metric_from_mapping(
        "AF",
        {
            "support": 100,
            "precision": 90 / 95,
            "recall": 0.9,
            "specificity": 0.95,
            "f1": 2 * (90 / 95) * 0.9 / ((90 / 95) + 0.9),
            "tp": 90,
            "fp": 5,
            "tn": 95,
            "fn": 10,
        },
    )
    metric.executable_predicate = True

    selected, audited = select_well_classified({"AF": metric})

    assert metric.metric_lower_bound is not None
    assert not selected
    assert "probabilities_not_proven_calibrated" in audited["AF"].reason
    assert "split_integrity_not_verified" in audited["AF"].reason
    assert "ambiguous_metric_provenance" in audited["AF"].reason


def test_training_partition_prediction_audit_is_ineligible(tmp_path):
    path = tmp_path / "leaky_predictions.csv"
    path.write_text(
        "record_id,true_label,pred_label,confidence,evaluation_partition,"
        "threshold_partition,patient_disjoint,probabilities_calibrated,"
        "calibration_error,fold_stability\n"
        "1,AF,AF,0.95,train,validation_only,true,true,0.02,0.95\n"
        "2,AF,AF,0.95,train,validation_only,true,true,0.02,0.95\n",
        encoding="utf-8",
    )
    metrics = _load_prediction_metrics(
        path,
        expected_evaluation_ids={"1", "2"},
        verified_patient_disjoint=True,
        split_manifest_sha256="fixture-manifest",
    )
    metrics["AF"].executable_predicate = True

    selected, audited = select_well_classified(
        metrics,
        SelectionPolicy(
            min_support=2,
            min_metric_lower_bound=0.0,
            min_specificity=0.0,
        ),
    )

    assert not selected
    assert "not_held_out_evidence" in audited["AF"].reason


def test_missing_metrics_never_activates_research_fallback(tmp_path):
    config = SimpleNamespace(paths=SimpleNamespace(reports=tmp_path))

    selected, audit = _selection_audit(
        config,
        "b2",
        ["AF"] * 100,
        {"AF": True},
        SelectionPolicy(),
        allow_research_fallback=True,
    )

    assert not selected
    assert audit["status"] == "failed_no_model_metrics"
    assert audit["research_fallback_requested"] is True
    assert audit["research_fallback_applied"] is False


def test_failed_gate_revokes_stale_rulebook(tmp_path):
    report_directory = tmp_path / "dss"
    report_directory.mkdir()
    stale_json = report_directory / "dss_rulebook_b1.json"
    stale_markdown = report_directory / "dss_rulebook_b1.md"
    stale_json.write_text(
        json.dumps({"status": "selected", "production_rules": [{"rule_id": "old"}]}),
        encoding="utf-8",
    )
    stale_markdown.write_text("# stale eligible rulebook\n", encoding="utf-8")
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    stale_manifest = manifests / "dss_b1.json"
    stale_manifest.write_text(
        json.dumps({"status": "eligible", "rules": 8}),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        ontology_version="test-v2",
        paths=SimpleNamespace(reports=tmp_path, manifests=manifests),
    )

    failure_path = _write_ineligible_rulebook(
        config,
        "b1",
        tmp_path / "B1_raw_train.parquet",
        {"well_classified_labels": []},
    )

    replacement = json.loads(stale_json.read_text(encoding="utf-8"))
    assert failure_path.exists()
    assert replacement["status"] == "ineligible_no_rulebook"
    assert replacement["rules_eligible"] is False
    assert replacement["production_rules"] == []
    assert "revoked" in stale_markdown.read_text(encoding="utf-8")
    replacement_manifest = json.loads(stale_manifest.read_text(encoding="utf-8"))
    assert replacement_manifest["status"] == "ineligible_no_rulebook"
    assert replacement_manifest["rules_eligible"] is False
    assert replacement_manifest["selected_features"] == 0
    assert replacement_manifest["rules"] == 0

    from jsonschema import validate

    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "dss_rulebook_schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate(replacement, schema)


def _eligible_certificate_fixture(tmp_path):
    reports = tmp_path / "reports"
    manifests = tmp_path / "manifests"
    latents = tmp_path / "latents"
    metrics_dir = reports / "metrics"
    for directory in (manifests, latents, metrics_dir):
        directory.mkdir(parents=True)

    split_path = manifests / "ptbxl_split_index.csv"
    split_path.write_text(
        "record_id,patient_id,split,labels,included\n"
        "1,p1,train,AF,true\n"
        "2,p2,test,AF,true\n",
        encoding="utf-8",
    )
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    model_path = latents / "compatibility.joblib"
    model_path.write_bytes(b"locked model bytes")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metrics_path = metrics_dir / "b1_classification_metrics.json"
    source_metric = {
        "support": 100,
        "precision": 0.90,
        "recall": 0.90,
        "specificity": 0.95,
        "f1": 0.90,
        "tp": 90,
        "fp": 5,
        "tn": 95,
        "fn": 10,
        "mean_confidence": 0.92,
        "metric_lower_bound": 0.84,
        "analyzable_coverage": 0.90,
        "abstention_rate": 0.10,
        "fold_stability": 0.90,
        "calibration_error": 0.03,
        "probabilities_calibrated": True,
        "evaluation_partition": "held_out_test",
        "threshold_partition": "validation_only",
        "patient_disjoint": True,
        "split_manifest_sha256": split_hash,
        "evaluation_records": 200,
        "global_accuracy": 0.95,
        "global_accuracy_definition": "compatibility_subset_exact_match",
        "metric_provenance": "held_out_compatibility_head",
        "confidence_level": 0.95,
        "confidence_interval_method": "bootstrap_percentile",
    }
    metrics_payload = {
        "ontology_version": "test-v3",
        "split_manifest_sha256": split_hash,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "per_class_metrics": {"AF": source_metric},
    }
    metrics_path.write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        ontology_version="test-v3",
        dss={"minimum_precision": 0.85},
        paths=SimpleNamespace(
            root=tmp_path,
            reports=reports,
            manifests=manifests,
        ),
    )
    audited_metric = _metric_from_mapping(
        "AF",
        source_metric,
        context=metrics_payload,
        evidence_source=str(metrics_path),
    ).to_dict()
    audited_metric["selected"] = True
    selection_audit = {
        "well_classified_labels": ["AF"],
        "class_metrics": {"AF": audited_metric},
    }
    rulebook = {
        "dataset": "b1",
        "status": "eligible",
        "rules_eligible": True,
        "ontology_version": config.ontology_version,
        "selection_audit": selection_audit,
        "production_rules": [{"rule_id": "AF-1", "target_label": "AF"}],
    }
    return config, rulebook, split_path, model_path, metrics_path


def test_eligible_rulebook_certificate_binds_actual_evidence_files(tmp_path):
    config, rulebook, split_path, model_path, metrics_path = (
        _eligible_certificate_fixture(tmp_path)
    )

    certified = _certify_eligible_rulebook(config, "b1", rulebook)

    verify_rulebook_eligibility_certificate(certified, config)
    assert certified["eligibility_evidence"] == {
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    }
    assert "eligibility_certificate" not in rulebook


def test_eligibility_certificate_producer_rejects_changed_model(tmp_path):
    config, rulebook, _, model_path, _ = _eligible_certificate_fixture(tmp_path)
    model_path.write_bytes(b"different model bytes")

    with pytest.raises(EligibilityCertificateError, match="model hash"):
        _certify_eligible_rulebook(config, "b1", rulebook)


def test_eligibility_certificate_producer_rejects_changed_metrics(tmp_path):
    config, rulebook, _, _, metrics_path = _eligible_certificate_fixture(tmp_path)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_payload["per_class_metrics"]["AF"]["precision"] = 0.99
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    with pytest.raises(EligibilityCertificateError, match="inconsistent with the metrics"):
        _certify_eligible_rulebook(config, "b1", rulebook)


def test_eligibility_certificate_producer_rejects_empty_rulebook(tmp_path):
    config, rulebook, *_ = _eligible_certificate_fixture(tmp_path)
    rulebook["production_rules"] = []

    with pytest.raises(EligibilityCertificateError, match="empty rulebook"):
        _certify_eligible_rulebook(config, "b1", rulebook)


def test_predicate_executability_requires_real_quality_guard_features():
    predicates = default_medical_predicates()
    available = predicates["AF"].features()

    executable = _predicate_executability(predicates, available)
    missing_quality = _predicate_executability(
        predicates,
        available - {"lead_quality_min_db"},
    )

    assert executable["AF"] is True
    assert executable["Other / unmapped"] is False
    assert missing_quality["AF"] is False


def test_postprocess_rules_adds_physician_predicate_similarity():
    rows = [
        {"qrs_dur_med_ms": 140.0, "r_prime_v1_any": 1, "record_id": "r1"},
        {"qrs_dur_med_ms": 145.0, "r_prime_v1_any": 1, "record_id": "r2"},
    ]
    labels = ["RBBB spectrum", "RBBB spectrum"]
    plan = fit_wedd_discretization(rows, labels, features=["qrs_dur_med_ms", "r_prime_v1_any"], min_support=1)
    system = build_decision_system(rows, labels, plan, object_ids=["r1", "r2"])
    rules = induce_rules(system, plan, min_support=1, use_reducts=False)
    processed = postprocess_rules(rules, default_medical_predicates())
    assert processed
    assert processed[0].predicate_similarity["score"] > 0.0
    assert processed[0].physician_predicate_label is not None


def test_postprocess_rules_accepts_keyword_predicate_library():
    rows = [
        {"qrs_dur_med_ms": 140.0, "r_prime_v1_any": 1, "record_id": "r1"},
        {"qrs_dur_med_ms": 145.0, "r_prime_v1_any": 1, "record_id": "r2"},
    ]
    labels = ["RBBB spectrum", "RBBB spectrum"]
    plan = fit_wedd_discretization(rows, labels, features=["qrs_dur_med_ms", "r_prime_v1_any"], min_support=1)
    system = build_decision_system(rows, labels, plan, object_ids=["r1", "r2"])
    rules = induce_rules(system, plan, min_support=1, use_reducts=False)
    processed = postprocess_rules(rules, predicates=default_medical_predicates(), max_rules_per_label=1)
    assert len(processed) == 1
    assert processed[0].physician_predicate_label is not None


def test_predicate_template_does_not_conjoin_all_supportive_alternatives():
    from tm_ecg.dss.rules import predicate_template_rule

    predicates = default_medical_predicates()
    predicate = predicates["AF"]
    from tm_ecg.dss.models import DecisionSystem

    system = DecisionSystem(
        universe=["af1"],
        conditional_attributes=["af_irregularity_cv", "paced_like_beat_count", "f_wave_power_ratio"],
        decision_attribute="arrhythmia_label",
        attribute_domains={},
        information_function={
            "af1": {
                "af_irregularity_cv": "irregular",
                "paced_like_beat_count": "zero",
                "f_wave_power_ratio": "absent",
            }
        },
        decisions={"af1": "AF"},
    )
    rule = predicate_template_rule(predicate, system)
    states_by_feature = {}
    for condition in rule.antecedents:
        states_by_feature.setdefault(condition.feature, set()).add(condition.state)
    assert states_by_feature["af_irregularity_cv"] == {"irregular"}
    assert len(rule.antecedents) < len(predicate.required + predicate.supportive + predicate.contraindications)
