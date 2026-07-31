from __future__ import annotations

from tm_ecg.stages.release_audit_v3 import build_release_audit


def _scenario(
    scenario_id: str,
    *,
    endpoint: str,
    requirement: str,
    kappa: float,
    minimum: float,
    expected: float | None = None,
) -> dict[str, object]:
    scenario = {
        "scenario_id": scenario_id,
        "endpoint_version": endpoint,
        "requirement": requirement,
        "population": "b1_unique",
        "route": None,
        "minimum_kappa": minimum,
    }
    if expected is not None:
        scenario["expected_baseline_kappa"] = expected
        scenario["expected_baseline_status"] = "ok"
    return {
        "scenario": scenario,
        "result": {
            "sample_size": 100,
            "status": "ok",
            "kappa": kappa,
        },
    }


def _passing_inputs() -> dict[str, object]:
    scenarios = {
        "scenario_results": [
            _scenario(
                "raw_exact",
                endpoint="raw_immutable_v2",
                requirement="audit",
                kappa=0.191116,
                minimum=0.0,
                expected=0.191116,
            ),
            _scenario(
                "harmonized_b1_unique_exact",
                endpoint="harmonized_v3",
                requirement="required",
                kappa=0.72,
                minimum=0.70,
            ),
            _scenario(
                "harmonized_axis",
                endpoint="harmonized_v3",
                requirement="required",
                kappa=0.65,
                minimum=0.60,
            ),
        ]
    }
    return {
        "source_gate": {"passed": True, "workbook_write_attempts": 0},
        "coding_gate": {"passed": True},
        "scenario_bundle": scenarios,
        "compatibility_metrics": {
            "evaluation_partition": "sealed_internal_confirmatory",
            "confirmatory_labels_opened": True,
            "calibration_gate_passed": True,
            "selected_metrics": {
                "records": 1000,
                "compatibility_subset_exact_match": 0.91,
                "best_class_f1": 0.85,
            },
            "selection_constraints": {
                "rare_label_recall_floors": {"passed": True}
            },
        },
        "environment_audit": {
            "dependency_lock": {"valid": True},
            "git": {"commit": "abc", "dirty": False},
        },
        "clinical_artifact_manifest_present": True,
        "transition_gate": {"passed": True},
        "dss_gate": {"passed": True},
    }


def test_release_audit_v3_passes_only_complete_confirmatory_evidence() -> None:
    code, audit = build_release_audit(**_passing_inputs())
    assert code == 0
    assert audit["release_eligible"] is True
    assert all(audit["gates"].values())


def test_release_audit_v3_reports_first_ordered_failure() -> None:
    inputs = _passing_inputs()
    inputs["source_gate"] = {
        "passed": False,
        "workbook_write_attempts": 0,
    }
    code, audit = build_release_audit(**inputs)
    assert code == 10
    assert audit["first_failed_gate"] == "G0_source_authority"


def test_development_metrics_cannot_satisfy_confirmatory_gate() -> None:
    inputs = _passing_inputs()
    metrics = dict(inputs["compatibility_metrics"])
    metrics["evaluation_partition"] = "development_validation"
    metrics["confirmatory_labels_opened"] = False
    inputs["compatibility_metrics"] = metrics
    code, audit = build_release_audit(**inputs)
    assert code == 18
    assert audit["gates"]["G6_compatibility_best_f1"] is False
    assert audit["gates"]["G7_compatibility_exact"] is False
