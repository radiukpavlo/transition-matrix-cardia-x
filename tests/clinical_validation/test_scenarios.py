from __future__ import annotations

from pathlib import Path

import pytest

from tm_ecg.clinical_validation.models import (
    BenchmarkFindingSet,
    CaseIdentity,
    ClinicalFindingSet,
)
from tm_ecg.clinical_validation.scenarios import (
    evaluate_scenarios,
    gate_results,
    load_scenario_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def test_registry_contains_required_and_route_scenarios() -> None:
    _, scenarios = load_scenario_registry(
        ROOT / "clinical_validation/config/scenario_registry_v2.yaml"
    )
    ids = {scenario.scenario_id for scenario in scenarios}
    assert {"b1_unique_exact", "b1_unique_five_family", "b1_unique_binary", "b1_unique_dominant"} <= ids
    for route in ("exact_match", "conflict", "abstention"):
        for projection in ("exact", "five_family", "binary", "dominant"):
            assert f"route_{route}_{projection}" in ids


def test_population_retention_and_gate_on_synthetic_perfect_agreement() -> None:
    registry = {"dominant_priority": ["AF"], "bootstrap_replicates": 10}
    scenario_payload = {
        "scenario_id": "required",
        "description": "test",
        "population": "b1_unique",
        "required_case_count": 2,
        "deduplication": "original_case_id",
        "physician_projection": "dominant",
        "benchmark_projection": "dominant",
        "uncertain_policy": "exclude_uncertain_findings",
        "multilabel_policy": "priority",
        "requirement": "required",
        "estimability_policy": "report",
        "minimum_kappa": 0.70,
        "bootstrap_mode": "cluster_original_case",
        "bootstrap_seed": 7,
    }
    from tm_ecg.clinical_validation.models import ScenarioDefinition

    scenario = ScenarioDefinition.from_dict(scenario_payload)
    identities = [
        CaseIdentity("P001", "P001", "ptbxl", "1", "exact_match", None, 2),
        CaseIdentity("P002", "P002", "ptbxl", "2", "conflict_region", None, 3),
    ]
    physician = [
        ClinicalFindingSet("P001", rhythm=("af",), normality="abnormal"),
        ClinicalFindingSet("P002", rhythm=("sinus",), normality="normal"),
    ]
    benchmark = [
        BenchmarkFindingSet("P001", rhythm=("af",), normality="abnormal"),
        BenchmarkFindingSet("P002", rhythm=("sinus",), normality="normal"),
    ]
    results, ledger, _ = evaluate_scenarios(
        identities, physician, benchmark, registry, [scenario]
    )
    assert results[0]["result"]["kappa"] == 1.0
    assert ledger == []
    code, gate = gate_results(results, {"required_not_estimable": "fail"})
    assert code == 0 and gate["passed"]


def test_v3_registry_expands_and_freezes_all_scenario_contracts() -> None:
    registry, scenarios = load_scenario_registry(
        ROOT / "clinical_validation/config/scenario_registry_v3.yaml"
    )
    assert registry["frozen"] is True
    ids = {scenario.scenario_id for scenario in scenarios}
    assert "raw_b1_unique_exact" in ids
    assert "harmonized_b1_unique_exact" in ids
    required = [item for item in scenarios if item.requirement == "required"]
    assert len(required) == 12
    assert all(item.required_case_ids_hash for item in required)
    assert all(item.physician_coder_hash for item in scenarios)
    assert all(item.benchmark_mapping_hash for item in scenarios)
    assert all(item.projection_contract_hash for item in scenarios)


@pytest.mark.parametrize(
    ("scenario", "result", "expected_code"),
    [
        (
            {
                "scenario_id": "raw_b1_unique_exact",
                "requirement": "audit",
                "expected_baseline_kappa": 0.2,
                "expected_baseline_observed_agreement": 0.5,
                "expected_baseline_status": "ok",
            },
            {"status": "ok", "kappa": 0.1, "observed_agreement": 0.5},
            13,
        ),
        (
            {
                "scenario_id": "harmonized_b1_unique_exact",
                "requirement": "required",
            },
            {"status": "not_estimable", "passes_point_threshold": False},
            14,
        ),
        (
            {
                "scenario_id": "harmonized_b1_unique_exact",
                "requirement": "required",
            },
            {"status": "ok", "passes_point_threshold": False},
            15,
        ),
        (
            {
                "scenario_id": "harmonized_b1_unique_binary",
                "requirement": "required",
            },
            {"status": "ok", "passes_point_threshold": False},
            16,
        ),
    ],
)
def test_v3_gate_has_distinct_failure_codes(
    scenario: dict[str, object],
    result: dict[str, object],
    expected_code: int,
) -> None:
    code, gate = gate_results(
        [{"scenario": scenario, "result": result}],
        {"version": 3, "raw_baseline_tolerance": 1e-6},
    )
    assert code == expected_code
    assert gate["passed"] is False
