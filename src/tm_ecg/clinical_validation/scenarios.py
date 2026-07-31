"""Pre-registered population selection, projection, and scenario evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from tm_ecg.clinical_validation.audit import sha256_payload
from tm_ecg.clinical_validation.bootstrap import cluster_bootstrap_kappa
from tm_ecg.clinical_validation.models import (
    BenchmarkFindingSet,
    CaseIdentity,
    ClinicalFindingSet,
    ScenarioDefinition,
)
from tm_ecg.clinical_validation.ontology import load_json_yaml, projection_value


VALID_PROJECTIONS = frozenset(
    {
        "exact",
        "five_family",
        "binary",
        "dominant",
        "axis:rhythm",
        "axis:ectopy",
        "axis:conduction",
        "axis:repolarization",
        "axis:pacing",
        "axis:quality",
        "axis:interpretability",
        "axis_binary:rhythm",
        "axis_binary:ectopy",
        "axis_binary:conduction",
        "axis_binary:repolarization",
        "axis_binary:pacing",
    }
)


def load_scenario_registry(path: str | Path) -> tuple[dict[str, object], list[ScenarioDefinition]]:
    payload = load_json_yaml(path)
    if payload.get("version") not in {2, 3}:
        raise ValueError("Scenario registry must declare version 2 or 3")
    raw_scenarios = payload.get("scenarios", [])
    if not isinstance(raw_scenarios, list):
        raise ValueError("Scenario registry scenarios must be a list")
    contracts = payload.get("contracts", {})
    contract_map = contracts if isinstance(contracts, Mapping) else {}
    populations = payload.get("population_contracts", {})
    population_map = populations if isinstance(populations, Mapping) else {}
    expanded: list[dict[str, object]] = []
    for item in raw_scenarios:
        if not isinstance(item, Mapping):
            raise ValueError("Each scenario definition must be an object")
        endpoint_version = str(
            item.get("endpoint_version", "clinical_validation_v2")
        )
        population = str(item.get("population", ""))
        endpoint_contract = contract_map.get(endpoint_version, {})
        population_contract = population_map.get(population, {})
        merged: dict[str, object] = {}
        if isinstance(endpoint_contract, Mapping):
            merged.update(endpoint_contract)
        if isinstance(population_contract, Mapping):
            merged.update(population_contract)
        merged.update(item)
        expanded.append(merged)
    scenarios = [ScenarioDefinition.from_dict(item) for item in expanded]
    ids = [item.scenario_id for item in scenarios]
    if not scenarios or len(ids) != len(set(ids)):
        raise ValueError("Scenario registry must contain unique scenario IDs")
    for scenario in scenarios:
        if scenario.physician_projection not in VALID_PROJECTIONS:
            raise ValueError(f"Unknown physician projection: {scenario.physician_projection}")
        if scenario.benchmark_projection not in VALID_PROJECTIONS:
            raise ValueError(f"Unknown benchmark projection: {scenario.benchmark_projection}")
        if scenario.requirement not in {
            "required",
            "audit",
            "diagnostic",
            "not_applicable",
        }:
            raise ValueError(f"Invalid scenario requirement: {scenario.requirement}")
        if scenario.minimum_sample_size is not None:
            if (
                scenario.required_case_count is not None
                and scenario.minimum_sample_size > scenario.required_case_count
            ):
                raise ValueError(
                    f"{scenario.scenario_id} minimum sample exceeds required count"
                )
        if int(str(payload.get("version"))) == 3:
            required_hash_fields = (
                scenario.population_contract,
                scenario.projection_contract_hash,
                scenario.source_contract_hash,
                scenario.physician_coder_hash,
                scenario.benchmark_mapping_hash,
            )
            if not scenario.endpoint_version or not all(required_hash_fields):
                raise ValueError(
                    f"{scenario.scenario_id} is missing its version/hash contract"
                )
    return payload, scenarios


def _population(
    scenario: ScenarioDefinition,
    identities: list[CaseIdentity],
) -> list[CaseIdentity]:
    if scenario.population == "b1_unique":
        rows = [item for item in identities if item.dataset == "ptbxl" and not item.is_repeat]
    elif scenario.population == "b1_all_rows":
        rows = [item for item in identities if item.dataset == "ptbxl"]
    elif scenario.population == "b2_unique":
        rows = [item for item in identities if item.dataset == "ludb" and not item.is_repeat]
    else:
        raise ValueError(f"Unknown scenario population: {scenario.population}")
    if scenario.route:
        rows = [item for item in rows if item.route == scenario.route]
    if scenario.dataset:
        rows = [item for item in rows if item.dataset == scenario.dataset]
    rows.sort(key=lambda item: item.row_ordinal)
    if scenario.required_case_count is not None and len(rows) != scenario.required_case_count:
        raise ValueError(
            f"{scenario.scenario_id} expected {scenario.required_case_count} cases, found {len(rows)}"
        )
    if scenario.minimum_sample_size is not None and len(rows) < scenario.minimum_sample_size:
        raise ValueError(
            f"{scenario.scenario_id} requires at least "
            f"{scenario.minimum_sample_size} cases, found {len(rows)}"
        )
    if scenario.deduplication == "original_case_id":
        original_ids = [item.original_case_id for item in rows]
        if len(original_ids) != len(set(original_ids)):
            raise ValueError(f"{scenario.scenario_id} contains duplicate original case identities")
    return rows


def evaluate_scenarios(
    identities: list[CaseIdentity],
    physician_findings: list[ClinicalFindingSet],
    benchmark_findings: list[BenchmarkFindingSet],
    registry: Mapping[str, object],
    scenarios: list[ScenarioDefinition],
    *,
    bootstrap_replicates_override: int | None = None,
    raw_physician_findings: list[ClinicalFindingSet] | None = None,
    raw_benchmark_findings: list[BenchmarkFindingSet] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[float]]]:
    physician = {item.case_id: item for item in physician_findings}
    benchmark = {item.case_id: item for item in benchmark_findings}
    identity_ids = {item.workbook_case_id for item in identities}
    if set(physician) != identity_ids or set(benchmark) != identity_ids:
        raise ValueError("Physician, benchmark, and identity case sets must be identical")
    raw_physician = {
        item.case_id: item
        for item in (
            raw_physician_findings
            if raw_physician_findings is not None
            else physician_findings
        )
    }
    raw_benchmark = {
        item.case_id: item
        for item in (
            raw_benchmark_findings
            if raw_benchmark_findings is not None
            else benchmark_findings
        )
    }
    if set(raw_physician) != identity_ids or set(raw_benchmark) != identity_ids:
        raise ValueError("Raw physician, benchmark, and identity case sets must be identical")
    raw_priority = registry.get("dominant_priority", [])
    priority = tuple(
        str(item) for item in (raw_priority if isinstance(raw_priority, list) else [])
    )
    expected_primary_ids: tuple[str, ...] | None = None
    results: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    distributions: dict[str, list[float]] = {}
    for scenario in scenarios:
        if scenario.requirement == "not_applicable":
            results.append(
                {
                    "scenario": scenario.to_dict(),
                    "result": {
                        "status": "not_applicable",
                        "reason": scenario.description,
                        "sample_size": 0,
                        "kappa": None,
                    },
                }
            )
            continue
        rows = _population(scenario, identities)
        current_ids = tuple(sorted(item.original_case_id for item in rows))
        current_ids_hash = sha256_payload(list(current_ids))
        if (
            scenario.required_case_ids_hash is not None
            and current_ids_hash != scenario.required_case_ids_hash
        ):
            raise ValueError(
                f"{scenario.scenario_id} case identity hash mismatch; "
                f"expected={scenario.required_case_ids_hash}, actual={current_ids_hash}"
            )
        if scenario.population == "b1_unique":
            if expected_primary_ids is None:
                expected_primary_ids = current_ids
            elif scenario.route is None and current_ids != expected_primary_ids:
                missing = sorted(set(expected_primary_ids) - set(current_ids))
                extra = sorted(set(current_ids) - set(expected_primary_ids))
                raise ValueError(
                    f"Primary-case retention failure for {scenario.scenario_id}; missing={missing}, extra={extra}"
                )
        include_uncertain = scenario.uncertain_policy == "include_uncertain_findings"
        reference: list[str] = []
        comparison: list[str] = []
        case_ids: list[str] = []
        clusters: list[str] = []
        observations: list[dict[str, object]] = []
        scenario_physician = (
            raw_physician
            if scenario.endpoint_version.startswith("raw_immutable")
            else physician
        )
        scenario_benchmark = (
            raw_benchmark
            if scenario.endpoint_version.startswith("raw_immutable")
            else benchmark
        )
        for identity in rows:
            case_id = identity.workbook_case_id
            bench = projection_value(
                scenario_benchmark[case_id],
                scenario.benchmark_projection,
                include_uncertain=include_uncertain,
                dominant_priority=priority,
            )
            phys = projection_value(
                scenario_physician[case_id],
                scenario.physician_projection,
                include_uncertain=include_uncertain,
                dominant_priority=priority,
            )
            reference.append(bench)
            comparison.append(phys)
            case_ids.append(case_id)
            clusters.append(identity.original_case_id)
            observations.append(
                {
                    "case_id": case_id,
                    "original_case_id": identity.original_case_id,
                    "dataset": identity.dataset,
                    "route": identity.route,
                    "physician_projection": phys,
                    "benchmark_projection": bench,
                    "agrees": phys == bench,
                }
            )
        replicate_count = (
            bootstrap_replicates_override
            if bootstrap_replicates_override is not None
            else (
                scenario.bootstrap_replicates
                if scenario.bootstrap_replicates is not None
                else int(str(registry.get("bootstrap_replicates", 1000)))
            )
        )
        result, bootstrap_values = cluster_bootstrap_kappa(
            reference,
            comparison,
            case_ids,
            clusters,
            replicates=replicate_count,
            seed=scenario.bootstrap_seed,
            threshold=scenario.minimum_kappa,
        )
        distributions[scenario.scenario_id] = bootstrap_values
        result_payload = result.to_dict()
        result_payload["passes_point_threshold"] = (
            result.kappa is not None and result.kappa >= scenario.minimum_kappa
        )
        results.append({"scenario": scenario.to_dict(), "result": result_payload})
        if result.kappa is None or result.kappa < scenario.minimum_kappa:
            for observation in observations:
                if observation["agrees"]:
                    continue
                case_id = str(observation["case_id"])
                ledger.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "disagreement_family": (
                            scenario.physician_projection.split(":", 1)[-1]
                            if scenario.physician_projection.startswith("axis:")
                            else scenario.physician_projection
                        ),
                        **observation,
                        "physician_finding_set": json.dumps(
                            scenario_physician[case_id].to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "benchmark_finding_set": json.dumps(
                            scenario_benchmark[case_id].to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "physician_coding_trace": json.dumps(
                            [
                                item.to_dict()
                                for item in scenario_physician[case_id].evidence
                            ],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "benchmark_coding_trace": json.dumps(
                            {
                                "source_labels": scenario_benchmark[case_id].source_labels,
                                "accepted_source_labels": scenario_benchmark[
                                    case_id
                                ].accepted_source_labels,
                                "uncertain_source_labels": scenario_benchmark[
                                    case_id
                                ].uncertain_source_labels,
                                "ignored_source_labels": scenario_benchmark[
                                    case_id
                                ].ignored_source_labels,
                                "mapping_rule_ids": scenario_benchmark[
                                    case_id
                                ].mapping_rule_ids,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
    disagreement_frequency: dict[str, int] = {}
    for row in ledger:
        case_id = str(row["case_id"])
        disagreement_frequency[case_id] = disagreement_frequency.get(case_id, 0) + 1
    for row in ledger:
        row["case_disagreement_frequency"] = disagreement_frequency[str(row["case_id"])]
    ledger.sort(
        key=lambda item: (
            -int(str(item["case_disagreement_frequency"])),
            str(item["disagreement_family"]),
            str(item["case_id"]),
            str(item["scenario_id"]),
        )
    )
    return results, ledger, distributions


def gate_results(
    scenario_results: list[Mapping[str, object]],
    acceptance_policy: Mapping[str, object],
) -> tuple[int, dict[str, object]]:
    if int(str(acceptance_policy.get("version", 2))) == 3:
        return _gate_results_v3(scenario_results, acceptance_policy)
    below: list[str] = []
    non_estimable: list[str] = []
    for item in scenario_results:
        raw_scenario = item["scenario"]
        raw_result = item["result"]
        scenario = dict(raw_scenario) if isinstance(raw_scenario, Mapping) else {}
        result = dict(raw_result) if isinstance(raw_result, Mapping) else {}
        if scenario.get("requirement") != "required":
            continue
        status = str(result.get("status"))
        if status in {"empty", "insufficient", "not_estimable"}:
            if acceptance_policy.get("required_not_estimable", "fail") == "fail":
                non_estimable.append(str(scenario["scenario_id"]))
            continue
        if not bool(result.get("passes_point_threshold", False)):
            below.append(str(scenario["scenario_id"]))
    if non_estimable:
        return 6, {"passed": False, "below_threshold": below, "required_not_estimable": non_estimable}
    if below:
        return 5, {"passed": False, "below_threshold": below, "required_not_estimable": []}
    return 0, {"passed": True, "below_threshold": [], "required_not_estimable": []}


def _gate_results_v3(
    scenario_results: list[Mapping[str, object]],
    acceptance_policy: Mapping[str, object],
) -> tuple[int, dict[str, object]]:
    raw_regressions: list[str] = []
    non_estimable: list[str] = []
    exact_below: list[str] = []
    other_below: list[str] = []
    tolerance = float(str(acceptance_policy.get("raw_baseline_tolerance", 1e-6)))

    for item in scenario_results:
        raw_scenario = item.get("scenario", {})
        raw_result = item.get("result", {})
        if not isinstance(raw_scenario, Mapping) or not isinstance(
            raw_result, Mapping
        ):
            continue
        scenario_id = str(raw_scenario.get("scenario_id", ""))
        requirement = str(raw_scenario.get("requirement", "diagnostic"))
        status = str(raw_result.get("status", ""))

        if requirement == "audit":
            expected_status = raw_scenario.get("expected_baseline_status")
            expected_kappa = raw_scenario.get("expected_baseline_kappa")
            expected_agreement = raw_scenario.get(
                "expected_baseline_observed_agreement"
            )
            regression = False
            if expected_status not in {None, ""} and status != str(expected_status):
                regression = True
            if expected_kappa is not None:
                actual_kappa = raw_result.get("kappa")
                regression = regression or actual_kappa is None or abs(
                    float(str(actual_kappa)) - float(str(expected_kappa))
                ) > tolerance
            if expected_agreement is not None:
                actual_agreement = raw_result.get("observed_agreement")
                regression = regression or actual_agreement is None or abs(
                    float(str(actual_agreement)) - float(str(expected_agreement))
                ) > tolerance
            if regression:
                raw_regressions.append(scenario_id)
            continue

        if requirement != "required":
            continue
        if status in {"empty", "insufficient", "not_estimable"}:
            non_estimable.append(scenario_id)
            continue
        if not bool(raw_result.get("passes_point_threshold", False)):
            if scenario_id == "harmonized_b1_unique_exact":
                exact_below.append(scenario_id)
            else:
                other_below.append(scenario_id)

    gate = {
        "passed": not (
            raw_regressions or non_estimable or exact_below or other_below
        ),
        "raw_baseline_regressions": raw_regressions,
        "required_not_estimable": non_estimable,
        "harmonized_exact_below_threshold": exact_below,
        "other_required_below_threshold": other_below,
    }
    if raw_regressions:
        return 13, gate
    if non_estimable:
        return 14, gate
    if exact_below:
        return 15, gate
    if other_below:
        return 16, gate
    return 0, gate
