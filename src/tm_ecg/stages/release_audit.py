"""Build the hash-anchored CARDIA-X strict release audit."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from typing import cast
import xml.etree.ElementTree as ET

from tm_ecg.config import ProjectConfig
from tm_ecg.dss.eligibility import (
    EligibilityCertificateError,
    verify_rulebook_eligibility_certificate,
)
from tm_ecg.io.common import sha256_file, write_json
from tm_ecg.io.readers import find_table, read_table_frame


_EXACT_PHYSICIAN_SCENARIO_ID = "b1_unique_exact"
_PRIMARY_PHYSICIAN_CASE_COUNT = 100
_NON_ESTIMABLE_STATUSES = frozenset(
    {"empty", "insufficient", "not_estimable", "not_applicable"}
)
_PREREGISTERED_EXEMPTIONS = frozenset({"exempt", "not_applicable"})
_SCENARIO_BINDING_FIELDS = (
    "scenario_id",
    "population",
    "required_case_count",
    "deduplication",
    "physician_projection",
    "benchmark_projection",
    "uncertain_policy",
    "multilabel_policy",
    "requirement",
    "estimability_policy",
    "minimum_kappa",
    "bootstrap_mode",
    "bootstrap_seed",
    "route",
    "dataset",
)
_DEFAULT_CLINICAL_RUN_ID = "final_audited_20260721"
_DEFAULT_EVIDENCE_TAG = "20260721"
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _safe_component(value: object, *, field: str) -> str:
    component = str(value).strip()
    if _SAFE_COMPONENT.fullmatch(component) is None:
        raise ValueError(
            f"{field} must contain only letters, digits, '.', '_' or '-'"
        )
    return component


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _artifact(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _manifest_output_binding(
    artifact_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    expected = sha256_file(artifact_path) if artifact_path.exists() else None
    declared: object = None
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        raw_hashes = manifest.get("output_hashes")
        if isinstance(raw_hashes, Mapping):
            declared = raw_hashes.get(artifact_path.name)
    return {
        "artifact": str(artifact_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "declared": declared,
        "expected": expected,
        "passed": bool(expected is not None and declared == expected),
    }


def _rule_review_validation(
    rule_review_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Require a complete, hash-bound physician review of all ten rules."""

    payload = _read_json(rule_review_path)
    binding = _manifest_output_binding(rule_review_path, manifest_path)
    expected_rules = _integer(payload.get("expected_rules"))
    expected_b1_rules = _integer(payload.get("expected_b1_rules"))
    expected_b2_rules = _integer(payload.get("expected_b2_rules"))
    mean_likert = _finite_float(payload.get("mean_likert"))
    completeness_checks = {
        "status_complete": payload.get("status") == "complete",
        "expected_rules_10": expected_rules == 10,
        "expected_b1_rules_8": expected_b1_rules == 8,
        "expected_b2_rules_2": expected_b2_rules == 2,
        "template_rows_complete": _integer(payload.get("template_rows")) == 10,
        "rule_ids_complete": _integer(payload.get("supplied_rule_ids")) == 10,
        "definitions_complete": (
            _integer(payload.get("complete_rule_definitions")) == 10
        ),
        "rules_fully_scored": _integer(payload.get("fully_scored_rules")) == 10,
        "likert_cells_complete": (
            _integer(payload.get("observed_likert_cells")) == 40
        ),
        "mean_likert_valid": (
            mean_likert is not None and 1.0 <= mean_likert <= 5.0
        ),
    }
    complete = all(completeness_checks.values())
    passed = bool(binding["passed"] and complete and mean_likert is not None and mean_likert > 4.0)
    return {
        "status": payload.get("status"),
        "mean_likert": mean_likert,
        "threshold": 4.0,
        "comparison": ">",
        "complete": complete,
        "completeness_checks": completeness_checks,
        "manifest_binding": binding,
        "passed": passed,
        "artifact": _artifact(rule_review_path),
    }


def _scenario_bundle(
    path: Path,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    payload = _read_json(path)
    raw_items = payload.get("scenario_results")
    if not isinstance(raw_items, list):
        raise ValueError(f"Scenario bundle {path} must contain scenario_results")
    definitions: dict[str, dict[str, object]] = {}
    results: dict[str, dict[str, object]] = {}
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"Scenario result {index} in {path} is not an object")
        raw_scenario = raw_item.get("scenario")
        raw_result = raw_item.get("result")
        if not isinstance(raw_scenario, Mapping) or not isinstance(raw_result, Mapping):
            raise ValueError(f"Scenario result {index} in {path} is malformed")
        scenario = dict(raw_scenario)
        result = dict(raw_result)
        scenario_id = str(scenario.get("scenario_id", "")).strip()
        if not scenario_id:
            raise ValueError(f"Scenario result {index} in {path} has no scenario_id")
        if scenario_id in results:
            raise ValueError(f"Duplicate scenario result: {scenario_id}")
        definitions[scenario_id] = scenario
        results[scenario_id] = result
    return payload, definitions, results


def _scenario_map(path: Path) -> dict[str, dict[str, object]]:
    return _scenario_bundle(path)[2]


def _scenario_registry_map(path: Path) -> dict[str, dict[str, object]]:
    payload = _read_json(path)
    if payload.get("version") != 2:
        raise ValueError("Physician scenario registry must declare version 2")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("Physician scenario registry must contain scenarios")
    scenarios: dict[str, dict[str, object]] = {}
    for index, raw_scenario in enumerate(raw_scenarios):
        if not isinstance(raw_scenario, Mapping):
            raise ValueError(f"Registry scenario {index} is not an object")
        scenario = dict(raw_scenario)
        scenario_id = str(scenario.get("scenario_id", "")).strip()
        if not scenario_id:
            raise ValueError(f"Registry scenario {index} has no scenario_id")
        if scenario_id in scenarios:
            raise ValueError(f"Duplicate registry scenario: {scenario_id}")
        requirement = str(scenario.get("requirement", "")).strip().lower()
        if requirement not in {
            "required",
            "diagnostic",
            "not_applicable",
            "exempt",
        }:
            raise ValueError(
                f"Registry scenario {scenario_id} has invalid requirement: {requirement}"
            )
        scenarios[scenario_id] = scenario
    return scenarios


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(str(value))
    except (TypeError, ValueError):
        return None
    return numeric


def _is_preregistered_exemption(scenario: Mapping[str, object]) -> bool:
    requirement = str(scenario.get("requirement", "")).strip().lower()
    estimability = str(scenario.get("estimability_policy", "")).strip().lower()
    return (
        requirement in _PREREGISTERED_EXEMPTIONS
        or estimability in _PREREGISTERED_EXEMPTIONS
    )


def _physician_primary_validation(
    registry_path: Path,
    scenario_results_path: Path,
) -> dict[str, object]:
    """Audit every registry-required physician scenario without silent omissions."""

    registry = _scenario_registry_map(registry_path)
    bundle, embedded_scenarios, scenario_results = _scenario_bundle(
        scenario_results_path
    )
    registry_hash = sha256_file(registry_path)
    results_hash = sha256_file(scenario_results_path)
    integrity_failures: list[str] = []
    binding_checks: dict[str, dict[str, object]] = {}

    bundle_registry_hash = bundle.get("registry_hash")
    bundle_hash_matches = bundle_registry_hash == registry_hash
    binding_checks["scenario_results_registry_hash"] = {
        "declared": bundle_registry_hash,
        "expected": registry_hash,
        "passed": bundle_hash_matches,
    }
    if not bundle_hash_matches:
        integrity_failures.append("scenario_results_registry_hash_mismatch")

    manifest_path = scenario_results_path.parent / "run_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        manifest_registry_hash = manifest.get("scenario_registry_hash")
        manifest_registry_matches = manifest_registry_hash == registry_hash
        binding_checks["run_manifest_registry_hash"] = {
            "declared": manifest_registry_hash,
            "expected": registry_hash,
            "passed": manifest_registry_matches,
        }
        if not manifest_registry_matches:
            integrity_failures.append("run_manifest_registry_hash_mismatch")

        raw_output_hashes = manifest.get("output_hashes")
        output_hashes = (
            dict(raw_output_hashes) if isinstance(raw_output_hashes, Mapping) else {}
        )
        manifest_results_hash = output_hashes.get(scenario_results_path.name)
        manifest_results_match = manifest_results_hash == results_hash
        binding_checks["run_manifest_scenario_results_hash"] = {
            "declared": manifest_results_hash,
            "expected": results_hash,
            "passed": manifest_results_match,
        }
        if not manifest_results_match:
            integrity_failures.append("run_manifest_scenario_results_hash_mismatch")
    else:
        binding_checks["run_manifest_present"] = {
            "declared": False,
            "expected": True,
            "passed": False,
        }
        integrity_failures.append("run_manifest_missing")

    unregistered_results = sorted(set(scenario_results) - set(registry))
    if unregistered_results:
        integrity_failures.append("unregistered_scenario_results")

    definition_mismatches: dict[str, list[str]] = {}
    for scenario_id in sorted(set(registry) & set(embedded_scenarios)):
        expected = registry[scenario_id]
        embedded = embedded_scenarios[scenario_id]
        mismatched_fields = [
            field
            for field in _SCENARIO_BINDING_FIELDS
            if expected.get(field) != embedded.get(field)
        ]
        if mismatched_fields:
            definition_mismatches[scenario_id] = mismatched_fields
    if definition_mismatches:
        integrity_failures.append("embedded_scenario_registry_mismatch")

    required_ids = [
        scenario_id
        for scenario_id, scenario in registry.items()
        if str(scenario.get("requirement", "")).strip().lower() == "required"
    ]
    preregistered_exemptions = sorted(
        scenario_id
        for scenario_id, scenario in registry.items()
        if _is_preregistered_exemption(scenario)
    )
    if _EXACT_PHYSICIAN_SCENARIO_ID not in required_ids:
        integrity_failures.append("required_exact_scenario_missing_from_registry")

    missing_required = sorted(set(required_ids) - set(scenario_results))
    if missing_required:
        integrity_failures.append("required_scenario_results_missing")

    primary_results: dict[str, dict[str, object]] = {}
    required_non_estimable: list[str] = []
    below_threshold: list[str] = []
    case_sets: dict[str, frozenset[str]] = {}
    for scenario_id in required_ids:
        registered = registry[scenario_id]
        result = scenario_results.get(scenario_id)
        target_floor = 0.70 if scenario_id == _EXACT_PHYSICIAN_SCENARIO_ID else 0.60
        registered_threshold = _finite_float(registered.get("minimum_kappa"))
        threshold = max(target_floor, registered_threshold or target_floor)
        exempt = _is_preregistered_exemption(registered)
        failure_reasons: list[str] = []

        if registered_threshold is None or not 0.0 <= registered_threshold <= 1.0:
            failure_reasons.append("invalid_registry_minimum_kappa")

        registered_case_count = _integer(registered.get("required_case_count"))
        if registered_case_count != _PRIMARY_PHYSICIAN_CASE_COUNT:
            failure_reasons.append("registry_primary_case_count_not_100")

        embedded_mismatches = definition_mismatches.get(scenario_id, [])
        if embedded_mismatches:
            failure_reasons.append("embedded_scenario_registry_mismatch")

        status: str | None = None
        kappa: float | None = None
        confidence_interval: object = None
        case_count = 0
        unique_case_count = 0
        sample_size: int | None = None
        non_estimable = False
        if result is None:
            failure_reasons.append("missing_result")
        else:
            status = str(result.get("status", "")).strip().lower()
            kappa = _finite_float(result.get("kappa"))
            confidence_interval = result.get("confidence_interval")
            raw_case_ids = result.get("case_ids")
            if isinstance(raw_case_ids, list):
                case_ids = [str(value) for value in raw_case_ids]
                case_count = len(case_ids)
                unique_case_count = len(set(case_ids))
                if unique_case_count == _PRIMARY_PHYSICIAN_CASE_COUNT:
                    case_sets[scenario_id] = frozenset(case_ids)
            else:
                failure_reasons.append("case_ids_missing_or_invalid")
            if case_count != _PRIMARY_PHYSICIAN_CASE_COUNT:
                failure_reasons.append("case_count_not_100")
            if unique_case_count != _PRIMARY_PHYSICIAN_CASE_COUNT:
                failure_reasons.append("unique_case_count_not_100")

            if "sample_size" in result:
                sample_size = _integer(result.get("sample_size"))
                if sample_size != _PRIMARY_PHYSICIAN_CASE_COUNT:
                    failure_reasons.append("sample_size_not_100")

            non_estimable = status in _NON_ESTIMABLE_STATUSES
            if non_estimable:
                if kappa is not None:
                    failure_reasons.append("non_estimable_status_has_kappa")
                if not exempt:
                    failure_reasons.append("required_non_estimable")
                    required_non_estimable.append(scenario_id)
            elif status != "ok":
                failure_reasons.append("invalid_status")
            elif kappa is None:
                failure_reasons.append("missing_or_non_finite_kappa")

        threshold_passed: bool | None
        if non_estimable and exempt:
            threshold_passed = None
        else:
            threshold_passed = kappa is not None and kappa > threshold
            if status == "ok" and kappa is not None and not threshold_passed:
                below_threshold.append(scenario_id)

        integrity_passed = not failure_reasons
        release_gate_passed = integrity_passed and (
            threshold_passed is True or (non_estimable and exempt)
        )
        primary_results[scenario_id] = {
            "status": status,
            "kappa": kappa,
            "confidence_interval": confidence_interval,
            "case_count": case_count,
            "unique_case_count": unique_case_count,
            "sample_size": sample_size,
            "required_case_count": _PRIMARY_PHYSICIAN_CASE_COUNT,
            "requested_target_floor": target_floor,
            "preregistered_threshold": registered_threshold,
            "release_threshold": threshold,
            "threshold_comparison": ">",
            "threshold_passed": threshold_passed,
            "preregistered_exemption": exempt,
            "integrity_passed": integrity_passed,
            "release_gate_passed": release_gate_passed,
            "failure_reasons": failure_reasons,
        }
        integrity_failures.extend(
            f"{scenario_id}:{reason}" for reason in failure_reasons
        )

    canonical_case_set = case_sets.get(_EXACT_PHYSICIAN_SCENARIO_ID)
    if canonical_case_set is None and case_sets:
        canonical_case_set = next(iter(case_sets.values()))
    inconsistent_case_sets = sorted(
        scenario_id
        for scenario_id, case_set in case_sets.items()
        if canonical_case_set is not None and case_set != canonical_case_set
    )
    for scenario_id in inconsistent_case_sets:
        result = primary_results[scenario_id]
        raw_failures = result.get("failure_reasons")
        failures = (
            [str(value) for value in raw_failures]
            if isinstance(raw_failures, list)
            else []
        )
        failures.append("primary_case_set_mismatch")
        result["failure_reasons"] = failures
        result["integrity_passed"] = False
        result["release_gate_passed"] = False
        integrity_failures.append(f"{scenario_id}:primary_case_set_mismatch")

    integrity_passed = not integrity_failures
    exact_result = primary_results.get(_EXACT_PHYSICIAN_SCENARIO_ID)
    other_ids = [
        scenario_id
        for scenario_id in required_ids
        if scenario_id != _EXACT_PHYSICIAN_SCENARIO_ID
    ]
    other_kappas: list[float] = []
    for scenario_id in other_ids:
        result = primary_results[scenario_id]
        kappa = _finite_float(result.get("kappa"))
        if result.get("status") == "ok" and kappa is not None:
            other_kappas.append(kappa)
    binding_passed = all(
        bool(check["passed"]) for check in binding_checks.values()
    )
    exact_passed = bool(
        binding_passed
        and exact_result is not None
        and exact_result["release_gate_passed"]
    )
    other_passed = bool(
        binding_passed
        and other_ids
        and all(primary_results[item]["release_gate_passed"] for item in other_ids)
    )
    return {
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_hash,
        "scenario_results_sha256": results_hash,
        "run_manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else None,
        "binding_checks": binding_checks,
        "binding_passed": binding_passed,
        "required_scenario_ids": required_ids,
        "other_required_scenario_ids": other_ids,
        "preregistered_exemptions": preregistered_exemptions,
        "missing_required_scenarios": missing_required,
        "unregistered_result_scenarios": unregistered_results,
        "definition_mismatches": definition_mismatches,
        "inconsistent_primary_case_sets": inconsistent_case_sets,
        "required_non_estimable_scenarios": sorted(set(required_non_estimable)),
        "below_threshold_scenarios": sorted(set(below_threshold)),
        "integrity_failures": integrity_failures,
        "integrity_passed": integrity_passed,
        "exact_passed": exact_passed,
        "other_passed": other_passed,
        "minimum_other_kappa": min(other_kappas) if other_kappas else None,
        "primary_results": primary_results,
    }


def _quality_coverage(config: ProjectConfig, dataset: str) -> dict[str, object]:
    thresholds = {
        "lead_quality_min_db": float(config.thresholds["feature_quality_min_db"]),
        "delineation_confidence": float(
            config.thresholds["delineation_confidence_minimum"]
        ),
        "analyzable_duration_s": float(
            config.thresholds["minimum_af_analyzable_duration_s"]
        ),
    }
    partitions: dict[str, object] = {}
    for split in ("train", "val", "test"):
        path = find_table(config.paths.features, f"{dataset.upper()}_raw_{split}")
        if path is None:
            raise FileNotFoundError(f"Missing {dataset} raw {split} artifact")
        frame = read_table_frame(path)
        mask = frame.index.to_series().notna()
        condition_counts: dict[str, int] = {}
        for feature, minimum in thresholds.items():
            numeric = frame[feature].astype(float)
            condition = numeric.notna() & numeric.ge(minimum)
            condition_counts[feature] = int(condition.sum())
            mask &= condition
        partitions[split] = {
            "records": len(frame),
            "quality_eligible_records": int(mask.sum()),
            "quality_eligible_coverage": float(mask.mean()),
            "condition_pass_counts": condition_counts,
            "artifact": _artifact(path),
        }
    return {
        "definition": "all_three_waveform_measured_DSS_quality_guards",
        "thresholds": thresholds,
        "partitions": partitions,
        "minimum_required_for_DSS": float(
            config.dss["minimum_analyzable_coverage"]
        ),
        "does_not_imply_model_abstention": True,
    }


def _verification_evidence(
    config: ProjectConfig,
    *,
    evidence_tag: str = _DEFAULT_EVIDENCE_TAG,
) -> dict[str, object]:
    directory = config.paths.reports / "verification"
    safe_tag = _safe_component(evidence_tag, field="evidence_tag")
    pytest_path = directory / f"pytest_full_{safe_tag}.xml"
    ruff_path = directory / f"ruff_full_{safe_tag}.json"
    compile_path = directory / f"compileall_{safe_tag}.json"
    evidence: dict[str, object] = {}
    if pytest_path.exists():
        root = ET.parse(pytest_path).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            raise RuntimeError("Cannot parse pytest JUnit evidence")
        cases = root.findall(".//testcase")
        if root.tag == "testcase":
            cases = [root]
        if not cases:
            raise RuntimeError("Pytest JUnit evidence contains no test cases")
        tests = len(cases)
        failures = sum(case.find("failure") is not None for case in cases)
        errors = sum(case.find("error") is not None for case in cases)
        skipped = sum(case.find("skipped") is not None for case in cases)
        declared_tests = int(suite.attrib.get("tests", tests))
        evidence["pytest"] = {
            "tests": tests,
            "count_source": "testcase_nodes",
            "declared_tests": declared_tests,
            "declared_count_matches": declared_tests == tests,
            "additional_declared_checks": max(declared_tests - tests, 0),
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "passed": tests - failures - errors - skipped,
            "status": "passed" if failures == 0 and errors == 0 else "failed",
            "artifact": _artifact(pytest_path),
        }
    if ruff_path.exists():
        issues = json.loads(ruff_path.read_text(encoding="utf-8"))
        evidence["ruff"] = {
            "issues": len(issues),
            "status": "passed" if not issues else "failed",
            "artifact": _artifact(ruff_path),
        }
    if compile_path.exists():
        payload = _read_json(compile_path)
        evidence["compileall"] = {
            **payload,
            "artifact": _artifact(compile_path),
        }
    return evidence


def build_audit(
    config: ProjectConfig,
    *,
    clinical_run_id: str = _DEFAULT_CLINICAL_RUN_ID,
    evidence_tag: str = _DEFAULT_EVIDENCE_TAG,
) -> dict[str, object]:
    clinical_run_id = _safe_component(clinical_run_id, field="clinical_run_id")
    evidence_tag = _safe_component(evidence_tag, field="evidence_tag")
    metrics_dir = config.paths.reports / "metrics"
    dss_dir = config.paths.reports / "dss"
    compatibility_path = metrics_dir / "b1_classification_metrics.json"
    validation_lock_path = metrics_dir / "b1_validation_model_lock.json"
    ludb_path = metrics_dir / "ludb_fiducial_validation.json"
    transition_path = metrics_dir / "transition_validation_metrics.json"
    physician_path = (
        config.paths.root
        / "clinical_validation"
        / "results"
        / clinical_run_id
        / "scenario_results.json"
    )
    physician_registry_path = (
        config.paths.root
        / "clinical_validation"
        / "config"
        / "scenario_registry_v2.yaml"
    )
    physician_manifest_path = physician_path.parent / "run_manifest.json"
    rule_review_path = physician_path.parent / "rule_review_summary.json"
    abstention_path = physician_path.parent / "abstention_concordance.json"
    assisted_review_path = physician_path.parent / "assisted_review_summary.json"
    semantic_path = (
        config.paths.root
        / "clinical_validation"
        / "results"
        / "physician_kappa_semantic_development_20260721"
        / "scenario_results.json"
    )

    compatibility = _read_json(compatibility_path)
    validation_lock = _read_json(validation_lock_path)
    ludb = _read_json(ludb_path)
    transition = _read_json(transition_path)
    physician_evidence = _physician_primary_validation(
        physician_registry_path,
        physician_path,
    )
    rule_review_evidence = _rule_review_validation(
        rule_review_path,
        physician_manifest_path,
    )
    abstention_evidence = _read_json(abstention_path)
    assisted_review_evidence = _read_json(assisted_review_path)
    semantic = _scenario_map(semantic_path)
    per_class = dict(compatibility["per_class_metrics"])
    best_f1 = max(float(dict(value)["f1"]) for value in per_class.values())
    exact_accuracy = float(compatibility["compatibility_subset_exact_match"])
    bitwise_accuracy = float(compatibility["per_label_bitwise_accuracy"])

    physician_results = cast(
        dict[str, dict[str, object]],
        physician_evidence["primary_results"],
    )
    other_required_physician_ids = cast(
        list[str],
        physician_evidence["other_required_scenario_ids"],
    )

    dss_results: dict[str, object] = {}
    for dataset in ("b1", "b2"):
        failure_path = dss_dir / f"dss_selection_failure_{dataset}.json"
        rulebook_path = dss_dir / f"dss_rulebook_{dataset}.json"
        failure = _read_json(failure_path)
        rulebook = _read_json(rulebook_path)
        rules_eligible = bool(rulebook["rules_eligible"])
        certificate_valid = False
        certificate_error: str | None = None
        if rules_eligible:
            try:
                verify_rulebook_eligibility_certificate(rulebook, config)
                certificate_valid = True
            except EligibilityCertificateError as exc:
                certificate_error = str(exc)
        dss_results[dataset] = {
            "selection_status": failure["status"],
            "rules_eligible": rules_eligible,
            "eligibility_certificate_valid": certificate_valid,
            "eligibility_certificate_error": certificate_error,
            "production_rule_count": len(rulebook["production_rules"]),
            "research_fallback_applied": bool(
                dict(failure["selection_audit"]).get(
                    "research_fallback_applied", False
                )
            ),
            "failure_audit": _artifact(failure_path),
            "rulebook": _artifact(rulebook_path),
        }
    verification = _verification_evidence(config, evidence_tag=evidence_tag)
    required_verification = ("pytest", "ruff", "compileall")
    verification_passed = all(
        name in verification and dict(verification[name]).get("status") == "passed"
        for name in required_verification
    )
    dss_release_ready = all(
        bool(dict(result)["rules_eligible"])
        and bool(dict(result)["eligibility_certificate_valid"])
        and int(dict(result)["production_rule_count"]) > 0
        and not bool(dict(result)["research_fallback_applied"])
        for result in dss_results.values()
    )

    artifact_paths = {
        "project_config": config.paths.root / "configs" / "defaults.toml",
        "release_audit_implementation": Path(__file__),
        "compatibility_metrics": compatibility_path,
        "compatibility_validation_lock": validation_lock_path,
        "compatibility_predictions": Path(str(compatibility["predictions_path"])),
        "compatibility_model": Path(str(compatibility["model_path"])),
        "ludb_fiducial_metrics": ludb_path,
        "ludb_fiducial_details": Path(str(ludb["details_path"])),
        "transition_metrics": transition_path,
        "physician_scenarios": physician_path,
        "physician_scenario_registry": physician_registry_path,
        "physician_run_manifest": physician_manifest_path,
        "physician_rule_review": rule_review_path,
        "physician_abstention_concordance": abstention_path,
        "physician_assisted_review": assisted_review_path,
        "semantic_development_scenarios": semantic_path,
        "signature_artifact": config.paths.transition / "signature_artifact_v1.json",
        "ptbxl_manifest": config.paths.manifests / "ptbxl_split_index.parquet",
        "ludb_manifest": config.paths.manifests / "ludb_split_index_repeat_1.parquet",
        "b1_operator_metadata": config.paths.transition / "B1_operator_metadata.json",
        "b2_operator_metadata": config.paths.transition / "B2_operator_metadata.json",
    }
    artifacts = {name: _artifact(path) for name, path in artifact_paths.items()}

    raw_exact_physician = dict(
        physician_results.get(_EXACT_PHYSICIAN_SCENARIO_ID, {})
    ).get("kappa")
    exact_physician = _finite_float(raw_exact_physician)
    failed_other_scenarios = [
        scenario_id
        for scenario_id in other_required_physician_ids
        if not bool(physician_results[scenario_id]["release_gate_passed"])
    ]
    gates = {
        "best_compatibility_class_f1_above_0_80": {
            "value": best_f1,
            "threshold": 0.80,
            "passed": best_f1 > 0.80,
        },
        "classification_subset_exact_match_above_0_90": {
            "value": exact_accuracy,
            "confidence_interval": dict(compatibility["global_metric_intervals"])[
                "intervals"
            ]["compatibility_subset_exact_match"],
            "threshold": 0.90,
            "passed": exact_accuracy > 0.90,
            "authoritative_accuracy_definition": "compatibility_subset_exact_match",
        },
        "per_label_bitwise_accuracy_above_0_90_non_authoritative": {
            "value": bitwise_accuracy,
            "threshold": 0.90,
            "passed": bitwise_accuracy > 0.90,
            "cannot_substitute_for_exact_match": True,
        },
        "physician_exact_kappa_above_0_70": {
            "value": exact_physician,
            "threshold": 0.70,
            "comparison": ">",
            "passed": bool(physician_evidence["exact_passed"]),
            "judgments_immutable": True,
        },
        "all_other_estimable_primary_physician_kappas_above_0_60": {
            "minimum_value": physician_evidence["minimum_other_kappa"],
            "threshold": 0.60,
            "comparison": ">",
            "passed": bool(physician_evidence["other_passed"]),
            "required_scenarios": physician_evidence["other_required_scenario_ids"],
            "failed_scenarios": failed_other_scenarios,
            "not_estimable_scenarios": physician_evidence[
                "required_non_estimable_scenarios"
            ],
            "required_non_estimable_policy": (
                "fail_unless_preregistered_exempt_or_not_applicable"
            ),
            "preregistered_exemptions": physician_evidence[
                "preregistered_exemptions"
            ],
        },
        "physician_primary_scenario_registry_integrity": {
            "value": bool(physician_evidence["integrity_passed"]),
            "threshold": True,
            "passed": bool(physician_evidence["integrity_passed"]),
            "required_scenarios": physician_evidence["required_scenario_ids"],
            "required_case_count": _PRIMARY_PHYSICIAN_CASE_COUNT,
            "binding_checks": physician_evidence["binding_checks"],
            "failures": physician_evidence["integrity_failures"],
        },
        "physician_rule_soundness_mean_likert_above_4_0": {
            "value": rule_review_evidence["mean_likert"],
            "threshold": 4.0,
            "comparison": ">",
            "passed": bool(rule_review_evidence["passed"]),
            "complete": rule_review_evidence["complete"],
            "manifest_binding": rule_review_evidence["manifest_binding"],
        },
        "software_verification_complete": {
            "value": verification_passed,
            "threshold": True,
            "passed": verification_passed,
            "required_checks": list(required_verification),
        },
        "strict_dss_rulebooks_release_ready": {
            "value": dss_release_ready,
            "threshold": True,
            "passed": dss_release_ready,
            "requires": "eligible nonempty B1 and B2 rulebooks with no fallback",
        },
    }
    required_gate_names = (
        "best_compatibility_class_f1_above_0_80",
        "classification_subset_exact_match_above_0_90",
        "physician_primary_scenario_registry_integrity",
        "physician_rule_soundness_mean_likert_above_4_0",
        "physician_exact_kappa_above_0_70",
        "all_other_estimable_primary_physician_kappas_above_0_60",
        "software_verification_complete",
        "strict_dss_rulebooks_release_ready",
    )
    overall_passed = all(
        bool(gates[name]["passed"])
        for name in required_gate_names
    )

    return {
        "artifact_version": 1,
        "release_id": f"cardia_x_strict_{evidence_tag}",
        "ontology_version": config.ontology_version,
        "status": "eligible" if overall_passed else "not_eligible",
        "all_requested_gates_passed": overall_passed,
        "required_gate_names": list(required_gate_names),
        "gates": gates,
        "compatibility_model": {
            "evaluation_partition": compatibility["evaluation_partition"],
            "test_records": compatibility["test_records"],
            "patient_disjoint": compatibility["patient_disjoint"],
            "micro_f1": compatibility["micro_f1"],
            "best_class_f1": best_f1,
            "per_class_metrics": per_class,
            "validation_exact_match": validation_lock[
                "validation_compatibility_subset_exact_match"
            ],
            "test_exact_match": exact_accuracy,
            "test_bitwise_accuracy": bitwise_accuracy,
        },
        "physician_validation": {
            "audited_bundle": str(physician_path.resolve()),
            "claim_eligible": bool(
                physician_evidence["integrity_passed"]
                and physician_evidence["exact_passed"]
                and physician_evidence["other_passed"]
            ),
            "primary_results": physician_results,
            "registry_enforcement": {
                key: value
                for key, value in physician_evidence.items()
                if key != "primary_results"
            },
            "rule_review": rule_review_evidence,
            "abstention_validation": {
                "summary": abstention_evidence,
                "manifest_binding": _manifest_output_binding(
                    abstention_path,
                    physician_manifest_path,
                ),
                "assisted_review_manifest_binding": _manifest_output_binding(
                    assisted_review_path,
                    physician_manifest_path,
                ),
                "composite_includes_benchmark_disagreement": True,
                "independent_physician_confirmation": {
                    "low_confidence_le_2": dict(
                        abstention_evidence["components"]
                    )["low_confidence_le_2"],
                    "physician_marked_ambiguous": dict(
                        abstention_evidence["components"]
                    )["physician_marked_ambiguous"],
                    "physician_marked_noisy": dict(
                        abstention_evidence["components"]
                    )["physician_marked_noisy"],
                    "structural_abstention_appropriate": assisted_review_evidence[
                        "structural_abstention_appropriate"
                    ],
                },
            },
            "semantic_development_only": {
                "claim_eligible": False,
                "exact_kappa": semantic["b1_unique_exact"].get("kappa"),
                "reason": "post-audit semantic sensitivity analysis; not the locked claim bundle",
                "artifact": _artifact(semantic_path),
            },
            "required_external_state_change": (
                "new prospectively locked blinded physician read with controlled "
                "multiaxial labels and positive pacing cases"
            ),
        },
        "quality_coverage": {
            "b1": _quality_coverage(config, "b1"),
            "b2": _quality_coverage(config, "b2"),
        },
        "external_ludb_fiducial_validation": {
            "records_evaluated": ludb["records_evaluated"],
            "record_failures": ludb["record_failures"],
            "r_peak_detection": ludb["r_peak_detection"],
            "matched_consensus_beats": ludb["matched_consensus_beats"],
            "pipeline_accepted_matched_beats": ludb[
                "pipeline_accepted_matched_beats"
            ],
            "landmark_metrics": ludb["landmark_metrics"],
        },
        "transition_validation": transition,
        "strict_dss": dss_results,
        "verification": verification,
        "artifacts": artifacts,
        "integrity_statement": (
            "No physician judgment, benchmark label, or held-out test label was altered "
            "to improve a metric. Unmet gates remain unmet and both DSS rulebooks fail closed."
        ),
    }


def _render_markdown(audit: dict[str, object]) -> str:
    gates = dict(audit["gates"])
    compatibility = dict(audit["compatibility_model"])
    physician = dict(audit["physician_validation"])
    ludb = dict(audit["external_ludb_fiducial_validation"])
    transition = dict(audit["transition_validation"])
    dss = dict(audit["strict_dss"])
    lines = [
        f"# CARDIA-X strict release audit — {audit['release_id']}",
        "",
        f"Release status: **{str(audit['status']).upper()}**.",
        "",
        "This audit distinguishes achieved engineering endpoints from unmet empirical "
        "endpoints. It never substitutes bitwise accuracy for exact label-set accuracy "
        "and never changes a physician judgment or benchmark label.",
        "",
        "## Requested gates",
        "",
    ]
    for name, raw_gate in gates.items():
        gate = dict(raw_gate)
        value = gate.get("value", gate.get("minimum_value"))
        lines.append(
            f"- `{name}`: **{'PASS' if gate['passed'] else 'FAIL'}**; "
            f"value={value}, threshold={gate['threshold']}."
        )
    lines.extend(
        [
            "",
            "## Compatibility model",
            "",
            f"- Held-out test records: {compatibility['test_records']}; patient disjoint: "
            f"{compatibility['patient_disjoint']}.",
            f"- Validation exact match: {compatibility['validation_exact_match']:.6f}.",
            f"- Held-out exact match: {compatibility['test_exact_match']:.6f}.",
            f"- Held-out bitwise accuracy: {compatibility['test_bitwise_accuracy']:.6f} "
            "(secondary; not an exact-match substitute).",
            f"- Held-out micro-F1: {compatibility['micro_f1']:.6f}; best class F1: "
            f"{compatibility['best_class_f1']:.6f}.",
            "",
            "## Immutable physician validation",
            "",
        ]
    )
    for scenario_id, raw_result in dict(physician["primary_results"]).items():
        result = dict(raw_result)
        lines.append(
            f"- `{scenario_id}`: status={result['status']}, kappa={result['kappa']}, "
            f"95% CI={result['confidence_interval']}, cases={result['case_count']}."
        )
    lines.extend(
        [
            "",
            "The current locked responses cannot legitimately satisfy the requested κ "
            "thresholds. The required next evidence is a new prospectively locked, "
            "blinded physician read with controlled multiaxial labels and positive pacing cases.",
            "",
            "## External LUDB waveform validation",
            "",
            f"- Records evaluated: {ludb['records_evaluated']}; failures: "
            f"{len(ludb['record_failures'])}.",
            f"- R-peak F1: {dict(ludb['r_peak_detection'])['f1']:.6f}; 95% CI: "
            f"{dict(ludb['r_peak_detection'])['f1_ci_95']}.",
            f"- Matched consensus beats: {ludb['matched_consensus_beats']}; "
            f"pipeline-accepted beats: {ludb['pipeline_accepted_matched_beats']}.",
            "",
            "## Transition validation",
            "",
        ]
    )
    for dataset, raw_splits in dict(transition["datasets"]).items():
        for split, raw_metric in dict(raw_splits).items():
            metric = dict(raw_metric)
            lines.append(
                f"- {dataset.upper()} {split}: MAE={metric['b_fit_mae']:.6f}, "
                f"95% record-cluster CI={metric['b_fit_mae_ci_95']}, "
                f"record coverage={metric['record_coverage']:.6f}."
            )
    lines.extend(["", "## Strict DSS outcome", ""])
    for dataset, raw_result in dss.items():
        result = dict(raw_result)
        lines.append(
            f"- {dataset.upper()}: selection={result['selection_status']}; "
            f"rules eligible={result['rules_eligible']}; production rules="
            f"{result['production_rule_count']}; certificate valid="
            f"{result['eligibility_certificate_valid']}; fallback applied="
            f"{result['research_fallback_applied']}."
        )
    verification = dict(audit["verification"])
    lines.extend(["", "## Software verification", ""])
    if "pytest" in verification:
        pytest_evidence = dict(verification["pytest"])
        lines.append(
            f"- Pytest: {pytest_evidence['passed']} concrete test cases passed; "
            f"{pytest_evidence['additional_declared_checks']} additional plugin-counted "
            "subtests/checks; zero failures or errors."
        )
    else:
        lines.append("- Pytest: current tagged evidence is missing.")
    if "ruff" in verification:
        ruff_evidence = dict(verification["ruff"])
        lines.append(
            f"- Ruff: {ruff_evidence['issues']} issues; "
            f"status={ruff_evidence['status']}."
        )
    else:
        lines.append("- Ruff: current tagged evidence is missing.")
    if "compileall" in verification:
        compile_evidence = dict(verification["compileall"])
        lines.append(
            f"- Source compilation: status={compile_evidence['status']}."
        )
    else:
        lines.append("- Source compilation: current tagged evidence is missing.")
    lines.extend(
        [
            "",
            "## Integrity conclusion",
            "",
            str(audit["integrity_statement"]),
            "",
        ]
    )
    return "\n".join(lines)


def run(config: ProjectConfig, args: object) -> int:
    clinical_run_id = getattr(args, "clinical_run_id", _DEFAULT_CLINICAL_RUN_ID)
    evidence_tag = getattr(args, "evidence_tag", _DEFAULT_EVIDENCE_TAG)
    safe_tag = _safe_component(evidence_tag, field="evidence_tag")
    audit = build_audit(
        config,
        clinical_run_id=str(clinical_run_id),
        evidence_tag=safe_tag,
    )
    json_path = config.paths.reports / f"CARDIA-X_strict_release_audit_{safe_tag}.json"
    markdown_path = config.paths.reports / f"CARDIA-X_strict_release_audit_{safe_tag}.md"
    write_json(json_path, audit)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(audit), encoding="utf-8")
    print(f"Release audit written to {json_path} and {markdown_path}")
    return 0 if audit["all_requested_gates_passed"] else 5
