from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tm_ecg.io.common import sha256_file
from tm_ecg.stages.release_audit import (
    _physician_primary_validation,
    _quality_coverage,
    _rule_review_validation,
    _scenario_map,
    _verification_evidence,
)


_PRESENCE_SCENARIOS = (
    "b1_axis_rhythm_presence",
    "b1_axis_ectopy_presence",
    "b1_axis_conduction_presence",
    "b1_axis_repolarization_presence",
)


def _registered_scenario(
    scenario_id: str,
    *,
    estimability_policy: str = "report",
    minimum_kappa: float | None = None,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "description": scenario_id,
        "population": "b1_unique",
        "required_case_count": 100,
        "deduplication": "original_case_id",
        "physician_projection": (
            "exact"
            if scenario_id == "b1_unique_exact"
            else f"axis_binary:{scenario_id.removeprefix('b1_axis_').removesuffix('_presence')}"
        ),
        "benchmark_projection": (
            "exact"
            if scenario_id == "b1_unique_exact"
            else f"axis_binary:{scenario_id.removeprefix('b1_axis_').removesuffix('_presence')}"
        ),
        "uncertain_policy": "exclude_uncertain_findings",
        "multilabel_policy": "single",
        "requirement": "required",
        "estimability_policy": estimability_policy,
        "minimum_kappa": (
            minimum_kappa
            if minimum_kappa is not None
            else (0.70 if scenario_id == "b1_unique_exact" else 0.60)
        ),
        "bootstrap_mode": "cluster_original_case",
        "bootstrap_seed": 1701,
    }


def _result(
    *,
    kappa: float | None,
    status: str = "ok",
    case_count: int = 100,
) -> dict[str, object]:
    return {
        "status": status,
        "kappa": kappa,
        "confidence_interval": [0.5, 0.8] if kappa is not None else [None, None],
        "sample_size": case_count,
        "case_ids": [f"CX-B1-{index:04d}" for index in range(1, case_count + 1)],
    }


def _write_physician_scenarios(
    tmp_path: Path,
    scenarios: list[dict[str, object]],
    results: dict[str, dict[str, object]],
    *,
    declared_registry_hash: str | None = None,
) -> tuple[Path, Path]:
    registry_path = tmp_path / "scenario_registry_v2.yaml"
    registry_path.write_text(
        json.dumps({"version": 2, "scenarios": scenarios}),
        encoding="utf-8",
    )
    scenario_path = tmp_path / "scenario_results.json"
    scenario_path.write_text(
        json.dumps(
            {
                "registry_hash": declared_registry_hash or sha256_file(registry_path),
                "scenario_results": [
                    {"scenario": scenario, "result": results[scenario["scenario_id"]]}
                    for scenario in scenarios
                    if scenario["scenario_id"] in results
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "scenario_registry_hash": sha256_file(registry_path),
                "output_hashes": {
                    scenario_path.name: sha256_file(scenario_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return registry_path, scenario_path


def test_physician_gate_enforces_every_registered_presence_scenario(tmp_path) -> None:
    scenarios = [_registered_scenario("b1_unique_exact")]
    scenarios.extend(_registered_scenario(name) for name in _PRESENCE_SCENARIOS)
    results = {
        "b1_unique_exact": _result(kappa=0.71),
        **{name: _result(kappa=0.61) for name in _PRESENCE_SCENARIOS},
    }
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
    )

    audit = _physician_primary_validation(registry_path, scenario_path)

    assert set(_PRESENCE_SCENARIOS).issubset(audit["required_scenario_ids"])
    assert set(audit["primary_results"]) == {
        "b1_unique_exact",
        *_PRESENCE_SCENARIOS,
    }
    assert audit["integrity_passed"] is True
    assert audit["exact_passed"] is True
    assert audit["other_passed"] is True
    for scenario_id in _PRESENCE_SCENARIOS:
        assert audit["primary_results"][scenario_id]["release_threshold"] == 0.60


def test_physician_gate_fails_when_registered_presence_scenario_is_omitted(
    tmp_path,
) -> None:
    omitted = "b1_axis_repolarization_presence"
    scenarios = [_registered_scenario("b1_unique_exact")]
    scenarios.extend(_registered_scenario(name) for name in _PRESENCE_SCENARIOS)
    results = {
        "b1_unique_exact": _result(kappa=0.71),
        **{
            name: _result(kappa=0.61)
            for name in _PRESENCE_SCENARIOS
            if name != omitted
        },
    }
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
    )

    audit = _physician_primary_validation(registry_path, scenario_path)

    assert audit["missing_required_scenarios"] == [omitted]
    assert audit["integrity_passed"] is False
    assert audit["other_passed"] is False
    assert audit["primary_results"][omitted]["failure_reasons"] == ["missing_result"]


def test_physician_gate_fails_required_non_estimable_scenario(tmp_path) -> None:
    pacing = _registered_scenario("b1_axis_pacing_presence")
    scenarios = [_registered_scenario("b1_unique_exact"), pacing]
    results = {
        "b1_unique_exact": _result(kappa=0.71),
        "b1_axis_pacing_presence": _result(kappa=None, status="not_estimable"),
    }
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
    )

    audit = _physician_primary_validation(registry_path, scenario_path)

    assert audit["required_non_estimable_scenarios"] == [
        "b1_axis_pacing_presence"
    ]
    assert audit["other_passed"] is False
    assert "required_non_estimable" in audit["primary_results"][
        "b1_axis_pacing_presence"
    ]["failure_reasons"]


def test_physician_gate_honors_preregistered_non_estimable_exemption(
    tmp_path,
) -> None:
    pacing = _registered_scenario(
        "b1_axis_pacing_presence",
        estimability_policy="not_applicable",
    )
    scenarios = [_registered_scenario("b1_unique_exact"), pacing]
    results = {
        "b1_unique_exact": _result(kappa=0.71),
        "b1_axis_pacing_presence": _result(kappa=None, status="not_applicable"),
    }
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
    )

    audit = _physician_primary_validation(registry_path, scenario_path)

    assert audit["required_non_estimable_scenarios"] == []
    assert audit["primary_results"]["b1_axis_pacing_presence"][
        "preregistered_exemption"
    ] is True
    assert audit["other_passed"] is True


def test_physician_gate_uses_strict_greater_than_kappa_thresholds(tmp_path) -> None:
    scenarios = [
        _registered_scenario("b1_unique_exact"),
        _registered_scenario("b1_axis_rhythm_presence"),
    ]
    results = {
        "b1_unique_exact": _result(kappa=0.70),
        "b1_axis_rhythm_presence": _result(kappa=0.60),
    }
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
    )

    audit = _physician_primary_validation(registry_path, scenario_path)

    assert audit["integrity_passed"] is True
    assert audit["exact_passed"] is False
    assert audit["other_passed"] is False
    assert audit["below_threshold_scenarios"] == [
        "b1_axis_rhythm_presence",
        "b1_unique_exact",
    ]


def test_physician_gate_never_weakens_preregistered_threshold(tmp_path) -> None:
    scenarios = [
        _registered_scenario("b1_unique_exact"),
        _registered_scenario(
            "b1_axis_rhythm_presence",
            minimum_kappa=0.70,
        ),
    ]
    results = {
        "b1_unique_exact": _result(kappa=0.71),
        "b1_axis_rhythm_presence": _result(kappa=0.65),
    }
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
    )

    audit = _physician_primary_validation(registry_path, scenario_path)

    result = audit["primary_results"]["b1_axis_rhythm_presence"]
    assert result["requested_target_floor"] == 0.60
    assert result["preregistered_threshold"] == 0.70
    assert result["release_threshold"] == 0.70
    assert result["release_gate_passed"] is False


def test_physician_gate_requires_hash_binding_manifest(tmp_path) -> None:
    scenarios = [_registered_scenario("b1_unique_exact")]
    results = {"b1_unique_exact": _result(kappa=0.71)}
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
    )
    (tmp_path / "run_manifest.json").unlink()

    audit = _physician_primary_validation(registry_path, scenario_path)

    assert audit["binding_passed"] is False
    assert audit["integrity_passed"] is False
    assert "run_manifest_missing" in audit["integrity_failures"]


def test_rule_review_gate_requires_complete_hash_bound_scores(tmp_path) -> None:
    summary_path = tmp_path / "rule_review_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_rules": 10,
                "expected_b1_rules": 8,
                "expected_b2_rules": 2,
                "template_rows": 10,
                "supplied_rule_ids": 10,
                "complete_rule_definitions": 10,
                "fully_scored_rules": 10,
                "observed_likert_cells": 40,
                "mean_likert": 4.1,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output_hashes": {
                    summary_path.name: sha256_file(summary_path),
                }
            }
        ),
        encoding="utf-8",
    )

    evidence = _rule_review_validation(summary_path, manifest_path)

    assert evidence["complete"] is True
    assert evidence["manifest_binding"]["passed"] is True
    assert evidence["passed"] is True

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["mean_likert"] = 4.0
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    evidence = _rule_review_validation(summary_path, manifest_path)
    assert evidence["passed"] is False


def test_physician_gate_fails_registry_hash_and_primary_case_count(
    tmp_path,
) -> None:
    scenarios = [
        _registered_scenario("b1_unique_exact"),
        _registered_scenario("b1_axis_rhythm_presence"),
    ]
    results = {
        "b1_unique_exact": _result(kappa=0.71),
        "b1_axis_rhythm_presence": _result(kappa=0.61, case_count=99),
    }
    registry_path, scenario_path = _write_physician_scenarios(
        tmp_path,
        scenarios,
        results,
        declared_registry_hash="stale-registry-hash",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "scenario_registry_hash": sha256_file(registry_path),
                "output_hashes": {"scenario_results.json": "stale-results-hash"},
            }
        ),
        encoding="utf-8",
    )

    audit = _physician_primary_validation(registry_path, scenario_path)

    assert audit["binding_passed"] is False
    assert audit["integrity_passed"] is False
    assert audit["other_passed"] is False
    assert "scenario_results_registry_hash_mismatch" in audit["integrity_failures"]
    assert "run_manifest_scenario_results_hash_mismatch" in audit[
        "integrity_failures"
    ]
    assert "case_count_not_100" in audit["primary_results"][
        "b1_axis_rhythm_presence"
    ]["failure_reasons"]


def test_release_audit_reads_registered_scenario_ids(tmp_path) -> None:
    path = tmp_path / "scenario_results.json"
    path.write_text(
        json.dumps(
            {
                "scenario_results": [
                    {
                        "scenario": {"scenario_id": "exact"},
                        "result": {"status": "ok", "kappa": 0.5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _scenario_map(path) == {
        "exact": {"status": "ok", "kappa": 0.5}
    }


def test_release_audit_quality_coverage_requires_all_three_guards(tmp_path) -> None:
    contents = (
        "record_id,lead_quality_min_db,delineation_confidence,analyzable_duration_s\n"
        "r1,6,0.8,8\n"
        "r2,6,0.4,8\n"
    )
    for split in ("train", "val", "test"):
        (tmp_path / f"B1_raw_{split}.csv").write_text(contents, encoding="utf-8")
    config = SimpleNamespace(
        thresholds={
            "feature_quality_min_db": 5.0,
            "delineation_confidence_minimum": 0.5,
            "minimum_af_analyzable_duration_s": 7.5,
        },
        dss={"minimum_analyzable_coverage": 0.8},
        paths=SimpleNamespace(features=tmp_path),
    )

    audit = _quality_coverage(config, "b1")

    assert audit["partitions"]["val"]["quality_eligible_coverage"] == pytest.approx(
        0.5
    )
    assert audit["does_not_imply_model_abstention"] is True


def test_release_audit_counts_concrete_junit_testcases(tmp_path) -> None:
    directory = tmp_path / "verification"
    directory.mkdir()
    (directory / "pytest_full_20260721.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite tests="99" failures="0" errors="0">'
        '<testcase classname="tests.test_example" name="test_one" />'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    config = SimpleNamespace(paths=SimpleNamespace(reports=tmp_path))

    pytest_audit = _verification_evidence(config)["pytest"]

    assert pytest_audit["tests"] == 1
    assert pytest_audit["passed"] == 1
    assert pytest_audit["declared_tests"] == 99
    assert pytest_audit["declared_count_matches"] is False
    assert pytest_audit["additional_declared_checks"] == 98
    assert pytest_audit["status"] == "passed"
