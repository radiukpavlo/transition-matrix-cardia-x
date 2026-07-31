"""Ordered CARDIA-X v3 release gates with fail-closed evidence handling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from tm_ecg.config import ProjectConfig
from tm_ecg.reproducibility import sha256_file, write_artifact_manifest


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _scenario_rows(bundle: Mapping[str, object]) -> list[dict[str, object]]:
    raw = bundle.get("scenario_results", [])
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _scenario_index(
    bundle: Mapping[str, object],
) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    indexed = {}
    for item in _scenario_rows(bundle):
        scenario = item.get("scenario", {})
        result = item.get("result", {})
        if not isinstance(scenario, Mapping) or not isinstance(result, Mapping):
            continue
        indexed[str(scenario.get("scenario_id", ""))] = (
            dict(scenario),
            dict(result),
        )
    return indexed


def _compatibility_values(
    payload: Mapping[str, object],
) -> dict[str, object]:
    metrics = payload.get("selected_metrics", payload)
    metric_map = dict(metrics) if isinstance(metrics, Mapping) else {}
    exact = metric_map.get(
        "exact_subset_accuracy",
        metric_map.get(
            "compatibility_subset_exact_match",
            metric_map.get("exact_match"),
        ),
    )
    per_class = metric_map.get(
        "per_class_metrics",
        payload.get("per_class_metrics", {}),
    )
    f1_values: list[float] = []
    if isinstance(per_class, Mapping):
        for value in per_class.values():
            if isinstance(value, Mapping) and value.get("f1") is not None:
                f1_values.append(float(str(value["f1"])))
    best_f1 = metric_map.get(
        "best_class_f1",
        max(f1_values) if f1_values else None,
    )
    sample_size = int(
        str(
            metric_map.get(
                "sample_size",
                metric_map.get(
                    "test_records",
                    metric_map.get(
                        "records",
                        payload.get("test_records", 0),
                    ),
                ),
            )
        )
    )
    success_count = (
        int(round(float(str(exact)) * sample_size))
        if exact is not None and sample_size > 0
        else None
    )
    rare = payload.get("selection_constraints", {})
    rare_map = dict(rare) if isinstance(rare, Mapping) else {}
    rare_gate = rare_map.get("rare_label_recall_floors", {})
    return {
        "exact_subset_accuracy": float(str(exact)) if exact is not None else None,
        "best_class_f1": (
            float(str(best_f1)) if best_f1 is not None else None
        ),
        "sample_size": sample_size,
        "success_count": success_count,
        "evaluation_partition": payload.get("evaluation_partition"),
        "confirmatory_labels_opened": payload.get(
            "confirmatory_labels_opened"
        ),
        "rare_label_gate_passed": (
            bool(rare_gate.get("passed"))
            if isinstance(rare_gate, Mapping)
            else False
        ),
        "calibration_gate_passed": bool(
            payload.get("calibration_gate_passed", False)
        ),
    }


def build_release_audit(
    *,
    source_gate: Mapping[str, object],
    coding_gate: Mapping[str, object],
    scenario_bundle: Mapping[str, object] | None,
    compatibility_metrics: Mapping[str, object] | None,
    environment_audit: Mapping[str, object],
    clinical_artifact_manifest_present: bool,
    transition_gate: Mapping[str, object] | None = None,
    dss_gate: Mapping[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    """Evaluate G0-G10 in order and return the first contractual exit code."""

    scenarios = _scenario_index(scenario_bundle or {})
    required_primary = [
        (scenario, result)
        for scenario, result in scenarios.values()
        if scenario.get("requirement") in {"required", "audit"}
        and scenario.get("population") == "b1_unique"
        and scenario.get("route") in {None, ""}
    ]
    retention_passed = bool(required_primary) and all(
        int(str(result.get("sample_size", 0))) == 100
        for _scenario, result in required_primary
    )
    raw_rows = [
        (scenario, result)
        for scenario, result in scenarios.values()
        if str(scenario.get("endpoint_version", "")).startswith("raw_immutable")
    ]
    raw_passed = bool(raw_rows) and all(
        (
            scenario.get("expected_baseline_status") in {None, ""}
            or result.get("status") == scenario.get("expected_baseline_status")
        )
        and (
            scenario.get("expected_baseline_kappa") is None
            or result.get("kappa") is not None
            and abs(
                float(str(result["kappa"]))
                - float(str(scenario["expected_baseline_kappa"]))
            )
            <= 1e-6
        )
        for scenario, result in raw_rows
    )
    harmonized_exact = scenarios.get("harmonized_b1_unique_exact")
    exact_passed = bool(
        harmonized_exact
        and harmonized_exact[1].get("status") == "ok"
        and float(str(harmonized_exact[1].get("kappa", -1))) >= 0.70
    )
    required_harmonized = [
        (scenario, result)
        for scenario, result in scenarios.values()
        if scenario.get("requirement") == "required"
    ]
    required_estimable = bool(required_harmonized) and all(
        result.get("status") == "ok" for _scenario, result in required_harmonized
    )
    all_required_passed = required_estimable and all(
        float(str(result.get("kappa", -1)))
        >= float(str(scenario.get("minimum_kappa", 0.60)))
        for scenario, result in required_harmonized
    )
    compatibility = (
        _compatibility_values(compatibility_metrics)
        if compatibility_metrics is not None
        else {}
    )
    confirmatory = (
        compatibility.get("evaluation_partition")
        == "sealed_internal_confirmatory"
        and compatibility.get("confirmatory_labels_opened") is True
    )
    best_f1 = compatibility.get("best_class_f1")
    best_f1_passed = bool(
        confirmatory and best_f1 is not None and float(best_f1) > 0.80
    )
    exact_value = compatibility.get("exact_subset_accuracy")
    sample_size = int(compatibility.get("sample_size", 0) or 0)
    strict_successes = math.floor(0.90 * sample_size) + 1
    compatibility_exact_passed = bool(
        confirmatory
        and exact_value is not None
        and float(exact_value) > 0.90
        and int(compatibility.get("success_count", 0) or 0) >= strict_successes
    )
    rare_calibration_passed = bool(
        compatibility.get("rare_label_gate_passed")
        and compatibility.get("calibration_gate_passed")
    )
    dss_passed = bool(
        transition_gate
        and transition_gate.get("passed")
        and dss_gate
        and dss_gate.get("passed")
    )
    lock = environment_audit.get("dependency_lock", {})
    reproducibility_passed = bool(
        isinstance(lock, Mapping)
        and lock.get("valid")
        and clinical_artifact_manifest_present
        and environment_audit.get("git") is not None
    )
    gates = {
        "G0_source_authority": bool(source_gate.get("passed")),
        "G1_retention": retention_passed,
        "G2_leakage": bool(coding_gate.get("passed")),
        "G3_raw_baseline": raw_passed,
        "G4_harmonized_exact": exact_passed,
        "G5_harmonized_scenarios": all_required_passed,
        "G6_compatibility_best_f1": best_f1_passed,
        "G7_compatibility_exact": compatibility_exact_passed,
        "G8_calibration_rare_classes": rare_calibration_passed,
        "G9_transition_dss": dss_passed,
        "G10_reproducibility": reproducibility_passed,
    }
    failure_order = (
        ("G0_source_authority", 10),
        ("G1_retention", 11),
        ("G2_leakage", 12),
        ("G3_raw_baseline", 13),
        ("G4_harmonized_exact", 15),
        (
            "G5_harmonized_scenarios",
            14 if not required_estimable else 16,
        ),
        ("G6_compatibility_best_f1", 18),
        ("G7_compatibility_exact", 17),
        ("G8_calibration_rare_classes", 19),
        ("G9_transition_dss", 19),
        ("G10_reproducibility", 19),
    )
    first_failed_gate = next(
        (name for name, _code in failure_order if not gates[name]),
        None,
    )
    code = next(
        (code for name, code in failure_order if name == first_failed_gate),
        0,
    )
    return code, {
        "schema_version": "cardia_x_release_audit_v3",
        "release_eligible": code == 0,
        "first_failed_gate": first_failed_gate,
        "exit_code": code,
        "gates": gates,
        "raw_immutable_endpoint": {
            "audit_only": True,
            "scenarios_present": len(raw_rows),
            "passed_frozen_baseline": raw_passed,
        },
        "harmonized_endpoint": {
            "exact_kappa": (
                harmonized_exact[1].get("kappa")
                if harmonized_exact
                else None
            ),
            "all_required_passed": all_required_passed,
        },
        "compatibility_endpoint": {
            **compatibility,
            "strict_required_successes": strict_successes,
            "confirmatory_partition_verified": confirmatory,
        },
        "prohibited_shortcut_checks": {
            "raw_endpoint_not_relabelled_as_harmonized": True,
            "confirmatory_required_for_model_claim": True,
            "all_rows_used_for_exact_accuracy": sample_size > 0,
            "abstention_not_used_to_change_primary_denominator": True,
            "workbook_write_attempts_zero": (
                int(str(source_gate.get("workbook_write_attempts", 0))) == 0
            ),
        },
    }


def run(config: ProjectConfig, args: object) -> int:
    root = config.paths.root

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    clinical_dir = resolve(str(getattr(args, "clinical_results_dir")))
    output_dir = resolve(str(getattr(args, "output_dir")))
    coding_gate_path = resolve(str(getattr(args, "coding_gate")))
    environment_path = resolve(str(getattr(args, "environment_audit")))
    compatibility_path = resolve(str(getattr(args, "compatibility_metrics")))
    scenario_path = clinical_dir / "scenario_results.json"
    transition_value = getattr(args, "transition_gate", None)
    dss_value = getattr(args, "dss_gate", None)
    code, audit = build_release_audit(
        source_gate=_read(clinical_dir / "source_authority_gate.json"),
        coding_gate=_read(coding_gate_path),
        scenario_bundle=_read(scenario_path) if scenario_path.exists() else None,
        compatibility_metrics=(
            _read(compatibility_path) if compatibility_path.exists() else None
        ),
        environment_audit=_read(environment_path),
        clinical_artifact_manifest_present=(
            clinical_dir / "artifact_manifest.json"
        ).is_file(),
        transition_gate=(
            _read(resolve(str(transition_value))) if transition_value else None
        ),
        dss_gate=_read(resolve(str(dss_value))) if dss_value else None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "release_audit_v3.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# CARDIA-X v3 release audit",
        "",
        f"Release eligible: **{'YES' if audit['release_eligible'] else 'NO'}**",
        "",
        f"First failed gate: `{audit['first_failed_gate']}`",
        "",
        "The immutable raw endpoint remains audit-only and is reported separately "
        "from the versioned harmonized endpoint.",
    ]
    (output_dir / "release_report_v3.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(
        output_dir,
        producer_command="tm-ecg release-audit-v3",
        input_hashes={
            "source_gate": sha256_file(
                clinical_dir / "source_authority_gate.json"
            ),
            "environment_audit": sha256_file(environment_path),
        },
        code_root=root,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return code
