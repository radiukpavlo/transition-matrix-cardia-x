"""Command-line orchestration for independent clinical-validation passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from tm_ecg.clinical_validation.audit import (
    environment_manifest,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_payload,
    write_csv,
    write_json,
    write_jsonl,
)
from tm_ecg.clinical_validation.benchmark_coder import (
    code_benchmarks,
    load_benchmark_mapping,
)
from tm_ecg.clinical_validation.coding_lint import lint_physician_coding
from tm_ecg.clinical_validation.baseline import (
    compute_baseline_reference,
    write_baseline_artifacts,
)
from tm_ecg.clinical_validation.field_policy import (
    PROHIBITED_FIELDS,
    FieldPolicyViolation,
    PhysicianView,
)
from tm_ecg.clinical_validation.metrics import compute_cohen_kappa
from tm_ecg.clinical_validation.models import (
    BenchmarkFindingSet,
    CaseIdentity,
    ClinicalFindingSet,
    RawPhysicianResponse,
    ValidationRunManifest,
)
from tm_ecg.clinical_validation.ontology import (
    load_json_yaml,
    project_exact,
)
from tm_ecg.clinical_validation.physician_coder import (
    code_physician_response,
    load_physician_rules,
)
from tm_ecg.clinical_validation.report import render_validation_report
from tm_ecg.clinical_validation.scenarios import (
    evaluate_scenarios,
    gate_results,
    load_scenario_registry,
)
from tm_ecg.clinical_validation.scenario_lint import lint_scenario_registry
from tm_ecg.clinical_validation.source_authority import (
    SourceAuthorityError,
    allows_unsigned_source_exploration,
    load_response_authority,
    resolve_unassisted_source,
    write_authority_audit,
)
from tm_ecg.clinical_validation.validation_tracks import (
    compute_abstention_concordance,
    evaluate_cardia_x_track,
    summarize_assisted_review,
)
from tm_ecg.reproducibility import write_artifact_manifest
from tm_ecg.clinical_validation.workbook_reader import (
    WorkbookSchemaError,
    load_provenance,
    read_completed_workbook,
)
from tm_ecg.config import ProjectConfig


PACKAGE_VERSION = "2.0.0"


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _root(config: ProjectConfig) -> Path:
    return config.paths.root


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _load_policy(root: Path, path: str | Path) -> tuple[Path, dict[str, object]]:
    policy_path = _path(root, path)
    policy = load_json_yaml(policy_path)
    if policy.get("version") not in {2, 3}:
        raise ValueError("Acceptance policy must declare version 2 or 3")
    return policy_path, policy


def _default_path(root: Path, policy: Mapping[str, object], key: str) -> Path:
    return _path(root, str(policy[key]))


def extract_pass(
    *,
    workbook: Path,
    provenance: Path,
    output_dir: Path,
    response_authority: Path | None = None,
    allow_unsigned_source_exploration: bool = False,
) -> dict[str, Path]:
    extraction = read_completed_workbook(
        workbook,
        provenance,
        reconciliation_mode=(
            "authority_manifest" if response_authority is not None else "strict_reconciliation"
        ),
        response_authority_path=response_authority,
        authority_output_dir=output_dir if response_authority is not None else None,
        allow_unsigned_source_exploration=allow_unsigned_source_exploration,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "responses": write_jsonl(
            output_dir / "unassisted_responses.jsonl",
            [item.to_dict() for item in extraction.physician_responses],
        ),
        "identities": write_jsonl(
            output_dir / "case_identities.jsonl",
            [item.to_dict() for item in extraction.identities],
        ),
        "b2": write_jsonl(output_dir / "b2_responses.jsonl", extraction.b2_responses),
        "cardia_x": write_jsonl(output_dir / "cardia_x_outputs.jsonl", extraction.cardia_x_outputs),
        "assisted_review": write_jsonl(
            output_dir / "assisted_review_responses.jsonl",
            extraction.assisted_review_responses,
        ),
        "audit": write_json(
            output_dir / "response_import_audit.json", extraction.response_import_audit
        ),
        "rule_review": write_json(
            output_dir / "rule_review_summary.json", extraction.rule_review_summary
        ),
    }
    return paths


def audit_workbook_pass(
    *,
    workbook: Path,
    provenance: Path,
    response_authority: Path,
    output_dir: Path,
) -> tuple[int, dict[str, Path]]:
    manifest = load_response_authority(response_authority)
    provenance_rows = load_provenance(provenance)
    original_case_ids = {
        case_id: (str(row.get("duplicate_of_case_id", "")).strip() or case_id)
        for case_id, row in provenance_rows.items()
    }
    resolution = resolve_unassisted_source(
        workbook,
        manifest,
        original_case_ids=original_case_ids,
    )
    paths = write_authority_audit(output_dir, resolution)
    return (0 if resolution.gate["passed"] else 10), paths


def physician_pass(*, responses: Path, rules_path: Path, output_dir: Path) -> dict[str, Path]:
    rules = load_physician_rules(rules_path)
    raw_rows = read_jsonl(responses)
    findings: list[ClinicalFindingSet] = []
    allowed_serialized = set(RawPhysicianResponse.__dataclass_fields__)
    for row in raw_rows:
        prohibited = sorted(set(row) & PROHIBITED_FIELDS)
        unknown = sorted(set(row) - allowed_serialized)
        if prohibited or unknown:
            raise FieldPolicyViolation(
                f"Physician input crossed field policy; prohibited={prohibited}, unknown={unknown}"
            )
        view = PhysicianView(RawPhysicianResponse.from_dict(row))
        findings.append(code_physician_response(view, rules))
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path = write_jsonl(
        output_dir / "physician_findings.jsonl", [item.to_dict() for item in findings]
    )
    manifest = {
        "input": str(responses),
        "input_hash": sha256_file(responses),
        "rules": str(rules_path),
        "rules_hash": sha256_file(rules_path),
        "case_count": len(findings),
        "contains_benchmark_fields": False,
        "contains_assisted_fields": False,
        "output": str(findings_path),
        "output_hash": sha256_file(findings_path),
    }
    return {
        "findings": findings_path,
        "manifest": write_json(output_dir / "physician_coding_manifest.json", manifest),
    }


def benchmark_pass(
    *,
    identities_path: Path,
    provenance_path: Path,
    mapping_path: Path,
    output_dir: Path,
    ptbxl_index: Path | None,
    ludb_index: Path | None,
) -> dict[str, Path]:
    identities = [CaseIdentity.from_dict(row) for row in read_jsonl(identities_path)]
    provenance = load_provenance(provenance_path)
    mapping = load_benchmark_mapping(mapping_path)
    findings = code_benchmarks(
        identities,
        provenance,
        mapping,
        ptbxl_index=ptbxl_index,
        ludb_index=ludb_index,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path = write_jsonl(
        output_dir / "benchmark_findings.jsonl", [item.to_dict() for item in findings]
    )
    manifest = {
        "provenance": str(provenance_path),
        "provenance_hash": sha256_file(provenance_path),
        "mapping": str(mapping_path),
        "mapping_hash": sha256_file(mapping_path),
        "ptbxl_index": str(ptbxl_index) if ptbxl_index else None,
        "ludb_index": str(ludb_index) if ludb_index else None,
        "case_count": len(findings),
        "contains_physician_text": False,
        "output": str(findings_path),
        "output_hash": sha256_file(findings_path),
    }
    return {
        "findings": findings_path,
        "manifest": write_json(output_dir / "benchmark_coding_manifest.json", manifest),
    }


def _repeatability(
    identities: list[CaseIdentity],
    physician: list[ClinicalFindingSet],
) -> dict[str, object]:
    findings = {item.case_id: item for item in physician}
    original_labels: list[str] = []
    repeat_labels: list[str] = []
    case_ids: list[str] = []
    for identity in identities:
        if not identity.is_repeat:
            continue
        original = findings.get(identity.original_case_id)
        repeated = findings.get(identity.workbook_case_id)
        if original is None or repeated is None:
            raise ValueError(f"Missing repeat pair for {identity.workbook_case_id}")
        original_labels.append(project_exact(original))
        repeat_labels.append(project_exact(repeated))
        case_ids.append(identity.workbook_case_id)
    result = compute_cohen_kappa(original_labels, repeat_labels, case_ids=case_ids)
    return result.to_dict()


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float | None]:
    if n <= 0:
        return [None, None]
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    half = z * ((proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) ** 0.5) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _b2_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"n": len(rows)}
    for field in (
        "p_wave_delineation_plausible",
        "qrs_delineation_plausible",
        "t_wave_delineation_plausible",
    ):
        values = [str(row.get(field, "")).strip().lower() for row in rows]
        usable = [value for value in values if value in {"yes", "no"}]
        count = sum(value == "yes" for value in usable)
        result[field] = {
            "n": len(usable),
            "yes": count,
            "proportion": count / len(usable) if usable else None,
            "wilson_ci_95": _wilson_interval(count, len(usable)),
        }
    for field in ("lead_quality_reasonable", "lead_quality_utility"):
        numeric_values: list[float] = []
        for row in rows:
            try:
                numeric_values.append(float(str(row.get(field))))
            except (TypeError, ValueError):
                continue
        result[field] = {
            "n": len(numeric_values),
            "mean": mean(numeric_values) if numeric_values else None,
            "median": median(numeric_values) if numeric_values else None,
            "minimum": min(numeric_values) if numeric_values else None,
            "maximum": max(numeric_values) if numeric_values else None,
        }
    issue_counts: dict[str, int] = {}
    for row in rows:
        issue = str(row.get("morphology_issue", "")).strip() or "blank"
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    result["morphology_issue_counts"] = dict(sorted(issue_counts.items()))
    return result


def evaluate_pass(
    *,
    identities_path: Path,
    physician_path: Path,
    benchmark_path: Path,
    b2_path: Path,
    raw_responses_path: Path | None,
    cardia_x_path: Path | None,
    assisted_review_path: Path | None,
    registry_path: Path,
    ontology_path: Path,
    output_dir: Path,
    bootstrap_replicates_override: int | None = None,
    raw_physician_path: Path | None = None,
    raw_benchmark_path: Path | None = None,
) -> dict[str, Path]:
    identities = [CaseIdentity.from_dict(row) for row in read_jsonl(identities_path)]
    physician = [ClinicalFindingSet.from_dict(row) for row in read_jsonl(physician_path)]
    benchmark = [BenchmarkFindingSet.from_dict(row) for row in read_jsonl(benchmark_path)]
    registry, scenarios = load_scenario_registry(registry_path)
    results, ledger, distributions = evaluate_scenarios(
        identities,
        physician,
        benchmark,
        registry,
        scenarios,
        bootstrap_replicates_override=bootstrap_replicates_override,
        raw_physician_findings=(
            [ClinicalFindingSet.from_dict(row) for row in read_jsonl(raw_physician_path)]
            if raw_physician_path is not None
            else None
        ),
        raw_benchmark_findings=(
            [BenchmarkFindingSet.from_dict(row) for row in read_jsonl(raw_benchmark_path)]
            if raw_benchmark_path is not None
            else None
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "registry_hash": sha256_file(registry_path),
        "ontology_hash": sha256_file(ontology_path),
        "physician_findings_hash": sha256_file(physician_path),
        "benchmark_findings_hash": sha256_file(benchmark_path),
        "scenario_results": results,
    }
    results_path = write_json(output_dir / "scenario_results.json", bundle)
    summary_rows: list[Mapping[str, object]] = []
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir = output_dir / "bootstrap_distributions"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    for item in results:
        raw_scenario = item["scenario"]
        raw_result = item["result"]
        scenario = dict(raw_scenario) if isinstance(raw_scenario, Mapping) else {}
        result = dict(raw_result) if isinstance(raw_result, Mapping) else {}
        ci = result.get("confidence_interval", [None, None])
        summary_rows.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "requirement": scenario.get("requirement"),
                "sample_size": result.get("sample_size"),
                "status": result.get("status"),
                "observed_agreement": result.get("observed_agreement"),
                "expected_agreement": result.get("expected_agreement"),
                "kappa": result.get("kappa"),
                "ci_lower": ci[0] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
                "ci_upper": ci[1] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
                "minimum_kappa": scenario.get("minimum_kappa"),
                "passes": result.get("passes_point_threshold"),
                "maximum_attainable_kappa": result.get("maximum_attainable_kappa"),
                "fixed_margin_threshold_attainable": result.get(
                    "fixed_margin_threshold_attainable"
                ),
                "approximate_additional_agreements": result.get(
                    "approximate_additional_agreements_for_threshold"
                ),
            }
        )
        matrix = result.get("confusion_matrix", {})
        matrix_rows: list[Mapping[str, object]] = []
        if isinstance(matrix, dict):
            for reference_label, columns in matrix.items():
                if not isinstance(columns, Mapping):
                    continue
                for comparison_label, count in columns.items():
                    matrix_rows.append(
                        {
                            "reference_label": reference_label,
                            "comparison_label": comparison_label,
                            "count": count,
                        }
                    )
        if matrix_rows:
            write_csv(confusion_dir / f"{scenario['scenario_id']}.csv", matrix_rows)
        bootstrap_values = distributions.get(str(scenario["scenario_id"]), [])
        if bootstrap_values:
            write_csv(
                bootstrap_dir / f"{scenario['scenario_id']}.csv",
                [
                    {"replicate": index, "kappa": value}
                    for index, value in enumerate(bootstrap_values, start=1)
                ],
            )
    summary_path = write_csv(output_dir / "scenario_summary.csv", summary_rows)
    ledger_path = write_csv(output_dir / "disagreement_ledger.csv", ledger)
    repeatability = _repeatability(identities, physician)
    repeatability_path = write_json(output_dir / "reader_repeatability.json", repeatability)
    b2_summary = _b2_summary(read_jsonl(b2_path))
    b2_summary_path = write_json(output_dir / "b2_plausibility_summary.json", b2_summary)
    outputs = {
        "results": results_path,
        "summary": summary_path,
        "ledger": ledger_path,
        "repeatability": repeatability_path,
        "b2_summary": b2_summary_path,
    }
    if raw_responses_path is not None and cardia_x_path is not None:
        raw_responses = [
            RawPhysicianResponse.from_dict(row) for row in read_jsonl(raw_responses_path)
        ]
        abstention_summary, abstention_ledger = compute_abstention_concordance(
            identities, raw_responses, physician, benchmark
        )
        outputs["abstention_summary"] = write_json(
            output_dir / "abstention_concordance.json", abstention_summary
        )
        outputs["abstention_ledger"] = write_csv(
            output_dir / "abstention_concordance_ledger.csv", abstention_ledger
        )
        cardia_x_summary, model_distributions = evaluate_cardia_x_track(
            identities,
            read_jsonl(cardia_x_path),
            benchmark,
            _string_tuple(registry.get("dominant_priority", [])),
            bootstrap_replicates=(
                int(bootstrap_replicates_override)
                if bootstrap_replicates_override is not None
                else int(str(registry.get("bootstrap_replicates", 1000)))
            ),
        )
        outputs["cardia_x_track"] = write_json(
            output_dir / "cardia_x_vs_benchmark.json", cardia_x_summary
        )
        outputs["cardia_x_gate"] = write_json(
            output_dir / "cardia_x_gate_result.json",
            cardia_x_summary["acceptance"],
        )
        model_bootstrap_dir = output_dir / "cardia_x_bootstrap_distributions"
        model_bootstrap_dir.mkdir(parents=True, exist_ok=True)
        for name, values in sorted(model_distributions.items()):
            if values:
                write_csv(
                    model_bootstrap_dir / f"{name}.csv",
                    [
                        {"replicate": index, "kappa": value}
                        for index, value in enumerate(values, start=1)
                    ],
                )
    if assisted_review_path is not None:
        outputs["assisted_review"] = write_json(
            output_dir / "assisted_review_summary.json",
            summarize_assisted_review(identities, read_jsonl(assisted_review_path)),
        )
    return outputs


def gate_pass(
    *,
    scenario_results_path: Path,
    policy_path: Path,
    registry_path: Path,
    ontology_path: Path,
    output_dir: Path,
) -> tuple[int, dict[str, object]]:
    bundle = read_json(scenario_results_path)
    if bundle.get("registry_hash") != sha256_file(registry_path):
        return 7, {"passed": False, "reason": "scenario_registry_hash_mismatch"}
    if bundle.get("ontology_hash") != sha256_file(ontology_path):
        return 7, {"passed": False, "reason": "ontology_hash_mismatch"}
    policy = load_json_yaml(policy_path)
    code, gate = gate_results(bundle["scenario_results"], policy)
    write_json(output_dir / "gate_result.json", gate)
    return code, gate


def _classify_gate_failures(
    scenario_results: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> dict[str, object]:
    raw_below_groups = [
        gate.get("below_threshold", []),
        gate.get("harmonized_exact_below_threshold", []),
        gate.get("other_required_below_threshold", []),
    ]
    raw_non_estimable = gate.get("required_not_estimable", [])
    below = {str(item) for group in raw_below_groups if isinstance(group, list) for item in group}
    non_estimable = (
        {str(item) for item in raw_non_estimable} if isinstance(raw_non_estimable, list) else set()
    )
    classifications: list[dict[str, object]] = []
    for item in scenario_results:
        raw_scenario = item.get("scenario", {})
        raw_result = item.get("result", {})
        if not isinstance(raw_scenario, Mapping) or not isinstance(raw_result, Mapping):
            continue
        scenario_id = str(raw_scenario.get("scenario_id", ""))
        if scenario_id not in below | non_estimable:
            continue
        projection = str(raw_scenario.get("physician_projection", ""))
        status = str(raw_result.get("status", ""))
        sample_size = int(str(raw_result.get("sample_size", 0)))
        observed = raw_result.get("observed_agreement")
        disagreement_count = (
            round(sample_size * (1.0 - float(str(observed)))) if observed is not None else None
        )
        if status == "not_estimable":
            classification = "non_estimability_from_degenerate_observed_margins"
        elif projection.startswith("axis"):
            classification = "residual_axis_level_physician_benchmark_disagreement"
        else:
            classification = "residual_fixed_reader_benchmark_disagreement_after_coding_audit"
        classifications.append(
            {
                "scenario_id": scenario_id,
                "projection": projection,
                "status": status,
                "classification": classification,
                "disagreement_count": disagreement_count,
                "kappa": raw_result.get("kappa"),
                "minimum_kappa": raw_scenario.get("minimum_kappa"),
                "resolution": (
                    "report_as_unmet; do_not_edit_fixed_responses_or_add_benchmark-aware_rules"
                ),
            }
        )
    return {
        "classification_policy": (
            "Only reproducible, diagnosis-independent semantic defects may be corrected. "
            "Remaining disagreements are empirical outcomes, not labels to hand-edit."
        ),
        "generalizable_defects_corrected_before_final_run": [
            "wandering/migrating atrial pacemaker no longer implies electronic pacing",
            "source-detail recovery is limited to curated Other/unmapped benchmark cases",
            "specific benchmark mappings supersede the BENCH-OTHER catch-all",
            "benchmark pacing defaults to absent rather than indeterminate",
        ],
        "failed_required_scenarios": classifications,
        "failed_required_scenario_count": len(classifications),
        "case_level_evidence": "disagreement_ledger.csv",
    }


def _output_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            hashes[str(path.relative_to(output_dir)).replace("\\", "/")] = sha256_file(path)
    return hashes


def run_all(config: ProjectConfig, args: argparse.Namespace) -> int:
    root = _root(config)
    policy_path, policy = _load_policy(root, args.policy)
    workbook = (
        _path(root, args.workbook)
        if args.workbook
        else _default_path(root, policy, "default_workbook")
    )
    provenance = (
        _path(root, args.provenance)
        if args.provenance
        else _default_path(root, policy, "default_provenance")
    )
    ontology_path = _default_path(root, policy, "ontology")
    rules_path = _default_path(root, policy, "physician_rules")
    mapping_path = _default_path(root, policy, "benchmark_mapping")
    registry_path = _default_path(root, policy, "scenario_registry")
    policy_version = int(str(policy.get("version", 2)))
    raw_ontology_path = (
        _default_path(root, policy, "raw_ontology") if policy_version >= 3 else ontology_path
    )
    raw_rules_path = (
        _default_path(root, policy, "raw_physician_rules") if policy_version >= 3 else rules_path
    )
    raw_mapping_path = (
        _default_path(root, policy, "raw_benchmark_mapping")
        if policy_version >= 3
        else mapping_path
    )
    config_paths = [
        policy_path,
        ontology_path,
        rules_path,
        mapping_path,
        registry_path,
        raw_ontology_path,
        raw_rules_path,
        raw_mapping_path,
    ]
    run_digest = sha256_payload(
        {
            "workbook": sha256_file(workbook),
            "provenance": sha256_file(provenance),
            "configs": {path.name: sha256_file(path) for path in config_paths},
        }
    )
    run_id = args.run_id or f"run_{run_digest[:12]}"
    output_dir = _path(root, args.results_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    response_authority = (
        _path(root, args.response_authority)
        if args.response_authority
        else (_default_path(root, policy, "response_authority") if policy_version >= 3 else None)
    )
    exploratory_unsigned_source = False
    if response_authority is not None:
        authority_code, authority_paths = audit_workbook_pass(
            workbook=workbook,
            provenance=provenance,
            response_authority=response_authority,
            output_dir=output_dir,
        )
        if authority_code != 0:
            authority_gate = read_json(authority_paths["gate"])
            exploratory_unsigned_source = bool(
                getattr(args, "exploratory_unsigned_source", False)
                and allows_unsigned_source_exploration(authority_gate)
            )
            if exploratory_unsigned_source:
                write_json(
                    output_dir / "exploratory_source_status.json",
                    {
                        "status": "exploratory_only",
                        "release_eligible": False,
                        "source_authority_gate_passed": False,
                        "source_authority_status": "source_authority_unresolved",
                        "continuation_reason": (
                            "All workbook, payload, population, mirror-conflict, "
                            "and immutability checks passed; project-owner signoff "
                            "is the sole unresolved G0 check."
                        ),
                        "metric_use_policy": (
                            "Diagnostic only. These metrics cannot satisfy a release "
                            "gate or be represented as confirmatory evidence."
                        ),
                    },
                )
                print(
                    "Clinical-validation source-authority gate: FAIL "
                    "(continuing in explicitly exploratory, release-ineligible mode)"
                )
            else:
                write_artifact_manifest(
                    output_dir,
                    producer_command="tm-ecg clinical-validation run-all",
                    input_hashes={"workbook": sha256_file(workbook)},
                    code_root=root,
                )
                print(
                    "Clinical-validation source-authority gate: FAIL "
                    f"(see {output_dir / 'source_authority_gate.json'})"
                )
                return authority_code
    extracted = extract_pass(
        workbook=workbook,
        provenance=provenance,
        output_dir=output_dir,
        response_authority=response_authority,
        allow_unsigned_source_exploration=exploratory_unsigned_source,
    )
    harmonized_output_dir = output_dir / "harmonized_v3" if policy_version >= 3 else output_dir
    raw_output_dir = output_dir / "raw_v2" if policy_version >= 3 else output_dir
    physician = physician_pass(
        responses=extracted["responses"],
        rules_path=rules_path,
        output_dir=harmonized_output_dir,
    )
    raw_physician = (
        physician_pass(
            responses=extracted["responses"],
            rules_path=raw_rules_path,
            output_dir=raw_output_dir,
        )
        if policy_version >= 3
        else physician
    )
    ptbxl_index = config.paths.manifests / "ptbxl_index.parquet"
    ludb_index = config.paths.manifests / "ludb_index.parquet"
    benchmark = benchmark_pass(
        identities_path=extracted["identities"],
        provenance_path=provenance,
        mapping_path=mapping_path,
        output_dir=harmonized_output_dir,
        ptbxl_index=ptbxl_index if ptbxl_index.exists() else None,
        ludb_index=ludb_index if ludb_index.exists() else None,
    )
    raw_benchmark = (
        benchmark_pass(
            identities_path=extracted["identities"],
            provenance_path=provenance,
            mapping_path=raw_mapping_path,
            output_dir=raw_output_dir,
            ptbxl_index=ptbxl_index if ptbxl_index.exists() else None,
            ludb_index=ludb_index if ludb_index.exists() else None,
        )
        if policy_version >= 3
        else benchmark
    )
    registry_payload, _registered_scenarios = load_scenario_registry(registry_path)
    baseline = compute_baseline_reference(
        [CaseIdentity.from_dict(row) for row in read_jsonl(extracted["identities"])],
        [RawPhysicianResponse.from_dict(row) for row in read_jsonl(extracted["responses"])],
        [ClinicalFindingSet.from_dict(row) for row in read_jsonl(raw_physician["findings"])],
        [BenchmarkFindingSet.from_dict(row) for row in read_jsonl(raw_benchmark["findings"])],
        load_physician_rules(raw_rules_path),
        _string_tuple(registry_payload.get("dominant_priority", [])),
    )
    write_baseline_artifacts(output_dir, baseline)
    evaluated = evaluate_pass(
        identities_path=extracted["identities"],
        physician_path=physician["findings"],
        benchmark_path=benchmark["findings"],
        b2_path=extracted["b2"],
        raw_responses_path=extracted["responses"],
        cardia_x_path=extracted["cardia_x"],
        assisted_review_path=extracted["assisted_review"],
        registry_path=registry_path,
        ontology_path=ontology_path,
        output_dir=output_dir,
        bootstrap_replicates_override=args.bootstrap_replicates,
        raw_physician_path=raw_physician["findings"],
        raw_benchmark_path=raw_benchmark["findings"],
    )
    gate_code, gate = gate_pass(
        scenario_results_path=evaluated["results"],
        policy_path=policy_path,
        registry_path=registry_path,
        ontology_path=ontology_path,
        output_dir=output_dir,
    )
    scenario_results = read_json(evaluated["results"])["scenario_results"]
    failure_classification = _classify_gate_failures(scenario_results, gate)
    failure_classification_path = write_json(
        output_dir / "failure_classification.json", failure_classification
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = ValidationRunManifest(
        run_id=run_id,
        input_hashes={
            "workbook": sha256_file(workbook),
            "provenance": sha256_file(provenance),
            "acceptance_policy": sha256_file(policy_path),
            "physician_rules": sha256_file(rules_path),
            "benchmark_mapping": sha256_file(mapping_path),
            "raw_ontology": sha256_file(raw_ontology_path),
            "raw_physician_rules": sha256_file(raw_rules_path),
            "raw_benchmark_mapping": sha256_file(raw_mapping_path),
        },
        package_version=PACKAGE_VERSION,
        ontology_hash=sha256_file(ontology_path),
        scenario_registry_hash=sha256_file(registry_path),
        random_seed=config.seed,
        environment=environment_manifest(),
        timestamp_utc=timestamp,
    ).to_dict()
    manifest["evaluation_status"] = (
        "exploratory_source_authority_unresolved"
        if exploratory_unsigned_source
        else "release_gate_eligible"
    )
    manifest["release_eligible"] = not exploratory_unsigned_source
    report_path = output_dir / "validation_report.md"
    render_validation_report(
        run_manifest=manifest,
        import_audit=read_json(extracted["audit"]),
        scenario_results=scenario_results,
        gate=gate,
        repeatability=read_json(evaluated["repeatability"]),
        b2_summary=read_json(evaluated["b2_summary"]),
        abstention_concordance=read_json(evaluated["abstention_summary"]),
        cardia_x_track=read_json(evaluated["cardia_x_track"]),
        assisted_review=read_json(evaluated["assisted_review"]),
        failure_classification=read_json(failure_classification_path),
        baseline=baseline,
        rule_review=read_json(extracted["rule_review"]),
        output_path=report_path,
    )
    manifest["output_hashes"] = _output_hashes(output_dir)
    write_json(output_dir / "run_manifest.json", manifest)
    write_artifact_manifest(
        output_dir,
        producer_command="tm-ecg clinical-validation run-all",
        input_hashes=manifest["input_hashes"],
        code_root=root,
    )
    latest_name = "latest_exploratory.json" if exploratory_unsigned_source else "latest.json"
    write_json(
        _path(root, args.results_root) / latest_name,
        {
            "run_id": run_id,
            "run_manifest": str(output_dir / "run_manifest.json"),
            "release_eligible": not exploratory_unsigned_source,
        },
    )
    print(f"Clinical-validation result bundle: {output_dir}")
    cardia_x_gate = read_json(evaluated["cardia_x_gate"])
    model_passed = bool(cardia_x_gate.get("passed"))
    print(f"Physician acceptance gate: {'PASS' if gate_code == 0 else 'FAIL'}")
    print(f"CARDIA-X acceptance gate: {'PASS' if model_passed else 'FAIL'}")
    if exploratory_unsigned_source:
        print(
            "Overall release eligibility: FAIL "
            "(source authority is unsigned; all metrics are exploratory)"
        )
        return 10
    return gate_code if gate_code != 0 else 0 if model_passed else 5


def _individual(config: ProjectConfig, args: argparse.Namespace) -> int:
    root = _root(config)
    policy_path, policy = _load_policy(root, args.policy)
    command = args.clinical_command
    output_dir = _path(root, args.output_dir)
    if command == "audit-workbook":
        workbook = _path(root, args.workbook)
        provenance = (
            _path(root, args.provenance)
            if args.provenance
            else _default_path(root, policy, "default_provenance")
        )
        code, paths = audit_workbook_pass(
            workbook=workbook,
            provenance=provenance,
            response_authority=_path(root, args.response_authority),
            output_dir=output_dir,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return code
    if command == "extract":
        workbook = (
            _path(root, args.workbook)
            if args.workbook
            else _default_path(root, policy, "default_workbook")
        )
        provenance = (
            _path(root, args.provenance)
            if args.provenance
            else _default_path(root, policy, "default_provenance")
        )
        extract_pass(
            workbook=workbook,
            provenance=provenance,
            output_dir=output_dir,
            response_authority=(
                _path(root, args.response_authority) if args.response_authority else None
            ),
        )
        return 0
    if command == "code-physician":
        physician_pass(
            responses=_path(root, args.responses),
            rules_path=_path(root, args.rules),
            output_dir=output_dir,
        )
        return 0
    if command == "lint-physician-coding":
        code, paths = lint_physician_coding(
            responses_path=_path(root, args.responses),
            rules_path=_path(root, args.rules),
            output_dir=output_dir,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return code
    if command == "lint-scenarios":
        code, report = lint_scenario_registry(
            registry_path=_path(root, args.scenarios),
            policy_path=policy_path,
            output_dir=output_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return code
    if command == "code-benchmark":
        benchmark_pass(
            identities_path=_path(root, args.identities),
            provenance_path=_path(root, args.provenance),
            mapping_path=_path(root, args.mapping),
            output_dir=output_dir,
            ptbxl_index=_path(root, args.ptbxl_index) if args.ptbxl_index else None,
            ludb_index=_path(root, args.ludb_index) if args.ludb_index else None,
        )
        return 0
    if command == "evaluate":
        evaluate_pass(
            identities_path=_path(root, args.identities),
            physician_path=_path(root, args.physician_findings),
            benchmark_path=_path(root, args.benchmark_findings),
            b2_path=_path(root, args.b2_responses),
            raw_responses_path=(
                _path(root, args.unassisted_responses) if args.unassisted_responses else None
            ),
            cardia_x_path=(_path(root, args.cardia_x_outputs) if args.cardia_x_outputs else None),
            assisted_review_path=(
                _path(root, args.assisted_review_responses)
                if args.assisted_review_responses
                else None
            ),
            registry_path=_path(root, args.scenarios),
            ontology_path=_path(root, args.ontology),
            output_dir=output_dir,
            bootstrap_replicates_override=args.bootstrap_replicates,
        )
        return 0
    if command == "gate":
        code, gate = gate_pass(
            scenario_results_path=_path(root, args.scenario_results),
            policy_path=policy_path,
            registry_path=_path(root, args.scenarios),
            ontology_path=_path(root, args.ontology),
            output_dir=output_dir,
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return code
    if command == "report":
        manifest_path = _path(root, args.run_manifest)
        manifest = read_json(manifest_path)
        result_dir = manifest_path.parent
        results = read_json(result_dir / "scenario_results.json")["scenario_results"]
        render_validation_report(
            run_manifest=manifest,
            import_audit=read_json(result_dir / "response_import_audit.json"),
            scenario_results=results,
            gate=read_json(result_dir / "gate_result.json"),
            repeatability=read_json(result_dir / "reader_repeatability.json"),
            b2_summary=read_json(result_dir / "b2_plausibility_summary.json"),
            abstention_concordance=(
                read_json(result_dir / "abstention_concordance.json")
                if (result_dir / "abstention_concordance.json").exists()
                else None
            ),
            cardia_x_track=(
                read_json(result_dir / "cardia_x_vs_benchmark.json")
                if (result_dir / "cardia_x_vs_benchmark.json").exists()
                else None
            ),
            assisted_review=(
                read_json(result_dir / "assisted_review_summary.json")
                if (result_dir / "assisted_review_summary.json").exists()
                else None
            ),
            failure_classification=(
                read_json(result_dir / "failure_classification.json")
                if (result_dir / "failure_classification.json").exists()
                else None
            ),
            baseline=(
                read_json(result_dir / "baseline_reference.json")
                if (result_dir / "baseline_reference.json").exists()
                else None
            ),
            rule_review=(
                read_json(result_dir / "rule_review_summary.json")
                if (result_dir / "rule_review_summary.json").exists()
                else None
            ),
            output_path=result_dir / "validation_report.md",
        )
        return 0
    raise ValueError(f"Unknown clinical-validation command: {command}")


def dispatch(config: ProjectConfig, args: argparse.Namespace) -> int:
    try:
        return (
            run_all(config, args)
            if args.clinical_command == "run-all"
            else _individual(config, args)
        )
    except FieldPolicyViolation as exc:
        print(f"Clinical-validation leakage-policy failure: {exc}")
        return 4
    except WorkbookSchemaError as exc:
        print(f"Clinical-validation input/schema failure: {exc}")
        return 2
    except SourceAuthorityError as exc:
        print(f"Clinical-validation source-authority failure: {exc}")
        return 10
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Clinical-validation input/configuration failure: {exc}")
        return 2


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("clinical-validation")
    commands = parser.add_subparsers(dest="clinical_command", required=True)
    common_policy = "clinical_validation/config/acceptance_policy_v2.yaml"

    audit_workbook = commands.add_parser("audit-workbook")
    audit_workbook.add_argument(
        "--workbook",
        default="clinical_validation/review/CARDIA-X_clinical_validation_v6.xlsx",
    )
    audit_workbook.add_argument(
        "--response-authority",
        default="clinical_validation/config/response_authority_v1.yaml",
    )
    audit_workbook.add_argument("--provenance")
    audit_workbook.add_argument("--policy", default=common_policy)
    audit_workbook.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        default="clinical_validation/results/v3_source_audit",
    )

    extract = commands.add_parser("extract")
    extract.add_argument("--workbook")
    extract.add_argument("--provenance")
    extract.add_argument("--response-authority")
    extract.add_argument("--policy", default=common_policy)
    extract.add_argument("--output-dir", default="clinical_validation/derived")

    physician = commands.add_parser("code-physician")
    physician.add_argument("--responses", required=True)
    physician.add_argument(
        "--rules", default="clinical_validation/config/physician_coding_rules_v2.yaml"
    )
    physician.add_argument("--policy", default=common_policy)
    physician.add_argument("--output-dir", default="clinical_validation/derived")

    physician_lint = commands.add_parser("lint-physician-coding")
    physician_lint.add_argument("--responses", required=True)
    physician_lint.add_argument(
        "--rules",
        default="clinical_validation/config/physician_coding_rules_v3.yaml",
    )
    physician_lint.add_argument("--policy", default=common_policy)
    physician_lint.add_argument(
        "--output-dir",
        default="clinical_validation/results/v3_physician_coding_lint",
    )

    scenario_lint = commands.add_parser("lint-scenarios")
    scenario_lint.add_argument(
        "--scenarios",
        default="clinical_validation/config/scenario_registry_v3.yaml",
    )
    scenario_lint.add_argument(
        "--policy",
        default="clinical_validation/config/acceptance_policy_v3.yaml",
    )
    scenario_lint.add_argument(
        "--output-dir",
        default="clinical_validation/results/v3_scenario_registry_lint",
    )

    benchmark = commands.add_parser("code-benchmark")
    benchmark.add_argument("--identities", required=True)
    benchmark.add_argument("--provenance", required=True)
    benchmark.add_argument(
        "--mapping", default="clinical_validation/config/benchmark_mapping_v2.yaml"
    )
    benchmark.add_argument("--ptbxl-index")
    benchmark.add_argument("--ludb-index")
    benchmark.add_argument("--policy", default=common_policy)
    benchmark.add_argument("--output-dir", default="clinical_validation/derived")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--identities", required=True)
    evaluate.add_argument("--physician-findings", required=True)
    evaluate.add_argument("--benchmark-findings", required=True)
    evaluate.add_argument("--b2-responses", required=True)
    evaluate.add_argument("--unassisted-responses")
    evaluate.add_argument("--cardia-x-outputs")
    evaluate.add_argument("--assisted-review-responses")
    evaluate.add_argument(
        "--scenarios", default="clinical_validation/config/scenario_registry_v2.yaml"
    )
    evaluate.add_argument(
        "--ontology", default="clinical_validation/config/validation_ontology_v2.yaml"
    )
    evaluate.add_argument("--bootstrap-replicates", type=int)
    evaluate.add_argument("--policy", default=common_policy)
    evaluate.add_argument("--output-dir", default="clinical_validation/results/latest")

    gate = commands.add_parser("gate")
    gate.add_argument("--scenario-results", required=True)
    gate.add_argument("--scenarios", default="clinical_validation/config/scenario_registry_v2.yaml")
    gate.add_argument(
        "--ontology", default="clinical_validation/config/validation_ontology_v2.yaml"
    )
    gate.add_argument("--policy", default=common_policy)
    gate.add_argument("--minimum-kappa", type=float, default=0.70)
    gate.add_argument("--output-dir", default="clinical_validation/results/latest")

    report = commands.add_parser("report")
    report.add_argument("--run-manifest", required=True)
    report.add_argument("--policy", default=common_policy)
    report.add_argument("--output-dir", default="clinical_validation/results/latest")

    run_all_parser = commands.add_parser("run-all")
    run_all_parser.add_argument("--workbook")
    run_all_parser.add_argument("--provenance")
    run_all_parser.add_argument("--response-authority")
    run_all_parser.add_argument(
        "--policy",
        "--acceptance-policy",
        dest="policy",
        default=common_policy,
    )
    run_all_parser.add_argument("--results-root", default="clinical_validation/results")
    run_all_parser.add_argument("--run-id")
    run_all_parser.add_argument("--bootstrap-replicates", type=int)
    run_all_parser.add_argument(
        "--exploratory-unsigned-source",
        action="store_true",
        help=(
            "Continue only when missing owner signoff is the sole source-authority "
            "failure; outputs remain release-ineligible and the command exits 10."
        ),
    )

    for command in (
        audit_workbook,
        extract,
        physician,
        physician_lint,
        scenario_lint,
        benchmark,
        evaluate,
        gate,
        report,
        run_all_parser,
    ):
        command.set_defaults(handler=dispatch)
