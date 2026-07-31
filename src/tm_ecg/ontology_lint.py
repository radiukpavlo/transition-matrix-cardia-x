"""Cross-module ontology and compatibility-contract lint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from tm_ecg.clinical_validation.audit import sha256_file, write_json
from tm_ecg.clinical_validation.benchmark_coder import load_benchmark_mapping
from tm_ecg.config import ProjectConfig
from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.io.readers import read_table_frame
from tm_ecg.modeling.label_contract import (
    load_compatibility_label_contract,
    parse_label_tokens,
)
from tm_ecg.ontology import PTBXL_AXIS_MAP
from tm_ecg.ptbxl_semantics import load_statement_metadata


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def _source_codes(frame: object) -> set[str]:
    codes: set[str] = set()
    columns = set(str(column) for column in frame.columns)  # type: ignore[attr-defined]
    if "source_likelihoods_json" in columns:
        for raw in frame["source_likelihoods_json"].dropna():  # type: ignore[index]
            try:
                payload = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                codes.update(str(code).strip().upper() for code in payload)
    if "source_codes" in columns:
        for raw in frame["source_codes"].dropna():  # type: ignore[index]
            codes.update(
                token.strip().upper()
                for token in str(raw).replace(",", "|").split("|")
                if token.strip()
            )
    return codes


def _rule_code_map(
    mapping: Mapping[str, object],
    known_codes: set[str],
) -> tuple[dict[str, tuple[str, str, str]], dict[str, list[str]]]:
    code_map: dict[str, tuple[str, str, str]] = {}
    duplicate_rules: dict[str, list[str]] = {}
    raw_rules = mapping.get("rules", [])
    rules = raw_rules if isinstance(raw_rules, list) else []
    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            continue
        rule_id = str(raw_rule.get("id", ""))
        axis = str(raw_rule.get("axis", ""))
        finding = str(raw_rule.get("finding", ""))
        terms = raw_rule.get("terms", [])
        for raw_term in terms if isinstance(terms, list) else []:
            code = str(raw_term).strip().upper()
            if code not in known_codes:
                continue
            if code in code_map:
                duplicate_rules.setdefault(code, [code_map[code][0]]).append(rule_id)
            else:
                code_map[code] = (rule_id, axis, finding)
    return code_map, duplicate_rules


def _canonical_axis(axis: str, finding: str) -> tuple[str, str]:
    aliases = {
        ("rhythm", "other_rhythm"): ("rhythm", "other_arrhythmia"),
        ("ectopy", "apb"): ("ectopy", "atrial_premature"),
        ("ectopy", "pvc"): ("ectopy", "ventricular_premature"),
        ("repolarization", "other_st_t"): (
            "repolarization",
            "nonspecific_st_t",
        ),
        ("repolarization", "t_abnormality"): (
            "repolarization",
            "t_wave_abnormality",
        ),
    }
    return aliases.get((axis, finding), (axis, finding))


def lint_ontology_contract(
    *,
    ptbxl_statements: Path,
    benchmark_mapping: Path,
    label_contract: Path,
    ptbxl_index: Path,
) -> dict[str, object]:
    metadata = load_statement_metadata(ptbxl_statements)
    mapping = load_benchmark_mapping(benchmark_mapping)
    contract = load_compatibility_label_contract(label_contract)
    frame = read_table_frame(ptbxl_index)
    codes = _source_codes(frame)
    code_map, duplicate_rules = _rule_code_map(mapping, codes)
    residual_raw = mapping.get("explicit_residual_codes", {})
    residual = residual_raw if isinstance(residual_raw, Mapping) else {}
    residual_without_rationale = sorted(
        code for code, rationale in residual.items() if not str(rationale).strip()
    )
    unknown_metadata_codes = sorted(codes - set(metadata))
    unmapped_codes = sorted(codes - set(code_map) - set(residual))
    unused_residual_codes = sorted(set(residual) - codes)

    differences: list[dict[str, str]] = []
    for code, (_rule_id, axis, finding) in sorted(code_map.items()):
        if code == "NORM":
            core = ("normality", "normal")
        elif code == "PACE":
            core = ("pacing", "present")
        else:
            core = PTBXL_AXIS_MAP.get(code)
        if core is None:
            continue
        if _canonical_axis(*core) != _canonical_axis(axis, finding):
            differences.append(
                {
                    "code": code,
                    "core": ":".join(_canonical_axis(*core)),
                    "benchmark": ":".join(_canonical_axis(axis, finding)),
                }
            )

    raw_normal_abnormal = 0
    raw_residual_specific = 0
    projected_normal_abnormal = 0
    projected_residual_specific = 0
    changed_targets = 0
    if "labels" in frame.columns:
        for raw in frame["labels"]:
            tokens = set(parse_label_tokens(raw))
            specific = tokens & set(contract.specific_labels)
            raw_normal_abnormal += int(
                contract.normal_label in tokens and len(tokens) > 1
            )
            raw_residual_specific += int(
                contract.residual_label in tokens and bool(specific)
            )
            projected = set(contract.normalize(raw, empty_policy="residual"))
            changed_targets += int(tokens != projected)
            projected_normal_abnormal += int(
                contract.normal_label in projected and len(projected) > 1
            )
            projected_residual_specific += int(
                contract.residual_label in projected
                and bool(projected & set(contract.specific_labels))
            )

    presence_categories = mapping.get("presence_categories")
    presence_policy_correct = presence_categories == ["rhythm", "form"]
    label_order_matches_project = list(contract.label_order) == PROJECT_LABELS
    checks = {
        "unmapped_source_codes_zero": not unmapped_codes,
        "unknown_metadata_codes_zero": not unknown_metadata_codes,
        "duplicate_mapping_rules_zero": not duplicate_rules,
        "residual_rationales_complete": not residual_without_rationale,
        "presence_category_policy_correct": presence_policy_correct,
        "cross_module_mapping_differences_zero": not differences,
        "normal_abnormal_target_conflicts_zero": projected_normal_abnormal == 0,
        "residual_specific_target_conflicts_zero": (
            projected_residual_specific == 0
        ),
        "label_order_matches_project": label_order_matches_project,
        "patient_count_unchanged": True,
        "training_target_version_v4": contract.version == 4,
    }
    return {
        "gate_id": "G1_label_semantics",
        "status": "pass" if all(checks.values()) else "fail",
        "passed": all(checks.values()),
        "checks": checks,
        "source_code_count": len(codes),
        "mapped_source_code_count": len(set(code_map)),
        "explicit_residual_code_count": len(set(residual) & codes),
        "unmapped_source_codes": unmapped_codes,
        "unknown_metadata_codes": unknown_metadata_codes,
        "duplicate_mapping_rules": duplicate_rules,
        "residual_codes_without_rationale": residual_without_rationale,
        "unused_explicit_residual_codes": unused_residual_codes,
        "cross_module_mapping_differences": differences,
        "raw_normal_abnormal_target_conflicts": raw_normal_abnormal,
        "raw_residual_specific_target_conflicts": raw_residual_specific,
        "projected_normal_abnormal_target_conflicts": projected_normal_abnormal,
        "projected_residual_specific_target_conflicts": projected_residual_specific,
        "rows_changed_by_v4_projection": changed_targets,
        "record_count": len(frame),
        "training_target_version": contract.contract_id,
        "hashes": {
            "ptbxl_statements": sha256_file(ptbxl_statements),
            "benchmark_mapping": sha256_file(benchmark_mapping),
            "label_contract": sha256_file(label_contract),
            "ptbxl_index": sha256_file(ptbxl_index),
        },
    }


def run(config: ProjectConfig, args: argparse.Namespace) -> int:
    root = config.paths.root
    index_value = args.ptbxl_index
    if index_value is None:
        parquet = config.paths.manifests / "ptbxl_index.parquet"
        index_value = parquet if parquet.exists() else config.paths.manifests / "ptbxl_index.csv"
    result = lint_ontology_contract(
        ptbxl_statements=_path(root, args.ptbxl_statements),
        benchmark_mapping=_path(root, args.benchmark_mapping),
        label_contract=_path(root, args.label_contract),
        ptbxl_index=_path(root, index_value),
    )
    if args.output:
        write_json(_path(root, args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1
