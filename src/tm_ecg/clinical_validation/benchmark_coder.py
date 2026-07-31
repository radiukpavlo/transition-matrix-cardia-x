"""Independent benchmark-to-validation-ontology coding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from tm_ecg.clinical_validation.models import (
    BenchmarkFindingSet,
    CaseIdentity,
    SourceStatementTrace,
)
from tm_ecg.clinical_validation.ontology import load_json_yaml, normalize_axis_values
from tm_ecg.ptbxl_semantics import (
    StatementMetadata,
    load_statement_metadata,
    source_statement_trace,
)


def _split_labels(value: object) -> list[str]:
    text = str(value or "")
    delimiter = "|" if "|" in text else ","
    return [item.strip() for item in text.split(delimiter) if item.strip()]


def _index_by_record(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    if path.suffix == ".parquet":
        import pandas as pd  # type: ignore[import-untyped]

        frame = pd.read_parquet(path)
        return {
            str(row["record_id"]): {str(key): value for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        }
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row["record_id"]): dict(row) for row in csv.DictReader(handle)}


def _rule_matches(rule: Mapping[str, object], label: str) -> bool:
    mode = str(rule.get("match", "exact_ci"))
    candidate = label.strip().lower()
    raw_terms = rule.get("terms", [])
    terms = [
        str(term).strip().lower()
        for term in (raw_terms if isinstance(raw_terms, list) else [])
    ]
    if mode == "contains_ci":
        return any(term in candidate for term in terms)
    return candidate in terms


def _extract_axis_targets(row: Mapping[str, object]) -> dict[str, object] | None:
    raw = row.get("axis_targets_json")
    if raw in {None, ""}:
        return None
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_source_likelihoods(row: Mapping[str, object]) -> dict[str, float]:
    raw = row.get("source_likelihoods_json")
    if raw in {None, ""}:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            result[str(key)] = float(str(value))
        except (TypeError, ValueError):
            continue
    return result


def _confidence_thresholds(mapping: Mapping[str, object]) -> tuple[float, float]:
    raw = mapping.get("confidence_bands", {})
    bands = raw if isinstance(raw, Mapping) else {}
    accepted = float(str(bands.get("accepted_min", 80.0)))
    uncertain = float(str(bands.get("uncertain_min", 50.0)))
    if not 0.0 <= uncertain <= accepted <= 100.0:
        raise ValueError(
            "Benchmark confidence bands must satisfy 0 <= uncertain_min <= accepted_min <= 100"
        )
    return accepted, uncertain


def _categorical_presence_terms(mapping: Mapping[str, object]) -> set[str]:
    raw = mapping.get("categorical_presence_terms", [])
    if not isinstance(raw, list):
        raise ValueError("categorical_presence_terms must be a list")
    return {str(term).strip().upper() for term in raw if str(term).strip()}


def _mapped_uncertain_findings(
    labels: list[str], rules: list[Mapping[str, object]]
) -> tuple[str, ...]:
    findings: set[str] = set()
    for label in labels:
        for rule in rules:
            if str(rule.get("id")) == "BENCH-OTHER" or not _rule_matches(rule, label):
                continue
            axis = str(rule["axis"])
            finding = str(rule["finding"])
            findings.add(f"{axis}:{finding}")
    return tuple(sorted(findings))


def code_benchmark_case(
    identity: CaseIdentity,
    provenance_row: Mapping[str, object],
    mapping: Mapping[str, object],
    index_row: Mapping[str, object] | None = None,
) -> BenchmarkFindingSet:
    curated_labels = _split_labels(provenance_row.get("hidden_benchmark_label"))
    index = dict(index_row or {})
    raw_rules = mapping.get("rules", [])
    rules = [rule for rule in raw_rules if isinstance(rule, Mapping)] if isinstance(raw_rules, list) else []
    likelihoods = _extract_source_likelihoods(index)
    accepted_min, uncertain_min = _confidence_thresholds(mapping)
    categorical_terms = _categorical_presence_terms(mapping)
    mapping_version = int(str(mapping.get("version", 2)))
    raw_statement_metadata = mapping.get("_statement_metadata", {})
    statement_metadata = (
        raw_statement_metadata
        if isinstance(raw_statement_metadata, Mapping)
        else {}
    )
    statement_trace_rows: tuple[dict[str, object], ...] = ()
    source_labels: list[str]
    accepted_labels: list[str]
    uncertain_labels: list[str]
    ignored_labels: list[str]
    if identity.dataset == "ptbxl" and likelihoods:
        # The legacy project projection was created before confidence-aware
        # harmonization.  Use the preserved PTB-XL source statements instead,
        # applying the already frozen confidence policy from the mapping file.
        if mapping_version >= 3:
            typed_metadata = {
                str(code): metadata
                for code, metadata in statement_metadata.items()
                if isinstance(metadata, StatementMetadata)
            }
            statement_trace_rows = source_statement_trace(
                likelihoods,
                typed_metadata,
                accepted_min=accepted_min,
                uncertain_min=uncertain_min,
            )
            accepted_labels = sorted(
                str(row["code"])
                for row in statement_trace_rows
                if row["state"] == "present"
            )
            uncertain_labels = sorted(
                str(row["code"])
                for row in statement_trace_rows
                if row["state"] == "uncertain"
            )
            ignored_labels = sorted(
                str(row["code"])
                for row in statement_trace_rows
                if row["state"] == "ignored"
            )
        else:
            accepted_labels = sorted(
                code
                for code, likelihood in likelihoods.items()
                if likelihood >= accepted_min or code.strip().upper() in categorical_terms
            )
            uncertain_labels = sorted(
                code
                for code, likelihood in likelihoods.items()
                if code.strip().upper() not in categorical_terms
                and uncertain_min <= likelihood < accepted_min
            )
            ignored_labels = sorted(
                code
                for code, likelihood in likelihoods.items()
                if code.strip().upper() not in categorical_terms
                and likelihood < uncertain_min
            )
        source_labels = sorted(likelihoods)
    else:
        source_labels = list(curated_labels)
        uncertain_labels = []
        ignored_labels = []
        for field in ("raw_source_labels", "source_codes", "scp_codes_list", "diagnosis_text"):
            source_labels.extend(_split_labels(index.get(field)))
        accepted_labels = list(dict.fromkeys(source_labels))
    source_labels = list(dict.fromkeys(source_labels))

    axes: dict[str, object] = {
        "rhythm": [],
        "ectopy": [],
        "conduction": [],
        "repolarization": [],
        "pacing": "absent",
        "normality": "indeterminate",
        "quality": "indeterminate",
        "interpretability": "indeterminate",
        "residual_abnormal": False,
    }
    rule_ids: list[str] = []
    embedded = _extract_axis_targets(index)
    if embedded:
        if identity.dataset == "ptbxl" and likelihoods:
            # Axis lists in the shared index use a different (50%) threshold;
            # only scalar metadata independent of statement confidence may be
            # reused here.
            for key in ("pacing", "quality", "interpretability"):
                if key in embedded:
                    axes[key] = embedded[key]
        else:
            axes.update(embedded)
    nonfallback_match_exists = any(
        str(rule.get("id")) != "BENCH-OTHER" and _rule_matches(rule, label)
        for label in accepted_labels
        for rule in rules
    )
    for label in accepted_labels:
        matched = False
        for rule in rules:
            if not _rule_matches(rule, label):
                continue
            if str(rule.get("id")) == "BENCH-OTHER" and nonfallback_match_exists:
                # Consume the workbook's catch-all marker without allowing it
                # to erase a more specific source diagnosis.
                matched = True
                continue
            matched = True
            axis = str(rule["axis"])
            finding = str(rule["finding"])
            if axis in {"pacing", "normality", "quality", "interpretability"}:
                axes[axis] = finding
            elif axis == "residual":
                axes["residual_abnormal"] = True
            else:
                current = axes.get(axis, [])
                values = list(current) if isinstance(current, list) else []
                if finding not in values:
                    values.append(finding)
                axes[axis] = values
            rule_ids.append(str(rule["id"]))
        if not matched and label.lower() not in {"", "none"}:
            axes["residual_abnormal"] = True
            rule_ids.append("BENCH-UNKNOWN-RESIDUAL")

    normalized = normalize_axis_values(axes)
    abnormal = bool(
        normalized["ectopy"]
        or normalized["conduction"]
        or normalized["repolarization"]
        or set(normalized["rhythm"]) & {"af", "afl", "other_arrhythmia"}  # type: ignore[arg-type]
        or normalized["pacing"] == "present"
        or normalized["residual_abnormal"]
    )
    normality = "abnormal" if abnormal else str(normalized["normality"])
    return BenchmarkFindingSet(
        case_id=identity.workbook_case_id,
        rhythm=normalized["rhythm"],  # type: ignore[arg-type]
        ectopy=normalized["ectopy"],  # type: ignore[arg-type]
        conduction=normalized["conduction"],  # type: ignore[arg-type]
        repolarization=normalized["repolarization"],  # type: ignore[arg-type]
        pacing=str(normalized["pacing"]),
        normality=normality,
        quality=str(normalized["quality"]),
        interpretability=str(normalized["interpretability"]),
        residual_abnormal=bool(normalized["residual_abnormal"]),
        uncertain_findings=_mapped_uncertain_findings(uncertain_labels, rules),
        source_labels=tuple(source_labels),
        accepted_source_labels=tuple(accepted_labels),
        uncertain_source_labels=tuple(uncertain_labels),
        ignored_source_labels=tuple(ignored_labels),
        mapping_rule_ids=tuple(sorted(set(rule_ids))),
        source_statement_trace=tuple(
            SourceStatementTrace.from_dict(row) for row in statement_trace_rows
        ),
    )


def code_benchmarks(
    identities: list[CaseIdentity],
    provenance: Mapping[str, Mapping[str, object]],
    mapping: Mapping[str, object],
    ptbxl_index: Path | None = None,
    ludb_index: Path | None = None,
) -> list[BenchmarkFindingSet]:
    indexes = {
        "ptbxl": _index_by_record(ptbxl_index),
        "ludb": _index_by_record(ludb_index),
    }
    results = []
    for identity in identities:
        row = provenance.get(identity.workbook_case_id)
        if row is None:
            raise ValueError(f"Missing provenance for {identity.workbook_case_id}")
        index_row = indexes.get(identity.dataset, {}).get(identity.source_record_id, {})
        results.append(code_benchmark_case(identity, row, mapping, index_row))
    return results


def load_benchmark_mapping(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = load_json_yaml(source)
    version = payload.get("version")
    if version not in {2, 3}:
        raise ValueError("Benchmark mapping must declare version 2 or 3")
    if version == 3:
        metadata_value = payload.get("statement_metadata_path")
        if not isinstance(metadata_value, str) or not metadata_value.strip():
            raise ValueError(
                "Benchmark mapping v3 must declare statement_metadata_path"
            )
        metadata_path = Path(metadata_value)
        if not metadata_path.is_absolute():
            repository_root = source.resolve().parents[2]
            metadata_path = repository_root / metadata_path
        payload["_statement_metadata"] = load_statement_metadata(metadata_path)
    return payload
