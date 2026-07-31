"""Coverage and leakage lint for benchmark-blind physician coding."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Mapping, Sequence

from tm_ecg.clinical_validation.audit import (
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from tm_ecg.clinical_validation.field_policy import (
    ALLOWED_UNASSISTED_FIELDS,
    PROHIBITED_FIELDS,
    build_physician_view,
)
from tm_ecg.clinical_validation.models import ClinicalAssertion, ClinicalFindingSet
from tm_ecg.clinical_validation.physician_coder import (
    code_physician_response,
    load_physician_rules,
    normalize_clinical_text,
)


SPECIFIC_FINDINGS = frozenset(
    {
        "af",
        "afl",
        "atrial_premature",
        "ventricular_premature",
        "rbbb_spectrum",
        "lbbb_spectrum",
        "st_elevation",
        "st_depression",
        "t_wave_abnormality",
        "qtc_abnormality",
    }
)
UNSPECIFIED_PAIRS = (
    ("ectopy", frozenset({"atrial_premature", "ventricular_premature", "mixed_ectopy"}), "unspecified_ectopy"),
    ("conduction", frozenset({"rbbb_spectrum", "lbbb_spectrum", "other_conduction"}), "unspecified_conduction"),
    (
        "repolarization",
        frozenset(
            {
                "st_elevation",
                "st_depression",
                "t_wave_abnormality",
                "qtc_abnormality",
            }
        ),
        "unspecified_repolarization",
    ),
)
TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9-]*")
TOKEN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "ecg",
        "electrocardiogram",
        "for",
        "from",
        "in",
        "is",
        "it",
        "lead",
        "leads",
        "no",
        "not",
        "of",
        "on",
        "or",
        "rhythm",
        "seen",
        "the",
        "to",
        "with",
        "without",
    }
)


def _authority_row_to_payload(row: Mapping[str, object]) -> dict[str, object]:
    fields = row.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("Authority-normalized response row is missing fields")
    payload: dict[str, object] = {
        "case_id": str(row.get("workbook_case_id", "")),
        "source_cells": {},
    }
    source_cells: dict[str, str] = {}
    for field in sorted(ALLOWED_UNASSISTED_FIELDS - {"case_id"}):
        item = fields.get(field)
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Authority-normalized response is missing canonical field {field}"
            )
        payload[field] = item.get("raw_value", "")
        source_cells[field] = str(item.get("source_cell", ""))
    payload["source_cells"] = source_cells
    return payload


def _raw_row_to_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key in ALLOWED_UNASSISTED_FIELDS or key == "source_cells"
    }


def _payload_from_row(row: Mapping[str, object]) -> dict[str, object]:
    return (
        _authority_row_to_payload(row)
        if isinstance(row.get("fields"), Mapping)
        else _raw_row_to_payload(row)
    )


def _list_values(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, list) else ()


def _definite_assertions(
    findings: ClinicalFindingSet,
) -> set[tuple[str, str]]:
    return {
        (item.axis, item.finding)
        for item in (*findings.explicit_diagnoses, *findings.axis_derivations)
        if item.assertion_status == "definite"
    }


def _emitted_findings(
    findings: ClinicalFindingSet,
) -> set[tuple[str, str]]:
    emitted = {
        (axis, value)
        for axis in ("rhythm", "ectopy", "conduction", "repolarization")
        for value in getattr(findings, axis)
    }
    if findings.pacing == "present":
        emitted.add(("pacing", "present"))
    if findings.residual_abnormal:
        emitted.add(("residual", "other_abnormal"))
    if findings.normality == "normal":
        emitted.add(("normality", "normal"))
    return emitted


def _unmatched_tokens(
    payload: Mapping[str, object],
    assertions: Sequence[ClinicalAssertion],
) -> set[str]:
    covered = {
        token
        for item in assertions
        for token in TOKEN_PATTERN.findall(normalize_clinical_text(item.normalized_span))
    }
    text = normalize_clinical_text(
        f"{payload.get('primary_diagnosis', '')} {payload.get('rationale', '')}"
    )
    return {
        token
        for token in TOKEN_PATTERN.findall(text)
        if token not in covered and token not in TOKEN_STOPWORDS and len(token) > 1
    }


def lint_physician_coding(
    *,
    responses_path: str | Path,
    rules_path: str | Path,
    output_dir: str | Path,
) -> tuple[int, dict[str, Path]]:
    """Code every response and emit a complete, benchmark-blind audit bundle."""

    response_source = Path(responses_path)
    rules_source = Path(rules_path)
    destination = Path(output_dir)
    rows = read_jsonl(response_source)
    rules = load_physician_rules(rules_source)
    if int(str(rules.get("version", 0))) != 3:
        raise ValueError("Physician coding lint requires version 3 rules")

    coverage: Counter[tuple[str, str, str, str]] = Counter()
    unmatched: Counter[str] = Counter()
    traces: list[dict[str, object]] = []
    evidence_only_specific: list[dict[str, str]] = []
    untraced: list[dict[str, str]] = []
    unresolved_normality: list[str] = []
    duplicate_states: list[dict[str, str]] = []
    uncertain_promotions: list[dict[str, str]] = []
    indeterminate_cases: list[str] = []
    b1_original_ids: set[str] = set()

    for row in rows:
        payload = _payload_from_row(row)
        # build_physician_view is the executable information boundary.  The
        # linter never passes authority metadata or any project outcome fields.
        view = build_physician_view(payload)
        findings = code_physician_response(view, rules)
        assertions = (
            *findings.explicit_diagnoses,
            *findings.observations,
            *findings.axis_derivations,
        )
        for item in assertions:
            coverage[
                (
                    item.rule_id,
                    item.source_field,
                    item.normalized_span,
                    item.assertion_status,
                )
            ] += 1
        unmatched.update(_unmatched_tokens(payload, assertions))

        definite_explicit = {
            (item.axis, item.finding)
            for item in findings.explicit_diagnoses
            if item.assertion_status == "definite"
        }
        definite_traces = _definite_assertions(findings)
        emitted = _emitted_findings(findings)
        for axis, finding in sorted(emitted):
            if finding in SPECIFIC_FINDINGS and (axis, finding) not in definite_explicit:
                evidence_only_specific.append(
                    {
                        "case_id": findings.case_id,
                        "axis": axis,
                        "finding": finding,
                    }
                )
            if (axis, finding) not in definite_traces:
                untraced.append(
                    {
                        "case_id": findings.case_id,
                        "axis": axis,
                        "finding": finding,
                    }
                )

        for axis, specifics, unspecified in UNSPECIFIED_PAIRS:
            values = set(getattr(findings, axis))
            if unspecified in values and values.intersection(specifics):
                duplicate_states.append(
                    {
                        "case_id": findings.case_id,
                        "axis": axis,
                        "finding": unspecified,
                    }
                )

        uncertain = {
            (item.axis, item.finding)
            for item in findings.explicit_diagnoses
            if item.assertion_status == "uncertain"
        }
        for axis, finding in sorted(uncertain.intersection(emitted)):
            if (axis, finding) not in definite_explicit:
                uncertain_promotions.append(
                    {
                        "case_id": findings.case_id,
                        "axis": axis,
                        "finding": finding,
                    }
                )

        explicit_normal = ("normality", "normal") in definite_explicit
        if (
            explicit_normal
            and findings.normality == "abnormal"
            and not findings.conflict_resolution_trace
        ):
            unresolved_normality.append(findings.case_id)
        if findings.normality == "indeterminate":
            indeterminate_cases.append(findings.case_id)

        if str(row.get("dataset_branch", "")) == "B1":
            b1_original_ids.add(
                str(row.get("original_case_id", row.get("workbook_case_id", "")))
            )
        traces.append(
            {
                "case_id": findings.case_id,
                "original_case_id": str(row.get("original_case_id", findings.case_id)),
                "dataset_branch": str(row.get("dataset_branch", "")),
                "input_row_sha256": row.get("row_sha256"),
                "input_fields": sorted(payload),
                "finding_set": findings.to_dict(),
            }
        )

    visible_input_fields = {
        str(key)
        for trace in traces
        for key in _list_values(trace.get("input_fields"))
    }
    benchmark_visible = any(
        key in PROHIBITED_FIELDS or "benchmark" in key.lower()
        for key in visible_input_fields
    )
    post_assistance_visible = any(
        any(marker in key.lower() for marker in ("assisted", "cardia_x", "utility"))
        for key in visible_input_fields
    )
    gate = {
        "benchmark_fields_visible_to_coder": benchmark_visible,
        "post_assistance_fields_visible_to_coder": post_assistance_visible,
        "evidence_only_specific_diagnoses": len(evidence_only_specific),
        "untraced_emitted_findings": len(untraced),
        "normal_abnormal_unresolved_conflicts": len(unresolved_normality),
        "specific_unspecified_duplicates": len(duplicate_states),
        "all_150_rows_coded": len(traces) == 150,
        "all_100_unique_b1_rows_retained": len(b1_original_ids) == 100,
    }
    gate["passed"] = (
        not benchmark_visible
        and not post_assistance_visible
        and not evidence_only_specific
        and not untraced
        and not unresolved_normality
        and not duplicate_states
        and len(traces) == 150
        and len(b1_original_ids) == 100
    )
    report = {
        "version": 1,
        "responses": str(response_source),
        "responses_sha256": sha256_file(response_source),
        "rules": str(rules_source),
        "rules_sha256": sha256_file(rules_source),
        "coded_row_count": len(traces),
        "unique_b1_case_count": len(b1_original_ids),
        "rule_coverage": [
            {
                "rule_id": rule_id,
                "source_field": field,
                "normalized_phrase_or_value": phrase,
                "assertion_status": status,
                "count": count,
            }
            for (rule_id, field, phrase, status), count in sorted(coverage.items())
        ],
        "unmatched_clinical_tokens": [
            {"token": token, "count": count}
            for token, count in unmatched.most_common()
        ],
        "normal_abnormal_unresolved_conflicts": unresolved_normality,
        "specific_unspecified_duplicates": duplicate_states,
        "uncertain_findings_promoted_to_definite": uncertain_promotions,
        "evidence_only_specific_diagnoses": evidence_only_specific,
        "untraced_emitted_findings": untraced,
        "indeterminate_normality_cases": indeterminate_cases,
        "gate": gate,
    }

    destination.mkdir(parents=True, exist_ok=True)
    coverage_rows = report["rule_coverage"]
    paths = {
        "report": write_json(destination / "physician_coding_lint.json", report),
        "trace": write_jsonl(
            destination / "physician_coding_trace.jsonl", traces
        ),
        "coverage": write_csv(
            destination / "physician_rule_coverage.csv",
            coverage_rows if isinstance(coverage_rows, list) else [],
        ),
        "gate": write_json(destination / "physician_coding_gate.json", gate),
    }
    return (0 if gate["passed"] else 11), paths
