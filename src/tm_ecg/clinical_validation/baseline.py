"""Reproduce primary-only and broad unassisted starting points."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from tm_ecg.clinical_validation.audit import write_json
from tm_ecg.clinical_validation.metrics import compute_cohen_kappa
from tm_ecg.clinical_validation.models import (
    BenchmarkFindingSet,
    CaseIdentity,
    ClinicalFindingSet,
    RawPhysicianResponse,
)
from tm_ecg.clinical_validation.ontology import projection_value
from tm_ecg.clinical_validation.physician_coder import code_physician_response
from tm_ecg.clinical_validation.field_policy import PhysicianView


PLAN_REFERENCE_APPROXIMATIONS = {
    "strict": {"exact": 0.048, "five_family": 0.045, "binary": 0.022, "dominant": 0.047},
    "broader": {"exact": 0.382, "five_family": 0.423, "binary": 0.341, "dominant": 0.411},
}


def compute_baseline_reference(
    identities: list[CaseIdentity],
    responses: list[RawPhysicianResponse],
    broad_findings: list[ClinicalFindingSet],
    benchmark_findings: list[BenchmarkFindingSet],
    physician_rules: Mapping[str, object],
    dominant_priority: tuple[str, ...],
) -> dict[str, object]:
    response_by_id = {item.case_id: item for item in responses}
    broad_by_id = {item.case_id: item for item in broad_findings}
    benchmark_by_id = {item.case_id: item for item in benchmark_findings}
    rows = sorted(
        (
            item
            for item in identities
            if item.dataset == "ptbxl" and not item.is_repeat
        ),
        key=lambda item: item.row_ordinal,
    )
    if len(rows) != 100:
        raise ValueError(f"Baseline requires 100 unique B1 rows, found {len(rows)}")
    strict: dict[str, ClinicalFindingSet] = {}
    for identity in rows:
        response = response_by_id[identity.workbook_case_id]
        primary_only = RawPhysicianResponse(
            case_id=response.case_id,
            primary_diagnosis=response.primary_diagnosis,
            rationale="",
            diagnostic_confidence=None,
            ecg_quality=None,
            ambiguous="",
            requires_additional_information="",
            evidence=(),
            source_cells=tuple(
                item for item in response.source_cells if item[0] == "primary_diagnosis"
            ),
        )
        strict[identity.workbook_case_id] = code_physician_response(
            PhysicianView(primary_only), physician_rules
        )

    output: dict[str, object] = {
        "population": "100 unique B1 cases",
        "strict_definition": "primary_diagnosis only",
        "broader_definition": (
            "primary_diagnosis, rationale, and pre-assistance evidence fields; "
            "confidence/quality affect interpretability only"
        ),
        "plan_reference_approximations": PLAN_REFERENCE_APPROXIMATIONS,
    }
    for track, findings in (("strict", strict), ("broader", broad_by_id)):
        metrics: dict[str, object] = {}
        for projection in ("exact", "five_family", "binary", "dominant"):
            reference = []
            comparison = []
            case_ids = []
            for identity in rows:
                case_id = identity.workbook_case_id
                reference.append(
                    projection_value(
                        benchmark_by_id[case_id],
                        projection,
                        include_uncertain=False,
                        dominant_priority=dominant_priority,
                    )
                )
                comparison.append(
                    projection_value(
                        findings[case_id],
                        projection,
                        include_uncertain=False,
                        dominant_priority=dominant_priority,
                    )
                )
                case_ids.append(case_id)
            result = compute_cohen_kappa(reference, comparison, case_ids=case_ids)
            payload = result.to_dict()
            expected = PLAN_REFERENCE_APPROXIMATIONS[track][projection]
            payload["plan_reference_approximation"] = expected
            payload["difference_from_plan_approximation"] = (
                result.kappa - expected if result.kappa is not None else None
            )
            metrics[projection] = payload
        output[track] = metrics
    return output


def write_baseline_artifacts(
    output_dir: str | Path, payload: Mapping[str, object]
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    json_path = write_json(destination / "baseline_reference.json", payload)
    lines = [
        "# CARDIA-X Baseline Reproduction",
        "",
            "The strict track uses only `primary_diagnosis`; the broader track uses all permitted unassisted evidence. Values are recomputed from the locked workbook and current versioned coders. Earlier documented approximations are retained beside the measured values, so any version drift is explicit rather than silently overwritten.",
        "",
        "| Track | Projection | n | Observed agreement | Kappa | Plan approximation | Difference |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for track in ("strict", "broader"):
        track_payload = payload[track]
        if not isinstance(track_payload, Mapping):
            continue
        for projection, raw in track_payload.items():
            if not isinstance(raw, Mapping):
                continue
            metric = dict(raw)
            lines.append(
                f"| {track} | {projection} | {metric['sample_size']} | "
                f"{float(str(metric['observed_agreement'])):.6f} | {float(str(metric['kappa'])):.6f} | "
                f"{float(str(metric['plan_reference_approximation'])):.3f} | "
                f"{float(str(metric['difference_from_plan_approximation'])):+.6f} |"
            )
    markdown_path = destination / "baseline_reference.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
