from __future__ import annotations

import json
from pathlib import Path

from tm_ecg.clinical_validation.cli import physician_pass
from tm_ecg.clinical_validation.models import ClinicalFindingSet


ROOT = Path(__file__).resolve().parents[2]


def test_v3_physician_pass_needs_no_benchmark_artifacts(tmp_path: Path) -> None:
    response_path = tmp_path / "unassisted_responses.jsonl"
    response_path.write_text(
        json.dumps(
            {
                "case_id": "P001",
                "primary_diagnosis": "Sinus rhythm",
                "rationale": "No evidence of atrial fibrillation",
                "diagnostic_confidence": 4,
                "ecg_quality": 4,
                "ambiguous": "No",
                "requires_additional_information": "No",
                "evidence": {"evidence_rr_irregularity": "Yes"},
                "source_cells": {
                    "primary_diagnosis": "Response_Export!O2",
                    "rationale": "Response_Export!AD2",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert not any("benchmark" in path.name.lower() for path in tmp_path.iterdir())

    outputs = physician_pass(
        responses=response_path,
        rules_path=(
            ROOT / "clinical_validation/config/physician_coding_rules_v3.yaml"
        ),
        output_dir=tmp_path / "output",
    )

    finding = ClinicalFindingSet.from_dict(
        json.loads(outputs["findings"].read_text(encoding="utf-8").splitlines()[0])
    )
    assert finding.rhythm == ("sinus",)
    assert finding.normality == "indeterminate"
    assert finding.projection_labels == ()
    assert not any("benchmark" in path.name.lower() for path in tmp_path.rglob("*"))


def test_v3_axis_derivation_round_trip_is_lossless() -> None:
    from tm_ecg.clinical_validation.field_policy import build_physician_view
    from tm_ecg.clinical_validation.physician_coder import (
        code_physician_response,
        load_physician_rules,
    )

    rules = load_physician_rules(
        ROOT / "clinical_validation/config/physician_coding_rules_v3.yaml"
    )
    findings = code_physician_response(
        build_physician_view(
            {
                "case_id": "P002",
                "primary_diagnosis": "",
                "rationale": "",
                "evidence_bundle_branch_pattern": "Yes",
            }
        ),
        rules,
    )
    restored = ClinicalFindingSet.from_dict(findings.to_dict())
    assert restored == findings
    assert restored.axis_derivations[0].derivation_status == "evidence-supported"

