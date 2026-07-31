from __future__ import annotations

import json
from pathlib import Path

from tm_ecg.cli import main
from tm_ecg.clinical_validation.cli import _classify_gate_failures


ROOT = Path(__file__).resolve().parents[2]


def test_run_all_cli_writes_audited_bundle(synthetic_validation_inputs, tmp_path) -> None:
    workbook, provenance = synthetic_validation_inputs
    results = tmp_path / "results"
    code = main(
        [
            "--config",
            str(ROOT / "configs/defaults.toml"),
            "clinical-validation",
            "run-all",
            "--workbook",
            str(workbook),
            "--provenance",
            str(provenance),
            "--results-root",
            str(results),
            "--run-id",
            "synthetic",
            "--bootstrap-replicates",
            "10",
        ]
    )
    assert code == 5
    bundle = results / "synthetic"
    manifest = json.loads((bundle / "run_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((bundle / "gate_result.json").read_text(encoding="utf-8"))
    assert gate["passed"]
    model_gate = json.loads((bundle / "cardia_x_gate_result.json").read_text(encoding="utf-8"))
    assert not model_gate["passed"]
    assert manifest["output_hashes"]
    assert (bundle / "disagreement_ledger.csv").exists()
    assert (bundle / "cardia_x_vs_benchmark.json").exists()
    assert (bundle / "abstention_concordance.json").exists()
    assert (bundle / "abstention_concordance_ledger.csv").exists()
    assert (bundle / "assisted_review_summary.json").exists()
    assert (bundle / "validation_report.md").exists()


def test_cli_returns_input_error_for_missing_workbook(tmp_path) -> None:
    code = main(
        [
            "--config",
            str(ROOT / "configs/defaults.toml"),
            "clinical-validation",
            "run-all",
            "--workbook",
            str(tmp_path / "missing.xlsx"),
            "--provenance",
            str(tmp_path / "missing.json"),
        ]
    )
    assert code == 2


def test_failure_classification_supports_v3_gate_groups() -> None:
    scenario_results = [
        {
            "scenario": {
                "scenario_id": "harmonized_b1_unique_exact",
                "physician_projection": "exact",
                "minimum_kappa": 0.7,
            },
            "result": {
                "status": "ok",
                "sample_size": 100,
                "observed_agreement": 0.29,
                "kappa": 0.12,
            },
        },
        {
            "scenario": {
                "scenario_id": "harmonized_b1_axis_rhythm_exact",
                "physician_projection": "axis_rhythm_exact",
                "minimum_kappa": 0.6,
            },
            "result": {
                "status": "ok",
                "sample_size": 100,
                "observed_agreement": 0.79,
                "kappa": 0.17,
            },
        },
    ]
    gate = {
        "harmonized_exact_below_threshold": ["harmonized_b1_unique_exact"],
        "other_required_below_threshold": ["harmonized_b1_axis_rhythm_exact"],
        "required_not_estimable": [],
    }

    result = _classify_gate_failures(scenario_results, gate)

    assert result["failed_required_scenario_count"] == 2
    assert {item["scenario_id"] for item in result["failed_required_scenarios"]} == {
        "harmonized_b1_unique_exact",
        "harmonized_b1_axis_rhythm_exact",
    }
