from __future__ import annotations

import json

import pandas as pd
import pytest

from tm_ecg.modeling.semantic_targets import (
    build_oof_semantic_target_matrix,
)


def _contract(path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "contract_id": "test_semantics",
                "fit_policy": {"forbidden_inputs": ["scp_codes"]},
                "features": [
                    {
                        "feature_id": "heart_rate_bpm",
                        "group": "rate",
                        "units": "bpm",
                        "bounds": [20, 300],
                        "missingness": "missing",
                        "provenance": "waveform_rr_measurement",
                    },
                    {
                        "feature_id": "af_probability",
                        "group": "rhythm",
                        "units": "probability",
                        "bounds": [0, 1],
                        "missingness": "missing",
                        "provenance": "cross_fitted_atrial_specialist",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_semantic_matrix_requires_complete_oof_provenance(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    specialist = tmp_path / "specialist.parquet"
    contract = tmp_path / "contract.json"
    output = tmp_path / "B_semantic.parquet"
    _contract(contract)
    pd.DataFrame(
        {
            "record_id": ["1", "2"],
            "patient_id": ["p1", "p2"],
            "heart_rate_bpm": [70.0, 80.0],
        }
    ).to_parquet(source, index=False)
    pd.DataFrame(
        {
            "record_id": ["1", "2"],
            "patient_id": ["p1", "p2"],
            "prediction_mode": ["out_of_fold", "out_of_fold"],
            "crossfit_fold": [0, 1],
            "af_probability": [0.1, 0.9],
        }
    ).to_parquet(specialist, index=False)
    manifest = build_oof_semantic_target_matrix(
        source_measurements_path=source,
        specialist_oof_path=specialist,
        contract_path=contract,
        output_path=output,
    )
    assert manifest["all_specialist_predictions_out_of_fold"] is True
    assert manifest["rows"] == 2
    result = pd.read_parquet(output)
    assert result["af_probability"].tolist() == [0.1, 0.9]


def test_semantic_matrix_rejects_in_sample_specialist_rows(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    specialist = tmp_path / "specialist.parquet"
    contract = tmp_path / "contract.json"
    _contract(contract)
    pd.DataFrame(
        {
            "record_id": ["1"],
            "patient_id": ["p1"],
            "heart_rate_bpm": [70.0],
        }
    ).to_parquet(source, index=False)
    pd.DataFrame(
        {
            "record_id": ["1"],
            "patient_id": ["p1"],
            "prediction_mode": ["in_sample"],
            "crossfit_fold": [0],
            "af_probability": [0.1],
        }
    ).to_parquet(specialist, index=False)
    with pytest.raises(ValueError, match="In-sample"):
        build_oof_semantic_target_matrix(
            source_measurements_path=source,
            specialist_oof_path=specialist,
            contract_path=contract,
            output_path=tmp_path / "output.parquet",
        )
