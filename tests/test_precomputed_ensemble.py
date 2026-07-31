import json

import numpy as np
import pandas as pd
import pytest

from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.modeling.precomputed_ensemble import (
    _load_precomputed_view,
    _probability_column,
)


def _write_view(tmp_path, *, physical_access: bool) -> None:
    report = {
        "confirmatory_labels_opened": False,
        "label_order": PROJECT_LABELS,
        "estimators": ["logistic"],
        "design": {
            "label_access_audit": (
                {
                    "mode": (
                        "physical_parquet_row_filter_before_label_materialization"
                    ),
                    "confirmatory_label_rows_materialized": 0,
                }
                if physical_access
                else None
            )
        },
    }
    (tmp_path / "structured_compatibility_development_metrics.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    crossfit = tmp_path / "crossfit_logistic"
    crossfit.mkdir()
    np.savez_compressed(
        crossfit / "crossfit_probabilities.npz",
        probabilities=np.full((2, len(PROJECT_LABELS)), 0.25),
        fold_assignments=np.array([0, 1]),
    )
    row = {"record_id": "v1"}
    row.update({_probability_column(label): 0.25 for label in PROJECT_LABELS})
    pd.DataFrame([row]).to_parquet(
        tmp_path / "development_validation_predictions.parquet",
        index=False,
    )


def test_precomputed_view_requires_physical_label_restriction(tmp_path) -> None:
    _write_view(tmp_path, physical_access=False)

    with pytest.raises(RuntimeError, match="physically restricted"):
        _load_precomputed_view(
            tmp_path,
            training_rows=2,
            validation_record_ids=["v1"],
        )


def test_precomputed_view_loads_aligned_probabilities(tmp_path) -> None:
    _write_view(tmp_path, physical_access=True)

    oof, validation, audit = _load_precomputed_view(
        tmp_path,
        training_rows=2,
        validation_record_ids=["v1"],
    )

    assert oof.shape == (2, len(PROJECT_LABELS))
    assert validation.shape == (1, len(PROJECT_LABELS))
    assert audit["label_access_audit"]["confirmatory_label_rows_materialized"] == 0
