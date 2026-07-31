from __future__ import annotations

import pytest
from types import SimpleNamespace

from tm_ecg.stages.report import (
    _cluster_bootstrap_mean_ci,
    _record_error_clusters,
    _table,
)


def test_transition_report_prefers_current_parquet_artifact(tmp_path) -> None:
    (tmp_path / "B1_fit_val.csv").write_text("legacy", encoding="utf-8")
    (tmp_path / "B1_fit_val.parquet").write_bytes(b"current")
    config = SimpleNamespace(paths=SimpleNamespace(features=tmp_path))

    assert _table(config, "features", "B1_fit_val").suffix == ".parquet"


def test_transition_mae_resamples_records_and_preserves_missing_targets() -> None:
    truth = [
        {"record_id": "r1", "a": 1.0, "b": 2.0},
        {"record_id": "r2", "a": 3.0, "b": None},
    ]
    predicted = [
        {"record_id": "r2", "a": 4.0, "b": 9.0},
        {"record_id": "r1", "a": 2.0, "b": 4.0},
    ]

    clusters, opportunities = _record_error_clusters(truth, predicted)
    first = _cluster_bootstrap_mean_ci(clusters, replicates=50, seed=17)
    second = _cluster_bootstrap_mean_ci(clusters, replicates=50, seed=17)

    assert opportunities == 4
    assert clusters == {"r1": [1.0, 2.0], "r2": [1.0]}
    assert first == second
    assert first[0] == pytest.approx(4.0 / 3.0)


def test_transition_reporting_rejects_undocumented_record_mismatch() -> None:
    with pytest.raises(RuntimeError, match="documented alignment"):
        _record_error_clusters(
            [{"record_id": "r1", "a": 1.0}],
            [{"record_id": "r2", "a": 1.0}],
        )


def test_transition_reporting_accepts_only_documented_latent_exclusion() -> None:
    clusters, opportunities = _record_error_clusters(
        [
            {"record_id": "r1", "a": 1.0},
            {"record_id": "r2", "a": 2.0},
        ],
        [{"record_id": "r1", "a": 1.5}],
        allowed_missing_prediction_ids={"r2"},
    )

    assert clusters == {"r1": [0.5]}
    assert opportunities == 1
