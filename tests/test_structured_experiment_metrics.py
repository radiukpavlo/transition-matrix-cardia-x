import numpy as np
import pandas as pd

from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.modeling.structured_experiment import (
    _metrics,
    _read_physically_restricted_development_index,
)


def test_structured_metrics_report_exact_secondary_and_clustered_results() -> None:
    truth = np.zeros((4, len(PROJECT_LABELS)), dtype=int)
    prediction = np.zeros_like(truth)
    normal = PROJECT_LABELS.index("Normal")
    pvc = PROJECT_LABELS.index("PVC")
    residual = PROJECT_LABELS.index("Other / unmapped")
    truth[0, normal] = 1
    truth[1, pvc] = 1
    truth[2, residual] = 1
    truth[3, pvc] = 1
    prediction[:] = truth
    prediction[3] = 0
    prediction[3, residual] = 1

    metrics = _metrics(
        truth,
        prediction,
        groups=["p1", "p2", "p3", "p3"],
        bootstrap_seed=17,
    )

    assert metrics["exact_successes"] == 3
    assert metrics["compatibility_subset_exact_match"] == 0.75
    assert 0.0 <= metrics["weighted_f1"] <= 1.0
    assert 0.0 <= metrics["hamming_loss"] <= 1.0
    assert 0.0 <= metrics["sample_jaccard"] <= 1.0
    assert metrics["prediction_contract"]["invalid_sets"] == 0
    assert metrics["patient_cluster_bootstrap_95"]["patient_groups"] == 3


def test_structured_metrics_detect_contract_conflicts() -> None:
    truth = np.zeros((1, len(PROJECT_LABELS)), dtype=int)
    prediction = np.zeros_like(truth)
    normal = PROJECT_LABELS.index("Normal")
    pvc = PROJECT_LABELS.index("PVC")
    truth[0, normal] = 1
    prediction[0, normal] = 1
    prediction[0, pvc] = 1

    metrics = _metrics(truth, prediction)

    assert metrics["prediction_contract"]["invalid_sets"] == 1
    assert metrics["prediction_contract"]["normal_abnormal_conflicts"] == 1


def test_development_label_reader_physically_filters_sealed_rows(tmp_path) -> None:
    path = tmp_path / "index.parquet"
    pd.DataFrame(
        [
            {
                "record_id": "d1",
                "patient_id": "p1",
                "strat_fold": "1",
                "labels": "Normal",
            },
            {
                "record_id": "v1",
                "patient_id": "p2",
                "strat_fold": "2",
                "labels": "PVC",
            },
            {
                "record_id": "s1",
                "patient_id": "p3",
                "strat_fold": "3",
                "labels": "SEALED_SENTINEL",
            },
            {
                "record_id": "h1",
                "patient_id": "p4",
                "strat_fold": "10",
                "labels": "HISTORICAL_SENTINEL",
            },
        ]
    ).to_parquet(path, index=False)
    frame, audit = _read_physically_restricted_development_index(
        path,
        {
            "p1": "development_train",
            "p2": "development_validation",
            "p3": "sealed_internal_confirmatory",
        },
    )

    assert set(frame["record_id"]) == {"d1", "v1"}
    assert "SEALED_SENTINEL" not in set(frame["labels"])
    assert audit["confirmatory_label_rows_materialized"] == 0
    assert audit["historical_label_rows_materialized"] == 0
