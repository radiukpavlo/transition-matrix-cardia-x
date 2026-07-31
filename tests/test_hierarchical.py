import numpy as np

from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.modeling.hierarchical import (
    HierarchicalGate,
    crossfit_hierarchical_probabilities,
)

NORMAL_INDEX = PROJECT_LABELS.index("Normal")
RESIDUAL_INDEX = PROJECT_LABELS.index("Other / unmapped")


def test_hierarchical_gate_reconciles_normal_and_residual() -> None:
    rng = np.random.default_rng(5)
    probabilities = rng.uniform(0.05, 0.95, size=(90, 9))
    targets = np.zeros((90, 9), dtype=int)
    targets[:30, NORMAL_INDEX] = 1
    targets[30:60, RESIDUAL_INDEX] = 1
    targets[60:, PROJECT_LABELS.index("PVC")] = 1
    transformed = HierarchicalGate().fit(probabilities, targets).transform(
        probabilities
    )
    assert transformed.shape == probabilities.shape
    assert np.isfinite(transformed).all()
    assert ((transformed > 0) & (transformed < 1)).all()


def test_hierarchical_crossfit_covers_each_patient_group() -> None:
    rng = np.random.default_rng(8)
    probabilities = rng.uniform(0.05, 0.95, size=(120, 9))
    targets = np.zeros((120, 9), dtype=int)
    targets[:40, NORMAL_INDEX] = 1
    targets[40:80, RESIDUAL_INDEX] = 1
    targets[80:, PROJECT_LABELS.index("APB")] = 1
    groups = [f"p{index}" for index in range(120)]
    oof, validation, audit = crossfit_hierarchical_probabilities(
        probabilities,
        targets,
        groups,
        probabilities[:10],
        n_splits=4,
    )
    assert oof.shape == probabilities.shape
    assert validation.shape == (10, 9)
    assert audit["all_training_rows_transformed_out_of_fold"] is True
    assert sum(audit["fold_counts"].values()) == 120
