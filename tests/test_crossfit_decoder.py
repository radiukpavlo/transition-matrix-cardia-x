from __future__ import annotations

import numpy as np
import pytest

from tm_ecg.modeling.calibration import fit_best_binary_calibrator
from tm_ecg.modeling.crossfit import generate_crossfit_probabilities
from tm_ecg.modeling.label_contract import (
    DEFAULT_COMPATIBILITY_CONTRACT_V4,
    LabelContractError,
)
from tm_ecg.modeling.label_set_decoder import StructuredLabelSetDecoder


class _LeakageGuardEstimator:
    def __init__(self) -> None:
        self.training_groups: set[int] = set()
        self.classes_ = np.asarray([0, 1])

    def fit(self, features: object, targets: object) -> "_LeakageGuardEstimator":
        matrix = np.asarray(features)
        self.training_groups = set(matrix[:, 0].astype(int).tolist())
        return self

    def predict_proba(self, features: object) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        prediction_groups = set(matrix[:, 0].astype(int).tolist())
        assert self.training_groups.isdisjoint(prediction_groups)
        probability = 1.0 / (1.0 + np.exp(-matrix[:, 1]))
        return np.column_stack((1.0 - probability, probability))


def test_crossfit_is_patient_disjoint_complete_and_deterministic() -> None:
    groups = [f"p{index // 2:03d}" for index in range(200)]
    group_number = np.asarray([index // 2 for index in range(200)])
    signal = np.where(group_number % 2 == 0, 2.0, -2.0)
    features = np.column_stack((group_number, signal))
    targets = np.column_stack(
        (
            (group_number % 2 == 0).astype(int),
            (group_number % 5 == 0).astype(int),
        )
    )

    first = generate_crossfit_probabilities(
        features,
        targets,
        groups,
        ["even", "fifth"],
        lambda _label, _seed: _LeakageGuardEstimator(),
        n_splits=5,
        seed=31,
    )
    second = generate_crossfit_probabilities(
        features,
        targets,
        groups,
        ["even", "fifth"],
        lambda _label, _seed: _LeakageGuardEstimator(),
        n_splits=5,
        seed=31,
    )

    assert np.isfinite(first.probabilities).all()
    assert np.allclose(first.probabilities, second.probabilities)
    assert np.array_equal(first.fold_assignments, second.fold_assignments)
    assert first.metadata["probability_checksum"] == second.metadata[
        "probability_checksum"
    ]


def test_nested_calibration_selection_is_bounded_and_support_aware() -> None:
    probabilities = np.linspace(0.05, 0.95, 200)
    targets = (probabilities > 0.65).astype(int)

    calibrator = fit_best_binary_calibrator(probabilities, targets)
    calibrated = calibrator.predict(probabilities)

    assert calibrator.method in {"identity", "platt", "beta", "isotonic"}
    assert np.isfinite(calibrated).all()
    assert ((calibrated > 0) & (calibrated < 1)).all()
    assert calibrator.positive_support == int(targets.sum())


def test_structured_decoder_enforces_contract_and_decodes_nonempty_sets() -> None:
    contract = DEFAULT_COMPATIBILITY_CONTRACT_V4
    label_index = {label: index for index, label in enumerate(contract.label_order)}
    target_sets = [
        ("Normal",),
        ("Other / unmapped",),
        ("AF",),
        ("PVC",),
        ("RBBB spectrum", "AF"),
    ]
    targets = np.zeros((250, len(contract.label_order)), dtype=int)
    probabilities = np.full_like(targets, 0.08, dtype=float)
    for row in range(len(targets)):
        labels = target_sets[row % len(target_sets)]
        for label in labels:
            targets[row, label_index[label]] = 1
            probabilities[row, label_index[label]] = 0.78
        probabilities[row] += ((row % 7) - 3) * 0.01
    probabilities = np.clip(probabilities, 0.01, 0.99)

    decoder = StructuredLabelSetDecoder(
        regularization=0.5,
        pairwise_regularization=2.0,
    ).fit(probabilities, targets)
    predictions = decoder.predict(probabilities)

    contract.validate_prediction_matrix(predictions)
    assert predictions.any(axis=1).all()
    assert np.all(predictions == targets, axis=1).mean() > 0.95
    assert decoder.fit_metadata["candidate_count"] >= len(target_sets)


def test_structured_decoder_rejects_af_afl_without_source_override() -> None:
    contract = DEFAULT_COMPATIBILITY_CONTRACT_V4
    targets = np.zeros((2, len(contract.label_order)), dtype=int)
    targets[0, contract.label_order.index("AF")] = 1
    targets[0, contract.label_order.index("AFL")] = 1
    targets[1, contract.label_order.index("Normal")] = 1

    with pytest.raises(LabelContractError, match="Invalid structured target"):
        StructuredLabelSetDecoder().build_candidate_sets(targets)

    candidates = StructuredLabelSetDecoder(
        source_permits_af_afl=True
    ).build_candidate_sets(targets)
    assert ("AF", "AFL") in candidates

