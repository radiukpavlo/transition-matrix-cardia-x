from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tm_ecg.modeling.label_contract import (
    DEFAULT_COMPATIBILITY_CONTRACT_V4,
    LabelContractError,
    load_compatibility_label_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v4_contract_is_residual_and_normal_exclusive() -> None:
    contract = DEFAULT_COMPATIBILITY_CONTRACT_V4
    assert contract.normalize(["AF", "Other / unmapped"]) == ("AF",)
    assert contract.normalize(["Normal", "PVC"]) == ("PVC",)
    assert contract.normalize(["Normal", "PVC", "Other / unmapped"]) == ("PVC",)
    assert contract.normalize(["Other / unmapped"]) == ("Other / unmapped",)


def test_v4_training_targets_reject_empty_sets() -> None:
    with pytest.raises(LabelContractError, match="empty"):
        DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize([], empty_policy="error")
    assert DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
        [], empty_policy="residual"
    ) == ("Other / unmapped",)


def test_v4_contract_loads_with_frozen_label_order() -> None:
    contract = load_compatibility_label_contract(
        ROOT / "configs" / "compatibility_label_contract_v4.yaml"
    )
    assert contract.label_order == DEFAULT_COMPATIBILITY_CONTRACT_V4.label_order
    assert contract.source_sha256
    assert contract.normalize(["PAC", "Other abnormal"]) == ("APB",)


def test_prediction_matrix_rejects_residual_specific_overlap() -> None:
    contract = DEFAULT_COMPATIBILITY_CONTRACT_V4
    matrix = np.zeros((1, len(contract.label_order)), dtype=int)
    matrix[0, contract.label_order.index("AF")] = 1
    matrix[0, contract.label_order.index("Other / unmapped")] = 1
    with pytest.raises(LabelContractError, match="co-occurs"):
        contract.validate_prediction_matrix(matrix)
