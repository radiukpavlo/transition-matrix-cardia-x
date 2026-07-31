from __future__ import annotations

import math

import pytest
from sklearn.metrics import cohen_kappa_score

from tm_ecg.clinical_validation.metrics import (
    compute_cohen_kappa,
    permutation_test_kappa,
)


def test_kappa_matches_sklearn_reference() -> None:
    left = ["a", "a", "b", "b", "c", "c", "a", "b"]
    right = ["a", "b", "b", "b", "c", "a", "a", "c"]
    result = compute_cohen_kappa(left, right)
    assert result.status == "ok"
    assert result.kappa == pytest.approx(cohen_kappa_score(left, right), abs=1e-12)
    assert result.observed_agreement == pytest.approx(5 / 8)
    assert result.observed_agreement_count == 5
    assert result.maximum_fixed_margin_agreement_count == 8
    assert set(result.threshold_attainability) == {"0.60", "0.70"}


@pytest.mark.parametrize(
    ("left", "right", "status"),
    [([], [], "empty"), (["a"], ["a"], "insufficient"), (["a", "a"], ["a", "a"], "not_estimable")],
)
def test_degenerate_inputs_do_not_report_false_perfection(left, right, status) -> None:
    result = compute_cohen_kappa(left, right)
    assert result.status == status
    assert result.kappa is None
    assert not math.isnan(result.observed_agreement) if result.observed_agreement is not None else True


def test_fixed_margin_diagnostic_does_not_offer_impossible_agreement_count() -> None:
    reference = ["normal"] * 90 + ["abnormal"] * 10
    comparison = ["normal"] * 100
    result = compute_cohen_kappa(reference, comparison, threshold=0.70)
    assert result.maximum_attainable_kappa == pytest.approx(0.0)
    assert result.fixed_margin_threshold_attainable is False
    assert result.approximate_additional_agreements_for_threshold is None
    assert (
        result.threshold_attainability["0.60"][
            "attainable_under_fixed_margins"
        ]
        is False
    )


def test_permutation_test_is_seed_deterministic() -> None:
    reference = ["a", "a", "a", "b", "b", "b", "c", "c"]
    comparison = ["a", "a", "a", "b", "b", "c", "c", "c"]
    first = permutation_test_kappa(reference, comparison, replicates=200, seed=3)
    second = permutation_test_kappa(reference, comparison, replicates=200, seed=3)
    assert first == second
    assert first[1] == 200
    assert first[0] is not None and 0.0 < first[0] <= 1.0
