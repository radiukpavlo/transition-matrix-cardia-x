"""Transparent Cohen-kappa calculation with safe degenerate handling."""

from __future__ import annotations

from collections import Counter
from math import ceil, isclose
import random
from typing import Sequence

from tm_ecg.clinical_validation.models import KappaResult


def _class_metrics(
    labels: tuple[str, ...],
    matrix: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int | None]]:
    n = sum(sum(row.values()) for row in matrix.values())
    result: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        tp = matrix[label][label]
        fn = sum(matrix[label][other] for other in labels if other != label)
        fp = sum(matrix[other][label] for other in labels if other != label)
        tn = n - tp - fn - fp
        result[label] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "positive_agreement": (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "positive_predictive_value": tp / (tp + fp) if tp + fp else None,
            "negative_predictive_value": tn / (tn + fn) if tn + fn else None,
        }
    return result


def compute_cohen_kappa(
    reference: Sequence[str],
    comparison: Sequence[str],
    *,
    case_ids: Sequence[str] | None = None,
    threshold: float = 0.70,
    tolerance: float = 1e-12,
) -> KappaResult:
    if len(reference) != len(comparison):
        raise ValueError("label sequences must be aligned")
    if case_ids is not None and len(case_ids) != len(reference):
        raise ValueError("case_ids and labels must be aligned")
    n = len(reference)
    ids = tuple(case_ids or (str(index) for index in range(n)))
    labels = tuple(sorted(set(reference) | set(comparison)))
    matrix = {
        row: {column: 0 for column in labels}
        for row in labels
    }
    for left, right in zip(reference, comparison, strict=True):
        matrix[left][right] += 1
    ref_counts = Counter(reference)
    cmp_counts = Counter(comparison)
    ref_margins = {label: ref_counts.get(label, 0) for label in labels}
    cmp_margins = {label: cmp_counts.get(label, 0) for label in labels}
    if n == 0:
        return KappaResult(
            sample_size=0,
            labels=(),
            observed_agreement=None,
            expected_agreement=None,
            kappa=None,
            confidence_interval=(None, None),
            status="empty",
            reason="No aligned observations exist",
            confusion_matrix={},
            reference_margins={},
            comparison_margins={},
            case_ids=(),
        )
    observed_count = sum(matrix[label][label] for label in labels)
    observed = observed_count / n
    expected = sum((ref_counts[label] / n) * (cmp_counts[label] / n) for label in labels)
    if n < 2:
        status = "insufficient"
        reason = "At least two aligned observations are required"
        kappa = None
    elif isclose(1.0 - expected, 0.0, abs_tol=tolerance):
        status = "not_estimable"
        reason = "Expected agreement is one; the kappa denominator is zero"
        kappa = None
    else:
        status = "ok"
        reason = ""
        kappa = (observed - expected) / (1.0 - expected)

    max_observed = sum(min(ref_counts[label], cmp_counts[label]) for label in labels) / n
    max_observed_count = sum(
        min(ref_counts[label], cmp_counts[label]) for label in labels
    )
    max_kappa = (
        (max_observed - expected) / (1.0 - expected)
        if not isclose(1.0 - expected, 0.0, abs_tol=tolerance)
        else None
    )
    fixed_margin_attainable = (
        max_kappa is not None and max_kappa + tolerance >= threshold
    )
    if fixed_margin_attainable:
        target_observed = threshold * (1.0 - expected) + expected
        additional = max(0, ceil(target_observed * n - observed_count - tolerance))
    else:
        additional = None
    threshold_attainability: dict[
        str, dict[str, float | int | bool | None]
    ] = {}
    for requested_threshold in sorted({0.60, 0.70, float(threshold)}):
        threshold_observed: float | None = (
            requested_threshold * (1.0 - expected) + expected
            if not isclose(1.0 - expected, 0.0, abs_tol=tolerance)
            else None
        )
        required_count = (
            ceil(threshold_observed * n - tolerance)
            if threshold_observed is not None
            else None
        )
        attainable = (
            required_count is not None and required_count <= max_observed_count
        )
        threshold_attainability[f"{requested_threshold:.2f}"] = {
            "threshold": requested_threshold,
            "attainable_under_fixed_margins": attainable,
            "required_diagonal_agreement_count": required_count,
            "additional_diagonal_agreements_required": (
                max(0, required_count - observed_count)
                if attainable and required_count is not None
                else None
            ),
                "required_observed_agreement": threshold_observed,
        }
    prevalence_index = None
    bias_index = None
    if len(labels) == 2:
        first, second = labels
        prevalence_index = abs(matrix[first][first] - matrix[second][second]) / n
        bias_index = abs(matrix[first][second] - matrix[second][first]) / n
    return KappaResult(
        sample_size=n,
        labels=labels,
        observed_agreement=observed,
        expected_agreement=expected,
        kappa=kappa,
        confidence_interval=(None, None),
        status=status,
        reason=reason,
        confusion_matrix=matrix,
        reference_margins=ref_margins,
        comparison_margins=cmp_margins,
        case_ids=ids,
        class_metrics=_class_metrics(labels, matrix),
        prevalence_index=prevalence_index,
        bias_index=bias_index,
        maximum_attainable_kappa=max_kappa,
        fixed_margin_threshold_attainable=fixed_margin_attainable,
        approximate_additional_agreements_for_threshold=additional,
        observed_agreement_count=observed_count,
        maximum_fixed_margin_agreement_count=max_observed_count,
        threshold_attainability=threshold_attainability,
    )


def permutation_test_kappa(
    reference: Sequence[str],
    comparison: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = 19,
) -> tuple[float | None, int]:
    """One-sided deterministic permutation test for agreement beyond chance."""

    observed = compute_cohen_kappa(reference, comparison)
    if observed.kappa is None or replicates <= 0:
        return None, 0
    shuffled = list(comparison)
    rng = random.Random(seed)
    greater_or_equal = 0
    valid = 0
    for _ in range(replicates):
        rng.shuffle(shuffled)
        result = compute_cohen_kappa(reference, shuffled)
        if result.kappa is None:
            continue
        valid += 1
        greater_or_equal += int(result.kappa >= observed.kappa)
    if valid == 0:
        return None, 0
    return (greater_or_equal + 1) / (valid + 1), valid
