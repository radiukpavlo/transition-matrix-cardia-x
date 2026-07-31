"""Validate the aggregate results reported for CARDIA-X."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import isclose, sqrt
from pathlib import Path
from typing import Any


Z_95 = 1.959963984540054


def _close(name: str, actual: float, expected: float, tolerance: float = 2e-6) -> None:
    if not isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{name}: calculated {actual:.12g}, stored {expected:.12g}")


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    proportion = successes / total
    denominator = 1.0 + Z_95 * Z_95 / total
    centre = (proportion + Z_95 * Z_95 / (2.0 * total)) / denominator
    half = (
        Z_95
        * sqrt((proportion * (1.0 - proportion) + Z_95 * Z_95 / (4.0 * total)) / total)
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == total else min(1.0, centre + half)
    return lower, upper


def _check_binomial(name: str, payload: dict[str, Any]) -> None:
    successes = int(payload["count"])
    total = int(payload["denominator"])
    _close(f"{name}.estimate", successes / total, payload["estimate"])
    if payload.get("ci_method") == "wilson_score_95":
        lower, upper = _wilson(successes, total)
        _close(f"{name}.ci_lower", lower, payload["ci"][0])
        _close(f"{name}.ci_upper", upper, payload["ci"][1])


def _verify_reader_audit(results: dict[str, Any]) -> int:
    reader = results["reader_audit"]
    agreement = reader["historical_formula_linked_agreement"]
    reference: list[str] = []
    physician: list[str] = []
    for cell in agreement["contingency_counts"]:
        count = int(cell["count"])
        reference.extend([str(cell["benchmark"])] * count)
        physician.extend([str(cell["physician"])] * count)

    total = len(reference)
    observed = sum(left == right for left, right in zip(reference, physician, strict=True)) / total
    reference_counts = Counter(reference)
    physician_counts = Counter(physician)
    labels = set(reference_counts) | set(physician_counts)
    expected = sum(reference_counts[label] * physician_counts[label] for label in labels) / total**2
    kappa = (observed - expected) / (1.0 - expected)
    maximum_observed = sum(min(reference_counts[label], physician_counts[label]) for label in labels) / total
    maximum_kappa = (maximum_observed - expected) / (1.0 - expected)

    _close("reader.observed_agreement", observed, agreement["observed_agreement"])
    _close("reader.expected_agreement", expected, agreement["expected_agreement"])
    _close("reader.kappa", kappa, agreement["kappa"])
    _close("reader.maximum_kappa", maximum_kappa, agreement["maximum_kappa"])
    if not (agreement["kappa_ci"][0] <= agreement["kappa"] <= agreement["kappa_ci"][1]):
        raise AssertionError("reader.kappa_ci does not contain the point estimate")

    for endpoint in reader["assisted_binary_endpoints"]:
        _check_binomial(f"reader.{endpoint['endpoint']}", endpoint)
    for endpoint in reader["abstention_risk_markers"]:
        _check_binomial(f"reader.{endpoint['endpoint']}", endpoint)
    for endpoint in reader["b2_morphology_and_delineation"]:
        _check_binomial(f"reader.{endpoint['endpoint']}", endpoint)

    for row in reader["assisted_rating_summary"]:
        score_counts = row["score_counts"]
        count = sum(int(score_counts[str(score)]) for score in range(1, 6))
        weighted = sum(score * int(score_counts[str(score)]) for score in range(1, 6))
        if count != int(row["n"]):
            raise AssertionError(f"rating count mismatch for {row['route']} / {row['domain']}")
        _close(f"rating.{row['route']}.{row['domain']}", weighted / count, row["mean"])

    rule_rows = reader["rule_review"]["artifacts"]
    all_scores = [
        float(row[domain])
        for row in rule_rows
        for domain in ("utility", "soundness", "safety", "clarity")
    ]
    _close("rule_review.overall_mean", sum(all_scores) / len(all_scores), reader["rule_review"]["overall_mean"])
    soundness_failures = sum(float(row["soundness"]) == 3.0 for row in rule_rows)
    if soundness_failures != int(reader["rule_review"]["soundness_score_3_count"]):
        raise AssertionError("rule-review soundness failure count mismatch")
    return total


def _verify_compatibility(results: dict[str, Any]) -> int:
    compatibility = results["computational"]["compatibility"]
    global_metrics = compatibility["global"]
    records = int(compatibility["test_records"])
    _close(
        "compatibility.exact_subset_accuracy",
        global_metrics["exact_matches"] / records,
        global_metrics["exact_subset_accuracy"],
    )

    true_positive = false_positive = false_negative = 0
    for label, row in compatibility["per_class"].items():
        tp = int(row["tp"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        true_positive += tp
        false_positive += fp
        false_negative += fn
        _close(f"{label}.precision", tp / (tp + fp), row["precision"])
        _close(f"{label}.recall", tp / (tp + fn), row["recall"])
        _close(f"{label}.f1", 2 * tp / (2 * tp + fp + fn), row["f1"])
        if not (row["f1_ci"][0] <= row["f1"] <= row["f1_ci"][1]):
            raise AssertionError(f"{label}.f1_ci does not contain the point estimate")

    micro_f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    _close("compatibility.micro_f1", micro_f1, global_metrics["micro_f1"])
    bitwise_opportunities = int(global_metrics["bitwise_opportunities"])
    bitwise_accuracy = 1.0 - (false_positive + false_negative) / bitwise_opportunities
    _close("compatibility.bitwise_accuracy", bitwise_accuracy, global_metrics["bitwise_accuracy"])
    return records


def _verify_transition_and_quality(results: dict[str, Any]) -> int:
    computational = results["computational"]
    for row in computational["semantic_transition"]:
        _close(
            f"transition.{row['branch']}.{row['split']}.coverage",
            int(row["observed_targets"]) / int(row["total_targets"]),
            row["coverage"],
        )
    for row in computational["signal_eligibility"]:
        _check_binomial(
            f"signal_eligibility.{row['branch']}.{row['split']}.{row['condition']}",
            row,
        )
    return len(computational["semantic_transition"])


def _verify_ludb(results: dict[str, Any]) -> int:
    detection = results["computational"]["ludb"]["r_peak_detection"]
    true_positive = int(detection["tp"])
    false_positive = int(detection["fp"])
    false_negative = int(detection["fn"])
    sensitivity = true_positive / (true_positive + false_negative)
    ppv = true_positive / (true_positive + false_positive)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    _close("ludb.sensitivity", sensitivity, detection["sensitivity"])
    _close("ludb.ppv", ppv, detection["ppv"])
    _close("ludb.f1", f1, detection["f1"])
    if not (detection["f1_ci"][0] <= detection["f1"] <= detection["f1_ci"][1]):
        raise AssertionError("ludb.f1_ci does not contain the point estimate")
    return len(results["computational"]["ludb"]["landmarks"])


def verify(payload: dict[str, Any]) -> dict[str, int]:
    """Recalculate deterministic quantities and return a compact verification summary."""
    if payload.get("schema_version") != "cardia_x_reported_results_v1":
        raise AssertionError("unsupported reported-result schema")
    if payload.get("research_only") is not True:
        raise AssertionError("research-only status must remain explicit")
    return {
        "reader_cases": _verify_reader_audit(payload),
        "compatibility_test_records": _verify_compatibility(payload),
        "semantic_transition_rows": _verify_transition_and_quality(payload),
        "ludb_landmarks": _verify_ludb(payload),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/reported_metrics.json"),
        help="Path to the curated reported-result snapshot.",
    )
    parser.add_argument("--json", action="store_true", help="Print the verification summary as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    summary = verify(payload)
    if args.json:
        print(json.dumps({"status": "verified", **summary}, indent=2, sort_keys=True))
    else:
        print(
            "Verified CARDIA-X reported results: "
            f"{summary['compatibility_test_records']:,} PTB-XL test records, "
            f"{summary['reader_cases']} reader-audit cases, "
            f"{summary['semantic_transition_rows']} transition estimates, and "
            f"{summary['ludb_landmarks']} LUDB landmarks."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
