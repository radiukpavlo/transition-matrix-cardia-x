"""Independent CARDIA-X, abstention, and assisted-review validation tracks."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
import re
from typing import Mapping, Sequence

from tm_ecg.clinical_validation.bootstrap import cluster_bootstrap_kappa
from tm_ecg.clinical_validation.models import (
    BenchmarkFindingSet,
    CaseIdentity,
    ClinicalFindingSet,
    RawPhysicianResponse,
)
from tm_ecg.clinical_validation.ontology import projection_value, project_exact


def _yes(value: object) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _number(value: object) -> float | None:
    try:
        return float(str(value)) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float | None]:
    if n <= 0:
        return [None, None]
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    half = z * (
        (proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) ** 0.5
    ) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _cardia_x_finding(case_id: str, label: object) -> ClinicalFindingSet:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(label or "").lower()).strip()
    if normalized in {"normal", "sinus", "sinus rhythm"}:
        return ClinicalFindingSet(
            case_id=case_id,
            rhythm=("sinus",),
            normality="normal",
            interpretability="interpretable",
        )
    rhythm = {
        "af": ("af",),
        "atrial fibrillation": ("af",),
        "afl": ("afl",),
        "atrial flutter": ("afl",),
    }.get(normalized, ())
    ectopy = {
        "pvc": ("ventricular_premature",),
        "apb": ("atrial_premature",),
    }.get(normalized, ())
    conduction = {
        "rbbb": ("rbbb_spectrum",),
        "lbbb": ("lbbb_spectrum",),
    }.get(normalized, ())
    pacing = "present" if normalized == "paced" else "absent"
    if not (rhythm or ectopy or conduction or pacing == "present"):
        return ClinicalFindingSet(
            case_id=case_id,
            normality="abnormal",
            interpretability="interpretable",
            residual_abnormal=True,
        )
    return ClinicalFindingSet(
        case_id=case_id,
        rhythm=rhythm,
        ectopy=ectopy,
        conduction=conduction,
        pacing=pacing,
        normality="abnormal",
        interpretability="interpretable",
    )


def evaluate_cardia_x_track(
    identities: Sequence[CaseIdentity],
    cardia_x_rows: Sequence[Mapping[str, object]],
    benchmark: Sequence[BenchmarkFindingSet],
    dominant_priority: tuple[str, ...],
    *,
    bootstrap_replicates: int = 1000,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    """Evaluate the frozen displayed CARDIA-X output without physician fields.

    A structural-abstention route is an operational deferral even when the
    workbook also displays an internal candidate label.  The candidate is
    retained for audit but is never counted as an issued diagnosis.
    """

    row_by_id = {str(row.get("case_id")): row for row in cardia_x_rows}
    benchmark_by_id = {item.case_id: item for item in benchmark}
    cohort = [
        item for item in identities
        if item.dataset == "ptbxl" and not item.is_repeat
    ]
    if len(cohort) != 100:
        raise ValueError(f"CARDIA-X track expected 100 unique B1 cases, got {len(cohort)}")

    covered = [item for item in cohort if item.route != "structural_abstention"]
    coverage = len(covered) / len(cohort)
    projections = ("exact", "five_family", "binary", "dominant")
    results: dict[str, object] = {}
    distributions: dict[str, list[float]] = {}
    audit_rows: list[dict[str, object]] = []
    for identity in cohort:
        row = row_by_id.get(identity.workbook_case_id)
        truth = benchmark_by_id.get(identity.workbook_case_id)
        if row is None or truth is None:
            raise ValueError(f"Missing CARDIA-X or benchmark row for {identity.workbook_case_id}")
        audit_rows.append(
            {
                "case_id": identity.workbook_case_id,
                "route": identity.route,
                "operational_status": (
                    "abstained" if identity.route == "structural_abstention" else "diagnosed"
                ),
                "displayed_internal_candidate": str(
                    row.get("primary_label_or_abstention", "")
                ),
                "uncertainty_flags": str(row.get("uncertainty_flags", "")),
                "activated_rule_ids": str(row.get("activated_rule_ids", "")),
                "benchmark_exact": project_exact(truth),
            }
        )

    for projection in projections:
        covered_reference: list[str] = []
        covered_prediction: list[str] = []
        covered_ids: list[str] = []
        all_reference: list[str] = []
        all_prediction: list[str] = []
        all_ids: list[str] = []
        for identity in cohort:
            case_id = identity.workbook_case_id
            row = row_by_id[case_id]
            truth = benchmark_by_id[case_id]
            truth_label = projection_value(
                truth,
                projection,
                include_uncertain=False,
                dominant_priority=dominant_priority,
            )
            all_reference.append(truth_label)
            all_ids.append(case_id)
            if identity.route == "structural_abstention":
                all_prediction.append("Abstain")
                continue
            prediction = _cardia_x_finding(
                case_id, row.get("primary_label_or_abstention")
            )
            predicted_label = projection_value(
                prediction,
                projection,
                include_uncertain=False,
                dominant_priority=dominant_priority,
            )
            all_prediction.append(predicted_label)
            covered_reference.append(truth_label)
            covered_prediction.append(predicted_label)
            covered_ids.append(case_id)

        covered_result, covered_distribution = cluster_bootstrap_kappa(
            covered_reference,
            covered_prediction,
            covered_ids,
            covered_ids,
            replicates=bootstrap_replicates,
            seed=2800 + projections.index(projection),
        )
        all_result, all_distribution = cluster_bootstrap_kappa(
            all_reference,
            all_prediction,
            all_ids,
            all_ids,
            replicates=bootstrap_replicates,
            seed=2900 + projections.index(projection),
        )
        results[projection] = {
            "covered_cases": covered_result.to_dict(),
            "intention_to_diagnose": all_result.to_dict(),
        }
        distributions[f"covered_{projection}"] = covered_distribution
        distributions[f"intention_to_diagnose_{projection}"] = all_distribution

    route_counts = Counter(item.route for item in cohort)
    required_model_scenarios: dict[str, dict[str, object]] = {}
    for projection in projections:
        projection_result = results[projection]
        projection_mapping = (
            projection_result if isinstance(projection_result, Mapping) else {}
        )
        all_case_result = projection_mapping.get("intention_to_diagnose", {})
        required_model_scenarios[projection] = (
            dict(all_case_result) if isinstance(all_case_result, Mapping) else {}
        )
    below_threshold = [
        projection
        for projection, result in required_model_scenarios.items()
        if result.get("kappa") is not None and float(str(result["kappa"])) < 0.70
    ]
    not_estimable = [
        projection
        for projection, result in required_model_scenarios.items()
        if result.get("kappa") is None
    ]
    return (
        {
            "track": "frozen_cardia_x_output_vs_independent_benchmark",
            "population": "100 unique B1/PTB-XL cases",
            "n": len(cohort),
            "diagnosed_n": len(covered),
            "abstained_n": len(cohort) - len(covered),
            "coverage": coverage,
            "coverage_wilson_ci_95": _wilson_interval(len(covered), len(cohort)),
            "route_counts": dict(sorted(route_counts.items())),
            "operational_abstention_rule": "route == structural_abstention",
            "candidate_label_policy": (
                "The displayed internal candidate is audit-only for structural-abstention "
                "cases and is not counted as an issued diagnosis."
            ),
            "results": results,
            "acceptance": {
                "passed": not below_threshold and not not_estimable,
                "minimum_point_kappa": 0.70,
                "required_population": "intention_to_diagnose_with_abstention_explicit",
                "required_projections": list(projections),
                "below_threshold": below_threshold,
                "not_estimable": not_estimable,
            },
            "case_audit": audit_rows,
            "clinical_gate_effect": "diagnostic_only_not_used_to_modify_physician_gate",
        },
        distributions,
    )


def compute_abstention_concordance(
    identities: Sequence[CaseIdentity],
    raw_responses: Sequence[RawPhysicianResponse],
    physician: Sequence[ClinicalFindingSet],
    benchmark: Sequence[BenchmarkFindingSet],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Audit the predeclared low-confidence/noise/disagreement concordance endpoint."""

    raw_by_id = {item.case_id: item for item in raw_responses}
    physician_by_id = {item.case_id: item for item in physician}
    benchmark_by_id = {item.case_id: item for item in benchmark}
    cohort = [
        item for item in identities
        if item.dataset == "ptbxl"
        and not item.is_repeat
        and item.route == "structural_abstention"
    ]
    if len(cohort) != 40:
        raise ValueError(f"Abstention audit expected 40 unique cases, got {len(cohort)}")

    ledger: list[dict[str, object]] = []
    for identity in cohort:
        case_id = identity.workbook_case_id
        raw = raw_by_id[case_id]
        physician_finding = physician_by_id[case_id]
        benchmark_finding = benchmark_by_id[case_id]
        evidence = dict(raw.evidence)
        free_text = f"{raw.primary_diagnosis} {raw.rationale}".lower()
        low_confidence = raw.diagnostic_confidence is not None and raw.diagnostic_confidence <= 2
        noisy = (
            _yes(evidence.get("evidence_lead_noise"))
            or (raw.ecg_quality is not None and raw.ecg_quality <= 2)
            or bool(re.search(r"\b(nois|artifact|artefact|baseline wander)", free_text))
        )
        disagrees = project_exact(physician_finding) != project_exact(benchmark_finding)
        concordant = low_confidence or noisy or disagrees
        ledger.append(
            {
                "case_id": case_id,
                "low_confidence_le_2": low_confidence,
                "physician_marked_noisy": noisy,
                "physician_benchmark_exact_disagreement": disagrees,
                "physician_marked_ambiguous": _yes(raw.ambiguous),
                "additional_information_requested": _yes(
                    raw.requires_additional_information
                ),
                "concordant_by_preregistered_composite": concordant,
                "physician_exact": project_exact(physician_finding),
                "benchmark_exact": project_exact(benchmark_finding),
            }
        )

    def endpoint(field: str) -> dict[str, object]:
        count = sum(bool(row[field]) for row in ledger)
        return {
            "count": count,
            "n": len(ledger),
            "proportion": count / len(ledger),
            "wilson_ci_95": _wilson_interval(count, len(ledger)),
        }

    return (
        {
            "population": "40 unique B1 structural-abstention cases",
            "composite_definition": (
                "diagnostic confidence <=2 OR physician marked noise/poor quality/artifact "
                "OR physician exact finding set disagreed with the independent benchmark"
            ),
            "concordance_of_ambiguity": endpoint(
                "concordant_by_preregistered_composite"
            ),
            "components": {
                "low_confidence_le_2": endpoint("low_confidence_le_2"),
                "physician_marked_noisy": endpoint("physician_marked_noisy"),
                "physician_benchmark_exact_disagreement": endpoint(
                    "physician_benchmark_exact_disagreement"
                ),
                "physician_marked_ambiguous": endpoint(
                    "physician_marked_ambiguous"
                ),
                "additional_information_requested": endpoint(
                    "additional_information_requested"
                ),
            },
            "interpretation": (
                "Descriptive internal concordance endpoint; it does not prove prospective "
                "safety, improved outcomes, or external generalization."
            ),
        },
        ledger,
    )


def summarize_assisted_review(
    identities: Sequence[CaseIdentity],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize post-assistance ratings without leaking them into physician coding."""

    row_by_id = {str(row.get("case_id")): row for row in rows}
    cohort = [
        item for item in identities
        if item.dataset == "ptbxl" and not item.is_repeat
    ]
    score_fields = ("utility", "soundness", "safety", "clarity")

    def score_summary(selected: Sequence[CaseIdentity]) -> dict[str, object]:
        summary: dict[str, object] = {"n": len(selected)}
        for field in score_fields:
            values = [
                value
                for item in selected
                if (value := _number(row_by_id[item.workbook_case_id].get(field))) is not None
            ]
            summary[field] = {
                "n": len(values),
                "mean": mean(values) if values else None,
                "median": median(values) if values else None,
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
            }
        changed = sum(
            _yes(row_by_id[item.workbook_case_id].get("diagnosis_changed"))
            for item in selected
        )
        summary["diagnosis_changed"] = {
            "count": changed,
            "proportion": changed / len(selected) if selected else None,
        }
        return summary

    routes = sorted(set(item.route for item in cohort))
    route_summaries = {
        route: score_summary([item for item in cohort if item.route == route])
        for route in routes
    }

    def binary_rating(route: str, field: str) -> dict[str, object]:
        values = [
            str(row_by_id[item.workbook_case_id].get(field, "")).strip().lower()
            for item in cohort
            if item.route == route
        ]
        usable = [value for value in values if value in {"yes", "no"}]
        positive = sum(value == "yes" for value in usable)
        return {
            "yes": positive,
            "no": len(usable) - positive,
            "n": len(usable),
            "proportion_yes": positive / len(usable) if usable else None,
            "wilson_ci_95": _wilson_interval(positive, len(usable)),
        }

    return {
        "population": "100 unique B1 cases, post-assistance fields only",
        "separation_policy": (
            "These ratings were excluded from unassisted physician coding and agreement."
        ),
        "overall": score_summary(cohort),
        "by_route": route_summaries,
        "structural_abstention_appropriate": binary_rating(
            "structural_abstention", "abstention_appropriate"
        ),
        "conflict_region_plausible": binary_rating(
            "conflict_region", "conflict_plausible"
        ),
    }
