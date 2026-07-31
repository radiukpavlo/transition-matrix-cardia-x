"""Weighted Entropy-Density Discretization (WEDD) for clinician features.

WEDD extends entropy-based supervised discretization by preferring class-separating
thresholds that lie in low-density regions of the observed feature distribution.  It
therefore balances decision purity and natural gaps in matrix B.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
import math
import random
import statistics
from typing import Iterable, Mapping

from tm_ecg.dss.models import AttributeDomain, DiscretizationPlan, IntervalBin, ThresholdRecord
from tm_ecg.features.registry import FEATURE_SPECS


MISSING_STATE = "missing"


CLINICAL_PRIORITIES: dict[str, float] = {
    "qrs_dur_med_ms": 2.2,
    "qrs_wide_any": 2.2,
    "r_prime_v1_any": 2.0,
    "broad_r_v6_any": 2.0,
    "p_present_ratio": 2.0,
    "af_irregularity_cv": 2.0,
    "pvc_like_beat_count": 1.9,
    "apb_like_beat_count": 1.6,
    "paced_like_beat_count": 2.1,
    "prematurity_index_min": 1.8,
    "comp_pause_ratio_max": 1.8,
    "rr_sdnn_ms": 1.6,
    "rr_iqr_ms": 1.6,
    "pr_med_ms": 1.4,
    "qt_med_ms": 1.3,
    "qtc_med_ms": 1.5,
    "st_level_v1_mV": 1.4,
    "st_level_v5_mV": 1.4,
    "t_inverted_right_any": 1.3,
    "delineation_confidence": 1.6,
    "lead_quality_min_db": 1.8,
}


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _entropy(labels: Iterable[str]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    result = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        result -= p * math.log2(p)
    return result


def conditional_entropy(values: list[float], labels: list[str], threshold: float) -> float:
    left_labels = [label for value, label in zip(values, labels, strict=False) if value <= threshold]
    right_labels = [label for value, label in zip(values, labels, strict=False) if value > threshold]
    total = len(values)
    if not total:
        return 0.0
    return (len(left_labels) / total) * _entropy(left_labels) + (len(right_labels) / total) * _entropy(right_labels)


def candidate_thresholds(values: list[float]) -> list[float]:
    ordered = sorted(set(values))
    return [(left + right) / 2.0 for left, right in zip(ordered, ordered[1:], strict=False) if left != right]


def _robust_bandwidth(values_sorted: list[float]) -> float:
    if len(values_sorted) <= 1:
        return 1.0
    q25 = values_sorted[int(0.25 * (len(values_sorted) - 1))]
    q75 = values_sorted[int(0.75 * (len(values_sorted) - 1))]
    iqr = max(q75 - q25, 0.0)
    mean = sum(values_sorted) / len(values_sorted)
    variance = sum((value - mean) ** 2 for value in values_sorted) / max(len(values_sorted) - 1, 1)
    sigma = math.sqrt(max(variance, 0.0))
    scale = sigma if iqr <= 0 else min(sigma, iqr / 1.349) if sigma > 0 else iqr / 1.349
    if scale <= 0:
        nonzero_gaps = [abs(b - a) for a, b in zip(values_sorted, values_sorted[1:], strict=False) if b != a]
        return max(min(nonzero_gaps), 1e-6) if nonzero_gaps else 1.0
    return max(0.9 * scale * (len(values_sorted) ** (-1 / 5)), 1e-6)


def window_density_at(values_sorted: list[float], threshold: float, bandwidth: float | None = None) -> float:
    """Robust uniform-kernel density estimate at a candidate threshold."""

    if not values_sorted:
        return 0.0
    h = bandwidth if bandwidth is not None else _robust_bandwidth(values_sorted)
    if h <= 0:
        h = 1.0
    left = bisect_left(values_sorted, threshold - h)
    right = bisect_right(values_sorted, threshold + h)
    return (right - left) / (len(values_sorted) * 2.0 * h)


def wedd_candidate_scores(
    values: list[float],
    labels: list[str],
    alpha: float = 0.65,
    min_leaf: int = 3,
) -> list[ThresholdRecord]:
    """Return WEDD scores for all candidate thresholds in one feature segment."""

    if len(values) != len(labels):
        raise ValueError("values and labels must be aligned")
    candidates = candidate_thresholds(values)
    if not candidates:
        return []
    sorted_values = sorted(values)
    bandwidth = _robust_bandwidth(sorted_values)
    raw: list[tuple[float, int, int, float, float]] = []
    for threshold in candidates:
        left_size = sum(1 for value in values if value <= threshold)
        right_size = len(values) - left_size
        if left_size < min_leaf or right_size < min_leaf:
            continue
        ent = conditional_entropy(values, labels, threshold)
        den = window_density_at(sorted_values, threshold, bandwidth)
        raw.append((threshold, left_size, right_size, ent, den))
    if not raw:
        return []
    ent_values = [item[3] for item in raw]
    den_values = [item[4] for item in raw]
    ent_min, ent_max = min(ent_values), max(ent_values)
    den_min, den_max = min(den_values), max(den_values)
    records = []
    for threshold, left_size, right_size, ent, den in raw:
        ent_norm = 0.0 if ent_max == ent_min else (ent - ent_min) / (ent_max - ent_min)
        den_norm = 0.0 if den_max == den_min else (den - den_min) / (den_max - den_min)
        objective = alpha * ent_norm + (1.0 - alpha) * den_norm
        records.append(
            ThresholdRecord(
                threshold_id="candidate",
                feature="",
                value=threshold,
                left_size=left_size,
                right_size=right_size,
                entropy=ent,
                density=den,
                objective=objective,
                source="WEDD",
                alpha=alpha,
                accepted=False,
            )
        )
    records.sort(key=lambda item: (item.objective, item.entropy, item.density, item.value))
    return records


def threshold_perturbation_flip_rate(
    values: list[float],
    threshold: float,
    tolerance: float,
) -> float:
    """Upper-bound the state-flip rate under a benign bounded perturbation."""

    if not values:
        return 0.0
    width = max(float(tolerance), 0.0)
    return sum(abs(value - threshold) <= width for value in values) / len(values)


def bootstrap_threshold_stability(
    values: list[float],
    labels: list[str],
    patient_ids: list[str],
    *,
    alpha: float = 0.65,
    min_leaf: int = 3,
    n_bootstrap: int = 100,
    seed: int = 19,
) -> list[float]:
    """Fit the best one-split WEDD threshold in patient-cluster bootstraps."""

    if not (len(values) == len(labels) == len(patient_ids)):
        raise ValueError("values, labels, and patient_ids must be aligned")
    if n_bootstrap <= 0 or not values:
        return []
    by_patient: dict[str, list[int]] = {}
    for index, patient_id in enumerate(patient_ids):
        by_patient.setdefault(str(patient_id), []).append(index)
    patients = sorted(by_patient)
    if not patients:
        return []
    rng = random.Random(seed)
    fitted: list[float] = []
    for _ in range(n_bootstrap):
        sampled = [rng.choice(patients) for _ in patients]
        indices = [index for patient_id in sampled for index in by_patient[patient_id]]
        sample_values = [values[index] for index in indices]
        sample_labels = [labels[index] for index in indices]
        candidates = wedd_candidate_scores(
            sample_values,
            sample_labels,
            alpha=alpha,
            min_leaf=min_leaf,
        )
        if candidates:
            fitted.append(float(candidates[0].value))
    return fitted


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _annotate_threshold_stability(
    records: list[ThresholdRecord],
    *,
    values: list[float],
    labels: list[str],
    patient_ids: list[str] | None,
    alpha: float,
    min_support: int,
    n_bootstrap: int,
    stability_tolerance: float,
    min_selection_frequency: float,
    perturbation_tolerance: float,
    max_perturbation_flip_rate: float,
    seed: int,
) -> list[ThresholdRecord]:
    bootstrapped = (
        bootstrap_threshold_stability(
            values,
            labels,
            patient_ids,
            alpha=alpha,
            min_leaf=min_support,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if patient_ids is not None and n_bootstrap > 0
        else []
    )
    retained: list[ThresholdRecord] = []
    for record in sorted(records, key=lambda item: item.value):
        record.perturbation_flip_rate = threshold_perturbation_flip_rate(
            values,
            record.value,
            perturbation_tolerance,
        )
        if bootstrapped:
            near = [
                value
                for value in bootstrapped
                if abs(value - record.value) <= max(stability_tolerance, 0.0)
            ]
            record.selection_frequency = len(near) / n_bootstrap
            record.bootstrap_median = statistics.median(bootstrapped)
            record.bootstrap_iqr = _quantile(bootstrapped, 0.75) - _quantile(
                bootstrapped, 0.25
            )
        unstable = (
            record.selection_frequency is not None
            and record.selection_frequency < min_selection_frequency
        )
        fragile = (
            record.perturbation_flip_rate is not None
            and record.perturbation_flip_rate > max_perturbation_flip_rate
        )
        if unstable or fragile:
            record.accepted = False
            reasons = []
            if unstable:
                reasons.append("bootstrap_unstable")
            if fragile:
                reasons.append("excessive_perturbation_flip_rate")
            record.notes = f"{record.notes}; rejected={'|'.join(reasons)}".strip("; ")
            continue
        retained.append(record)
    return retained


def _recursive_wedd(
    feature: str,
    values: list[float],
    labels: list[str],
    alpha: float,
    min_leaf: int,
    max_depth: int,
    min_gain: float,
    depth: int = 0,
) -> list[ThresholdRecord]:
    if len(values) < 2 * min_leaf or depth >= max_depth or len(set(labels)) <= 1:
        return []
    parent_entropy = _entropy(labels)
    scored = wedd_candidate_scores(values, labels, alpha=alpha, min_leaf=min_leaf)
    if not scored:
        return []
    best = scored[0]
    gain = parent_entropy - best.entropy
    if gain < min_gain:
        return []
    best.feature = feature
    best.threshold_id = f"{feature}:wedd:{depth}:{best.value:.8g}"
    best.accepted = True
    best.depth = depth
    best.notes = f"information_gain={gain:.6f}; parent_entropy={parent_entropy:.6f}"
    left_values: list[float] = []
    left_labels: list[str] = []
    right_values: list[float] = []
    right_labels: list[str] = []
    for value, label in zip(values, labels, strict=False):
        if value <= best.value:
            left_values.append(value)
            left_labels.append(label)
        else:
            right_values.append(value)
            right_labels.append(label)
    return (
        _recursive_wedd(feature, left_values, left_labels, alpha, min_leaf, max_depth, min_gain, depth + 1)
        + [best]
        + _recursive_wedd(feature, right_values, right_labels, alpha, min_leaf, max_depth, min_gain, depth + 1)
    )


def _bins_from_thresholds(feature: str, thresholds: list[ThresholdRecord], value_type: str, provenance: str) -> list[IntervalBin]:
    ordered = sorted(thresholds, key=lambda item: item.value)
    if not ordered:
        return [IntervalBin(0, "observed", None, None, True, True, provenance)]
    bins: list[IntervalBin] = []
    labels = _generic_state_labels(len(ordered) + 1, feature)
    lower = None
    for idx, label in enumerate(labels):
        upper = ordered[idx].value if idx < len(ordered) else None
        bins.append(
            IntervalBin(
                code=idx,
                label=label,
                lower=lower,
                upper=upper,
                include_lower=True,
                include_upper=idx == len(labels) - 1,
                provenance=provenance,
                threshold_ids=[ordered[idx].threshold_id] if idx < len(ordered) else [],
            )
        )
        lower = upper
    return bins


def _generic_state_labels(count: int, feature: str) -> list[str]:
    if count == 2:
        return ["low", "high"]
    if count == 3:
        return ["low", "middle", "high"]
    return [f"interval_{idx}" for idx in range(count)]


def _clinical_bins(
    feature: str,
    signature_thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> list[IntervalBin] | None:
    # Clinically anchored screening bins used by the rule layer. These bins are
    # intentionally conservative and audit-friendly; they do not replace expert ECG review.
    specs: dict[str, list[tuple[str, float | None, float | None]]] = {
        "hr_med_bpm": [("bradycardic", None, 60.0), ("normal_rate", 60.0, 100.0), ("tachycardic", 100.0, None)],
        "rr_med_ms": [("tachycardic_rr", None, 600.0), ("normal_rr", 600.0, 1000.0), ("bradycardic_rr", 1000.0, None)],
        "rr_iqr_ms": [("stable", None, 50.0), ("variable", 50.0, 120.0), ("highly_variable", 120.0, None)],
        "rr_sdnn_ms": [("stable", None, 50.0), ("variable", 50.0, 120.0), ("highly_variable", 120.0, None)],
        "prematurity_index_min": [("very_premature", None, 0.8), ("premature", 0.8, 0.9), ("not_premature", 0.9, None)],
        "comp_pause_ratio_max": [("no_compensatory_pause", None, 1.15), ("possible_compensatory_pause", 1.15, 1.8), ("compensatory_pause", 1.8, None)],
        "pvc_like_beat_count": [("zero", None, 0.5), ("present", 0.5, 3.0), ("frequent", 3.0, None)],
        "apb_like_beat_count": [("zero", None, 0.5), ("present", 0.5, 3.0), ("frequent", 3.0, None)],
        "paced_like_beat_count": [("zero", None, 0.5), ("present", 0.5, 3.0), ("frequent", 3.0, None)],
        "af_irregularity_cv": [("regular", None, 0.10), ("mildly_irregular", 0.10, 0.20), ("irregular", 0.20, None)],
        "f_wave_power_ratio": [("no_f_wave", None, 0.20), ("possible_f_wave", 0.20, 0.50), ("f_wave_present", 0.50, None)],
        "p_present_ratio": [("absent_or_low", None, 0.30), ("intermittent", 0.30, 0.80), ("present", 0.80, None)],
        "p_dur_med_ms": [("short", None, 80.0), ("normal", 80.0, 120.0), ("prolonged", 120.0, None)],
        "pr_med_ms": [("short_pr", None, 120.0), ("normal_pr", 120.0, 200.0), ("prolonged_pr", 200.0, None)],
        "pr_iqr_ms": [("stable", None, 20.0), ("variable", 20.0, 60.0), ("highly_variable", 60.0, None)],
        "qrs_dur_med_ms": [("narrow_qrs", None, 110.0), ("borderline_qrs", 110.0, 120.0), ("wide_qrs", 120.0, None)],
        "qrs_dur_iqr_ms": [("stable", None, 20.0), ("variable", 20.0, 50.0), ("highly_variable", 50.0, None)],
        "qrs_deformed_prob": [("not_deformed", None, 0.5), ("deformed", 0.5, None)],
        "st_level_v1_mV": [("depressed", None, -0.10), ("isoelectric", -0.10, 0.10), ("elevated", 0.10, None)],
        "st_level_v5_mV": [("depressed", None, -0.10), ("isoelectric", -0.10, 0.10), ("elevated", 0.10, None)],
        "st_slope_v5_uV_per_ms": [("downsloping", None, -1.0), ("flat", -1.0, 1.0), ("upsloping", 1.0, None)],
        "t_amp_v5_med_mV": [("low_or_negative", None, 0.0), ("low_positive", 0.0, 0.15), ("positive", 0.15, None)],
        "t_dur_med_ms": [("short", None, 100.0), ("normal", 100.0, 250.0), ("prolonged", 250.0, None)],
        "qt_med_ms": [("normal_qt", None, 450.0), ("prolonged_qt", 450.0, 500.0), ("very_prolonged_qt", 500.0, None)],
        "qtc_med_ms": [("normal_qtc", None, 450.0), ("prolonged_qtc", 450.0, 500.0), ("very_prolonged_qtc", 500.0, None)],
        "qrs_axis_deg": [("left_axis", None, -30.0), ("normal_axis", -30.0, 90.0), ("right_axis", 90.0, None)],
        "lead_quality_min_db": [("low_quality", None, 5.0), ("marginal_quality", 5.0, 10.0), ("good_quality", 10.0, None)],
        "delineation_confidence": [("low_confidence", None, 0.50), ("medium_confidence", 0.50, 0.80), ("high_confidence", 0.80, None)],
        "rhythm_valid_beat_fraction": [("insufficient", None, 0.50), ("limited", 0.50, 0.80), ("sufficient", 0.80, None)],
        "atrial_valid_beat_fraction": [("insufficient", None, 0.50), ("limited", 0.50, 0.80), ("sufficient", 0.80, None)],
        "qrs_valid_beat_fraction": [("insufficient", None, 0.50), ("limited", 0.50, 0.80), ("sufficient", 0.80, None)],
        "st_t_valid_beat_fraction": [("insufficient", None, 0.50), ("limited", 0.50, 0.80), ("sufficient", 0.80, None)],
        "atrial_lead_coverage": [("insufficient", None, 0.67), ("sufficient", 0.67, None)],
        "qrs_lead_coverage": [("insufficient", None, 0.60), ("sufficient", 0.60, None)],
        "st_t_lead_coverage": [("insufficient", None, 0.60), ("sufficient", 0.60, None)],
        "detector_agreement": [("low", None, 0.50), ("moderate", 0.50, 0.80), ("high", 0.80, None)],
        "analyzable_duration_s": [("too_short", None, 7.5), ("adequate", 7.5, None)],
        "u_amp_v2_mV": [("absent_or_low", None, 0.05), ("prominent", 0.05, None)],
    }
    if feature.endswith("_signature_score"):
        threshold = dict((signature_thresholds or {}).get(feature, {}))
        if not threshold:
            return None
        negative = float(threshold["negative_max_logodds"])
        positive = float(threshold["positive_min_logodds"])
        if not negative < positive:
            raise ValueError(
                f"Signature thresholds must satisfy negative < positive for {feature}"
            )
        specs[feature] = [
            ("negative", None, negative),
            ("neutral", negative, positive),
            ("positive", positive, None),
        ]
    rows = specs.get(feature)
    if rows is None:
        return None
    bins = []
    for idx, (label, lower, upper) in enumerate(rows):
        bins.append(
            IntervalBin(
                code=idx,
                label=label,
                lower=lower,
                upper=upper,
                include_lower=True,
                include_upper=idx == len(rows) - 1,
                provenance="clinical_anchor",
                threshold_ids=[f"{feature}:clinical:{idx}"],
            )
        )
    return bins


def _binary_domain(feature: str, unit: str, family: str) -> AttributeDomain:
    return AttributeDomain(
        name=feature,
        value_type="binary",
        unit=unit,
        family=family,
        allowed_states=["absent", "present"],
        clinical_priority=CLINICAL_PRIORITIES.get(feature, 1.0),
        provenance="binary_threshold_0.5",
    )


def fit_wedd_discretization(
    rows: list[Mapping[str, object]],
    labels: list[str],
    features: list[str] | None = None,
    alpha: float = 0.65,
    min_support: int = 5,
    max_depth: int = 3,
    min_gain: float = 1e-6,
    prefer_clinical_bins: bool = True,
    signature_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    patient_ids: list[str] | None = None,
    fit_partition: str = "training_or_oof",
    n_bootstrap: int = 0,
    stability_tolerance: float = 0.05,
    min_selection_frequency: float = 0.50,
    perturbation_tolerance: float = 0.0,
    max_perturbation_flip_rate: float = 1.0,
    random_seed: int = 19,
) -> DiscretizationPlan:
    if len(rows) != len(labels):
        raise ValueError("rows and labels must be aligned")
    if patient_ids is not None and len(patient_ids) != len(rows):
        raise ValueError("patient_ids must align with rows")
    if fit_partition not in {"training", "oof", "training_or_oof"}:
        raise ValueError("WEDD thresholds may only be fitted on training or OOF values")
    selected_features = features or [feature for feature in FEATURE_SPECS if feature in (rows[0].keys() if rows else FEATURE_SPECS)]
    domains: dict[str, AttributeDomain] = {}
    thresholds: list[ThresholdRecord] = []
    candidates: list[ThresholdRecord] = []
    for feature in selected_features:
        spec = FEATURE_SPECS.get(feature, ("", "", "continuous", ""))
        family, unit, value_type, _formula = spec
        if value_type == "binary":
            domains[feature] = _binary_domain(feature, unit, family)
            continue
        clinical_bins = (
            _clinical_bins(feature, signature_thresholds) if prefer_clinical_bins else None
        )
        if clinical_bins is not None:
            domains[feature] = AttributeDomain(
                name=feature,
                value_type=value_type,
                unit=unit,
                family=family,
                bins=clinical_bins,
                clinical_priority=CLINICAL_PRIORITIES.get(feature, 1.0),
                provenance="clinical_anchor",
            )
            for interval in clinical_bins:
                if interval.upper is not None:
                    thresholds.append(
                        ThresholdRecord(
                            threshold_id=interval.threshold_ids[0] if interval.threshold_ids else f"{feature}:clinical:{interval.code}",
                            feature=feature,
                            value=interval.upper,
                            left_size=0,
                            right_size=0,
                            entropy=0.0,
                            density=0.0,
                            objective=0.0,
                            source="clinical_anchor",
                            alpha=alpha,
                            accepted=True,
                            notes="Clinically anchored screening bin; not learned from this cohort.",
                            unit=unit,
                        )
                    )
            continue
        values: list[float] = []
        used_labels: list[str] = []
        used_patient_ids: list[str] = []
        for row_index, (row, label) in enumerate(zip(rows, labels, strict=False)):
            numeric = _as_float(row.get(feature))
            if numeric is None:
                continue
            values.append(numeric)
            used_labels.append(label)
            if patient_ids is not None:
                used_patient_ids.append(str(patient_ids[row_index]))
        feature_candidates = wedd_candidate_scores(
            values,
            used_labels,
            alpha=alpha,
            min_leaf=min_support,
        )
        for index, record in enumerate(feature_candidates):
            record.feature = feature
            record.threshold_id = f"{feature}:candidate:{index}:{record.value:.8g}"
            record.unit = unit
        candidates.extend(feature_candidates)
        learned_thresholds = _recursive_wedd(
            feature,
            values,
            used_labels,
            alpha=alpha,
            min_leaf=min_support,
            max_depth=max_depth,
            min_gain=min_gain,
        )
        for record in learned_thresholds:
            record.unit = unit
        learned_thresholds = _annotate_threshold_stability(
            learned_thresholds,
            values=values,
            labels=used_labels,
            patient_ids=used_patient_ids if patient_ids is not None else None,
            alpha=alpha,
            min_support=min_support,
            n_bootstrap=n_bootstrap,
            stability_tolerance=stability_tolerance,
            min_selection_frequency=min_selection_frequency,
            perturbation_tolerance=perturbation_tolerance,
            max_perturbation_flip_rate=max_perturbation_flip_rate,
            seed=random_seed,
        )
        thresholds.extend(learned_thresholds)
        domains[feature] = AttributeDomain(
            name=feature,
            value_type=value_type,
            unit=unit,
            family=family,
            bins=_bins_from_thresholds(feature, learned_thresholds, value_type, "WEDD"),
            clinical_priority=CLINICAL_PRIORITIES.get(feature, 1.0),
            provenance="WEDD" if learned_thresholds else "observed_no_split",
        )
    return DiscretizationPlan(
        feature_domains=domains,
        thresholds=thresholds,
        class_labels=sorted(set(labels)),
        alpha=alpha,
        min_support=min_support,
        max_depth=max_depth,
        candidate_thresholds=candidates,
        fit_partition=fit_partition,
    )


def quantize_row(row: Mapping[str, object], plan: DiscretizationPlan) -> dict[str, str]:
    return {feature: domain.state_for_value(row.get(feature)) for feature, domain in plan.feature_domains.items()}


def quantize_rows(rows: list[Mapping[str, object]], plan: DiscretizationPlan) -> list[dict[str, str]]:
    return [quantize_row(row, plan) for row in rows]


def build_decision_table(
    rows: list[Mapping[str, object]],
    labels: list[str],
    plan: DiscretizationPlan,
    object_ids: list[str] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if len(rows) != len(labels):
        raise ValueError("rows and labels must be aligned")
    ids = object_ids or [str(row.get("record_id", idx)) for idx, row in enumerate(rows)]
    if len(ids) != len(rows):
        raise ValueError("object_ids and rows must be aligned")
    information = {object_id: quantize_row(row, plan) for object_id, row in zip(ids, rows, strict=False)}
    decisions = {object_id: label for object_id, label in zip(ids, labels, strict=False)}
    return information, decisions
