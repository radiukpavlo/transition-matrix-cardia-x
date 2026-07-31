"""RBBB/LBBB and generic conduction-presence specialist."""

from __future__ import annotations

from typing import Mapping

from tm_ecg.modeling.specialists.base import SpecialistOutput, logistic_score


def conduction_specialist(
    measurements: Mapping[str, object],
    *,
    signal_eligible: bool = True,
) -> SpecialistOutput:
    """Produce quality-gated probabilistic conduction features."""

    import math

    feature_names = (
        "qrs_duration_ms",
        "v1_rsr_score",
        "v1_terminal_r_score",
        "i_v6_terminal_s_score",
        "v1_qs_rs_score",
        "i_v6_broad_notched_r_score",
        "frontal_axis_deg",
        "r_peak_time_ms",
        "qrs_area_concordance",
        "pr_interval_ms",
        "morphology_confidence",
    )
    features: dict[str, float | None] = {}
    for name in feature_names:
        try:
            value = float(measurements[name])
        except (KeyError, TypeError, ValueError):
            features[name] = None
            continue
        features[name] = value if math.isfinite(value) else None
    confidence = features.get("morphology_confidence")
    quality_flags: list[str] = []
    if features.get("qrs_duration_ms") is None:
        quality_flags.append("qrs_duration_unavailable")
    if confidence is None or confidence < 0.30:
        quality_flags.append("low_morphology_confidence")
    if not signal_eligible:
        quality_flags.append("shared_signal_quality_ineligible")
    eligible = (
        signal_eligible
        and features.get("qrs_duration_ms") is not None
        and confidence is not None
        and confidence >= 0.30
    )
    if not eligible:
        probabilities: dict[str, float | None] = {
            "RBBB spectrum": None,
            "LBBB spectrum": None,
            "other_conduction_presence": None,
        }
    else:
        rbbb = logistic_score(
            features,
            {
                "qrs_duration_ms": 0.025,
                "v1_rsr_score": 2.5,
                "v1_terminal_r_score": 1.8,
                "i_v6_terminal_s_score": 1.8,
                "morphology_confidence": 1.0,
            },
            intercept=-4.0,
        )
        lbbb = logistic_score(
            features,
            {
                "qrs_duration_ms": 0.027,
                "v1_qs_rs_score": 2.0,
                "i_v6_broad_notched_r_score": 2.5,
                "r_peak_time_ms": 0.01,
                "morphology_confidence": 1.0,
            },
            intercept=-4.4,
        )
        other = logistic_score(
            features,
            {
                "qrs_duration_ms": 0.02,
                "pr_interval_ms": 0.006,
                "qrs_area_concordance": -0.8,
            },
            intercept=-3.8,
        )
        probabilities = {
            "RBBB spectrum": rbbb,
            "LBBB spectrum": lbbb,
            "other_conduction_presence": other,
        }
    return SpecialistOutput(
        specialist_id="conduction_v1",
        probabilities=probabilities,
        eligible=eligible,
        quality_flags=tuple(quality_flags),
        features=features,
        model_version="quality_gated_morphology_logistic_seed_v1",
    )

