"""Premature-beat APB/PVC specialist with robust record aggregation."""

from __future__ import annotations

from typing import Mapping, Sequence

from tm_ecg.modeling.specialists.base import (
    SpecialistOutput,
    logistic_score,
    robust_summary,
)


def ectopy_specialist(
    beats: Sequence[Mapping[str, object]],
    *,
    signal_eligible: bool = True,
) -> SpecialistOutput:
    """Score none/atrial/ventricular ectopy without a direct label override."""

    import numpy as np  # type: ignore

    numeric_names = (
        "prematurity_index",
        "compensatory_pause_ratio",
        "qrs_duration_ms",
        "qrs_morphology_distance",
        "preceding_p_wave_probability",
        "pr_interval_ms",
        "axis_concordance",
    )
    columns: dict[str, list[float]] = {name: [] for name in numeric_names}
    candidate_count = 0
    atrial_votes = 0
    ventricular_votes = 0
    for beat in beats:
        try:
            premature = float(beat.get("prematurity_index", 1.0)) < 0.85
        except (TypeError, ValueError):
            premature = False
        candidate_count += int(premature)
        for name in numeric_names:
            try:
                value = float(beat[name])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                columns[name].append(value)
        if premature:
            p_probability = float(beat.get("preceding_p_wave_probability", 0.0) or 0.0)
            qrs_duration = float(beat.get("qrs_duration_ms", 0.0) or 0.0)
            morphology = float(beat.get("qrs_morphology_distance", 0.0) or 0.0)
            atrial_votes += int(p_probability >= 0.55 and qrs_duration < 120.0)
            ventricular_votes += int(qrs_duration >= 120.0 or morphology >= 0.45)

    features: dict[str, float | None] = {
        "beat_count": float(len(beats)),
        "candidate_premature_beat_count": float(candidate_count),
        "candidate_premature_beat_fraction": (
            candidate_count / len(beats) if beats else None
        ),
        "atrial_candidate_fraction": (
            atrial_votes / candidate_count if candidate_count else None
        ),
        "ventricular_candidate_fraction": (
            ventricular_votes / candidate_count if candidate_count else None
        ),
    }
    for name, values in columns.items():
        summary = robust_summary(values)
        for statistic, value in summary.items():
            features[f"{name}_{statistic}"] = value
    quality_flags: list[str] = []
    if len(beats) < 5:
        quality_flags.append("insufficient_usable_beats")
    if not signal_eligible:
        quality_flags.append("shared_signal_quality_ineligible")
    eligible = signal_eligible and len(beats) >= 5
    if not eligible:
        probabilities: dict[str, float | None] = {
            "none": None,
            "APB": None,
            "PVC": None,
            "mixed_or_uncertain": None,
        }
    else:
        apb = logistic_score(
            features,
            {
                "candidate_premature_beat_fraction": 4.0,
                "atrial_candidate_fraction": 3.5,
                "preceding_p_wave_probability_median": 2.0,
                "qrs_duration_ms_median": -0.012,
                "qrs_morphology_distance_median": -1.5,
            },
            intercept=-2.5,
        )
        pvc = logistic_score(
            features,
            {
                "candidate_premature_beat_fraction": 4.0,
                "ventricular_candidate_fraction": 3.5,
                "qrs_duration_ms_median": 0.018,
                "qrs_morphology_distance_median": 2.5,
                "compensatory_pause_ratio_median": 1.5,
            },
            intercept=-4.0,
        )
        mixed = min(apb, pvc)
        none = max(0.0, 1.0 - max(apb, pvc))
        probabilities = {
            "none": none,
            "APB": apb,
            "PVC": pvc,
            "mixed_or_uncertain": mixed,
        }
    return SpecialistOutput(
        specialist_id="ectopy_v1",
        probabilities=probabilities,
        eligible=eligible,
        quality_flags=tuple(quality_flags),
        features=features,
        model_version="quality_gated_beat_aggregation_v1",
    )

