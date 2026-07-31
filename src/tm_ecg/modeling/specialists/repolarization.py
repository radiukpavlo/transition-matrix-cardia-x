"""Lead-aware ST/T/QT repolarization specialist."""

from __future__ import annotations

from typing import Mapping, Sequence

from tm_ecg.modeling.specialists.base import SpecialistOutput, logistic_score


def _lead_group_max(
    values: Mapping[str, float],
    lead_groups: Sequence[Sequence[str]],
    *,
    absolute: bool = False,
) -> float | None:
    candidates: list[float] = []
    for group in lead_groups:
        present = [values[lead] for lead in group if lead in values]
        if len(present) >= 2:
            candidates.append(
                max(abs(value) for value in present)
                if absolute
                else max(present)
            )
    return max(candidates) if candidates else None


def repolarization_specialist(
    measurements: Mapping[str, object],
    *,
    signal_eligible: bool = True,
) -> SpecialistOutput:
    """Return semantic repolarization probabilities for validation and DSS."""

    import math

    leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    st: dict[str, float] = {}
    t_amplitude: dict[str, float] = {}
    for lead in leads:
        for prefix, destination in (
            ("st_j60_mv_", st),
            ("t_amplitude_mv_", t_amplitude),
        ):
            try:
                value = float(measurements[f"{prefix}{lead}"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                destination[lead] = value
    contiguous = (
        ("II", "III", "aVF"),
        ("I", "aVL", "V5", "V6"),
        ("V1", "V2", "V3", "V4"),
    )
    try:
        qtc = float(measurements.get("qtc_ms"))
        qtc = qtc if math.isfinite(qtc) else None
    except (TypeError, ValueError):
        qtc = None
    features: dict[str, float | None] = {
        "contiguous_st_elevation_max_mv": _lead_group_max(st, contiguous),
        "contiguous_st_depression_max_mv": (
            _lead_group_max(
                {lead: -value for lead, value in st.items()},
                contiguous,
            )
        ),
        "t_wave_absolute_amplitude_max_mv": _lead_group_max(
            t_amplitude,
            contiguous,
            absolute=True,
        ),
        "t_wave_negative_lead_fraction": (
            sum(value < 0 for value in t_amplitude.values()) / len(t_amplitude)
            if t_amplitude
            else None
        ),
        "qtc_ms": qtc,
        "baseline_reference_stability": _finite_value(
            measurements.get("baseline_reference_stability")
        ),
        "fiducial_confidence": _finite_value(
            measurements.get("fiducial_confidence")
        ),
    }
    confidence = features["fiducial_confidence"]
    quality_flags: list[str] = []
    if len(st) < 6:
        quality_flags.append("insufficient_st_lead_coverage")
    if confidence is None or confidence < 0.30:
        quality_flags.append("low_fiducial_confidence")
    if not signal_eligible:
        quality_flags.append("shared_signal_quality_ineligible")
    eligible = (
        signal_eligible
        and len(st) >= 6
        and confidence is not None
        and confidence >= 0.30
    )
    if not eligible:
        probabilities: dict[str, float | None] = {
            "st_elevation": None,
            "st_depression": None,
            "t_wave_abnormality": None,
            "qtc_abnormality": None,
        }
    else:
        probabilities = {
            "st_elevation": logistic_score(
                features,
                {
                    "contiguous_st_elevation_max_mv": 16.0,
                    "baseline_reference_stability": 1.0,
                },
                intercept=-2.5,
            ),
            "st_depression": logistic_score(
                features,
                {
                    "contiguous_st_depression_max_mv": 16.0,
                    "baseline_reference_stability": 1.0,
                },
                intercept=-2.5,
            ),
            "t_wave_abnormality": logistic_score(
                features,
                {
                    "t_wave_negative_lead_fraction": 4.0,
                    "t_wave_absolute_amplitude_max_mv": 0.5,
                },
                intercept=-2.0,
            ),
            "qtc_abnormality": logistic_score(
                features,
                {"qtc_ms": 0.025},
                intercept=-11.25,
            ),
        }
    return SpecialistOutput(
        specialist_id="repolarization_v1",
        probabilities=probabilities,
        eligible=eligible,
        quality_flags=tuple(quality_flags),
        features=features,
        model_version="quality_gated_contiguous_lead_seed_v1",
    )


def _finite_value(value: object) -> float | None:
    import math

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None

