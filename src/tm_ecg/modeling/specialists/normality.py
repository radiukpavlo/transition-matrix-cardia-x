"""Hierarchical normal-versus-abnormal and specific-versus-residual gates."""

from __future__ import annotations

from typing import Mapping

from tm_ecg.modeling.specialists.base import SpecialistOutput, logistic_score


def normality_specialist(
    specialist_probabilities: Mapping[str, float | None],
    measurements: Mapping[str, object],
    *,
    signal_eligible: bool = True,
) -> SpecialistOutput:
    """Combine source-independent abnormality evidence into two hierarchy gates."""

    import math

    finite_probabilities = {
        name: float(value)
        for name, value in specialist_probabilities.items()
        if value is not None and math.isfinite(float(value))
    }
    features: dict[str, float | None] = {
        "maximum_specific_abnormality_probability": (
            max(finite_probabilities.values()) if finite_probabilities else None
        ),
        "mean_specific_abnormality_probability": (
            sum(finite_probabilities.values()) / len(finite_probabilities)
            if finite_probabilities
            else None
        ),
        "interval_deviation_score": _finite(
            measurements.get("interval_deviation_score")
        ),
        "axis_deviation_score": _finite(
            measurements.get("axis_deviation_score")
        ),
        "waveform_anomaly_score": _finite(
            measurements.get("waveform_anomaly_score")
        ),
        "latent_novelty_score": _finite(
            measurements.get("latent_novelty_score")
        ),
        "measurement_missing_fraction": _finite(
            measurements.get("measurement_missing_fraction")
        ),
    }
    quality_flags: list[str] = []
    if not finite_probabilities:
        quality_flags.append("specialist_probabilities_unavailable")
    if not signal_eligible:
        quality_flags.append("shared_signal_quality_ineligible")
    eligible = signal_eligible and bool(finite_probabilities)
    if not eligible:
        probabilities: dict[str, float | None] = {
            "Normal": None,
            "abnormal": None,
            "specific_given_abnormal": None,
            "Other / unmapped_given_abnormal": None,
        }
    else:
        abnormal = logistic_score(
            features,
            {
                "maximum_specific_abnormality_probability": 5.0,
                "mean_specific_abnormality_probability": 2.0,
                "interval_deviation_score": 1.5,
                "axis_deviation_score": 1.0,
                "waveform_anomaly_score": 2.0,
                "latent_novelty_score": 1.0,
                "measurement_missing_fraction": -0.5,
            },
            intercept=-2.5,
        )
        specific_given_abnormal = logistic_score(
            features,
            {
                "maximum_specific_abnormality_probability": 5.0,
                "mean_specific_abnormality_probability": 1.0,
                "latent_novelty_score": -1.5,
                "waveform_anomaly_score": -0.5,
            },
            intercept=-1.5,
        )
        probabilities = {
            "Normal": 1.0 - abnormal,
            "abnormal": abnormal,
            "specific_given_abnormal": specific_given_abnormal,
            "Other / unmapped_given_abnormal": 1.0
            - specific_given_abnormal,
        }
    return SpecialistOutput(
        specialist_id="normality_v1",
        probabilities=probabilities,
        eligible=eligible,
        quality_flags=tuple(quality_flags),
        features=features,
        model_version="hierarchical_quality_adjusted_seed_v1",
    )


def _finite(value: object) -> float | None:
    import math

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None

