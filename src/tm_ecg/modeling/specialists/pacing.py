"""Quality-gated electronic pacing specialist."""

from __future__ import annotations

from tm_ecg.modeling.specialists.base import SpecialistOutput, logistic_score
from tm_ecg.signal.pacing import pacing_spike_features


def pacing_specialist(
    signal: object,
    *,
    sampling_rate_hz: float,
    rpeaks: object | None = None,
    paced_qrs_width_ms: float | None = None,
    paced_qrs_morphology_score: float | None = None,
    signal_eligible: bool = True,
) -> SpecialistOutput:
    features = pacing_spike_features(
        signal,
        sampling_rate_hz=sampling_rate_hz,
        rpeaks=rpeaks,
    )
    features["paced_qrs_width_ms"] = paced_qrs_width_ms
    features["paced_qrs_morphology_score"] = paced_qrs_morphology_score
    quality_flags: list[str] = []
    if int(features["spike_event_count"]) == 0:
        quality_flags.append("no_candidate_spikes")
    if float(features["high_frequency_noise_suppressor"]) > 0.8:
        quality_flags.append("high_frequency_noise")
    if not signal_eligible:
        quality_flags.append("shared_signal_quality_ineligible")
    eligible = (
        signal_eligible
        and float(features["high_frequency_noise_suppressor"]) <= 0.8
    )
    probability = (
        logistic_score(
            features,
            {
                "multi_lead_spike_count": 0.8,
                "median_lead_concurrence": 2.0,
                "median_adaptive_spike_score": 0.2,
                "pre_qrs_spike_fraction": 3.0,
                "spike_latency_iqr_ms": -0.03,
                "paced_qrs_width_ms": 0.012,
                "paced_qrs_morphology_score": 1.5,
                "high_frequency_noise_suppressor": -2.0,
            },
            intercept=-4.0,
        )
        if eligible
        else None
    )
    return SpecialistOutput(
        specialist_id="pacing_v1",
        probabilities={"Paced": probability},
        eligible=eligible,
        quality_flags=tuple(quality_flags),
        features={
            str(key): float(value) if value is not None else None
            for key, value in features.items()
        },
        model_version="adaptive_multilead_spike_seed_v1",
    )
