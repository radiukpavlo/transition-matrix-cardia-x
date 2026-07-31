"""Quality-gated AF/AFL waveform and rhythm specialist features."""

from __future__ import annotations

from typing import Mapping, Sequence

from tm_ecg.modeling.specialists.base import SpecialistOutput, logistic_score


def _sample_entropy(values: object, tolerance_scale: float = 0.2) -> float | None:
    import numpy as np  # type: ignore

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 8 or float(array.std()) <= 1e-9:
        return None
    tolerance = tolerance_scale * float(array.std())

    def matches(length: int) -> int:
        count = 0
        for left in range(len(array) - length):
            template = array[left : left + length]
            for right in range(left + 1, len(array) - length + 1):
                if np.max(np.abs(template - array[right : right + length])) <= tolerance:
                    count += 1
        return count

    two = matches(2)
    three = matches(3)
    if two == 0 or three == 0:
        return None
    return float(-np.log(three / two))


def _turning_point_ratio(values: object) -> float | None:
    import numpy as np  # type: ignore

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 3:
        return None
    turns = (
        ((array[1:-1] > array[:-2]) & (array[1:-1] > array[2:]))
        | ((array[1:-1] < array[:-2]) & (array[1:-1] < array[2:]))
    )
    return float(turns.mean())


def _atrial_spectrum(
    segments: object | None,
    sampling_rate_hz: float,
) -> dict[str, float | None]:
    import numpy as np  # type: ignore

    if segments is None:
        return {
            "dominant_atrial_frequency_hz": None,
            "atrial_harmonic_ratio": None,
            "sawtooth_score": None,
        }
    array = np.asarray(segments, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 16:
        return {
            "dominant_atrial_frequency_hz": None,
            "atrial_harmonic_ratio": None,
            "sawtooth_score": None,
        }
    median = np.nanmedian(array, axis=0)
    median = np.nan_to_num(median - np.nanmedian(median))
    spectrum = np.abs(np.fft.rfft(median)) ** 2
    frequencies = np.fft.rfftfreq(len(median), d=1.0 / sampling_rate_hz)
    band = (frequencies >= 3.0) & (frequencies <= 12.0)
    if not band.any() or float(spectrum[band].sum()) <= 1e-12:
        dominant = None
        harmonic = None
    else:
        indices = np.flatnonzero(band)
        dominant_index = int(indices[int(np.argmax(spectrum[band]))])
        dominant = float(frequencies[dominant_index])
        harmonic_frequency = 2.0 * dominant
        harmonic_index = int(np.argmin(np.abs(frequencies - harmonic_frequency)))
        harmonic = float(
            spectrum[harmonic_index] / max(spectrum[dominant_index], 1e-12)
        )
    derivative = np.diff(median)
    sawtooth = (
        float(abs(np.mean(derivative**3)) / (np.std(derivative) ** 3 + 1e-9))
        if len(derivative)
        else None
    )
    return {
        "dominant_atrial_frequency_hz": dominant,
        "atrial_harmonic_ratio": harmonic,
        "sawtooth_score": sawtooth,
    }


def atrial_rhythm_specialist(
    rr_intervals_ms: Sequence[float],
    *,
    p_wave_confidence_by_lead: Mapping[str, Sequence[float]] | None = None,
    p_to_qrs_association: Sequence[float] | None = None,
    atrial_segments: object | None = None,
    sampling_rate_hz: float = 500.0,
    signal_eligible: bool = True,
) -> SpecialistOutput:
    """Return AF/AFL probabilities and eligibility without hard overrides."""

    import numpy as np  # type: ignore

    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    rr_cv = float(rr.std() / rr.mean()) if len(rr) >= 3 and rr.mean() else None
    p_values = [
        float(value)
        for lead in ("II", "V1")
        for value in (p_wave_confidence_by_lead or {}).get(lead, ())
        if np.isfinite(value)
    ]
    p_presence = float(np.mean(p_values)) if p_values else None
    association_values = np.asarray(
        p_to_qrs_association
        if p_to_qrs_association is not None
        else (),
        dtype=float,
    )
    association_values = association_values[np.isfinite(association_values)]
    association = (
        float(association_values.mean()) if len(association_values) else None
    )
    features: dict[str, float | None] = {
        "rr_cv": rr_cv,
        "rr_sample_entropy": _sample_entropy(rr),
        "rr_turning_point_ratio": _turning_point_ratio(rr),
        "p_wave_presence_confidence": p_presence,
        "p_to_qrs_association_consistency": association,
        **_atrial_spectrum(atrial_segments, sampling_rate_hz),
    }
    quality_flags: list[str] = []
    if len(rr) < 8:
        quality_flags.append("insufficient_rr_intervals")
    if p_presence is None:
        quality_flags.append("p_wave_confidence_unavailable")
    if not signal_eligible:
        quality_flags.append("shared_signal_quality_ineligible")
    eligible = signal_eligible and len(rr) >= 8
    if not eligible:
        probabilities: dict[str, float | None] = {
            "sinus_family": None,
            "AF": None,
            "AFL": None,
        }
    else:
        af = logistic_score(
            features,
            {
                "rr_cv": 6.0,
                "rr_sample_entropy": 1.2,
                "rr_turning_point_ratio": 1.0,
                "p_wave_presence_confidence": -3.0,
                "p_to_qrs_association_consistency": -2.0,
            },
            intercept=-2.3,
        )
        afl = logistic_score(
            features,
            {
                "dominant_atrial_frequency_hz": 0.35,
                "atrial_harmonic_ratio": 1.2,
                "sawtooth_score": 0.5,
                "rr_cv": -1.0,
                "p_wave_presence_confidence": -1.5,
            },
            intercept=-3.0,
        )
        sinus = logistic_score(
            features,
            {
                "rr_cv": -5.0,
                "rr_sample_entropy": -0.8,
                "p_wave_presence_confidence": 2.5,
                "p_to_qrs_association_consistency": 2.0,
            },
            intercept=0.0,
        )
        probabilities = {"sinus_family": sinus, "AF": af, "AFL": afl}
    return SpecialistOutput(
        specialist_id="atrial_rhythm_v1",
        probabilities=probabilities,
        eligible=eligible,
        quality_flags=tuple(quality_flags),
        features=features,
        model_version="quality_gated_shrunken_logistic_seed_v1",
    )
