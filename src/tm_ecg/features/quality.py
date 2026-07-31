"""Shared signal-quality features used to gate ECG specialists."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SignalQualityVector:
    lead_names: tuple[str, ...]
    flatline_rate: tuple[float, ...]
    clipping_rate: tuple[float, ...]
    baseline_wander_energy_ratio: tuple[float, ...]
    high_frequency_noise_ratio: tuple[float, ...]
    powerline_contamination_ratio: tuple[float, ...]
    lead_dropout_mask: tuple[bool, ...]
    rpeak_detection_confidence: float
    beat_count: int
    usable_beat_fraction: float
    median_beat_stability: float | None
    fiducial_completeness: float | None
    cross_lead_timing_consistency: float | None
    eligible_lead_fraction: float
    globally_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _band_energy_ratio(
    signal: object,
    sampling_rate_hz: float,
    lower_hz: float,
    upper_hz: float,
) -> float:
    import numpy as np  # type: ignore

    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return 1.0
    centered = np.where(finite, values, np.nanmedian(values[finite]))
    centered = centered - float(np.mean(centered))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate_hz)
    total = float(spectrum[1:].sum())
    if total <= 1e-12:
        return 0.0
    mask = (frequencies >= lower_hz) & (frequencies <= upper_hz)
    return float(spectrum[mask].sum() / total)


def _median_beat_stability(beats: object | None) -> float | None:
    import numpy as np  # type: ignore

    if beats is None:
        return None
    matrix = np.asarray(beats, dtype=float)
    if matrix.ndim < 2 or matrix.shape[0] < 2:
        return None
    flattened = matrix.reshape(matrix.shape[0], -1)
    median = np.nanmedian(flattened, axis=0)
    scale = float(np.nanmedian(np.abs(median - np.nanmedian(median)))) + 1e-6
    deviations = np.nanmedian(np.abs(flattened - median), axis=1) / scale
    return float(1.0 / (1.0 + np.nanmedian(deviations)))


def _fiducial_completeness(
    fiducials: Mapping[str, Sequence[object]] | None,
    beat_count: int,
) -> float | None:
    if fiducials is None or beat_count <= 0:
        return None
    required = ("p_onset", "p_peak", "qrs_onset", "r_peak", "qrs_offset", "t_peak", "t_offset")
    observed = 0
    possible = beat_count * len(required)
    for name in required:
        values = fiducials.get(name, ())
        observed += sum(value is not None for value in values[:beat_count])
    return observed / possible if possible else None


def _timing_consistency(
    lead_rpeaks: Sequence[Sequence[int]] | None,
    sampling_rate_hz: float,
) -> float | None:
    import numpy as np  # type: ignore

    if not lead_rpeaks or len(lead_rpeaks) < 2:
        return None
    usable = [np.asarray(values, dtype=float) for values in lead_rpeaks if len(values)]
    if len(usable) < 2:
        return None
    minimum = min(len(values) for values in usable)
    if minimum < 2:
        return None
    aligned = np.vstack([values[:minimum] for values in usable])
    dispersion_ms = np.median(np.std(aligned, axis=0)) * 1000.0 / sampling_rate_hz
    return float(1.0 / (1.0 + dispersion_ms / 20.0))


def compute_signal_quality(
    signal: object,
    *,
    sampling_rate_hz: float,
    lead_names: Sequence[str] | None = None,
    rpeaks: Sequence[int] | None = None,
    lead_rpeaks: Sequence[Sequence[int]] | None = None,
    beats: object | None = None,
    fiducials: Mapping[str, Sequence[object]] | None = None,
) -> SignalQualityVector:
    """Compute leakage-safe quality channels from waveform measurements only."""

    import numpy as np  # type: ignore

    matrix = np.asarray(signal, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 8:
        raise ValueError("ECG signal must be samples-by-leads with at least 8 samples")
    if not math_is_finite_positive(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be finite and positive")
    names = tuple(
        str(value)
        for value in (
            lead_names
            if lead_names is not None
            else [f"lead_{index}" for index in range(matrix.shape[1])]
        )
    )
    if len(names) != matrix.shape[1]:
        raise ValueError("lead_names must align with the waveform columns")

    flatline: list[float] = []
    clipping: list[float] = []
    baseline: list[float] = []
    high_frequency: list[float] = []
    powerline: list[float] = []
    dropout: list[bool] = []
    nyquist = sampling_rate_hz / 2.0
    for column in range(matrix.shape[1]):
        lead = matrix[:, column]
        finite = np.isfinite(lead)
        if finite.sum() < 8:
            flatline.append(1.0)
            clipping.append(1.0)
            baseline.append(1.0)
            high_frequency.append(1.0)
            powerline.append(1.0)
            dropout.append(True)
            continue
        values = lead[finite]
        amplitude_scale = float(np.nanpercentile(values, 99) - np.nanpercentile(values, 1))
        diff = np.abs(np.diff(values))
        tolerance = max(amplitude_scale * 1e-5, 1e-8)
        flat_rate = float((diff <= tolerance).mean()) if len(diff) else 1.0
        lower = float(np.nanpercentile(values, 0.1))
        upper = float(np.nanpercentile(values, 99.9))
        clip_tolerance = max(amplitude_scale * 1e-4, 1e-8)
        clip_rate = float(
            ((values <= lower + clip_tolerance) | (values >= upper - clip_tolerance)).mean()
        )
        flatline.append(flat_rate)
        clipping.append(clip_rate)
        baseline.append(
            _band_energy_ratio(values, sampling_rate_hz, 0.01, min(0.5, nyquist))
        )
        high_frequency.append(
            _band_energy_ratio(values, sampling_rate_hz, min(40.0, nyquist), nyquist)
            if nyquist > 40.0
            else 0.0
        )
        mains = max(
            _band_energy_ratio(values, sampling_rate_hz, 49.0, min(51.0, nyquist))
            if nyquist >= 49.0
            else 0.0,
            _band_energy_ratio(values, sampling_rate_hz, 59.0, min(61.0, nyquist))
            if nyquist >= 59.0
            else 0.0,
        )
        powerline.append(mains)
        dropout.append(
            amplitude_scale < 1e-4 or flat_rate > 0.20 or finite.mean() < 0.95
        )

    peaks = sorted(
        set(
            int(value)
            for value in (rpeaks if rpeaks is not None else ())
        )
    )
    rr = np.diff(peaks) / sampling_rate_hz if len(peaks) >= 2 else np.asarray([])
    plausible_rr = (rr >= 0.25) & (rr <= 2.5)
    usable_fraction = float(plausible_rr.mean()) if len(rr) else 0.0
    expected_minimum = max(int(matrix.shape[0] / sampling_rate_hz / 3.0), 1)
    count_score = min(len(peaks) / expected_minimum, 1.0)
    rpeak_confidence = float(count_score * usable_fraction)
    timing = _timing_consistency(lead_rpeaks, sampling_rate_hz)
    stable = _median_beat_stability(beats)
    completeness = _fiducial_completeness(fiducials, len(peaks))
    eligible_leads = [
        not is_dropout
        and flatline[index] < 0.10
        and clipping[index] < 0.10
        and high_frequency[index] < 0.40
        for index, is_dropout in enumerate(dropout)
    ]
    eligible_fraction = sum(eligible_leads) / len(eligible_leads)
    globally_eligible = (
        eligible_fraction >= 0.5
        and rpeak_confidence >= 0.25
        and (stable is None or stable >= 0.20)
        and (completeness is None or completeness >= 0.40)
    )
    return SignalQualityVector(
        lead_names=names,
        flatline_rate=tuple(flatline),
        clipping_rate=tuple(clipping),
        baseline_wander_energy_ratio=tuple(baseline),
        high_frequency_noise_ratio=tuple(high_frequency),
        powerline_contamination_ratio=tuple(powerline),
        lead_dropout_mask=tuple(dropout),
        rpeak_detection_confidence=rpeak_confidence,
        beat_count=len(peaks),
        usable_beat_fraction=usable_fraction,
        median_beat_stability=stable,
        fiducial_completeness=completeness,
        cross_lead_timing_consistency=timing,
        eligible_lead_fraction=eligible_fraction,
        globally_eligible=globally_eligible,
    )


def math_is_finite_positive(value: object) -> bool:
    import math

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0
