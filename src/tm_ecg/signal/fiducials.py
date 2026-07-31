"""Signal-derived fiducial delineation, detector agreement, and beat acceptance.

The routines in this module are deliberately conservative.  A boundary is emitted
only when the corresponding morphology has enough amplitude/prominence and a
physiologically plausible duration.  Missing boundaries therefore mean
"unavailable", never a fixed-offset surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from tm_ecg.types import BeatAcceptance, BeatFiducials


DELINEATION_METHOD = "adaptive_signal_envelope_v1"
SECONDARY_R_DETECTOR = "bandpass_energy_prominence_v1"


@dataclass(frozen=True, slots=True)
class BeatDelineation:
    """Fiducials plus component-level evidence for one beat."""

    fiducials: BeatFiducials
    p_confidence: float
    qrs_confidence: float
    t_confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RDetectorAgreement:
    """One-to-one temporal agreement between independent R detectors."""

    score: float | None
    matches: tuple[tuple[int, int], ...]
    matched_primary_indices: tuple[int, ...]
    primary_match_fraction: float | None
    secondary_match_fraction: float | None


@dataclass(frozen=True, slots=True)
class _WaveBounds:
    onset: int
    peak: int
    offset: int
    confidence: float


def cross_lead_fiducial_consistency(
    fiducials_by_lead: Mapping[str, BeatFiducials],
    *,
    sampling_rate_hz: float,
) -> dict[str, float | int | None]:
    """Summarize completeness and timing agreement without imputing boundaries."""

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    fields = (
        "p_on",
        "p_peak",
        "p_off",
        "qrs_on",
        "r_peak",
        "qrs_off",
        "t_on",
        "t_peak",
        "t_off",
    )
    per_field_mad_ms: list[float] = []
    observed = 0
    possible = len(fiducials_by_lead) * len(fields)
    for field in fields:
        values = sorted(
            float(value)
            for fiducials in fiducials_by_lead.values()
            if (value := getattr(fiducials, field)) is not None
            and math.isfinite(float(value))
        )
        observed += len(values)
        if len(values) < 2:
            continue
        middle = values[len(values) // 2]
        deviations = sorted(abs(value - middle) for value in values)
        mad_samples = deviations[len(deviations) // 2]
        per_field_mad_ms.append(
            1000.0 * mad_samples / sampling_rate_hz
        )
    timing_mad_ms = (
        sum(per_field_mad_ms) / len(per_field_mad_ms)
        if per_field_mad_ms
        else None
    )
    completeness = observed / possible if possible else 0.0
    confidence_values = [
        float(item.confidence)
        for item in fiducials_by_lead.values()
        if math.isfinite(float(item.confidence))
    ]
    return {
        "lead_count": len(fiducials_by_lead),
        "fiducial_completeness": completeness,
        "cross_lead_timing_mad_ms": timing_mad_ms,
        "mean_delineation_confidence": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),
        "cross_lead_consistent": (
            int(timing_mad_ms <= 20.0)
            if timing_mad_ms is not None and completeness >= 0.50
            else 0
        ),
    }


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _complete_triplet(values: tuple[float | None, float | None, float | None]) -> bool:
    present = [_finite(value) for value in values]
    return not any(present) or all(present)


def verify_fiducial_order(fiducials: BeatFiducials) -> bool:
    """Validate complete wave triplets and strict physiological ordering.

    A wholly absent P or T wave is allowed at this structural-validation layer;
    :func:`accept_beat` applies the stricter measurement requirements.  Partially
    emitted wave triplets are always invalid.
    """

    p = (fiducials.p_on, fiducials.p_peak, fiducials.p_off)
    qrs = (fiducials.qrs_on, fiducials.r_peak, fiducials.qrs_off)
    t = (fiducials.t_on, fiducials.t_peak, fiducials.t_off)
    if not all(_complete_triplet(group) for group in (p, qrs, t)):
        return False

    complete_groups = [group for group in (p, qrs, t) if all(_finite(v) for v in group)]
    if any(not (float(group[0]) < float(group[1]) < float(group[2])) for group in complete_groups):
        return False
    if all(_finite(value) for value in p + qrs) and float(p[2]) >= float(qrs[0]):
        return False
    if all(_finite(value) for value in qrs + t) and float(qrs[2]) >= float(t[0]):
        return False
    return True


def _robust_scale(values, np) -> float:  # type: ignore[no-untyped-def]
    if values.size == 0:
        return 0.0
    centre = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - centre)))


def _smooth_detrended(segment, sampling_rate_hz: float, smooth_ms: float, np, sp_signal):  # type: ignore[no-untyped-def]
    """Remove a linear local baseline and apply deterministic Savitzky-Golay smoothing."""

    values = np.asarray(segment, dtype=float)
    if values.size < 5 or not np.all(np.isfinite(values)):
        return None
    detrended = sp_signal.detrend(values, type="linear")
    window = max(5, int(round(sampling_rate_hz * smooth_ms / 1000.0)))
    if window % 2 == 0:
        window += 1
    if window >= values.size:
        window = values.size - 1 if values.size % 2 == 0 else values.size
    if window < 5:
        return detrended
    return sp_signal.savgol_filter(detrended, window_length=window, polyorder=2, mode="interp")


def _amplitude_wave_bounds(
    signal,
    start: int,
    stop: int,
    sampling_rate_hz: float,
    *,
    minimum_amplitude: float,
    minimum_duration_ms: float,
    maximum_duration_ms: float,
    smooth_ms: float,
    np,
    sp_signal,
) -> _WaveBounds | None:  # type: ignore[no-untyped-def]
    """Delineate a P/T-like wave from prominence and data-derived width."""

    if stop - start < max(7, int(0.04 * sampling_rate_hz)):
        return None
    segment = np.asarray(signal[start:stop], dtype=float)
    smoothed = _smooth_detrended(segment, sampling_rate_hz, smooth_ms, np, sp_signal)
    if smoothed is None:
        return None
    envelope = np.abs(smoothed)
    detrended = sp_signal.detrend(segment, type="linear")
    # The smoothing residual estimates high-frequency noise without treating the
    # tails of a broad physiological wave as baseline noise.
    noise = _robust_scale(detrended - smoothed, np)
    detection_floor = max(float(minimum_amplitude), 4.0 * noise)

    candidates, properties = sp_signal.find_peaks(
        envelope,
        prominence=max(0.5 * minimum_amplitude, 3.0 * noise),
        distance=max(1, int(0.025 * sampling_rate_hz)),
    )
    if candidates.size == 0:
        return None
    prominences = properties["prominences"]
    best_position = int(np.argmax(prominences))
    peak = int(candidates[best_position])
    prominence = float(prominences[best_position])
    if float(envelope[peak]) < detection_floor or prominence < 0.5 * detection_floor:
        return None

    widths, _height, left_ips, right_ips = sp_signal.peak_widths(envelope, [peak], rel_height=0.80)
    duration_ms = 1000.0 * float(widths[0]) / sampling_rate_hz
    if not minimum_duration_ms <= duration_ms <= maximum_duration_ms:
        return None
    onset = start + int(math.floor(float(left_ips[0])))
    offset = start + int(math.ceil(float(right_ips[0])))
    absolute_peak = start + peak
    if not onset < absolute_peak < offset:
        return None

    evidence_ratio = prominence / max(detection_floor, 1e-12)
    confidence = max(0.0, min(1.0, evidence_ratio / 3.0))
    return _WaveBounds(onset, absolute_peak, offset, confidence)


def _qrs_bounds(signal, r_peak: int, sampling_rate_hz: float, np, sp_signal) -> _WaveBounds | None:  # type: ignore[no-untyped-def]
    """Delineate QRS from a short-window derivative-energy component."""

    start = max(0, r_peak - int(round(0.16 * sampling_rate_hz)))
    stop = min(len(signal), r_peak + int(round(0.20 * sampling_rate_hz)) + 1)
    smoothed = _smooth_detrended(
        np.asarray(signal[start:stop], dtype=float), sampling_rate_hz, 5.0, np, sp_signal
    )
    if smoothed is None:
        return None

    local_r = r_peak - start
    refine_radius = max(1, int(round(0.05 * sampling_rate_hz)))
    refine_start = max(0, local_r - refine_radius)
    refine_stop = min(smoothed.size, local_r + refine_radius + 1)
    refined_r = refine_start + int(np.argmax(np.abs(smoothed[refine_start:refine_stop])))

    gradient = np.gradient(smoothed)
    integration_samples = max(3, int(round(0.012 * sampling_rate_hz)))
    kernel = np.ones(integration_samples, dtype=float) / integration_samples
    energy = np.convolve(gradient * gradient, kernel, mode="same")
    edge_width = max(2, min(energy.size // 6, int(round(0.04 * sampling_rate_hz))))
    edge_energy = np.concatenate((energy[:edge_width], energy[-edge_width:]))
    energy_floor = float(np.median(edge_energy))
    energy_noise = _robust_scale(edge_energy, np)
    peak_energy = float(np.max(energy))
    threshold = energy_floor + max(6.0 * energy_noise, 0.04 * (peak_energy - energy_floor))
    active = np.flatnonzero(energy >= threshold)
    if active.size < 2 or peak_energy <= energy_floor:
        return None

    max_gap = max(2, int(round(0.07 * sampling_rate_hz)))
    groups: list[list[int]] = [[int(active[0])]]
    for index in active[1:]:
        value = int(index)
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    containing = [
        group
        for group in groups
        if group[0] <= refined_r <= group[-1]
        or min(abs(group[0] - refined_r), abs(group[-1] - refined_r))
        <= int(0.03 * sampling_rate_hz)
    ]
    if not containing:
        return None
    selected = max(containing, key=lambda group: float(np.max(energy[group[0] : group[-1] + 1])))
    padding = max(1, int(round(0.008 * sampling_rate_hz)))
    onset_local = max(0, selected[0] - padding)
    offset_local = min(smoothed.size - 1, selected[-1] + padding)
    if not onset_local < refined_r < offset_local:
        return None

    duration_ms = 1000.0 * (offset_local - onset_local) / sampling_rate_hz
    if not 35.0 <= duration_ms <= 220.0:
        return None
    edge_signal = np.concatenate((smoothed[:edge_width], smoothed[-edge_width:]))
    amplitude_noise = _robust_scale(edge_signal, np)
    qrs_amplitude = float(np.max(smoothed[onset_local : offset_local + 1])) - float(
        np.min(smoothed[onset_local : offset_local + 1])
    )
    amplitude_floor = max(0.05, 5.0 * amplitude_noise)
    if qrs_amplitude < amplitude_floor:
        return None

    amplitude_evidence = qrs_amplitude / amplitude_floor
    energy_evidence = (peak_energy - energy_floor) / max(energy_noise, peak_energy * 0.01, 1e-12)
    confidence = max(0.0, min(1.0, min(amplitude_evidence / 4.0, energy_evidence / 10.0)))
    return _WaveBounds(
        start + onset_local,
        start + refined_r,
        start + offset_local,
        confidence,
    )


def delineate_beat(
    signal,
    r_peak: int,
    sampling_rate_hz: float,
    beat_id: str,
    record_id: str,
    *,
    previous_r_peak: int | None = None,
    next_r_peak: int | None = None,
) -> BeatDelineation:  # type: ignore[no-untyped-def]
    """Derive P/QRS/T timing from one filtered ECG lead.

    QRS and T are required for a measurement-eligible beat.  P is optional because
    genuine atrial-wave absence must remain representable; downstream atrial
    features are marked unavailable when no defensible P boundary exists.
    """

    try:
        import numpy as np  # type: ignore
        from scipy import signal as sp_signal  # type: ignore
    except ImportError as exc:  # pragma: no cover - package dependencies are locked
        raise RuntimeError("delineate_beat requires numpy and scipy") from exc

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 5 or sampling_rate_hz <= 0:
        raise ValueError(
            "signal must be a non-empty one-dimensional array and sampling rate positive"
        )
    if r_peak <= 0 or r_peak >= values.size - 1:
        raise ValueError("r_peak must be inside the signal")

    qrs = _qrs_bounds(values, int(r_peak), sampling_rate_hz, np, sp_signal)
    reasons: list[str] = []
    if qrs is None:
        reasons.append("qrs_unavailable")
        fiducials = BeatFiducials(
            beat_id=beat_id,
            record_id=record_id,
            r_peak=float(r_peak),
            source=DELINEATION_METHOD,
            confidence=0.0,
        )
        return BeatDelineation(fiducials, 0.0, 0.0, 0.0, tuple(reasons))

    prior_rr = r_peak - previous_r_peak if previous_r_peak is not None else None
    p_horizon = min(
        int(round(0.32 * sampling_rate_hz)),
        int(round(0.40 * prior_rr))
        if prior_rr is not None and prior_rr > 0
        else int(0.32 * sampling_rate_hz),
    )
    p_start = max(0, qrs.onset - p_horizon)
    p_stop = max(p_start, qrs.onset - int(round(0.035 * sampling_rate_hz)))
    p = _amplitude_wave_bounds(
        values,
        p_start,
        p_stop,
        sampling_rate_hz,
        minimum_amplitude=0.015,
        minimum_duration_ms=25.0,
        maximum_duration_ms=180.0,
        smooth_ms=15.0,
        np=np,
        sp_signal=sp_signal,
    )
    if p is None:
        reasons.append("p_unavailable")

    following_rr = next_r_peak - r_peak if next_r_peak is not None else None
    t_horizon = min(
        int(round(0.52 * sampling_rate_hz)),
        int(round(0.60 * following_rr))
        if following_rr is not None and following_rr > 0
        else int(round(0.52 * sampling_rate_hz)),
    )
    t_start = min(values.size, qrs.offset + int(round(0.035 * sampling_rate_hz)))
    t_stop = min(values.size, r_peak + t_horizon)
    if next_r_peak is not None:
        t_stop = min(t_stop, next_r_peak - int(round(0.10 * sampling_rate_hz)))
    t = _amplitude_wave_bounds(
        values,
        t_start,
        t_stop,
        sampling_rate_hz,
        minimum_amplitude=0.02,
        minimum_duration_ms=60.0,
        maximum_duration_ms=450.0,
        smooth_ms=25.0,
        np=np,
        sp_signal=sp_signal,
    )
    if t is None:
        reasons.append("t_unavailable")

    overall_confidence = min(qrs.confidence, t.confidence if t is not None else 0.0)
    fiducials = BeatFiducials(
        beat_id=beat_id,
        record_id=record_id,
        p_on=float(p.onset) if p is not None else None,
        p_peak=float(p.peak) if p is not None else None,
        p_off=float(p.offset) if p is not None else None,
        qrs_on=float(qrs.onset),
        r_peak=float(qrs.peak),
        qrs_off=float(qrs.offset),
        t_on=float(t.onset) if t is not None else None,
        t_peak=float(t.peak) if t is not None else None,
        t_off=float(t.offset) if t is not None else None,
        source=DELINEATION_METHOD,
        confidence=overall_confidence,
    )
    if not verify_fiducial_order(fiducials):
        reasons.append("invalid_fiducial_order")
        fiducials = BeatFiducials(
            beat_id=beat_id,
            record_id=record_id,
            r_peak=float(r_peak),
            source=DELINEATION_METHOD,
            confidence=0.0,
        )
        return BeatDelineation(fiducials, 0.0, 0.0, 0.0, tuple(dict.fromkeys(reasons)))
    return BeatDelineation(
        fiducials,
        p.confidence if p is not None else 0.0,
        qrs.confidence,
        t.confidence if t is not None else 0.0,
        tuple(reasons),
    )


def detect_secondary_r_peaks(signal, sampling_rate_hz: float) -> list[int]:  # type: ignore[no-untyped-def]
    """Detect R peaks independently using band-pass energy and prominence."""

    try:
        import numpy as np  # type: ignore
        from scipy import signal as sp_signal  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("detect_secondary_r_peaks requires numpy and scipy") from exc

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or sampling_rate_hz <= 0 or values.size < int(sampling_rate_hz):
        return []
    if not np.all(np.isfinite(values)):
        return []
    nyquist = sampling_rate_hz / 2.0
    lower = 5.0 / nyquist
    upper = min(25.0 / nyquist, 0.95)
    if lower >= upper:
        return []
    sos = sp_signal.butter(3, [lower, upper], btype="bandpass", output="sos")
    try:
        qrs_band = sp_signal.sosfiltfilt(sos, values)
    except ValueError:
        return []
    derivative = np.diff(qrs_band, prepend=qrs_band[0])
    integration_samples = max(3, int(round(0.08 * sampling_rate_hz)))
    integrated = np.convolve(
        derivative * derivative,
        np.ones(integration_samples, dtype=float) / integration_samples,
        mode="same",
    )
    centre = float(np.median(integrated))
    scale = _robust_scale(integrated, np)
    prominence = max(6.0 * scale, 0.10 * max(0.0, float(np.percentile(integrated, 95)) - centre))
    if prominence <= 0:
        return []
    candidates, _ = sp_signal.find_peaks(
        integrated,
        distance=max(1, int(round(0.25 * sampling_rate_hz))),
        prominence=prominence,
    )
    refinement = max(1, int(round(0.10 * sampling_rate_hz)))
    refined: list[int] = []
    for candidate in candidates:
        left = max(0, int(candidate) - refinement)
        right = min(values.size, int(candidate) + refinement + 1)
        peak = left + int(np.argmax(np.abs(qrs_band[left:right])))
        if refined and peak - refined[-1] < int(round(0.20 * sampling_rate_hz)):
            if abs(qrs_band[peak]) > abs(qrs_band[refined[-1]]):
                refined[-1] = peak
            continue
        refined.append(peak)
    return refined


def match_r_peaks(
    primary_peaks: Sequence[int],
    secondary_peaks: Sequence[int],
    tolerance_samples: int,
) -> RDetectorAgreement:
    """Match sorted peak trains one-to-one and report their symmetric F1 agreement."""

    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative")
    primary = sorted(int(value) for value in primary_peaks)
    secondary = sorted(int(value) for value in secondary_peaks)
    if not primary and not secondary:
        return RDetectorAgreement(None, (), (), None, None)

    matches: list[tuple[int, int]] = []
    matched_primary: list[int] = []
    primary_index = 0
    secondary_index = 0
    while primary_index < len(primary) and secondary_index < len(secondary):
        delta = primary[primary_index] - secondary[secondary_index]
        if abs(delta) <= tolerance_samples:
            matches.append((primary[primary_index], secondary[secondary_index]))
            matched_primary.append(primary_index)
            primary_index += 1
            secondary_index += 1
        elif delta < 0:
            primary_index += 1
        else:
            secondary_index += 1

    match_count = len(matches)
    primary_fraction = match_count / len(primary) if primary else None
    secondary_fraction = match_count / len(secondary) if secondary else None
    score = 2.0 * match_count / (len(primary) + len(secondary))
    return RDetectorAgreement(
        score,
        tuple(matches),
        tuple(matched_primary),
        primary_fraction,
        secondary_fraction,
    )


def analyzable_duration_from_beats(
    r_peaks: Sequence[int],
    quality_valid: Sequence[bool],
    sampling_rate_hz: float,
    *,
    minimum_rr_s: float = 0.25,
    maximum_rr_s: float = 2.5,
) -> float:
    """Sum only intervals bounded by adjacent accepted, quality-valid beats."""

    if len(r_peaks) != len(quality_valid):
        raise ValueError("r_peaks and quality_valid must have the same length")
    if sampling_rate_hz <= 0 or minimum_rr_s <= 0 or maximum_rr_s <= minimum_rr_s:
        raise ValueError("invalid sampling rate or analyzable RR bounds")
    total = 0.0
    for index in range(len(r_peaks) - 1):
        if not (quality_valid[index] and quality_valid[index + 1]):
            continue
        duration = (int(r_peaks[index + 1]) - int(r_peaks[index])) / sampling_rate_hz
        if minimum_rr_s <= duration <= maximum_rr_s:
            total += duration
    return float(total)


def accept_beat(
    fiducials: BeatFiducials,
    lead_quality_db: float,
    delineation_confidence: float,
    pacing_contaminated: bool,
    *,
    minimum_lead_quality_db: float | None = None,
    minimum_delineation_confidence: float | None = None,
    r_detector_matched: bool = True,
) -> BeatAcceptance:
    """Fail closed for absent required waves or inadequate extraction evidence."""

    reasons: list[str] = []
    if not verify_fiducial_order(fiducials):
        reasons.append("invalid_fiducial_order")
    required = (
        fiducials.qrs_on,
        fiducials.r_peak,
        fiducials.qrs_off,
        fiducials.t_on,
        fiducials.t_peak,
        fiducials.t_off,
    )
    if not all(_finite(value) for value in required):
        reasons.append("required_fiducials_unavailable")
    if minimum_lead_quality_db is not None and (
        not math.isfinite(float(lead_quality_db)) or lead_quality_db < minimum_lead_quality_db
    ):
        reasons.append("lead_quality_below_minimum")
    if minimum_delineation_confidence is not None and (
        not math.isfinite(float(delineation_confidence))
        or delineation_confidence < minimum_delineation_confidence
    ):
        reasons.append("delineation_confidence_below_minimum")
    if not r_detector_matched:
        reasons.append("r_detector_disagreement")
    if pacing_contaminated:
        reasons.append("pacing_contamination")
    accepted = not reasons
    fiducial_values = (
        fiducials.p_on,
        fiducials.p_peak,
        fiducials.p_off,
        fiducials.qrs_on,
        fiducials.r_peak,
        fiducials.qrs_off,
        fiducials.t_on,
        fiducials.t_peak,
        fiducials.t_off,
    )
    return BeatAcceptance(
        beat_id=fiducials.beat_id,
        record_id=fiducials.record_id,
        accepted=accepted,
        reasons=reasons or ["accepted"],
        lead_quality_db=lead_quality_db,
        fiducial_completeness=sum(_finite(value) for value in fiducial_values) / 9.0,
        pacing_corrected=not pacing_contaminated,
    )
