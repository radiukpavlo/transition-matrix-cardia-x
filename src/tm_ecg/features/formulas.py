"""Pure-Python implementations of the locked feature formulas."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean, median, pstdev

from tm_ecg.constants import QT_CORRECTION_CODE
from tm_ecg.types import FeatureQualityState


def _clean(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _median(values: list[float | None]) -> float | None:
    cleaned = _clean(values)
    return median(cleaned) if cleaned else None


def _iqr(values: list[float | None]) -> float | None:
    cleaned = sorted(_clean(values))
    if len(cleaned) < 4:
        return None
    lower_idx = int(0.25 * (len(cleaned) - 1))
    upper_idx = int(0.75 * (len(cleaned) - 1))
    return cleaned[upper_idx] - cleaned[lower_idx]


def _std(values: list[float | None]) -> float | None:
    cleaned = _clean(values)
    return pstdev(cleaned) if len(cleaned) >= 2 else None


def median_angle_deg(angles_deg: list[float]) -> float | None:
    if not angles_deg:
        return None
    sin_med = median([math.sin(math.radians(angle)) for angle in angles_deg])
    cos_med = median([math.cos(math.radians(angle)) for angle in angles_deg])
    return math.degrees(math.atan2(sin_med, cos_med))


def st_offset_seconds(hr_bpm: float) -> float:
    return 0.06 if hr_bpm >= 100.0 else 0.08


@dataclass(slots=True)
class BeatMeasurement:
    beat_id: str
    rr_s: float | None = None
    rr_prev_s: float | None = None
    rr_next_s: float | None = None
    p_present: bool = False
    p_amp_ii_mv: float | None = None
    p_dur_ms: float | None = None
    pr_ms: float | None = None
    q_amp_ii_mv: float | None = None
    r_amp_ii_mv: float | None = None
    s_amp_ii_mv: float | None = None
    qrs_dur_ms: float | None = None
    qrs_deformed_prob: float | None = None
    qrs_secondary_extrema: int = 0
    r_prime_v1: bool = False
    broad_r_v6: bool = False
    st_level_v1_mv: float | None = None
    st_level_v5_mv: float | None = None
    st_slope_v5_uv_per_ms: float | None = None
    t_amp_v5_mv: float | None = None
    t_dur_ms: float | None = None
    t_amp_right_mv: float | None = None
    t_negative_duration_ms: float | None = None
    qt_ms: float | None = None
    qrs_net_area_i_mv_ms: float | None = None
    qrs_net_area_avf_mv_ms: float | None = None
    lead_quality_db: float | None = None
    delineation_confidence: float | None = None
    u_present_v2: bool | None = None
    u_amp_v2_mv: float | None = None
    pvc_like: bool = False
    apb_like: bool = False
    paced_like: bool = False
    is_ectopic: bool = False
    is_paced: bool = False
    is_artifact: bool = False
    rhythm_valid: bool = True
    atrial_valid: bool = True
    qrs_valid: bool = True
    st_t_valid: bool = True
    detector_agreement: float | None = None
    prematurity_index: float | None = None
    compensatory_pause_ratio: float | None = None
    qrs_morphology_distance: float | None = None
    preceding_p_wave_probability: float | None = None
    axis_concordance: float | None = None
    beat_template_cluster: str | None = None


@dataclass(slots=True)
class RecordMeasurements:
    record_id: str
    beats: list[BeatMeasurement] = field(default_factory=list)
    tq_power_ratios: list[float] = field(default_factory=list)
    sampling_rate_hz: float = 500.0
    qrs_def_threshold: float = 0.5
    lead_quality_by_lead_db: dict[str, float] = field(default_factory=dict)
    analyzable_duration_s: float | None = None


def _nn_intervals(beats: list[BeatMeasurement]) -> list[float]:
    values = []
    for beat in beats:
        if beat.rr_s is None:
            continue
        if beat.is_ectopic or beat.is_paced or beat.is_artifact or not beat.rhythm_valid:
            continue
        values.append(beat.rr_s)
    return values


def _fraction(beats: list[BeatMeasurement], attribute: str) -> float | None:
    if not beats:
        return None
    return sum(bool(getattr(beat, attribute)) for beat in beats) / len(beats)


def _lead_coverage(record: RecordMeasurements, leads: tuple[str, ...], threshold: float) -> float:
    if not leads:
        return 0.0
    quality_by_normalized_lead = {
        str(name).strip().lower(): value
        for name, value in record.lead_quality_by_lead_db.items()
    }
    available = sum(
        quality_by_normalized_lead.get(lead.lower(), float("-inf")) >= threshold
        for lead in leads
    )
    return available / len(leads)


def compute_feature_quality_states(
    record: RecordMeasurements,
    thresholds: dict[str, object],
) -> dict[str, FeatureQualityState]:
    minimum_beats = int(thresholds.get("minimum_valid_beats", 5))
    minimum_fraction = float(thresholds.get("minimum_analyzable_fraction", 0.5))
    quality_threshold = float(thresholds.get("feature_quality_min_db", 5.0))
    detector_minimum = float(thresholds.get("detector_agreement_minimum", 0.5))
    family_spec = {
        "rhythm": ("rhythm_valid", ("II",), float(thresholds.get("minimum_rhythm_lead_coverage", 1.0))),
        "atrial": ("atrial_valid", ("I", "II", "V1"), float(thresholds.get("minimum_p_wave_lead_coverage", 0.67))),
        "qrs": ("qrs_valid", ("I", "V1", "V2", "V5", "V6"), float(thresholds.get("minimum_qrs_lead_coverage", 0.6))),
        "st_t": ("st_t_valid", ("I", "II", "V1", "V5", "V6"), float(thresholds.get("minimum_st_t_lead_coverage", 0.6))),
    }
    agreement = _median([beat.detector_agreement for beat in record.beats])
    states: dict[str, FeatureQualityState] = {}
    for family, (attribute, leads, minimum_coverage) in family_spec.items():
        fraction = _fraction(record.beats, attribute)
        valid_count = sum(bool(getattr(beat, attribute)) for beat in record.beats)
        coverage = _lead_coverage(record, leads, quality_threshold)
        reasons: list[str] = []
        if valid_count < minimum_beats:
            reasons.append(f"valid_beats<{minimum_beats}")
        if fraction is None or fraction < minimum_fraction:
            reasons.append(f"valid_fraction<{minimum_fraction:.2f}")
        if coverage < minimum_coverage:
            reasons.append(f"lead_coverage<{minimum_coverage:.2f}")
        if agreement is None or agreement < detector_minimum:
            reasons.append(f"detector_agreement<{detector_minimum:.2f}")
        if valid_count == 0 or coverage == 0:
            state = "unavailable"
        elif reasons:
            state = "low_confidence"
        else:
            state = "observed"
        states[family] = FeatureQualityState(
            state=state,
            valid_beat_fraction=fraction,
            lead_coverage=coverage,
            detector_agreement=agreement,
            reasons=tuple(reasons),
        )
    return states


def _qtc_fridericia_ms(qt_ms: float | None, rr_s: float | None) -> float | None:
    if qt_ms is None or rr_s is None or rr_s <= 0:
        return None
    return 1000.0 * ((qt_ms / 1000.0) / (rr_s ** (1.0 / 3.0)))


def compute_record_features(record: RecordMeasurements, thresholds: dict[str, object]) -> dict[str, float | int | None]:
    beats = record.beats
    quality_states = compute_feature_quality_states(record, thresholds)
    rhythm_beats = [beat for beat in beats if beat.rhythm_valid]
    atrial_beats = [beat for beat in beats if beat.atrial_valid]
    qrs_beats = [beat for beat in beats if beat.qrs_valid]
    st_t_beats = [beat for beat in beats if beat.st_t_valid]
    rr_values = [beat.rr_s for beat in rhythm_beats]
    hr_values = [60.0 / rr for rr in _clean(rr_values) if rr > 0]
    nn_values = _nn_intervals(beats)
    rr_n = median(nn_values) if nn_values else None

    prematurity = []
    comp_pause = []
    for beat in rhythm_beats:
        if rr_n and beat.rr_prev_s is not None:
            prematurity.append(beat.rr_prev_s / rr_n)
        if rr_n and beat.is_ectopic and beat.rr_prev_s is not None and beat.rr_next_s is not None:
            comp_pause.append((beat.rr_prev_s + beat.rr_next_s) / (2.0 * rr_n))

    qrs_angles = []
    for beat in qrs_beats:
        if beat.qrs_net_area_i_mv_ms is None or beat.qrs_net_area_avf_mv_ms is None:
            continue
        qrs_angles.append(math.degrees(math.atan2(beat.qrs_net_area_avf_mv_ms, beat.qrs_net_area_i_mv_ms)))

    qtc_values = [_qtc_fridericia_ms(beat.qt_ms, beat.rr_s) for beat in st_t_beats]
    axis_deg = median_angle_deg(qrs_angles)

    features: dict[str, float | int | None] = {
        "hr_med_bpm": _median(hr_values),
        "rr_med_ms": None if _median(rr_values) is None else 1000.0 * _median(rr_values),  # type: ignore[arg-type]
        "rr_iqr_ms": None if _iqr(rr_values) is None else 1000.0 * _iqr(rr_values),  # type: ignore[arg-type]
        "rr_sdnn_ms": None if _std(nn_values) is None or len(nn_values) < 5 else 1000.0 * _std(nn_values),  # type: ignore[arg-type]
        "prematurity_index_min": min(prematurity) if prematurity else None,
        "comp_pause_ratio_max": max(comp_pause) if comp_pause else None,
        "pvc_like_beat_count": sum(1 for beat in rhythm_beats if beat.pvc_like),
        "apb_like_beat_count": sum(1 for beat in rhythm_beats if beat.apb_like),
        "paced_like_beat_count": sum(1 for beat in rhythm_beats if beat.paced_like),
        "af_irregularity_cv": None if not nn_values or mean(nn_values) == 0 or len(nn_values) < 2 else pstdev(nn_values) / mean(nn_values),
        "f_wave_power_ratio": _median(record.tq_power_ratios),
        "p_present_ratio": (sum(1 for beat in atrial_beats if beat.p_present) / len(atrial_beats)) if atrial_beats else None,
        "p_amp_ii_med_mV": _median([beat.p_amp_ii_mv for beat in atrial_beats]),
        "p_dur_med_ms": _median([beat.p_dur_ms for beat in atrial_beats]),
        "pr_med_ms": _median([beat.pr_ms for beat in atrial_beats]),
        "pr_iqr_ms": _iqr([beat.pr_ms for beat in atrial_beats]),
        "q_amp_ii_med_mV": _median([beat.q_amp_ii_mv for beat in qrs_beats]),
        "r_amp_ii_med_mV": _median([beat.r_amp_ii_mv for beat in qrs_beats]),
        "s_amp_ii_med_mV": _median([beat.s_amp_ii_mv for beat in qrs_beats]),
        "qrs_dur_med_ms": _median([beat.qrs_dur_ms for beat in qrs_beats]),
        "qrs_dur_iqr_ms": _iqr([beat.qrs_dur_ms for beat in qrs_beats]),
        "qrs_deformed_prob": _median([beat.qrs_deformed_prob for beat in qrs_beats]),
        "qrs_deformed_any": None if not qrs_beats else int(any((beat.qrs_deformed_prob or 0.0) >= record.qrs_def_threshold for beat in qrs_beats)),
        "qrs_fragmented_any": None if not qrs_beats else int(any(beat.qrs_secondary_extrema >= 2 for beat in qrs_beats)),
        "qrs_wide_any": None if not qrs_beats else int(any((beat.qrs_dur_ms or 0.0) >= float(thresholds["qrs_wide_ms"]) for beat in qrs_beats)),
        "r_prime_v1_any": None if not qrs_beats else int(any(beat.r_prime_v1 for beat in qrs_beats)),
        "broad_r_v6_any": None if not qrs_beats else int(any(beat.broad_r_v6 for beat in qrs_beats)),
        "st_level_v1_mV": _median([beat.st_level_v1_mv for beat in st_t_beats]),
        "st_level_v5_mV": _median([beat.st_level_v5_mv for beat in st_t_beats]),
        "st_slope_v5_uV_per_ms": _median([beat.st_slope_v5_uv_per_ms for beat in st_t_beats]),
        "t_amp_v5_med_mV": _median([beat.t_amp_v5_mv for beat in st_t_beats]),
        "t_dur_med_ms": _median([beat.t_dur_ms for beat in st_t_beats]),
        "t_inverted_right_any": int(
            any(
                (beat.t_amp_right_mv is not None and beat.t_amp_right_mv <= float(thresholds["t_inverted_threshold_mv"]))
                and (beat.t_negative_duration_ms is not None and beat.t_negative_duration_ms >= float(thresholds["t_inverted_duration_ms"]))
                for beat in st_t_beats
            )
        ),
        "qt_med_ms": _median([beat.qt_ms for beat in st_t_beats]),
        "qtc_med_ms": _median(qtc_values),
        "qrs_net_area_i_mV_ms": _median([beat.qrs_net_area_i_mv_ms for beat in qrs_beats]),
        "qrs_net_area_avf_mV_ms": _median([beat.qrs_net_area_avf_mv_ms for beat in qrs_beats]),
        "qrs_axis_deg": axis_deg,
        "qrs_axis_sin": None if axis_deg is None else math.sin(math.radians(axis_deg)),
        "qrs_axis_cos": None if axis_deg is None else math.cos(math.radians(axis_deg)),
        "lead_quality_min_db": min(_clean([beat.lead_quality_db for beat in beats])) if _clean([beat.lead_quality_db for beat in beats]) else None,
        "delineation_confidence": _median([beat.delineation_confidence for beat in beats]),
        "rhythm_valid_beat_fraction": quality_states["rhythm"].valid_beat_fraction,
        "atrial_valid_beat_fraction": quality_states["atrial"].valid_beat_fraction,
        "qrs_valid_beat_fraction": quality_states["qrs"].valid_beat_fraction,
        "st_t_valid_beat_fraction": quality_states["st_t"].valid_beat_fraction,
        "atrial_lead_coverage": quality_states["atrial"].lead_coverage,
        "qrs_lead_coverage": quality_states["qrs"].lead_coverage,
        "st_t_lead_coverage": quality_states["st_t"].lead_coverage,
        "detector_agreement": quality_states["rhythm"].detector_agreement,
        "analyzable_duration_s": record.analyzable_duration_s,
        "u_present_v2_any": None if record.sampling_rate_hz < 500 else int(any(bool(beat.u_present_v2) for beat in beats if beat.u_present_v2 is not None)),
        "u_amp_v2_mV": None if record.sampling_rate_hz < 500 else _median([beat.u_amp_v2_mv for beat in beats]),
    }
    features["qtc_formula_code"] = QT_CORRECTION_CODE
    return features
