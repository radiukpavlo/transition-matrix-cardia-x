"""Beat-level measurement and ECG extraction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import os
from dataclasses import asdict
from typing import Any

from tm_ecg.config import ProjectConfig
from tm_ecg.features.formulas import BeatMeasurement
from tm_ecg.modeling.triads import build_triad_memberships
from tm_ecg.signal.filtering import preprocess_signal
from tm_ecg.signal.rpeaks import detect_r_peaks
from tm_ecg.signal.fiducials import (
    DELINEATION_METHOD,
    SECONDARY_R_DETECTOR,
    accept_beat,
    analyzable_duration_from_beats,
    delineate_beat,
    detect_secondary_r_peaks,
    match_r_peaks,
)
from tm_ecg.types import BeatAcceptance
from tm_ecg.io.wfdb_loader import _runtime, _parse_labels, split_entries, _load_record, _lead_index


def _window(signal, center: int, left: int, right: int, np):
    start = center - left
    end = center + right
    width = left + right
    if start >= 0 and end <= signal.shape[0]:
        return signal[start:end]
    output = np.zeros((width, signal.shape[1]), dtype=np.float32)
    valid_start = max(0, start)
    valid_end = min(signal.shape[0], end)
    out_start = valid_start - start
    out_end = out_start + (valid_end - valid_start)
    output[out_start:out_end] = signal[valid_start:valid_end]
    return output


def _count_secondary_extrema(values, np) -> int:
    if values.size < 5:
        return 0
    derivative = np.diff(values)
    signs = np.sign(derivative)
    return int(np.sum(np.abs(np.diff(signs)) > 1))


def _lead_quality(values, peaks: list[int], np, sampling_rate_hz: float = 500.0) -> float:
    """Estimate robust QRS-to-baseline SNR in dB for one lead."""

    if not peaks:
        return 0.0
    samples = np.asarray(values, dtype=float)
    qrs_half_width = max(1, int(round(0.08 * sampling_rate_hz)))
    rr_samples = np.diff(sorted(int(peak) for peak in peaks))
    median_rr = float(np.median(rr_samples[rr_samples > 0])) if np.any(rr_samples > 0) else 0.0
    adaptive_exclusion = int(round(0.35 * median_rr)) if median_rr > 0 else qrs_half_width
    exclusion_half_width = max(
        qrs_half_width,
        min(int(round(0.20 * sampling_rate_hz)), adaptive_exclusion),
    )
    qrs_amplitudes: list[float] = []
    baseline_mask = np.ones(samples.size, dtype=bool)
    for peak in peaks:
        qrs_start = max(0, int(peak) - qrs_half_width)
        qrs_stop = min(samples.size, int(peak) + qrs_half_width + 1)
        if qrs_stop > qrs_start:
            segment = samples[qrs_start:qrs_stop]
            qrs_amplitudes.append(float(np.max(segment) - np.min(segment)))
        exclude_start = max(0, int(peak) - exclusion_half_width)
        exclude_stop = min(samples.size, int(peak) + exclusion_half_width + 1)
        baseline_mask[exclude_start:exclude_stop] = False
    if not qrs_amplitudes or not np.any(baseline_mask):
        return 0.0
    baseline = samples[baseline_mask]
    baseline_centre = float(np.median(baseline))
    baseline_noise = float(1.4826 * np.median(np.abs(baseline - baseline_centre)))
    qrs_signal = float(np.median(qrs_amplitudes))
    if qrs_signal <= 0.0:
        return 0.0
    return float(max(-20.0, min(60.0, 20.0 * math.log10(qrs_signal / max(baseline_noise, 1e-6)))))


def _st_offset_samples(rr_s: float | None, fs: float) -> int:
    hr = 60.0 / rr_s if rr_s and rr_s > 0 else 0.0
    return int((0.06 if hr >= 100.0 else 0.08) * fs)


def _strict_triad_rows(record_id: str, acceptances: list[BeatAcceptance]) -> list[dict[str, Any]]:
    """Build triads only from three adjacent accepted detector-consensus beats."""

    acceptance_positions = {item.beat_id: index for index, item in enumerate(acceptances)}
    return [
        item.to_dict()
        for item in build_triad_memberships(record_id, acceptances)
        if acceptance_positions[item.previous_beat_id] + 1
        == acceptance_positions[item.current_beat_id]
        and acceptance_positions[item.current_beat_id] + 1
        == acceptance_positions[item.next_beat_id]
    ]


def _one_record_measurements(  # noqa: PLR0915
    signal,
    fs: float,
    sig_names: list[str],
    record_id: str,
    config: ProjectConfig,
):
    np, _torch, _wfdb, sp_signal = _runtime()
    diagnostic = preprocess_signal(signal, fs, config.filters["diagnostic"])
    lead_ii = _lead_index(sig_names, "II")
    lead_i = _lead_index(sig_names, "I")
    lead_avf = _lead_index(sig_names, "aVF")
    lead_v1 = _lead_index(sig_names, "V1")
    lead_v2 = _lead_index(sig_names, "V2")
    lead_v3 = _lead_index(sig_names, "V3")
    lead_v5 = _lead_index(sig_names, "V5")

    primary_peaks, _meta = detect_r_peaks(diagnostic[:, lead_ii], fs)
    primary_peaks = sorted(
        {
            int(peak)
            for peak in primary_peaks
            if int(0.25 * fs) <= int(peak) < signal.shape[0] - int(0.45 * fs)
        }
    )
    secondary_peaks = sorted(
        {
            int(peak)
            for peak in detect_secondary_r_peaks(diagnostic[:, lead_ii], fs)
            if int(0.25 * fs) <= int(peak) < signal.shape[0] - int(0.45 * fs)
        }
    )
    tolerance_ms = float(config.thresholds.get("r_peak_match_tolerance_ms", 60.0))
    detector_agreement = match_r_peaks(
        primary_peaks,
        secondary_peaks,
        tolerance_samples=max(1, int(round(tolerance_ms * fs / 1000.0))),
    )
    # Only one-to-one matches are promoted to physiological beats.  Unmatched
    # candidates remain explicit provenance failures and cannot constrain P/T
    # search windows or contribute analyzable time.
    peaks = [primary_peaks[index] for index in detector_agreement.matched_primary_indices]
    extraction_provenance: dict[str, Any] = {
        "fiducial_method": DELINEATION_METHOD,
        "primary_r_detector": "adaptive_derivative_energy_v1",
        "secondary_r_detector": SECONDARY_R_DETECTOR,
        "r_peak_match_tolerance_ms": tolerance_ms,
        "r_detector_agreement_f1": detector_agreement.score,
        "primary_r_peak_count": len(primary_peaks),
        "secondary_r_peak_count": len(secondary_peaks),
        "matched_r_peak_count": len(detector_agreement.matches),
        "unmatched_primary_candidate_count": len(primary_peaks) - len(detector_agreement.matches),
        "unmatched_secondary_candidate_count": len(secondary_peaks)
        - len(detector_agreement.matches),
        "analyzable_duration_definition": (
            "sum_of_adjacent_RR_intervals_bounded_by_accepted_quality_valid_beats"
        ),
    }
    if len(peaks) < 3:
        extraction_provenance["analyzable_duration_s"] = 0.0
        return [], [], [], {}, extraction_provenance

    rr_intervals_s = np.diff(np.asarray(peaks, dtype=float)) / fs
    template_radius = max(1, int(round(0.12 * fs)))
    qrs_windows = [
        diagnostic[peak - template_radius : peak + template_radius + 1, lead_ii]
        for peak in peaks
        if peak - template_radius >= 0
        and peak + template_radius + 1 <= diagnostic.shape[0]
    ]
    median_qrs_template = (
        np.nanmedian(np.stack(qrs_windows), axis=0) if qrs_windows else None
    )

    quality_by_lead = {
        str(name): _lead_quality(diagnostic[:, index], peaks, np, fs)
        for index, name in enumerate(sig_names)
    }
    quality_lookup = {name.lower(): value for name, value in quality_by_lead.items()}
    quality_minimum = float(config.thresholds.get("feature_quality_min_db", 5.0))
    confidence_minimum = float(config.thresholds.get("delineation_confidence_minimum", 0.5))
    detector_minimum = float(config.thresholds.get("detector_agreement_minimum", 0.5))

    def leads_good(*names: str) -> bool:
        return all(
            quality_lookup.get(name.lower(), float("-inf")) >= quality_minimum for name in names
        )

    measurements: list[dict[str, Any]] = []
    acceptances: list[BeatAcceptance] = []
    tq_ratios: list[float] = []
    rhythm_quality_valid = [False] * len(peaks)
    rejection_reason_counts: dict[str, int] = {
        "r_detector_disagreement": len(primary_peaks) - len(detector_agreement.matches)
    }
    if rejection_reason_counts["r_detector_disagreement"] == 0:
        rejection_reason_counts = {}
    delineation_reason_counts: dict[str, int] = {}

    for idx, peak in enumerate(peaks):
        rr_prev = (peak - peaks[idx - 1]) / fs if idx > 0 else None
        rr_next = (peaks[idx + 1] - peak) / fs if idx < len(peaks) - 1 else None
        rr_s = rr_next if rr_next is not None else rr_prev
        local_left = max(0, idx - 3)
        local_right = min(len(rr_intervals_s), idx + 3)
        local_rr = rr_intervals_s[local_left:local_right]
        local_rr_baseline = (
            float(np.nanmedian(local_rr)) if len(local_rr) else None
        )
        prematurity_index = (
            rr_prev / local_rr_baseline
            if rr_prev is not None
            and local_rr_baseline is not None
            and local_rr_baseline > 0
            else None
        )
        compensatory_pause_ratio = (
            (rr_prev + rr_next) / (2.0 * local_rr_baseline)
            if rr_prev is not None
            and rr_next is not None
            and local_rr_baseline is not None
            and local_rr_baseline > 0
            else None
        )
        beat_id = f"{record_id}-beat-{idx:04d}"
        delineation = delineate_beat(
            diagnostic[:, lead_ii],
            peak,
            fs,
            beat_id,
            record_id,
            previous_r_peak=peaks[idx - 1] if idx > 0 else None,
            next_r_peak=peaks[idx + 1] if idx < len(peaks) - 1 else None,
        )
        fiducials = delineation.fiducials
        for reason in delineation.reasons:
            delineation_reason_counts[reason] = delineation_reason_counts.get(reason, 0) + 1
        quality = float(quality_by_lead[str(sig_names[lead_ii])])
        acceptance = accept_beat(
            fiducials,
            lead_quality_db=quality,
            delineation_confidence=fiducials.confidence,
            pacing_contaminated=False,
            minimum_lead_quality_db=quality_minimum,
            minimum_delineation_confidence=confidence_minimum,
            r_detector_matched=True,
        )
        acceptances.append(acceptance)
        if not acceptance.accepted:
            for reason in acceptance.reasons:
                rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
            continue

        qrs_on = int(round(float(fiducials.qrs_on)))
        r_peak = int(round(float(fiducials.r_peak)))
        qrs_off = int(round(float(fiducials.qrs_off)))
        t_on = int(round(float(fiducials.t_on)))
        t_off = int(round(float(fiducials.t_off)))
        p_available = all(
            value is not None for value in (fiducials.p_on, fiducials.p_peak, fiducials.p_off)
        )
        p_on = int(round(float(fiducials.p_on))) if p_available else None
        p_peak = int(round(float(fiducials.p_peak))) if p_available else None
        p_off = int(round(float(fiducials.p_off))) if p_available else None

        baseline_window = diagnostic[
            max(0, qrs_on - int(0.06 * fs)) : max(qrs_on - int(0.02 * fs), 1)
        ]
        baseline = (
            np.median(baseline_window, axis=0)
            if baseline_window.size
            else np.zeros(signal.shape[1], dtype=np.float32)
        )

        qrs_sig_ii = diagnostic[qrs_on:qrs_off, lead_ii]
        qrs_sig_i = diagnostic[qrs_on:qrs_off, lead_i]
        qrs_sig_avf = diagnostic[qrs_on:qrs_off, lead_avf]
        t_sig_v5 = diagnostic[t_on:t_off, lead_v5]
        st_idx = min(signal.shape[0] - 1, qrs_off + _st_offset_samples(rr_s, fs))

        tq_start = min(signal.shape[0] - 1, t_off + int(0.04 * fs))
        tq_end = min(
            signal.shape[0],
            (peaks[idx + 1] - int(0.04 * fs)) if idx < len(peaks) - 1 else tq_start + int(0.2 * fs),
        )
        tq_segment = diagnostic[tq_start:tq_end, lead_ii]
        if tq_segment.size >= 16:
            freqs, psd = sp_signal.welch(tq_segment, fs=fs, nperseg=min(128, tq_segment.size))
            f_mask = (freqs >= 4.0) & (freqs <= 10.0)
            all_mask = (freqs >= 0.5) & (freqs <= 20.0)
            p_f = float(np.trapezoid(psd[f_mask], freqs[f_mask])) if np.any(f_mask) else 0.0
            p_all = float(np.trapezoid(psd[all_mask], freqs[all_mask])) if np.any(all_mask) else 0.0
            tq_ratios.append(p_f / p_all if p_all > 0 else 0.0)

        qrs_dur_ms = 1000.0 * (qrs_off - qrs_on) / fs
        fixed_qrs = diagnostic[
            max(0, peak - template_radius) : min(
                diagnostic.shape[0], peak + template_radius + 1
            ),
            lead_ii,
        ]
        morphology_distance = None
        if (
            median_qrs_template is not None
            and len(fixed_qrs) == len(median_qrs_template)
        ):
            scale = float(np.nanstd(median_qrs_template))
            morphology_distance = float(
                np.sqrt(np.nanmean((fixed_qrs - median_qrs_template) ** 2))
                / max(scale, 1e-6)
            )
        qrs_areas = np.asarray(
            [
                float(np.sum(diagnostic[qrs_on:qrs_off, lead_index] - baseline[lead_index]))
                for lead_index in range(diagnostic.shape[1])
            ],
            dtype=float,
        )
        nonzero_area = qrs_areas[np.abs(qrs_areas) > 1e-9]
        axis_concordance = (
            float(
                max(
                    np.mean(nonzero_area > 0),
                    np.mean(nonzero_area < 0),
                )
            )
            if len(nonzero_area)
            else None
        )
        secondary_extrema = _count_secondary_extrema(qrs_sig_ii, np)
        qrs_def_prob = 1.0 / (
            1.0 + math.exp(-((qrs_dur_ms - 110.0) / 10.0 + 0.5 * secondary_extrema))
        )
        right_t = diagnostic[t_on:t_off, [lead_v1, lead_v2, lead_v3]]
        min_t = float(np.min(right_t)) if right_t.size else 0.0
        neg_duration_ms = (
            float(np.sum(right_t[:, 0] < float(config.thresholds["t_inverted_threshold_mv"])))
            * 1000.0
            / fs
            if right_t.size
            else 0.0
        )
        u_idx = min(signal.shape[0] - 1, t_off + int(0.08 * fs))

        agreement_score = detector_agreement.score
        rhythm_valid = bool(
            leads_good("II") and agreement_score is not None and agreement_score >= detector_minimum
        )
        atrial_valid = bool(
            p_available
            and leads_good("I", "II", "V1")
            and delineation.p_confidence >= confidence_minimum
            and rhythm_valid
        )
        qrs_valid = bool(
            leads_good("I", "V1", "V2", "V5", "V6")
            and delineation.qrs_confidence >= confidence_minimum
            and rhythm_valid
        )
        st_t_valid = bool(
            leads_good("I", "II", "V1", "V5", "V6")
            and delineation.t_confidence >= confidence_minimum
            and rhythm_valid
        )
        rhythm_quality_valid[idx] = rhythm_valid
        p_amplitude = (
            float(diagnostic[p_peak, lead_ii] - baseline[lead_ii]) if p_peak is not None else None
        )

        beat = BeatMeasurement(
            beat_id=beat_id,
            rr_s=rr_s,
            rr_prev_s=rr_prev,
            rr_next_s=rr_next,
            p_present=bool(atrial_valid and p_amplitude is not None and abs(p_amplitude) >= 0.02),
            p_amp_ii_mv=p_amplitude if atrial_valid else None,
            p_dur_ms=(1000.0 * (p_off - p_on) / fs) if p_available else None,
            pr_ms=(1000.0 * (qrs_on - p_on) / fs) if p_available else None,
            q_amp_ii_mv=float(np.min(qrs_sig_ii) - baseline[lead_ii]) if qrs_sig_ii.size else 0.0,
            r_amp_ii_mv=float(diagnostic[r_peak, lead_ii] - baseline[lead_ii]),
            s_amp_ii_mv=float(qrs_sig_ii[-1] - baseline[lead_ii]) if qrs_sig_ii.size else 0.0,
            qrs_dur_ms=qrs_dur_ms,
            qrs_deformed_prob=float(max(0.0, min(1.0, qrs_def_prob))),
            qrs_secondary_extrema=secondary_extrema,
            r_prime_v1=bool(np.max(diagnostic[qrs_on:qrs_off, lead_v1] - baseline[lead_v1]) > 0.15)
            if qrs_off > qrs_on
            else False,
            broad_r_v6=bool(np.max(diagnostic[qrs_on:qrs_off, lead_v5] - baseline[lead_v5]) > 0.2)
            if qrs_off > qrs_on
            else False,
            st_level_v1_mv=float(diagnostic[st_idx, lead_v1] - baseline[lead_v1]),
            st_level_v5_mv=float(diagnostic[st_idx, lead_v5] - baseline[lead_v5]),
            st_slope_v5_uv_per_ms=float(
                (
                    (diagnostic[st_idx, lead_v5] - diagnostic[qrs_off, lead_v5])
                    / max(1, st_idx - qrs_off)
                )
                * fs
                * 1000.0
            ),
            t_amp_v5_mv=float(np.max(t_sig_v5) - baseline[lead_v5]) if t_sig_v5.size else 0.0,
            t_dur_ms=1000.0 * (t_off - t_on) / fs,
            t_amp_right_mv=min_t,
            t_negative_duration_ms=neg_duration_ms,
            qt_ms=1000.0 * (t_off - qrs_on) / fs,
            qrs_net_area_i_mv_ms=float((1000.0 / fs) * np.sum(qrs_sig_i - baseline[lead_i]))
            if qrs_sig_i.size
            else 0.0,
            qrs_net_area_avf_mv_ms=float((1000.0 / fs) * np.sum(qrs_sig_avf - baseline[lead_avf]))
            if qrs_sig_avf.size
            else 0.0,
            lead_quality_db=quality,
            delineation_confidence=fiducials.confidence,
            u_present_v2=bool(fs >= 500 and diagnostic[u_idx, lead_v2] > baseline[lead_v2] + 0.02),
            u_amp_v2_mv=float(diagnostic[u_idx, lead_v2] - baseline[lead_v2])
            if fs >= 500
            else None,
            pvc_like=bool(
                qrs_dur_ms >= float(config.thresholds["qrs_wide_ms"])
                and rr_prev is not None
                and rr_s is not None
                and rr_prev < rr_s
            ),
            apb_like=bool(
                rr_prev is not None
                and rr_s is not None
                and rr_prev < rr_s
                and qrs_dur_ms < float(config.thresholds["qrs_wide_ms"])
            ),
            paced_like=False,
            is_ectopic=bool(rr_prev is not None and rr_s is not None and rr_prev < 0.85 * rr_s),
            is_paced=False,
            is_artifact=False,
            rhythm_valid=rhythm_valid,
            atrial_valid=atrial_valid,
            qrs_valid=qrs_valid,
            st_t_valid=st_t_valid,
            detector_agreement=agreement_score,
            prematurity_index=prematurity_index,
            compensatory_pause_ratio=compensatory_pause_ratio,
            qrs_morphology_distance=morphology_distance,
            preceding_p_wave_probability=(
                delineation.p_confidence if p_available else 0.0
            ),
            axis_concordance=axis_concordance,
            beat_template_cluster=(
                "ventricular_like"
                if prematurity_index is not None
                and prematurity_index < 0.85
                and (
                    qrs_dur_ms >= float(config.thresholds["qrs_wide_ms"])
                    or (morphology_distance or 0.0) >= 0.45
                )
                else (
                    "atrial_like"
                    if prematurity_index is not None
                    and prematurity_index < 0.85
                    else "normal_like"
                )
            ),
        )
        measurements.append(asdict(beat))

    triad_rows = _strict_triad_rows(record_id, acceptances)
    analyzable_duration_s = analyzable_duration_from_beats(
        peaks,
        rhythm_quality_valid,
        fs,
        minimum_rr_s=float(config.thresholds.get("minimum_analyzable_rr_s", 0.25)),
        maximum_rr_s=float(config.thresholds.get("maximum_analyzable_rr_s", 2.5)),
    )
    extraction_provenance["analyzable_duration_s"] = analyzable_duration_s
    extraction_provenance["accepted_quality_valid_beat_count"] = sum(rhythm_quality_valid)
    extraction_provenance["accepted_morphology_beat_count"] = sum(
        acceptance.accepted for acceptance in acceptances
    )
    extraction_provenance["beat_rejection_reason_counts"] = rejection_reason_counts
    extraction_provenance["delineation_reason_counts"] = delineation_reason_counts
    return measurements, triad_rows, tq_ratios, quality_by_lead, extraction_provenance


_MEASUREMENT_WORKER_CONFIG: ProjectConfig | None = None
_MEASUREMENT_WORKER_DATASET: str | None = None


def _measurement_record(
    config: ProjectConfig,
    dataset: str,
    split: str,
    entry: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signal, fs, sig_names = _load_record(config, dataset, entry)
    (
        measurements,
        triad_rows,
        tq_ratios,
        quality_by_lead,
        extraction_provenance,
    ) = _one_record_measurements(
        signal,
        fs,
        sig_names,
        str(entry["record_id"]),
        config,
    )
    return (
        {
            "record_id": str(entry["record_id"]),
            "split": split,
            "sampling_rate_hz": fs,
            "qrs_def_threshold": 0.5,
            "labels": _parse_labels(entry["labels"]),
            "beats": measurements,
            "tq_power_ratios": tq_ratios,
            "lead_quality_by_lead_db": quality_by_lead,
            "analyzable_duration_s": extraction_provenance[
                "analyzable_duration_s"
            ],
            "extraction_provenance": extraction_provenance,
        },
        triad_rows,
    )


def _initialize_measurement_worker(config: ProjectConfig, dataset: str) -> None:
    global _MEASUREMENT_WORKER_CONFIG, _MEASUREMENT_WORKER_DATASET
    _MEASUREMENT_WORKER_CONFIG = config
    _MEASUREMENT_WORKER_DATASET = dataset


def _measurement_worker(
    task: tuple[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _MEASUREMENT_WORKER_CONFIG is None or _MEASUREMENT_WORKER_DATASET is None:
        raise RuntimeError("Measurement worker was not initialized")
    split, entry = task
    return _measurement_record(
        _MEASUREMENT_WORKER_CONFIG,
        _MEASUREMENT_WORKER_DATASET,
        split,
        entry,
    )


def build_measurement_records(
    config: ProjectConfig, dataset: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = split_entries(config, dataset)
    tasks = [
        (split, entry)
        for split, entries in grouped.items()
        for entry in entries
    ]
    return build_measurement_records_for_tasks(config, dataset, tasks)


def build_measurement_records_for_tasks(
    config: ProjectConfig,
    dataset: str,
    tasks: list[tuple[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure an explicit, pre-authorized record list."""

    records: list[dict[str, Any]] = []
    triads: list[dict[str, Any]] = []

    configured_workers = int(config.thresholds.get("extraction_workers", 1))
    worker_count = max(
        1,
        min(configured_workers, os.cpu_count() or 1, len(tasks) or 1),
    )
    executor = (
        ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_measurement_worker,
            initargs=(config, dataset),
        )
        if worker_count > 1
        else None
    )
    iterator = (
        executor.map(_measurement_worker, tasks, chunksize=1)
        if executor is not None
        else (
            _measurement_record(config, dataset, split, entry)
            for split, entry in tasks
        )
    )
    try:
        for index, (record, triad_rows) in enumerate(iterator, start=1):
            records.append(record)
            triads.extend(triad_rows)
            if index % 250 == 0 or index == len(tasks):
                print(
                    f"Measured {index}/{len(tasks)} {dataset} records "
                    f"with {worker_count} worker(s)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return records, triads
