"""Synthetic regressions for conservative, signal-derived ECG delineation."""

from __future__ import annotations

import numpy as np
import pytest

from tm_ecg.signal.fiducials import (
    DELINEATION_METHOD,
    accept_beat,
    analyzable_duration_from_beats,
    delineate_beat,
    detect_secondary_r_peaks,
    match_r_peaks,
    verify_fiducial_order,
)
from tm_ecg.types import BeatFiducials


FS = 500.0


def _gaussian(samples: np.ndarray, centre: float, sigma_ms: float, amplitude: float) -> np.ndarray:
    sigma_samples = sigma_ms * FS / 1000.0
    return amplitude * np.exp(-0.5 * ((samples - centre) / sigma_samples) ** 2)


def _synthetic_beat(*, qrs_sigma_ms: float, t_sigma_ms: float) -> tuple[np.ndarray, int]:
    samples = np.arange(1000, dtype=float)
    r_peak = 400
    signal = _gaussian(samples, 320, 24.0, 0.12)
    signal += _gaussian(samples, r_peak, qrs_sigma_ms, 1.0)
    signal += _gaussian(samples, 535, t_sigma_ms, 0.28)
    return signal, r_peak


def test_signal_delineation_adapts_qrs_and_t_widths() -> None:
    narrow_signal, r_peak = _synthetic_beat(qrs_sigma_ms=8.0, t_sigma_ms=38.0)
    broad_signal, _ = _synthetic_beat(qrs_sigma_ms=19.0, t_sigma_ms=68.0)

    narrow = delineate_beat(narrow_signal, r_peak, FS, "narrow", "synthetic")
    broad = delineate_beat(broad_signal, r_peak, FS, "broad", "synthetic")

    assert narrow.fiducials.source == DELINEATION_METHOD
    assert verify_fiducial_order(narrow.fiducials)
    assert verify_fiducial_order(broad.fiducials)
    assert narrow.fiducials.qrs_on is not None and narrow.fiducials.qrs_off is not None
    assert broad.fiducials.qrs_on is not None and broad.fiducials.qrs_off is not None
    assert narrow.fiducials.t_on is not None and narrow.fiducials.t_off is not None
    assert broad.fiducials.t_on is not None and broad.fiducials.t_off is not None
    narrow_qrs = narrow.fiducials.qrs_off - narrow.fiducials.qrs_on
    broad_qrs = broad.fiducials.qrs_off - broad.fiducials.qrs_on
    narrow_t = narrow.fiducials.t_off - narrow.fiducials.t_on
    broad_t = broad.fiducials.t_off - broad.fiducials.t_on
    assert broad_qrs > narrow_qrs + 5
    assert broad_t > narrow_t + 10


def test_flat_morphology_fails_closed_without_fixed_boundaries() -> None:
    result = delineate_beat(np.zeros(1000), 400, FS, "flat", "synthetic")

    assert result.fiducials.qrs_on is None
    assert result.fiducials.qrs_off is None
    assert result.fiducials.t_on is None
    assert result.fiducials.confidence == 0.0
    assert "qrs_unavailable" in result.reasons
    acceptance = accept_beat(
        result.fiducials,
        lead_quality_db=20.0,
        delineation_confidence=result.fiducials.confidence,
        pacing_contaminated=False,
        minimum_delineation_confidence=0.5,
    )
    assert not acceptance.accepted
    assert "required_fiducials_unavailable" in acceptance.reasons


def test_partial_wave_and_equal_boundaries_are_invalid() -> None:
    partial = BeatFiducials(
        beat_id="partial",
        record_id="synthetic",
        qrs_on=100,
        r_peak=100,
        qrs_off=120,
        t_on=150,
        t_peak=180,
        t_off=210,
    )
    assert not verify_fiducial_order(partial)


def test_detector_agreement_is_one_to_one_temporal_f1() -> None:
    agreement = match_r_peaks([100, 200, 300], [103, 198, 500], tolerance_samples=5)

    assert agreement.matches == ((100, 103), (200, 198))
    assert agreement.matched_primary_indices == (0, 1)
    assert agreement.score == pytest.approx(2.0 / 3.0)
    assert agreement.primary_match_fraction == pytest.approx(2.0 / 3.0)
    assert agreement.secondary_match_fraction == pytest.approx(2.0 / 3.0)


def test_secondary_detector_finds_synthetic_qrs_train() -> None:
    samples = np.arange(2500, dtype=float)
    expected = [400, 900, 1400, 1900]
    signal = sum(_gaussian(samples, centre, 10.0, 1.0) for centre in expected)

    detected = detect_secondary_r_peaks(signal, FS)
    agreement = match_r_peaks(expected, detected, tolerance_samples=int(0.06 * FS))

    assert agreement.score is not None
    assert agreement.score >= 0.85


def test_analyzable_duration_excludes_invalid_and_unbounded_time() -> None:
    # Only 100->200 and 400->500 are bounded by adjacent valid beats.  The
    # record edges and both intervals touching the invalid beat are excluded.
    duration = analyzable_duration_from_beats(
        [100, 200, 300, 400, 500],
        [True, True, False, True, True],
        sampling_rate_hz=100.0,
    )

    assert duration == pytest.approx(2.0)
    assert duration < 5.0


def test_acceptance_preserves_quality_and_detector_failure_reasons() -> None:
    fiducials = BeatFiducials(
        beat_id="b",
        record_id="r",
        qrs_on=100,
        r_peak=120,
        qrs_off=145,
        t_on=180,
        t_peak=220,
        t_off=280,
        confidence=0.4,
    )
    result = accept_beat(
        fiducials,
        lead_quality_db=3.0,
        delineation_confidence=0.4,
        pacing_contaminated=False,
        minimum_lead_quality_db=5.0,
        minimum_delineation_confidence=0.5,
        r_detector_matched=False,
    )

    assert not result.accepted
    assert result.fiducial_completeness == pytest.approx(6.0 / 9.0)
    assert set(result.reasons) == {
        "lead_quality_below_minimum",
        "delineation_confidence_below_minimum",
        "r_detector_disagreement",
    }
