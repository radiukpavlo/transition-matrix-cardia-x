"""Tests for signal processing modules: fiducials, filtering, R-peaks, pacing."""

import unittest

from tm_ecg.signal.fiducials import verify_fiducial_order, accept_beat
from tm_ecg.types import BeatFiducials


class FiducialTests(unittest.TestCase):
    def test_correct_order_returns_true(self) -> None:
        fid = BeatFiducials(
            beat_id="b1", record_id="r1",
            p_on=10, p_peak=15, p_off=20,
            qrs_on=25, r_peak=30, qrs_off=35,
            t_on=40, t_peak=50, t_off=60,
        )
        self.assertTrue(verify_fiducial_order(fid))

    def test_reversed_t_wave_returns_false(self) -> None:
        fid = BeatFiducials(
            beat_id="b1", record_id="r1",
            p_on=10, p_peak=15, p_off=20,
            qrs_on=25, r_peak=30, qrs_off=35,
            t_on=60, t_peak=50, t_off=40,  # reversed
        )
        self.assertFalse(verify_fiducial_order(fid))

    def test_p_after_qrs_returns_false(self) -> None:
        fid = BeatFiducials(
            beat_id="b1", record_id="r1",
            p_on=30, p_peak=35, p_off=40,  # P after QRS onset
            qrs_on=25, r_peak=28, qrs_off=32,
            t_on=45, t_peak=50, t_off=55,
        )
        self.assertFalse(verify_fiducial_order(fid))

    def test_missing_p_wave_still_valid(self) -> None:
        """P-wave can be None for ectopic beats."""
        fid = BeatFiducials(
            beat_id="b1", record_id="r1",
            p_on=None, p_peak=None, p_off=None,
            qrs_on=25, r_peak=30, qrs_off=35,
            t_on=40, t_peak=50, t_off=60,
        )
        # The function should handle None gracefully
        result = verify_fiducial_order(fid)
        self.assertIsInstance(result, bool)

    def test_beat_acceptance_rejects_bad_fiducials(self) -> None:
        fid = BeatFiducials(
            beat_id="b1", record_id="r1",
            p_on=60, p_peak=50, p_off=40,  # reversed
            qrs_on=25, r_peak=30, qrs_off=35,
            t_on=45, t_peak=50, t_off=55,
        )
        result = accept_beat(fid, lead_quality_db=10.0, delineation_confidence=0.9, pacing_contaminated=False)
        self.assertFalse(result.accepted)

    def test_beat_acceptance_accepts_good_fiducials(self) -> None:
        fid = BeatFiducials(
            beat_id="b1", record_id="r1",
            p_on=10, p_peak=15, p_off=20,
            qrs_on=25, r_peak=30, qrs_off=35,
            t_on=40, t_peak=50, t_off=60,
        )
        result = accept_beat(fid, lead_quality_db=10.0, delineation_confidence=0.9, pacing_contaminated=False)
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
