import unittest

from tm_ecg.features.formulas import BeatMeasurement, RecordMeasurements, compute_record_features, st_offset_seconds


THRESHOLDS = {
    "qrs_wide_ms": 120.0,
    "t_inverted_threshold_mv": -0.1,
    "t_inverted_duration_ms": 80.0,
}


class FormulaTests(unittest.TestCase):
    def test_st_offset_rule_is_locked(self) -> None:
        self.assertEqual(st_offset_seconds(99.9), 0.08)
        self.assertEqual(st_offset_seconds(100.0), 0.06)

    def test_nn_only_sdnn_and_qtcf_and_u_wave_block(self) -> None:
        beats = [
            BeatMeasurement(beat_id="b1", rr_s=1.0, rr_prev_s=1.0, rr_next_s=1.0, qt_ms=400.0),
            BeatMeasurement(beat_id="b2", rr_s=1.0, rr_prev_s=1.0, rr_next_s=1.0, qt_ms=410.0),
            BeatMeasurement(beat_id="b3", rr_s=0.7, rr_prev_s=1.0, rr_next_s=1.1, qt_ms=390.0, is_ectopic=True),
            BeatMeasurement(beat_id="b4", rr_s=1.0, rr_prev_s=1.1, rr_next_s=1.0, qt_ms=405.0),
            BeatMeasurement(beat_id="b5", rr_s=1.0, rr_prev_s=1.0, rr_next_s=1.0, qt_ms=400.0),
            BeatMeasurement(beat_id="b6", rr_s=1.0, rr_prev_s=1.0, rr_next_s=1.0, qt_ms=395.0),
        ]
        record = RecordMeasurements(record_id="r1", beats=beats, sampling_rate_hz=100.0)
        features = compute_record_features(record, THRESHOLDS)
        self.assertEqual(features["rr_sdnn_ms"], 0.0)
        self.assertAlmostEqual(features["qtc_med_ms"], 402.5, places=1)
        self.assertEqual(features["qtc_formula_code"], "QTcF")
        self.assertIsNone(features["u_present_v2_any"])
        self.assertIsNone(features["u_amp_v2_mV"])

    def test_qrs_and_t_flags(self) -> None:
        beats = [
            BeatMeasurement(
                beat_id="b1",
                rr_s=1.0,
                qrs_dur_ms=130.0,
                qrs_deformed_prob=0.9,
                qrs_secondary_extrema=3,
                r_prime_v1=True,
                broad_r_v6=True,
                t_amp_right_mv=-0.2,
                t_negative_duration_ms=100.0,
            )
        ]
        record = RecordMeasurements(record_id="r2", beats=beats, sampling_rate_hz=500.0, qrs_def_threshold=0.5)
        features = compute_record_features(record, THRESHOLDS)
        self.assertEqual(features["qrs_wide_any"], 1)
        self.assertEqual(features["qrs_deformed_any"], 1)
        self.assertEqual(features["qrs_fragmented_any"], 1)
        self.assertEqual(features["r_prime_v1_any"], 1)
        self.assertEqual(features["broad_r_v6_any"], 1)
        self.assertEqual(features["t_inverted_right_any"], 1)

    def test_lead_coverage_is_case_insensitive(self) -> None:
        record = RecordMeasurements(
            record_id="lowercase-leads",
            beats=[BeatMeasurement(beat_id="b1")],
            lead_quality_by_lead_db={
                lead.lower(): 12.0
                for lead in ("I", "II", "V1", "V2", "V5", "V6")
            },
        )

        features = compute_record_features(record, THRESHOLDS)

        self.assertEqual(features["atrial_lead_coverage"], 1.0)
        self.assertEqual(features["qrs_lead_coverage"], 1.0)
        self.assertEqual(features["st_t_lead_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
