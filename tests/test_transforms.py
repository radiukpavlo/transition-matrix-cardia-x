import unittest

from tm_ecg.transition.ridge import singular_value_keep_mask
from tm_ecg.transition.typed_transforms import fit_transform_bundle, inverse_rows, transform_rows


class TransformTests(unittest.TestCase):
    def test_typed_transform_round_trip(self) -> None:
        rows = [
            {
                "record_id": "r1",
                "qt_med_ms": 380.0,
                "pvc_like_beat_count": 0,
                "qrs_deformed_any": 0,
                "p_present_ratio": 0.2,
            },
            {
                "record_id": "r2",
                "qt_med_ms": 400.0,
                "pvc_like_beat_count": 3,
                "qrs_deformed_any": 1,
                "p_present_ratio": 0.8,
            },
        ]
        columns = ["qt_med_ms", "pvc_like_beat_count", "qrs_deformed_any", "p_present_ratio"]
        bundle = fit_transform_bundle(rows, columns)
        transformed = transform_rows(rows, bundle)
        restored = inverse_rows(transformed, bundle)
        self.assertEqual(len(transformed), 2)
        self.assertAlmostEqual(float(restored[0]["qt_med_ms"]), 380.0, delta=0.11)
        self.assertAlmostEqual(float(restored[1]["pvc_like_beat_count"]), 3.0, places=4)
        self.assertLess(float(restored[0]["qrs_deformed_any"]), 0.1)
        self.assertGreater(float(restored[1]["qrs_deformed_any"]), 0.9)

    def test_singular_value_mask(self) -> None:
        mask = singular_value_keep_mask([10.0, 1e-18, 1e-20], m=100, r=3)
        self.assertEqual(mask[0], True)
        self.assertEqual(mask[-1], False)


if __name__ == "__main__":
    unittest.main()
