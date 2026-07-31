"""Tests for the transition operator fitting and transforms."""

import unittest

from pathlib import Path
import tempfile

from tm_ecg.transition.ridge import (
    apply_transition,
    fit_masked_ridge_transition,
    fit_ridge_transition,
    load_operator_package,
    save_operator_package,
    singular_value_keep_mask,
)
from tm_ecg.transition.typed_transforms import fit_transform_bundle, transform_rows, inverse_rows


class RidgeTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")

    def test_identity_mapping_small_lambda(self) -> None:
        """When A == B, T should approximate the identity for small lambda."""
        A = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        B = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        result = fit_ridge_transition(A, B, lambda_value=1e-8, rank_cap=2)
        predicted = apply_transition(A, result["operator"])
        for true_row, pred_row in zip(B, predicted):
            for true_val, pred_val in zip(true_row, pred_row):
                self.assertAlmostEqual(true_val, pred_val, places=3)

    def test_rank_cap_limits_singular_values(self) -> None:
        A = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        B = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        result = fit_ridge_transition(A, B, lambda_value=0.01, rank_cap=1)
        self.assertIn("operator", result)
        self.assertIn("lambda_value", result)

    def test_masked_ridge_uses_featurewise_observed_rows(self) -> None:
        a_rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
        b_rows = [
            [2.0, None, None],
            [None, 3.0, None],
            [2.0, 3.0, None],
            [4.0, 3.0, 1.0],
        ]

        result = fit_masked_ridge_transition(
            a_rows,
            b_rows,
            lambda_value=1e-8,
            rank_cap=2,
            minimum_target_rows=2,
        )
        predicted = apply_transition(a_rows, result["operator"])

        self.assertEqual(result["target_support"], [3, 3, 1])
        self.assertEqual(result["target_status"][:2], ["ok", "ok"])
        self.assertEqual(
            result["target_status"][2],
            "not_estimable_insufficient_observations",
        )
        self.assertAlmostEqual(predicted[0][0], 2.0, places=3)
        self.assertAlmostEqual(predicted[1][1], 3.0, places=3)
        self.assertEqual(predicted[0][2], 0.0)

    def test_apply_transition_output_dimensions(self) -> None:
        operator = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        A = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        predicted = apply_transition(A, operator)
        self.assertEqual(len(predicted), 3)
        self.assertEqual(len(predicted[0]), 3)

    def test_singular_value_mask_filters_tiny(self) -> None:
        mask = singular_value_keep_mask([10.0, 5.0, 1e-18], m=100, r=3)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])

    def test_npz_operator_roundtrip(self) -> None:
        result = fit_ridge_transition([[1.0, 0.0], [0.0, 1.0]], [[2.0], [3.0]], lambda_value=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_operator_package(Path(tmp) / "T_ridge.npz", result)
            loaded = load_operator_package(path)
        self.assertEqual(loaded["retained_rank"], result["retained_rank"])
        self.assertEqual(len(loaded["operator"]), 2)


class TypedTransformTests(unittest.TestCase):
    def test_roundtrip_continuous(self) -> None:
        rows = [
            {"record_id": "r1", "qt_med_ms": 380.0},
            {"record_id": "r2", "qt_med_ms": 420.0},
        ]
        bundle = fit_transform_bundle(rows, ["qt_med_ms"])
        transformed = transform_rows(rows, bundle)
        restored = inverse_rows(transformed, bundle)
        self.assertAlmostEqual(float(restored[0]["qt_med_ms"]), 380.0, delta=0.5)
        self.assertAlmostEqual(float(restored[1]["qt_med_ms"]), 420.0, delta=0.5)

    def test_roundtrip_binary(self) -> None:
        rows = [
            {"record_id": "r1", "qrs_deformed_any": 0},
            {"record_id": "r2", "qrs_deformed_any": 1},
        ]
        bundle = fit_transform_bundle(rows, ["qrs_deformed_any"])
        transformed = transform_rows(rows, bundle)
        restored = inverse_rows(transformed, bundle)
        self.assertLess(float(restored[0]["qrs_deformed_any"]), 0.1)
        self.assertGreater(float(restored[1]["qrs_deformed_any"]), 0.9)

    def test_roundtrip_count(self) -> None:
        rows = [
            {"record_id": "r1", "pvc_like_beat_count": 0},
            {"record_id": "r2", "pvc_like_beat_count": 5},
        ]
        bundle = fit_transform_bundle(rows, ["pvc_like_beat_count"])
        transformed = transform_rows(rows, bundle)
        restored = inverse_rows(transformed, bundle)
        self.assertAlmostEqual(float(restored[1]["pvc_like_beat_count"]), 5.0, places=2)


if __name__ == "__main__":
    unittest.main()
