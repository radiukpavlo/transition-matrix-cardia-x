"""Tests for training-only matrix-A preprocessing."""

import unittest

from tm_ecg.transition.a_preprocess import (
    apply_a_preprocess_bundle,
    fit_a_preprocess_bundle,
)


class APreprocessTests(unittest.TestCase):
    def test_zero_variance_columns_are_dropped_and_val_uses_train_bundle(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")

        train = [
            {"record_id": "r1", "split": "train", "latent_0000": 1.0, "latent_0001": 5.0, "latent_0002": 0.0},
            {"record_id": "r2", "split": "train", "latent_0000": 2.0, "latent_0001": 5.0, "latent_0002": 1.0},
            {"record_id": "r3", "split": "train", "latent_0000": 3.0, "latent_0001": 5.0, "latent_0002": 2.0},
        ]
        bundle, reduced_train = fit_a_preprocess_bundle(train, dataset="b1", rank_cap=2)
        val = [{"record_id": "v1", "split": "val", "latent_0000": 4.0, "latent_0001": 999.0, "latent_0002": 3.0}]
        reduced_val = apply_a_preprocess_bundle(val, bundle)
        self.assertEqual(bundle.dropped_zero_variance_columns, ["latent_0001"])
        self.assertEqual(len(reduced_train), 3)
        self.assertEqual(len([key for key in reduced_val[0] if key.startswith("a_red_")]), len(bundle.components))


if __name__ == "__main__":
    unittest.main()
