"""Tests for the feature registry module."""

import unittest

from tm_ecg.features.registry import FEATURE_SPECS, feature_dictionary_rows, fit_columns
from tm_ecg.constants import B_COLUMNS


class FeatureRegistryTests(unittest.TestCase):
    def test_all_b_columns_have_specs(self) -> None:
        """Every column in B_COLUMNS must have an entry in FEATURE_SPECS."""
        for column in B_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(column, FEATURE_SPECS, f"{column} missing from FEATURE_SPECS")

    def test_feature_dictionary_rows_complete(self) -> None:
        rows = feature_dictionary_rows()
        self.assertGreater(len(rows), 0)
        for row in rows:
            d = row.to_dict()
            self.assertIn("column", d)
            self.assertIn("family", d)
            self.assertIn("unit", d)
            self.assertIn("value_type", d)

    def test_feature_specs_value_types_are_valid(self) -> None:
        valid_types = {"continuous", "binary", "count", "bounded", "angle", "logit", "categorical", "circular"}
        for feature, spec in FEATURE_SPECS.items():
            with self.subTest(feature=feature):
                self.assertIn(spec[2], valid_types, f"{feature} has invalid value_type: {spec[2]}")

    def test_fit_columns_excludes_reserved(self) -> None:
        """fit_columns should never include record_id, split, or qtc_formula_code."""
        rows = [
            {"record_id": "r1", "split": "train", "qt_med_ms": 400.0, "qrs_dur_med_ms": 90.0},
            {"record_id": "r2", "split": "train", "qt_med_ms": 410.0, "qrs_dur_med_ms": 95.0},
        ]
        columns = fit_columns(rows)
        self.assertNotIn("record_id", columns)
        self.assertNotIn("split", columns)
        self.assertNotIn("qtc_formula_code", columns)

    def test_fit_columns_returns_list(self) -> None:
        rows = [
            {"record_id": "r1", "qt_med_ms": 400.0},
            {"record_id": "r2", "qt_med_ms": 410.0},
        ]
        columns = fit_columns(rows)
        self.assertIsInstance(columns, list)
        self.assertGreater(len(columns), 0)


if __name__ == "__main__":
    unittest.main()
