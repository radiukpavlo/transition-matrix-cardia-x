"""Tests for project constants integrity."""

import unittest

from tm_ecg.constants import B_COLUMNS, DATASET_MAP, PROJECT_LABELS


class ConstantsTests(unittest.TestCase):
    def test_b_columns_unique(self) -> None:
        """B_COLUMNS must not contain duplicates."""
        self.assertEqual(len(B_COLUMNS), len(set(B_COLUMNS)))

    def test_dataset_map_keys(self) -> None:
        self.assertEqual(DATASET_MAP["b1"], "ptbxl")
        self.assertEqual(DATASET_MAP["b2"], "ludb")

    def test_project_labels_nonempty(self) -> None:
        self.assertGreater(len(PROJECT_LABELS), 0)

    def test_project_labels_contains_normal(self) -> None:
        self.assertIn("Normal", PROJECT_LABELS)

    def test_project_labels_contains_other(self) -> None:
        self.assertIn("Other / unmapped", PROJECT_LABELS)

    def test_project_labels_unique(self) -> None:
        self.assertEqual(len(PROJECT_LABELS), len(set(PROJECT_LABELS)))

    def test_dataset_map_has_exactly_two_entries(self) -> None:
        self.assertEqual(len(DATASET_MAP), 2)


if __name__ == "__main__":
    unittest.main()
