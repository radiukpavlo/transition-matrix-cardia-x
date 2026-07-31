"""Tests for the ontology mapping module."""

import unittest

from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.ontology import (
    appendix_d_mapping,
    compatibility_projection,
    map_ludb_text,
    map_ptbxl_labels,
)
from tm_ecg.types import MultiAxialTarget


class OntologyMappingTests(unittest.TestCase):
    def test_all_project_labels_have_mappings(self) -> None:
        """Every PROJECT_LABELS entry (except Other) should appear in at least one mapping."""
        mapping = appendix_d_mapping()
        mapped_labels = set()
        for item in mapping:
            mapped_labels.add(item.project_label)
        for label in PROJECT_LABELS:
            if label == "Other / unmapped":
                continue
            with self.subTest(label=label):
                self.assertIn(label, mapped_labels, f"{label} has no mapping in appendix_d")

    def test_ptbxl_normal_maps_correctly(self) -> None:
        labels = map_ptbxl_labels({"scp_codes": "{'NORM': 100}"})
        self.assertIn("Normal", labels)

    def test_ptbxl_pacemaker_flag_adds_paced(self) -> None:
        labels = map_ptbxl_labels({"scp_codes": "{'NORM': 100}", "pacemaker": "1"})
        self.assertIn("Paced", labels)

    def test_ludb_rbbb_maps_correctly(self) -> None:
        labels = map_ludb_text("Right bundle branch block")
        self.assertIn("RBBB spectrum", labels)

    def test_ludb_unknown_returns_other(self) -> None:
        labels = map_ludb_text("some totally unknown diagnosis xyz")
        self.assertTrue(
            any("Other" in label or "unmapped" in label for label in labels) or len(labels) == 0,
            f"Expected Other/unmapped for unknown diagnosis, got {labels}"
        )

    def test_appendix_d_mapping_is_nonempty(self) -> None:
        mapping = appendix_d_mapping()
        self.assertGreater(len(mapping), 0)

    def test_specific_compatibility_label_excludes_residual(self) -> None:
        target = MultiAxialTarget(
            rhythm=("af",),
            repolarization=("st_depression",),
        )
        self.assertEqual(compatibility_projection(target), ["AF"])


if __name__ == "__main__":
    unittest.main()
