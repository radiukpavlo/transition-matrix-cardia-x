import unittest

from tm_ecg.modeling.triads import build_triad_memberships
from tm_ecg.ontology import map_ludb_text, map_ptbxl_axes, map_ptbxl_labels
from tm_ecg.signal.fiducials import verify_fiducial_order
from tm_ecg.types import BeatAcceptance, BeatFiducials


class OntologyAndTriadTests(unittest.TestCase):
    def test_ptbxl_mapping_and_paced_flag(self) -> None:
        labels = map_ptbxl_labels({"scp_codes": "{'NORM': 100, 'AFIB': 80}", "pacemaker": "1"})
        self.assertNotIn("Normal", labels)
        self.assertIn("AF", labels)
        self.assertIn("Paced", labels)

    def test_normal_is_suppressed_by_definite_abnormality(self) -> None:
        labels = map_ptbxl_labels({"scp_codes": "{'NORM': 100, 'PVC': 100}"})
        self.assertEqual(labels, ["PVC"])

    def test_ptbxl_mapping_uses_statement_type_aware_likelihoods(self) -> None:
        labels = map_ptbxl_labels(
            {"scp_codes": "{'NORM': 15, 'CRBBB': 35, 'AFIB': 0, 'PAC': 0}"}
        )
        self.assertNotIn("Normal", labels)
        self.assertNotIn("RBBB spectrum", labels)
        self.assertIn("AF", labels)
        self.assertIn("APB", labels)

    def test_all_relevant_form_statements_are_presence_coded(self) -> None:
        labels = map_ptbxl_labels(
            {"scp_codes": "{'STD_': 0, 'STE_': 0, 'NT_': 0, 'TAB_': 0}"}
        )
        self.assertEqual(labels, ["Other / unmapped"])
        axes = map_ptbxl_axes(
            {"scp_codes": "{'STD_': 0, 'STE_': 0, 'NT_': 0, 'TAB_': 0}"}
        )
        self.assertEqual(
            set(axes.repolarization),
            {"st_depression", "st_elevation", "t_abnormality"},
        )

    def test_ptbxl_pacing_uses_scp_and_localized_metadata(self) -> None:
        self.assertIn("Paced", map_ptbxl_labels({"scp_codes": "{'PACE': 0}"}))
        self.assertIn(
            "Paced",
            map_ptbxl_labels(
                {"scp_codes": "{}", "pacemaker": "ja, pacemaker"}
            ),
        )

    def test_source_index_quality_remains_unknown(self) -> None:
        axes = map_ptbxl_axes({"scp_codes": "{'NORM': 100}"})
        self.assertEqual(axes.quality, "unknown")

    def test_sinus_rhythm_alone_does_not_imply_global_normality(self) -> None:
        sinus_only = map_ptbxl_axes({"scp_codes": "{'SR': 0}"})
        self.assertEqual(sinus_only.rhythm, ("sinus",))
        self.assertEqual(sinus_only.normality, "unknown")
        self.assertEqual(map_ptbxl_labels({"scp_codes": "{'SR': 0}"}), ["Other / unmapped"])
        self.assertEqual(
            map_ptbxl_labels({"scp_codes": "{'NORM': 100, 'SR': 0}"}),
            ["Normal"],
        )

    def test_ludb_mapping(self) -> None:
        labels = map_ludb_text("Right bundle branch block; atrial flutter")
        self.assertIn("RBBB spectrum", labels)
        self.assertIn("AFL", labels)

    def test_ludb_normal_is_suppressed_by_definite_abnormality(self) -> None:
        labels = map_ludb_text("Normal sinus rhythm; right bundle branch block")
        self.assertEqual(labels, ["RBBB spectrum"])

    def test_ptbxl_mapping_accepts_parsed_codes_and_missing_pacemaker(self) -> None:
        labels = map_ptbxl_labels({"scp_codes": {"NORM": 100}, "pacemaker": float("nan")})
        self.assertEqual(labels, ["Normal"])

    def test_triad_memberships_and_fiducial_order(self) -> None:
        acceptances = [
            BeatAcceptance("b1", "r1", True, ["accepted"]),
            BeatAcceptance("b2", "r1", True, ["accepted"]),
            BeatAcceptance("b3", "r1", False, ["artifact"]),
            BeatAcceptance("b4", "r1", True, ["accepted"]),
            BeatAcceptance("b5", "r1", True, ["accepted"]),
        ]
        triads = build_triad_memberships("r1", acceptances)
        self.assertEqual(len(triads), 2)
        fiducials = BeatFiducials(
            beat_id="b1",
            record_id="r1",
            p_on=1,
            p_peak=2,
            p_off=3,
            qrs_on=4,
            r_peak=5,
            qrs_off=6,
            t_on=7,
            t_peak=8,
            t_off=9,
        )
        self.assertTrue(verify_fiducial_order(fiducials))


if __name__ == "__main__":
    unittest.main()
