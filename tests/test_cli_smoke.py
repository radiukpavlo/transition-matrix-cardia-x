"""Smoke tests for the CLI parser and subcommand registration."""

import unittest

from tm_ecg.cli import _build_parser


class CLISmokeTests(unittest.TestCase):
    def test_parser_builds_without_error(self) -> None:
        parser = _build_parser()
        self.assertIsNotNone(parser)

    def test_every_subcommand_is_registered(self) -> None:
        parser = _build_parser()
        subcommands = [
            "bootstrap-env",
            "ingest",
            "index",
            "splits",
            "preprocess",
            "pace",
            "rpeaks",
            "delineate",
            "triads",
            "extract-a",
            "train-classifier",
            "build-b",
            "fit-transition",
            "explain",
            "dss",
            "report",
            "release-audit",
            "freeze",
            "doctor",
        ]
        # Verify all subcommands are registered in the parser
        # Access _subparsers to check action choices
        for action in parser._subparsers._actions:
            if hasattr(action, "choices") and action.choices:
                registered = set(action.choices.keys())
                for cmd in subcommands:
                    with self.subTest(cmd=cmd):
                        self.assertIn(cmd, registered, f"Subcommand '{cmd}' not registered")

    def test_verbose_and_quiet_flags_exist(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["-v", "bootstrap-env"])
        self.assertEqual(args.verbose, 1)
        self.assertFalse(args.quiet)

        args_quiet = parser.parse_args(["-q", "bootstrap-env"])
        self.assertTrue(args_quiet.quiet)

    def test_double_verbose(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["-vv", "bootstrap-env"])
        self.assertEqual(args.verbose, 2)

    def test_dss_subcommand_has_required_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "dss", "--dataset", "b1",
            "--min-support", "5",
            "--max-rules-per-label", "10",
        ])
        self.assertEqual(args.dataset, "b1")
        self.assertEqual(args.min_support, 5)

    def test_explain_subcommand_has_split_arg(self) -> None:
        parser = _build_parser()
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                args = parser.parse_args(["explain", "--dataset", "b1", "--split", split])
                self.assertEqual(args.split, split)

    def test_pipeline_task_commands_match_cli_contract(self) -> None:
        parser = _build_parser()
        commands = [
            ["train-classifier", "--dataset", "ptbxl"],
            ["report", "--experiment", "b1"],
            ["freeze", "--experiment", "b1"],
            ["doctor"],
        ]
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(parser.parse_args(command).handler)

    def test_release_audit_accepts_pinned_evidence_inputs(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "release-audit",
            "--clinical-run-id", "strict_recheck_20260722",
            "--evidence-tag", "20260722",
        ])
        self.assertEqual(args.clinical_run_id, "strict_recheck_20260722")
        self.assertEqual(args.evidence_tag, "20260722")


if __name__ == "__main__":
    unittest.main()
