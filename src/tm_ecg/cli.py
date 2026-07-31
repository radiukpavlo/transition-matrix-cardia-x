"""Command-line interface for the locked ECG baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from tm_ecg.config import ProjectConfig
from tm_ecg.clinical_validation import cli as clinical_validation_cli
from tm_ecg.log import setup_logging
from tm_ecg.modeling import fiducial_validation
from tm_ecg.modeling import confirmatory_split
from tm_ecg.modeling import target_migration
from tm_ecg.modeling import structured_experiment
from tm_ecg.modeling import precomputed_ensemble
from tm_ecg.modeling import sealed_waveform_features
from tm_ecg.modeling import semantic_targets
from tm_ecg import ontology_lint
from tm_ecg.stages import (
    bootstrap_env,
    delineate,
    dss,
    explain,
    features,
    fit_transition,
    freeze,
    index,
    ingest,
    pace,
    preprocess,
    report,
    release_audit,
    release_audit_v3,
    rpeaks,
    splits,
    train_classifier,
    train_signatures,
    triads,
)


StageFn = Callable[[ProjectConfig, argparse.Namespace], int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tm-ecg")
    parser.add_argument("--config", default="configs/defaults.toml")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (use -vv for debug)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ontology_lint_parser = subparsers.add_parser("ontology-lint")
    ontology_lint_parser.add_argument(
        "--ptbxl-statements",
        default=(
            "data/raw/ptbxl/ptb-xl-a-large-publicly-available-"
            "electrocardiography-dataset-1.0.3/scp_statements.csv"
        ),
    )
    ontology_lint_parser.add_argument(
        "--benchmark-mapping",
        default="clinical_validation/config/benchmark_mapping_v3.yaml",
    )
    ontology_lint_parser.add_argument(
        "--label-contract",
        default="configs/compatibility_label_contract_v4.yaml",
    )
    ontology_lint_parser.add_argument("--ptbxl-index")
    ontology_lint_parser.add_argument("--output")
    ontology_lint_parser.set_defaults(handler=ontology_lint.run)

    bootstrap = subparsers.add_parser("bootstrap-env")
    bootstrap.set_defaults(handler=bootstrap_env.run)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--source", choices=("zip", "download"), default="zip")
    ingest_parser.set_defaults(handler=ingest.run)

    index_parser = subparsers.add_parser("index")
    index_parser.set_defaults(handler=index.run)

    split_parser = subparsers.add_parser("splits")
    split_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    split_parser.set_defaults(handler=splits.run)

    confirmatory_split_parser = subparsers.add_parser(
        "create-confirmatory-split"
    )
    confirmatory_split_parser.add_argument(
        "--index",
        default="artifacts/manifests/ptbxl_index.parquet",
    )
    confirmatory_split_parser.add_argument(
        "--output",
        default="artifacts/manifests/sealed_internal_confirmatory_v1.json",
    )
    confirmatory_split_parser.add_argument(
        "--salt",
        default=confirmatory_split.DEFAULT_SPLIT_SALT,
        help="Deterministic split salt; only its SHA-256 is persisted.",
    )
    confirmatory_split_parser.set_defaults(handler=confirmatory_split.run)

    target_migration_parser = subparsers.add_parser(
        "audit-compatibility-target-migration"
    )
    target_migration_parser.add_argument(
        "--index",
        default="artifacts/manifests/ptbxl_index.parquet",
    )
    target_migration_parser.add_argument(
        "--historical-predictions",
        default="artifacts/reports/metrics/ptbxl_12sl_test_predictions.parquet",
    )
    target_migration_parser.add_argument(
        "--output",
        default="artifacts/reports/metrics/compatibility_target_migration_v4.json",
    )
    target_migration_parser.set_defaults(handler=target_migration.run)

    structured_experiment_parser = subparsers.add_parser(
        "experiment-structured-compatibility"
    )
    structured_experiment_parser.add_argument(
        "--index",
        default="artifacts/manifests/ptbxl_index.parquet",
    )
    structured_experiment_parser.add_argument(
        "--sealed-manifest",
        default="artifacts/manifests/sealed_internal_confirmatory_v1.json",
    )
    structured_experiment_parser.add_argument(
        "--output-dir",
        default="artifacts/reports/metrics/compatibility_structured_v4",
    )
    structured_experiment_parser.add_argument(
        "--estimators",
        default="hgb,logistic",
        help="Comma-separated base views: hgb, logistic, extra_trees.",
    )
    structured_experiment_parser.add_argument(
        "--crossfit-folds",
        type=int,
        default=5,
    )
    structured_experiment_parser.add_argument("--seed", type=int, default=1701)
    structured_experiment_parser.add_argument(
        "--project-feature-paths",
        default="",
        help=(
            "Comma-separated governed project waveform feature tables; "
            "never accepts labels or diagnostic statements as inputs."
        ),
    )
    structured_experiment_parser.set_defaults(handler=structured_experiment.run)

    ensemble_parser = subparsers.add_parser(
        "experiment-precomputed-ensemble"
    )
    ensemble_parser.add_argument(
        "--index",
        default="artifacts/manifests/ptbxl_index.parquet",
    )
    ensemble_parser.add_argument(
        "--sealed-manifest",
        default="artifacts/manifests/sealed_internal_confirmatory_v1.json",
    )
    ensemble_parser.add_argument(
        "--view-dirs",
        required=True,
        help="Two comma-separated governed development output directories.",
    )
    ensemble_parser.add_argument(
        "--output-dir",
        default="artifacts/reports/metrics/compatibility_precomputed_ensemble_v4",
    )
    ensemble_parser.add_argument(
        "--weights",
        default="0,0.25,0.5,0.75,1",
        help="First-view weight grid; the second weight is one minus it.",
    )
    ensemble_parser.add_argument("--crossfit-folds", type=int, default=5)
    ensemble_parser.add_argument("--seed", type=int, default=2901)
    ensemble_parser.set_defaults(handler=precomputed_ensemble.run)

    semantic_targets_parser = subparsers.add_parser(
        "build-semantic-targets"
    )
    semantic_targets_parser.add_argument("--source-measurements", required=True)
    semantic_targets_parser.add_argument("--specialist-oof", required=True)
    semantic_targets_parser.add_argument(
        "--contract",
        default="configs/semantic_target_contract_v3.yaml",
    )
    semantic_targets_parser.add_argument(
        "--output",
        default="data/processed/features/B1_semantic_targets_v3_oof.parquet",
    )
    semantic_targets_parser.set_defaults(handler=semantic_targets.run)

    sealed_waveform_parser = subparsers.add_parser(
        "extract-sealed-waveform-features"
    )
    sealed_waveform_parser.add_argument(
        "--index",
        default="artifacts/manifests/ptbxl_index.parquet",
    )
    sealed_waveform_parser.add_argument(
        "--sealed-manifest",
        default="artifacts/manifests/sealed_internal_confirmatory_v1.json",
    )
    sealed_waveform_parser.add_argument(
        "--existing-features",
        default="data/processed/features/B1_raw_train.parquet",
    )
    sealed_waveform_parser.add_argument(
        "--output",
        default="data/processed/features/B1_raw_sealed_development.parquet",
    )
    sealed_waveform_parser.set_defaults(handler=sealed_waveform_features.run)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    preprocess_parser.set_defaults(handler=preprocess.run)

    pace_parser = subparsers.add_parser("pace")
    pace_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    pace_parser.set_defaults(handler=pace.run)

    rpeak_parser = subparsers.add_parser("rpeaks")
    rpeak_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    rpeak_parser.set_defaults(handler=rpeaks.run)

    delineate_parser = subparsers.add_parser("delineate")
    delineate_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    delineate_parser.set_defaults(handler=delineate.run)

    triads_parser = subparsers.add_parser("triads")
    triads_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    triads_parser.set_defaults(handler=triads.run)

    fiducial_validation_parser = subparsers.add_parser("validate-fiducials")
    fiducial_validation_parser.add_argument(
        "--bootstrap-replicates", type=int, default=1000
    )
    fiducial_validation_parser.set_defaults(handler=fiducial_validation.run)

    train_parser = subparsers.add_parser("train-classifier")
    train_parser.add_argument("--dataset", choices=("ptbxl",), required=True)
    train_parser.set_defaults(handler=train_classifier.run)

    signature_parser = subparsers.add_parser("train-signatures")
    signature_parser.add_argument("--dataset", choices=("b1",), default="b1")
    signature_parser.set_defaults(handler=train_signatures.run)

    extract_parser = subparsers.add_parser("extract-a")
    extract_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    extract_parser.set_defaults(handler=triads.extract_latents)

    build_b = subparsers.add_parser("build-b")
    build_b.add_argument("--dataset", choices=("b1", "b2"), required=True)
    build_b.set_defaults(handler=features.run)

    fit_parser = subparsers.add_parser("fit-transition")
    fit_parser.add_argument("--dataset", choices=("b1", "b2"), required=True)
    fit_parser.set_defaults(handler=fit_transition.run)

    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    explain_parser.add_argument("--dataset", choices=("b1", "b2"), default="b1")
    explain_parser.set_defaults(handler=explain.run)


    dss_parser = subparsers.add_parser("dss")
    dss_parser.add_argument("--dataset", choices=("b1", "b2"), required=True)
    dss_parser.add_argument("--alpha", type=float, default=0.65)
    dss_parser.add_argument("--min-support", type=int, default=5)
    dss_parser.add_argument("--max-depth", type=int, default=3)
    dss_parser.add_argument("--max-rules-per-label", type=int, default=25)
    dss_parser.add_argument("--top-transition-features", type=int, default=0)
    dss_parser.add_argument("--predictions", default=None, help="Optional CSV/Parquet/JSON prediction audit table for Step 1.")
    dss_parser.add_argument("--prediction-id-col", default="record_id")
    dss_parser.add_argument("--true-label-col", default="true_label")
    dss_parser.add_argument("--pred-label-col", default="pred_label")
    dss_parser.add_argument("--confidence-col", default="confidence")
    dss_parser.add_argument("--min-precision", type=float, default=0.85)
    dss_parser.add_argument("--min-recall", type=float, default=0.85)
    dss_parser.add_argument("--min-f1", type=float, default=0.85)
    dss_parser.add_argument("--min-specificity", type=float, default=0.85)
    dss_parser.add_argument("--min-class-support", type=int, default=20)
    dss_parser.add_argument("--min-calibration-confidence", type=float, default=0.0)
    dss_parser.add_argument("--no-clinical-bins", action="store_true")
    dss_parser.add_argument("--no-reducts", action="store_true")
    dss_parser.add_argument("--no-predicate-rules", action="store_true")
    dss_parser.add_argument(
        "--allow-research-feature-fallback",
        action="store_true",
        help="Deprecated audit flag; strict execution records but never applies a fallback.",
    )
    dss_parser.set_defaults(handler=dss.run)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--experiment", required=True)
    report_parser.set_defaults(handler=report.run)

    release_audit_parser = subparsers.add_parser("release-audit")
    release_audit_parser.add_argument(
        "--clinical-run-id",
        default="final_audited_20260721",
        help="Pinned clinical-validation result directory under clinical_validation/results.",
    )
    release_audit_parser.add_argument(
        "--evidence-tag",
        default="20260721",
        help="Pinned suffix for verification inputs and release-audit outputs.",
    )
    release_audit_parser.set_defaults(handler=release_audit.run)

    release_audit_v3_parser = subparsers.add_parser("release-audit-v3")
    release_audit_v3_parser.add_argument(
        "--clinical-results-dir",
        default="clinical_validation/results/implementation_v3_blocked",
    )
    release_audit_v3_parser.add_argument(
        "--coding-gate",
        default=(
            "clinical_validation/results/v3_physician_coding_lint/"
            "physician_coding_gate.json"
        ),
    )
    release_audit_v3_parser.add_argument(
        "--compatibility-metrics",
        default=(
            "artifacts/reports/metrics/compatibility_structured_v4/"
            "structured_compatibility_development_metrics.json"
        ),
    )
    release_audit_v3_parser.add_argument(
        "--environment-audit",
        default="artifacts/reports/metrics/environment_audit.json",
    )
    release_audit_v3_parser.add_argument("--transition-gate")
    release_audit_v3_parser.add_argument("--dss-gate")
    release_audit_v3_parser.add_argument(
        "--output-dir",
        default="artifacts/reports/release/cardia_x_v3",
    )
    release_audit_v3_parser.set_defaults(handler=release_audit_v3.run)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--experiment", default="default")
    freeze_parser.set_defaults(handler=freeze.run)

    doctor_parser = subparsers.add_parser("doctor")
    from tm_ecg.stages import doctor

    doctor_parser.set_defaults(handler=doctor.run)
    clinical_validation_cli.register_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    verbosity = 0 if args.quiet else 1 + args.verbose
    setup_logging(verbosity)
    config = ProjectConfig.load(Path(args.config))
    config.ensure_directories()
    handler: StageFn = args.handler
    return handler(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
