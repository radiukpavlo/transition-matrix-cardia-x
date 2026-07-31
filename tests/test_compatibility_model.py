from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.modeling.compatibility import (
    _build_compatibility_target_frame,
    _canonical_12sl_feature_allowlist,
    _class_metrics,
    _load_waveform_b_features,
    _multilabel_cluster_bootstrap,
    _prediction_coverage,
    _require_artifact_hash,
    _require_reproduced_validation_evidence,
    _split_audit,
    _validate_12sl_feature_schema,
    expected_calibration_error,
    normalize_compatibility_labels,
    project_compatibility_predictions,
    select_f1_threshold,
    select_joint_thresholds,
)


def _validation_evidence_frame(
    record_ids: list[str],
    truth: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    split_manifest_sha256: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index, record_id in enumerate(record_ids):
        row: dict[str, object] = {
            "record_id": record_id,
            "evaluation_partition": "validation_model_selection",
            "threshold_partition": "validation_only",
            "split_manifest_sha256": split_manifest_sha256,
        }
        for column, label in enumerate(PROJECT_LABELS):
            key = label.lower().replace(" / ", "_").replace(" ", "_")
            row[f"true::{key}"] = int(truth[row_index, column])
            row[f"probability::{key}"] = float(probabilities[row_index, column])
            row[f"predicted::{key}"] = int(predictions[row_index, column])
        rows.append(row)
    return pd.DataFrame(rows)


def test_normal_is_removed_when_an_abnormal_compatibility_label_cooccurs() -> None:
    assert normalize_compatibility_labels("Normal,PVC") == ["PVC"]
    assert normalize_compatibility_labels("AF,Other / unmapped") == ["AF"]
    assert normalize_compatibility_labels("Normal") == ["Normal"]
    assert normalize_compatibility_labels("") == ["Other / unmapped"]


def test_threshold_selection_is_validation_only_and_deterministic() -> None:
    targets = [1, 1, 0, 0]
    probabilities = [0.9, 0.7, 0.4, 0.1]

    first = select_f1_threshold(targets, probabilities)
    second = select_f1_threshold(targets, probabilities)

    assert first == second
    assert first["threshold_partition"] == "validation_only"
    assert first["validation_f1"] == 1.0
    assert 0.4 < first["threshold"] <= 0.7


def test_expected_calibration_error_is_bounded_and_rejects_misalignment() -> None:
    error = expected_calibration_error([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert error == pytest.approx(0.15)
    assert 0.0 <= error <= 1.0

    with pytest.raises(ValueError, match="aligned"):
        expected_calibration_error([1], [0.8, 0.2])


def test_split_audit_requires_patient_isolation() -> None:
    isolated = pd.DataFrame(
        [
            {"record_id": "1", "patient_id": "a", "split": "train"},
            {"record_id": "2", "patient_id": "b", "split": "val"},
            {"record_id": "3", "patient_id": "c", "split": "test"},
        ]
    )
    assert _split_audit(isolated)["patient_overlap_count"] == 0

    leaking = isolated.copy()
    leaking.loc[2, "patient_id"] = "a"
    with pytest.raises(RuntimeError, match="Patient leakage"):
        _split_audit(leaking)


def test_metric_interval_resamples_patient_clusters() -> None:
    metrics = _class_metrics(
        [1, 1, 0, 0],
        [0.9, 0.8, 0.2, 0.1],
        0.5,
        seed=17,
        patient_groups=["p1", "p1", "p2", "p3"],
        replicates=50,
    )

    assert metrics["f1"] == 1.0
    assert metrics["bootstrap_cluster_count"] == 3
    assert metrics["confidence_interval_method"] == "patient_cluster_bootstrap_percentile_50"


def test_class_metrics_accepts_ontology_projected_predictions() -> None:
    metrics = _class_metrics(
        [1, 0],
        [0.9, 0.9],
        0.5,
        seed=17,
        replicates=20,
        predictions=[1, 0],
    )

    assert metrics["f1"] == 1.0
    assert metrics["fp"] == 0


def test_prediction_projection_enforces_normal_exclusivity_and_nonempty_output() -> None:
    probabilities = np.full((3, len(PROJECT_LABELS)), 0.1)
    normal = PROJECT_LABELS.index("Normal")
    af = PROJECT_LABELS.index("AF")
    lbbb = PROJECT_LABELS.index("LBBB spectrum")
    probabilities[0, normal] = 0.9
    probabilities[0, af] = 0.8
    probabilities[1, normal] = 0.7
    probabilities[1, af] = 0.9
    probabilities[2, lbbb] = 0.4

    predicted = project_compatibility_predictions(
        probabilities,
        np.full(len(PROJECT_LABELS), 0.5),
    )

    assert predicted[0, normal] and predicted[0].sum() == 1
    assert predicted[1, af] and not predicted[1, normal]
    assert predicted[2, lbbb] and predicted[2].sum() == 1


def test_prediction_projection_removes_residual_when_specific_label_is_present() -> None:
    probabilities = np.full((1, len(PROJECT_LABELS)), 0.1)
    probabilities[0, PROJECT_LABELS.index("AF")] = 0.9
    probabilities[0, PROJECT_LABELS.index("Other / unmapped")] = 0.95
    predicted = project_compatibility_predictions(
        probabilities,
        np.full(len(PROJECT_LABELS), 0.5),
    )
    assert predicted[0, PROJECT_LABELS.index("AF")]
    assert not predicted[0, PROJECT_LABELS.index("Other / unmapped")]


def test_prediction_coverage_is_derived_from_non_abstained_rows() -> None:
    predictions = np.asarray([[1, 0], [0, 1], [0, 0]], dtype=int)

    coverage = _prediction_coverage(predictions)

    assert coverage["analyzable_coverage"] == pytest.approx(2 / 3)
    assert coverage["abstention_rate"] == pytest.approx(1 / 3)
    assert "emitted compatibility label" in coverage["analyzable_coverage_definition"]

    projected = project_compatibility_predictions(
        np.full((2, len(PROJECT_LABELS)), 0.1),
        np.full(len(PROJECT_LABELS), 0.9),
    )
    projected_coverage = _prediction_coverage(projected)
    assert projected_coverage["analyzable_coverage"] == 1.0
    assert projected_coverage["abstention_rate"] == 0.0


def test_joint_threshold_selection_is_deterministic_and_never_degrades_exact_match() -> None:
    truth = np.zeros((6, len(PROJECT_LABELS)), dtype=int)
    probabilities = np.full_like(truth, 0.05, dtype=float)
    labels = [
        "Normal",
        "AF",
        "RBBB spectrum",
        "Normal",
        "AF",
        "Other / unmapped",
    ]
    for row, label in enumerate(labels):
        column = PROJECT_LABELS.index(label)
        truth[row, column] = 1
        probabilities[row, column] = 0.75
    # Create ontology conflicts that deterministic projection must resolve.
    probabilities[0, PROJECT_LABELS.index("AF")] = 0.55
    probabilities[1, PROJECT_LABELS.index("Normal")] = 0.55
    initial = np.full(len(PROJECT_LABELS), 0.5)

    first_thresholds, first_audit = select_joint_thresholds(
        truth, probabilities, initial, maximum_passes=3
    )
    second_thresholds, second_audit = select_joint_thresholds(
        truth, probabilities, initial, maximum_passes=3
    )
    prediction = project_compatibility_predictions(probabilities, first_thresholds)

    assert np.array_equal(first_thresholds, second_thresholds)
    assert first_audit == second_audit
    assert (
        first_audit["final_objective"]["compatibility_subset_exact_match"]
        >= first_audit["initial_objective"]["compatibility_subset_exact_match"]
    )
    assert prediction.any(axis=1).all()
    assert not (
        prediction[:, PROJECT_LABELS.index("Normal")]
        & (prediction.sum(axis=1) > 1)
    ).any()


def test_global_metric_intervals_resample_patient_clusters() -> None:
    truth = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]])
    prediction = truth.copy()

    intervals = _multilabel_cluster_bootstrap(
        truth,
        prediction,
        ["p1", "p1", "p2", "p3"],
        seed=17,
        replicates=30,
    )

    assert intervals["bootstrap_cluster_count"] == 3
    assert intervals["bootstrap_replicates"] == 30
    assert intervals["intervals"]["compatibility_subset_exact_match"] == [1.0, 1.0]


def test_waveform_b_features_are_prefixed_and_split_bound(tmp_path) -> None:
    manifest = pd.DataFrame(
        [
            {"record_id": "1", "split": "train"},
            {"record_id": "2", "split": "val"},
            {"record_id": "3", "split": "test"},
        ]
    )
    for split, record_id in (("train", "1"), ("val", "2"), ("test", "3")):
        (tmp_path / f"B1_raw_{split}.csv").write_text(
            f"record_id,hr_med_bpm,labels\n{record_id},72,AF\n",
            encoding="utf-8",
        )
    config = SimpleNamespace(paths=SimpleNamespace(features=tmp_path))

    frame, columns, provenance = _load_waveform_b_features(config, manifest)

    assert columns == ["waveform_b::hr_med_bpm"]
    assert frame["record_id"].tolist() == ["1", "2", "3"]
    assert "labels" not in frame.columns
    assert provenance["waveform_b_feature_count"] == 1

    (tmp_path / "B1_raw_test.csv").write_text(
        "record_id,hr_med_bpm\nwrong,72\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="IDs do not match"):
        _load_waveform_b_features(config, manifest)


def test_12sl_schema_is_constrained_to_canonical_feature_description_ids(
    tmp_path,
) -> None:
    description = tmp_path / "feature_description.csv"
    description.write_text(
        "id,description\n"
        "P_Amp_X,P amplitude\n"
        "HR__Global,heart rate\n"
        "P_Term_V1,P terminal force\n",
        encoding="utf-8",
    )

    allowed = _canonical_12sl_feature_allowlist(description)
    assert "P_Amp_I" in allowed
    assert "P_Amp_V6" in allowed
    assert "P_Term_V1" in allowed
    assert "P_Term_V2" not in allowed

    audit = _validate_12sl_feature_schema(
        ["ecg_id", "P_Amp_I", "P_Amp_V6", "HR__Global", "P_Term_V1"],
        description,
    )
    assert audit["feature_allowlist_expanded_count"] == 14
    assert audit["feature_allowlist_policy"].startswith("canonical")

    with pytest.raises(RuntimeError, match="outside the canonical"):
        _validate_12sl_feature_schema(
            ["ecg_id", "P_Amp_I", "target_AF"],
            description,
        )


def test_validation_target_frame_does_not_parse_held_out_labels() -> None:
    class HeldOutLabel:
        def __str__(self) -> str:
            raise AssertionError("held-out label was parsed")

    frame = pd.DataFrame(
        [
            {"split": "train", "labels": "AF"},
            {"split": "val", "labels": "Normal"},
            {"split": "test", "labels": HeldOutLabel()},
        ]
    )

    targets, support = _build_compatibility_target_frame(frame, ("train", "val"))

    assert set(support) == {"train", "val"}
    assert targets.loc[0, "target::AF"] == 1
    assert targets.loc[1, "target::Normal"] == 1
    assert targets.loc[2].isna().all()


def test_reproduced_validation_evidence_is_checked_before_test_scoring() -> None:
    record_ids = ["v2", "v1"]
    truth = np.zeros((2, len(PROJECT_LABELS)), dtype=int)
    truth[0, PROJECT_LABELS.index("AF")] = 1
    truth[1, PROJECT_LABELS.index("Normal")] = 1
    probabilities = np.full((2, len(PROJECT_LABELS)), 0.05, dtype=float)
    probabilities[0, PROJECT_LABELS.index("AF")] = 0.9
    probabilities[1, PROJECT_LABELS.index("Normal")] = 0.8
    predictions = project_compatibility_predictions(
        probabilities,
        np.full(len(PROJECT_LABELS), 0.5),
    ).astype(int)
    split_hash = "a" * 64
    evidence = _validation_evidence_frame(
        record_ids,
        truth,
        probabilities,
        predictions,
        split_manifest_sha256=split_hash,
    ).iloc[::-1]

    _require_reproduced_validation_evidence(
        evidence,
        record_ids,
        truth,
        probabilities,
        predictions,
        split_manifest_sha256=split_hash,
    )

    af_key = "probability::af"
    drifted = evidence.copy()
    drifted.loc[drifted["record_id"].eq("v2"), af_key] += 1e-4
    with pytest.raises(RuntimeError, match="probability evidence is not reproducible"):
        _require_reproduced_validation_evidence(
            drifted,
            record_ids,
            truth,
            probabilities,
            predictions,
            split_manifest_sha256=split_hash,
        )


def test_sealed_artifact_hash_rejects_mutation(tmp_path) -> None:
    import hashlib

    artifact = tmp_path / "predictions.parquet"
    artifact.write_bytes(b"sealed")
    expected = hashlib.sha256(b"sealed").hexdigest()

    _require_artifact_hash(
        artifact,
        expected,
        artifact_name="Frozen predictions",
    )
    artifact.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _require_artifact_hash(
            artifact,
            expected,
            artifact_name="Frozen predictions",
        )
