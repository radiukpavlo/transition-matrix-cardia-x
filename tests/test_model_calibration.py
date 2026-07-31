from tm_ecg.modeling.calibration import (
    compatibility_from_calibrated_axes,
    fit_temperature_calibration,
)
from tm_ecg.modeling.training import _per_class_validation_metrics


def test_temperature_calibration_uses_validation_rows_and_is_deterministic() -> None:
    logits = [[10.0], [-10.0], [-10.0], [10.0]]
    targets = [[1.0], [0.0], [1.0], [0.0]]
    first = fit_temperature_calibration(logits, targets, ["finding"])
    second = fit_temperature_calibration(logits, targets, ["finding"])
    assert first == second
    assert first["fit_partition"] == "validation_only"
    assert first["temperature"] > 1.0
    assert first["after"]["negative_log_likelihood"] < first["before"][
        "negative_log_likelihood"
    ]


def test_compatibility_projection_uses_calibrated_axis_thresholds() -> None:
    scores = {
        "rhythm": {"sinus": 0.1, "af": 0.8, "afl": 0.05},
        "ectopy": {"pvc": 0.2, "apb": 0.1},
        "conduction": {"rbbb_spectrum": 0.1, "lbbb_spectrum": 0.1},
        "pacing": {"present": 0.05},
        "repolarization": {"none": 0.9},
    }
    calibration = {
        "axes": {
            "rhythm": {
                "class_thresholds": {
                    "af": {"threshold": 0.7},
                    "afl": {"threshold": 0.7},
                    "sinus": {"threshold": 0.7},
                }
            }
        }
    }
    label, trace = compatibility_from_calibrated_axes(scores, calibration)
    assert label == "AF"
    assert trace["source"] == "calibrated_axis_projection"


def test_per_class_metrics_use_validation_threshold_on_held_out_rows() -> None:
    metrics = _per_class_validation_metrics(
        logits=[[3.0], [-3.0], [2.0], [-2.0]],
        targets=[[1.0], [0.0], [0.0], [1.0]],
        calibration={
            "temperature": 1.0,
            "class_thresholds": {"finding": {"threshold": 0.5}},
        },
        labels=["finding"],
    )["finding"]

    assert metrics["evaluation_partition"] == "held_out_test"
    assert metrics["threshold_partition"] == "validation_only"
    assert metrics["support"] == 2
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f1"] == 0.5
