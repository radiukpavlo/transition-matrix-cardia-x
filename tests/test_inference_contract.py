from tm_ecg.inference import _compatibility_set_from_scores


def test_inference_compatibility_set_applies_residual_and_normal_exclusivity() -> None:
    scores = {
        "AF": 0.9,
        "AFL": 0.1,
        "APB": 0.1,
        "PVC": 0.1,
        "Paced": 0.1,
        "RBBB spectrum": 0.1,
        "LBBB spectrum": 0.1,
        "Normal": 0.8,
        "Other / unmapped": 0.7,
    }
    assert _compatibility_set_from_scores(scores, None) == ["AF"]


def test_inference_empty_threshold_set_uses_residual_fallback() -> None:
    scores = {
        "AF": 0.1,
        "AFL": 0.1,
        "APB": 0.1,
        "PVC": 0.1,
        "Paced": 0.1,
        "RBBB spectrum": 0.1,
        "LBBB spectrum": 0.1,
        "Normal": 0.1,
        "Other / unmapped": 0.1,
    }
    assert _compatibility_set_from_scores(scores, None) == [
        "Other / unmapped"
    ]
