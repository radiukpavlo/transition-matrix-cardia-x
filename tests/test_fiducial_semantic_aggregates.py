import pytest

from tm_ecg.features.signatures import robust_semantic_aggregates
from tm_ecg.signal.fiducials import cross_lead_fiducial_consistency
from tm_ecg.types import BeatFiducials


def test_cross_lead_fiducials_report_missingness_and_timing() -> None:
    leads = {
        "II": BeatFiducials(
            "b",
            "r",
            qrs_on=100,
            r_peak=120,
            qrs_off=150,
            confidence=0.9,
        ),
        "V1": BeatFiducials(
            "b",
            "r",
            qrs_on=102,
            r_peak=121,
            qrs_off=151,
            confidence=0.8,
        ),
    }
    result = cross_lead_fiducial_consistency(
        leads,
        sampling_rate_hz=500.0,
    )
    assert result["lead_count"] == 2
    assert result["cross_lead_consistent"] == 0  # P/T missing: fail closed.
    assert result["cross_lead_timing_mad_ms"] is not None


def test_robust_semantic_aggregates_are_missingness_explicit() -> None:
    result = robust_semantic_aggregates(
        [{"score": 0.1}, {"score": 0.9}, {"score": None}, {}],
        feature_names=["score"],
    )
    assert result["score_available_fraction"] == 0.5
    assert result["score_median"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="forbidden"):
        robust_semantic_aggregates(
            [{"target_label": 1}],
            feature_names=["target_label"],
        )
