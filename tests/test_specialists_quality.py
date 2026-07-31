from __future__ import annotations

import numpy as np
import pytest

from tm_ecg.features.quality import compute_signal_quality
from tm_ecg.features.registry import (
    governed_12sl_feature_specs,
    governed_project_feature_specs,
)
from tm_ecg.modeling.specialists.atrial_rhythm import (
    atrial_rhythm_specialist,
)
from tm_ecg.modeling.specialists.conduction import conduction_specialist
from tm_ecg.modeling.specialists.ectopy import ectopy_specialist
from tm_ecg.modeling.specialists.normality import normality_specialist
from tm_ecg.modeling.specialists.pacing import pacing_specialist
from tm_ecg.modeling.specialists.repolarization import (
    repolarization_specialist,
)


def test_shared_quality_layer_detects_usable_and_dropped_leads() -> None:
    sampling_rate = 500.0
    time = np.arange(5000) / sampling_rate
    signal = np.column_stack(
        (
            0.5 * np.sin(2 * np.pi * 1.2 * time),
            np.zeros_like(time),
        )
    )
    quality = compute_signal_quality(
        signal,
        sampling_rate_hz=sampling_rate,
        lead_names=("II", "V1"),
        rpeaks=np.arange(250, 5000, 500),
    )

    assert quality.lead_dropout_mask == (False, True)
    assert quality.beat_count == 10
    assert quality.rpeak_detection_confidence > 0.8
    assert quality.eligible_lead_fraction == 0.5
    assert quality.globally_eligible is True


def test_atrial_and_ectopy_specialists_abstain_when_quality_is_inadequate() -> None:
    atrial = atrial_rhythm_specialist(
        [800.0, 610.0, 950.0],
        signal_eligible=False,
    )
    ectopy = ectopy_specialist([], signal_eligible=False)

    assert atrial.eligible is False
    assert atrial.probabilities["AF"] is None
    assert ectopy.eligible is False
    assert ectopy.probabilities["PVC"] is None


def test_specialists_return_probabilities_and_never_hard_override() -> None:
    atrial = atrial_rhythm_specialist(
        [520, 880, 610, 1030, 570, 940, 650, 1100, 540, 920],
        p_wave_confidence_by_lead={"II": [0.1] * 10, "V1": [0.15] * 10},
        p_to_qrs_association=[0.2] * 10,
    )
    beats = [
        {
            "prematurity_index": 0.7,
            "compensatory_pause_ratio": 1.8,
            "qrs_duration_ms": 145,
            "qrs_morphology_distance": 0.7,
            "preceding_p_wave_probability": 0.1,
        }
        for _ in range(8)
    ]
    ectopy = ectopy_specialist(beats)
    conduction = conduction_specialist(
        {
            "qrs_duration_ms": 150,
            "v1_rsr_score": 0.9,
            "v1_terminal_r_score": 0.8,
            "i_v6_terminal_s_score": 0.8,
            "morphology_confidence": 0.9,
        }
    )
    normality = normality_specialist(
        {
            "AF": atrial.probabilities["AF"],
            "PVC": ectopy.probabilities["PVC"],
            "RBBB": conduction.probabilities["RBBB spectrum"],
        },
        {"waveform_anomaly_score": 0.8},
    )

    assert atrial.eligible and 0.0 <= float(atrial.probabilities["AF"]) <= 1.0
    assert ectopy.eligible and float(ectopy.probabilities["PVC"]) > 0.5
    assert conduction.eligible
    assert normality.eligible
    assert set(normality.probabilities) == {
        "Normal",
        "abnormal",
        "specific_given_abnormal",
        "Other / unmapped_given_abnormal",
    }


def test_pacing_and_repolarization_specialists_are_quality_gated() -> None:
    sampling_rate = 500.0
    signal = np.zeros((2500, 2), dtype=float)
    rpeaks = np.asarray([500, 1000, 1500, 2000])
    for peak in rpeaks:
        signal[peak - 10, :] = 1.0
    pacing = pacing_specialist(
        signal,
        sampling_rate_hz=sampling_rate,
        rpeaks=rpeaks,
        paced_qrs_width_ms=150,
        paced_qrs_morphology_score=0.8,
    )
    repolarization = repolarization_specialist(
        {
            **{
                f"st_j60_mv_{lead}": 0.2
                for lead in (
                    "I",
                    "II",
                    "III",
                    "aVR",
                    "aVL",
                    "aVF",
                    "V1",
                    "V2",
                )
            },
            "fiducial_confidence": 0.9,
            "baseline_reference_stability": 0.9,
            "qtc_ms": 470,
        }
    )

    assert pacing.eligible
    assert pacing.probabilities["Paced"] is not None
    assert repolarization.eligible
    assert repolarization.probabilities["st_elevation"] is not None


def test_feature_governance_rejects_target_or_fold_features() -> None:
    specs = governed_project_feature_specs()
    assert specs
    assert all(spec.inference_safe for spec in specs.values())
    assert governed_12sl_feature_specs(("P_Amp_II", "QRS_Dur_Global"))
    with pytest.raises(ValueError, match="prohibited"):
        governed_12sl_feature_specs(("P_Amp_II", "target_AF"))

