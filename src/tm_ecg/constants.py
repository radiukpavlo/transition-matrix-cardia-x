"""Project constants and locked ontology definitions."""

from __future__ import annotations

DATASET_MAP: dict[str, str] = {"b1": "ptbxl", "b2": "ludb"}

PROJECT_LABELS = [
    "Normal",
    "PVC",
    "APB",
    "RBBB spectrum",
    "LBBB spectrum",
    "AF",
    "AFL",
    "Paced",
    "Other / unmapped",
]

# Canonical multiaxial targets.  PROJECT_LABELS remains the compatibility
# projection consumed by the historical single-head model and reports.
AXIS_LABELS: dict[str, tuple[str, ...]] = {
    "rhythm": (
        "sinus",
        "sinus_bradycardia",
        "sinus_tachycardia",
        "af",
        "afl",
        "other_arrhythmia",
    ),
    "ectopy": ("none", "apb", "pvc", "mixed_ectopy", "unspecified_ectopy"),
    "conduction": ("none", "rbbb_spectrum", "lbbb_spectrum", "other_conduction"),
    "repolarization": (
        "none",
        "st_elevation",
        "st_depression",
        "t_abnormality",
        "other_st_t",
        "qtc_abnormality",
    ),
    "pacing": ("absent", "present"),
    "normality": ("unknown", "normal", "abnormal"),
    # Source-label indexes do not contain measured waveform quality.  Unknown
    # must remain distinct from an observed analyzable state until a signal
    # quality stage supplies evidence.
    "quality": ("unknown", "analyzable", "limited", "uninterpretable"),
}

SIGNATURE_FEATURES = (
    "rbbb_signature_score",
    "lbbb_signature_score",
    "pvc_signature_score",
    "af_signature_score",
    "paced_signature_score",
)

PTBXL_TO_PROJECT = {
    "NORM": "Normal",
    "PVC": "PVC",
    "VPB": "PVC",
    "PAC": "APB",
    "APB": "APB",
    "SPAC": "APB",
    "SVPB": "APB",
    "RBBB": "RBBB spectrum",
    "CRBBB": "RBBB spectrum",
    "IRBBB": "RBBB spectrum",
    "LBBB": "LBBB spectrum",
    "CLBBB": "LBBB spectrum",
    "AFIB": "AF",
    "AFLT": "AFL",
}

LUDB_TO_PROJECT = {
    "normal sinus rhythm": "Normal",
    "ventricular extrasystole": "PVC",
    "pvc": "PVC",
    "atrial extrasystole": "APB",
    "supraventricular ectopy": "APB",
    "right bundle branch block": "RBBB spectrum",
    "left bundle branch block": "LBBB spectrum",
    "atrial fibrillation": "AF",
    "atrial flutter": "AFL",
    "paced rhythm": "Paced",
    "pacemaker": "Paced",
}

LEADS_12 = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

ARCHIVE_NAMES = {
    "ptbxl": "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
    "ptbxl_plus": "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip",
    "ludb": "lobachevsky-university-electrocardiography-database-1.0.1.zip",
}

QT_CORRECTION_CODE = "QTcF"

B_COLUMNS = [
    "hr_med_bpm",
    "rr_med_ms",
    "rr_iqr_ms",
    "rr_sdnn_ms",
    "prematurity_index_min",
    "comp_pause_ratio_max",
    "pvc_like_beat_count",
    "apb_like_beat_count",
    "paced_like_beat_count",
    "af_irregularity_cv",
    "f_wave_power_ratio",
    "p_present_ratio",
    "p_amp_ii_med_mV",
    "p_dur_med_ms",
    "pr_med_ms",
    "pr_iqr_ms",
    "q_amp_ii_med_mV",
    "r_amp_ii_med_mV",
    "s_amp_ii_med_mV",
    "qrs_dur_med_ms",
    "qrs_dur_iqr_ms",
    "qrs_deformed_prob",
    "qrs_deformed_any",
    "qrs_fragmented_any",
    "qrs_wide_any",
    "r_prime_v1_any",
    "broad_r_v6_any",
    "st_level_v1_mV",
    "st_level_v5_mV",
    "st_slope_v5_uV_per_ms",
    "t_amp_v5_med_mV",
    "t_dur_med_ms",
    "t_inverted_right_any",
    "qt_med_ms",
    "qtc_med_ms",
    "qrs_net_area_i_mV_ms",
    "qrs_net_area_avf_mV_ms",
    "qrs_axis_deg",
    "qrs_axis_sin",
    "qrs_axis_cos",
    "rbbb_signature_score",
    "lbbb_signature_score",
    "pvc_signature_score",
    "af_signature_score",
    "paced_signature_score",
    "lead_quality_min_db",
    "delineation_confidence",
    "rhythm_valid_beat_fraction",
    "atrial_valid_beat_fraction",
    "qrs_valid_beat_fraction",
    "st_t_valid_beat_fraction",
    "atrial_lead_coverage",
    "qrs_lead_coverage",
    "st_t_lead_coverage",
    "detector_agreement",
    "analyzable_duration_s",
    "u_present_v2_any",
    "u_amp_v2_mV",
]
