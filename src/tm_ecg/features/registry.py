"""Feature dictionary and B-matrix schema helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from tm_ecg.constants import B_COLUMNS
import json

from tm_ecg.features.formulas import (
    RecordMeasurements,
    compute_feature_quality_states,
    compute_record_features,
)
from tm_ecg.types import FeatureDictionaryRow


FEATURE_SPECS: dict[str, tuple[str, str, str, str]] = {
    "hr_med_bpm": ("rhythm", "bpm", "continuous", "F1"),
    "rr_med_ms": ("rhythm", "ms", "continuous", "F2"),
    "rr_iqr_ms": ("rhythm", "ms", "continuous", "F2"),
    "rr_sdnn_ms": ("rhythm", "ms", "continuous", "F3"),
    "prematurity_index_min": ("rhythm", "ratio", "bounded", "F4"),
    "comp_pause_ratio_max": ("rhythm", "ratio", "bounded", "F5"),
    "pvc_like_beat_count": ("burden", "count", "count", "F6"),
    "apb_like_beat_count": ("burden", "count", "count", "F6"),
    "paced_like_beat_count": ("burden", "count", "count", "F6"),
    "af_irregularity_cv": ("rhythm", "ratio", "bounded", "F7"),
    "f_wave_power_ratio": ("atrial", "ratio", "bounded", "F8"),
    "p_present_ratio": ("atrial", "ratio", "bounded", "F9"),
    "p_amp_ii_med_mV": ("atrial", "mV", "continuous", "F10"),
    "p_dur_med_ms": ("atrial", "ms", "continuous", "F11"),
    "pr_med_ms": ("atrial", "ms", "continuous", "F12"),
    "pr_iqr_ms": ("atrial", "ms", "continuous", "F12"),
    "q_amp_ii_med_mV": ("qrs", "mV", "continuous", "F10"),
    "r_amp_ii_med_mV": ("qrs", "mV", "continuous", "F10"),
    "s_amp_ii_med_mV": ("qrs", "mV", "continuous", "F10"),
    "qrs_dur_med_ms": ("qrs", "ms", "continuous", "F13"),
    "qrs_dur_iqr_ms": ("qrs", "ms", "continuous", "F13"),
    "qrs_deformed_prob": ("qrs", "prob", "bounded", "F14"),
    "qrs_deformed_any": ("qrs", "binary", "binary", "F14"),
    "qrs_fragmented_any": ("qrs", "binary", "binary", "F15"),
    "qrs_wide_any": ("qrs", "binary", "binary", "F15"),
    "r_prime_v1_any": ("qrs", "binary", "binary", "F16"),
    "broad_r_v6_any": ("qrs", "binary", "binary", "F16"),
    "st_level_v1_mV": ("st", "mV", "continuous", "F17"),
    "st_level_v5_mV": ("st", "mV", "continuous", "F17"),
    "st_slope_v5_uV_per_ms": ("st", "uV/ms", "continuous", "F18"),
    "t_amp_v5_med_mV": ("t", "mV", "continuous", "F19"),
    "t_dur_med_ms": ("t", "ms", "continuous", "F19"),
    "t_inverted_right_any": ("t", "binary", "binary", "F20"),
    "qt_med_ms": ("qt", "ms", "continuous", "F21"),
    "qtc_med_ms": ("qt", "ms", "continuous", "F21"),
    "qrs_net_area_i_mV_ms": ("axis", "mV*ms", "continuous", "F22"),
    "qrs_net_area_avf_mV_ms": ("axis", "mV*ms", "continuous", "F22"),
    "qrs_axis_deg": ("axis", "deg", "circular", "F23"),
    "qrs_axis_sin": ("axis", "unitless", "continuous", "F23"),
    "qrs_axis_cos": ("axis", "unitless", "continuous", "F23"),
    "rbbb_signature_score": ("signature", "logodds", "continuous", "F24"),
    "lbbb_signature_score": ("signature", "logodds", "continuous", "F24"),
    "pvc_signature_score": ("signature", "logodds", "continuous", "F24"),
    "af_signature_score": ("signature", "logodds", "continuous", "F24"),
    "paced_signature_score": ("signature", "logodds", "continuous", "F24"),
    "lead_quality_min_db": ("quality", "dB", "continuous", "F25"),
    "delineation_confidence": ("quality", "0-1", "bounded", "F26"),
    "rhythm_valid_beat_fraction": ("quality", "0-1", "bounded", "F28"),
    "atrial_valid_beat_fraction": ("quality", "0-1", "bounded", "F28"),
    "qrs_valid_beat_fraction": ("quality", "0-1", "bounded", "F28"),
    "st_t_valid_beat_fraction": ("quality", "0-1", "bounded", "F28"),
    "atrial_lead_coverage": ("quality", "0-1", "bounded", "F29"),
    "qrs_lead_coverage": ("quality", "0-1", "bounded", "F29"),
    "st_t_lead_coverage": ("quality", "0-1", "bounded", "F29"),
    "detector_agreement": ("quality", "0-1", "bounded", "F30"),
    "analyzable_duration_s": ("quality", "s", "continuous", "F31"),
    "u_present_v2_any": ("u", "binary", "binary", "F27"),
    "u_amp_v2_mV": ("u", "mV", "continuous", "F27"),
}


OPTIONAL_COLUMNS = {"u_present_v2_any", "u_amp_v2_mV"}
SIGNATURE_COLUMNS = {
    "rbbb_signature_score",
    "lbbb_signature_score",
    "pvc_signature_score",
    "af_signature_score",
    "paced_signature_score",
}


@dataclass(frozen=True, slots=True)
class GovernedFeatureSpec:
    feature_id: str
    version: str
    units: str
    lead_scope: str
    algorithm_or_source: str
    clinical_interpretation: str
    valid_range: str
    missingness_rule: str
    quality_prerequisites: str
    inference_safe: bool
    target_leakage_risk: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _lead_scope(feature_id: str) -> str:
    lowered = feature_id.lower()
    for lead in (
        "avf",
        "avl",
        "avr",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "ii",
        "iii",
        "i",
    ):
        if lowered.endswith(f"_{lead}") or f"_{lead}_" in lowered:
            return lead.upper()
    return "global_or_multilead"


def governed_project_feature_specs() -> dict[str, GovernedFeatureSpec]:
    """Return governance records for every project-resident B feature."""

    output: dict[str, GovernedFeatureSpec] = {}
    for feature_id in B_COLUMNS:
        family, units, value_type, formula = FEATURE_SPECS[feature_id]
        output[feature_id] = GovernedFeatureSpec(
            feature_id=feature_id,
            version="cardia_x_feature_v3",
            units=units,
            lead_scope=_lead_scope(feature_id),
            algorithm_or_source=f"project_waveform_formula::{formula}",
            clinical_interpretation=f"{family} measurement ({value_type})",
            valid_range=(
                "[0,1]"
                if value_type in {"binary", "bounded"}
                else "finite; domain bounds enforced by feature formula"
            ),
            missingness_rule=(
                "missing when required fiducials/leads fail quality prerequisites"
            ),
            quality_prerequisites=f"{family}_feature_state must be usable",
            inference_safe=True,
            target_leakage_risk=(
                "medium_requires_cross_fitted_generation"
                if feature_id in SIGNATURE_COLUMNS
                else "low_waveform_measurement_only"
            ),
        )
    return output


FEATURE_NAME_DENYLIST = (
    "scp",
    "diagnostic_statement",
    "source_label",
    "compatibility_label",
    "target_",
    "strat_fold",
    "fold_assignment",
    "validation_outcome",
    "benchmark",
)


def governed_12sl_feature_specs(
    feature_names: list[str] | tuple[str, ...],
) -> dict[str, GovernedFeatureSpec]:
    """Govern a concrete allowlisted PTB-XL+ 12SL measurement schema."""

    output: dict[str, GovernedFeatureSpec] = {}
    rejected = [
        feature
        for feature in feature_names
        if any(token in feature.lower() for token in FEATURE_NAME_DENYLIST)
    ]
    if rejected:
        raise ValueError(
            "Target-leaking or partition-derived features are prohibited: "
            f"{sorted(rejected)}"
        )
    for feature_id in feature_names:
        output[feature_id] = GovernedFeatureSpec(
            feature_id=feature_id,
            version="ptbxl_plus_12sl_1.0.1",
            units="source_dictionary",
            lead_scope=_lead_scope(feature_id),
            algorithm_or_source="PTB-XL+ 12SL numeric measurement",
            clinical_interpretation="vendor-derived ECG measurement; see feature_description.csv",
            valid_range="finite source-defined measurement range",
            missingness_rule="median imputation plus explicit missingness indicator",
            quality_prerequisites="source measurement available; missingness indicator otherwise",
            inference_safe=True,
            target_leakage_risk="low_measurement_only_allowlist_enforced",
        )
    return output


def feature_dictionary_rows() -> list[FeatureDictionaryRow]:
    return [
        FeatureDictionaryRow(
            column=column,
            family=FEATURE_SPECS[column][0],
            unit=FEATURE_SPECS[column][1],
            value_type=FEATURE_SPECS[column][2],
            level="record",
            formula=FEATURE_SPECS[column][3],
            included=column not in SIGNATURE_COLUMNS,
            notes=(
                "Optional"
                if column in OPTIONAL_COLUMNS
                else "Declared F24 feature; unavailable until calibrated signature-score artifact is fitted"
                if column in SIGNATURE_COLUMNS
                else ""
            ),
        )
        for column in B_COLUMNS
    ]


def feature_types() -> dict[str, str]:
    return {column: FEATURE_SPECS[column][2] for column in B_COLUMNS}


def build_raw_feature_row(record: RecordMeasurements, thresholds: dict[str, object]) -> dict[str, object]:
    features = compute_record_features(record, thresholds)
    quality = compute_feature_quality_states(record, thresholds)
    row: dict[str, object] = {column: features.get(column) for column in B_COLUMNS}
    row["record_id"] = record.record_id
    row["qtc_formula_code"] = features["qtc_formula_code"]
    row["feature_quality_json"] = json.dumps(
        {family: state.to_dict() for family, state in quality.items()},
        ensure_ascii=False,
        sort_keys=True,
    )
    for family, state in quality.items():
        row[f"{family}_feature_state"] = state.state
    return row


def fit_columns(rows: list[dict[str, object]], optional_missingness_threshold: float = 0.2) -> list[str]:
    selected = []
    row_count = max(len(rows), 1)
    for column in B_COLUMNS:
        if column == "qrs_axis_deg":
            continue
        missingness = sum(1 for row in rows if row.get(column) is None) / row_count
        if column in SIGNATURE_COLUMNS and missingness > 0.0:
            # A signature is executable only when one compatible calibrated
            # artifact populated every training row.
            continue
        if column in OPTIONAL_COLUMNS and missingness > optional_missingness_threshold:
            continue
        selected.append(column)
    return selected
