"""Medical predicate library for project ECG arrhythmia labels.

The predicates are conservative screening predicates for a research DSS. They use
clinician-readable matrix-B states and should be reviewed by cardiology experts before
clinical deployment.
"""

from __future__ import annotations

from typing import Mapping

from tm_ecg.constants import LUDB_TO_PROJECT, PROJECT_LABELS, PTBXL_TO_PROJECT
from tm_ecg.dss.models import MedicalPredicate, RuleCondition


STANDARD_REFERENCES = [
    "AHA/ACCF/HRS recommendations for standardization and interpretation of the ECG, "
    "Part III: intraventricular conduction disturbances; Heart Rhythm Society resource page "
    "https://www.hrsonline.org/resource/ahaaccfhrs-recommendations-standardization-and-interpretation-electrocardiogram-part-iii/",
    "AHA/ACCF/HRS recommendations for standardization and interpretation of the ECG, "
    "Part II: diagnostic statement list, and Part IV: ST segment, T and U waves, and QT interval.",
    "2023 ACC/AHA/ACCP/HRS guideline for diagnosis and management of atrial fibrillation; "
    "ECG documentation is required for AF diagnosis.",
    "ANSI/AAMI EC57:2012(R2020): testing and reporting performance of cardiac rhythm "
    "and ST-segment measurement algorithms; FDA recognition number 3-118.",
    "PhysioNet PTB-XL and LUDB dataset documentation; PTB-XL uses cardiologist-reviewed "
    "SCP-ECG diagnostic, form, and rhythm statements.",
]


def c(feature: str, state: str, criticality: float = 1.0, concept: str | None = None) -> RuleCondition:
    return RuleCondition(
        feature=feature,
        state=state,
        family="clinical_ecg",
        clinical_concept=concept or feature,
        source="medical_predicate_library",
        criticality=criticality,
    )


def not_c(feature: str, state: str, criticality: float = 1.0, concept: str | None = None) -> RuleCondition:
    base = c(feature, state, criticality, concept)
    return RuleCondition(
        feature=base.feature,
        state=base.state,
        family=base.family,
        clinical_concept=base.clinical_concept,
        source=base.source,
        criticality=base.criticality,
        negated=True,
    )


def quality_guard() -> list[RuleCondition]:
    """Return the mandatory observability conditions for an executable predicate.

    A morphology or rhythm predicate is not clinically executable when the signal,
    delineation, or analyzable duration is inadequate.  Keeping these conditions in
    every non-fallback predicate makes quality failure an explicit non-match rather
    than allowing a confident rule to operate on unobservable physiology.
    """

    return [
        c("lead_quality_min_db", "good_quality", 1.8, "signal_quality"),
        c("delineation_confidence", "high_confidence", 1.7, "delineation_reliability"),
        c("analyzable_duration_s", "adequate", 1.6, "minimum_observation_duration"),
    ]


def _source_labels_for(project_label: str) -> dict[str, list[str]]:
    ptbxl = sorted(code for code, mapped in PTBXL_TO_PROJECT.items() if mapped == project_label)
    ludb = sorted(text for text, mapped in LUDB_TO_PROJECT.items() if mapped == project_label)
    return {"PTB-XL": ptbxl, "LUDB": ludb, "MIT-BIH/AAMI-style": [project_label]}


def default_medical_predicates() -> dict[str, MedicalPredicate]:
    """Return one executable medical predicate per locked project label."""

    predicates = {
        "Normal": MedicalPredicate(
            label="Normal",
            source_labels=_source_labels_for("Normal"),
            required=quality_guard(),
            supportive=[
                c("af_irregularity_cv", "regular", 1.4, "regular_rr"),
                c("p_present_ratio", "present", 1.4, "sinus_p_waves"),
                c("pr_med_ms", "normal_pr", 1.0),
                c("qrs_dur_med_ms", "narrow_qrs", 1.5),
                c("pvc_like_beat_count", "zero", 1.1),
                c("apb_like_beat_count", "zero", 1.0),
                c("paced_like_beat_count", "zero", 1.2),
            ],
            contraindications=[
                c("af_irregularity_cv", "irregular", 1.8),
                c("qrs_dur_med_ms", "wide_qrs", 1.6),
                c("pvc_like_beat_count", "frequent", 1.4),
                c("paced_like_beat_count", "present", 1.6),
            ],
            references=STANDARD_REFERENCES,
            explanation="No strong ectopic, pacing, AF/flutter, or bundle-branch pattern is present.",
        ),
        "PVC": MedicalPredicate(
            label="PVC",
            source_labels=_source_labels_for("PVC"),
            required=quality_guard()
            + [c("pvc_like_beat_count", "present", 1.8, "ventricular_ectopy")],
            supportive=[
                c("pvc_like_beat_count", "frequent", 1.9, "ventricular_ectopy"),
                c("prematurity_index_min", "very_premature", 1.6),
                c("prematurity_index_min", "premature", 1.4),
                c("comp_pause_ratio_max", "compensatory_pause", 1.6),
                c("qrs_dur_med_ms", "wide_qrs", 1.2),
                c("qrs_deformed_prob", "deformed", 1.2),
            ],
            contraindications=[c("paced_like_beat_count", "present", 1.5)],
            references=STANDARD_REFERENCES,
            explanation="Premature ventricular-complex pattern with supportive pause or wide/deformed QRS evidence.",
        ),
        "APB": MedicalPredicate(
            label="APB",
            source_labels=_source_labels_for("APB"),
            required=quality_guard()
            + [c("apb_like_beat_count", "present", 1.6, "atrial_ectopy")],
            supportive=[
                c("apb_like_beat_count", "frequent", 1.7, "atrial_ectopy"),
                c("prematurity_index_min", "premature", 1.3),
                c("qrs_dur_med_ms", "narrow_qrs", 1.1),
                c("p_present_ratio", "present", 0.9),
                c("p_present_ratio", "intermittent", 0.8),
            ],
            contraindications=[c("qrs_dur_med_ms", "wide_qrs", 1.0), c("paced_like_beat_count", "present", 1.2)],
            references=STANDARD_REFERENCES,
            explanation="Premature atrial/supraventricular ectopy pattern, preferably narrow-QRS and non-paced.",
        ),
        "RBBB spectrum": MedicalPredicate(
            label="RBBB spectrum",
            source_labels=_source_labels_for("RBBB spectrum"),
            required=quality_guard()
            + [c("qrs_dur_med_ms", "wide_qrs", 1.7), c("r_prime_v1_any", "present", 1.8)],
            supportive=[
                c("rbbb_signature_score", "positive", 1.5),
                c("qrs_deformed_prob", "deformed", 1.0),
                c("qrs_axis_deg", "normal_axis", 0.5),
                c("qrs_lead_coverage", "sufficient", 1.4, "required_bundle_branch_leads"),
            ],
            contraindications=[c("broad_r_v6_any", "present", 1.4, "lbbb_like_broad_r_v6"), c("qrs_lead_coverage", "insufficient", 1.7)],
            references=STANDARD_REFERENCES,
            explanation="Wide-QRS right bundle-branch morphology with right-precordial r-prime evidence.",
        ),
        "LBBB spectrum": MedicalPredicate(
            label="LBBB spectrum",
            source_labels=_source_labels_for("LBBB spectrum"),
            required=quality_guard()
            + [c("qrs_dur_med_ms", "wide_qrs", 1.7), c("broad_r_v6_any", "present", 1.8)],
            supportive=[
                c("lbbb_signature_score", "positive", 1.5),
                c("qrs_deformed_prob", "deformed", 1.0),
                c("qrs_axis_deg", "left_axis", 0.5),
                c("qrs_lead_coverage", "sufficient", 1.4, "required_bundle_branch_leads"),
            ],
            contraindications=[c("r_prime_v1_any", "present", 1.4, "rbbb_like_r_prime"), c("qrs_lead_coverage", "insufficient", 1.7)],
            references=STANDARD_REFERENCES,
            explanation="Wide-QRS left bundle-branch morphology with lateral broad-R evidence.",
        ),
        "AF": MedicalPredicate(
            label="AF",
            source_labels=_source_labels_for("AF"),
            required=quality_guard()
            + [c("af_irregularity_cv", "irregular", 1.9, "irregularly_irregular_rr")],
            supportive=[
                c("p_present_ratio", "absent_or_low", 1.5, "absent_p_waves"),
                c("p_present_ratio", "intermittent", 1.0),
                c("rr_iqr_ms", "highly_variable", 1.3),
                c("f_wave_power_ratio", "possible_f_wave", 0.9),
                c("af_signature_score", "positive", 1.3),
                c("atrial_lead_coverage", "sufficient", 1.5, "atrial_observability"),
                c("atrial_valid_beat_fraction", "sufficient", 1.3),
                c("analyzable_duration_s", "adequate", 1.3),
            ],
            contraindications=[c("paced_like_beat_count", "frequent", 1.4), c("pvc_like_beat_count", "frequent", 1.7), c("apb_like_beat_count", "frequent", 1.5), c("atrial_lead_coverage", "insufficient", 1.8), c("f_wave_power_ratio", "f_wave_present", 0.5)],
            references=STANDARD_REFERENCES,
            explanation="Irregularly irregular RR pattern with absent/intermittent organized P waves.",
        ),
        "AFL": MedicalPredicate(
            label="AFL",
            source_labels=_source_labels_for("AFL"),
            required=quality_guard()
            + [c("f_wave_power_ratio", "f_wave_present", 1.8, "flutter_waves")],
            supportive=[
                c("hr_med_bpm", "tachycardic", 1.0),
                c("rr_med_ms", "tachycardic_rr", 1.0),
                c("af_irregularity_cv", "regular", 0.7),
                c("af_irregularity_cv", "mildly_irregular", 0.6),
                c("p_present_ratio", "absent_or_low", 1.0),
                c("atrial_lead_coverage", "sufficient", 1.4),
                c("analyzable_duration_s", "adequate", 1.2),
            ],
            contraindications=[c("af_irregularity_cv", "irregular", 0.8), c("atrial_lead_coverage", "insufficient", 1.8)],
            references=STANDARD_REFERENCES,
            explanation="Flutter-wave spectral/morphology evidence, commonly with regular or patterned atrial tachyarrhythmia.",
        ),
        "Paced": MedicalPredicate(
            label="Paced",
            source_labels=_source_labels_for("Paced"),
            required=quality_guard()
            + [c("paced_like_beat_count", "present", 1.9, "paced_complex")],
            supportive=[
                c("paced_like_beat_count", "frequent", 2.0, "paced_complex"),
                c("paced_signature_score", "positive", 1.6),
                c("qrs_dur_med_ms", "wide_qrs", 1.2),
                c("qrs_deformed_prob", "deformed", 1.0),
            ],
            contraindications=[c("pvc_like_beat_count", "frequent", 0.8)],
            references=STANDARD_REFERENCES,
            explanation="Pacing-spike or paced-complex signature with supportive wide/deformed QRS morphology.",
        ),
        "Other / unmapped": MedicalPredicate(
            label="Other / unmapped",
            source_labels=_source_labels_for("Other / unmapped"),
            required=[],
            supportive=[
                c("delineation_confidence", "low_confidence", 1.0),
                c("lead_quality_min_db", "low_quality", 1.0),
                c("qtc_med_ms", "very_prolonged_qtc", 1.0),
                c("st_level_v5_mV", "elevated", 1.0),
                c("st_level_v5_mV", "depressed", 1.0),
            ],
            contraindications=[],
            references=STANDARD_REFERENCES,
            explanation="Fallback class for weak, overlapping, unsupported, or non-mapped diagnostic signatures.",
            weak_signature=True,
        ),
    }
    return {label: predicates[label] for label in PROJECT_LABELS}


def predicate_feature_map(
    predicates: Mapping[str, MedicalPredicate] | None = None,
) -> dict[str, set[str]]:
    library = predicates or default_medical_predicates()
    return {label: predicate.features() for label, predicate in library.items()}


def predicates_to_dict(predicates: Mapping[str, MedicalPredicate]) -> dict[str, object]:
    return {label: predicate.to_dict() for label, predicate in predicates.items()}
