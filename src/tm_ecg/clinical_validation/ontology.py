"""Versioned multiaxial ontology and deterministic scenario projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tm_ecg.clinical_validation.models import BenchmarkFindingSet, ClinicalFindingSet


FindingSet = ClinicalFindingSet | BenchmarkFindingSet

ABNORMAL_RHYTHMS = frozenset(
    {
        "af",
        "afl",
        "other_arrhythmia",
        "sinus_bradycardia",
        "sinus_tachycardia",
    }
)
ABNORMAL_ECTOPY = frozenset(
    {
        "atrial_premature",
        "ventricular_premature",
        "mixed_ectopy",
        "unspecified_ectopy",
    }
)
ABNORMAL_CONDUCTION = frozenset(
    {
        "rbbb_spectrum",
        "lbbb_spectrum",
        "other_conduction",
        "unspecified_conduction",
    }
)
ABNORMAL_REPOLARIZATION = frozenset(
    {
        "st_elevation",
        "st_depression",
        "t_wave_abnormality",
        "nonspecific_st_t",
        "qtc_abnormality",
        "unspecified_repolarization",
    }
)

# The dataset index and the clinical-validation package historically used two
# names for several identical states.  Normalize those names at the boundary
# rather than allowing duplicate aliases to become distinct kappa categories.
AXIS_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "ectopy": {
        "apb": "atrial_premature",
        "pvc": "ventricular_premature",
    },
    "repolarization": {
        "other_st_t": "nonspecific_st_t",
        "t_abnormality": "t_wave_abnormality",
    },
}


def load_json_yaml(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file without adding a parser dependency.

    All versioned validation configuration files intentionally use the JSON
    subset of YAML.  JSON is valid YAML and remains straightforward to hash and
    parse in restricted environments.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source} must use the dependency-free JSON subset of YAML: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object at the root of {source}")
    return payload


def has_supported_abnormality(findings: FindingSet, include_uncertain: bool = False) -> bool:
    abnormal = bool(
        set(findings.rhythm) & ABNORMAL_RHYTHMS
        or set(findings.ectopy) & ABNORMAL_ECTOPY
        or set(findings.conduction) & ABNORMAL_CONDUCTION
        or set(findings.repolarization) & ABNORMAL_REPOLARIZATION
        or findings.pacing == "present"
        or findings.residual_abnormal
    )
    return abnormal or (include_uncertain and bool(findings.uncertain_findings))


def exact_labels(findings: FindingSet, include_uncertain: bool = False) -> tuple[str, ...]:
    labels: set[str] = set()
    rhythm_map = {
        "af": "AF",
        "afl": "AFL",
        "other_arrhythmia": "Other abnormal",
        "sinus_bradycardia": "Sinus bradycardia",
        "sinus_tachycardia": "Sinus tachycardia",
    }
    ectopy_map = {
        "atrial_premature": "APB",
        "ventricular_premature": "PVC",
        "mixed_ectopy": "Mixed ectopy",
        "unspecified_ectopy": "Unspecified ectopy",
    }
    conduction_map = {
        "rbbb_spectrum": "RBBB spectrum",
        "lbbb_spectrum": "LBBB spectrum",
        "other_conduction": "Other conduction",
        "unspecified_conduction": "Unspecified conduction",
    }
    repolarization_map = {
        "st_elevation": "ST elevation",
        "st_depression": "ST depression",
        "t_wave_abnormality": "T-wave abnormality",
        "nonspecific_st_t": "Nonspecific ST-T",
        "qtc_abnormality": "QTc abnormality",
        "unspecified_repolarization": "Unspecified repolarization",
    }
    for value in findings.rhythm:
        if value in rhythm_map:
            labels.add(rhythm_map[value])
    for value in findings.ectopy:
        if value in ectopy_map:
            labels.add(ectopy_map[value])
    for value in findings.conduction:
        if value in conduction_map:
            labels.add(conduction_map[value])
    for value in findings.repolarization:
        if value in repolarization_map:
            labels.add(repolarization_map[value])
    if findings.pacing == "present":
        labels.add("Paced")
    if findings.residual_abnormal:
        labels.add("Other abnormal")
    # Sinus rhythm is a rhythm state, not proof that the whole ECG is normal.
    # Emit the mutually exclusive global Normal label only when no definite
    # abnormal finding was reconstructed on any clinical axis.
    if not labels and findings.normality == "normal":
        labels.add("Normal")
    if include_uncertain:
        labels.update(f"Possible {item}" for item in findings.uncertain_findings)
    if not labels:
        if findings.interpretability == "uninterpretable":
            labels.add("Uninterpretable")
        elif findings.normality == "normal":
            labels.add("Normal")
        else:
            labels.add("Indeterminate")
    return tuple(sorted(labels))


def canonical_label_set_token(labels: tuple[str, ...]) -> str:
    return " | ".join(sorted(set(labels)))


def project_exact(findings: FindingSet, include_uncertain: bool = False) -> str:
    return canonical_label_set_token(exact_labels(findings, include_uncertain))


def project_five_family(findings: FindingSet, include_uncertain: bool = False) -> str:
    families: set[str] = set()
    if set(findings.rhythm) & ABNORMAL_RHYTHMS:
        families.add("Rhythm")
    if set(findings.ectopy) & ABNORMAL_ECTOPY:
        families.add("Ectopy")
    if set(findings.conduction) & ABNORMAL_CONDUCTION:
        families.add("Conduction")
    if (
        set(findings.repolarization) & ABNORMAL_REPOLARIZATION
        or findings.pacing == "present"
        or findings.residual_abnormal
    ):
        families.add("Other abnormal")
    if include_uncertain and findings.uncertain_findings:
        families.add("Other abnormal")
    if not families:
        if findings.interpretability == "uninterpretable":
            families.add("Uninterpretable")
        elif findings.normality == "normal":
            families.add("Normal")
        else:
            families.add("Indeterminate")
    return canonical_label_set_token(tuple(families))


def project_binary(findings: FindingSet, include_uncertain: bool = False) -> str:
    if findings.interpretability == "uninterpretable":
        return "Uninterpretable"
    if has_supported_abnormality(findings, include_uncertain):
        return "Abnormal"
    if findings.normality == "normal":
        return "Normal"
    return "Indeterminate"


def project_dominant(
    findings: FindingSet,
    priority: tuple[str, ...],
    include_uncertain: bool = False,
) -> str:
    available = set(exact_labels(findings, include_uncertain))
    for label in priority:
        if label in available:
            return label
    return sorted(available)[0]


def project_axis(findings: FindingSet, axis: str, include_uncertain: bool = False) -> str:
    if axis == "rhythm":
        values = tuple(value for value in findings.rhythm if value != "indeterminate")
        return canonical_label_set_token(values or ("absent",))
    if axis == "ectopy":
        values = tuple(value for value in findings.ectopy if value != "indeterminate")
        return canonical_label_set_token(values or ("absent",))
    if axis == "conduction":
        values = tuple(value for value in findings.conduction if value != "indeterminate")
        return canonical_label_set_token(values or ("absent",))
    if axis == "repolarization":
        values = tuple(value for value in findings.repolarization if value != "indeterminate")
        return canonical_label_set_token(values or ("absent",))
    if axis == "pacing":
        return findings.pacing
    if axis == "quality":
        return findings.quality
    if axis == "interpretability":
        return findings.interpretability
    raise ValueError(f"Unknown validation axis: {axis}")


def project_axis_binary(findings: FindingSet, axis: str) -> str:
    if axis == "rhythm":
        return "present" if set(findings.rhythm) & ABNORMAL_RHYTHMS else "absent"
    if axis == "ectopy":
        return "present" if set(findings.ectopy) & ABNORMAL_ECTOPY else "absent"
    if axis == "conduction":
        return "present" if set(findings.conduction) & ABNORMAL_CONDUCTION else "absent"
    if axis == "repolarization":
        return "present" if set(findings.repolarization) & ABNORMAL_REPOLARIZATION else "absent"
    if axis == "pacing":
        return "present" if findings.pacing == "present" else "absent"
    raise ValueError(f"Unknown binary validation axis: {axis}")


def projection_value(
    findings: FindingSet,
    projection: str,
    *,
    include_uncertain: bool,
    dominant_priority: tuple[str, ...],
) -> str:
    if projection == "exact":
        return project_exact(findings, include_uncertain)
    if projection == "five_family":
        return project_five_family(findings, include_uncertain)
    if projection == "binary":
        return project_binary(findings, include_uncertain)
    if projection == "dominant":
        return project_dominant(findings, dominant_priority, include_uncertain)
    if projection.startswith("axis:"):
        return project_axis(findings, projection.split(":", 1)[1], include_uncertain)
    if projection.startswith("axis_binary:"):
        return project_axis_binary(findings, projection.split(":", 1)[1])
    raise ValueError(f"Unknown projection: {projection}")


def normalize_axis_values(values: Mapping[str, object]) -> dict[str, tuple[str, ...] | str | bool]:
    def items(key: str) -> tuple[str, ...]:
        value = values.get(key, [])
        if isinstance(value, (list, tuple, set, frozenset)):
            raw_items = tuple(str(item) for item in value)
        elif value not in {None, ""}:
            raw_items = (str(value),)
        else:
            raw_items = ()
        aliases = AXIS_VALUE_ALIASES.get(key, {})
        return tuple(sorted({aliases.get(item, item) for item in raw_items}))

    return {
        "rhythm": items("rhythm"),
        "ectopy": items("ectopy"),
        "conduction": items("conduction"),
        "repolarization": items("repolarization"),
        "pacing": str(values.get("pacing", "indeterminate")),
        "normality": str(values.get("normality", "indeterminate")),
        "quality": str(values.get("quality", "indeterminate")),
        "interpretability": str(values.get("interpretability", "indeterminate")),
        "residual_abnormal": bool(values.get("residual_abnormal", False)),
    }
