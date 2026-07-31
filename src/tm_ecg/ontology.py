"""Ontology helpers shared across datasets and reports."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping

from tm_ecg.constants import AXIS_LABELS
from tm_ecg.modeling.label_contract import DEFAULT_COMPATIBILITY_CONTRACT_V4
from tm_ecg.ptbxl_semantics import StatementMetadata, statement_state
from tm_ecg.types import MultiAxialTarget, OntologyMapping


PTBXL_AXIS_MAP: dict[str, tuple[str, str]] = {
    "SR": ("rhythm", "sinus"),
    "AFIB": ("rhythm", "af"),
    "AFLT": ("rhythm", "afl"),
    "STACH": ("rhythm", "sinus_tachycardia"),
    "SARRH": ("rhythm", "sinus"),
    "SBRAD": ("rhythm", "sinus_bradycardia"),
    "SVARR": ("rhythm", "other_arrhythmia"),
    "SVTAC": ("rhythm", "other_arrhythmia"),
    "PSVT": ("rhythm", "other_arrhythmia"),
    "PVC": ("ectopy", "pvc"),
    "VPB": ("ectopy", "pvc"),
    "PAC": ("ectopy", "apb"),
    "APB": ("ectopy", "apb"),
    "SPAC": ("ectopy", "apb"),
    "SVPB": ("ectopy", "apb"),
    "BIGU": ("ectopy", "unspecified_ectopy"),
    "TRIGU": ("ectopy", "unspecified_ectopy"),
    "PRC(S)": ("ectopy", "unspecified_ectopy"),
    "RBBB": ("conduction", "rbbb_spectrum"),
    "CRBBB": ("conduction", "rbbb_spectrum"),
    "IRBBB": ("conduction", "rbbb_spectrum"),
    "LBBB": ("conduction", "lbbb_spectrum"),
    "CLBBB": ("conduction", "lbbb_spectrum"),
    "IVCD": ("conduction", "other_conduction"),
    "LAFB": ("conduction", "other_conduction"),
    "LPFB": ("conduction", "other_conduction"),
    "WPW": ("conduction", "other_conduction"),
    "LPR": ("conduction", "other_conduction"),
    "STE_": ("repolarization", "st_elevation"),
    "STD_": ("repolarization", "st_depression"),
    "NDT": ("repolarization", "t_abnormality"),
    "NT_": ("repolarization", "t_abnormality"),
    "TAB_": ("repolarization", "t_abnormality"),
    "INVT": ("repolarization", "t_abnormality"),
    "LOWT": ("repolarization", "t_abnormality"),
    "NST_": ("repolarization", "other_st_t"),
    "DIG": ("repolarization", "other_st_t"),
    "LNGQT": ("repolarization", "qtc_abnormality"),
    "ISC_": ("repolarization", "other_st_t"),
    "ISCAL": ("repolarization", "other_st_t"),
    "ISCAN": ("repolarization", "other_st_t"),
    "ISCAS": ("repolarization", "other_st_t"),
    "ISCIL": ("repolarization", "other_st_t"),
    "ISCIN": ("repolarization", "other_st_t"),
    "ISCLA": ("repolarization", "other_st_t"),
}

def _parse_scp_mapping(raw_value: object) -> Mapping[object, object] | None:
    if isinstance(raw_value, Mapping):
        return raw_value
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _normalized_scalar_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip().lower()


def _ptbxl_pacemaker_metadata_is_present(row: Mapping[str, object]) -> bool:
    pacemaker = _normalized_scalar_text(row.get("pacemaker"))
    if not pacemaker:
        pacemaker = _normalized_scalar_text(row.get("pacemaker_present"))
    return (
        pacemaker in {"1", "true", "yes", "y", "ja"}
        or pacemaker.startswith("ja,")
        or "pacemaker" in pacemaker
        or "pace" in pacemaker
    )


def parse_ptbxl_scp_codes(raw_value: object) -> list[str]:
    parsed = _parse_scp_mapping(raw_value)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        return [str(key) for key in parsed.keys()]
    return [str(key) for key in parsed]


def parse_ptbxl_scp_likelihoods(raw_value: object) -> dict[str, float]:
    parsed = _parse_scp_mapping(raw_value)
    if parsed is None:
        return {}
    result: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            result[str(key).upper()] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def map_ptbxl_axes(
    row: Mapping[str, object],
    accepted_min: float = 80.0,
    *,
    uncertain_min: float = 50.0,
    statement_metadata: Mapping[str, StatementMetadata] | None = None,
) -> MultiAxialTarget:
    values: dict[str, set[str]] = {
        "rhythm": set(),
        "ectopy": set(),
        "conduction": set(),
        "repolarization": set(),
    }
    unsupported: list[str] = []
    pacing_present = _ptbxl_pacemaker_metadata_is_present(row)
    normality = "unknown"
    likelihoods = parse_ptbxl_scp_likelihoods(row.get("scp_codes"))
    for code, likelihood in likelihoods.items():
        metadata = statement_metadata.get(code) if statement_metadata is not None else None
        if (
            statement_state(
                code,
                likelihood,
                metadata,
                accepted_min=accepted_min,
                uncertain_min=uncertain_min,
            )
            != "present"
        ):
            continue
        if code == "NORM":
            normality = "normal"
            continue
        if code == "PACE":
            pacing_present = True
            continue
        mapping = PTBXL_AXIS_MAP.get(code)
        if mapping is None:
            unsupported.append(code)
            continue
        axis, value = mapping
        values[axis].add(value)
    return MultiAxialTarget(
        rhythm=tuple(sorted(values["rhythm"])),
        ectopy=tuple(sorted(values["ectopy"])),
        conduction=tuple(sorted(values["conduction"])),
        repolarization=tuple(sorted(values["repolarization"])),
        pacing="present" if pacing_present else "absent",
        normality=normality,
        unsupported_source_labels=tuple(sorted(unsupported)),
    )


def map_ludb_axes(text: str | None) -> MultiAxialTarget:
    raw_text = (text or "").strip()
    lowered = raw_text.lower()
    axes: dict[str, set[str]] = {
        "rhythm": set(),
        "ectopy": set(),
        "conduction": set(),
        "repolarization": set(),
    }
    phrases: tuple[tuple[str, str, str], ...] = (
        ("normal sinus rhythm", "rhythm", "sinus"),
        ("sinus rhythm", "rhythm", "sinus"),
        ("sinus bradycardia", "rhythm", "other_rhythm"),
        ("sinus tachycardia", "rhythm", "other_rhythm"),
        ("sinus arrhythmia", "rhythm", "other_rhythm"),
        ("irregular sinus rhythm", "rhythm", "other_rhythm"),
        ("wandering atrial pacemaker", "rhythm", "other_rhythm"),
        ("atrial fibrillation", "rhythm", "af"),
        ("atrial flutter", "rhythm", "afl"),
        ("ventricular extrasystole", "ectopy", "pvc"),
        ("pvc", "ectopy", "pvc"),
        ("atrial extrasystole", "ectopy", "apb"),
        ("supraventricular ectopy", "ectopy", "apb"),
        ("right bundle branch block", "conduction", "rbbb_spectrum"),
        ("left bundle branch block", "conduction", "lbbb_spectrum"),
        ("atrioventricular block", "conduction", "other_conduction"),
        ("av block", "conduction", "other_conduction"),
        ("av-block", "conduction", "other_conduction"),
        ("left anterior hemiblock", "conduction", "other_conduction"),
        ("intraventricular conduction delay", "conduction", "other_conduction"),
        ("intravintricular conduction delay", "conduction", "other_conduction"),
        ("aberrant conduction", "conduction", "other_conduction"),
        ("sinoatrial blockade", "conduction", "other_conduction"),
        ("st elevation", "repolarization", "st_elevation"),
        ("st depression", "repolarization", "st_depression"),
        ("t wave abnormal", "repolarization", "t_abnormality"),
        ("repolarization abnormal", "repolarization", "other_st_t"),
        ("ischemia", "repolarization", "other_st_t"),
        ("scar formation", "repolarization", "other_st_t"),
        ("early repolarization", "repolarization", "st_elevation"),
    )
    raw_clauses = [
        clause.strip(" .")
        for clause in re.split(r"\s*(?:\||;|\r?\n)\s*", raw_text)
        if clause.strip(" .")
    ]
    unmatched: list[str] = []
    pacing_terms = (
        "paced rhythm",
        "ventricular pacing",
        "atrial pacing",
        "biventricular pacing",
        "cardiac pacemaker",
    )
    pacing = "present" if any(term in lowered for term in pacing_terms) else "absent"
    for raw_clause in raw_clauses:
        clause = raw_clause.lower()
        matched = False
        for phrase, axis, value in phrases:
            if phrase in clause:
                axes[axis].add(value)
                matched = True
        if re.search(r"\bstemi\b", clause) and not re.search(r"\bnstemi\b", clause):
            axes["repolarization"].add("st_elevation")
            matched = True
        if any(term in clause for term in pacing_terms):
            matched = True
        if clause in {"normal ecg", "normal sinus rhythm"}:
            matched = True
        if not matched:
            unmatched.append(raw_clause)
    normality = (
        "normal"
        if any(term in lowered for term in ("normal sinus rhythm", "normal ecg"))
        else "unknown"
    )
    return MultiAxialTarget(
        rhythm=tuple(sorted(axes["rhythm"])),
        ectopy=tuple(sorted(axes["ectopy"])),
        conduction=tuple(sorted(axes["conduction"])),
        repolarization=tuple(sorted(axes["repolarization"])),
        pacing=pacing,
        normality=normality,
        unsupported_source_labels=tuple(sorted(set(unmatched))),
    )


def validate_multiaxial_target(target: MultiAxialTarget) -> None:
    for axis in ("rhythm", "ectopy", "conduction", "repolarization"):
        unknown = set(getattr(target, axis)) - set(AXIS_LABELS[axis])
        if unknown:
            raise ValueError(f"Unknown {axis} target values: {sorted(unknown)}")
    if (
        target.pacing not in AXIS_LABELS["pacing"]
        or target.normality not in AXIS_LABELS["normality"]
        or target.quality not in AXIS_LABELS["quality"]
    ):
        raise ValueError("Invalid pacing, normality, or quality target")


def compatibility_projection(target: MultiAxialTarget) -> list[str]:
    labels: list[str] = []
    mapping = {
        "af": "AF",
        "afl": "AFL",
        "apb": "APB",
        "pvc": "PVC",
        "rbbb_spectrum": "RBBB spectrum",
        "lbbb_spectrum": "LBBB spectrum",
    }
    for value in (*target.rhythm, *target.ectopy, *target.conduction):
        label = mapping.get(value)
        if label and label not in labels:
            labels.append(label)
    if target.pacing == "present":
        labels.append("Paced")
    has_unprojected_axis_value = any(
        value not in mapping and value != "sinus"
        for value in (*target.rhythm, *target.ectopy, *target.conduction)
    )
    has_other_abnormality = bool(
        target.repolarization
        or target.unsupported_source_labels
        or has_unprojected_axis_value
    )
    if has_other_abnormality:
        labels.append("Other / unmapped")
    if target.normality == "normal" and not labels:
        labels.append("Normal")
    return list(
        DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
            labels,
            empty_policy="residual",
        )
    )


def map_ptbxl_labels(
    row: Mapping[str, object],
    accepted_min: float = 80.0,
    *,
    uncertain_min: float = 50.0,
    statement_metadata: Mapping[str, StatementMetadata] | None = None,
) -> list[str]:
    return compatibility_projection(
        map_ptbxl_axes(
            row,
            accepted_min=accepted_min,
            uncertain_min=uncertain_min,
            statement_metadata=statement_metadata,
        )
    )


def map_ludb_text(text: str | None) -> list[str]:
    return compatibility_projection(map_ludb_axes(text))


def normalize_project_labels(labels: list[str]) -> list[str]:
    return list(
        DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
            labels,
            empty_policy="residual",
        )
    )


def appendix_d_mapping() -> list[OntologyMapping]:
    rows = [
        ("ptbxl", "PTB-XL NORM", "Normal", "Appendix D"),
        ("ludb", "normal sinus rhythm / no pathologic rhythm label", "Normal", "Appendix D"),
        ("ptbxl", "PTB-XL PVC / VPB / ventricular ectopy statements", "PVC", "Appendix D"),
        ("ludb", "ventricular extrasystole / PVC diagnosis", "PVC", "Appendix D"),
        ("ptbxl", "PTB-XL PAC/APB/SPAC/SVPB", "APB", "Appendix D"),
        ("ludb", "atrial extrasystole / supraventricular ectopy diagnosis", "APB", "Appendix D"),
        ("ptbxl", "PTB-XL RBBB / CRBBB / IRBBB", "RBBB spectrum", "Appendix D"),
        ("ludb", "right bundle branch block diagnosis", "RBBB spectrum", "Appendix D"),
        ("ptbxl", "PTB-XL LBBB / CLBBB", "LBBB spectrum", "Appendix D"),
        ("ludb", "left bundle branch block diagnosis", "LBBB spectrum", "Appendix D"),
        ("ptbxl", "PTB-XL AFIB", "AF", "Appendix D"),
        ("ludb", "atrial fibrillation rhythm field", "AF", "Appendix D"),
        ("ptbxl", "PTB-XL AFLT", "AFL", "Appendix D"),
        ("ludb", "atrial flutter rhythm field", "AFL", "Appendix D"),
        ("ptbxl", "PTB-XL paced / pacemaker metadata", "Paced", "Appendix D"),
        ("ludb", "pacemaker-present rhythm/metadata", "Paced", "Appendix D"),
        ("ptbxl", "PTB-XL other rhythm/form statements", "Other / unmapped", "Appendix D"),
        ("ludb", "unmatched diagnoses", "Other / unmapped", "Appendix D"),
    ]
    return [
        OntologyMapping(
            source_dataset=dataset,
            source_label=source_label,
            project_label=project_label,
            notes=notes,
        )
        for dataset, source_label, project_label, notes in rows
    ]
