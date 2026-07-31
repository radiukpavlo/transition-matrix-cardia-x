"""Enforce the information boundary for unassisted physician coding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tm_ecg.clinical_validation.models import RawPhysicianResponse


ALLOWED_UNASSISTED_FIELDS = frozenset(
    {
        "case_id",
        "primary_diagnosis",
        "rationale",
        "diagnostic_confidence",
        "ecg_quality",
        "ambiguous",
        "requires_additional_information",
        "evidence_rr_irregularity",
        "evidence_absent_p_waves",
        "evidence_wide_qrs",
        "evidence_bundle_branch_pattern",
        "evidence_premature_beat",
        "evidence_paced_morphology",
        "evidence_st_deviation",
        "evidence_qtc_abnormality",
        "evidence_lead_noise",
        "evidence_other",
    }
)

PROHIBITED_FIELDS = frozenset(
    {
        "hidden_benchmark_label",
        "benchmark_labels",
        "source_labels",
        "cardia_x_route",
        "route_from_provenance",
        "primary_label_or_abstention",
        "activated_rule_ids",
        "uncertainty_flags",
        "safety_note",
        "diagnosis_changed",
        "assisted_confidence",
        "utility",
        "soundness",
        "safety",
        "clarity",
        "abstention_appropriate",
        "conflict_plausible",
        "missed_feature",
        "misleading_or_unsafe_feature",
        "final_comment",
        "previous_kappa",
        "scenario_result",
    }
)

EVIDENCE_FIELDS = tuple(
    sorted(field for field in ALLOWED_UNASSISTED_FIELDS if field.startswith("evidence_"))
)


class FieldPolicyViolation(ValueError):
    """Raised when a physician-coding input crosses the declared boundary."""


@dataclass(frozen=True, slots=True)
class PhysicianView:
    """A wrapper whose only payload is the allowed immutable response model."""

    response: RawPhysicianResponse


def enforce_raw_field_policy(payload: Mapping[str, object]) -> None:
    prohibited = sorted(set(payload) & PROHIBITED_FIELDS)
    if prohibited:
        raise FieldPolicyViolation(f"Prohibited physician-coder fields: {prohibited}")
    unknown = sorted(set(payload) - ALLOWED_UNASSISTED_FIELDS - {"source_cells"})
    if unknown:
        raise FieldPolicyViolation(f"Unregistered physician-coder fields: {unknown}")


def build_physician_view(payload: Mapping[str, object]) -> PhysicianView:
    enforce_raw_field_policy(payload)
    evidence = {field: str(payload.get(field, "")) for field in EVIDENCE_FIELDS}
    normalized = {
        "case_id": payload.get("case_id", ""),
        "primary_diagnosis": payload.get("primary_diagnosis", ""),
        "rationale": payload.get("rationale", ""),
        "diagnostic_confidence": payload.get("diagnostic_confidence"),
        "ecg_quality": payload.get("ecg_quality"),
        "ambiguous": payload.get("ambiguous", ""),
        "requires_additional_information": payload.get(
            "requires_additional_information", ""
        ),
        "evidence": evidence,
        "source_cells": payload.get("source_cells", {}),
    }
    return PhysicianView(RawPhysicianResponse.from_dict(normalized))


def assert_sentinel_absent(value: object, sentinel: str) -> None:
    """Recursively prove that planted prohibited content did not propagate."""

    if sentinel in repr(value):
        raise FieldPolicyViolation("A prohibited-field sentinel reached physician-coder output")

