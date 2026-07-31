"""Versioned compatibility-label contracts and deterministic projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from tm_ecg.clinical_validation.audit import sha256_file
from tm_ecg.clinical_validation.ontology import load_json_yaml


class LabelContractError(ValueError):
    """Raised when a target or prediction violates the compatibility contract."""


def parse_label_tokens(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [
            token.strip()
            for token in re.split(r"\s*(?:\||,|;)\s*", raw)
            if token.strip()
        ]
    if isinstance(raw, Iterable):
        return [str(token).strip() for token in raw if str(token).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


@dataclass(frozen=True, slots=True)
class CompatibilityLabelContractV4:
    version: int
    contract_id: str
    label_order: tuple[str, ...]
    specific_labels: tuple[str, ...]
    normal_label: str
    residual_label: str
    af_afl_mutually_exclusive: bool
    mixed_ectopy_allowed: bool
    pacing_conduction_cooccurrence: bool
    aliases: tuple[tuple[str, str], ...]
    source_path: str | None = None
    source_sha256: str | None = None

    @property
    def alias_map(self) -> dict[str, str]:
        return dict(self.aliases)

    def normalize(
        self,
        raw: object,
        *,
        empty_policy: str = "error",
    ) -> tuple[str, ...]:
        """Normalize aliases and enforce normal/residual exclusivity."""

        if empty_policy not in {"error", "residual"}:
            raise ValueError("empty_policy must be error or residual")
        aliases = self.alias_map
        labels: list[str] = []
        unknown: list[str] = []
        for token in parse_label_tokens(raw):
            label = aliases.get(token, token)
            if label not in self.label_order:
                unknown.append(token)
                continue
            if label not in labels:
                labels.append(label)
        if unknown:
            raise LabelContractError(f"Unknown compatibility labels: {sorted(unknown)}")
        specific = [label for label in self.specific_labels if label in labels]
        if specific:
            labels = specific
        elif self.residual_label in labels:
            labels = [self.residual_label]
        elif self.normal_label in labels:
            labels = [self.normal_label]
        if not labels:
            if empty_policy == "residual":
                labels = [self.residual_label]
            else:
                raise LabelContractError(
                    "Training target projection produced an empty compatibility set"
                )
        return tuple(label for label in self.label_order if label in labels)

    def is_valid(self, labels: object) -> bool:
        try:
            normalized = self.normalize(labels, empty_policy="error")
        except LabelContractError:
            return False
        tokens = set(parse_label_tokens(labels))
        canonical_tokens = {self.alias_map.get(token, token) for token in tokens}
        return canonical_tokens == set(normalized)

    def validate_prediction_matrix(self, predictions: object) -> None:
        import numpy as np  # type: ignore

        matrix = np.asarray(predictions)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.label_order):
            raise LabelContractError(
                "Prediction matrix must have one column per frozen compatibility label"
            )
        boolean = matrix.astype(bool)
        if (~boolean.any(axis=1)).any():
            raise LabelContractError("Compatibility predictions may not be empty")
        normal_index = self.label_order.index(self.normal_label)
        residual_index = self.label_order.index(self.residual_label)
        specific_indices = np.asarray(
            [self.label_order.index(label) for label in self.specific_labels]
        )
        if (
            boolean[:, normal_index]
            & boolean[:, np.arange(len(self.label_order)) != normal_index].any(axis=1)
        ).any():
            raise LabelContractError("Normal co-occurs with an abnormal prediction")
        if (
            boolean[:, residual_index] & boolean[:, specific_indices].any(axis=1)
        ).any():
            raise LabelContractError(
                "Other / unmapped co-occurs with a specific prediction"
            )


def load_compatibility_label_contract(
    path: str | Path,
) -> CompatibilityLabelContractV4:
    source = Path(path)
    payload = load_json_yaml(source)
    if payload.get("version") != 4:
        raise LabelContractError("Compatibility label contract must declare version 4")
    label_order_raw = payload.get("label_order")
    specific_raw = payload.get("specific_labels")
    constraints = payload.get("constraints")
    aliases_raw = payload.get("aliases", {})
    if not isinstance(label_order_raw, list) or not isinstance(specific_raw, list):
        raise LabelContractError("label_order and specific_labels must be lists")
    if not isinstance(constraints, dict) or not isinstance(aliases_raw, dict):
        raise LabelContractError("constraints and aliases must be objects")
    label_order = tuple(str(label) for label in label_order_raw)
    specific = tuple(str(label) for label in specific_raw)
    normal = str(payload.get("normal_label", ""))
    residual = str(payload.get("residual_label", ""))
    if len(set(label_order)) != len(label_order) or not label_order:
        raise LabelContractError("label_order must be non-empty and unique")
    if not set(specific) <= set(label_order):
        raise LabelContractError("specific_labels must be included in label_order")
    if normal not in label_order or residual not in label_order or normal == residual:
        raise LabelContractError("Normal and residual labels must be distinct registered labels")
    if set(specific) & {normal, residual}:
        raise LabelContractError("Normal and residual labels may not be specific labels")
    aliases = tuple(sorted((str(key), str(value)) for key, value in aliases_raw.items()))
    if not all(value in label_order for _, value in aliases):
        raise LabelContractError("Every compatibility alias must resolve to label_order")
    return CompatibilityLabelContractV4(
        version=4,
        contract_id=str(payload.get("contract_id", "compatibility_v4")),
        label_order=label_order,
        specific_labels=specific,
        normal_label=normal,
        residual_label=residual,
        af_afl_mutually_exclusive=bool(
            constraints.get("af_afl_mutually_exclusive", True)
        ),
        mixed_ectopy_allowed=bool(constraints.get("mixed_ectopy_allowed", True)),
        pacing_conduction_cooccurrence=bool(
            constraints.get("pacing_conduction_cooccurrence", True)
        ),
        aliases=aliases,
        source_path=str(source),
        source_sha256=sha256_file(source),
    )


DEFAULT_COMPATIBILITY_CONTRACT_V4 = CompatibilityLabelContractV4(
    version=4,
    contract_id="compatibility_v4",
    label_order=(
        "Normal",
        "PVC",
        "APB",
        "RBBB spectrum",
        "LBBB spectrum",
        "AF",
        "AFL",
        "Paced",
        "Other / unmapped",
    ),
    specific_labels=(
        "PVC",
        "APB",
        "RBBB spectrum",
        "LBBB spectrum",
        "AF",
        "AFL",
        "Paced",
    ),
    normal_label="Normal",
    residual_label="Other / unmapped",
    af_afl_mutually_exclusive=True,
    mixed_ectopy_allowed=True,
    pacing_conduction_cooccurrence=True,
    aliases=(),
)
