"""Immutable models for the audited clinical-validation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Iterable
from typing import Any, Mapping


def _tuple_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _mapping_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


@dataclass(frozen=True, slots=True)
class CaseIdentity:
    workbook_case_id: str
    original_case_id: str
    dataset: str
    source_record_id: str
    route: str
    repeat_group_id: str | None
    row_ordinal: int

    @property
    def is_repeat(self) -> bool:
        return self.workbook_case_id != self.original_case_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CaseIdentity":
        return cls(
            workbook_case_id=str(payload["workbook_case_id"]),
            original_case_id=str(payload["original_case_id"]),
            dataset=str(payload["dataset"]),
            source_record_id=str(payload["source_record_id"]),
            route=str(payload.get("route", "")),
            repeat_group_id=(
                str(payload["repeat_group_id"])
                if payload.get("repeat_group_id") not in {None, ""}
                else None
            ),
            row_ordinal=int(str(payload["row_ordinal"])),
        )


@dataclass(frozen=True, slots=True)
class RawPhysicianResponse:
    """Only fields visible before CARDIA-X assistance."""

    case_id: str
    primary_diagnosis: str
    rationale: str
    diagnostic_confidence: float | None
    ecg_quality: float | None
    ambiguous: str
    requires_additional_information: str
    evidence: tuple[tuple[str, str], ...] = ()
    source_cells: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = dict(self.evidence)
        payload["source_cells"] = dict(self.source_cells)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RawPhysicianResponse":
        evidence = payload.get("evidence", {})
        cells = payload.get("source_cells", {})
        evidence_map = evidence if isinstance(evidence, Mapping) else {}
        cells_map = cells if isinstance(cells, Mapping) else {}
        return cls(
            case_id=str(payload["case_id"]),
            primary_diagnosis=str(payload.get("primary_diagnosis", "")),
            rationale=str(payload.get("rationale", "")),
            diagnostic_confidence=(
                float(str(payload["diagnostic_confidence"]))
                if payload.get("diagnostic_confidence") not in {None, ""}
                else None
            ),
            ecg_quality=(
                float(str(payload["ecg_quality"]))
                if payload.get("ecg_quality") not in {None, ""}
                else None
            ),
            ambiguous=str(payload.get("ambiguous", "")),
            requires_additional_information=str(
                payload.get("requires_additional_information", "")
            ),
            evidence=tuple(sorted((str(k), str(v)) for k, v in evidence_map.items())),
            source_cells=tuple(sorted((str(k), str(v)) for k, v in cells_map.items())),
        )


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    source_field: str
    source_cell: str
    normalized_text_span: str
    matched_rule_id: str
    polarity: str
    temporality: str
    certainty: str
    axis: str
    finding: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvidenceSpan":
        return cls(**{name: str(payload.get(name, "")) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class SourceStatementTrace:
    code: str
    likelihood: float
    category: str
    presence_coded: bool
    state: str
    metadata_source_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourceStatementTrace":
        return cls(
            code=str(payload.get("code", "")),
            likelihood=float(str(payload.get("likelihood", 0.0))),
            category=str(payload.get("category", "unknown")),
            presence_coded=bool(payload.get("presence_coded", False)),
            state=str(payload.get("state", "ignored")),
            metadata_source_sha256=(
                str(payload["metadata_source_sha256"])
                if payload.get("metadata_source_sha256") not in {None, ""}
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ClinicalAssertion:
    source_field: str
    source_cell: str
    raw_text_or_value: str
    normalized_span: str
    rule_id: str
    rule_version: str
    assertion_status: str
    derivation_status: str
    axis: str
    finding: str

    def __post_init__(self) -> None:
        if self.assertion_status not in {
            "definite",
            "uncertain",
            "negated",
            "absent",
            "unknown",
        }:
            raise ValueError(f"Unknown assertion status: {self.assertion_status}")
        if self.derivation_status not in {
            "explicit",
            "evidence-supported",
            "inferred",
            "observation",
        }:
            raise ValueError(f"Unknown derivation status: {self.derivation_status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ClinicalAssertion":
        return cls(
            **{
                name: str(payload.get(name, ""))
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class ClinicalFindingSet:
    case_id: str
    rhythm: tuple[str, ...] = ()
    ectopy: tuple[str, ...] = ()
    conduction: tuple[str, ...] = ()
    repolarization: tuple[str, ...] = ()
    pacing: str = "indeterminate"
    normality: str = "indeterminate"
    quality: str = "indeterminate"
    interpretability: str = "indeterminate"
    residual_abnormal: bool = False
    uncertain_findings: tuple[str, ...] = ()
    evidence: tuple[EvidenceSpan, ...] = ()
    explicit_diagnoses: tuple[ClinicalAssertion, ...] = ()
    observations: tuple[ClinicalAssertion, ...] = ()
    axis_derivations: tuple[ClinicalAssertion, ...] = ()
    projection_labels: tuple[str, ...] = ()
    conflict_resolution_trace: tuple[str, ...] = ()
    coding_rule_version: str = "physician_coding_v2"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        payload["explicit_diagnoses"] = [
            item.to_dict() for item in self.explicit_diagnoses
        ]
        payload["observations"] = [item.to_dict() for item in self.observations]
        payload["axis_derivations"] = [
            item.to_dict() for item in self.axis_derivations
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ClinicalFindingSet":
        raw_evidence = payload.get("evidence", [])
        evidence_rows = raw_evidence if isinstance(raw_evidence, list) else []
        return cls(
            case_id=str(payload["case_id"]),
            rhythm=_tuple_strings(payload.get("rhythm")),
            ectopy=_tuple_strings(payload.get("ectopy")),
            conduction=_tuple_strings(payload.get("conduction")),
            repolarization=_tuple_strings(payload.get("repolarization")),
            pacing=str(payload.get("pacing", "indeterminate")),
            normality=str(payload.get("normality", "indeterminate")),
            quality=str(payload.get("quality", "indeterminate")),
            interpretability=str(payload.get("interpretability", "indeterminate")),
            residual_abnormal=bool(payload.get("residual_abnormal", False)),
            uncertain_findings=_tuple_strings(payload.get("uncertain_findings")),
            evidence=tuple(
                EvidenceSpan.from_dict(item)
                for item in evidence_rows
                if isinstance(item, Mapping)
            ),
            explicit_diagnoses=tuple(
                ClinicalAssertion.from_dict(item)
                for item in _mapping_rows(payload.get("explicit_diagnoses"))
            ),
            observations=tuple(
                ClinicalAssertion.from_dict(item)
                for item in _mapping_rows(payload.get("observations"))
            ),
            axis_derivations=tuple(
                ClinicalAssertion.from_dict(item)
                for item in _mapping_rows(payload.get("axis_derivations"))
            ),
            projection_labels=_tuple_strings(payload.get("projection_labels")),
            conflict_resolution_trace=_tuple_strings(
                payload.get("conflict_resolution_trace")
            ),
            coding_rule_version=str(
                payload.get("coding_rule_version", "physician_coding_v2")
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkFindingSet:
    case_id: str
    rhythm: tuple[str, ...] = ()
    ectopy: tuple[str, ...] = ()
    conduction: tuple[str, ...] = ()
    repolarization: tuple[str, ...] = ()
    pacing: str = "indeterminate"
    normality: str = "indeterminate"
    quality: str = "indeterminate"
    interpretability: str = "indeterminate"
    residual_abnormal: bool = False
    uncertain_findings: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()
    accepted_source_labels: tuple[str, ...] = ()
    uncertain_source_labels: tuple[str, ...] = ()
    ignored_source_labels: tuple[str, ...] = ()
    mapping_rule_ids: tuple[str, ...] = ()
    source_statement_trace: tuple[SourceStatementTrace, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_statement_trace"] = [
            item.to_dict() for item in self.source_statement_trace
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BenchmarkFindingSet":
        return cls(
            case_id=str(payload["case_id"]),
            rhythm=_tuple_strings(payload.get("rhythm")),
            ectopy=_tuple_strings(payload.get("ectopy")),
            conduction=_tuple_strings(payload.get("conduction")),
            repolarization=_tuple_strings(payload.get("repolarization")),
            pacing=str(payload.get("pacing", "indeterminate")),
            normality=str(payload.get("normality", "indeterminate")),
            quality=str(payload.get("quality", "indeterminate")),
            interpretability=str(payload.get("interpretability", "indeterminate")),
            residual_abnormal=bool(payload.get("residual_abnormal", False)),
            uncertain_findings=_tuple_strings(payload.get("uncertain_findings")),
            source_labels=_tuple_strings(payload.get("source_labels")),
            accepted_source_labels=_tuple_strings(payload.get("accepted_source_labels")),
            uncertain_source_labels=_tuple_strings(payload.get("uncertain_source_labels")),
            ignored_source_labels=_tuple_strings(payload.get("ignored_source_labels")),
            mapping_rule_ids=_tuple_strings(payload.get("mapping_rule_ids")),
            source_statement_trace=tuple(
                SourceStatementTrace.from_dict(item)
                for item in _mapping_rows(payload.get("source_statement_trace"))
            ),
        )


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    scenario_id: str
    description: str
    population: str
    required_case_count: int | None
    deduplication: str
    physician_projection: str
    benchmark_projection: str
    uncertain_policy: str
    multilabel_policy: str
    requirement: str
    estimability_policy: str
    minimum_kappa: float
    bootstrap_mode: str
    bootstrap_seed: int
    route: str | None = None
    dataset: str | None = None
    endpoint_version: str = "clinical_validation_v2"
    scenario_group: str = "legacy"
    population_contract: str = ""
    required_case_ids_hash: str | None = None
    minimum_sample_size: int | None = None
    projection_contract_hash: str = ""
    source_contract_hash: str = ""
    physician_coder_hash: str = ""
    benchmark_mapping_hash: str = ""
    cluster_key: str = "original_case_id"
    bootstrap_replicates: int | None = None
    expected_baseline_kappa: float | None = None
    expected_baseline_observed_agreement: float | None = None
    expected_baseline_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ScenarioDefinition":
        return cls(
            scenario_id=str(payload["scenario_id"]),
            description=str(payload.get("description", "")),
            population=str(payload["population"]),
            required_case_count=(
                int(str(payload["required_case_count"]))
                if payload.get("required_case_count") is not None
                else None
            ),
            deduplication=str(payload.get("deduplication", "original_case_id")),
            physician_projection=str(payload["physician_projection"]),
            benchmark_projection=str(payload["benchmark_projection"]),
            uncertain_policy=str(payload.get("uncertain_policy", "exclude_uncertain_findings")),
            multilabel_policy=str(payload.get("multilabel_policy", "canonical_token")),
            requirement=str(payload.get("requirement", "diagnostic")),
            estimability_policy=str(payload.get("estimability_policy", "report")),
            minimum_kappa=float(str(payload.get("minimum_kappa", 0.70))),
            bootstrap_mode=str(payload.get("bootstrap_mode", "cluster_original_case")),
            bootstrap_seed=int(str(payload.get("bootstrap_seed", 17))),
            route=str(payload["route"]) if payload.get("route") else None,
            dataset=str(payload["dataset"]) if payload.get("dataset") else None,
            endpoint_version=str(
                payload.get("endpoint_version", "clinical_validation_v2")
            ),
            scenario_group=str(payload.get("scenario_group", "legacy")),
            population_contract=str(payload.get("population_contract", "")),
            required_case_ids_hash=(
                str(payload["required_case_ids_hash"])
                if payload.get("required_case_ids_hash") not in {None, ""}
                else None
            ),
            minimum_sample_size=(
                int(str(payload["minimum_sample_size"]))
                if payload.get("minimum_sample_size") is not None
                else None
            ),
            projection_contract_hash=str(
                payload.get("projection_contract_hash", "")
            ),
            source_contract_hash=str(payload.get("source_contract_hash", "")),
            physician_coder_hash=str(payload.get("physician_coder_hash", "")),
            benchmark_mapping_hash=str(
                payload.get("benchmark_mapping_hash", "")
            ),
            cluster_key=str(payload.get("cluster_key", "original_case_id")),
            bootstrap_replicates=(
                int(str(payload["bootstrap_replicates"]))
                if payload.get("bootstrap_replicates") is not None
                else None
            ),
            expected_baseline_kappa=(
                float(str(payload["expected_baseline_kappa"]))
                if payload.get("expected_baseline_kappa") is not None
                else None
            ),
            expected_baseline_observed_agreement=(
                float(str(payload["expected_baseline_observed_agreement"]))
                if payload.get("expected_baseline_observed_agreement") is not None
                else None
            ),
            expected_baseline_status=(
                str(payload["expected_baseline_status"])
                if payload.get("expected_baseline_status") not in {None, ""}
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class KappaResult:
    sample_size: int
    labels: tuple[str, ...]
    observed_agreement: float | None
    expected_agreement: float | None
    kappa: float | None
    confidence_interval: tuple[float | None, float | None]
    status: str
    reason: str
    confusion_matrix: dict[str, dict[str, int]]
    reference_margins: dict[str, int]
    comparison_margins: dict[str, int]
    case_ids: tuple[str, ...]
    class_metrics: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    prevalence_index: float | None = None
    bias_index: float | None = None
    maximum_attainable_kappa: float | None = None
    fixed_margin_threshold_attainable: bool | None = None
    approximate_additional_agreements_for_threshold: int | None = None
    bootstrap_replicates: int = 0
    bootstrap_failed_replicates: int = 0
    observed_agreement_count: int = 0
    maximum_fixed_margin_agreement_count: int | None = None
    threshold_attainability: dict[str, dict[str, float | int | bool | None]] = field(
        default_factory=dict
    )
    permutation_p_value: float | None = None
    permutation_replicates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationRunManifest:
    run_id: str
    input_hashes: dict[str, str]
    package_version: str
    ontology_hash: str
    scenario_registry_hash: str
    random_seed: int
    environment: dict[str, str]
    timestamp_utc: str
    output_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
