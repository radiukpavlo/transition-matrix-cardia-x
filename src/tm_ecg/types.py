"""Typed models used across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SourceArchive:
    dataset: str
    archive_path: str
    sha256: str
    extracted_dir: str
    extracted_at_utc: str
    archive_version: str
    verified_against_manifest: bool = False
    source_mode: str = "zip"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceManifest:
    generated_at_utc: str
    archives: list[SourceArchive]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "archives": [item.to_dict() for item in self.archives],
            "notes": self.notes,
        }


@dataclass(slots=True)
class OntologyMapping:
    source_dataset: str
    source_label: str
    project_label: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecordIndexRow:
    row_id: str
    record_id: str
    patient_id: str | None
    dataset: str
    split: str
    labels: list[str]
    source_path: str
    preprocessing_hash: str
    ontology_version: str
    raw_source_labels: list[str] = field(default_factory=list)
    axis_targets: dict[str, list[str] | str] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    included: bool = True
    exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["labels"] = ",".join(self.labels)
        payload["raw_source_labels"] = "|".join(self.raw_source_labels)
        payload.pop("axis_targets", None)
        payload.pop("source_metadata", None)
        payload["axis_targets_json"] = json.dumps(
            self.axis_targets, ensure_ascii=False, sort_keys=True
        )
        payload["source_metadata_json"] = json.dumps(
            self.source_metadata, ensure_ascii=False, sort_keys=True
        )
        return payload


@dataclass(frozen=True, slots=True)
class MultiAxialTarget:
    """Non-exclusive ECG target used before compatibility projection."""

    rhythm: tuple[str, ...] = ()
    ectopy: tuple[str, ...] = ()
    conduction: tuple[str, ...] = ()
    repolarization: tuple[str, ...] = ()
    pacing: str = "absent"
    normality: str = "unknown"
    quality: str = "unknown"
    unsupported_source_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureQualityState:
    """Availability and confidence carried with a derived feature."""

    state: str
    valid_beat_fraction: float | None = None
    lead_coverage: float | None = None
    detector_agreement: float | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in {"observed", "inferred", "low_confidence", "unavailable"}:
            raise ValueError(f"Unknown feature-quality state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TypedAbstention:
    reason: str
    failed_requirement: str
    needed_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SplitManifest:
    dataset: str
    generated_at_utc: str
    seed: int
    split_assignments: list[RecordIndexRow]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "generated_at_utc": self.generated_at_utc,
            "seed": self.seed,
            "split_assignments": [row.to_dict() for row in self.split_assignments],
            "notes": self.notes,
        }


@dataclass(slots=True)
class BeatFiducials:
    beat_id: str
    record_id: str
    p_on: float | None = None
    p_peak: float | None = None
    p_off: float | None = None
    qrs_on: float | None = None
    r_peak: float | None = None
    qrs_off: float | None = None
    t_on: float | None = None
    t_peak: float | None = None
    t_off: float | None = None
    source: str = "algorithmic"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BeatAcceptance:
    beat_id: str
    record_id: str
    accepted: bool
    reasons: list[str]
    lead_quality_db: float | None = None
    fiducial_completeness: float | None = None
    pacing_corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = ",".join(self.reasons)
        return payload


@dataclass(slots=True)
class TriadMembership:
    triad_id: str
    record_id: str
    previous_beat_id: str
    current_beat_id: str
    next_beat_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureDictionaryRow:
    column: str
    family: str
    unit: str
    value_type: str
    level: str
    formula: str
    included: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TransformColumnStats:
    column: str
    family: str
    mean: float
    std: float
    lower: float | None = None
    upper: float | None = None
    eps: float = 1e-3
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TransformBundle:
    dataset: str
    stats: list[TransformColumnStats]
    fit_columns: list[str]
    dropped_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "fit_columns": self.fit_columns,
            "dropped_columns": self.dropped_columns,
            "stats": [item.to_dict() for item in self.stats],
        }


@dataclass(slots=True)
class TransitionOperatorPackage:
    dataset: str
    lambda_value: float
    retained_rank: int
    zero_variance_columns: list[str]
    standardized_columns: list[str]
    singular_values: list[float]
    operator_path: str
    transform_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExplanationPacket:
    record_id: str
    prediction: str
    measured_features_path: str
    translated_features_path: str
    discrepancy_summary: list[str]
    waveform_snapshot_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    data_lock: Path
    raw: Path
    interim: Path
    features: Path
    latents: Path
    models: Path
    transition: Path
    reports: Path
    manifests: Path
    logs: Path
