"""Single category-aware PTB-XL statement semantics contract."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from tm_ecg.clinical_validation.audit import sha256_file


# Frozen compatibility fallback for callers that cannot supply scp_statements.csv.
# This is the PTB-XL 1.0.3 rhythm/form vocabulary formerly embedded in
# tm_ecg.ontology. Production indexing and clinical validation supply the hashed
# metadata table; the fallback exists only to preserve category semantics at
# lightweight API boundaries and is intentionally centralized here.
PTBXL_1_0_3_PRESENCE_OVERRIDES = frozenset(
    {
        "ABQRS",
        "AFIB",
        "AFLT",
        "BIGU",
        "DIG",
        "HVOLT",
        "INVT",
        "LNGQT",
        "LOWT",
        "LPR",
        "LVOLT",
        "NDT",
        "NST_",
        "NT_",
        "PAC",
        "PACE",
        "PRC(S)",
        "PSVT",
        "PVC",
        "QWAVE",
        "SARRH",
        "SBRAD",
        "SR",
        "STACH",
        "STD_",
        "STE_",
        "SVARR",
        "SVTAC",
        "TAB_",
        "TRIGU",
        "VCLVH",
        # Compatibility aliases used by supported source vocabularies.
        "VPB",
        "APB",
        "SPAC",
        "SVPB",
    }
)


@dataclass(frozen=True, slots=True)
class StatementMetadata:
    code: str
    description: str
    diagnostic: bool
    rhythm: bool
    form: bool
    statement_category: str
    diagnostic_class: str
    diagnostic_subclass: str
    source_sha256: str

    @property
    def category(self) -> str:
        categories = [
            name
            for name, present in (
                ("diagnostic", self.diagnostic),
                ("rhythm", self.rhythm),
                ("form", self.form),
            )
            if present
        ]
        if len(categories) > 1:
            return "mixed_category"
        return f"{categories[0]}_statement" if categories else "unknown"

    @property
    def presence_coded(self) -> bool:
        return self.rhythm or self.form

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "category": self.category}


def _flag(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "1.0", "true", "yes"}


def load_statement_metadata(path: str | Path) -> dict[str, StatementMetadata]:
    """Load the published PTB-XL statement categories with a source hash."""

    source = Path(path)
    source_hash = sha256_file(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"PTB-XL statement metadata is empty: {source}")
        code_field = reader.fieldnames[0]
        required = {"description", "diagnostic", "form", "rhythm"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"PTB-XL statement metadata is missing columns: {missing}")
        metadata: dict[str, StatementMetadata] = {}
        for row_number, row in enumerate(reader, start=2):
            code = str(row.get(code_field, "") or "").strip().upper()
            if not code:
                continue
            if code in metadata:
                raise ValueError(
                    f"Duplicate PTB-XL statement code {code} at row {row_number}"
                )
            metadata[code] = StatementMetadata(
                code=code,
                description=str(row.get("description", "") or "").strip(),
                diagnostic=_flag(row.get("diagnostic")),
                rhythm=_flag(row.get("rhythm")),
                form=_flag(row.get("form")),
                statement_category=str(row.get("Statement Category", "") or "").strip(),
                diagnostic_class=str(row.get("diagnostic_class", "") or "").strip(),
                diagnostic_subclass=str(
                    row.get("diagnostic_subclass", "") or ""
                ).strip(),
                source_sha256=source_hash,
            )
    if not metadata:
        raise ValueError(f"PTB-XL statement metadata contains no codes: {source}")
    return metadata


def statement_state(
    code: str,
    likelihood: float,
    metadata: StatementMetadata | None,
    *,
    accepted_min: float = 80.0,
    uncertain_min: float = 50.0,
) -> str:
    """Classify a source statement as present, uncertain, or ignored."""

    if not 0.0 <= uncertain_min <= accepted_min <= 100.0:
        raise ValueError(
            "Statement confidence bands must satisfy "
            "0 <= uncertain_min <= accepted_min <= 100"
        )
    normalized_code = code.strip().upper()
    if metadata is not None and metadata.code != normalized_code:
        raise ValueError(
            f"Statement metadata/code mismatch: {metadata.code} != {normalized_code}"
        )
    if (
        metadata is not None
        and metadata.presence_coded
        or metadata is None
        and normalized_code in PTBXL_1_0_3_PRESENCE_OVERRIDES
    ):
        return "present"
    if likelihood >= accepted_min:
        return "present"
    if likelihood >= uncertain_min:
        return "uncertain"
    return "ignored"


def source_statement_trace(
    likelihoods: Mapping[str, float],
    metadata: Mapping[str, StatementMetadata],
    *,
    accepted_min: float = 80.0,
    uncertain_min: float = 50.0,
) -> tuple[dict[str, object], ...]:
    """Return a deterministic, fully traced state for every source code."""

    trace: list[dict[str, object]] = []
    for raw_code, raw_likelihood in sorted(likelihoods.items()):
        code = str(raw_code).strip().upper()
        likelihood = float(raw_likelihood)
        statement = metadata.get(code)
        trace.append(
            {
                "code": code,
                "likelihood": likelihood,
                "category": (
                    statement.category
                    if statement
                    else (
                        "frozen_presence_override"
                        if code in PTBXL_1_0_3_PRESENCE_OVERRIDES
                        else "unknown"
                    )
                ),
                "presence_coded": (
                    statement.presence_coded
                    if statement
                    else code in PTBXL_1_0_3_PRESENCE_OVERRIDES
                ),
                "state": statement_state(
                    code,
                    likelihood,
                    statement,
                    accepted_min=accepted_min,
                    uncertain_min=uncertain_min,
                ),
                "metadata_source_sha256": (
                    statement.source_sha256 if statement else None
                ),
            }
        )
    return tuple(trace)
