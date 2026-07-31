from __future__ import annotations

from pathlib import Path

import pytest

from tm_ecg.ptbxl_semantics import (
    StatementMetadata,
    load_statement_metadata,
    statement_state,
)


ROOT = Path(__file__).resolve().parents[1]
STATEMENTS = (
    ROOT
    / "data"
    / "raw"
    / "ptbxl"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    / "scp_statements.csv"
)


@pytest.fixture(scope="module")
def metadata() -> dict[str, StatementMetadata]:
    if not STATEMENTS.is_file():
        pytest.skip("PTB-XL metadata is available after the public dataset is installed")
    return load_statement_metadata(STATEMENTS)


@pytest.mark.parametrize("code", ["SR", "AFIB", "AFLT", "PAC", "PVC"])
def test_presence_coded_rhythm_and_form_statements_are_present_at_zero(
    metadata: dict[str, StatementMetadata],
    code: str,
) -> None:
    assert metadata[code].presence_coded
    assert statement_state(code, 0.0, metadata[code]) == "present"


def test_diagnostic_statement_uses_likelihood_bands(
    metadata: dict[str, StatementMetadata],
) -> None:
    assert metadata["NORM"].category == "diagnostic_statement"
    assert statement_state("NORM", 100.0, metadata["NORM"]) == "present"
    assert statement_state("NORM", 60.0, metadata["NORM"]) == "uncertain"
    assert statement_state("NORM", 0.0, metadata["NORM"]) == "ignored"


def test_statement_metadata_hash_and_classification_are_deterministic(
    metadata: dict[str, StatementMetadata],
) -> None:
    second = load_statement_metadata(STATEMENTS)
    assert metadata == second
    assert len({row.source_sha256 for row in metadata.values()}) == 1
    assert metadata["NDT"].category == "mixed_category"


def test_invalid_likelihood_bands_fail_closed(
    metadata: dict[str, StatementMetadata],
) -> None:
    with pytest.raises(ValueError, match="confidence bands"):
        statement_state(
            "SR",
            0.0,
            metadata["SR"],
            accepted_min=40.0,
            uncertain_min=50.0,
        )
