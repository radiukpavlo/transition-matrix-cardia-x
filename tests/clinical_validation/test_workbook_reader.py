from __future__ import annotations

from openpyxl import load_workbook
import pytest

from tm_ecg.clinical_validation.workbook_reader import (
    WorkbookSchemaError,
    normalize_cell,
    read_completed_workbook,
)


def test_reader_reconciles_locked_populations(synthetic_validation_inputs) -> None:
    workbook, provenance = synthetic_validation_inputs
    extraction = read_completed_workbook(workbook, provenance)
    assert extraction.response_import_audit["total_rows"] == 150
    assert extraction.response_import_audit["b1_unique"] == 100
    assert extraction.response_import_audit["b1_repeats"] == 10
    assert extraction.response_import_audit["b2_rows"] == 40
    assert extraction.response_import_audit["unique_b1_route_counts"] == {
        "exact_match": 30,
        "conflict_region": 30,
        "structural_abstention": 40,
    }
    assert extraction.rule_review_summary["status"] == "not_estimable_incomplete_rulebook"


def test_reader_fails_closed_on_case_sheet_disagreement(synthetic_validation_inputs) -> None:
    workbook_path, provenance = synthetic_validation_inputs
    workbook = load_workbook(workbook_path)
    workbook["P001"]["M7"] = "tampered"
    workbook.save(workbook_path)
    with pytest.raises(WorkbookSchemaError, match="disagrees"):
        read_completed_workbook(workbook_path, provenance)


def test_unicode_cell_normalization_is_stable() -> None:
    assert normalize_cell("  A\u00a0 B  ") == "A B"
