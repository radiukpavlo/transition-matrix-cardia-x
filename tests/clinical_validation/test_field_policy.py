from __future__ import annotations

import pytest

from tm_ecg.clinical_validation.field_policy import (
    FieldPolicyViolation,
    assert_sentinel_absent,
    build_physician_view,
)


def test_prohibited_benchmark_field_is_rejected() -> None:
    with pytest.raises(FieldPolicyViolation, match="hidden_benchmark_label"):
        build_physician_view({"case_id": "P001", "hidden_benchmark_label": "SENTINEL"})


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(FieldPolicyViolation, match="Unregistered"):
        build_physician_view({"case_id": "P001", "future_leaky_column": "x"})


def test_recursive_sentinel_check() -> None:
    with pytest.raises(FieldPolicyViolation):
        assert_sentinel_absent({"nested": ["LEAK_SENTINEL"]}, "LEAK_SENTINEL")

