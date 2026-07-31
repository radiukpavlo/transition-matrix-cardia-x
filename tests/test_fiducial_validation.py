from __future__ import annotations

import pytest

from tm_ecg.modeling.fiducial_validation import parse_ludb_annotation_waves


def test_ludb_wave_parser_extracts_triplets_and_ignores_u_waves() -> None:
    samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    symbols = ["(", "p", ")", "(", "N", ")", "(", "t", ")", "(", "u", ")"]

    waves = parse_ludb_annotation_waves(samples, symbols)

    assert [(wave.kind, wave.onset, wave.peak, wave.offset) for wave in waves] == [
        ("p", 10, 20, 30),
        ("qrs", 40, 50, 60),
        ("t", 70, 80, 90),
    ]


def test_ludb_wave_parser_rejects_misaligned_annotations() -> None:
    with pytest.raises(ValueError, match="aligned"):
        parse_ludb_annotation_waves([1], ["(", "N", ")"])


def test_ludb_wave_parser_skips_incomplete_triplet() -> None:
    assert parse_ludb_annotation_waves([10, 20], ["(", "N"]) == []
