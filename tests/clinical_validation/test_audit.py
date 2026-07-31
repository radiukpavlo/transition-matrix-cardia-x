from __future__ import annotations

from tm_ecg.clinical_validation.audit import (
    canonical_json,
    read_jsonl,
    sha256_payload,
    write_jsonl,
)


def test_hash_is_independent_of_mapping_insertion_order() -> None:
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_jsonl_round_trip_is_utf8_and_deterministic(tmp_path) -> None:
    path = write_jsonl(tmp_path / "rows.jsonl", [{"case": "Łódź", "value": 1}])
    assert read_jsonl(path) == [{"case": "Łódź", "value": 1}]

