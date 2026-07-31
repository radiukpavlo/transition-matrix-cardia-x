from __future__ import annotations

import json

import pytest

from tm_ecg.reproducibility import (
    build_artifact_manifest,
    load_exact_lock,
    write_artifact_manifest,
)


def test_exact_lock_rejects_ranges_and_duplicate_names(tmp_path) -> None:
    ranged = tmp_path / "ranged.lock"
    ranged.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact"):
        load_exact_lock(ranged)
    duplicate = tmp_path / "duplicate.lock"
    duplicate.write_text("scikit_learn==1\nscikit-learn==1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_exact_lock(duplicate)


def test_artifact_manifest_is_deterministic_and_self_excluding(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    first = build_artifact_manifest(
        tmp_path,
        producer_command="tm-ecg test",
        input_hashes={"source": "abc"},
        code_root=tmp_path,
    )
    destination = write_artifact_manifest(
        tmp_path,
        producer_command="tm-ecg test",
        input_hashes={"source": "abc"},
        code_root=tmp_path,
    )
    second = build_artifact_manifest(
        tmp_path,
        producer_command="tm-ecg test",
        input_hashes={"source": "abc"},
        code_root=tmp_path,
    )
    assert first == second
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["artifact_count"] == 2
    assert {item["path"] for item in payload["artifacts"]} == {
        "a.txt",
        "nested/b.txt",
    }
