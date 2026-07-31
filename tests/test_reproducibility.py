from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tm_ecg.reproducibility import (
    build_artifact_manifest,
    load_exact_lock,
    verify_public_manifest,
    write_public_manifest,
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


def _git(*arguments: str, cwd: Path) -> None:
    subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True)


def test_public_manifest_is_deterministic_and_self_excluding(tmp_path) -> None:
    _git("init", "--quiet", cwd=tmp_path)
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")
    _git("add", "alpha.txt", "nested/beta.txt", cwd=tmp_path)

    destination = write_public_manifest(tmp_path)
    _git("add", "MANIFEST.sha256", cwd=tmp_path)
    result = verify_public_manifest(tmp_path)

    assert destination.name == "MANIFEST.sha256"
    assert result["valid"] is True
    assert result["public_file_count"] == 2
    assert result["manifest_entry_count"] == 2
    assert "MANIFEST.sha256" not in destination.read_text(encoding="utf-8")


def test_public_manifest_reports_missing_extra_and_modified_files(tmp_path) -> None:
    _git("init", "--quiet", cwd=tmp_path)
    (tmp_path / "tracked.txt").write_text("original", encoding="utf-8")
    _git("add", "tracked.txt", cwd=tmp_path)
    write_public_manifest(tmp_path)
    _git("add", "MANIFEST.sha256", cwd=tmp_path)

    (tmp_path / "tracked.txt").write_text("modified", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    _git("add", "extra.txt", cwd=tmp_path)
    result = verify_public_manifest(tmp_path)

    assert result["valid"] is False
    assert result["missing_from_manifest"] == ["extra.txt"]
    assert result["hash_mismatches"] == ["tracked.txt"]
