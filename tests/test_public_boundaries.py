from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from tm_ecg import __version__
from tm_ecg.config import ProjectConfig


ROOT = Path(__file__).resolve().parents[1]


def _run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_manuscript_boundary_remains_ignored_and_untracked() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/manuscript/" in gitignore
    assert _run_git("check-ignore", "--quiet", "--", "manuscript/main.tex").returncode == 0
    assert _run_git("ls-files", "--error-unmatch", "manuscript/main.tex").returncode != 0
    assert _run_git("check-ignore", "--quiet", "--", "data/archives/.gitkeep").returncode != 0


def test_runtime_and_packaging_versions_are_consistent() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        package = tomllib.load(handle)["project"]
    with (ROOT / "configs" / "defaults.toml").open("rb") as handle:
        config_payload = tomllib.load(handle)

    assert __version__ == package["version"] == config_payload["project"]["version"] == "1.0.0"
    assert package["requires-python"] == ">=3.12,<3.14"
    assert ProjectConfig.load(ROOT / "configs" / "defaults.toml").version == __version__


def test_dockerfile_contains_offline_runtime_contracts() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for path in ("configs", "clinical_validation/config", "results", "schemas"):
        assert f"COPY {path} " in dockerfile
    assert "tm-ecg-verify-reported" in dockerfile
