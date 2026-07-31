"""Environment bootstrap and compatibility inspection."""

from __future__ import annotations

import platform
import sys

from tm_ecg.config import ProjectConfig
from tm_ecg.stages.shared import write_stage_manifest


CORE_PACKAGES = ["numpy", "scipy", "pandas", "pyarrow", "sklearn", "wfdb"]
TRAIN_PACKAGES = ["torch"]


def run(config: ProjectConfig, args: object) -> int:
    report = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "core_packages": {},
        "train_packages": {},
    }
    missing_core = []
    for package in CORE_PACKAGES:
        try:
            __import__(package)
            report["core_packages"][package] = "installed"
        except ImportError:
            report["core_packages"][package] = "missing"
            missing_core.append(package)

    for package in TRAIN_PACKAGES:
        try:
            __import__(package)
            report["train_packages"][package] = "installed"
        except ImportError:
            report["train_packages"][package] = "missing"

    major, minor = sys.version_info[:2]
    report["recommended_runtime"] = (
        "python3.12"
        if major == 3 and minor >= 13 and missing_core
        else f"python{major}.{minor}"
    )
    report["notes"] = [
        "Use the installed interpreter when wheels exist.",
        "Fall back to Python 3.12 when scientific wheels are unavailable on 3.13.",
    ]
    write_stage_manifest(config, "bootstrap_env", report)
    print(f"Environment report written. Recommended runtime: {report['recommended_runtime']}")
    return 0
