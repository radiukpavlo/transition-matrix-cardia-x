"""Test bootstrap for source-tree execution.

The repository is usually run either as an installed package or via PYTHONPATH=src.
This file makes plain `pytest` work in a freshly unpacked archive, which is useful for
reproducibility reviews and offline artifact checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
