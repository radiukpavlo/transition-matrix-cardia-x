from __future__ import annotations

import json
from pathlib import Path

from tm_ecg.reported_results import verify


ROOT = Path(__file__).resolve().parents[1]


def test_reported_result_snapshot_recalculates() -> None:
    payload = json.loads((ROOT / "results" / "reported_metrics.json").read_text(encoding="utf-8"))
    summary = verify(payload)
    assert summary == {
        "reader_cases": 100,
        "compatibility_test_records": 2198,
        "semantic_transition_rows": 4,
        "ludb_landmarks": 9,
    }
