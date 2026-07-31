from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tm_ecg.modeling.confirmatory_split import (
    SPLIT_NAMES,
    create_sealed_confirmatory_split,
    verify_sealed_confirmatory_split,
)


def _synthetic_index(path: Path) -> None:
    labels = (
        "Normal",
        "Other / unmapped",
        "AF,Other / unmapped",
        "PVC",
        "APB",
        "RBBB spectrum",
        "LBBB spectrum",
        "Paced",
        "AFL",
    )
    rows: list[dict[str, object]] = []
    for patient in range(1, 81):
        for record in range(2):
            rows.append(
                {
                    "record_id": f"{patient}-{record}",
                    "patient_id": f"patient-{patient:03d}",
                    "strat_fold": str((patient % 8) + 1),
                    "labels": labels[(patient + record) % len(labels)],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_sealed_confirmatory_split_is_deterministic_and_patient_disjoint(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.csv"
    output = tmp_path / "sealed_internal_confirmatory_v1.json"
    _synthetic_index(index)

    first = create_sealed_confirmatory_split(
        index_path=index,
        output_path=output,
        salt="fixed-test-salt",
    )
    second = create_sealed_confirmatory_split(
        index_path=index,
        output_path=output,
        salt="fixed-test-salt",
    )

    assert first["salted_split_hash"] == second["salted_split_hash"]
    patient_sets = {
        split: set(first["patient_ids"][split]) for split in SPLIT_NAMES
    }
    assert set.union(*patient_sets.values()) == {
        f"patient-{patient:03d}" for patient in range(1, 81)
    }
    assert all(
        patient_sets[left].isdisjoint(patient_sets[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )
    verification = verify_sealed_confirmatory_split(output)
    assert verification["passed"] is True
    assert first["confirmatory_labels_opened"] is False


def test_sealed_manifest_hash_detects_mutation(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    output = tmp_path / "sealed_internal_confirmatory_v1.json"
    _synthetic_index(index)
    create_sealed_confirmatory_split(
        index_path=index,
        output_path=output,
        salt="fixed-test-salt",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["frozen"] = False
    output.write_text(json.dumps(payload), encoding="utf-8")

    verification = verify_sealed_confirmatory_split(output)

    assert verification["passed"] is False
    assert verification["checks"]["manifest_hash_matches"] is False
    assert verification["checks"]["frozen"] is False

