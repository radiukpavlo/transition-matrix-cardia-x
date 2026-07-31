"""Audit the compatibility-v3 to residual-exclusive v4 target migration."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from tm_ecg.config import ProjectConfig
from tm_ecg.constants import PROJECT_LABELS
from tm_ecg.io.readers import read_table_frame
from tm_ecg.modeling.label_contract import (
    DEFAULT_COMPATIBILITY_CONTRACT_V4,
    parse_label_tokens,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(labels: Sequence[str]) -> str:
    return " | ".join(label for label in PROJECT_LABELS if label in set(labels))


def _historical_prediction_migration(frame: object) -> dict[str, object]:
    import numpy as np  # type: ignore

    keys = [
        label.lower().replace(" / ", "_").replace(" ", "_")
        for label in PROJECT_LABELS
    ]
    true_columns = [f"true::{key}" for key in keys]
    predicted_columns = [f"predicted::{key}" for key in keys]
    missing = set(true_columns + predicted_columns) - set(frame.columns)  # type: ignore[union-attr]
    if missing:
        raise ValueError(
            f"Historical prediction table is missing columns: {sorted(missing)}"
        )
    truth_v3 = frame[true_columns].to_numpy(dtype=int)  # type: ignore[index]
    prediction_v3 = frame[predicted_columns].to_numpy(dtype=int)  # type: ignore[index]
    truth_v4 = truth_v3.copy()
    prediction_v4 = prediction_v3.copy()
    normal = PROJECT_LABELS.index("Normal")
    residual = PROJECT_LABELS.index("Other / unmapped")
    specific = [
        index
        for index in range(len(PROJECT_LABELS))
        if index not in {normal, residual}
    ]
    truth_v4[truth_v4[:, specific].any(axis=1), residual] = 0
    prediction_v4[prediction_v4[:, specific].any(axis=1), residual] = 0
    exact_v3 = np.all(truth_v3 == prediction_v3, axis=1)
    exact_v4 = np.all(truth_v4 == prediction_v4, axis=1)
    records = int(len(truth_v3))
    strict_minimum = math.floor(0.90 * records) + 1
    return {
        "partition": "consumed_historical_fold_10",
        "records": records,
        "v3_exact_successes": int(exact_v3.sum()),
        "v3_exact_subset_accuracy": float(exact_v3.mean()),
        "v4_exact_successes_with_same_predictions": int(exact_v4.sum()),
        "v4_exact_subset_accuracy_with_same_predictions": float(exact_v4.mean()),
        "successes_attributable_only_to_contract_migration": int(
            exact_v4.sum() - exact_v3.sum()
        ),
        "strict_minimum_successes_above_0_90": strict_minimum,
        "remaining_success_gap_after_migration": int(
            strict_minimum - exact_v4.sum()
        ),
        "truth_rows_changed": int(np.any(truth_v3 != truth_v4, axis=1).sum()),
        "prediction_rows_changed": int(
            np.any(prediction_v3 != prediction_v4, axis=1).sum()
        ),
        "confirmatory_evidence": False,
    }


def build_target_migration_report(
    *,
    index_path: str | Path,
    historical_predictions_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    index_source = Path(index_path).resolve()
    predictions_source = Path(historical_predictions_path).resolve()
    output = Path(output_path).resolve()
    index = read_table_frame(index_source)
    if not {"record_id", "patient_id", "labels"} <= set(index.columns):
        raise ValueError("Index must contain record_id, patient_id, and labels")

    old_frequency: Counter[str] = Counter()
    new_frequency: Counter[str] = Counter()
    old_support: Counter[str] = Counter()
    new_support: Counter[str] = Counter()
    rows_changed = 0
    old_residual_specific = 0
    new_residual_specific = 0
    migration_rows: list[dict[str, object]] = []
    for row in index.itertuples(index=False):
        old_tokens = tuple(
            label for label in PROJECT_LABELS if label in parse_label_tokens(row.labels)
        )
        new_tokens = DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
            row.labels,
            empty_policy="error",
        )
        old_set = _canonical(old_tokens)
        new_set = _canonical(new_tokens)
        old_frequency[old_set] += 1
        new_frequency[new_set] += 1
        old_support.update(old_tokens)
        new_support.update(new_tokens)
        old_conflict = (
            "Other / unmapped" in old_tokens
            and bool(
                set(old_tokens).intersection(
                    DEFAULT_COMPATIBILITY_CONTRACT_V4.specific_labels
                )
            )
        )
        new_conflict = (
            "Other / unmapped" in new_tokens
            and bool(
                set(new_tokens).intersection(
                    DEFAULT_COMPATIBILITY_CONTRACT_V4.specific_labels
                )
            )
        )
        old_residual_specific += int(old_conflict)
        new_residual_specific += int(new_conflict)
        if old_tokens != new_tokens:
            rows_changed += 1
            migration_rows.append(
                {
                    "record_id": str(row.record_id),
                    "old_label_set": old_set,
                    "new_label_set": new_set,
                }
            )

    predictions = read_table_frame(predictions_source)
    report = {
        "version": 1,
        "source_contract": "compatibility_v3",
        "target_contract": "compatibility_v4",
        "index": str(index_source),
        "index_sha256": _sha256_file(index_source),
        "historical_predictions": str(predictions_source),
        "historical_predictions_sha256": _sha256_file(predictions_source),
        "record_count_before": int(len(index)),
        "record_count_after": int(len(index)),
        "patient_count_before": int(
            index["patient_id"].fillna(index["record_id"]).astype(str).nunique()
        ),
        "patient_count_after": int(
            index["patient_id"].fillna(index["record_id"]).astype(str).nunique()
        ),
        "patient_count_unchanged": True,
        "rows_changed": rows_changed,
        "residual_specific_conflicts_before": old_residual_specific,
        "residual_specific_conflicts_after": new_residual_specific,
        "residual_specific_conflicts_removed": (
            old_residual_specific - new_residual_specific
        ),
        "old_label_set_frequency": dict(sorted(old_frequency.items())),
        "new_label_set_frequency": dict(sorted(new_frequency.items())),
        "per_class_support_before": {
            label: old_support[label] for label in PROJECT_LABELS
        },
        "per_class_support_after": {
            label: new_support[label] for label in PROJECT_LABELS
        },
        "historical_evaluation": _historical_prediction_migration(predictions),
        "sealed_confirmatory_labels_opened": False,
        "migration_rows": migration_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run(config: ProjectConfig, args: object) -> int:
    root = config.paths.root

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    report = build_target_migration_report(
        index_path=resolve(str(getattr(args, "index"))),
        historical_predictions_path=resolve(
            str(getattr(args, "historical_predictions"))
        ),
        output_path=resolve(str(getattr(args, "output"))),
    )
    summary = dict(report["historical_evaluation"])  # type: ignore[arg-type]
    summary.update(
        {
            "rows_changed": report["rows_changed"],
            "residual_specific_conflicts_removed": report[
                "residual_specific_conflicts_removed"
            ],
            "output": str(resolve(str(getattr(args, "output")))),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

