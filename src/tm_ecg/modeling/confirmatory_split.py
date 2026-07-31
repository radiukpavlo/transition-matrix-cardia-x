"""Create and verify the sealed patient-level compatibility partition."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from tm_ecg.config import ProjectConfig
from tm_ecg.io.readers import read_table_frame
from tm_ecg.modeling.label_contract import DEFAULT_COMPATIBILITY_CONTRACT_V4


SPLIT_ID = "sealed_internal_confirmatory_v1"
SPLIT_NAMES = (
    "development_train",
    "development_validation",
    "sealed_internal_confirmatory",
)
SPLIT_RATIOS = {
    "development_train": 0.70,
    "development_validation": 0.15,
    "sealed_internal_confirmatory": 0.15,
}
DEFAULT_SPLIT_SALT = "CARDIA-X::sealed_internal_confirmatory_v1::2026-07-24"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _patient_key(patient_id: object, record_id: object) -> str:
    text = str(patient_id).strip()
    return text if text and text.lower() != "nan" else f"record::{record_id}"


def _salted_rank(patient_id: str, salt: str) -> str:
    return _sha256_bytes(f"{salt}\0{patient_id}".encode("utf-8"))


def _target_labels(value: object) -> tuple[str, ...]:
    return DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
        value,
        empty_policy="error",
    )


def _target_contract_hash() -> str:
    contract = DEFAULT_COMPATIBILITY_CONTRACT_V4
    return _canonical_hash(
        {
            "version": contract.version,
            "contract_id": contract.contract_id,
            "label_order": contract.label_order,
            "specific_labels": contract.specific_labels,
            "normal_label": contract.normal_label,
            "residual_label": contract.residual_label,
            "af_afl_mutually_exclusive": contract.af_afl_mutually_exclusive,
            "mixed_ectopy_allowed": contract.mixed_ectopy_allowed,
            "pacing_conduction_cooccurrence": (
                contract.pacing_conduction_cooccurrence
            ),
            "aliases": contract.aliases,
        }
    )


def _patient_profiles(frame: object) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for row in frame.itertuples(index=False):  # type: ignore[union-attr]
        record_id = str(getattr(row, "record_id"))
        patient_id = _patient_key(getattr(row, "patient_id"), record_id)
        labels = _target_labels(getattr(row, "labels"))
        label_set = " | ".join(labels)
        profile = profiles.setdefault(
            patient_id,
            {
                "record_count": 0,
                "label_counts": Counter(),
                "set_counts": Counter(),
            },
        )
        profile["record_count"] = int(profile["record_count"]) + 1
        label_counts = profile["label_counts"]
        set_counts = profile["set_counts"]
        if not isinstance(label_counts, Counter) or not isinstance(
            set_counts, Counter
        ):
            raise TypeError("Invalid patient profile")
        label_counts.update(labels)
        set_counts.update([label_set])
    return profiles


def _greedy_stratified_assignment(
    profiles: Mapping[str, Mapping[str, object]],
    *,
    salt: str,
) -> dict[str, list[str]]:
    total_records = sum(int(profile["record_count"]) for profile in profiles.values())
    total_patients = len(profiles)
    label_totals: Counter[str] = Counter()
    set_totals: Counter[str] = Counter()
    patient_label_support: Counter[str] = Counter()
    patient_set_support: Counter[str] = Counter()
    for profile in profiles.values():
        label_counts = profile["label_counts"]
        set_counts = profile["set_counts"]
        if not isinstance(label_counts, Counter) or not isinstance(
            set_counts, Counter
        ):
            raise TypeError("Invalid patient profile counters")
        label_totals.update(label_counts)
        set_totals.update(set_counts)
        patient_label_support.update(label_counts.keys())
        patient_set_support.update(set_counts.keys())

    def rarity(patient_id: str) -> float:
        profile = profiles[patient_id]
        label_counts = profile["label_counts"]
        set_counts = profile["set_counts"]
        if not isinstance(label_counts, Counter) or not isinstance(
            set_counts, Counter
        ):
            return 0.0
        label_rarity = max(
            (1.0 / patient_label_support[label] for label in label_counts),
            default=0.0,
        )
        set_rarity = max(
            (1.0 / patient_set_support[label_set] for label_set in set_counts),
            default=0.0,
        )
        return max(label_rarity, set_rarity)

    ordered = sorted(
        profiles,
        key=lambda patient_id: (
            -rarity(patient_id),
            -int(profiles[patient_id]["record_count"]),
            _salted_rank(patient_id, salt),
        ),
    )
    assigned = {name: [] for name in SPLIT_NAMES}
    record_counts = Counter({name: 0 for name in SPLIT_NAMES})
    patient_counts = Counter({name: 0 for name in SPLIT_NAMES})
    label_counts_by_split = {name: Counter() for name in SPLIT_NAMES}
    set_counts_by_split = {name: Counter() for name in SPLIT_NAMES}

    def incremental_score(split: str, patient_id: str) -> float:
        ratio = SPLIT_RATIOS[split]
        profile = profiles[patient_id]
        patient_records = int(profile["record_count"])
        labels = profile["label_counts"]
        sets = profile["set_counts"]
        if not isinstance(labels, Counter) or not isinstance(sets, Counter):
            raise TypeError("Invalid patient profile counters")

        def delta(current: float, addition: float, target: float) -> float:
            scale = max(target, 1.0)
            return (
                ((current + addition - target) / scale) ** 2
                - ((current - target) / scale) ** 2
            )

        score = 4.0 * delta(
            record_counts[split],
            patient_records,
            total_records * ratio,
        )
        score += 1.5 * delta(
            patient_counts[split],
            1,
            total_patients * ratio,
        )
        for label, count in labels.items():
            score += 2.0 * delta(
                label_counts_by_split[split][label],
                count,
                label_totals[label] * ratio,
            )
        for label_set, count in sets.items():
            # Exact-set balance is useful, but strongly regularize very rare
            # sets so one patient cannot dominate the entire split objective.
            weight = min(1.0, set_totals[label_set] / 10.0)
            score += weight * delta(
                set_counts_by_split[split][label_set],
                count,
                set_totals[label_set] * ratio,
            )
        target_records = total_records * ratio
        projected = record_counts[split] + patient_records
        if projected > math.ceil(target_records * 1.03):
            score += 100.0 * (projected - target_records) / max(
                target_records, 1.0
            )
        return score

    for patient_id in ordered:
        split = min(
            SPLIT_NAMES,
            key=lambda name: (
                incremental_score(name, patient_id),
                _salted_rank(f"{patient_id}\0{name}", salt),
            ),
        )
        assigned[split].append(patient_id)
        profile = profiles[patient_id]
        record_counts[split] += int(profile["record_count"])
        patient_counts[split] += 1
        labels = profile["label_counts"]
        sets = profile["set_counts"]
        if isinstance(labels, Counter):
            label_counts_by_split[split].update(labels)
        if isinstance(sets, Counter):
            set_counts_by_split[split].update(sets)

    for split in SPLIT_NAMES:
        assigned[split].sort()
        if not assigned[split]:
            raise RuntimeError(f"Sealed split algorithm produced empty {split}")
    return assigned


def _manifest_payload(
    *,
    source_path: Path,
    profiles: Mapping[str, Mapping[str, object]],
    assignments: Mapping[str, list[str]],
    salt: str,
) -> dict[str, object]:
    all_ids = [item for split in SPLIT_NAMES for item in assignments[split]]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(profiles):
        raise RuntimeError("Patient assignment is incomplete or overlapping")
    counts = {
        split: {
            "patients": len(assignments[split]),
            "records": sum(
                int(profiles[patient_id]["record_count"])
                for patient_id in assignments[split]
            ),
        }
        for split in SPLIT_NAMES
    }
    split_hash = _canonical_hash(
        {
            "salt": salt,
            "assignments": {
                split: assignments[split] for split in SPLIT_NAMES
            },
        }
    )
    return {
        "version": 1,
        "split_id": SPLIT_ID,
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "frozen": True,
        "confirmatory_labels_opened": False,
        "source_index": str(source_path),
        "source_index_sha256": _sha256_file(source_path),
        "source_folds": [1, 2, 3, 4, 5, 6, 7, 8],
        "excluded_historical_folds": {
            "9": "transport_validation_only",
            "10": "consumed_historical_test_only",
        },
        "target_contract": DEFAULT_COMPATIBILITY_CONTRACT_V4.contract_id,
        "target_contract_hash": _target_contract_hash(),
        "algorithm": "patient_grouped_greedy_multilabel_stratification_v1",
        "split_ratios": SPLIT_RATIOS,
        "salt_sha256": _sha256_bytes(salt.encode("utf-8")),
        "salted_split_hash": split_hash,
        "counts": counts,
        "patient_id_hashes": {
            split: _canonical_hash(assignments[split]) for split in SPLIT_NAMES
        },
        "patient_ids": {
            split: assignments[split] for split in SPLIT_NAMES
        },
    }


def create_sealed_confirmatory_split(
    *,
    index_path: str | Path,
    output_path: str | Path,
    salt: str = DEFAULT_SPLIT_SALT,
) -> dict[str, object]:
    source = Path(index_path).resolve()
    output = Path(output_path).resolve()
    frame = read_table_frame(source)
    required = {"record_id", "patient_id", "strat_fold", "labels"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PTB-XL index is missing split columns: {sorted(missing)}")
    development = frame[
        frame["strat_fold"].astype(str).isin(
            {"1", "2", "3", "4", "5", "6", "7", "8"}
        )
    ].copy()
    if development.empty:
        raise ValueError("No PTB-XL development rows found in folds 1-8")
    profiles = _patient_profiles(development)
    assignments = _greedy_stratified_assignment(profiles, salt=salt)
    manifest = _manifest_payload(
        source_path=source,
        profiles=profiles,
        assignments=assignments,
        salt=salt,
    )
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        comparable_fields = (
            "split_id",
            "source_index_sha256",
            "target_contract_hash",
            "algorithm",
            "salt_sha256",
            "salted_split_hash",
            "counts",
            "patient_id_hashes",
            "patient_ids",
        )
        if any(existing.get(key) != manifest.get(key) for key in comparable_fields):
            raise RuntimeError(
                "Sealed confirmatory manifest already exists with different content"
            )
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    output.with_suffix(".sha256").write_text(
        f"{_sha256_file(output)}  {output.name}\n",
        encoding="utf-8",
    )
    return manifest


def verify_sealed_confirmatory_split(
    manifest_path: str | Path,
) -> dict[str, object]:
    source = Path(manifest_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    assignments = payload.get("patient_ids", {})
    if not isinstance(assignments, Mapping):
        raise ValueError("Sealed manifest patient_ids must be an object")
    sets = {
        split: set(str(item) for item in assignments.get(split, []))
        for split in SPLIT_NAMES
    }
    overlap = {
        f"{left}::{right}": sorted(sets[left].intersection(sets[right]))
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
        if sets[left].intersection(sets[right])
    }
    hash_path = source.with_suffix(".sha256")
    recorded_hash = (
        hash_path.read_text(encoding="utf-8").split()[0]
        if hash_path.exists()
        else ""
    )
    checks = {
        "frozen": payload.get("frozen") is True,
        "confirmatory_labels_unopened": (
            payload.get("confirmatory_labels_opened") is False
        ),
        "all_partitions_present": all(sets[split] for split in SPLIT_NAMES),
        "patient_disjoint": not overlap,
        "manifest_hash_matches": (
            bool(recorded_hash) and recorded_hash == _sha256_file(source)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "overlap": overlap,
        "counts": payload.get("counts", {}),
        "salted_split_hash": payload.get("salted_split_hash"),
        "manifest_sha256": _sha256_file(source),
    }


def run(config: ProjectConfig, args: object) -> int:
    root = config.paths.root
    index_value = str(
        getattr(args, "index", "artifacts/manifests/ptbxl_index.parquet")
    )
    output_value = str(
        getattr(
            args,
            "output",
            "artifacts/manifests/sealed_internal_confirmatory_v1.json",
        )
    )
    index_path = Path(index_value)
    output_path = Path(output_value)
    if not index_path.is_absolute():
        index_path = root / index_path
    if not output_path.is_absolute():
        output_path = root / output_path
    manifest = create_sealed_confirmatory_split(
        index_path=index_path,
        output_path=output_path,
        salt=str(getattr(args, "salt", DEFAULT_SPLIT_SALT)),
    )
    verification = verify_sealed_confirmatory_split(output_path)
    print(
        json.dumps(
            {
                "split_id": manifest["split_id"],
                "output": str(output_path),
                "counts": manifest["counts"],
                "verification": verification,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if verification["passed"] else 19
