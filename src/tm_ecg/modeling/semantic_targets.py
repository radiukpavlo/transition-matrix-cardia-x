"""Versioned, leakage-resistant semantic target matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from tm_ecg.config import ProjectConfig
from tm_ecg.clinical_validation.ontology import load_json_yaml
from tm_ecg.io.readers import read_table_frame


@dataclass(frozen=True, slots=True)
class SemanticTargetSpec:
    feature_id: str
    group: str
    units: str
    lower: float | None
    upper: float | None
    missingness: str
    provenance: str

    @property
    def requires_crossfit(self) -> bool:
        return self.provenance.startswith("cross_fitted_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_semantic_target_contract(
    path: str | Path,
) -> tuple[dict[str, object], tuple[SemanticTargetSpec, ...]]:
    source = Path(path)
    payload = load_json_yaml(source)
    if payload.get("version") != 3:
        raise ValueError("Semantic target contract must declare version 3")
    rows = payload.get("features", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Semantic target contract contains no features")
    specs: list[SemanticTargetSpec] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Semantic target feature definitions must be objects")
        bounds = row.get("bounds", [None, None])
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Semantic target bounds must contain lower and upper")
        specs.append(
            SemanticTargetSpec(
                feature_id=str(row["feature_id"]),
                group=str(row["group"]),
                units=str(row["units"]),
                lower=float(bounds[0]) if bounds[0] is not None else None,
                upper=float(bounds[1]) if bounds[1] is not None else None,
                missingness=str(row["missingness"]),
                provenance=str(row["provenance"]),
            )
        )
    names = [item.feature_id for item in specs]
    if len(names) != len(set(names)):
        raise ValueError("Semantic target feature IDs must be unique")
    return payload, tuple(specs)


def build_oof_semantic_target_matrix(
    *,
    source_measurements_path: str | Path,
    specialist_oof_path: str | Path,
    contract_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Combine source measurements with held-out specialist predictions."""

    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    source_path = Path(source_measurements_path).resolve()
    specialist_path = Path(specialist_oof_path).resolve()
    contract_source = Path(contract_path).resolve()
    destination = Path(output_path).resolve()
    contract, specs = load_semantic_target_contract(contract_source)
    source = read_table_frame(source_path).copy()
    specialist = read_table_frame(specialist_path).copy()
    required_identity = {"record_id", "patient_id"}
    if not required_identity <= set(source.columns):
        raise ValueError("Source semantic measurements lack record_id/patient_id")
    required_specialist = {
        "record_id",
        "patient_id",
        "prediction_mode",
        "crossfit_fold",
    }
    if not required_specialist <= set(specialist.columns):
        raise ValueError(
            "Specialist OOF table lacks identity/crossfit provenance columns"
        )
    if not specialist["prediction_mode"].astype(str).eq("out_of_fold").all():
        raise ValueError("In-sample specialist predictions are forbidden in B")
    for frame in (source, specialist):
        frame["record_id"] = frame["record_id"].astype(str)
        frame["patient_id"] = frame["patient_id"].astype(str)
        if frame["record_id"].duplicated().any():
            raise ValueError("Semantic target inputs contain duplicate record IDs")
    joined = source.merge(
        specialist,
        on=["record_id", "patient_id"],
        how="left",
        validate="one_to_one",
        suffixes=("::source", "::oof"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Every source row requires exactly one OOF specialist row")
    output = joined[["record_id", "patient_id"]].copy()
    missing_counts: dict[str, int] = {}
    for spec in specs:
        suffix = "::oof" if spec.requires_crossfit else "::source"
        preferred = (
            f"{spec.feature_id}{suffix}"
            if f"{spec.feature_id}{suffix}" in joined.columns
            else spec.feature_id
        )
        if preferred not in joined.columns:
            raise ValueError(
                f"Semantic input is missing {spec.feature_id} "
                f"({spec.provenance})"
            )
        values = pd.to_numeric(joined[preferred], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        finite = values.dropna()
        if spec.lower is not None and (finite < spec.lower).any():
            raise ValueError(f"{spec.feature_id} violates its lower bound")
        if spec.upper is not None and (finite > spec.upper).any():
            raise ValueError(f"{spec.feature_id} violates its upper bound")
        output[spec.feature_id] = values
        output[f"missing::{spec.feature_id}"] = values.isna().astype(int)
        missing_counts[spec.feature_id] = int(values.isna().sum())
    output["crossfit_fold"] = joined["crossfit_fold"].astype(int)
    if not np.isfinite(output["crossfit_fold"]).all():
        raise ValueError("Crossfit fold assignments must be finite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    record_ids = output["record_id"].astype(str).tolist()
    manifest = {
        "version": 1,
        "contract_id": contract.get("contract_id"),
        "contract_sha256": _sha256_file(contract_source),
        "source_measurements_sha256": _sha256_file(source_path),
        "specialist_oof_sha256": _sha256_file(specialist_path),
        "output_sha256": _sha256_file(destination),
        "rows": len(output),
        "patients": int(output["patient_id"].nunique()),
        "semantic_feature_count": len(specs),
        "missing_indicator_count": len(specs),
        "record_id_hash": hashlib.sha256(
            json.dumps(
                record_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "missing_counts": missing_counts,
        "all_specialist_predictions_out_of_fold": True,
        "forbidden_inputs_present": sorted(
            set(contract.get("fit_policy", {}).get("forbidden_inputs", []))
            .intersection(output.columns)
        )
        if isinstance(contract.get("fit_policy"), Mapping)
        else [],
    }
    if manifest["forbidden_inputs_present"]:
        raise ValueError("Forbidden inputs reached the semantic target matrix")
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run(config: ProjectConfig, args: object) -> int:
    root = config.paths.root

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    manifest = build_oof_semantic_target_matrix(
        source_measurements_path=resolve(str(getattr(args, "source_measurements"))),
        specialist_oof_path=resolve(str(getattr(args, "specialist_oof"))),
        contract_path=resolve(str(getattr(args, "contract"))),
        output_path=resolve(str(getattr(args, "output"))),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
