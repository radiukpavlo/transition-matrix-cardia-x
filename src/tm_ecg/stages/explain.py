"""Apply the trained transition operator to validation or test latents."""

from __future__ import annotations

from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import read_json, sha256_file
from tm_ecg.io.readers import dataset_a_path, find_table, read_table_rows
from tm_ecg.io.tabular import write_records_table
from tm_ecg.stages.shared import write_stage_manifest
from tm_ecg.transition.a_preprocess import apply_a_preprocess_bundle, read_a_preprocess_bundle
from tm_ecg.transition.ridge import apply_transition, load_operator_package
from tm_ecg.transition.typed_transforms import inverse_rows
from tm_ecg.types import TransformBundle, TransformColumnStats




def _load_bundle(path: Path) -> TransformBundle:
    payload = read_json(path)
    return TransformBundle(
        dataset=str(payload["dataset"]),
        fit_columns=list(payload["fit_columns"]),
        dropped_columns=list(payload.get("dropped_columns", [])),
        stats=[TransformColumnStats(**item) for item in payload["stats"]],
    )


def _validate_explanation_inputs(
    config: ProjectConfig,
    dataset: str,
    split: str,
    a_path: Path,
    bundle_path: Path,
    operator_path: Path,
) -> dict[str, object]:
    metadata_path = config.paths.transition / f"{dataset.upper()}_operator_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Transition explanation lacks strict provenance metadata")
    metadata = read_json(metadata_path)
    if metadata.get("artifact_version") != 2:
        raise RuntimeError("Transition explanation metadata contract is stale")
    if metadata.get("ontology_version") != config.ontology_version:
        raise RuntimeError("Transition explanation ontology mismatch")
    operator_hash_key = (
        "operator_sha256" if operator_path.suffix == ".npz" else "legacy_operator_sha256"
    )
    if metadata.get(operator_hash_key) != sha256_file(operator_path):
        raise RuntimeError("Transition explanation operator hash mismatch")
    if metadata.get("transform_bundle_sha256") != sha256_file(bundle_path):
        raise RuntimeError("Transition explanation bundle hash mismatch")
    reduced_evidence = dict(metadata.get("a_red_output_artifacts", {})).get(split)
    if not isinstance(reduced_evidence, dict) or reduced_evidence.get(
        "sha256"
    ) != sha256_file(a_path):
        raise RuntimeError("Transition explanation latent hash mismatch")
    return metadata


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    split = getattr(args, "split")
    a_path = find_table(config.paths.latents, f"A_{dataset}_{split}_red")
    bundle_path = config.paths.transition / f"{dataset.upper()}_transform_bundle.json"
    operator_path = config.paths.transition / f"{dataset.upper()}_T_ridge.npz"
    if not operator_path.exists():
        operator_path = config.paths.transition / f"{dataset.upper()}_T_ridge.json"
    if a_path is None:
        raw_a_path = dataset_a_path(config, dataset, split)
        a_bundle_path = config.paths.transition / f"{dataset.upper()}_A_preprocess_bundle.json"
        if raw_a_path is not None and a_bundle_path.exists():
            raw_a_rows = read_table_rows(raw_a_path)
            a_rows_for_write = apply_a_preprocess_bundle(raw_a_rows, read_a_preprocess_bundle(a_bundle_path))
            a_path = write_records_table(config.paths.latents / f"A_{dataset}_{split}_red.parquet", a_rows_for_write)
    if a_path is None or not bundle_path.exists() or not operator_path.exists():
        write_stage_manifest(
            config,
            f"explain_{dataset}_{split}",
            {"dataset": dataset, "split": split, "status": "waiting_for_dependencies"},
        )
        print(f"Missing artifacts for explain {dataset} {split}")
        return 0

    metadata = _validate_explanation_inputs(
        config,
        dataset,
        split,
        a_path,
        bundle_path,
        operator_path,
    )

    a_rows = read_table_rows(a_path)
    a_columns = [column for column in a_rows[0].keys() if column not in {"record_id", "split"}]
    matrix = [[float(row[column]) for column in a_columns] for row in a_rows]
    operator_payload = load_operator_package(operator_path)
    predicted_fit = apply_transition(matrix, operator_payload["operator"])
    bundle = _load_bundle(bundle_path)
    fit_rows = []
    for row, predicted in zip(a_rows, predicted_fit, strict=False):
        fit_row = {"record_id": row["record_id"]}
        for column, value in zip(bundle.fit_columns, predicted, strict=False):
            fit_row[column] = value
        fit_rows.append(fit_row)
    raw_rows = inverse_rows(fit_rows, bundle)

    fit_output = write_records_table(config.paths.transition / f"{dataset.upper()}_B_hat_fit_{split}.parquet", fit_rows)
    raw_output = write_records_table(config.paths.transition / f"{dataset.upper()}_B_hat_raw_{split}.parquet", raw_rows)
    write_stage_manifest(
        config,
        f"explain_{dataset}_{split}",
        {
            "dataset": dataset,
            "split": split,
            "status": "complete",
            "fit_output": str(fit_output),
            "fit_output_sha256": sha256_file(fit_output),
            "raw_output": str(raw_output),
            "raw_output_sha256": sha256_file(raw_output),
            "operator_metadata": str(
                config.paths.transition / f"{dataset.upper()}_operator_metadata.json"
            ),
            "operator_metadata_sha256": sha256_file(
                config.paths.transition / f"{dataset.upper()}_operator_metadata.json"
            ),
            "operator_sha256": metadata[
                "operator_sha256"
                if operator_path.suffix == ".npz"
                else "legacy_operator_sha256"
            ],
            "latent_sha256": sha256_file(a_path),
            "ontology_version": config.ontology_version,
        },
    )
    print(f"Predicted feature outputs written for {dataset} {split}")
    return 0
