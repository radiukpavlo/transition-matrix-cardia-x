"""Project configuration loading and directory management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from tm_ecg.types import ProjectPaths


@dataclass(slots=True)
class DatasetRuntimeConfig:
    name: str
    version: str
    archive: str
    extract_dir: str
    metadata_csv: str | None = None
    train_folds: tuple[int, ...] = ()
    val_folds: tuple[int, ...] = ()
    test_folds: tuple[int, ...] = ()
    repeats: int | None = None
    folds: int | None = None


@dataclass(slots=True)
class ProjectConfig:
    name: str
    version: str
    ontology_version: str
    seed: int
    paths: ProjectPaths
    datasets: dict[str, DatasetRuntimeConfig]
    splits: dict[str, object]
    filters: dict[str, dict[str, object]]
    thresholds: dict[str, object]
    latents: dict[str, object]
    transition: dict[str, object]
    training: dict[str, object]
    reporting: dict[str, object]
    dss: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ProjectConfig":
        path = Path(config_path or "configs/defaults.toml")
        with path.open("rb") as handle:
            payload = tomllib.load(handle)

        root = Path(payload["paths"]["root"]).resolve()
        paths = ProjectPaths(
            root=root,
            data_lock=(root / payload["paths"]["data_lock"]).resolve(),
            raw=(root / payload["paths"]["raw"]).resolve(),
            interim=(root / payload["paths"]["interim"]).resolve(),
            features=(root / payload["paths"]["features"]).resolve(),
            latents=(root / payload["paths"]["latents"]).resolve(),
            models=(root / payload["paths"]["models"]).resolve(),
            transition=(root / payload["paths"]["transition"]).resolve(),
            reports=(root / payload["paths"]["reports"]).resolve(),
            manifests=(root / payload["paths"]["manifests"]).resolve(),
            logs=(root / payload["paths"]["logs"]).resolve(),
        )

        datasets = {}
        for key, value in payload["datasets"].items():
            datasets[key] = DatasetRuntimeConfig(
                name=value["name"],
                version=value["version"],
                archive=value["archive"],
                extract_dir=value["extract_dir"],
                metadata_csv=value.get("metadata_csv"),
                train_folds=tuple(value.get("train_folds", [])),
                val_folds=tuple(value.get("val_folds", [])),
                test_folds=tuple(value.get("test_folds", [])),
                repeats=value.get("repeats"),
                folds=value.get("folds"),
            )

        project = payload["project"]
        return cls(
            name=project["name"],
            version=project["version"],
            ontology_version=project["ontology_version"],
            seed=int(project["seed"]),
            paths=paths,
            datasets=datasets,
            splits=payload["splits"],
            filters=payload["filters"],
            thresholds=payload["thresholds"],
            latents=payload["latents"],
            transition=payload["transition"],
            training=payload["training"],
            reporting=payload["reporting"],
            dss=payload.get("dss", {}),
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.paths.data_lock,
            self.paths.raw,
            self.paths.interim,
            self.paths.features,
            self.paths.latents,
            self.paths.models,
            self.paths.transition,
            self.paths.reports,
            self.paths.manifests,
            self.paths.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)
