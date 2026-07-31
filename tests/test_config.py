"""Tests for ProjectConfig loading and directory management."""

import unittest
from pathlib import Path

from tm_ecg.config import ProjectConfig


class ConfigTests(unittest.TestCase):
    def test_config_loads_defaults(self) -> None:
        config_path = Path("configs/defaults.toml")
        if not config_path.exists():
            self.skipTest("configs/defaults.toml not found")
        config = ProjectConfig.load(config_path)
        self.assertIsInstance(config, ProjectConfig)
        self.assertIsNotNone(config.seed)

    def test_config_datasets_complete(self) -> None:
        config_path = Path("configs/defaults.toml")
        if not config_path.exists():
            self.skipTest("configs/defaults.toml not found")
        config = ProjectConfig.load(config_path)
        for key in ("ptbxl", "ptbxl_plus", "ludb"):
            with self.subTest(key=key):
                self.assertIn(key, config.datasets)

    def test_config_seed_is_deterministic(self) -> None:
        config_path = Path("configs/defaults.toml")
        if not config_path.exists():
            self.skipTest("configs/defaults.toml not found")
        config = ProjectConfig.load(config_path)
        self.assertEqual(config.seed, 17)

    def test_config_paths_have_required_fields(self) -> None:
        config_path = Path("configs/defaults.toml")
        if not config_path.exists():
            self.skipTest("configs/defaults.toml not found")
        config = ProjectConfig.load(config_path)
        self.assertIsNotNone(config.paths.features)
        self.assertIsNotNone(config.paths.latents)
        self.assertIsNotNone(config.paths.models)
        self.assertIsNotNone(config.paths.transition)
        self.assertIsNotNone(config.paths.reports)
        self.assertIsNotNone(config.paths.manifests)
        self.assertIsNotNone(config.paths.logs)


if __name__ == "__main__":
    unittest.main()
