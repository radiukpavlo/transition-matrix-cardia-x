from pathlib import Path
import shutil
import unittest
import uuid

from tm_ecg.config import DatasetRuntimeConfig, ProjectConfig
from tm_ecg.stages.splits import _freeze_ptbxl
from tm_ecg.types import ProjectPaths


class SplitTests(unittest.TestCase):
    def test_ptbxl_locked_train_count(self) -> None:
        root = Path.cwd() / ".tmp_test_splits" / str(uuid.uuid4())
        manifests = root / "manifests"
        manifests.mkdir(parents=True)
        try:
            (manifests / "ptbxl_index.csv").write_text(
                "\n".join(
                    [
                        "dataset,record_id,patient_id,strat_fold,labels,filename_hr",
                        "ptbxl,1,p1,1,Normal,a",
                        "ptbxl,2,p2,1,PVC,b",
                        "ptbxl,3,p3,1,AF,c",
                        "ptbxl,4,p4,1,APB,d",
                        "ptbxl,5,p5,9,Normal,e",
                        "ptbxl,6,p6,10,Normal,f",
                    ]
                ),
                encoding="utf-8",
            )
            config = ProjectConfig(
                name="tm",
                version="0.1.0",
                ontology_version="v1",
                seed=17,
                paths=ProjectPaths(
                    root=root,
                    data_lock=root / "data" / "archives",
                    raw=root / "data" / "raw",
                    interim=root / "data" / "interim",
                    features=root / "data" / "processed" / "features",
                    latents=root / "data" / "processed" / "latents",
                    models=root / "artifacts" / "models",
                    transition=root / "artifacts" / "transition",
                    reports=root / "artifacts" / "reports",
                    manifests=manifests,
                    logs=root / "artifacts" / "logs",
                ),
                datasets={
                    "ptbxl": DatasetRuntimeConfig("ptbxl", "1.0.3", "x.zip", "ptbxl"),
                    "ludb": DatasetRuntimeConfig("ludb", "1.0.1", "y.zip", "ludb", repeats=1, folds=5),
                },
                splits={"ptbxl_train_target_rows": 4},
                filters={},
                thresholds={},
                latents={},
                transition={},
                training={},
                reporting={},
            )
            manifest = _freeze_ptbxl(config)
            train_rows = [row for row in manifest.split_assignments if row.split == "train"]
            self.assertEqual(len(train_rows), 4)
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
