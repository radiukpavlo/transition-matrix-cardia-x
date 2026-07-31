from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tm_ecg.io.common import sha256_file
from tm_ecg.stages.explain import _validate_explanation_inputs


def test_explanation_inputs_are_bound_to_transition_metadata(tmp_path) -> None:
    operator = tmp_path / "B1_T_ridge.npz"
    bundle = tmp_path / "B1_transform_bundle.json"
    latent = tmp_path / "A_b1_val_red.parquet"
    operator.write_bytes(b"operator")
    bundle.write_text("{}", encoding="utf-8")
    latent.write_bytes(b"latent")
    metadata = {
        "artifact_version": 2,
        "ontology_version": "test-v3",
        "operator_sha256": sha256_file(operator),
        "transform_bundle_sha256": sha256_file(bundle),
        "a_red_output_artifacts": {
            "val": {"sha256": sha256_file(latent)}
        },
    }
    (tmp_path / "B1_operator_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    config = SimpleNamespace(
        ontology_version="test-v3",
        paths=SimpleNamespace(transition=tmp_path),
    )

    observed = _validate_explanation_inputs(
        config,
        "b1",
        "val",
        latent,
        bundle,
        operator,
    )
    assert observed["artifact_version"] == 2

    latent.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="latent hash mismatch"):
        _validate_explanation_inputs(
            config,
            "b1",
            "val",
            latent,
            bundle,
            operator,
        )
