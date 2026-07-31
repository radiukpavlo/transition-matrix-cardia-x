from __future__ import annotations

import json

import pytest

from tm_ecg.dss.discretization import _clinical_bins
from tm_ecg.dss.models import IntervalBin
from tm_ecg.features.signatures import fit_signature_artifact, load_signature_artifact


def _signature_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for index in range(80):
        positive = index % 4 == 0
        rows.append(
            {
                "record_id": f"r{index}",
                "labels": "LBBB spectrum" if positive else "Normal",
                "broad_r_v6_any": int(positive),
                "qrs_dur_med_ms": 145.0 if positive else 90.0,
            }
        )
    return rows[:60], rows[60:]


def test_fitted_signature_bands_are_strictly_ordered() -> None:
    train, validation = _signature_rows()
    artifact = fit_signature_artifact(train, validation, random_seed=17)

    for model in dict(artifact["signatures"]).values():
        if model["status"] != "available":
            continue
        bands = model["threshold_bands"]
        assert bands["negative_max_probability"] < bands["positive_min_probability"]
        assert bands["negative_max_logodds"] < bands["positive_min_logodds"]


def test_reversed_signature_artifact_is_rejected(tmp_path) -> None:
    train, validation = _signature_rows()
    artifact = fit_signature_artifact(train, validation, random_seed=17)
    model = artifact["signatures"]["lbbb_signature_score"]
    model["threshold_bands"]["negative_max_probability"] = 0.8
    model["threshold_bands"]["positive_min_probability"] = 0.2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="reversed probability bands"):
        load_signature_artifact(path)


def test_reversed_signature_discretization_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative < positive"):
        _clinical_bins(
            "lbbb_signature_score",
            {
                "lbbb_signature_score": {
                    "negative_max_logodds": 0.0,
                    "positive_min_logodds": -1.0,
                }
            },
        )


def test_interval_bin_rejects_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="lower bound exceeds"):
        IntervalBin(code=0, label="invalid", lower=2.0, upper=1.0)
