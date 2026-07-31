from __future__ import annotations

from pathlib import Path

from tm_ecg.ontology import map_ludb_axes, map_ludb_text
from tm_ecg.stages.index import _parse_ludb_header


def test_ludb_section_diagnoses_are_preserved(tmp_path: Path) -> None:
    header = tmp_path / "1.hea"
    header.write_text(
        "1 12 500 5000\n"
        "#<age>: 51\n"
        "#<sex>: F\n"
        "#<diagnoses>:\n"
        "#Rhythm: Sinus rhythm.\n"
        "#Complete right bundle branch block.\n"
        "#Non-specific repolarization abnormalities: anterior wall.\n",
        encoding="utf-8",
    )

    parsed = _parse_ludb_header(header)

    assert parsed["age"] == "51"
    assert parsed["sex"] == "F"
    assert parsed["diagnoses"].split(" | ") == [
        "Rhythm: Sinus rhythm.",
        "Complete right bundle branch block.",
        "Non-specific repolarization abnormalities: anterior wall.",
    ]
    axes = map_ludb_axes(parsed["diagnoses"])
    assert axes.rhythm == ("sinus",)
    assert axes.conduction == ("rbbb_spectrum",)
    assert axes.repolarization == ("other_st_t",)
    assert map_ludb_text(parsed["diagnoses"]) == ["RBBB spectrum"]


def test_wandering_atrial_pacemaker_is_not_device_pacing() -> None:
    axes = map_ludb_axes("Rhythm: Wandering atrial pacemaker.")
    assert axes.rhythm == ("other_rhythm",)
    assert axes.pacing == "absent"


def test_ludb_pacing_and_ectopy_are_mapped() -> None:
    axes = map_ludb_axes(
        "BIpolar ventricular pacing. | Ventricular extrasystole, type: single PVC."
    )
    assert axes.pacing == "present"
    assert axes.ectopy == ("pvc",)
    assert map_ludb_text(
        "BIpolar ventricular pacing. | Ventricular extrasystole, type: single PVC."
    ) == ["PVC", "Paced"]


def test_known_and_unknown_ludb_clauses_preserve_residual_abnormality() -> None:
    axes = map_ludb_axes("Rhythm: Atrial fibrillation. | Left ventricular hypertrophy.")
    assert axes.rhythm == ("af",)
    assert axes.unsupported_source_labels == ("Left ventricular hypertrophy",)
    assert map_ludb_text(
        "Rhythm: Atrial fibrillation. | Left ventricular hypertrophy."
    ) == ["AF"]
