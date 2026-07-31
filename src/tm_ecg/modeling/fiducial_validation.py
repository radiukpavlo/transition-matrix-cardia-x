"""External LUDB validation for the adaptive ECG fiducial extractor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from tm_ecg.config import ProjectConfig
from tm_ecg.features.beat_extraction import _lead_quality
from tm_ecg.io.common import sha256_file, write_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.io.wfdb_loader import _lead_index
from tm_ecg.signal.fiducials import (
    DELINEATION_METHOD,
    SECONDARY_R_DETECTOR,
    accept_beat,
    delineate_beat,
    detect_secondary_r_peaks,
    match_r_peaks,
)
from tm_ecg.signal.filtering import preprocess_signal
from tm_ecg.signal.rpeaks import detect_r_peaks
from tm_ecg.stages.shared import dataset_root, find_single_file


_QRS_SYMBOLS = frozenset("NLRaVFAJSEj/QenfrB")
_LANDMARKS = (
    "p_on",
    "p_peak",
    "p_off",
    "qrs_on",
    "r_peak",
    "qrs_off",
    "t_on",
    "t_peak",
    "t_off",
)


@dataclass(frozen=True, slots=True)
class ReferenceWave:
    kind: str
    onset: int
    peak: int
    offset: int


def parse_ludb_annotation_waves(
    samples: Sequence[int],
    symbols: Sequence[str],
) -> list[ReferenceWave]:
    """Parse LUDB ``( wave )`` triplets without treating U waves as QRS."""

    if len(samples) != len(symbols):
        raise ValueError("Annotation samples and symbols must be aligned")
    waves: list[ReferenceWave] = []
    index = 0
    while index < len(symbols):
        if symbols[index] != "(":
            index += 1
            continue
        close = index + 1
        while close < len(symbols) and symbols[close] != ")":
            close += 1
        if close >= len(symbols):
            break
        centres = [
            position
            for position in range(index + 1, close)
            if symbols[position] not in {"(", ")"}
        ]
        if len(centres) == 1:
            centre = centres[0]
            symbol = symbols[centre]
            normalized = symbol.lower()
            kind = (
                "p"
                if normalized == "p"
                else "t"
                if normalized == "t"
                else "qrs"
                if symbol in _QRS_SYMBOLS
                else ""
            )
            if kind:
                waves.append(
                    ReferenceWave(
                        kind=kind,
                        onset=int(samples[index]),
                        peak=int(samples[centre]),
                        offset=int(samples[close]),
                    )
                )
        index = close + 1
    return waves


def _associated_wave(
    waves: Sequence[ReferenceWave],
    qrs_peak: int,
    sampling_rate_hz: float,
    kind: str,
) -> ReferenceWave | None:
    if kind == "p":
        candidates = [
            wave
            for wave in waves
            if wave.kind == kind
            and 0.04 <= (qrs_peak - wave.peak) / sampling_rate_hz <= 0.45
        ]
        return max(candidates, key=lambda wave: wave.peak, default=None)
    candidates = [
        wave
        for wave in waves
        if wave.kind == kind
        and 0.04 <= (wave.peak - qrs_peak) / sampling_rate_hz <= 0.65
    ]
    return min(candidates, key=lambda wave: wave.peak, default=None)


def _bootstrap_detection_ci(
    records: Sequence[dict[str, object]],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    import numpy as np  # type: ignore

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(len(records), size=len(records), replace=True)
        tp = sum(int(records[index]["r_tp"]) for index in sampled)
        fp = sum(int(records[index]["r_fp"]) for index in sampled)
        fn = sum(int(records[index]["r_fn"]) for index in sampled)
        values.append(2.0 * tp / (2.0 * tp + fp + fn) if tp or fp or fn else 0.0)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _landmark_summary(
    rows: Sequence[dict[str, object]],
    landmark: str,
    *,
    accepted_only: bool,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    scope = [row for row in rows if not accepted_only or bool(row["pipeline_accepted"])]
    reference_key = f"reference_{landmark}_sample"
    prediction_key = f"predicted_{landmark}_sample"
    error_key = f"error_{landmark}_ms"
    reference_rows = [row for row in scope if row.get(reference_key) is not None]
    paired = [row for row in reference_rows if row.get(prediction_key) is not None]
    errors = np.asarray([float(row[error_key]) for row in paired], dtype=float)
    if errors.size == 0:
        return {
            "reference_opportunities": len(reference_rows),
            "paired_predictions": 0,
            "availability": 0.0,
            "status": "not_estimable",
        }
    tolerance_ms = 40.0 if landmark.startswith("qrs") or landmark == "r_peak" else 80.0
    by_record: dict[str, list[float]] = {}
    for row in paired:
        by_record.setdefault(str(row["record_id"]), []).append(abs(float(row[error_key])))
    record_ids = sorted(by_record)
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(record_ids, size=len(record_ids), replace=True)
        values = [value for record_id in sampled for value in by_record[str(record_id)]]
        bootstrap.append(float(np.median(values)))
    ci = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "status": "ok",
        "reference_opportunities": len(reference_rows),
        "paired_predictions": len(paired),
        "availability": len(paired) / len(reference_rows) if reference_rows else 0.0,
        "signed_error_mean_ms": float(errors.mean()),
        "absolute_error_median_ms": float(np.median(np.abs(errors))),
        "absolute_error_mean_ms": float(np.mean(np.abs(errors))),
        "absolute_error_p95_ms": float(np.quantile(np.abs(errors), 0.95)),
        "within_tolerance_ms": tolerance_ms,
        "within_tolerance_fraction": float(np.mean(np.abs(errors) <= tolerance_ms)),
        "median_absolute_error_ci_95_ms": [float(ci[0]), float(ci[1])],
        "confidence_interval_method": (
            f"record_cluster_bootstrap_percentile_{replicates}"
        ),
    }


def _record_validation(
    record_name: str,
    base_directory: Path,
    config: ProjectConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np  # type: ignore
    import wfdb  # type: ignore

    record_path = base_directory / record_name
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "ii")
    waves = parse_ludb_annotation_waves(annotation.sample, annotation.symbol)
    reference_qrs = [wave for wave in waves if wave.kind == "qrs"]
    signal = np.asarray(record.p_signal, dtype=np.float32)
    sampling_rate_hz = float(record.fs)
    signal_names = [str(value) for value in record.sig_name]
    lead_ii = _lead_index(signal_names, "II")
    diagnostic = preprocess_signal(
        signal,
        sampling_rate_hz,
        config.filters["diagnostic"],
    )
    primary, _metadata = detect_r_peaks(diagnostic[:, lead_ii], sampling_rate_hz)
    secondary = detect_secondary_r_peaks(diagnostic[:, lead_ii], sampling_rate_hz)
    lower = int(0.25 * sampling_rate_hz)
    upper = signal.shape[0] - int(0.45 * sampling_rate_hz)
    primary = sorted({int(value) for value in primary if lower <= int(value) < upper})
    secondary = sorted(
        {int(value) for value in secondary if lower <= int(value) < upper}
    )
    detector_tolerance = max(
        1,
        int(
            round(
                float(config.thresholds.get("r_peak_match_tolerance_ms", 60.0))
                * sampling_rate_hz
                / 1000.0
            )
        ),
    )
    detector_agreement = match_r_peaks(primary, secondary, detector_tolerance)
    consensus = [pair[0] for pair in detector_agreement.matches]
    reference_tolerance = max(1, int(round(0.10 * sampling_rate_hz)))
    reference_match = match_r_peaks(
        [wave.peak for wave in reference_qrs],
        consensus,
        reference_tolerance,
    )
    quality = _lead_quality(diagnostic[:, lead_ii], consensus, np, sampling_rate_hz)
    qrs_by_peak = {wave.peak: wave for wave in reference_qrs}
    consensus_positions = {peak: index for index, peak in enumerate(consensus)}
    landmark_rows: list[dict[str, object]] = []
    for reference_peak, predicted_peak in reference_match.matches:
        position = consensus_positions[predicted_peak]
        delineation = delineate_beat(
            diagnostic[:, lead_ii],
            predicted_peak,
            sampling_rate_hz,
            f"{record_name}-validation-{position:04d}",
            record_name,
            previous_r_peak=consensus[position - 1] if position > 0 else None,
            next_r_peak=(
                consensus[position + 1] if position + 1 < len(consensus) else None
            ),
        )
        fiducials = delineation.fiducials
        accepted = accept_beat(
            fiducials,
            lead_quality_db=quality,
            delineation_confidence=fiducials.confidence,
            pacing_contaminated=False,
            minimum_lead_quality_db=float(
                config.thresholds.get("feature_quality_min_db", 5.0)
            ),
            minimum_delineation_confidence=float(
                config.thresholds.get("delineation_confidence_minimum", 0.5)
            ),
            r_detector_matched=True,
        )
        qrs = qrs_by_peak[reference_peak]
        p_wave = _associated_wave(waves, reference_peak, sampling_rate_hz, "p")
        t_wave = _associated_wave(waves, reference_peak, sampling_rate_hz, "t")
        references = {
            "p_on": p_wave.onset if p_wave else None,
            "p_peak": p_wave.peak if p_wave else None,
            "p_off": p_wave.offset if p_wave else None,
            "qrs_on": qrs.onset,
            "r_peak": qrs.peak,
            "qrs_off": qrs.offset,
            "t_on": t_wave.onset if t_wave else None,
            "t_peak": t_wave.peak if t_wave else None,
            "t_off": t_wave.offset if t_wave else None,
        }
        row: dict[str, object] = {
            "record_id": record_name,
            "reference_qrs_peak_sample": reference_peak,
            "predicted_r_peak_sample": predicted_peak,
            "pipeline_accepted": accepted.accepted,
            "acceptance_reasons": "|".join(accepted.reasons),
            "lead_ii_quality_db": quality,
            "delineation_confidence": fiducials.confidence,
            "detector_agreement_f1": detector_agreement.score,
        }
        for landmark in _LANDMARKS:
            reference_value = references[landmark]
            predicted_value = getattr(fiducials, landmark)
            row[f"reference_{landmark}_sample"] = reference_value
            row[f"predicted_{landmark}_sample"] = predicted_value
            row[f"error_{landmark}_ms"] = (
                1000.0
                * (float(predicted_value) - float(reference_value))
                / sampling_rate_hz
                if reference_value is not None and predicted_value is not None
                else None
            )
        landmark_rows.append(row)
    record_metrics: dict[str, object] = {
        "record_id": record_name,
        "reference_qrs_count": len(reference_qrs),
        "consensus_r_count": len(consensus),
        "r_tp": len(reference_match.matches),
        "r_fp": len(consensus) - len(reference_match.matches),
        "r_fn": len(reference_qrs) - len(reference_match.matches),
        "pipeline_accepted_matched_beats": sum(
            bool(row["pipeline_accepted"]) for row in landmark_rows
        ),
        "detector_agreement_f1": detector_agreement.score,
        "lead_ii_quality_db": quality,
    }
    return record_metrics, landmark_rows


def run(config: ProjectConfig, args: argparse.Namespace) -> int:
    """Validate current fiducials against cardiologist LUDB lead-II annotations."""

    root = dataset_root(config, "ludb")
    records_file = find_single_file(root, "RECORDS")
    base_directory = records_file.parent
    record_names = [
        value.strip()
        for value in records_file.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    per_record: list[dict[str, object]] = []
    landmark_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, record_name in enumerate(record_names, start=1):
        try:
            record_metrics, record_landmarks = _record_validation(
                record_name,
                base_directory,
                config,
            )
            per_record.append(record_metrics)
            landmark_rows.extend(record_landmarks)
        except (FileNotFoundError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            failures.append({"record_id": record_name, "error": str(exc)})
        if index % 25 == 0 or index == len(record_names):
            print(f"Validated {index}/{len(record_names)} LUDB records", flush=True)

    replicates = int(getattr(args, "bootstrap_replicates", 1000))
    tp = sum(int(row["r_tp"]) for row in per_record)
    fp = sum(int(row["r_fp"]) for row in per_record)
    fn = sum(int(row["r_fn"]) for row in per_record)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if tp or fp or fn else 0.0
    landmark_summary = {
        scope: {
            landmark: _landmark_summary(
                landmark_rows,
                landmark,
                accepted_only=scope == "pipeline_accepted",
                seed=config.seed + position,
                replicates=replicates,
            )
            for position, landmark in enumerate(_LANDMARKS)
        }
        for scope in ("all_matched_consensus_beats", "pipeline_accepted")
    }
    reports_directory = config.paths.reports / "metrics"
    details_path = write_records_table(
        reports_directory / "ludb_fiducial_validation_details.parquet",
        landmark_rows,
    )
    record_path = write_records_table(
        reports_directory / "ludb_fiducial_validation_records.csv",
        per_record,
    )
    manifest_path = config.paths.manifests / "ludb_index.parquet"
    payload: dict[str, object] = {
        "artifact_version": 1,
        "status": "complete" if not failures else "complete_with_record_failures",
        "dataset": "LUDB 1.0.1",
        "annotation_lead": "II",
        "ontology_version": config.ontology_version,
        "fiducial_method": DELINEATION_METHOD,
        "secondary_r_detector": SECONDARY_R_DETECTOR,
        "records_expected": len(record_names),
        "records_evaluated": len(per_record),
        "record_failures": failures,
        "r_peak_reference_tolerance_ms": 100.0,
        "r_peak_detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sensitivity": sensitivity,
            "positive_predictive_value": precision,
            "f1": f1,
            "f1_ci_95": _bootstrap_detection_ci(
                per_record,
                seed=config.seed,
                replicates=replicates,
            )
            if per_record
            else None,
            "confidence_interval_method": (
                f"record_cluster_bootstrap_percentile_{replicates}"
            ),
        },
        "matched_consensus_beats": len(landmark_rows),
        "pipeline_accepted_matched_beats": sum(
            bool(row["pipeline_accepted"]) for row in landmark_rows
        ),
        "landmark_metrics": landmark_summary,
        "details_path": str(details_path),
        "details_sha256": sha256_file(details_path),
        "per_record_path": str(record_path),
        "per_record_sha256": sha256_file(record_path),
        "ludb_index_sha256": (
            sha256_file(manifest_path) if manifest_path.exists() else None
        ),
        "limitations": [
            "Validation uses the cardiologist LUDB lead-II annotations only; it does not establish performance on every lead or external device population.",
            "Selective pipeline-accepted errors must always be interpreted together with accepted-beat coverage.",
            "This is algorithm validation for research decision support, not medical-device validation.",
        ],
    }
    output_path = reports_directory / "ludb_fiducial_validation.json"
    write_json(output_path, payload)
    print(json.dumps({"report": str(output_path), "status": payload["status"]}, indent=2))
    return 0 if not failures else 8


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/defaults.toml")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ProjectConfig.load(args.config)
    config.ensure_directories()
    return run(config, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
