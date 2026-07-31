"""Classifier training loop and latent row extraction."""

from __future__ import annotations

import json
from typing import Any

from tm_ecg.config import ProjectConfig

from tm_ecg.constants import AXIS_LABELS, LEADS_12, PROJECT_LABELS
from tm_ecg.io.common import write_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.modeling.classifier import build_model
from tm_ecg.modeling.calibration import fit_temperature_calibration
from tm_ecg.modeling.latents import trimmed_mean_pool
from tm_ecg.signal.filtering import preprocess_signal
from tm_ecg.signal.rpeaks import detect_r_peaks
from tm_ecg.io.wfdb_loader import _runtime, split_entries, _load_record, _parse_labels
from tm_ecg.features.beat_extraction import _window


def _label_vector(labels: list[str]) -> list[float]:
    return [1.0 if label in labels else 0.0 for label in PROJECT_LABELS]


def _axis_targets(entry: dict[str, Any]) -> dict[str, list[float]]:
    raw = entry.get("axis_targets") or entry.get("axis_targets_json") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    payload = dict(raw) if isinstance(raw, dict) else {}
    result: dict[str, list[float]] = {}
    for axis, classes in AXIS_LABELS.items():
        observed = payload.get(axis, [])
        if isinstance(observed, str):
            values = {observed}
        else:
            values = {str(value) for value in observed}
        if not values:
            if "unknown" in classes:
                values = {"unknown"}
            elif "none" in classes:
                values = {"none"}
            elif axis == "pacing":
                values = {"absent"}
        result[axis] = [1.0 if value in values else 0.0 for value in classes]
    return result


def _per_class_validation_metrics(
    logits: list[list[float]],
    targets: list[list[float]],
    calibration: dict[str, object],
    labels: list[str],
) -> dict[str, dict[str, float | int | str | None]]:
    """Score a held-out partition using validation-fitted calibration only."""

    import numpy as np  # type: ignore

    z = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=float)
    if z.ndim != 2 or y.ndim != 2 or z.shape != y.shape or z.shape[1] != len(labels):
        raise ValueError("Metric logits, targets, and labels must have aligned 2D shapes")
    temperature = float(calibration.get("temperature", 1.0) or 1.0)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(z / temperature, -40.0, 40.0)))
    threshold_rows = dict(calibration.get("class_thresholds", {}))
    result: dict[str, dict[str, float | int | str | None]] = {}
    for column, label in enumerate(labels):
        truth = y[:, column].astype(int)
        threshold_row = dict(threshold_rows.get(label, {}))
        threshold = float(threshold_row.get("threshold", 0.5))
        prediction = (probabilities[:, column] >= threshold).astype(int)
        tp = int(((prediction == 1) & (truth == 1)).sum())
        tn = int(((prediction == 0) & (truth == 0)).sum())
        fp = int(((prediction == 1) & (truth == 0)).sum())
        fn = int(((prediction == 0) & (truth == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        predicted_positive = probabilities[prediction == 1, column]
        result[label] = {
            "status": "ok" if len(truth) else "not_estimable_no_rows",
            "evaluation_partition": "held_out_test",
            "threshold_partition": "validation_only",
            "threshold": threshold,
            "support": int(truth.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "mean_confidence": (
                float(predicted_positive.mean()) if len(predicted_positive) else None
            ),
            "analyzable_coverage": 1.0 if len(truth) else 0.0,
            "abstention_rate": 0.0 if len(truth) else 1.0,
        }
    return result


def triad_tensors_for_record(signal, fs: float, sig_names: list[str], config: ProjectConfig):
    np, _torch, _wfdb, sp_signal = _runtime()
    from tm_ecg.io.wfdb_loader import _lead_index
    detection = preprocess_signal(signal, fs, config.filters["detection"])
    lead_ii = _lead_index(sig_names, "II")
    peaks, _meta = detect_r_peaks(detection[:, lead_ii], fs)
    peaks = [int(peak) for peak in peaks if int(0.25 * fs) <= int(peak) < signal.shape[0] - int(0.45 * fs)]
    if len(peaks) < 3:
        return []
    left = int(float(config.training["pre_r_seconds"]) * fs)
    right = int(float(config.training["post_r_seconds"]) * fs)
    samples_per_beat = int(config.training["samples_per_beat"])
    tensors = []
    for center in range(1, len(peaks) - 1):
        triad_peaks = peaks[center - 1 : center + 2]
        parts = []
        for peak in triad_peaks:
            beat = _window(detection, peak, left, right, np)
            part = sp_signal.resample(beat, samples_per_beat, axis=0).T.astype(np.float32)
            parts.append(part)
        tensors.append((np.concatenate(parts, axis=0), triad_peaks))
    return tensors


def representative_triad_tensor(signal, fs: float, sig_names: list[str], config: ProjectConfig):
    tensors = triad_tensors_for_record(signal, fs, sig_names, config)
    if not tensors:
        return None
    return tensors[len(tensors) // 2]


def _build_split_samples(config: ProjectConfig, dataset: str) -> dict[str, list[dict[str, Any]]]:
    grouped = split_entries(config, dataset)
    output = {"train": [], "val": [], "test": []}
    for split, entries in grouped.items():
        for entry in entries:
            signal, fs, sig_names = _load_record(config, dataset, entry)
            triads = triad_tensors_for_record(signal, fs, sig_names, config)
            if not triads:
                continue
            tensor, triad_peaks = triads[len(triads) // 2]
            output[split].append(
                {
                    "record_id": str(entry["record_id"]),
                    "tensor": tensor,
                    "target": _label_vector(_parse_labels(entry["labels"])),
                    "axis_targets": _axis_targets(entry),
                    "labels": _parse_labels(entry["labels"]),
                    "triad_peaks": triad_peaks,
                }
            )
    return output


def train_ptbxl_classifier(config: ProjectConfig) -> dict[str, str]:
    np, torch, _wfdb, _sp_signal = _runtime()
    if torch is None:
        raise RuntimeError("train_ptbxl_classifier requires torch. Install with: pip install 'tm-ecg[train]'")
    samples = _build_split_samples(config, "ptbxl")
    model = build_model(
        in_leads=len(LEADS_12),
        triad_length=3,
        samples_per_beat=int(config.training["samples_per_beat"]),
        latent_dim=int(config.latents["penultimate_dim"]),
        num_classes=len(PROJECT_LABELS),
        axis_classes={axis: len(values) for axis, values in AXIS_LABELS.items()},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training["learning_rate"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    axis_loss_fns: dict[str, Any] = {}
    for axis, classes in AXIS_LABELS.items():
        counts = np.sum(
            np.asarray([item["axis_targets"][axis] for item in samples["train"]]), axis=0
        )
        negatives = max(len(samples["train"]), 1) - counts
        positive_weight = np.divide(
            negatives,
            np.maximum(counts, 1.0),
        )
        axis_loss_fns[axis] = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, dtype=torch.float32)
        )
    batch_size = int(config.training["batch_size"])

    model.train()
    for _epoch in range(int(config.training["epochs"])):
        for start in range(0, len(samples["train"]), batch_size):
            batch = samples["train"][start : start + batch_size]
            x = torch.tensor(np.stack([item["tensor"] for item in batch]), dtype=torch.float32)
            y = torch.tensor(np.stack([item["target"] for item in batch]), dtype=torch.float32)
            optimizer.zero_grad()
            logits, axis_logits, _latent = model.forward_multiaxial(x)
            loss = loss_fn(logits, y)
            for axis in AXIS_LABELS:
                axis_target = torch.tensor(
                    np.stack([item["axis_targets"][axis] for item in batch]),
                    dtype=torch.float32,
                )
                loss = loss + axis_loss_fns[axis](axis_logits[axis], axis_target)
            loss.backward()
            optimizer.step()

    calibration: dict[str, object]
    if samples["val"]:
        compatibility_logits: list[list[float]] = []
        compatibility_targets: list[list[float]] = []
        axis_logits_rows: dict[str, list[list[float]]] = {
            axis: [] for axis in AXIS_LABELS
        }
        axis_target_rows: dict[str, list[list[float]]] = {
            axis: [] for axis in AXIS_LABELS
        }
        model.eval()
        with torch.no_grad():
            for start in range(0, len(samples["val"]), batch_size):
                batch = samples["val"][start : start + batch_size]
                x = torch.tensor(
                    np.stack([item["tensor"] for item in batch]),
                    dtype=torch.float32,
                )
                logits, axis_logits, _latent = model.forward_multiaxial(x)
                compatibility_logits.extend(logits.cpu().numpy().tolist())
                compatibility_targets.extend([item["target"] for item in batch])
                for axis in AXIS_LABELS:
                    axis_logits_rows[axis].extend(
                        axis_logits[axis].cpu().numpy().tolist()
                    )
                    axis_target_rows[axis].extend(
                        [item["axis_targets"][axis] for item in batch]
                    )
        calibration = {
            "version": 1,
            "fit_partition": "validation_only",
            "compatibility_head": fit_temperature_calibration(
                compatibility_logits, compatibility_targets, PROJECT_LABELS
            ),
            "axes": {
                axis: fit_temperature_calibration(
                    axis_logits_rows[axis], axis_target_rows[axis], labels
                )
                for axis, labels in AXIS_LABELS.items()
            },
        }
    else:
        calibration = {
            "version": 1,
            "status": "not_estimable_no_validation_rows",
            "fit_partition": "validation_only",
            "compatibility_head": {"temperature": 1.0},
            "axes": {
                axis: {"temperature": 1.0, "status": "not_estimable"}
                for axis in AXIS_LABELS
            },
        }

    per_class_metrics: dict[str, dict[str, float | int | str | None]] = {}
    if samples["test"]:
        test_logits: list[list[float]] = []
        test_targets: list[list[float]] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(samples["test"]), batch_size):
                batch = samples["test"][start : start + batch_size]
                x = torch.tensor(
                    np.stack([item["tensor"] for item in batch]),
                    dtype=torch.float32,
                )
                logits, _axis_logits, _latent = model.forward_multiaxial(x)
                test_logits.extend(logits.cpu().numpy().tolist())
                test_targets.extend([item["target"] for item in batch])
        per_class_metrics = _per_class_validation_metrics(
            test_logits,
            test_targets,
            dict(calibration.get("compatibility_head", {})),
            list(PROJECT_LABELS),
        )

    checkpoint_path = config.paths.models / "ptbxl_classifier.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": int(config.latents["penultimate_dim"]),
            "samples_per_beat": int(config.training["samples_per_beat"]),
            "multiaxial": True,
            "axis_labels": {axis: list(values) for axis, values in AXIS_LABELS.items()},
            "calibration": calibration,
        },
        checkpoint_path,
    )
    metrics_path = config.paths.reports / "metrics" / "ptbxl_training_metrics.json"
    calibration_path = config.paths.reports / "metrics" / "ptbxl_calibration.json"
    write_json(calibration_path, calibration)
    write_json(
        metrics_path,
        {
            "epochs": int(config.training["epochs"]),
            "train_records": len(samples["train"]),
            "val_records": len(samples["val"]),
            "test_records": len(samples["test"]),
            "per_class_metrics": per_class_metrics,
        },
    )
    return {
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "calibration": str(calibration_path),
    }


def _load_checkpoint(config: ProjectConfig, checkpoint_dataset: str = "ptbxl"):
    _np, torch, _wfdb, _sp_signal = _runtime()
    if torch is None:
        raise RuntimeError("Latent extraction requires torch. Install with: pip install 'tm-ecg[train]'")
    checkpoint = torch.load(config.paths.models / f"{checkpoint_dataset}_classifier.pt", map_location="cpu", weights_only=False)
    model = build_model(
        in_leads=len(LEADS_12),
        triad_length=3,
        samples_per_beat=int(config.training["samples_per_beat"]),
        latent_dim=int(config.latents["penultimate_dim"]),
        num_classes=len(PROJECT_LABELS),
        axis_classes={
            str(axis): len(values)
            for axis, values in dict(checkpoint.get("axis_labels", {})).items()
        },
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_samples_for_dataset(
    config: ProjectConfig,
    dataset: str,
    include_targets: bool = True,
    checkpoint_dataset: str = "ptbxl",
) -> dict[str, list[dict[str, Any]]]:
    np, torch, _wfdb, _sp_signal = _runtime()
    if torch is None:
        raise RuntimeError("Latent extraction requires torch. Install with: pip install 'tm-ecg[train]'")
    model = _load_checkpoint(config, checkpoint_dataset=checkpoint_dataset)
    grouped = split_entries(config, dataset)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    
    with torch.no_grad():
        for split, entries in grouped.items():
            for entry in entries:
                signal, fs, sig_names = _load_record(config, dataset, entry)
                triads = triad_tensors_for_record(signal, fs, sig_names, config)
                if not triads:
                    continue
                
                triad_latents = []
                for tensor, _peaks in triads:
                    x = torch.tensor(tensor[None, ...], dtype=torch.float32)
                    _logits, latent = model(x)
                    triad_latents.append(latent.squeeze(0).cpu().numpy().astype(np.float32).tolist())
                
                pooled = trimmed_mean_pool(triad_latents, trim_ratio=0.1)
                output[split].append(
                    {
                        "record_id": str(entry["record_id"]),
                        "split": split,
                        "latent": pooled,
                        "labels": _parse_labels(entry["labels"]) if include_targets else [],
                        "triad_count": len(triad_latents),
                    }
                )
    return output


def save_latent_rows(config: ProjectConfig, dataset: str, latent_rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    written: dict[str, str] = {}
    for split, rows in latent_rows_by_split.items():
        if not rows:
            continue
        formatted = []
        for row in rows:
            payload = {"record_id": row["record_id"], "split": split}
            for idx, value in enumerate(row["latent"]):
                payload[f"latent_{idx:04d}"] = value
            if "triad_count" in row:
                payload["triad_count"] = row["triad_count"]
            formatted.append(payload)
        dataset_path = write_records_table(config.paths.latents / f"A_{dataset}_{split}.parquet", formatted)
        written[f"{dataset}_{split}"] = str(dataset_path)
    return written
