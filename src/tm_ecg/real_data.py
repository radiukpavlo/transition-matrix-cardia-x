"""Executable helpers for real PTB-XL and LUDB pipeline stages.
This module is a facade re-exporting from io, features, and modeling.
"""

from __future__ import annotations

from tm_ecg.features.beat_extraction import (
    _count_secondary_extrema,
    _lead_quality,
    _one_record_measurements,
    _st_offset_samples,
    _window,
    build_measurement_records,
)
from tm_ecg.io.wfdb_loader import (
    _lead_index,
    _load_record,
    _parse_labels,
    _resolved_dataset_root,
    _runtime,
    split_entries,
)
from tm_ecg.modeling.training import (
    _build_split_samples,
    _label_vector,
    _load_checkpoint,
    build_samples_for_dataset,
    representative_triad_tensor,
    save_latent_rows,
    train_ptbxl_classifier,
)

__all__ = [
    "_count_secondary_extrema",
    "_lead_quality",
    "_one_record_measurements",
    "_st_offset_samples",
    "_window",
    "build_measurement_records",
    "_lead_index",
    "_load_record",
    "_parse_labels",
    "_resolved_dataset_root",
    "_runtime",
    "split_entries",
    "_build_split_samples",
    "_label_vector",
    "_load_checkpoint",
    "build_samples_for_dataset",
    "representative_triad_tensor",
    "save_latent_rows",
    "train_ptbxl_classifier",
]
