"""Signal preprocessing policies and optional scipy-backed filters."""

from __future__ import annotations


def branch_parameters(filter_config: dict[str, object]) -> dict[str, object]:
    return {
        "highpass_hz": float(filter_config["highpass_hz"]),
        "lowpass_hz": float(filter_config["lowpass_hz"]),
        "line_frequency_hz": float(filter_config["line_frequency_hz"]),
        "apply_notch_if_present": bool(filter_config["apply_notch_if_present"]),
    }


def preprocess_signal(signal, sampling_rate_hz: float, filter_config: dict[str, object]):  # type: ignore[no-untyped-def]
    try:
        import numpy as np  # type: ignore
        from scipy import signal as sp_signal  # type: ignore
    except ImportError as exc:
        raise RuntimeError("preprocess_signal requires numpy and scipy") from exc

    x = np.asarray(signal, dtype=float)
    params = branch_parameters(filter_config)
    nyquist = sampling_rate_hz / 2.0
    b_hp, a_hp = sp_signal.butter(2, params["highpass_hz"] / nyquist, btype="highpass")
    b_lp, a_lp = sp_signal.butter(4, params["lowpass_hz"] / nyquist, btype="lowpass")
    filtered = sp_signal.filtfilt(b_hp, a_hp, x, axis=0)
    filtered = sp_signal.filtfilt(b_lp, a_lp, filtered, axis=0)
    return filtered
