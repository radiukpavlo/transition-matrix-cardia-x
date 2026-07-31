"""Adaptive-window R-peak detection."""

from __future__ import annotations


def detect_r_peaks(signal, sampling_rate_hz: float, refractory_ratio: float = 0.25):  # type: ignore[no-untyped-def]
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("detect_r_peaks requires numpy") from exc

    x = np.asarray(signal, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    energy = np.abs(np.diff(x, prepend=x[0]))
    threshold = energy.mean() + 1.5 * energy.std()
    candidate_idx = np.where(energy >= threshold)[0]
    accepted = []
    rr_hat = sampling_rate_hz * 0.8
    refractory = int(rr_hat * refractory_ratio)
    for idx in candidate_idx:
        if accepted and idx - accepted[-1] < refractory:
            continue
        window = int(max(rr_hat * 0.2, sampling_rate_hz * 0.08))
        left = max(idx - window, 0)
        right = min(idx + window, len(x) - 1)
        local = left + int(np.argmax(x[left : right + 1]))
        accepted.append(local)
        if len(accepted) >= 2:
            rr_hat = accepted[-1] - accepted[-2]
            refractory = max(1, int(rr_hat * refractory_ratio))
    return accepted, {"rr_hat_samples": rr_hat, "refractory_samples": refractory}
