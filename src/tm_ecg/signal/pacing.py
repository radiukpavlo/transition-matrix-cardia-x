"""Pacing spike detection and interpolation-based removal."""

from __future__ import annotations


def detect_pacing_spikes(signal, threshold_scale: float = 8.0):  # type: ignore[no-untyped-def]
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("detect_pacing_spikes requires numpy") from exc

    x = np.asarray(signal, dtype=float)
    diff = np.abs(np.diff(x, axis=0))
    threshold = diff.mean() + threshold_scale * diff.std()
    spike_idx = np.argwhere(diff > threshold)
    return spike_idx.tolist()


def remove_pacing_spikes(signal, spike_idx):  # type: ignore[no-untyped-def]
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("remove_pacing_spikes requires numpy") from exc

    x = np.asarray(signal, dtype=float).copy()
    for sample_idx, lead_idx in spike_idx:
        left = max(sample_idx - 1, 0)
        right = min(sample_idx + 1, x.shape[0] - 1)
        x[sample_idx, lead_idx] = (x[left, lead_idx] + x[right, lead_idx]) / 2.0
    return x


def detect_multilead_pacing_spikes(
    signal,
    *,
    sampling_rate_hz: float,
    local_window_ms: float = 200.0,
    threshold_scale: float = 10.0,
    concurrence_ms: float = 4.0,
):  # type: ignore[no-untyped-def]
    """Adaptive narrow-spike detector with cross-lead concurrence."""

    import numpy as np  # type: ignore

    matrix = np.asarray(signal, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 8:
        raise ValueError("Pacing detector expects samples-by-leads ECG data")
    difference = np.abs(np.diff(matrix, axis=0))
    window = max(int(local_window_ms * sampling_rate_hz / 1000.0), 5)
    candidates: list[tuple[int, int, float]] = []
    for lead in range(matrix.shape[1]):
        values = difference[:, lead]
        for sample in range(len(values)):
            lower = max(0, sample - window // 2)
            upper = min(len(values), sample + window // 2 + 1)
            local = values[lower:upper]
            median = float(np.nanmedian(local))
            mad = float(np.nanmedian(np.abs(local - median))) + 1e-9
            threshold = median + threshold_scale * 1.4826 * mad
            if values[sample] > threshold and values[sample] > 1e-6:
                candidates.append((sample, lead, float(values[sample] / threshold)))
    tolerance = max(int(concurrence_ms * sampling_rate_hz / 1000.0), 1)
    events: list[dict[str, object]] = []
    for sample, lead, score in sorted(candidates):
        matching = next(
            (
                event
                for event in reversed(events)
                if sample - int(event["sample"]) <= tolerance
            ),
            None,
        )
        if matching is None:
            events.append(
                {
                    "sample": sample,
                    "leads": [lead],
                    "lead_scores": [score],
                }
            )
        else:
            leads = matching["leads"]
            scores = matching["lead_scores"]
            if isinstance(leads, list) and lead not in leads:
                leads.append(lead)
            if isinstance(scores, list):
                scores.append(score)
    for event in events:
        leads = event["leads"]
        scores = event["lead_scores"]
        event["lead_concurrence"] = (
            len(leads) / matrix.shape[1] if isinstance(leads, list) else 0.0
        )
        event["median_adaptive_score"] = (
            float(np.median(scores)) if isinstance(scores, list) and scores else 0.0
        )
    return events


def pacing_spike_features(
    signal,
    *,
    sampling_rate_hz: float,
    rpeaks: object | None = None,
):  # type: ignore[no-untyped-def]
    """Aggregate spike concurrence, timing, and noise-suppression features."""

    import numpy as np  # type: ignore

    matrix = np.asarray(signal, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    events = detect_multilead_pacing_spikes(
        matrix,
        sampling_rate_hz=sampling_rate_hz,
    )
    peaks = np.asarray(rpeaks if rpeaks is not None else (), dtype=int)
    latencies_ms: list[float] = []
    pre_qrs = 0
    for event in events:
        sample = int(event["sample"])
        following = peaks[peaks >= sample]
        if len(following):
            latency = float((following[0] - sample) * 1000.0 / sampling_rate_hz)
            if 0.0 <= latency <= 80.0:
                latencies_ms.append(latency)
                pre_qrs += 1
    difference = np.diff(matrix, axis=0)
    high_frequency_noise = float(
        np.nanmedian(np.abs(difference))
        / (np.nanpercentile(np.abs(matrix), 95) + 1e-9)
    )
    concurrence = [
        float(event["lead_concurrence"]) for event in events
    ]
    scores = [float(event["median_adaptive_score"]) for event in events]
    return {
        "spike_event_count": len(events),
        "multi_lead_spike_count": sum(value >= 0.25 for value in concurrence),
        "median_lead_concurrence": (
            float(np.median(concurrence)) if concurrence else 0.0
        ),
        "median_adaptive_spike_score": (
            float(np.median(scores)) if scores else 0.0
        ),
        "pre_qrs_spike_fraction": pre_qrs / len(events) if events else 0.0,
        "median_spike_to_qrs_latency_ms": (
            float(np.median(latencies_ms)) if latencies_ms else None
        ),
        "spike_latency_iqr_ms": (
            float(np.quantile(latencies_ms, 0.75) - np.quantile(latencies_ms, 0.25))
            if len(latencies_ms) >= 2
            else None
        ),
        "high_frequency_noise_suppressor": high_frequency_noise,
    }
