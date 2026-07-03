"""Multi-scale geometric local z-score (TextSeal localized_detect pattern).

Adapted from vendor/textseal/textseal/watermarking/detector.py
_geometric_cover_search: dyadic window lengths, stride L/2, max window z
assigned to all tokens in the winning window span.
"""

from __future__ import annotations

import numpy as np


def token_multiscale_z(signal: np.ndarray, min_length: int = 24,
                       base_var: float = 1.0) -> np.ndarray:
    """Per-token max localized z from geometric cover windows (O(n log n)).

    Matches TextSeal _geometric_cover_search: dyadic lengths, stride L/2,
    each window boosts all tokens it covers with its window z-score."""
    n = len(signal)
    if n == 0:
        return signal.astype(np.float64)
    out = np.zeros(n, dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(signal)])
    denom = np.sqrt(max(base_var, 1e-9))
    max_power = int(np.floor(np.log2(n)))
    for power in range(max_power + 1):
        L = 2 ** power
        if L < min_length or L > n:
            continue
        stride = max(1, L // 2)
        for start in range(0, n - L + 1, stride):
            end = start + L
            z = (prefix[end] - prefix[start]) / (np.sqrt(L) * denom)
            if z > 0:
                out[start:end] = np.maximum(out[start:end], z)
    return out


def boundary_smooth(scores: np.ndarray, window: int = 20,
                    threshold: float = 0.0) -> np.ndarray:
    """Moving-average boost (TextSeal _boundary_smoother), continuous variant."""
    n = len(scores)
    if n == 0 or window <= 1:
        return np.zeros(n, dtype=np.float64)
    half = window // 2
    boost = np.zeros(n, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(scores)])
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        m = (cum[hi] - cum[lo]) / (hi - lo)
        boost[i] = max(0.0, m - threshold)
    return boost
