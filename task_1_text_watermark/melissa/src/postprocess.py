"""Post-processing of per-token scores (span logic).

Watermarked regions are contiguous, so we can:
  1. smooth scores over neighbouring tokens (moving average / Gaussian),
  2. penalise isolated high spikes surrounded by clean tokens,
  3. optionally boost tokens inside a high-density local window (TextSeal region idea).

All parameters are tuned on validation only. Smoothing preserves the [0, 1] range.
"""

from __future__ import annotations

import math
from typing import Sequence


def _gaussian_kernel(radius: int, sigma: float) -> list[float]:
    ks = [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-radius, radius + 1)]
    s = sum(ks)
    return [k / s for k in ks]


def smooth(scores: Sequence[float], radius: int = 3, sigma: float = 1.5) -> list[float]:
    """Gaussian smoothing over neighbouring tokens (regions are contiguous)."""
    n = len(scores)
    if n == 0 or radius <= 0:
        return list(scores)
    kernel = _gaussian_kernel(radius, sigma)
    out = [0.0] * n
    for i in range(n):
        acc = 0.0
        wsum = 0.0
        for j, w in enumerate(kernel):
            idx = i + (j - radius)
            if 0 <= idx < n:
                acc += w * scores[idx]
                wsum += w
        out[i] = acc / wsum if wsum > 0 else scores[i]
    return out


def penalize_isolated(scores: Sequence[float], radius: int = 2,
                      thr: float = 0.5, factor: float = 0.5) -> list[float]:
    """Damp a high token whose neighbourhood mean is low (likely a chance spike)."""
    n = len(scores)
    out = list(scores)
    for i in range(n):
        if scores[i] < thr:
            continue
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        neigh = [scores[j] for j in range(lo, hi) if j != i]
        if neigh and (sum(neigh) / len(neigh)) < thr:
            out[i] = scores[i] * factor
    return out


def postprocess(scores: Sequence[float], smooth_radius: int = 3, smooth_sigma: float = 1.5,
                do_isolated: bool = True, iso_radius: int = 2, iso_thr: float = 0.5,
                iso_factor: float = 0.5) -> list[float]:
    out = smooth(scores, smooth_radius, smooth_sigma)
    if do_isolated:
        out = penalize_isolated(out, iso_radius, iso_thr, iso_factor)
    return [min(1.0, max(0.0, v)) for v in out]
