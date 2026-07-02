"""Gumbel-Max watermark detection score, per token.

For each token we recompute ``R_t = PRF(token_id, context, key)`` from the k preceding
tokens and the secret key, and use the Gumbel detection statistic

    s_t = -ln(1 - R_t)

which is Exp(1) (mean 1) for unwatermarked text and skewed large for watermarked text
(TextSeal §2.3.1). Returned unnormalised; downstream ``features.py`` standardises it.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..config import WatermarkConfig
from .prf import prf_uniform


def _gumbel_key(cfg: WatermarkConfig) -> int:
    keys = cfg.keys or {}
    for name in ("gumbel", "gumbel_max", "gumbelmax"):
        if name in keys:
            return int(keys[name])
    return 0  # key-free fallback


def gumbel_scores(token_ids: Sequence[int], cfg: WatermarkConfig,
                  key: int | None = None) -> list[float]:
    """Per-token Gumbel score ``-ln(1 - R_t)`` (first k tokens get 0.0)."""
    if key is None:
        key = _gumbel_key(cfg)
    k = cfg.context_width
    n = len(token_ids)
    out = [0.0] * n
    for t in range(k, n):
        context = token_ids[t - k:t]
        r = prf_uniform(token_ids[t], context, key)
        r = min(r, 1.0 - 1e-12)
        out[t] = -math.log(1.0 - r)
    return out
