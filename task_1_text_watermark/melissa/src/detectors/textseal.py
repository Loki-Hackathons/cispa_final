"""TextSeal detection score, per token (dual-key early fusion).

TextSeal (§3.1–3.2) generates with two secret keys, routing to key 2 with probability
``alpha``. The detector does not know which key produced a token, so it early-fuses the
two Gumbel scores per token:

    s_i = (1 - alpha) * s_i^(1) + alpha * s_i^(2),   s_i^(j) = -ln(1 - R_i^(j))

This dominates late (Fisher/Bonferroni) fusion because of reduced null variance
(TextSeal App. C.1.1). If only one key is available we fall back to single-key Gumbel.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..config import WatermarkConfig
from .prf import prf_uniform


def _textseal_keys(cfg: WatermarkConfig) -> tuple[int, int | None]:
    keys = cfg.keys or {}
    ts = keys.get("textseal", keys.get("text_seal", {}))
    if isinstance(ts, dict):
        k1 = int(ts.get("key1", ts.get("k1", 0)))
        k2 = ts.get("key2", ts.get("k2", None))
        return k1, (int(k2) if k2 is not None else None)
    if isinstance(ts, (list, tuple)) and ts:
        k1 = int(ts[0])
        k2 = int(ts[1]) if len(ts) > 1 else None
        return k1, k2
    if isinstance(ts, int):
        return int(ts), None
    return 0, None


def textseal_scores(token_ids: Sequence[int], cfg: WatermarkConfig) -> list[float]:
    """Per-token dual-key early-fusion Gumbel score."""
    k1, k2 = _textseal_keys(cfg)
    alpha = float(cfg.textseal_alpha)
    k = cfg.context_width
    n = len(token_ids)
    out = [0.0] * n
    for t in range(k, n):
        context = token_ids[t - k:t]
        r1 = min(prf_uniform(token_ids[t], context, k1), 1.0 - 1e-12)
        s1 = -math.log(1.0 - r1)
        if k2 is None:
            out[t] = s1
        else:
            r2 = min(prf_uniform(token_ids[t], context, k2), 1.0 - 1e-12)
            s2 = -math.log(1.0 - r2)
            out[t] = (1.0 - alpha) * s1 + alpha * s2
    return out
