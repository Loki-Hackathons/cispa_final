"""Unigram (fixed green-list) watermark features, per token.

The Unigram watermark (Zhao et al., "Provable Robust Watermarking") seeds a **single,
context-independent** green list over the whole vocabulary from the secret key, then
boosts green tokens at every step. Detection features per token:

- ``green`` : 1 if the token is in the green list, else 0
- ``local_green_frac`` : green fraction in a local window (contiguity signal)

The membership vector is deterministic given (key, vocab_size, gamma).
"""

from __future__ import annotations

from typing import Sequence

from ..config import WatermarkConfig
from .prf import prf_uniform


def _unigram_key(cfg: WatermarkConfig) -> int:
    keys = cfg.keys or {}
    for name in ("unigram", "uni_gram"):
        if name in keys:
            return int(keys[name])
    return 0


def _green_mask(cfg: WatermarkConfig, key: int) -> list[bool]:
    """Deterministic green-list membership over the whole vocab (context-free)."""
    gamma = float(cfg.gamma)
    vocab = int(cfg.vocab_size)
    # A fixed empty context makes the PRF depend only on (token, key) → unigram behaviour.
    return [prf_uniform(v, (), key) < gamma for v in range(vocab)]


def unigram_features(token_ids: Sequence[int], cfg: WatermarkConfig,
                     window: int = 25) -> tuple[list[float], list[float]]:
    """Return (green membership 0/1, local green fraction) per token."""
    key = _unigram_key(cfg)
    mask = _green_mask(cfg, key)
    vocab = len(mask)
    green = [1.0 if (0 <= int(t) < vocab and mask[int(t)]) else 0.0 for t in token_ids]

    n = len(green)
    frac = [0.0] * n
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        seg = green[lo:hi]
        frac[i] = sum(seg) / max(1, len(seg))
    return green, frac
