"""Per-token feature extraction for the watermark localization calibrator.

Stacks, for every token in a document:
  - detector signals: Gumbel, TextSeal, Unigram (green + local frac),
    KGW (green + local frac + running z), entropy;
  - key-free statistical signals (always available): piece length, whitespace/newline
    flags, local token-id novelty, local repetition;
  - local context: rolling means of the main detector scores over a small window;
  - normalised position in the document.

All detectors gracefully degrade to key-free/CPU fallbacks so the same feature matrix is
produced locally (for a baseline) and on the cluster (with real keys + CUDA KGW).
"""

from __future__ import annotations

from typing import Sequence

from .config import WatermarkConfig, load_watermark_config
from .detectors import (
    gumbel_scores,
    textseal_scores,
    unigram_features,
    kgw_features,
    entropy_scores,
)

FEATURE_NAMES = [
    "gumbel",
    "textseal",
    "uni_green",
    "uni_frac",
    "kgw_green",
    "kgw_frac",
    "kgw_z",
    "entropy",
    "piece_len",
    "is_space",
    "is_newline",
    "novelty",
    "repeat",
    "gumbel_ctx",
    "textseal_ctx",
    "kgw_frac_ctx",
    "pos",
]


def _local_mean(x: Sequence[float], window: int) -> list[float]:
    n = len(x)
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        seg = x[lo:hi]
        out[i] = sum(seg) / max(1, len(seg))
    return out


def _stat_features(token_ids: Sequence[int], pieces: Sequence[str],
                   window: int = 25) -> dict[str, list[float]]:
    n = len(token_ids)
    piece_len = [float(len(p)) for p in pieces]
    is_space = [1.0 if p.startswith(("Ġ", " ")) else 0.0 for p in pieces]
    is_newline = [1.0 if ("Ċ" in p or "\n" in p) else 0.0 for p in pieces]
    novelty = [0.0] * n
    repeat = [0.0] * n
    for i in range(n):
        lo = max(0, i - window)
        recent = token_ids[lo:i]
        if recent:
            rs = set(recent)
            novelty[i] = float(token_ids[i] not in rs)
            repeat[i] = float(sum(1 for t in recent if t == token_ids[i])) / len(recent)
    return {
        "piece_len": piece_len,
        "is_space": is_space,
        "is_newline": is_newline,
        "novelty": novelty,
        "repeat": repeat,
    }


def extract_features(token_ids: Sequence[int], pieces: Sequence[str],
                     cfg: WatermarkConfig | None = None,
                     use_detectors: bool = True,
                     use_entropy_lm: bool = True,
                     ctx_window: int = 5) -> list[list[float]]:
    """Return an (n_tokens × n_features) list-of-lists aligned to token order."""
    if cfg is None:
        cfg = load_watermark_config()
    n = len(token_ids)

    if use_detectors:
        gum = gumbel_scores(token_ids, cfg)
        ts = textseal_scores(token_ids, cfg)
        uni_g, uni_f = unigram_features(token_ids, cfg)
        kgw_g, kgw_f, kgw_z = kgw_features(token_ids, cfg)
        ent = entropy_scores(token_ids, use_proxy_lm=use_entropy_lm)
    else:  # key-free baseline: detector channels zeroed, stats + novelty entropy only
        gum = [0.0] * n
        ts = [0.0] * n
        uni_g = uni_f = [0.0] * n
        kgw_g = kgw_f = kgw_z = [0.0] * n
        ent = entropy_scores(token_ids, use_proxy_lm=False)

    stats = _stat_features(token_ids, pieces)
    gum_ctx = _local_mean(gum, ctx_window)
    ts_ctx = _local_mean(ts, ctx_window)
    kgw_f_ctx = _local_mean(kgw_f, ctx_window)
    pos = [(i / max(1, n - 1)) for i in range(n)]

    columns = {
        "gumbel": gum, "textseal": ts,
        "uni_green": uni_g, "uni_frac": uni_f,
        "kgw_green": kgw_g, "kgw_frac": kgw_f, "kgw_z": kgw_z,
        "entropy": ent,
        **stats,
        "gumbel_ctx": gum_ctx, "textseal_ctx": ts_ctx, "kgw_frac_ctx": kgw_f_ctx,
        "pos": pos,
    }
    return [[columns[name][i] for name in FEATURE_NAMES] for i in range(n)]
