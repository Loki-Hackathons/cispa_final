"""Per-token watermark detector signals for Task 1.

Computes raw per-token statistics from token_ids only (no model forward):
- TextSeal  (dual-key Gumbel PRF, ngram=3, alpha=0.5)   -> continuous, H0 ~ mean 1, var 0.5
- Gumbel-Max (single-key Gumbel PRF, ngram=2)           -> continuous, H0 ~ Exp(1): mean 1, var 1
- Unigram   (fixed greenlist, fraction=0.5)             -> binary,     H0 ~ Bernoulli(0.5)
- KGW       (CUDA Philox greenlists)                    -> computed separately on GPU (kgw_scores.py)

All PRF math reuses the pinned vendor code (textseal commit 788fe8b) so the
int64 wrap-around semantics match the organizers' generation exactly.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

_VENDOR = Path(__file__).resolve().parents[1] / "vendor"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_textseal_core():
    """Load vendor textseal core PRF without triggering the heavy package __init__."""
    root = _VENDOR / "textseal" / "textseal"
    for pkg_name, pkg_path in [("textseal", root), ("textseal.watermarking", root / "watermarking")]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_path)]
            sys.modules[pkg_name] = pkg
    _load_module("textseal.watermarking.config", root / "watermarking" / "config.py")
    return _load_module("textseal.watermarking.core", root / "watermarking" / "core.py")


prf_uniform = _load_textseal_core().prf_uniform

# Keys from task_1_text_watermark/watermark_config.yaml
TEXTSEAL_KEY_A = 947821031
TEXTSEAL_KEY_B = 1562881159
TEXTSEAL_NGRAM = 3
TEXTSEAL_ALPHA = 0.5

GUMBEL_KEY = 2004683203
GUMBEL_NGRAM = 2

UNIGRAM_KEY = 1873092841
UNIGRAM_FRACTION = 0.5

# H0 moments per signal (used for windowed z-scores)
H0_MOMENTS = {
    "textseal": (1.0, 0.5),
    "gumbelmax": (1.0, 1.0),
    "unigram": (0.5, 0.25),
    "kgw": (0.25, 0.1875),  # gamma = 0.25
}


def _windows_and_targets(token_ids: list[int], ngram: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (windows (n, ngram), targets (n,), first_scored_pos)."""
    toks = torch.tensor(token_ids, dtype=torch.long)
    n = len(token_ids)
    if n <= ngram:
        return torch.empty(0, ngram, dtype=torch.long), torch.empty(0, dtype=torch.long), n
    windows = toks.unfold(0, ngram, 1)[:-1]  # (n - ngram, ngram): context for positions ngram..n-1
    targets = toks[ngram:]
    return windows, targets, ngram


def _gumbel_increment(r: torch.Tensor) -> torch.Tensor:
    return -torch.log1p(-r.clamp(max=1.0 - 1e-9))


def textseal_signal(token_ids: list[int]) -> np.ndarray:
    """Fused dual-key Gumbel score per token. Positions < ngram get H0 mean."""
    out = np.full(len(token_ids), H0_MOMENTS["textseal"][0], dtype=np.float64)
    windows, targets, start = _windows_and_targets(token_ids, TEXTSEAL_NGRAM)
    if len(targets) == 0:
        return out
    r_a = prf_uniform(windows, targets, TEXTSEAL_KEY_A)
    r_b = prf_uniform(windows, targets, TEXTSEAL_KEY_B)
    fused = TEXTSEAL_ALPHA * _gumbel_increment(r_a) + (1 - TEXTSEAL_ALPHA) * _gumbel_increment(r_b)
    out[start:] = fused.numpy()
    return out


def gumbelmax_signal(token_ids: list[int]) -> np.ndarray:
    out = np.full(len(token_ids), H0_MOMENTS["gumbelmax"][0], dtype=np.float64)
    windows, targets, start = _windows_and_targets(token_ids, GUMBEL_NGRAM)
    if len(targets) == 0:
        return out
    r = prf_uniform(windows, targets, GUMBEL_KEY)
    out[start:] = _gumbel_increment(r).numpy()
    return out


def gumbelmax_r(token_ids: list[int]) -> np.ndarray:
    """Raw PRF draw r in [0,1] per token (H0: uniform). Positions < ngram get 0.5."""
    out = np.full(len(token_ids), 0.5, dtype=np.float64)
    windows, targets, start = _windows_and_targets(token_ids, GUMBEL_NGRAM)
    if len(targets) == 0:
        return out
    out[start:] = prf_uniform(windows, targets, GUMBEL_KEY).numpy()
    return out


def textseal_r(token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Raw dual-key PRF draws (r_a, r_b), each H0: uniform. < ngram -> 0.5."""
    out_a = np.full(len(token_ids), 0.5, dtype=np.float64)
    out_b = np.full(len(token_ids), 0.5, dtype=np.float64)
    windows, targets, start = _windows_and_targets(token_ids, TEXTSEAL_NGRAM)
    if len(targets) == 0:
        return out_a, out_b
    out_a[start:] = prf_uniform(windows, targets, TEXTSEAL_KEY_A).numpy()
    out_b[start:] = prf_uniform(windows, targets, TEXTSEAL_KEY_B).numpy()
    return out_a, out_b


_UNIGRAM_MASK_CACHE: dict[int, np.ndarray] = {}


def unigram_mask(vocab_size: int) -> np.ndarray:
    """Greenlist mask, replicating vendor gptwm.GPTWatermarkBase exactly."""
    if vocab_size not in _UNIGRAM_MASK_CACHE:
        import hashlib

        seed = int.from_bytes(hashlib.sha256(np.int64(UNIGRAM_KEY)).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        n_green = int(UNIGRAM_FRACTION * vocab_size)
        mask = np.array([True] * n_green + [False] * (vocab_size - n_green))
        rng.shuffle(mask)
        _UNIGRAM_MASK_CACHE[vocab_size] = mask
    return _UNIGRAM_MASK_CACHE[vocab_size]


def unigram_signal(token_ids: list[int], vocab_size: int = 151643) -> np.ndarray:
    """1 if token in greenlist. Out-of-vocab token ids get H0 mean (uninformative)."""
    mask = unigram_mask(vocab_size)
    ids = np.asarray(token_ids)
    out = np.full(len(ids), H0_MOMENTS["unigram"][0], dtype=np.float64)
    in_vocab = ids < vocab_size
    out[in_vocab] = mask[ids[in_vocab]].astype(np.float64)
    return out


def _dedup_mask(token_ids: list[int], ngram: int) -> np.ndarray:
    """True where the (context, target) n-gram is seen for the first time.

    Repeated n-grams re-emit the exact same PRF draw, so they carry no new
    evidence; counting them inflates window z-scores on repetitive text.
    """
    n = len(token_ids)
    keep = np.ones(n, dtype=bool)
    seen = set()
    for i in range(ngram, n):
        key = tuple(token_ids[i - ngram:i + 1])
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    return keep


def compute_signals(token_ids: list[int], unigram_vocab_size: int = 151643,
                    dedup: bool = True) -> dict[str, np.ndarray]:
    sigs = {
        "textseal": textseal_signal(token_ids),
        "gumbelmax": gumbelmax_signal(token_ids),
        "unigram": unigram_signal(token_ids, unigram_vocab_size),
    }
    if dedup:
        for name, ngram in [("textseal", TEXTSEAL_NGRAM), ("gumbelmax", GUMBEL_NGRAM)]:
            keep = _dedup_mask(token_ids, ngram)
            sigs[name][~keep] = H0_MOMENTS[name][0]
    return sigs
