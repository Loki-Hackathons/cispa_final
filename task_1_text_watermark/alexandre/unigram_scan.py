"""Unlabeled Unigram presence scan.

The Unigram greenlist is context-free, so its presence is testable WITHOUT
labels: sliding-window green-fraction z-scores over every document, comparing
the real watermark key against random decoy keys on the exact same text
(exact negative control: same token frequencies, same repetitions).

Generation-time constraint pins vocab_size: GPTWatermarkLogitsWarper adds
`strength * green_list_mask` directly to the model logits, so the mask must
have the logits dimension = model config vocab_size = 152064 (Qwen2.5-7B).

Token-level dedup: a repeated token id re-contributes the same mask bit (no
new evidence), so only first occurrences count — same fix as for the PRF
schemes (detectors._dedup_mask), but keyed on single token ids.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Two distinct vocab constants — do not merge them:
#
# BASE_VOCAB (151643): organizer-confirmed (Maitri Shah, 2026-07-02) as "the
# Unigram vocabulary"; special tokens (id >= 151643) are outside the scheme,
# so they are EXCLUDED from scoring eligibility.
#
# MASK_VOCAB (152064): the permutation size that actually matches the data.
# Head-to-head on the test set (20 decoy keys as empirical null), uniform
# eligibility rule: real key at 151643 ranks 21/21 (pure noise), at 151665
# 19/21, at 151936 8/21, at 152064 **1/21** with n(z>4)=28 vs null max 15.
# Same verdict on train/val GT labels (18 aligned windows vs 3).
# This is consistent with the vendor generation code: the LogitsWarper adds
# `strength * mask` directly to the logits tensor, which requires
# len(mask) == model logits dim == config vocab_size == 152064.
BASE_VOCAB = 151643   # eligibility cutoff (special tokens excluded)
VOCAB_SIZE = 152064   # greenlist permutation size (empirically pinned)
FRACTION = 0.5
REAL_KEY = 1873092841
WINDOW_LENGTHS = (31, 47, 63, 95, 159, 320)
MIN_KEPT = 15  # minimum unique eligible tokens in a window to score it


def greenlist_mask(key: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Replicates vendor GPTWatermarkBase exactly."""
    seed = int.from_bytes(hashlib.sha256(np.int64(key)).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    n_green = int(FRACTION * vocab_size)
    mask = np.array([True] * n_green + [False] * (vocab_size - n_green))
    rng.shuffle(mask)
    return mask


def token_dedup_mask(token_ids: list[int]) -> np.ndarray:
    """First occurrence of each ELIGIBLE token id; special tokens never kept."""
    seen: set[int] = set()
    keep = np.zeros(len(token_ids), dtype=bool)
    for i, t in enumerate(token_ids):
        if t < BASE_VOCAB and t not in seen:
            keep[i] = True
            seen.add(t)
    return keep


def doc_max_window_z(token_ids: list[int], mask: np.ndarray) -> float:
    """Max green-fraction z over sliding windows of canonical span lengths."""
    ids = np.asarray(token_ids)
    keep = token_dedup_mask(token_ids)
    green = np.zeros(len(ids), dtype=bool)
    green[keep] = mask[ids[keep]]
    ps_g = np.concatenate([[0], np.cumsum(green)])
    ps_k = np.concatenate([[0], np.cumsum(keep)])
    n = len(ids)
    best = -np.inf
    for L in WINDOW_LENGTHS:
        if L > n:
            continue
        s = np.arange(0, n - L + 1)
        g = ps_g[s + L] - ps_g[s]
        c = ps_k[s + L] - ps_k[s]
        ok = c >= MIN_KEPT
        if not ok.any():
            continue
        z = (g[ok] - FRACTION * c[ok]) / np.sqrt(FRACTION * (1 - FRACTION) * c[ok])
        best = max(best, float(z.max()))
    return best


def scan(records: list[dict], keys: list[int]) -> dict[int, np.ndarray]:
    """Per-key array of per-document max window z."""
    masks = {k: greenlist_mask(k) for k in keys}
    out = {k: np.empty(len(records)) for k in keys}
    for i, rec in enumerate(records):
        for k in keys:
            out[k][i] = doc_max_window_z(rec["token_ids"], masks[k])
    return out
