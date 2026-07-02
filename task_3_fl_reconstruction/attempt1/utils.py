"""IO, image mapping, quality scoring, dedup — no model forward required.

Everything here is label-free and cheap: it works purely on the provided
gradient tensors and on reconstructed pixel arrays.
"""
from __future__ import annotations

import math
import os
from typing import Sequence

import torch
import torch.nn.functional as F

import config


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_gradient(i: int) -> dict:
    """Load gradients/model{i}.pt -> dict with keys gradients/family/... ."""
    path = os.path.join(config.GRADIENTS_DIR, f"model{i}.pt")
    return torch.load(path, weights_only=False, map_location="cpu")


def load_state(i: int) -> dict:
    """Load models/model{i}.pt -> plain state_dict."""
    path = os.path.join(config.MODELS_DIR, f"model{i}.pt")
    return torch.load(path, weights_only=False, map_location="cpu")


def save_submission(submission: dict, path: str = None) -> str:
    path = path or config.SUBMISSION_PATH
    validate_submission(submission)
    torch.save(submission, path)
    return path


def validate_submission(submission: dict) -> None:
    """Fail loudly *before* wasting a 5-minute leaderboard cooldown."""
    expected = {f"model{i}" for i in range(1, config.NUM_MODELS + 1)}
    got = set(submission.keys())
    if got != expected:
        raise ValueError(f"keys mismatch: missing={expected - got} extra={got - expected}")
    for k, v in submission.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"{k}: not a tensor")
        if tuple(v.shape) != (config.BATCH, *config.IMG_SHAPE):
            raise ValueError(f"{k}: shape {tuple(v.shape)} != {(config.BATCH, *config.IMG_SHAPE)}")
        if v.dtype != torch.float32:
            raise TypeError(f"{k}: dtype {v.dtype} != float32")
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise ValueError(f"{k}: contains NaN/Inf")
        lo, hi = float(v.min()), float(v.max())
        if lo < -1e-4 or hi > 1 + 1e-4:
            raise ValueError(f"{k}: values out of [0,1] (min={lo:.4f} max={hi:.4f})")


# --------------------------------------------------------------------------- #
# Image normalization / mapping
# --------------------------------------------------------------------------- #
def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalize a batch (N,C,H,W) into [0,1].

    Robust to arbitrary scale/sign (needed when the layer has no bias or the
    bias-gradient scale is off). Flat images map to zeros.
    """
    n = x.shape[0]
    flat = x.reshape(n, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    span = (hi - lo).clamp_min(1e-8)
    flat = (flat - lo) / span
    return flat.reshape_as(x)


def features_to_image(rows: torch.Tensor, channels: int, hc: int, wc: int) -> torch.Tensor:
    """Map reconstructed feature rows -> (N,3,64,64) in [0,1].

    rows: (N, channels*hc*wc). Reshapes to (N,channels,hc,wc), collapses to 3
    channels, then bilinearly resizes to 64x64.
    """
    n = rows.shape[0]
    x = rows.reshape(n, channels, hc, wc)
    x = _to_three_channels(x)
    x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    return to_unit(x).float()


def flat_to_image(rows: torch.Tensor, feature_shape: Sequence[int]) -> torch.Tensor:
    """Map rows whose length == prod(feature_shape) to (N,3,64,64) in [0,1]."""
    n = rows.shape[0]
    c, h, w = feature_shape
    x = rows.reshape(n, c, h, w)
    x = _to_three_channels(x)
    if (h, w) != (64, 64):
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    return to_unit(x).float()


def _to_three_channels(x: torch.Tensor) -> torch.Tensor:
    """Collapse an arbitrary channel count to exactly 3 (RGB)."""
    c = x.shape[1]
    if c == 3:
        return x
    if c == 1:
        return x.repeat(1, 3, 1, 1)
    if c == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    # c > 3: average the channels into 3 contiguous groups.
    groups = torch.chunk(x, 3, dim=1)
    return torch.stack([g.mean(dim=1) for g in groups], dim=1)


def infer_square(n_features: int, channels: int) -> tuple[int, int]:
    """Best-effort (hc, wc) for a flattened conv feature map of `channels`."""
    per = max(1, n_features // max(1, channels))
    side = int(round(math.sqrt(per)))
    while side > 1 and per % side != 0:
        side -= 1
    hc = side
    wc = max(1, per // side)
    return hc, wc


# --------------------------------------------------------------------------- #
# Label-free quality scoring
# --------------------------------------------------------------------------- #
def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Mean anisotropic TV per image, (N,C,H,W) -> (N,)."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    return dh + dw


def quality_score(x: torch.Tensor) -> torch.Tensor:
    """Heuristic 'looks like a natural image' score, higher = better.

    Rewards contrast, penalizes noise (high TV). Works without ground truth,
    so it never risks leaderboard overfitting.
    """
    x = to_unit(x)
    contrast = x.reshape(x.shape[0], -1).std(dim=1)
    tv = total_variation(x)
    return contrast / (1.0 + 5.0 * tv)


# --------------------------------------------------------------------------- #
# Dedup + selection to exactly K distinct images
# --------------------------------------------------------------------------- #
def _fingerprints(x: torch.Tensor, size: int = 16) -> torch.Tensor:
    g = _to_three_channels(x).mean(dim=1, keepdim=True)  # gray
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    f = g.reshape(g.shape[0], -1)
    f = f - f.mean(dim=1, keepdim=True)
    return F.normalize(f, dim=1)


def dedup_select(
    imgs: torch.Tensor,
    scores: torch.Tensor,
    k: int = config.BATCH,
    sim_threshold: float = config.DEDUP_SIM_THRESHOLD,
) -> torch.Tensor:
    """Greedily pick up to k *distinct* images, best score first.

    Returns a (k,3,64,64) tensor. If fewer than k distinct candidates exist,
    pads with the next-best (possibly similar) ones, then with diverse noise.
    """
    imgs = to_unit(imgs)
    fp = _fingerprints(imgs)
    order = torch.argsort(scores, descending=True)

    chosen: list[int] = []
    chosen_fp: list[torch.Tensor] = []
    for idx in order.tolist():
        if len(chosen) >= k:
            break
        if chosen_fp:
            sims = torch.stack(chosen_fp) @ fp[idx]
            if float(sims.max()) > sim_threshold:
                continue
        chosen.append(idx)
        chosen_fp.append(fp[idx])

    # Not enough distinct -> relax and take remaining best (allow near-dups).
    if len(chosen) < k:
        remaining = [i for i in order.tolist() if i not in set(chosen)]
        chosen.extend(remaining[: k - len(chosen)])

    out = imgs[torch.tensor(chosen[:k], dtype=torch.long)] if chosen else imgs[:0]

    # Still short (e.g. no candidates at all) -> diverse noise fills the rest.
    if out.shape[0] < k:
        pad = torch.rand(k - out.shape[0], *config.IMG_SHAPE)
        out = torch.cat([out, pad], dim=0) if out.numel() else pad
    return out[:k].contiguous().float()
