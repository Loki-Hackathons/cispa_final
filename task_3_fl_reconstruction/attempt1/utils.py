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
    config.ensure_data_root()
    path = os.path.join(config.GRADIENTS_DIR, f"model{i}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing gradient file: {path}\n"
            f"  TASK3_DATA_ROOT={config.DATA_ROOT}\n"
            "Run: source setup_cluster.sh"
        )
    return torch.load(path, weights_only=False, map_location="cpu")


def load_state(i: int) -> dict:
    """Load models/model{i}.pt -> plain state_dict."""
    config.ensure_data_root()
    path = os.path.join(config.MODELS_DIR, f"model{i}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing model file: {path}\n"
            f"  TASK3_DATA_ROOT={config.DATA_ROOT}\n"
            "Run: source setup_cluster.sh"
        )
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


# --------------------------------------------------------------------------- #
# SSIM (the actual competition metric; used only for INTERNAL selection and
# cross-model diagnostics — never against ground truth or the leaderboard)
# --------------------------------------------------------------------------- #
def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)  # (size, size)


def ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5) -> torch.Tensor:
    """Mean structural similarity per image pair. a,b: (N,C,H,W) in [0,1].

    Returns (N,) SSIM values. Matches the standard Wang et al. formulation
    (Gaussian window, C1/C2 for dynamic range 1.0), averaged over channels.
    """
    a = to_unit(a).float()
    b = to_unit(b).float()
    c = a.shape[1]
    win = _gaussian_window(window_size, sigma).to(a.device)
    win = win.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def filt(x):
        return F.conv2d(x, win, padding=pad, groups=c)

    mu_a, mu_b = filt(a), filt(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = filt(a * a) - mu_a2
    sb = filt(b * b) - mu_b2
    sab = filt(a * b) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_ab + c1) * (2 * sab + c2)) / \
           ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return smap.mean(dim=(1, 2, 3))


def ssim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise SSIM between every image in a (Na) and b (Nb) -> (Na, Nb).

    O(Na*Nb) forward passes worth of work; fine for a few hundred images.
    """
    na, nb = a.shape[0], b.shape[0]
    out = torch.zeros(na, nb)
    for i in range(na):
        rep = a[i:i + 1].expand(nb, -1, -1, -1)
        out[i] = ssim(rep, b)
    return out


# --------------------------------------------------------------------------- #
# K-means diverse selection: better coverage of the distinct images than the
# greedy dedup (which can cluster around a few easy reconstructions).
# --------------------------------------------------------------------------- #
def kmeans_select(imgs: torch.Tensor, scores: torch.Tensor,
                  k: int = config.BATCH, iters: int = 25,
                  seed: int = config.SEED) -> torch.Tensor:
    """Cluster candidates into k groups; return the best-scoring member of each.

    Gives one representative per visual cluster, so the 128 outputs cover 128
    *distinct* reconstructions rather than 128 near-duplicates of the easiest
    few. Falls back to noise padding if fewer than k candidates exist.
    """
    imgs = to_unit(imgs)
    n = imgs.shape[0]
    if n == 0:
        return torch.rand(k, *config.IMG_SHAPE).float()
    if n <= k:
        out = imgs
        if n < k:
            pad = torch.rand(k - n, *config.IMG_SHAPE)
            out = torch.cat([out, pad], dim=0)
        return out[:k].contiguous().float()

    fp = _fingerprints(imgs)                     # (n, d) unit vectors
    g = torch.Generator().manual_seed(seed)
    # k-means++-ish init: seed with the top-score candidate then farthest points.
    centers = [int(torch.argmax(scores))]
    d2 = 1.0 - (fp @ fp[centers[0]]).clamp(-1, 1)
    for _ in range(1, k):
        nxt = int(torch.argmax(d2))
        centers.append(nxt)
        d2 = torch.minimum(d2, 1.0 - (fp @ fp[nxt]).clamp(-1, 1))
    cen = fp[torch.tensor(centers)]

    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        sims = fp @ cen.t()                      # (n, k) cosine sim
        assign = sims.argmax(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                cen[j] = F.normalize(fp[m].mean(dim=0), dim=0)

    chosen = []
    for j in range(k):
        m = (assign == j).nonzero(as_tuple=True)[0]
        if m.numel() == 0:
            continue
        best = m[torch.argmax(scores[m])]
        chosen.append(int(best))
    # Fill any empty clusters with next best unused candidates.
    if len(chosen) < k:
        order = torch.argsort(scores, descending=True).tolist()
        chosen_set = set(chosen)
        for idx in order:
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            if len(chosen) >= k:
                break
    return imgs[torch.tensor(chosen[:k], dtype=torch.long)].contiguous().float()
