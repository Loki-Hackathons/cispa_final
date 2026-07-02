"""Controlled experiments for Task 3 root-cause analysis.

This file is intentionally separate from `run.py`: the stable leaderboard
baseline remains unchanged while we test one hypothesis at a time.

Hypotheses we can test here:
  H1: per-image min-max normalization destroys useful absolute scale.
  H2: CNN 8-channel -> RGB projection is the weak link.
  H3: the analytic identity is exact only for isolated rows; mixed rows are
      weighted averages of multiple private images.

Usage on cluster:
  python experiment.py --synthetic
  python experiment.py --norm unit --channel-map chunks --out submission_exp.pt
  python experiment.py --norm robust --channel-map first3 --out submission_exp.pt
  python submit.py --check submission_exp.pt
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

import config
import extract
import utils


NORM_MODES = ("unit", "clamp", "sigmoid", "tanh", "robust", "zsigmoid")
CHANNEL_MAPS = ("chunks", "first3", "mean", "maxabs")


def normalize(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Map arbitrary image-like tensors to valid [0,1] submission images."""
    x = x.float()
    if mode == "unit":
        return utils.to_unit(x).float()
    if mode == "clamp":
        return x.clamp(0, 1).float()
    if mode == "sigmoid":
        return torch.sigmoid(x).float()
    if mode == "tanh":
        return ((torch.tanh(x) + 1) / 2).float()
    if mode == "zsigmoid":
        flat = x.reshape(x.shape[0], -1)
        mu = flat.mean(dim=1, keepdim=True)
        sd = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        z = ((flat - mu) / (2 * sd)).reshape_as(x)
        return torch.sigmoid(z).float()
    if mode == "robust":
        flat = x.reshape(x.shape[0], -1)
        lo = torch.quantile(flat, 0.01, dim=1, keepdim=True)
        hi = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        y = ((flat - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
        return y.reshape_as(x).float()
    raise ValueError(f"unknown norm mode {mode}")


def map_channels(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Collapse C feature channels to RGB with controlled alternatives."""
    c = x.shape[1]
    if c == 3:
        return x
    if mode == "chunks":
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        if c == 2:
            return torch.cat([x, x[:, :1]], dim=1)
        groups = torch.chunk(x, 3, dim=1)
        return torch.stack([g.mean(dim=1) for g in groups], dim=1)
    if mode == "first3":
        if c >= 3:
            return x[:, :3]
        return x.repeat(1, (3 + c - 1) // c, 1, 1)[:, :3]
    if mode == "mean":
        return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    if mode == "maxabs":
        # Pick the three most energetic channels per image. This tests whether
        # averaging all 8 CNN channels is washing out the signal.
        energy = x.abs().flatten(2).mean(dim=2)  # (N,C)
        idx = torch.argsort(energy, dim=1, descending=True)[:, :min(3, c)]
        out = []
        for n in range(x.shape[0]):
            chans = x[n:n + 1, idx[n]]
            if chans.shape[1] < 3:
                chans = chans.repeat(1, 3, 1, 1)[:, :3]
            out.append(chans)
        return torch.cat(out, dim=0)
    raise ValueError(f"unknown channel map {mode}")


def analytic_rows(gW: torch.Tensor, gb: torch.Tensor, eps: float = config.EPS):
    valid = gb.abs() > eps
    if valid.sum() == 0:
        return torch.empty(0, gW.shape[1])
    return gW[valid] / gb[valid].unsqueeze(1)


def candidate_images(i: int, norm: str, channel_map: str) -> torch.Tensor:
    """Return raw analytic candidates transformed by experimental knobs."""
    grad = utils.load_gradient(i)
    info = extract.introspect(i, grad)
    gr = info.grads

    if info.family == "mlp":
        rows = analytic_rows(gr["net.0.weight"], gr["net.0.bias"])
        if rows.numel() == 0:
            return torch.empty(0, *config.IMG_SHAPE)
        x = rows.reshape(rows.shape[0], *config.IMG_SHAPE)
        return normalize(map_channels(x, channel_map), norm)

    if info.family == "cnn" and "fc1.weight" in gr and "fc1.bias" in gr:
        rows = analytic_rows(gr["fc1.weight"], gr["fc1.bias"])
        if rows.numel() == 0:
            return torch.empty(0, *config.IMG_SHAPE)
        flat = rows.shape[1]
        fc, fh, fw = info.feature_shape
        if fc * fh * fw == flat:
            c, h, w = fc, fh, fw
        else:
            c = gr["conv.weight"].shape[0] if "conv.weight" in gr else 1
            h, w = utils.infer_square(flat, c)
            if c * h * w != flat:
                return torch.empty(0, *config.IMG_SHAPE)
        x = rows.reshape(rows.shape[0], c, h, w)
        x = map_channels(x, channel_map)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return normalize(x, norm)

    return torch.empty(0, *config.IMG_SHAPE)


def select_candidates(cands: torch.Tensor, mode: str) -> torch.Tensor:
    if cands.shape[0] == 0:
        return torch.rand(config.BATCH, *config.IMG_SHAPE)
    scores = utils.quality_score(cands)
    if mode == "kmeans":
        return utils.kmeans_select(cands, scores, k=config.BATCH)
    return utils.dedup_select(cands, scores, k=config.BATCH)


def make_submission(norm: str, channel_map: str, select: str, out: str, models: list[int]):
    submission = {f"model{i}": torch.rand(config.BATCH, *config.IMG_SHAPE)
                  for i in range(1, config.NUM_MODELS + 1)}
    for i in models:
        cands = candidate_images(i, norm=norm, channel_map=channel_map)
        imgs = select_candidates(cands, select).float()
        submission[f"model{i}"] = imgs
        q = utils.quality_score(imgs).mean()
        print(f"model{i:2d}: cands={cands.shape[0]:4d} "
              f"q={float(q):.4f} norm={norm} channels={channel_map}")
    path = utils.save_submission(submission, out)
    print(f"[experiment] wrote {path} (validated OK)")


def synthetic():
    """Prove the analytic identity's exact-vs-mixed behavior.

    We synthesize gradients for a linear layer. If each row receives gradient
    from exactly one sample, gW_i/gb_i recovers that sample. If a row receives
    gradient from several samples, the same ratio is only their weighted average.
    """
    torch.manual_seed(config.SEED)
    b, d, n = 16, 64, 64
    x = torch.rand(b, d)

    # Isolated rows: row i is tied to sample i % B.
    delta = torch.zeros(n, b)
    for i in range(n):
        delta[i, i % b] = torch.randn(()) * 0.5 + 1.0
    gW = delta @ x
    gb = delta.sum(dim=1)
    rec = gW / gb.unsqueeze(1)
    target = x[torch.arange(n) % b]
    iso_mse = F.mse_loss(rec, target).item()

    # Mixed rows: every row gets positive contributions from all samples.
    # Positive weights avoid near-zero cancellations and show the normal case:
    # gW_i/gb_i becomes a convex combination of private images.
    delta_m = torch.rand(n, b)
    gb_m = delta_m.sum(dim=1).clamp_min(1e-6)
    gW_m = delta_m @ x
    rec_m = gW_m / gb_m.unsqueeze(1)
    # nearest true image error remains high because rows are averages.
    dist = torch.cdist(rec_m, x).min(dim=1).values.mean().item()

    print("Synthetic analytic identity test")
    print(f"  isolated rows MSE to true sample: {iso_mse:.8f} (should be ~0)")
    print(f"  mixed rows nearest-image L2:      {dist:.4f} (weighted averages)")
    print("Root-cause implication: candidates are only true images when the model")
    print("creates isolated rows/trap weights; otherwise quality scoring is choosing")
    print("among mixtures, not real reconstructions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--norm", choices=NORM_MODES, default="unit")
    ap.add_argument("--channel-map", choices=CHANNEL_MAPS, default="chunks")
    ap.add_argument("--select", choices=["quality", "kmeans"], default="quality")
    ap.add_argument("--models", type=int, nargs="*", default=list(range(1, 13)))
    ap.add_argument("--out", type=str, default="submission_exp.pt")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    if args.synthetic:
        synthetic()
        return
    make_submission(args.norm, args.channel_map, args.select, args.out, args.models)


if __name__ == "__main__":
    main()
