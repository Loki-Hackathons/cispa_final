"""Diagnostics to guide strategy WITHOUT touching the leaderboard.

Three things it answers:
  1. Per-model: how many analytic candidates, quality distribution, how many
     survive dedup -> tells us which models are reconstructible and how well.
  2. Cross-model batch-share test: are the 128 private images the SAME across
     models, or distinct? (The task says "reconstruct 1536 images" = 12x128,
     which suggests distinct batches, but we verify empirically.) This decides
     whether cross-model image reuse is even valid.
  3. Preview: save reconstructions as PNG (via PIL/matplotlib if available) and
     always as .npy, so we can finally LOOK at what we produce.

Usage:
  python analyze.py --stats
  python analyze.py --crosscheck 5 8        # are two models' batches shared?
  python analyze.py --preview 5 8 3 --select kmeans
"""
from __future__ import annotations

import argparse
import os

import torch

import config
import extract
import utils


def _analytic(i, select="quality"):
    grad = utils.load_gradient(i)
    info = extract.introspect(i, grad)
    cands = extract.extract_analytic(info)
    if cands.shape[0] == 0:
        return info, cands, None
    scores = utils.quality_score(cands)
    if select == "kmeans":
        sel = utils.kmeans_select(cands, scores, k=config.BATCH)
    else:
        sel = utils.dedup_select(cands, scores, k=config.BATCH)
    return info, cands, sel


def stats():
    print(f"{'model':>5} | {'fam':3} | {'act':7} | {'cands':>5} | "
          f"{'q_mean':>7} | {'q_max':>7}")
    print("-" * 60)
    for i in range(1, config.NUM_MODELS + 1):
        info, cands, _ = _analytic(i)
        if cands.shape[0] == 0:
            print(f"{i:>5} | {info.family:3} | {info.activation:7} | "
                  f"{0:>5} | {'-':>7} | {'-':>7}")
            continue
        q = utils.quality_score(cands)
        print(f"{i:>5} | {info.family:3} | {info.activation:7} | "
              f"{cands.shape[0]:>5} | {float(q.mean()):>7.4f} | "
              f"{float(q.max()):>7.4f}")


def crosscheck(models, top=32):
    """Compare best reconstructions across models via SSIM nearest-match.

    High cross-model best-match SSIM => batches likely SHARED (same images).
    Low => batches DISTINCT (reconstruct each model independently).
    """
    sels = {}
    for i in models:
        _, _, sel = _analytic(i)
        if sel is not None:
            sels[i] = sel[:top]
    keys = list(sels.keys())
    print(f"cross-model best-match SSIM (top {top} by order), higher=more shared")
    print("     " + " ".join(f"{j:>6}" for j in keys))
    for a in keys:
        row = []
        for b in keys:
            if a == b:
                row.append("  1.00")
                continue
            m = utils.ssim_matrix(sels[a], sels[b])   # (top, top)
            best = m.max(dim=1).values.mean()          # avg nearest-match
            row.append(f"{float(best):6.2f}")
        print(f"{a:>4} " + " ".join(row))
    print("\nHeuristic: off-diagonal >~0.6 suggests shared images; "
          "<~0.3 suggests distinct batches.")


def _save_png(t, path):
    """Save a (N,3,H,W) montage; try torchvision, PIL, matplotlib in order."""
    t = utils.to_unit(t).float()
    try:
        from torchvision.utils import save_image
        save_image(t, path, nrow=8)
        return "torchvision"
    except Exception:
        pass
    # Build a grid manually -> (H*rows, W*cols, 3) uint8.
    import math
    n = t.shape[0]
    cols = 8
    rows = math.ceil(n / cols)
    _, c, h, w = t.shape
    grid = torch.zeros(3, rows * h, cols * w)
    for idx in range(n):
        r, cc = divmod(idx, cols)
        grid[:, r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t[idx]
    arr = (grid.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return "PIL"
    except Exception:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, arr)
        return "matplotlib"
    except Exception:
        return None


def preview(models, select="quality", n=64):
    os.makedirs(config.OUT_DIR, exist_ok=True)
    for i in models:
        _, cands, sel = _analytic(i, select=select)
        if sel is None:
            print(f"model{i}: no candidates")
            continue
        top = sel[:n]
        npy = os.path.join(config.OUT_DIR, f"preview_model{i}.npy")
        import numpy as np
        np.save(npy, utils.to_unit(top).cpu().numpy())
        png = os.path.join(config.OUT_DIR, f"preview_model{i}.png")
        how = _save_png(top, png)
        msg = f"model{i}: saved {top.shape[0]} imgs -> {npy}"
        msg += f" and {png} ({how})" if how else f" (png skipped: no PIL/mpl)"
        print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--crosscheck", type=int, nargs="*", default=None)
    ap.add_argument("--preview", type=int, nargs="*", default=None)
    ap.add_argument("--select", choices=["quality", "kmeans"], default="quality")
    args = ap.parse_args()

    if args.stats or (args.crosscheck is None and args.preview is None):
        stats()
    if args.crosscheck:
        crosscheck(args.crosscheck)
    if args.preview:
        preview(args.preview, select=args.select)


if __name__ == "__main__":
    main()
