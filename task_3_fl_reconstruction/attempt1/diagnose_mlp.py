"""Diagnostic for MLP analytic reconstruction quality (models 1, 4, 5, 8).

For an MLP, net.0 sees the flattened RGB image directly, so the analytic row
    row_i = gW0_i / gb0_i
IS an input reconstruction — exact when a single sample activated neuron i
(ReLU), a blurry mixture when every sample contributes (sigmoid/tanh).

This script tells us, per model, WHERE the headroom is before we spend GPU:
  - how many neurons are usable (|gb0| > eps)
  - the distribution of the natural-image quality score
  - the fraction of rows whose own neuron is actually active on the
    reconstruction (a principled "this is a real activating image" signal for
    ReLU; near-meaningless for smooth activations, reported anyway)
  - internal diversity of the selected 128 (mean pairwise SSIM; high = we are
    submitting near-duplicates, wasting matched slots)
  - a preview grid of the top-64 reconstructions

Usage:
  python diagnose_mlp.py --models 1 4 5 8 --n 64
"""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

import config
import extract
import utils


def grid(imgs: torch.Tensor, cols: int = 8) -> Image.Image:
    imgs = utils.to_unit(imgs).clamp(0, 1).cpu()
    n, _, h, w = imgs.shape
    rows = (n + cols - 1) // cols
    canvas = torch.zeros(3, rows * h, cols * w)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = imgs[i]
    arr = (canvas.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def mean_pairwise_ssim(imgs: torch.Tensor, sample: int = 64) -> float:
    """Rough internal-diversity probe: lower = more distinct images."""
    x = imgs[:sample]
    m = utils.ssim_matrix(x, x)
    n = m.shape[0]
    if n < 2:
        return float("nan")
    off = m.sum() - m.diagonal().sum()
    return float(off / (n * (n - 1)))


def diagnose(i: int, n_preview: int, outdir: str) -> None:
    grad = utils.load_gradient(i)
    info = extract.introspect(i, grad)
    if info.family != "mlp":
        print(f"model{i}: family={info.family} (not mlp), skipping")
        return

    gW = info.grads["net.0.weight"]         # (H, 12288) gradient
    gb = info.grads["net.0.bias"]           # (H,)
    valid = gb.abs() > config.EPS
    n_valid = int(valid.sum())
    rows = gW[valid] / gb[valid].unsqueeze(1)   # (N, 12288)

    imgs = utils.flat_to_image(rows, config.IMG_SHAPE)    # (N,3,64,64) normed
    q = utils.quality_score(imgs)

    # Own-neuron activation on its own reconstruction, using MODEL weights.
    state = utils.load_state(i)
    W0 = state["net.0.weight"][valid]       # (N, 12288) weights (not gradient)
    b0 = state["net.0.bias"][valid]
    own_act = (W0 * rows).sum(dim=1) + b0    # pre-activation a_i(r_i)
    frac_active = float((own_act > 0).float().mean())

    # Selected 128 (current pipeline: quality + greedy dedup).
    sel = utils.dedup_select(imgs, q, k=config.BATCH)
    div = mean_pairwise_ssim(sel)

    q_sorted = torch.sort(q, descending=True).values
    top128_q = float(q_sorted[:config.BATCH].mean()) if n_valid else float("nan")

    os.makedirs(outdir, exist_ok=True)
    order = torch.argsort(q, descending=True)
    grid(imgs[order[:n_preview]]).save(f"{outdir}/model{i}_top{n_preview}.png")

    print(f"model{i:2d} [{info.activation:7s}] "
          f"usable_neurons={n_valid:4d} "
          f"quality(top128 mean)={top128_q:.4f} "
          f"quality[min/med/max]="
          f"{float(q.min()):.3f}/{float(q.median()):.3f}/{float(q.max()):.3f} "
          f"own_active_frac={frac_active:.2f} "
          f"selected_meanSSIM={div:.3f}")
    print(f"          preview -> {outdir}/model{i}_top{n_preview}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=[1, 4, 5, 8])
    ap.add_argument("--n", type=int, default=64, help="preview grid size")
    ap.add_argument("--outdir", type=str, default="output/mlp_diag")
    args = ap.parse_args()

    print("=== MLP analytic diagnostic ===")
    print("high quality + high own_active_frac + low selected_meanSSIM = good.")
    print("low quality (blurry mixtures) => needs gradient-matching refine.\n")
    for i in args.models:
        diagnose(i, args.n, args.outdir)


if __name__ == "__main__":
    main()
