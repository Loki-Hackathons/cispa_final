"""v2 reconstruction pipeline — per-family analytic recovery + diagnostics.

Improvements over run.py (all label-free, none tuned against the leaderboard):

  * MLP: isolated-image recovery by clustering analytic rows (separation.py)
    instead of picking 128 possibly-mixture rows; clamp (not min/max stretch)
    keeps the true pixel scale for ReLU single-image rows -> better SSIM.
  * CNN: recover the fc1-input feature maps, then invert the KNOWN conv's
    transmit filters channel-by-channel (channels.py) instead of averaging all
    8 feature channels (which mixed image + noise channels). Falls back to the
    old channel averaging when no transmit structure exists.
  * Leftover slots filled with diverse augmented variants of real
    reconstructions rather than white noise.

Smooth-activation (sigmoid/tanh) and ViT models have no clean analytic path;
v2 gives their best analytic cluster here, then refine on GPU with the existing
`mlp_reconstruct.py --refine` (exact-forward gradient matching) as needed.

Usage:
  python reconstruct_v2.py --diagnose                 # no GPU: stats + previews
  python reconstruct_v2.py --out submission_v2.pt     # build all 12
  python reconstruct_v2.py --models 5 8 6 10 --out submission_v2.pt
  python reconstruct_v2.py --base submission.pt --models 6 10 12 --out sub_cnn.pt
"""
from __future__ import annotations

import argparse
import os

import torch

import channels
import config
import extract
import separation
import utils


# --------------------------------------------------------------------------- #
# Family-specific row -> RGB mappers
# --------------------------------------------------------------------------- #
def _mlp_to_rgb(activation: str):
    def f(rows: torch.Tensor) -> torch.Tensor:
        x = rows.reshape(rows.shape[0], *config.IMG_SHAPE)
        if activation == "relu":
            return x.clamp(0, 1).float()                 # true scale for single-image rows
        return utils.to_unit(x).float()                  # mixtures: scale unknown
    return f


def _cnn_to_rgb(i: int, feature_shape: tuple[int, int, int]):
    state = utils.load_state(i)
    grad = utils.load_gradient(i)
    conv_w = state["conv.weight"]
    conv_b = state.get("conv.bias")
    act = grad["activation"]
    c, h, w = feature_shape

    def f(rows: torch.Tensor) -> torch.Tensor:
        feats = rows.reshape(rows.shape[0], c, h, w)
        rgb, _ = channels.transmit_features_to_rgb(feats, conv_w, conv_b, act)
        return rgb
    return f


# --------------------------------------------------------------------------- #
# Per-model recovery
# --------------------------------------------------------------------------- #
def recover_mlp(i: int, info: extract.ModelInfo):
    gW = info.grads["net.0.weight"]
    gb = info.grads["net.0.bias"]
    rows, valid = separation.analytic_rows(gW, gb)
    if rows.shape[0] == 0:
        return separation.diversify_fill(torch.empty(0, *config.IMG_SHAPE),
                                         config.BATCH), "empty"

    # Own-margin (w_i.r_i+b_i)/||w_i|| from the KNOWN weights ranks likely
    # single-image rows first — the best label-free selector in bench_selection.
    state = utils.load_state(i)
    W0 = state["net.0.weight"][valid]
    b0 = state["net.0.bias"][valid]
    margin = separation.own_margin(rows, W0, b0)
    own_active = margin > 0

    imgs, conf = separation.isolated_recovery(
        rows, config.IMG_SHAPE, _mlp_to_rgb(info.activation),
        sim_threshold=0.90, own_active=own_active, row_priority=margin,
    )
    method = f"cluster+margin({info.activation}):{imgs.shape[0]}"
    out = separation.diversify_fill(imgs, config.BATCH)
    return out, method


def recover_cnn(i: int, info: extract.ModelInfo):
    gr = info.grads
    if "fc1.weight" not in gr or "fc1.bias" not in gr:
        return separation.diversify_fill(torch.empty(0, *config.IMG_SHAPE),
                                         config.BATCH), "no-fc1"
    gW, gb = gr["fc1.weight"], gr["fc1.bias"]
    rows, valid = separation.analytic_rows(gW, gb)
    if rows.shape[0] == 0:
        return separation.diversify_fill(torch.empty(0, *config.IMG_SHAPE),
                                         config.BATCH), "empty"

    # Resolve the fc1-input feature map shape (trust feature_shape, else infer).
    fc, fh, fw = info.feature_shape
    if fc * fh * fw != rows.shape[1]:
        fc = gr["conv.weight"].shape[0] if "conv.weight" in gr else 1
        fh, fw = utils.infer_square(rows.shape[1], fc)
        if fc * fh * fw != rows.shape[1]:
            return separation.diversify_fill(torch.empty(0, *config.IMG_SHAPE),
                                             config.BATCH), "reshape-fail"

    to_rgb = _cnn_to_rgb(i, (fc, fh, fw))
    st = utils.load_state(i)
    # fc1 own-margin: feed each recovered fc1-input row back through the KNOWN
    # fc1 weights; isolated rows reactivate their neuron strongly.
    fcW = st["fc1.weight"][valid]
    fcb = st["fc1.bias"][valid]
    margin = separation.own_margin(rows, fcW, fcb)
    imgs, conf = separation.isolated_recovery(
        rows, (fc, fh, fw), to_rgb, sim_threshold=0.90, row_priority=margin,
    )
    tm = channels.analyze_conv(st["conv.weight"], st.get("conv.bias"))
    tag = "transmit" if tm.is_transmit() else "avg"
    method = f"cluster+margin/{tag}({info.activation}):{imgs.shape[0]}"

    out = separation.diversify_fill(imgs, config.BATCH)
    return out, method


def recover_model(i: int):
    grad = utils.load_gradient(i)
    info = extract.introspect(i, grad)
    if info.family == "mlp":
        imgs, method = recover_mlp(i, info)
    elif info.family == "cnn":
        imgs, method = recover_cnn(i, info)
    else:  # vit: no clean analytic path here; leave to run.py --optimize / prior
        imgs = separation.diversify_fill(torch.empty(0, *config.IMG_SHAPE),
                                         config.BATCH)
        method = "vit-prior(fill)"
    imgs = imgs.clamp(0, 1).float()
    print(f"model{i:2d} | {info.family:3s}/{info.activation:7s} -> {method}")
    return imgs, method


# --------------------------------------------------------------------------- #
# Diagnostics (no ground truth, no leaderboard)
# --------------------------------------------------------------------------- #
def _save_grid(imgs: torch.Tensor, path: str, cols: int = 8) -> None:
    import math
    t = imgs.clamp(0, 1).float().cpu()
    n = t.shape[0]
    rows = math.ceil(n / cols)
    _, c, h, w = t.shape
    grid = torch.zeros(3, rows * h, cols * w)
    for idx in range(n):
        r, cc = divmod(idx, cols)
        grid[:, r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t[idx]
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        import numpy as np
        np.save(path.replace(".png", ".npy"), arr)


def diagnose(models: list[int], outdir: str, n_preview: int = 64) -> None:
    os.makedirs(outdir, exist_ok=True)
    print(f"{'model':>5} | {'fam':3} | {'act':7} | {'valid':>5} | "
          f"{'clusters':>8} | {'conf>.5':>7} | {'transmit':>8}")
    print("-" * 74)
    for i in models:
        grad = utils.load_gradient(i)
        info = extract.introspect(i, grad)
        gr = info.grads
        transmit = "-"
        try:
            if info.family == "mlp":
                gW, gb = gr["net.0.weight"], gr["net.0.bias"]
                rows, valid = separation.analytic_rows(gW, gb)
                st = utils.load_state(i)
                W0, b0 = st["net.0.weight"][valid], st["net.0.bias"][valid]
                margin = separation.own_margin(rows, W0, b0)
                imgs, conf = separation.isolated_recovery(
                    rows, config.IMG_SHAPE, _mlp_to_rgb(info.activation),
                    row_priority=margin)
            elif info.family == "cnn" and "fc1.weight" in gr:
                gW, gb = gr["fc1.weight"], gr["fc1.bias"]
                rows, valid = separation.analytic_rows(gW, gb)
                fc, fh, fw = info.feature_shape
                if fc * fh * fw != rows.shape[1]:
                    fc = gr["conv.weight"].shape[0]
                    fh, fw = utils.infer_square(rows.shape[1], fc)
                st = utils.load_state(i)
                margin = separation.own_margin(
                    rows, st["fc1.weight"][valid], st["fc1.bias"][valid])
                imgs, conf = separation.isolated_recovery(
                    rows, (fc, fh, fw), _cnn_to_rgb(i, (fc, fh, fw)),
                    row_priority=margin)
                tm = channels.analyze_conv(st["conv.weight"], st.get("conv.bias"))
                transmit = f"{tm.strength:.2f}{'*' if tm.is_transmit() else ''}"
            else:
                print(f"{i:>5} | {info.family:3} | {info.activation:7} | "
                      f"{'-':>5} | {'-':>8} | {'-':>7} | {transmit:>8}")
                continue
            n_valid = int(valid.sum())
            n_clu = imgs.shape[0]
            hi = int((conf > 0.5).sum())
            print(f"{i:>5} | {info.family:3} | {info.activation:7} | "
                  f"{n_valid:>5} | {n_clu:>8} | {hi:>7} | {transmit:>8}")
            _save_grid(imgs[:n_preview], os.path.join(outdir, f"v2_model{i}.png"))
        except Exception as e:
            print(f"{i:>5} | ERROR: {type(e).__name__}: {e}")
    print(f"\npreviews -> {outdir}/v2_model*.png "
          f"(high clusters + high conf>.5 = strongly recoverable)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=list(range(1, 13)))
    ap.add_argument("--out", type=str, default="submission_v2.pt")
    ap.add_argument("--base", type=str, default=None,
                    help="existing submission to update in place")
    ap.add_argument("--diagnose", action="store_true",
                    help="no build: print recoverability table + save previews")
    ap.add_argument("--outdir", type=str, default="output/v2_diag")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)

    if args.diagnose:
        diagnose(args.models, args.outdir)
        return

    if args.base and os.path.exists(args.base):
        submission = torch.load(args.base, weights_only=False)
        print(f"[v2] loaded base {args.base}")
    else:
        submission = {f"model{i}": torch.rand(config.BATCH, *config.IMG_SHAPE)
                      for i in range(1, config.NUM_MODELS + 1)}

    for i in args.models:
        imgs, _ = recover_model(i)
        submission[f"model{i}"] = imgs

    path = utils.save_submission(submission, args.out)
    print(f"[v2] wrote {path} (validated OK)")


if __name__ == "__main__":
    main()
