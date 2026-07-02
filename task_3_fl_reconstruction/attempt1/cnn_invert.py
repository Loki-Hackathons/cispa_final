"""Invert CNN analytic feature maps back to RGB pixels.

Why this exists:
  The current CNN "analytic" path extracts the input of `fc1`, i.e. an
  (8,H,W) feature map. We then collapse those 8 channels to RGB by averaging
  groups. That is a crude visualization, not an inverse model.

This script keeps the strong analytic fc1 extraction, then solves:

    activation(conv(x)) ~= reconstructed_feature_map

using only the *known first conv* and the reconstructed fc1 features. This is
much less speculative than full-model gradient matching: we do not invent the
classifier, labels, or loss. For lower-resolution feature maps, we match an
adaptive average pooled conv output, which is the only remaining mild guess.

Usage on cluster:
  python run.py --out submission_analytic.pt
  python cnn_invert.py --base submission_analytic.pt --out submission_cnninv.pt \
      --models 2 3 6 7 10 12 --steps 1500
  python submit.py --check submission_cnninv.pt
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

import config
import utils


def activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"unsupported CNN activation {name}")


def analytic_feature_candidates(i: int):
    """Return selected raw fc1-input feature maps for CNN model i.

    Selection uses the existing visualization quality score only to pick 128
    promising candidates. The optimization target remains the raw feature map,
    not the normalized RGB preview.
    """
    grad = utils.load_gradient(i)
    gr = grad["gradients"]
    gW, gb = gr["fc1.weight"], gr["fc1.bias"]
    valid = gb.abs() > config.EPS
    rows = gW[valid] / gb[valid].unsqueeze(1)
    if rows.shape[0] == 0:
        raise RuntimeError(f"model{i}: no fc1 analytic rows")

    c, h, w = tuple(grad["feature_shape"])
    if c * h * w != rows.shape[1]:
        # Fall back to conv out-channels + square factorization.
        c = gr["conv.weight"].shape[0]
        h, w = utils.infer_square(rows.shape[1], c)
        if c * h * w != rows.shape[1]:
            raise RuntimeError(
                f"model{i}: cannot reshape fc1 rows {rows.shape[1]} into feature map"
            )

    feats = rows.reshape(rows.shape[0], c, h, w).float()
    preview = utils.features_to_image(rows, c, h, w)
    scores = utils.quality_score(preview)

    # Reproduce dedup_select but keep indices so raw features and previews align.
    fp = utils._fingerprints(preview)
    order = torch.argsort(scores, descending=True)
    chosen, chosen_fp = [], []
    for idx in order.tolist():
        if len(chosen) >= config.BATCH:
            break
        if chosen_fp:
            sims = torch.stack(chosen_fp) @ fp[idx]
            if float(sims.max()) > config.DEDUP_SIM_THRESHOLD:
                continue
        chosen.append(idx)
        chosen_fp.append(fp[idx])
    if len(chosen) < config.BATCH:
        chosen_set = set(chosen)
        for idx in order.tolist():
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            if len(chosen) >= config.BATCH:
                break
    idx = torch.tensor(chosen[:config.BATCH], dtype=torch.long)
    return grad, feats[idx], preview[idx]


def conv_features(x: torch.Tensor, state: dict, act: str, hw: tuple[int, int]):
    w = state["conv.weight"].to(x.device)
    b = state["conv.bias"].to(x.device)
    kh, kw = w.shape[-2:]
    y = F.conv2d(x, w, b, padding=(kh // 2, kw // 2))
    y = activation(y, act)
    if y.shape[-2:] != hw:
        y = F.adaptive_avg_pool2d(y, hw)
    return y


def invert_one_model(i: int, init: torch.Tensor, steps: int, lr: float,
                     tv_weight: float, device: str):
    grad, target, preview = analytic_feature_candidates(i)
    state = utils.load_state(i)
    act = grad["activation"]
    target = target.to(device)
    hw = target.shape[-2:]

    if init is None:
        x = preview.clone()
    else:
        x = init.clone()
    x = utils.to_unit(x).to(device).requires_grad_(True)

    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = x.detach().clone()
    best_loss = float("inf")

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = conv_features(x, state, act, hw)
        feat_loss = F.mse_loss(pred, target)
        # Use a light TV prior. Too much TV erases texture and hurts SSIM.
        tv = utils.total_variation(x).mean()
        loss = feat_loss + tv_weight * tv
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
            val = float(feat_loss.detach())
            if val < best_loss:
                best_loss = val
                best = x.detach().clone()
        if step % 250 == 0:
            print(f"model{i:2d} step {step:4d} feature_mse={val:.6f} best={best_loss:.6f}")

    out = best.detach().cpu().float()
    print(f"model{i:2d}: cnn inversion done best_feature_mse={best_loss:.6f} "
          f"quality={float(utils.quality_score(out).mean()):.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=[2, 3, 6, 7, 10, 12])
    ap.add_argument("--base", type=str, default="submission_analytic.pt")
    ap.add_argument("--out", type=str, default="submission_cnninv.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--tv", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.SEED)
    if os.path.exists(args.base):
        submission = torch.load(args.base, weights_only=False)
        print(f"[cnn_invert] loaded base {args.base}")
    else:
        submission = {f"model{i}": torch.rand(config.BATCH, *config.IMG_SHAPE)
                      for i in range(1, config.NUM_MODELS + 1)}
        print("[cnn_invert] no base found; using noise for non-CNN models")

    print(f"[cnn_invert] device={device} models={args.models}")
    for i in args.models:
        init = submission.get(f"model{i}")
        submission[f"model{i}"] = invert_one_model(
            i, init=init, steps=args.steps, lr=args.lr,
            tv_weight=args.tv, device=device,
        )

    path = utils.save_submission(submission, args.out)
    print(f"[cnn_invert] wrote {path} (validated OK)")


if __name__ == "__main__":
    main()
