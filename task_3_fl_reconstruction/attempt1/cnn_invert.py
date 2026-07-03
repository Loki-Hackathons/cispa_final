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
import fc1_analytic
import reconstruct_v2
import separation
import utils


def activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"unsupported CNN activation {name}")


def analytic_feature_candidates(
    i: int,
    use_confidence: bool = True,
    confidence_weight: float = 0.55,
):
    """Return selected raw fc1-input feature maps for CNN model i.

  By default ranks candidates with isolation confidence (fc1_analytic) plus
  the legacy quality proxy, then dedups to 128 rows. The optimization target
  remains the raw feature map, not the RGB preview.
    """
    cands = fc1_analytic.extract_fc1_candidates(i)
    idx = fc1_analytic.select_fc1_candidates(
        cands,
        use_confidence=use_confidence,
        confidence_weight=confidence_weight,
    )
    grad = utils.load_gradient(i)
    if use_confidence:
        rep = fc1_analytic.model_confidence_report(cands)
        print(
            f"model{i:2d}: fc1 conf all={rep['conf_mean_all']:.3f} "
            f"sel={rep['conf_mean_selected']:.3f} "
            f"high_frac={rep['conf_high_frac']:.2f}"
        )
    return grad, cands.feats[idx], cands.preview[idx]


def analytic_feature_candidates_v2(i: int):
    """v2 target selection: cluster fc1 analytic rows (separation.py) instead of
    the legacy `fc1_analytic` per-row heuristic.

    `bench_selection.py` shows `own_margin` + clustering (isolated_recovery) is
    the best label-free row selector we have; using its DENOISED raw feature-map
    representatives (averaged/cleanest member per cluster) as the optimization
    target, instead of raw un-clustered fc1 rows, should give `invert_one_model`
    a cleaner target to fit. Returns (grad, feats, preview) like the legacy
    `analytic_feature_candidates`, but with at most one row per distinct image
    (K <= 128, not padded — callers pad the resulting RGB afterwards).
    """
    grad = utils.load_gradient(i)
    gr = grad["gradients"]
    gW, gb = gr["fc1.weight"], gr["fc1.bias"]
    rows, valid = separation.analytic_rows(gW, gb)
    if rows.shape[0] == 0:
        raise RuntimeError(f"model{i}: no fc1 analytic rows")

    fc, fh, fw = tuple(grad["feature_shape"])
    if fc * fh * fw != rows.shape[1]:
        fc = gr["conv.weight"].shape[0] if "conv.weight" in gr else 1
        fh, fw = utils.infer_square(rows.shape[1], fc)
        if fc * fh * fw != rows.shape[1]:
            raise RuntimeError(f"model{i}: cannot reshape fc1 rows ({rows.shape[1]}) "
                               f"into a ({fc},?,?) feature map")

    state = utils.load_state(i)
    fcW, fcb = state["fc1.weight"][valid], state["fc1.bias"][valid]
    margin = separation.own_margin(rows, fcW, fcb)
    # to_rgb is only used here for clustering fingerprints / quality tie-break,
    # never as the optimization target — cnn_invert fits pixels against the
    # RAW feature map below, which sidesteps the transmit-filter closed form
    # entirely (and thus the model12 sigmoid color-inversion bug).
    to_rgb = reconstruct_v2._cnn_to_rgb(i, (fc, fh, fw))
    _, _, raw = separation.isolated_recovery(
        rows, (fc, fh, fw), to_rgb, sim_threshold=0.90, row_priority=margin,
        return_raw=True,
    )
    feats = raw.reshape(raw.shape[0], fc, fh, fw)
    preview = utils.flat_to_image(raw, (fc, fh, fw))
    return grad, feats, preview


def conv_features(x: torch.Tensor, state: dict, act: str, hw: tuple[int, int]):
    w = state["conv.weight"].to(x.device)
    b = state["conv.bias"].to(x.device)
    kh, kw = w.shape[-2:]
    y = F.conv2d(x, w, b, padding=(kh // 2, kw // 2))
    y = activation(y, act)
    if y.shape[-2:] != hw:
        y = F.adaptive_avg_pool2d(y, hw)
    return y


def _tv_lr_for_shape(hw: tuple[int, int], lr: float, tv_weight: float) -> tuple[float, float]:
    """Scale TV/lr by feature resolution (8x8 needs stronger TV, 64x64 weaker)."""
    side = max(hw)
    if side >= 64:
        return lr, tv_weight * 0.35
    if side >= 16:
        return lr, tv_weight
    return lr * 1.15, tv_weight * 2.5


def invert_one_model(i: int, init: torch.Tensor, steps: int, lr: float,
                     tv_weight: float, device: str,
                     use_confidence: bool = True,
                     confidence_weight: float = 0.55,
                     selector: str = "legacy"):
    if selector == "v2":
        grad, target, preview = analytic_feature_candidates_v2(i)
        print(f"model{i:2d}: v2 selection -> {target.shape[0]} distinct targets "
              f"(clustered, own_margin)")
    else:
        grad, target, preview = analytic_feature_candidates(
            i,
            use_confidence=use_confidence,
            confidence_weight=confidence_weight,
        )
    state = utils.load_state(i)
    act = grad["activation"]
    target = target.to(device)
    hw = target.shape[-2:]
    lr, tv_weight = _tv_lr_for_shape(hw, lr, tv_weight)
    print(f"model{i:2d}: hw={hw} lr={lr:.4f} tv={tv_weight:.2e}")

    n_targets = target.shape[0]
    if init is None:
        x = preview.clone()
    else:
        # init may come from a 128-image base submission while the v2
        # selector can yield fewer than 128 distinct clustered targets.
        x = init[:n_targets].clone()
        if x.shape[0] < n_targets:
            x = torch.cat([x, preview[x.shape[0]:n_targets].clone()], dim=0)
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
    if out.shape[0] < config.BATCH:
        # v2 selector gives K <= 128 distinct clusters; pad with augmented
        # variants of the real optimized images (never noise) like the rest
        # of the v2 pipeline (separation.keep_structured_and_fill).
        out = separation.keep_structured_and_fill(out, config.BATCH)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=[2, 3, 6, 7, 10, 12])
    ap.add_argument("--base", type=str, default="submission_analytic.pt")
    ap.add_argument("--out", type=str, default="submission_cnninv.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--tv", type=float, default=1e-3)
    ap.add_argument("--no-confidence", action="store_true",
                    help="legacy selection: quality_score only")
    ap.add_argument("--confidence-weight", type=float, default=0.55)
    ap.add_argument("--selector", choices=["legacy", "v2"], default="v2",
                    help="target selection: legacy=fc1_analytic heuristic, "
                         "v2=separation.isolated_recovery (own_margin + "
                         "clustering, recommended: better on bench_selection.py)")
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

    print(f"[cnn_invert] device={device} models={args.models} selector={args.selector}")
    for i in args.models:
        init = submission.get(f"model{i}")
        submission[f"model{i}"] = invert_one_model(
            i, init=init, steps=args.steps, lr=args.lr,
            tv_weight=args.tv, device=device,
            use_confidence=not args.no_confidence,
            confidence_weight=args.confidence_weight,
            selector=args.selector,
        )

    path = utils.save_submission(submission, args.out)
    print(f"[cnn_invert] wrote {path} (validated OK)")


if __name__ == "__main__":
    main()
