"""Orchestrate reconstruction of all 12 models -> submission.pt.

Strategy (per model):
  1. Analytic extraction (eq. 6) -> candidate images. Cheap, no GPU.
  2. Label-free quality scoring + dedup -> exactly 128 distinct images.
  3. (Optional, --optimize) Geiping gradient matching (MLP/CNN/ViT),
     warm-started from the analytic candidates.
  4. Anti-regression selection: keep whichever of {analytic, optimized}
     reproduces the *observed* gradient better (label-free, never touches the
     leaderboard, so it cannot cause public-split overfitting).

The run *always* produces a valid submission (falls back to noise on failure)
so you can submit a reference early and improve incrementally.

Usage:
  python run.py                              # analytic only, all models
  python run.py --models 5 8                 # just a couple
  python run.py --optimize --models 9 11 --steps 6000   # GPU optimization
  python run.py --optimize --models 6 --save-part output/parts   # array job
  python run.py --out submission.pt
"""
from __future__ import annotations

import argparse
import os

import torch

import config
import extract
import rebuild
import utils


def _build_model(i: int, info: extract.ModelInfo):
    """Rebuild a differentiable forward model for family. May raise (ViT)."""
    state = utils.load_state(i)
    if info.family == "mlp":
        return rebuild.build_mlp(state, info.activation)
    if info.family == "cnn":
        return rebuild.build_cnn(state, info.activation, info.feature_shape)
    if info.family == "vit":
        return rebuild.build_vit(state)
    raise ValueError(f"unknown family {info.family}")


def _optimize(i, grad, info, init, steps):
    """Return (imgs, model, labels) or (None, None, None) if not possible."""
    model = _build_model(i, info)
    labels = rebuild.infer_labels(grad["gradients"])
    imgs = rebuild.gradient_match(
        model, grad["gradients"], labels,
        n_images=config.BATCH, steps=steps, init=init,
    )
    return imgs, model, labels


def reconstruct_model(i: int, use_opt: bool, steps: int) -> tuple[torch.Tensor, str]:
    grad = utils.load_gradient(i)
    info = extract.introspect(i, grad)

    cands = extract.extract_analytic(info)
    n_c = cands.shape[0]

    method = f"analytic({info.tier})"
    analytic_imgs = None
    if n_c > 0:
        scores = utils.quality_score(cands)
        analytic_imgs = utils.dedup_select(cands, scores, k=config.BATCH)

    imgs = analytic_imgs

    if use_opt:
        try:
            # Warm-start from the deduped top-128 analytic images when we have
            # them (better than raw candidate order); ViT starts from noise.
            init = analytic_imgs if analytic_imgs is not None else None
            opt_imgs, model, labels = _optimize(i, grad, info, init, steps)

            # Anti-regression selection via observed-gradient reproduction.
            if analytic_imgs is not None:
                d_analytic = rebuild.resim_distance(
                    model, analytic_imgs, labels, grad["gradients"])
                d_opt = rebuild.resim_distance(
                    model, opt_imgs, labels, grad["gradients"])
                print(f"  [select] model{i}: analytic={d_analytic:.4f} "
                      f"optimized={d_opt:.4f}")
                if d_opt <= d_analytic:
                    imgs, method = opt_imgs, f"optimize({info.family},{steps})"
                else:
                    imgs, method = analytic_imgs, f"analytic({info.tier})+kept"
            else:
                imgs, method = opt_imgs, f"optimize({info.family},{steps})"
        except Exception as e:
            print(f"  [run] model{i}: optimize skipped ({type(e).__name__}: {e})")

    if imgs is None or imgs.shape[0] == 0:
        imgs = torch.rand(config.BATCH, *config.IMG_SHAPE)
        method = "prior(noise)"

    imgs = utils.to_unit(imgs).float()
    print(f"model{i:2d} | {info.family:3s}/{info.activation:7s} | "
          f"cands={n_c:4d} | -> {method}")
    return imgs, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=list(range(1, 13)))
    ap.add_argument("--optimize", action="store_true", help="run gradient matching")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", type=str, default=config.SUBMISSION_PATH)
    ap.add_argument("--base", type=str, default=None,
                    help="existing submission.pt to update in place")
    ap.add_argument("--save-part", type=str, default=None,
                    help="dir to write per-model tensors model{i}.pt (array jobs)")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)

    if args.base and os.path.exists(args.base):
        submission = torch.load(args.base, weights_only=False)
        print(f"[run] loaded base submission {args.base}")
    else:
        submission = {f"model{i}": torch.rand(config.BATCH, *config.IMG_SHAPE)
                      for i in range(1, 13)}

    if args.save_part:
        os.makedirs(args.save_part, exist_ok=True)

    for i in args.models:
        imgs, _ = reconstruct_model(i, args.optimize, args.steps)
        submission[f"model{i}"] = imgs
        if args.save_part:
            part = os.path.join(args.save_part, f"model{i}.pt")
            torch.save(imgs, part)
            print(f"[run] wrote part {part}")

    if not args.save_part:
        path = utils.save_submission(submission, args.out)
        print(f"[run] wrote {path}  (validated OK)")


if __name__ == "__main__":
    main()
