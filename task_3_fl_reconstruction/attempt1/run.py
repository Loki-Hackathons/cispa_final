"""Orchestrate reconstruction of all 12 models -> submission.pt.

Strategy (per model):
  1. Analytic extraction (eq. 6) -> candidate images. Cheap, no GPU.
  2. Label-free quality scoring + dedup -> exactly 128 distinct images.
  3. (Optional, --optimize) Geiping gradient matching for hard models,
     warm-started from the analytic candidates.

The run *always* produces a valid submission (falls back to noise on failure)
so you can submit a reference early and improve incrementally.

Usage:
  python run.py                        # analytic only, all models
  python run.py --models 5 8           # just a couple
  python run.py --optimize --models 9 11 --steps 4000   # GPU optimization
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


def reconstruct_model(i: int, use_opt: bool, steps: int) -> tuple[torch.Tensor, str]:
    grad = utils.load_gradient(i)
    info = extract.introspect(i, grad)

    cands = extract.extract_analytic(info)
    n_c = cands.shape[0]

    method = f"analytic({info.tier})"
    imgs = None

    if n_c > 0:
        scores = utils.quality_score(cands)
        imgs = utils.dedup_select(cands, scores, k=config.BATCH)

    # Optimization path (hard models, or explicit request).
    if use_opt and info.family == "mlp":
        try:
            state = utils.load_state(i)
            model = rebuild.build_mlp(state, info.activation)
            labels = rebuild.infer_labels(grad["gradients"])
            init = cands if n_c > 0 else None
            imgs = rebuild.gradient_match(
                model, grad["gradients"], labels,
                n_images=config.BATCH, steps=steps, init=init,
            )
            method = f"optimize(mlp,{steps})"
        except Exception as e:
            print(f"  [run] model{i}: optimize failed ({e}); keeping analytic")

    if imgs is None or imgs.shape[0] == 0:
        # Last resort: diverse noise (valid, non-zero-SSIM in expectation).
        imgs = torch.rand(config.BATCH, *config.IMG_SHAPE)
        method = "prior(noise)"

    imgs = utils.to_unit(imgs).float()
    print(f"model{i:2d} | {info.family:3s}/{info.activation:7s} | "
          f"cands={n_c:4d} | -> {method}")
    return imgs, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=list(range(1, 13)))
    ap.add_argument("--optimize", action="store_true", help="run gradient matching (MLP)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", type=str, default=config.SUBMISSION_PATH)
    ap.add_argument("--base", type=str, default=None,
                    help="existing submission.pt to update in place")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)

    if args.base and os.path.exists(args.base):
        submission = torch.load(args.base, weights_only=False)
        print(f"[run] loaded base submission {args.base}")
    else:
        submission = {f"model{i}": torch.rand(config.BATCH, *config.IMG_SHAPE)
                      for i in range(1, 13)}

    for i in args.models:
        imgs, _ = reconstruct_model(i, args.optimize, args.steps)
        submission[f"model{i}"] = imgs

    path = utils.save_submission(submission, args.out)
    print(f"[run] wrote {path}  (validated OK)")


if __name__ == "__main__":
    main()
