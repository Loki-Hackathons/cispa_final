"""Assemble submission.pt from per-model parts produced by array jobs.

Each array task writes output/parts/model{i}.pt (a 128x3x64x64 tensor). This
merges whatever parts exist into a single validated submission, falling back to
a base submission (or diverse noise) for any model whose part is missing. Safe
to re-run as parts land, so you can submit incrementally.

Usage:
  python merge.py --parts output/parts --base submission.pt --out submission.pt
"""
from __future__ import annotations

import argparse
import os

import torch

import config
import utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=str, default="output/parts")
    ap.add_argument("--base", type=str, default=None,
                    help="fallback submission for missing parts")
    ap.add_argument("--out", type=str, default=config.SUBMISSION_PATH)
    args = ap.parse_args()

    if args.base and os.path.exists(args.base):
        base = torch.load(args.base, weights_only=False)
        print(f"[merge] base submission: {args.base}")
    else:
        base = {}

    submission = {}
    n_from_part = 0
    for i in range(1, config.NUM_MODELS + 1):
        key = f"model{i}"
        part = os.path.join(args.parts, f"{key}.pt")
        if os.path.exists(part):
            t = torch.load(part, weights_only=False)
            submission[key] = utils.to_unit(t).float()
            n_from_part += 1
            src = "part"
        elif key in base:
            submission[key] = base[key]
            src = "base"
        else:
            submission[key] = torch.rand(config.BATCH, *config.IMG_SHAPE)
            src = "noise"
        print(f"  {key}: {src}")

    path = utils.save_submission(submission, args.out)
    print(f"[merge] {n_from_part}/{config.NUM_MODELS} from parts -> {path} (validated OK)")


if __name__ == "__main__":
    main()
