"""Assemble submission.pt from per-model parts produced by array jobs.

Each array task writes output/parts/model{i}.pt (a 128x3x64x64 tensor). This
merges whatever parts exist into a single validated submission, falling back to
a base submission (or diverse noise) for any model whose part is missing. Safe
to re-run as parts land, so you can submit incrementally.

``--from-submission`` additionally lets you patch specific models straight out
of another *full* submission.pt (not per-model parts) — e.g. pull the proven
legacy ViT reconstructions (model9/model11 from `sub_vit_both.pt`, 0.2469 on
the leaderboard) into the v2 base (`submission_v2.pt`, 0.255), which currently
fills those two models with plain noise. Takes precedence over --parts/--base
for the models listed in --from-models.

Usage:
  python merge.py --parts output/parts --base submission.pt --out submission.pt
  python merge.py --base submission_v2.pt \
      --from-submission sub_vit_both.pt --from-models 9 11 \
      --out submission_v3.pt
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
    ap.add_argument("--from-submission", type=str, default=None,
                    help="another full submission.pt to pull specific models from")
    ap.add_argument("--from-models", type=int, nargs="*", default=[],
                    help="model ids to take from --from-submission (e.g. 9 11)")
    ap.add_argument("--out", type=str, default=config.SUBMISSION_PATH)
    args = ap.parse_args()

    if args.base and os.path.exists(args.base):
        base = torch.load(args.base, weights_only=False)
        print(f"[merge] base submission: {args.base}")
    else:
        base = {}

    patch = {}
    if args.from_submission:
        if not args.from_models:
            raise SystemExit("--from-submission given without --from-models")
        if not os.path.exists(args.from_submission):
            raise SystemExit(f"--from-submission not found: {args.from_submission}")
        donor = torch.load(args.from_submission, weights_only=False)
        for i in args.from_models:
            key = f"model{i}"
            if key not in donor:
                raise SystemExit(f"{key} missing in {args.from_submission}")
            patch[key] = utils.to_unit(donor[key]).float()
        print(f"[merge] patching {sorted(patch)} from {args.from_submission}")

    submission = {}
    n_from_part = 0
    n_from_patch = 0
    for i in range(1, config.NUM_MODELS + 1):
        key = f"model{i}"
        part = os.path.join(args.parts, f"{key}.pt")
        if key in patch:
            submission[key] = patch[key]
            n_from_patch += 1
            src = "from-submission"
        elif os.path.exists(part):
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
    print(f"[merge] {n_from_part}/{config.NUM_MODELS} from parts, "
          f"{n_from_patch} patched -> {path} (validated OK)")


if __name__ == "__main__":
    main()
