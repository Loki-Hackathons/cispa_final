"""Validate and (optionally) upload a submission.

The strict validator is the important part: a rejected upload still burns a
2-minute cooldown, so we always check shape/dtype/range/keys locally first.

The actual HTTP upload must match the organizers' `task_template.py` (its
submission block was truncated in our copy). Until confirmed on the cluster,
prefer submitting through the official template — this script wires the API
key from `.env` so nothing secret is hardcoded.

Usage:
  python submit.py --check submission.pt          # validate only
  python submit.py --check submission.pt --send    # validate + upload (see TODO)
"""
from __future__ import annotations

import argparse
import os

import torch

import config
import utils


def load_env(env_path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) into os.environ."""
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def report(submission: dict) -> None:
    for i in range(1, config.NUM_MODELS + 1):
        x = submission[f"model{i}"]
        q = utils.quality_score(x).mean()
        print(f"  model{i:2d}: mean_quality={float(q):.4f} "
              f"range=[{float(x.min()):.3f},{float(x.max()):.3f}]")


def upload(path: str) -> None:
    """Upload via the API. Endpoint/format must match task_template.py.

    We know from the task: single .pt file, team API key required, ~5 min
    cooldown. Confirm the exact URL/fields against the cluster template before
    relying on this; otherwise submit through task_template.py directly.
    """
    import requests  # noqa: F401  (only needed when actually sending)

    api_key = os.environ.get("CISPA_API_KEY")
    base_url = os.environ.get("CISPA_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("Set CISPA_API_KEY and CISPA_BASE_URL (source .env).")

    # TODO(cluster): align endpoint + field names with task_template.py tail.
    raise SystemExit(
        "Automatic upload not wired to the confirmed endpoint yet.\n"
        "Use the official task_template.py with SUBMIT=True and\n"
        f"API_KEY from env (key prefix {api_key[:6]}...), FILE_PATH={path}."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=str, default=config.SUBMISSION_PATH)
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--env", type=str, default=".env")
    args = ap.parse_args()

    load_env(args.env)
    submission = torch.load(args.check, weights_only=False)
    utils.validate_submission(submission)
    print(f"[submit] {args.check}: format VALID (12 keys, 128x3x64x64, float32, [0,1])")
    report(submission)

    if args.send:
        upload(args.check)


if __name__ == "__main__":
    main()
