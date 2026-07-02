#!/usr/bin/env python3
"""Build baseline submission.npz (unmodified images) under 200 MB API limit.

No PyTorch required — fast to run on the login node.

Usage:
  python build_quick_submission.py
  python build_quick_submission.py --output output/submission_baseline.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import PathsConfig  # noqa: E402
from submission_io import (  # noqa: E402
    API_MAX_BYTES,
    build_submission,
    load_reference_images,
    save_submission_npz,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Build baseline MGI submission npz")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output npz path (default: output/submission_baseline.npz)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help=(
            "Force a single JPEG quality pass (faster). "
            "Example: --jpeg-quality 80"
        ),
    )
    args = p.parse_args()

    paths = PathsConfig()
    out = args.output or (paths.output_dir / "submission_baseline.npz")

    print(f"Loading images from {paths.data_dir}")
    originals = load_reference_images(paths.data_dir)
    submission = build_submission(originals, {})
    size = save_submission_npz(
        submission,
        out,
        jpeg_quality=args.jpeg_quality,
        enforce_api_limit=True,
    )

    print(f"OK — {out} ({size} bytes, limit {API_MAX_BYTES})")
    if size > API_MAX_BYTES:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
