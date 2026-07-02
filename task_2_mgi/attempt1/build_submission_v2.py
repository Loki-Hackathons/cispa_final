#!/usr/bin/env python3
"""Per-direction MGI submission builder.

The score is mean(DetectorScore * (1 - MSE)) over 1800 slots, split into six
independent 300-slot directions (M_N, M_G, N_M, N_G, G_M, G_N). Because the
directions are independent, we can choose a *different* generation method per
direction and keep, cumulatively, whichever method gave the best API score for
each one.

Methods (all numpy/PIL only unless noted):
  swap            nearest target-class image (pixel-L2) -> genuine flip, high MSE
  ref             raw source reference (MSE=0, no flip) -> isolation/control only
  noise:S         source reference + Gaussian noise, std S on the 0-255 scale
  jpeg:Q          source reference JPEG round-tripped at quality Q
  blur:R          source reference Gaussian-blurred, radius R (PIL)
  shift:P         source reference rolled by P px (breaks exact memorization)
  block:PATH      load a precomputed (300,256,256,3) uint8 .npy block
                  (use this to plug in GPU results, e.g. RAR reconstructions)

Rationale per axis (the hidden detector has a membership head + a generation
head, per the task spec):
  * Remove membership (M->N):   tiny perturbation of the member image usually
    breaks the memorization signal at near-zero MSE.  Try noise/jpeg/shift.
  * Add generation fingerprint (*->G): pass the reference through RAR's
    autoencoder (reconstruct) -> lands on the model manifold, low MSE.
    Produce that block with a GPU script, then plug via block:PATH.
  * Add membership (*->M):       hard; keep `swap` as the safe fallback.

Usage
-----
  # rebuild the proven all-swap baseline (== build_content_swap.py):
  python task_2_mgi/attempt1/build_submission_v2.py

  # isolation diagnostic: only M_N uses a tiny perturbation, rest stay swap:
  python task_2_mgi/attempt1/build_submission_v2.py --dir M_N=noise:8

  # combine several proven directions + a precomputed reconstruction block:
  python task_2_mgi/attempt1/build_submission_v2.py \
      --dir M_N=jpeg:35 --dir M_G=block:output/blocks/M_G_recon.npy
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_SIZE, IMAGE_SIZE, PathsConfig, TOTAL_IMAGES  # noqa: E402
from submission_io import (  # noqa: E402
    API_MAX_BYTES,
    load_reference_images,
    save_submission_npz,
)

CLASS_RANGES = {
    "M": (0, CLASS_SIZE),
    "N": (CLASS_SIZE, 2 * CLASS_SIZE),
    "G": (2 * CLASS_SIZE, 3 * CLASS_SIZE),
}

# (name, submission_slot_start, source_class, target_class)
DIRECTIONS = [
    ("M_N", 0, "M", "N"),
    ("M_G", 300, "M", "G"),
    ("N_M", 600, "N", "M"),
    ("N_G", 900, "N", "G"),
    ("G_M", 1200, "G", "M"),
    ("G_N", 1500, "G", "N"),
]
DIR_BY_NAME = {d[0]: d for d in DIRECTIONS}


def nearest_target_indices(refs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    r = refs.reshape(len(refs), -1).astype(np.float32)
    t = targets.reshape(len(targets), -1).astype(np.float32)
    r2 = (r * r).sum(axis=1, keepdims=True)
    t2 = (t * t).sum(axis=1, keepdims=True).T
    d = r2 + t2 - 2.0 * (r @ t.T)
    return d.argmin(axis=1)


def method_swap(refs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = nearest_target_indices(refs, targets)
    return targets[idx].copy()


def method_noise(refs: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, std, size=refs.shape)
    out = refs.astype(np.float32) + noise
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def method_jpeg(refs: np.ndarray, quality: int) -> np.ndarray:
    out = np.empty_like(refs)
    for i in range(len(refs)):
        buf = io.BytesIO()
        Image.fromarray(refs[i]).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return out


def method_blur(refs: np.ndarray, radius: float) -> np.ndarray:
    out = np.empty_like(refs)
    for i in range(len(refs)):
        im = Image.fromarray(refs[i]).filter(ImageFilter.GaussianBlur(radius))
        out[i] = np.asarray(im, dtype=np.uint8)
    return out


def method_shift(refs: np.ndarray, px: int) -> np.ndarray:
    return np.roll(refs, shift=(px, px), axis=(1, 2))


def method_block(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.shape != (CLASS_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"block {path} has shape {arr.shape}, expected "
                         f"{(CLASS_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)}")
    return arr.astype(np.uint8)


def build_block(spec: str, refs: np.ndarray, targets: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """spec like 'swap', 'noise:8', 'jpeg:35', 'blur:1.2', 'shift:4',
    'ref', 'block:path/to.npy'."""
    if ":" in spec:
        name, arg = spec.split(":", 1)
    else:
        name, arg = spec, None

    if name == "swap":
        return method_swap(refs, targets)
    if name == "ref":
        return refs.copy()
    if name == "noise":
        return method_noise(refs, float(arg), rng)
    if name == "jpeg":
        return method_jpeg(refs, int(arg))
    if name == "blur":
        return method_blur(refs, float(arg))
    if name == "shift":
        return method_shift(refs, int(arg))
    if name == "block":
        return method_block(Path(arg))
    raise ValueError(f"unknown method spec: {spec!r}")


def main() -> int:
    p = argparse.ArgumentParser(description="Per-direction MGI submission builder")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--dir",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help="Override a direction's method, e.g. M_N=noise:8. Repeatable. "
             "Unspecified directions default to 'swap'.",
    )
    p.add_argument("--default", default="swap",
                   help="Method for directions not overridden (default: swap)")
    p.add_argument("--jpeg-quality", type=int, default=None,
                   help="Force submission JPEG quality (skip auto-fit search)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    overrides: dict[str, str] = {}
    for item in args.dir:
        if "=" not in item:
            print(f"ERROR: --dir expects NAME=SPEC, got {item!r}", file=sys.stderr)
            return 1
        name, spec = item.split("=", 1)
        if name not in DIR_BY_NAME:
            print(f"ERROR: unknown direction {name!r}. "
                  f"Choose from {list(DIR_BY_NAME)}", file=sys.stderr)
            return 1
        overrides[name] = spec

    rng = np.random.default_rng(args.seed)
    paths = PathsConfig()
    out = args.output or (paths.output_dir / "submission.npz")

    print(f"Loading reference images from {paths.data_dir}")
    originals = load_reference_images(paths.data_dir)

    submission = np.empty((TOTAL_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    per_dir = {}

    for name, slot_start, src_cls, tgt_cls in DIRECTIONS:
        spec = overrides.get(name, args.default)
        s0, s1 = CLASS_RANGES[src_cls]
        t0, t1 = CLASS_RANGES[tgt_cls]
        refs = originals[s0:s1]
        targets = originals[t0:t1]

        block = build_block(spec, refs, targets, rng)
        submission[slot_start:slot_start + CLASS_SIZE] = block

        diff = block.astype(np.float32) - refs.astype(np.float32)
        mse_norm = float(np.mean((diff / 255.0) ** 2))
        per_dir[name] = (spec, mse_norm)
        tag = "OVERRIDE" if name in overrides else "default"
        print(f"  {name:4s} [{tag:8s}] {spec:22s} "
              f"mse_norm={mse_norm:.5f}  (1-mse)={1.0 - mse_norm:.5f}")

    est = float(np.mean([1.0 - m for _, m in per_dir.values()]))
    print(f"\nUpper-bound score IF every slot flips: ~{est:.4f} "
          "(real score = flip_rate-weighted, MSE as computed by grader)")

    size = save_submission_npz(
        submission, out, jpeg_quality=args.jpeg_quality, enforce_api_limit=True,
    )
    print(f"OK — {out} ({size / 1e6:.2f} MB, limit {API_MAX_BYTES / 1e6:.0f} MB)")
    return 0 if size <= API_MAX_BYTES else 1


if __name__ == "__main__":
    raise SystemExit(main())
