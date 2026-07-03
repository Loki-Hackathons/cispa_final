#!/usr/bin/env python3
"""Local proxy-DCB scorer for an MGI submission.npz.

We do not have the hidden detector, so we reconstruct *our* DCB and score any
submission offline:

  Stage 1 (autoencoder): L_A = L_R + alpha * L_Q on the MaskGIT VQ-GAN.
                         L_A <= tau_G  =>  predicted "G" (generated).
  Stage 2 (latent gen):  ICAS = nll_uncond - nll_cond on RAR, only for
                         non-generated images. ICAS >= tau_MN => "M" else "N".

For each of the 6 direction blocks we report the proxy flip-rate (predicted
class == target class), the mean normalised MSE vs the *source* reference, and
the estimated per-direction score mean(flip * (1 - MSE)). The global estimate is
the mean over all 1800 slots.

IMPORTANT: this measures transfer to our *proxy*, not the hidden detector. But
the Stage-1 VQ-GAN is the exact tokenizer RAR decodes with, so a Stage-1 proxy
flip is a strong predictor. Use this to gate API submissions:
  * only submit a direction whose proxy flip-rate is >= 0.9.

Run on a GPU node (Stage 2 needs RAR). Stage 1 alone works without RAR:
  export ONED_TOKENIZER_ROOT=/p/scratch/training2625/dougnon1/Loki/1d-tokenizer
  export PYTHONPATH=$ONED_TOKENIZER_ROOT:$PWD/task_2_mgi/attempt1
  python task_2_mgi/attempt1/evaluate_submission.py output/submission.npz
  python task_2_mgi/attempt1/evaluate_submission.py output/submission.npz --stage1-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    CLASS_SIZE,
    DIRECTION_BLOCKS,
    IMAGE_SIZE,
    TOTAL_IMAGES,
    PathsConfig,
    setup_oned_tokenizer_path,
)
from proxy_dcb import (  # noqa: E402
    calibrate_thresholds,
    compute_LA,
    load_calibration,
    load_vqgan,
    uint8_to_tensor,
)
from submission_io import load_reference_images  # noqa: E402


def _optimal_member_threshold(icas_m: np.ndarray, icas_n: np.ndarray) -> float:
    """Threshold on ICAS where members (higher) beat non-members (lower)."""
    candidates = np.unique(np.concatenate([icas_m, icas_n]))
    best_tau, best_acc = float(np.median(candidates)), -1.0
    for tau in candidates:
        tp = (icas_m >= tau).mean()
        tn = (icas_n < tau).mean()
        acc = 0.5 * (tp + tn)
        if acc > best_acc:
            best_acc, best_tau = acc, float(tau)
    return best_tau


def _load_submission(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        images = data["images"]
    if images.shape != (TOTAL_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"submission shape {images.shape} != expected "
                         f"{(TOTAL_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3)}")
    if images.dtype != np.uint8:
        raise ValueError(f"expected uint8 submission, got {images.dtype}")
    return images


def _batch_la(vqgan, images_uint8: np.ndarray, alpha: float, eps: float,
              device: str, batch_size: int) -> np.ndarray:
    out = np.empty(len(images_uint8), dtype=np.float32)
    for start in range(0, len(images_uint8), batch_size):
        chunk = images_uint8[start:start + batch_size]
        x = uint8_to_tensor(chunk, device)
        with torch.no_grad():
            la = compute_LA(vqgan, x, alpha=alpha, eps=eps, reduction="none")
        out[start:start + len(chunk)] = la.detach().cpu().numpy()
    return out


def _batch_icas(generator, tokenizer, classifier, images_uint8: np.ndarray,
                device: str, batch_size: int) -> np.ndarray:
    from proxy_icas import compute_membership_stats, predict_imagenet_class
    out = np.empty(len(images_uint8), dtype=np.float32)
    for start in range(0, len(images_uint8), batch_size):
        chunk = images_uint8[start:start + batch_size]
        x = uint8_to_tensor(chunk, device)
        with torch.no_grad():
            labels = predict_imagenet_class(x, classifier)
            _, _, icas = compute_membership_stats(generator, tokenizer, x, labels)
        out[start:start + len(chunk)] = icas.detach().cpu().numpy()
    return out


def _predict(la: np.ndarray, icas: np.ndarray | None, tau_g: float,
             tau_mn: float | None) -> np.ndarray:
    """Return array of predicted labels 'M'/'N'/'G' (or 'G'/'?' in stage1-only)."""
    labels = np.where(la <= tau_g, "G", "?").astype("<U1")
    if icas is None or tau_mn is None:
        return labels
    non_g = la > tau_g
    labels[non_g] = np.where(icas[non_g] >= tau_mn, "M", "N")
    return labels


def main() -> int:
    p = argparse.ArgumentParser(description="Score an MGI submission against proxy DCB")
    p.add_argument("submission", type=Path)
    p.add_argument("--stage1-only", action="store_true",
                   help="Skip RAR/Stage-2 (only generated-vs-natural is scored)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--calibration", type=Path, default=None,
                   help="calibration.json (else recalibrate on references)")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    paths = PathsConfig()
    setup_oned_tokenizer_path(paths.oned_tokenizer_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not paths.tokenizer_ckpt.is_file():
        print(f"ERROR: tokenizer not found: {paths.tokenizer_ckpt}", file=sys.stderr)
        return 1

    submission = _load_submission(args.submission)
    originals = load_reference_images(paths.data_dir)
    vqgan = load_vqgan(paths.tokenizer_ckpt, device=device)
    eps = 1e-6

    # --- thresholds -------------------------------------------------------- #
    cal_path = args.calibration or paths.calibration_path
    if cal_path and Path(cal_path).is_file():
        cal = load_calibration(cal_path)
        alpha, tau_g = cal["alpha"], cal["tau_G"]
        print(f"Loaded calibration {cal_path}: alpha={alpha:.4f} tau_G={tau_g:.6f}")
    else:
        print("No calibration file; recalibrating on reference M/N/G ...")
        cal = calibrate_thresholds(
            vqgan,
            uint8_to_tensor(originals[0:CLASS_SIZE], device),
            uint8_to_tensor(originals[CLASS_SIZE:2 * CLASS_SIZE], device),
            uint8_to_tensor(originals[2 * CLASS_SIZE:3 * CLASS_SIZE], device),
            eps=eps, batch_size=args.batch_size,
        )
        alpha, tau_g = cal["alpha"], cal["tau_G"]
        print(f"Calibrated: alpha={alpha:.4f} tau_G={tau_g:.6f}")

    generator = tokenizer = classifier = None
    tau_mn = None
    if not args.stage1_only:
        from proxy_icas import load_class_predictor, load_rar_generator
        tokenizer = vqgan
        generator, _ = load_rar_generator(tokenizer_ckpt=paths.tokenizer_ckpt, device=device)
        classifier = load_class_predictor(device=device)
        icas_m = _batch_icas(generator, tokenizer, classifier,
                             originals[0:CLASS_SIZE], device, args.batch_size)
        icas_n = _batch_icas(generator, tokenizer, classifier,
                             originals[CLASS_SIZE:2 * CLASS_SIZE], device, args.batch_size)
        tau_mn = _optimal_member_threshold(icas_m, icas_n)
        print(f"Stage-2 tau_MN={tau_mn:+.4f} "
              f"(ICAS ref: M mean={icas_m.mean():+.4f}, N mean={icas_n.mean():+.4f})")

    # --- per-direction scoring -------------------------------------------- #
    results: dict[str, dict] = {}
    per_slot_score = np.zeros(TOTAL_IMAGES, dtype=np.float64)

    for name, slot_start, source_start, count, _mode in DIRECTION_BLOCKS:
        target = name.split("_")[1]
        block = submission[slot_start:slot_start + count]
        refs = originals[source_start:source_start + count]

        la = _batch_la(vqgan, block, alpha, eps, device, args.batch_size)
        icas = None
        if not args.stage1_only:
            icas = _batch_icas(generator, tokenizer, classifier, block,
                               device, args.batch_size)
        pred = _predict(la, icas, tau_g, tau_mn)

        diff = block.astype(np.float32) - refs.astype(np.float32)
        mse = np.mean((diff / 255.0) ** 2, axis=(1, 2, 3))

        if args.stage1_only and target != "G":
            # cannot judge M/N without Stage 2; only report generated-vs-natural
            flip = (pred != "G") if target in ("M", "N") else (pred == "G")
            note = "stage1-only: generated-vs-natural proxy"
        else:
            flip = pred == target
            note = ""

        slot_score = flip.astype(np.float64) * (1.0 - mse)
        per_slot_score[slot_start:slot_start + count] = slot_score

        results[name] = {
            "target": target,
            "flip_rate": float(flip.mean()),
            "mean_mse": float(mse.mean()),
            "est_score": float(slot_score.mean()),
            "note": note,
        }
        line = (f"  {name:4s} -> {target}: flip={flip.mean():.1%}  "
                f"mse={mse.mean():.5f}  est_score={slot_score.mean():.4f}")
        if note:
            line += f"  [{note}]"
        print(line)

    total = float(per_slot_score.mean())
    print(f"\nEstimated global score (proxy DCB): {total:.4f}")
    if args.stage1_only:
        print("  (stage1-only: M/N split not evaluated — upper bound on true score)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"global": total, "tau_G": tau_g, "tau_MN": tau_mn,
             "per_direction": results}, indent=2))
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
