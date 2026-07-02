#!/usr/bin/env python3
"""Smoke test: verify L_A(G) < L_A(M/N) on proxy VQ-GAN."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import AttackConfig, PathsConfig, setup_oned_tokenizer_path
from proxy_dcb import (
    calibrate_thresholds,
    classify_stage1,
    compute_LA,
    compute_LQ,
    compute_LR,
    load_vqgan,
    smoke_check,
    uint8_to_tensor,
)


def load_one_image(data_dir: Path, index: int, size: int = 256) -> np.ndarray:
    with Image.open(data_dir / f"img_{index:03d}.png") as img:
        return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def main() -> int:
    paths = PathsConfig()
    setup_oned_tokenizer_path(paths.oned_tokenizer_root)

    if not paths.tokenizer_ckpt.is_file():
        print(f"ERROR: tokenizer checkpoint not found: {paths.tokenizer_ckpt}", file=sys.stderr)
        return 1
    if not paths.data_dir.is_dir():
        print(f"ERROR: data dir not found: {paths.data_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Tokenizer: {paths.tokenizer_ckpt}")
    print(f"Data: {paths.data_dir}")

    vqgan = load_vqgan(paths.tokenizer_ckpt, device=device)
    cfg = AttackConfig(device=device)

    samples = {
        "M": load_one_image(paths.data_dir, 0),
        "N": load_one_image(paths.data_dir, 300),
        "G": load_one_image(paths.data_dir, 600),
    }

    print("\nPer-image proxy scores (alpha=1.0 placeholder):")
    for label, img in samples.items():
        x = uint8_to_tensor(img, device)
        lq = compute_LQ(vqgan, x).item()
        lr = compute_LR(vqgan, x, eps=cfg.eps).item()
        la = compute_LA(vqgan, x, alpha=1.0, eps=cfg.eps).item()
        print(f"  {label}: L_Q={lq:.6f}  L_R={lr:.6f}  L_A={la:.6f}")

    # Quick calibration on 30 images per class
    def load_slice(start: int, count: int) -> torch.Tensor:
        arrs = [load_one_image(paths.data_dir, start + i) for i in range(count)]
        return uint8_to_tensor(np.stack(arrs), device)

    print("\nCalibrating on 30 images per class...")
    cal = calibrate_thresholds(
        vqgan,
        load_slice(0, 30),
        load_slice(300, 30),
        load_slice(600, 30),
        batch_size=8,
    )
    alpha = cal["alpha"]
    tau_g = cal["tau_G"]
    print(f"  alpha={alpha:.4f}  tau_G={tau_g:.6f}")
    print(f"  stats: {cal['stats']}")

    ok = smoke_check(cal)
    print(f"\nSmoke check (mean L_A_G < L_A_M and L_A_G < L_A_N): {'PASS' if ok else 'FAIL'}")

    for label, img in samples.items():
        x = uint8_to_tensor(img, device)
        la = compute_LA(vqgan, x, alpha=alpha, eps=cfg.eps).item()
        print(f"  {label} stage1={classify_stage1(la, tau_g)} (L_A={la:.6f})")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
