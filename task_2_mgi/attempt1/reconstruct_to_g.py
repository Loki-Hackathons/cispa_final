#!/usr/bin/env python3
"""Produce '*->G' direction blocks by autoencoding references through the
MaskGIT VQ-GAN that RAR decodes with.

RAR-generated images are all rendered by this VQ-GAN decoder, so encoding a
natural reference to discrete tokens and decoding it back stamps the reference
with the *same* decoder fingerprint the generation-attribution head keys on,
while staying visually close to the original (low MSE). Much lower MSE than
swapping in a different generated image.

Writes uint8 (300,256,256,3) blocks to output/blocks/ that build_submission_v2
plugs in via `--dir M_G=block:...`.

Run on a GPU node (needs torch + the 1d-tokenizer VQ-GAN weights):
  export ONED_TOKENIZER_ROOT=/p/scratch/training2625/dougnon1/Loki/1d-tokenizer
  export PYTHONPATH=$ONED_TOKENIZER_ROOT:task_2_mgi/attempt1
  python task_2_mgi/attempt1/reconstruct_to_g.py --classes M N --passes 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_SIZE, PathsConfig, setup_oned_tokenizer_path  # noqa: E402
from proxy_dcb import load_vqgan, reconstruct, uint8_to_tensor  # noqa: E402
from submission_io import load_reference_images  # noqa: E402

CLASS_START = {"M": 0, "N": CLASS_SIZE, "G": 2 * CLASS_SIZE}


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """BCHW float [0,1] -> NHWC uint8."""
    x = x.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
    return (x.cpu().numpy() * 255.0).round().astype(np.uint8)


@torch.no_grad()
def reconstruct_block(vqgan, refs: np.ndarray, device: str,
                      passes: int, batch_size: int) -> np.ndarray:
    out = np.empty_like(refs)
    for start in range(0, len(refs), batch_size):
        chunk = refs[start:start + batch_size]
        x = uint8_to_tensor(chunk, device)
        for _ in range(passes):
            x = reconstruct(vqgan, x).clamp(0.0, 1.0)
        out[start:start + len(chunk)] = tensor_to_uint8(x)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="RAR VQ-GAN reconstruction blocks")
    p.add_argument("--classes", nargs="+", default=["M", "N"],
                   choices=["M", "N", "G"],
                   help="Source classes to reconstruct (M for M->G, N for N->G)")
    p.add_argument("--passes", type=int, default=1,
                   help="Number of encode->decode passes (>1 = stronger "
                        "fingerprint, higher MSE)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    paths = PathsConfig()
    setup_oned_tokenizer_path(paths.oned_tokenizer_root)
    if not paths.tokenizer_ckpt.is_file():
        print(f"ERROR: tokenizer checkpoint not found: {paths.tokenizer_ckpt}",
              file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or (paths.output_dir / "blocks")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  passes={args.passes}")
    originals = load_reference_images(paths.data_dir)
    vqgan = load_vqgan(paths.tokenizer_ckpt, device=device)

    for cls in args.classes:
        s = CLASS_START[cls]
        refs = originals[s:s + CLASS_SIZE]
        block = reconstruct_block(vqgan, refs, device, args.passes, args.batch_size)
        diff = block.astype(np.float32) - refs.astype(np.float32)
        mse_norm = float(np.mean((diff / 255.0) ** 2))
        name = f"{cls}_G_recon.npy"
        np.save(out_dir / name, block)
        print(f"  {cls}->G  saved {out_dir / name}  "
              f"mse_norm={mse_norm:.5f}  (1-mse)={1.0 - mse_norm:.5f}")

    print("\nPlug into a submission, e.g.:")
    print("  python task_2_mgi/attempt1/build_submission_v2.py \\")
    print(f"      --dir M_G=block:{out_dir / 'M_G_recon.npy'} \\")
    print(f"      --dir N_G=block:{out_dir / 'N_G_recon.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
