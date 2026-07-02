"""Print every model's shapes and dump a visual sanity-check of analytic
reconstructions. Run this first on the cluster to confirm assumptions.

Usage:
  python inspect_models.py                 # table for all 12
  python inspect_models.py --preview 5 8   # save preview grids for models 5,8
  python inspect_models.py --keys 9 11     # full param name+shape dump (ViT check)
"""
from __future__ import annotations

import argparse
import os

import torch

import config
import extract
import utils


def table():
    print(f"{'model':>5} | {'family':6} | {'act':7} | {'feat_shape':12} | first-layer shapes")
    print("-" * 90)
    for i in range(1, config.NUM_MODELS + 1):
        g = utils.load_gradient(i)
        gr = g["gradients"]
        names = list(gr.keys())
        first = names[0]
        shapes = ", ".join(f"{n}{tuple(gr[n].shape)}" for n in names[:2])
        print(f"{i:>5} | {g['family']:6} | {g['activation']:7} | "
              f"{str(tuple(g['feature_shape'])):12} | {shapes}")


def dump_keys(i: int):
    """Print every gradient/state parameter name + shape for one model.

    Use this to confirm the ViT (or any) architecture before trusting the
    rebuilt forward model used for gradient-matching optimization.
    """
    g = utils.load_gradient(i)
    gr = g["gradients"]
    print(f"\n=== model{i} | {g['family']}/{g['activation']} | "
          f"feature_shape={tuple(g['feature_shape'])} | "
          f"{len(gr)} params ===")
    for name, t in gr.items():
        print(f"  {name:40s} {tuple(t.shape)}")


def save_preview(i: int, n: int = 64):
    """Delegate to analyze.preview, which saves .npy always and PNG when a
    renderer (torchvision/PIL/matplotlib) is available — no hard dependency."""
    import analyze
    analyze.preview([i], select="quality", n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, nargs="*", default=[])
    ap.add_argument("--keys", type=int, nargs="*", default=[])
    args = ap.parse_args()
    table()
    for i in args.keys:
        dump_keys(i)
    for i in args.preview:
        save_preview(i)


if __name__ == "__main__":
    main()
