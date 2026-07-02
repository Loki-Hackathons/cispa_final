#!/usr/bin/env python3
"""Approach A for the '*->M' directions: rank the 300 genuine member images by
how strongly RAR has memorised them, then build N_M / G_M blocks from the
strongest members.

Why
---
For N->M and G->M the detector must output "member". Our img_000..299 ARE
training members of RAR, yet submitting them (content-swap) does NOT flip: the
detector's membership head (MIA) has low recall by design, so an average member
sits below its threshold. But membership strength varies across members; the
most strongly memorised ones (highest ICAS = conditional minus unconditional
likelihood under RAR) are the best chance to cross the bar.

This uses ONLY genuine member images (no adversarial perturbation), so it is the
most robust, least-overfit way to attack "->M". It is the membership analogue of
reconstruct_to_g.py.

Outputs (uint8, (300,256,256,3)):
  output/blocks/N_M_member.npy   for the N->M slots
  output/blocks/G_M_member.npy   for the G->M slots
plus output/blocks/member_ranking.json (ICAS score per member image index).

Selection strategy per reference slot: among the top-K most-memorised members,
pick the one nearest (pixel L2) to the reference, trading membership strength
against MSE. --top-k 1 fills every slot with the single strongest member
(max flip chance, worst MSE) — good for a first go/no-go API test.

Run on a GPU node (needs RAR-XL weights + the 1d-tokenizer VQ-GAN):
  export ONED_TOKENIZER_ROOT=/p/scratch/training2625/dougnon1/Loki/1d-tokenizer
  export PYTHONPATH=$ONED_TOKENIZER_ROOT:task_2_mgi/attempt1
  python task_2_mgi/attempt1/score_membership.py --top-k 1
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

from config import CLASS_SIZE, PathsConfig, setup_oned_tokenizer_path  # noqa: E402
from proxy_dcb import load_vqgan, uint8_to_tensor  # noqa: E402
from proxy_icas import (  # noqa: E402
    compute_membership_stats,
    load_class_predictor,
    load_rar_generator,
    predict_imagenet_class,
)
from submission_io import load_reference_images  # noqa: E402

CLASS_RANGES = {"M": (0, CLASS_SIZE), "N": (CLASS_SIZE, 2 * CLASS_SIZE),
                "G": (2 * CLASS_SIZE, 3 * CLASS_SIZE)}


@torch.no_grad()
def score_set(
    generator, tokenizer, classifier, images_uint8: np.ndarray,
    device: str, batch_size: int, tag: str = "",
) -> dict[str, np.ndarray]:
    """Return correct RAR membership signals (nll_cond, nll_uncond, icas)."""
    n = len(images_uint8)
    nll_c = np.empty(n, dtype=np.float32)
    nll_u = np.empty(n, dtype=np.float32)
    icas = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        chunk = images_uint8[start:start + batch_size]
        x = uint8_to_tensor(chunk, device)
        labels = predict_imagenet_class(x, classifier)
        c, u, i = compute_membership_stats(generator, tokenizer, x, labels)
        sl = slice(start, start + len(chunk))
        nll_c[sl] = c.detach().cpu().numpy()
        nll_u[sl] = u.detach().cpu().numpy()
        icas[sl] = i.detach().cpu().numpy()
        print(f"  scored {tag} {start + len(chunk)}/{n}", flush=True)
    return {"nll_cond": nll_c, "nll_uncond": nll_u, "icas": icas}


def membership_scores(
    generator, tokenizer, classifier, members_uint8: np.ndarray,
    device: str, batch_size: int,
) -> np.ndarray:
    """Higher score = more strongly memorised (more 'member')."""
    return score_set(generator, tokenizer, classifier, members_uint8,
                     device, batch_size, tag="members")["icas"]


def _fmt(a: np.ndarray) -> str:
    return (f"mean={a.mean():+.4f} std={a.std():.4f} "
            f"min={a.min():+.4f} max={a.max():+.4f}")


def run_diagnose(generator, tokenizer, classifier, originals, device, batch_size) -> None:
    """FREE diagnostic: does RAR likelihood separate members (M) from N and G?

    If members have clearly LOWER nll_cond (or HIGHER icas) than N, the
    membership signal is real and a '->M' attack has a concrete target.
    """
    sets = {}
    for cls, (a, b) in CLASS_RANGES.items():
        print(f"\nScoring set {cls} ...")
        sets[cls] = score_set(generator, tokenizer, classifier,
                               originals[a:b], device, batch_size, tag=cls)

    print("\n================ MEMBERSHIP DIAGNOSTIC ================")
    for signal in ("nll_cond", "icas"):
        print(f"\n[{signal}]  (member goal: "
              f"{'LOW nll_cond' if signal == 'nll_cond' else 'HIGH icas'})")
        for cls in ("M", "N", "G"):
            print(f"  {cls}: {_fmt(sets[cls][signal])}")

    # Separation of members (M) vs non-members (N) on the membership signal.
    m_icas, n_icas = sets["M"]["icas"], sets["N"]["icas"]
    thr = float(np.median(np.concatenate([m_icas, n_icas])))
    tpr = float((m_icas > thr).mean())
    fpr = float((n_icas > thr).mean())
    print("\n[icas] median-threshold separability M vs N:")
    print(f"  threshold={thr:+.4f}  TPR(M>thr)={tpr:.3f}  FPR(N>thr)={fpr:.3f}")
    print("  -> if TPR >> FPR, membership signal is real; an attack that raises")
    print("     icas above the M-set max can plausibly flip N/G into M.")
    print(f"  M-set icas max = {m_icas.max():+.4f}  (attack target: exceed this)")


def build_member_block(
    refs: np.ndarray, members: np.ndarray, ranked_idx: np.ndarray, top_k: int,
) -> np.ndarray:
    """For each ref, pick nearest (pixel L2) among the top_k strongest members."""
    pool_idx = ranked_idx[:max(top_k, 1)]
    pool = members[pool_idx].reshape(len(pool_idx), -1).astype(np.float32)
    r = refs.reshape(len(refs), -1).astype(np.float32)
    r2 = (r * r).sum(1, keepdims=True)
    p2 = (pool * pool).sum(1, keepdims=True).T
    d = r2 + p2 - 2.0 * (r @ pool.T)
    nearest_in_pool = d.argmin(axis=1)
    chosen = pool_idx[nearest_in_pool]
    return members[chosen].copy()


def main() -> int:
    p = argparse.ArgumentParser(description="Rank members by RAR memorisation, build ->M blocks")
    p.add_argument("--top-k", type=int, default=1,
                   help="Pool of strongest members to choose from per slot "
                        "(1 = single strongest everywhere; larger = lower MSE)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--diagnose", action="store_true",
                   help="FREE: score M/N/G sets and report if RAR likelihood "
                        "separates members from non-members (no blocks written)")
    args = p.parse_args()

    paths = PathsConfig()
    setup_oned_tokenizer_path(paths.oned_tokenizer_root)
    if not paths.tokenizer_ckpt.is_file():
        print(f"ERROR: tokenizer checkpoint not found: {paths.tokenizer_ckpt}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or (paths.output_dir / "blocks")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  top_k={args.top_k}")

    originals = load_reference_images(paths.data_dir)
    members = originals[CLASS_RANGES["M"][0]:CLASS_RANGES["M"][1]]
    refs_N = originals[CLASS_RANGES["N"][0]:CLASS_RANGES["N"][1]]
    refs_G = originals[CLASS_RANGES["G"][0]:CLASS_RANGES["G"][1]]

    tokenizer = load_vqgan(paths.tokenizer_ckpt, device=device)
    generator, _ = load_rar_generator(tokenizer_ckpt=paths.tokenizer_ckpt, device=device)
    classifier = load_class_predictor(device=device)

    if args.diagnose:
        run_diagnose(generator, tokenizer, classifier, originals, device, args.batch_size)
        return 0

    print("Scoring membership strength of the 300 member images...")
    scores = membership_scores(generator, tokenizer, classifier, members, device, args.batch_size)
    ranked_idx = np.argsort(-scores)  # descending: strongest member first

    ranking = {int(i): float(scores[i]) for i in range(len(scores))}
    (out_dir / "member_ranking.json").write_text(json.dumps(
        {"scores": ranking, "ranked_desc": [int(i) for i in ranked_idx],
         "top10": [int(i) for i in ranked_idx[:10]]}, indent=2))
    print(f"  ICAS range: [{scores.min():.4f}, {scores.max():.4f}]  "
          f"strongest members (idx): {list(ranked_idx[:5])}")

    block_nm = build_member_block(refs_N, members, ranked_idx, args.top_k)
    block_gm = build_member_block(refs_G, members, ranked_idx, args.top_k)

    for name, block, refs in (("N_M", block_nm, refs_N), ("G_M", block_gm, refs_G)):
        diff = block.astype(np.float32) - refs.astype(np.float32)
        mse = float(np.mean((diff / 255.0) ** 2))
        np.save(out_dir / f"{name}_member.npy", block)
        print(f"  {name}: saved {out_dir / f'{name}_member.npy'}  "
              f"mse_norm={mse:.5f}  (1-mse)={1.0 - mse:.5f}")

    print("\nTest these blocks (go/no-go on whether strong members flip ->M):")
    print("  python task_2_mgi/attempt1/build_submission_v2.py \\")
    print(f"      --dir N_M=block:{out_dir / 'N_M_member.npy'} \\")
    print(f"      --dir G_M=block:{out_dir / 'G_M_member.npy'} \\")
    print(f"      --dir M_G=block:{out_dir / 'M_G_recon.npy'} \\")
    print(f"      --dir N_G=block:{out_dir / 'N_G_recon.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
