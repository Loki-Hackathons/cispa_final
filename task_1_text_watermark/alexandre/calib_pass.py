"""Temperature/top-p calibrated realized-token probability for the exact
Gumbel-Max / TextSeal LLR (docs/task1/attempt1.md 19.4 point 1).

Why: the vendor generator (vendor/textseal/textseal/watermarking/generator.py
GumbelmaxGenerator/TextSealGenerator.sample_next) samples as:
    probs = softmax(logits / T)
    zero everything beyond the top-p nucleus, renormalize -> probs_sort
    next_token = argmax( r ^ (1 / probs_sort) )
i.e. the "p" driving the Aaronson reparameterization is the TEMPERATURE- AND
TOP-P-ADJUSTED, RENORMALIZED probability, not the raw softmax(logits) at
T=1 used so far (smm_scorer.gumbel_exact_llr / textseal_exact_llr). This is
the most likely reason the exact LLR made CV worse (docs/task1/attempt1.md
19.3): wrong p -> wrong LLR magnitude and even wrong sign in the tails.

watermark_config.yaml does not pin temperature/top_p (organizer-hidden), but
the vendor code's own defaults triangulate a narrow, plausible range:
  - ProcessingConfig default (config.py):      T=0.9, top_p=0.95
  - WmGenerator.generate() signature default:  T=0.8, top_p=0.95
  - main.py --help docstring examples:         T in [0.8, 1.0], top_p=0.95
This script recomputes the realized token's probability under a small grid
of (T, top_p) candidates spanning that range, reusing a single 7B forward
pass per document (the sort order is T-invariant; only softmax/cumsum/mask
differ per grid point, so the grid is cheap relative to the forward pass).

Output: calib_T{t}_p{p}_{split}.npz per grid point (document_id -> float32
log-probability array), same sentinel convention as logp_{split}.npz
(+1.0 = no context / first token, impossible for a real log-prob).

Usage (JURECA GPU node):
    python calib_pass.py --data-dir <dataset dir> --out-dir output \
        --splits train validation --temps 0.8 0.9 1.0 --top-ps 0.9 0.95 1.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from entropy_pass import load_model, read_jsonl


@torch.no_grad()
def doc_calib(model, token_ids, device, temps, top_ps) -> dict[str, np.ndarray]:
    n = len(token_ids)
    ids = torch.tensor([token_ids], device=device)
    logits = model(ids).logits[0].float()            # (n, V)
    cur_logits = logits[:-1]                          # predicts token 1..n-1
    target = ids[0, 1:]                                # (n-1,)

    sorted_logits, sorted_idx = torch.sort(cur_logits, dim=-1, descending=True)
    rank = torch.empty_like(sorted_idx)
    ar = torch.arange(sorted_idx.shape[1], device=device).unsqueeze(0).expand_as(sorted_idx)
    rank.scatter_(1, sorted_idx, ar)
    target_rank = rank[torch.arange(n - 1, device=device), target]
    row = torch.arange(n - 1, device=device)

    out = {}
    for T in temps:
        probs_sorted = torch.softmax(sorted_logits / T, dim=-1)
        cumsum = torch.cumsum(probs_sorted, dim=-1)
        for top_p in top_ps:
            if top_p >= 1.0:
                kept = probs_sorted
            else:
                mask = (cumsum - probs_sorted) > top_p
                kept = probs_sorted.masked_fill(mask, 0.0)
                kept = kept / kept.sum(dim=-1, keepdim=True)
            p_target = kept[row, target_rank].clamp_min(1e-30)
            arr = np.full(n, 1.0, dtype=np.float32)   # sentinel: no context
            arr[1:] = torch.log(p_target).cpu().numpy().astype(np.float32)
            out[f"T{T}_p{top_p}"] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.8, 0.9, 1.0])
    ap.add_argument("--top-ps", nargs="+", type=float, default=[0.9, 0.95, 1.0])
    args = ap.parse_args()

    model, name = load_model(args.device, require_primary=True)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [f"T{T}_p{P}" for T in args.temps for P in args.top_ps]
    for split in args.splits:
        records = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        out = {tag: {} for tag in tags}
        t0 = time.time()
        for i, rec in enumerate(records):
            sig = doc_calib(model, rec["token_ids"], device, args.temps, args.top_ps)
            did = str(rec["document_id"])
            for tag in tags:
                out[tag][did] = sig[tag]
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{split}: {i + 1}/{len(records)} ({rate:.2f} docs/s)", flush=True)
        for tag, data in out.items():
            np.savez_compressed(out_dir / f"calib_{tag}_{split}.npz", **data)
        print(f"{split}: wrote {len(tags)} calib_*_{split}.npz files "
              f"({len(records)} docs, {time.time() - t0:.0f}s, model={name})",
              flush=True)


if __name__ == "__main__":
    main()
