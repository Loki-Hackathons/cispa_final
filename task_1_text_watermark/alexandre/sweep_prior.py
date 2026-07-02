"""Sweep LOG_P_SPAN (span-entry prior) for the semi-Markov scorer.

Same protocol as sweep_shifts.py: TPR@0.1%FPR on train AND validation;
adopt only if it wins on both.
"""

from __future__ import annotations

import time

import numpy as np

import smm_scorer
from smm_scorer import read_jsonl, score_document

DATA = r"..\..\data\watermark_localization"
PRIORS = (0.002, 0.004, 0.008, 0.016)


def tpr_at_fpr(scores, labels, fpr=0.001):
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    clean = np.sort(s[y == 0])[::-1]
    k = max(int(len(clean) * fpr), 1)
    tau = clean[k - 1]
    return (s[y == 1] >= tau).mean()


def main() -> None:
    splits = {}
    for split in ("train", "validation"):
        recs = read_jsonl(rf"{DATA}\{split}.jsonl")
        kgw = np.load(f"output/kgw_{split}.npz")
        splits[split] = (recs, kgw)

    for p in PRIORS:
        smm_scorer.LOG_P_SPAN = np.log(p)
        line = [f"p_span={p}"]
        for split, (recs, kgw) in splits.items():
            t0 = time.time()
            scores, labels = [], []
            for r in recs:
                extra = ({"kgw": kgw[r["document_id"]].astype(np.float64)}
                         if r["document_id"] in kgw.files else None)
                scores.append(score_document(r["token_ids"], extra=extra))
                labels.append(np.array(r["labels"]))
            line.append(f"{split}={tpr_at_fpr(scores, labels):.4f} ({time.time() - t0:.0f}s)")
        print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
