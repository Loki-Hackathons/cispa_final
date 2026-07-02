"""Sweep per-scheme shift hypotheses for the semi-Markov scorer.

Evaluates TPR@0.1%FPR on BOTH train and validation to avoid picking a
config that only wins on one split (the KGW shift sweep done earlier used
val only — overfitting risk flagged in review).

Usage: python sweep_shifts.py
"""

from __future__ import annotations

import json
import time

import numpy as np

import smm_scorer
from smm_scorer import read_jsonl, score_document

DATA = r"..\..\data\watermark_localization"

CONFIGS = {
    "baseline (current)": {
        "textseal": (0.45, 0.65, 0.9),
        "gumbelmax": (0.55, 0.8, 1.1),
        "kgw": (0.6, 0.9, 1.3),
    },
    "weak kgw added": {
        "textseal": (0.45, 0.65, 0.9),
        "gumbelmax": (0.55, 0.8, 1.1),
        "kgw": (0.3, 0.6, 0.9, 1.3),
    },
    "weak all added": {
        "textseal": (0.25, 0.45, 0.65, 0.9),
        "gumbelmax": (0.3, 0.55, 0.8, 1.1),
        "kgw": (0.3, 0.6, 0.9, 1.3),
    },
    "weak all, denser": {
        "textseal": (0.2, 0.35, 0.5, 0.7, 0.9),
        "gumbelmax": (0.25, 0.45, 0.65, 0.9, 1.1),
        "kgw": (0.25, 0.5, 0.75, 1.0, 1.3),
    },
}


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

    for name, shifts in CONFIGS.items():
        smm_scorer.SHIFTS = shifts
        line = [name]
        for split, (recs, kgw) in splits.items():
            t0 = time.time()
            scores, labels = [], []
            for r in recs:
                extra = ({"kgw": kgw[r["document_id"]].astype(np.float64)}
                         if r["document_id"] in kgw.files else None)
                scores.append(score_document(r["token_ids"], extra=extra))
                labels.append(np.array(r["labels"]))
            tpr = tpr_at_fpr(scores, labels)
            line.append(f"{split}={tpr:.4f} ({time.time() - t0:.0f}s)")
        print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
