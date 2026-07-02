"""Evaluation: the exact competition metric — TPR @ 0.1 % FPR, pooled over tokens.

The leaderboard pools every token from every document, ranks by score, picks the
threshold giving 0.1 % false-positive rate on the clean (label 0) tokens, and reports
the true-positive rate on watermarked (label 1) tokens at that threshold.

We also report ROC-AUC and TPR@1% as sanity checks. This is a ranking metric, so only
the ordering of scores matters.

CLI:  python -m src.evaluate --pred outputs/val_pred.jsonl --split validation
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np


def tpr_at_fpr(scores: Sequence[float], labels: Sequence[int],
               target_fpr: float = 0.001) -> tuple[float, float]:
    """Return (TPR at the threshold achieving <= target_fpr, that threshold)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    # Threshold = the smallest score s.t. FPR(neg >= thr) <= target_fpr.
    # Use the (1 - target_fpr) quantile of the negative scores.
    thr = float(np.quantile(neg, 1.0 - target_fpr, method="higher"))
    tpr = float(np.mean(pos >= thr))
    return tpr, thr


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_pooled(scores: Sequence[float], labels: Sequence[int]) -> dict:
    tpr01, thr01 = tpr_at_fpr(scores, labels, 0.001)
    tpr1, _ = tpr_at_fpr(scores, labels, 0.01)
    return {
        "tpr@0.1%fpr": tpr01,
        "threshold@0.1%fpr": thr01,
        "tpr@1%fpr": tpr1,
        "auc": roc_auc(scores, labels),
        "n_tokens": len(scores),
        "n_pos": int(np.sum(np.asarray(labels) == 1)),
    }


def _load_predictions(path: str) -> dict[str, list[float]]:
    preds: dict[str, list[float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[str(obj["document_id"])] = list(obj["scores"])
    return preds


def evaluate_file(pred_path: str, split: str = "validation") -> dict:
    """Pool a prediction .jsonl against the labeled split and compute the metric."""
    from .load_data import load_split

    docs = load_split(split)
    preds = _load_predictions(pred_path)
    all_scores: list[float] = []
    all_labels: list[int] = []
    missing = 0
    for d in docs:
        if d.labels is None:
            continue
        s = preds.get(d.document_id)
        if s is None:
            missing += 1
            continue
        if len(s) != d.n_tokens:
            raise ValueError(
                f"{d.document_id}: scores length {len(s)} != tokens {d.n_tokens}")
        all_scores.extend(s)
        all_labels.extend(d.labels)
    if missing:
        print(f"[evaluate] WARNING: {missing} labeled docs missing from predictions")
    res = evaluate_pooled(all_scores, all_labels)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pooled TPR@0.1%FPR evaluation.")
    parser.add_argument("--pred", required=True, help="Prediction .jsonl path")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    evaluate_file(args.pred, args.split)
