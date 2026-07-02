"""Task 1 baseline: windowed z-score fusion of per-token detector signals.

For each scheme, per-token increments are smoothed with centered moving
windows of several sizes and converted to z-scores against the scheme's H0
moments. The final token score is the max z across windows and schemes,
squashed to [0, 1] with a sigmoid (monotone, so TPR@FPR is unchanged).

KGW greenlists require CUDA Philox: precompute them on the cluster with
kgw_scores.py and pass the resulting .npz via --kgw.

Usage:
    python build_scores.py --dataset train.jsonl --out train_scores.jsonl [--kgw kgw_train.npz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from detectors import H0_MOMENTS, compute_signals

WINDOW_SIZES = (16, 32, 64, 128)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def windowed_z(x: np.ndarray, mu0: float, sigma0: float, windows=WINDOW_SIZES) -> np.ndarray:
    """Max centered moving-window z-score per token."""
    n = len(x)
    best = np.full(n, -np.inf)
    centered = x - mu0
    cumsum = np.concatenate([[0.0], np.cumsum(centered)])
    for w in windows:
        if w > n:
            continue
        half = w // 2
        starts = np.clip(np.arange(n) - half, 0, n - w)
        sums = cumsum[starts + w] - cumsum[starts]
        z = sums / (sigma0 * np.sqrt(w))
        best = np.maximum(best, z)
    if not np.isfinite(best).all():  # doc shorter than the smallest window
        w = n
        z = float(centered.sum() / (sigma0 * np.sqrt(w)))
        best = np.where(np.isfinite(best), best, z)
    return best


def score_document(token_ids: list[int], kgw_signal: np.ndarray | None = None) -> np.ndarray:
    signals = compute_signals(token_ids)
    if kgw_signal is not None:
        signals["kgw"] = kgw_signal
    fused = np.full(len(token_ids), -np.inf)
    for name, sig in signals.items():
        mu0, var0 = H0_MOMENTS[name]
        fused = np.maximum(fused, windowed_z(sig, mu0, np.sqrt(var0)))
    return 1.0 / (1.0 + np.exp(-fused / 4.0))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.001) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kgw", default=None, help="npz with per-doc KGW green masks")
    args = parser.parse_args()

    records = read_jsonl(args.dataset)
    kgw = np.load(args.kgw) if args.kgw else None

    all_labels, all_scores = [], []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            kgw_sig = kgw[rec["document_id"]].astype(np.float64) if kgw is not None else None
            scores = score_document(rec["token_ids"], kgw_sig)
            f.write(json.dumps({"document_id": rec["document_id"],
                                "scores": [round(float(s), 6) for s in scores]}) + "\n")
            if "labels" in rec:
                all_labels.extend(rec["labels"])
                all_scores.extend(scores)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(records)} documents scored", flush=True)

    print(f"Wrote {out_path}")
    if all_labels:
        val = tpr_at_fpr(np.array(all_labels), np.array(all_scores))
        print(f"TPR @ 0.1% FPR: {val:.4f}  ({len(all_labels)} tokens)")


if __name__ == "__main__":
    main()
