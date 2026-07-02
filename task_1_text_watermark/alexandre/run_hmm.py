"""Fit the HMM scorer on train, evaluate on val, and score any split.

Usage:
    python run_hmm.py --data-dir ../../data/watermark_localization --out-dir output \
        [--kgw-dir <dir with kgw_train.npz / kgw_validation.npz / kgw_test.npz>] \
        [--splits validation test]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hmm_scorer import fit_hmm, posterior_scores, read_jsonl


def load_kgw(kgw_dir: Path | None, split: str) -> dict | None:
    if kgw_dir is None:
        return None
    path = kgw_dir / f"kgw_{split}.npz"
    if not path.exists():
        print(f"WARNING: {path} missing — KGW disabled for {split}")
        return None
    npz = np.load(path)
    return {k: npz[k].astype(np.float64) for k in npz.files}


def score_split(model, records, kgw: dict | None, out_path: Path) -> tuple[list, list]:
    labels, scores = [], []
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            extra = {"kgw": kgw[rec["document_id"]]} if kgw else None
            s = np.clip(posterior_scores(model, rec["token_ids"], extra), 0.0, 1.0)
            f.write(json.dumps({"document_id": rec["document_id"],
                                "scores": [round(float(x), 6) for x in s]}) + "\n")
            if "labels" in rec:
                labels.extend(rec["labels"])
                scores.extend(s)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(records)}", flush=True)
    return labels, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--kgw-dir", default=None)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--p-enter", type=float, default=0.005)
    parser.add_argument("--p-exit", type=float, default=0.010)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    kgw_dir = Path(args.kgw_dir) if args.kgw_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(data_dir / "train.jsonl")
    kgw_train = load_kgw(kgw_dir, "train")
    extra_train = {"kgw": kgw_train} if kgw_train else None
    print("Fitting HMM on train...")
    model = fit_hmm(train, extra_signals=extra_train,
                    p_enter=args.p_enter, p_exit=args.p_exit)

    for split in args.splits:
        records = read_jsonl(data_dir / f"{split}.jsonl")
        kgw = load_kgw(kgw_dir, split)
        name = "submission.jsonl" if split == "test" else f"{split}_scores.jsonl"
        print(f"Scoring {split} ({len(records)} docs)...")
        labels, scores = score_split(model, records, kgw, out_dir / name)
        if labels:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(np.array(labels), np.array(scores))
            print(f"  {split} TPR@0.1%FPR = {np.interp(0.001, fpr, tpr):.4f}")
        print(f"  wrote {out_dir / name}")


if __name__ == "__main__":
    main()
