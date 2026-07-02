"""Score a split with the semi-Markov scorer and write submission JSONL.

Usage:
  python run_smm.py --split validation --out val_scores.jsonl [--kgw kgw_validation.npz]
  python run_smm.py --split test --out submission.jsonl [--kgw kgw_test.npz]

Scores are written in full float precision (no rounding): TPR@0.1%FPR is a
ranking metric and rounding creates tie groups that cost real score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from smm_scorer import read_jsonl, score_document

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "watermark_localization"


def load_kgw(path: str | None) -> dict[int, np.ndarray] | None:
    if not path:
        return None
    npz = np.load(path)
    return {int(k): npz[k].astype(np.float64) for k in npz.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--data", default=None, help="override dataset jsonl path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kgw", default=None, help="npz with per-doc kgw green masks")
    args = ap.parse_args()

    data_path = args.data or DATA_DIR / f"{args.split}.jsonl"
    records = read_jsonl(data_path)
    kgw = load_kgw(args.kgw)

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            doc_id = rec["document_id"]
            extra = None
            if kgw is not None:
                key = int(doc_id) if not isinstance(doc_id, int) else doc_id
                if key in kgw:
                    extra = {"kgw": kgw[key]}
            scores = score_document(rec["token_ids"], extra=extra)
            f.write(json.dumps({"document_id": doc_id,
                                "scores": [float(s) for s in scores]}) + "\n")
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(records)} docs ({rate:.1f} docs/s)", flush=True)
    print(f"Wrote {len(records)} docs to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
