"""Train a logistic-regression calibrator on multi-scale detector features.

Trains on train.jsonl, evaluates TPR@0.1%FPR on validation.jsonl, then can
score test.jsonl into a submission file.

Usage:
    python train_calibrator.py --data-dir ../../data/watermark_localization \
        [--kgw-dir kgw_npz_dir] --out-dir output
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from features import doc_features


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.001) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def load_kgw(kgw_dir: Path | None, split: str):
    if kgw_dir is None:
        return None
    path = kgw_dir / f"kgw_{split}.npz"
    if not path.exists():
        print(f"WARNING: {path} not found, KGW features disabled for {split}")
        return None
    return np.load(path)


def build_matrix(records: list[dict], kgw, smooth: int = 0):
    feats, labels = [], []
    for i, rec in enumerate(records):
        kgw_sig = kgw[rec["document_id"]].astype(np.float64) if kgw is not None else None
        feats.append(doc_features(rec["token_ids"], kgw_sig))
        if "labels" in rec:
            labels.append(np.array(rec["labels"]))
        if (i + 1) % 200 == 0:
            print(f"  features {i + 1}/{len(records)}", flush=True)
    return feats, labels


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    kernel = np.ones(window) / window
    return np.convolve(scores, kernel, mode="same")


def predict_docs(model, feats: list[np.ndarray], smooth: int) -> list[np.ndarray]:
    out = []
    for X in feats:
        p = model.predict_proba(X)[:, 1]
        out.append(smooth_scores(p, smooth))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--kgw-dir", default=None)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--smooth", type=int, default=9, help="post-hoc moving-average window")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    kgw_dir = Path(args.kgw_dir) if args.kgw_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(data_dir / "train.jsonl")
    val = read_jsonl(data_dir / "validation.jsonl")

    print("Building train features...")
    train_feats, train_labels = build_matrix(train, load_kgw(kgw_dir, "train"))
    X_train = np.vstack(train_feats)
    y_train = np.concatenate(train_labels)

    print(f"Training logistic regression on {X_train.shape} ...")
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X_train, y_train)

    print("Building validation features...")
    val_feats, val_labels = build_matrix(val, load_kgw(kgw_dir, "validation"))

    best = (None, -1.0)
    for smooth in (1, 5, 9, 15, 21, 31):
        val_scores = predict_docs(model, val_feats, smooth)
        m = tpr_at_fpr(np.concatenate(val_labels), np.concatenate(val_scores))
        print(f"  smooth={smooth:>2}  val TPR@0.1%FPR = {m:.4f}")
        if m > best[1]:
            best = (smooth, m)
    smooth = best[0]
    print(f"Selected smooth={smooth} (val TPR@0.1%FPR={best[1]:.4f})")

    # Save val predictions for task1_eval / dashboard viewer
    val_scores = predict_docs(model, val_feats, smooth)
    with open(out_dir / "val_scores.jsonl", "w", encoding="utf-8") as f:
        for rec, s in zip(val, val_scores):
            f.write(json.dumps({"document_id": rec["document_id"],
                                "scores": [round(float(x), 6) for x in np.clip(s, 0, 1)]}) + "\n")

    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump({"model": model, "smooth": smooth}, f)

    if not args.skip_test:
        test = read_jsonl(data_dir / "test.jsonl")
        print(f"Scoring test ({len(test)} docs)...")
        test_feats, _ = build_matrix(test, load_kgw(kgw_dir, "test"))
        test_scores = predict_docs(model, test_feats, smooth)
        with open(out_dir / "submission.jsonl", "w", encoding="utf-8") as f:
            for rec, s in zip(test, test_scores):
                f.write(json.dumps({"document_id": rec["document_id"],
                                    "scores": [round(float(x), 6) for x in np.clip(s, 0, 1)]}) + "\n")
        print(f"Wrote {out_dir / 'submission.jsonl'}")


if __name__ == "__main__":
    main()
