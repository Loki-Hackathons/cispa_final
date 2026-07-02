"""Phase 1 baseline — key-free statistical logistic model.

Runs *without* the watermark YAML or a GPU: detector channels are zeroed and only
key-free statistical + novelty features are used. Purpose: exercise the full pipeline,
produce a valid submission early, and set a ranking floor. NOT the final method.

Usage:
    python -m src.baseline            # train on train, eval on val, write val + test preds
"""

from __future__ import annotations

import json

import numpy as np

from . import config
from .evaluate import evaluate_pooled
from .load_data import load_split
from .pipeline import build_matrix, scores_to_docs, write_submission


def run() -> dict:
    config.ensure_dirs()
    np.random.seed(config.SEED)

    print("[baseline] loading data ...")
    train = load_split("train")
    val = load_split("validation")

    print("[baseline] building key-free features ...")
    Xtr, ytr, _ = build_matrix(train, use_detectors=False, use_entropy_lm=False)
    Xva, yva, idx_va = build_matrix(val, use_detectors=False, use_entropy_lm=False)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)

    val_scores = clf.predict_proba(scaler.transform(Xva))[:, 1]
    metrics = evaluate_pooled(val_scores, yva)
    print("[baseline] validation:", json.dumps(metrics, indent=2))

    # Write val predictions (for evaluate.py) and a test submission.
    pred_val = scores_to_docs(idx_va, val_scores)
    write_submission(pred_val, str(config.OUTPUT_DIR / "baseline_val_pred.jsonl"))

    test = load_split("test")
    Xte, _, idx_te = build_matrix(test, use_detectors=False, use_entropy_lm=False,
                                  with_labels=False)
    test_scores = clf.predict_proba(scaler.transform(Xte))[:, 1]
    pred_test = scores_to_docs(idx_te, test_scores)
    write_submission(pred_test, str(config.OUTPUT_DIR / "baseline_submission.jsonl"))
    print(f"[baseline] wrote {config.OUTPUT_DIR / 'baseline_submission.jsonl'}")
    return metrics


if __name__ == "__main__":
    run()
