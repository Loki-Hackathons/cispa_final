"""Phase 3 — train the detector-fusion calibrator.

Builds the full per-token feature matrix (all four detectors + entropy + stats), trains a
lightweight calibrator on the train split, and selects on validation using the exact
competition metric (pooled TPR @ 0.1 % FPR). Saves the model for ``predict.py``.

Requires the watermark YAML (real keys) and, for correct KGW, a CUDA GPU. Without them it
still runs (degraded detectors) so you can smoke-test the wiring.

Usage:
    python -m src.train_calibrator --model logreg      # or: gboost
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from . import config
from .evaluate import evaluate_pooled
from .load_data import load_split
from .pipeline import build_matrix, scores_to_docs, write_submission
from .postprocess import postprocess


def _make_model(kind: str):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    if kind == "gboost":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=config.SEED,
        )
    raise ValueError(f"unknown model kind: {kind}")


def run(model_kind: str = "logreg", use_entropy_lm: bool = True,
        smooth_radius: int = 3, smooth_sigma: float = 1.5) -> dict:
    config.ensure_dirs()
    np.random.seed(config.SEED)
    cfg = config.load_watermark_config()
    if not config.WATERMARK_YAML:
        print("[train] WARNING: WML_WATERMARK_YAML not set — detectors use key-free "
              "fallbacks. Set it on the cluster for real keys.")

    print("[train] building features (train) ...")
    train = load_split("train")
    Xtr, ytr, _ = build_matrix(train, cfg, use_detectors=True, use_entropy_lm=use_entropy_lm)
    print("[train] building features (val) ...")
    val = load_split("validation")
    Xva, yva, idx_va = build_matrix(val, cfg, use_detectors=True, use_entropy_lm=use_entropy_lm)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    model = _make_model(model_kind)
    model.fit(scaler.transform(Xtr), ytr)

    raw = model.predict_proba(scaler.transform(Xva))[:, 1]
    metrics_raw = evaluate_pooled(raw, yva)

    # Apply span smoothing per document, then re-pool.
    pred_by_doc = scores_to_docs(idx_va, raw)
    smoothed_by_doc = {
        d: postprocess(s, smooth_radius=smooth_radius, smooth_sigma=smooth_sigma)
        for d, s in pred_by_doc.items()
    }
    # Re-pool smoothed scores in doc/token order to match labels.
    val_docs = {d.document_id: d for d in val}
    sm_scores, sm_labels = [], []
    for did, sc in smoothed_by_doc.items():
        sm_scores.extend(sc)
        sm_labels.extend(val_docs[did].labels)
    metrics_sm = evaluate_pooled(sm_scores, sm_labels)

    print("[train] val (raw)     :", json.dumps(metrics_raw, indent=2))
    print("[train] val (smoothed):", json.dumps(metrics_sm, indent=2))

    # Persist model + config for predict.py.
    artifact = {
        "model": model, "scaler": scaler, "model_kind": model_kind,
        "use_entropy_lm": use_entropy_lm,
        "smooth_radius": smooth_radius, "smooth_sigma": smooth_sigma,
        "metrics_raw": metrics_raw, "metrics_smoothed": metrics_sm,
    }
    out = config.OUTPUT_DIR / f"calibrator_{model_kind}.pkl"
    with open(out, "wb") as fh:
        pickle.dump(artifact, fh)
    print(f"[train] saved {out}")

    write_submission(smoothed_by_doc, str(config.OUTPUT_DIR / "val_pred.jsonl"))
    return metrics_sm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fusion calibrator.")
    parser.add_argument("--model", default="logreg", choices=["logreg", "gboost"])
    parser.add_argument("--no-entropy-lm", action="store_true",
                        help="Skip the proxy LM entropy feature (use novelty proxy).")
    parser.add_argument("--smooth-radius", type=int, default=3)
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    args = parser.parse_args()
    run(args.model, use_entropy_lm=not args.no_entropy_lm,
        smooth_radius=args.smooth_radius, smooth_sigma=args.smooth_sigma)
