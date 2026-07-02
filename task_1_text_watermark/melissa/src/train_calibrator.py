"""Phase 3 — train the detector-fusion calibrator.

Builds the full per-token feature matrix from the **correct, vendor-based** watermark
signals (TextSeal / Gumbel-Max / Unigram via the pinned repos, KGW via CUDA Philox) with
multi-scale windowed z-scores, trains a lightweight calibrator on the train split, and
selects the span-smoothing on validation using the exact competition metric
(pooled TPR @ 0.1 % FPR). Saves the model for ``predict.py``.

Requires the pinned vendor repos (``scripts/task1/sync_watermark_repos.sh`` on a login
node) and, for correct KGW, a CUDA GPU.

Usage:
    python -m src.train_calibrator --model gboost      # or: logreg
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from . import config
from .correct_features import doc_feature_matrix, feature_names, kgw_available
from .evaluate import evaluate_pooled
from .load_data import load_split
from .pipeline import write_submission
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


# Span-smoothing grid searched on validation (radius, sigma). (0, 0) = no smoothing.
_SMOOTH_GRID = [(0, 0.0), (3, 1.5), (5, 2.0), (7, 2.5), (9, 3.0)]


def _doc_features(docs, use_kgw: bool):
    feats, labels = [], []
    total = len(docs)
    for i, d in enumerate(docs, 1):
        feats.append(doc_feature_matrix(d.token_ids, use_kgw=use_kgw))
        labels.append(np.asarray(d.labels) if d.labels is not None else None)
        if i % 20 == 0 or i == total:
            print(f"  features {i}/{total}", flush=True)
    return feats, labels


def run(model_kind: str = "gboost", smooth_radius: int | None = None,
        smooth_sigma: float | None = None) -> dict:
    config.ensure_dirs()
    np.random.seed(config.SEED)
    use_kgw = kgw_available()
    if not use_kgw:
        print("[train] WARNING: KGW unavailable (no CUDA / vendor) — training without "
              "KGW features. Run on a GPU node for the full signal.")

    print("[train] building features (train) ...")
    train = load_split("train")
    tr_feats, tr_labels = _doc_features(train, use_kgw)
    Xtr = np.vstack(tr_feats)
    ytr = np.concatenate([l for l in tr_labels])

    print("[train] building features (val) ...")
    val = load_split("validation")
    va_feats, _ = _doc_features(val, use_kgw)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    model = _make_model(model_kind)
    print(f"[train] fitting {model_kind} on X={Xtr.shape} ...")
    model.fit(scaler.transform(Xtr), ytr)

    # Per-doc raw probabilities on validation.
    raw_by_doc = [model.predict_proba(scaler.transform(X))[:, 1] for X in va_feats]

    # Select the smoothing (or use the one requested) by pooled TPR@0.1%FPR.
    if smooth_radius is not None:
        grid = [(smooth_radius, smooth_sigma if smooth_sigma is not None else 1.5)]
    else:
        grid = _SMOOTH_GRID

    best = {"tpr@0.1%fpr": -1.0}
    best_rs = (0, 0.0)
    for (r, s) in grid:
        pooled_scores, pooled_labels = [], []
        for d, raw in zip(val, raw_by_doc):
            sc = list(raw) if r <= 0 else postprocess(list(raw), smooth_radius=r, smooth_sigma=s)
            pooled_scores.extend(sc)
            pooled_labels.extend(d.labels)
        m = evaluate_pooled(pooled_scores, pooled_labels)
        print(f"  smooth r={r} sigma={s}: TPR@0.1%FPR={m['tpr@0.1%fpr']:.4f} "
              f"AUC={m['auc']:.4f}")
        if m["tpr@0.1%fpr"] > best["tpr@0.1%fpr"]:
            best, best_rs = m, (r, s)

    r, s = best_rs
    print(f"[train] selected smoothing r={r} sigma={s} "
          f"(val TPR@0.1%FPR={best['tpr@0.1%fpr']:.4f})")

    # Persist model + config for predict.py.
    artifact = {
        "model": model, "scaler": scaler, "model_kind": model_kind,
        "use_kgw": use_kgw,
        "smooth_radius": r, "smooth_sigma": s,
        "feature_names": feature_names(with_kgw=use_kgw),
        "metrics_smoothed": best,
    }
    out = config.OUTPUT_DIR / f"calibrator_{model_kind}.pkl"
    with open(out, "wb") as fh:
        pickle.dump(artifact, fh)
    print(f"[train] saved {out}")

    # Write validation predictions (with the selected smoothing) for the eval script.
    val_pred = {}
    for d, raw in zip(val, raw_by_doc):
        sc = list(raw) if r <= 0 else postprocess(list(raw), smooth_radius=r, smooth_sigma=s)
        val_pred[d.document_id] = [min(1.0, max(0.0, float(x))) for x in sc]
    write_submission(val_pred, str(config.OUTPUT_DIR / "val_pred.jsonl"))
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fusion calibrator.")
    parser.add_argument("--model", default="gboost", choices=["logreg", "gboost"])
    parser.add_argument("--smooth-radius", type=int, default=None,
                        help="Fix the smoothing radius instead of searching on val.")
    parser.add_argument("--smooth-sigma", type=float, default=None)
    args = parser.parse_args()
    run(args.model, smooth_radius=args.smooth_radius, smooth_sigma=args.smooth_sigma)
