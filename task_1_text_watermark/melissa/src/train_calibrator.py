"""Phase 3 — fit several scorers and keep the one that wins on validation.

All heads consume the **correct, vendor-based** watermark signals (TextSeal / Gumbel-Max /
Unigram via the pinned repos, KGW via CUDA Philox):

  * ``logreg``  — logistic regression on the multi-scale windowed-z feature matrix
                  (Alexandre's proven head; low variance, ranks the tail well at 0.1% FPR).
  * ``gboost``  — histogram gradient boosting on the same features (nonlinear, but can
                  overfit the tail on few docs).
  * ``hmm``     — forward-backward posterior over per-scheme LLRs (span-aware).

For every head we also search the moving-average smoothing window and evaluate the exact
competition metric (pooled TPR @ 0.1 % FPR) on validation. The overall best
(head, window) is saved for ``predict.py`` — so we can never do worse than the best
option available.

Requires the pinned vendor repos (``scripts/task1/sync_watermark_repos.sh`` on a login
node) and, for correct KGW, a CUDA GPU.

Usage:
    python -m src.train_calibrator                  # try all heads, keep the best
    python -m src.train_calibrator --heads logreg   # restrict the search
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from . import config
from .correct_features import doc_feature_matrix, feature_names, kgw_available, kgw_signal
from .evaluate import evaluate_pooled
from .load_data import load_split
from .pipeline import write_submission
from .postprocess import apply_smoothing

# Moving-average smoothing windows searched on validation (1 = no smoothing).
_MA_WINDOWS = (1, 5, 9, 15, 21, 31, 41)


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


def _doc_features(docs, use_kgw: bool):
    feats = []
    total = len(docs)
    for i, d in enumerate(docs, 1):
        feats.append(doc_feature_matrix(d.token_ids, use_kgw=use_kgw))
        if i % 20 == 0 or i == total:
            print(f"  features {i}/{total}", flush=True)
    return feats


def _best_window(raw_by_doc, docs) -> tuple[int, dict]:
    """Search the MA window that maximises pooled TPR@0.1%FPR on `docs`."""
    best_w, best_m = 1, {"tpr@0.1%fpr": -1.0}
    for w in _MA_WINDOWS:
        pooled_s, pooled_y = [], []
        for d, raw in zip(docs, raw_by_doc):
            pooled_s.extend(apply_smoothing(list(raw), w))
            pooled_y.extend(d.labels)
        m = evaluate_pooled(pooled_s, pooled_y)
        print(f"    window={w:>2}: TPR@0.1%FPR={m['tpr@0.1%fpr']:.4f} AUC={m['auc']:.4f}")
        if m["tpr@0.1%fpr"] > best_m["tpr@0.1%fpr"]:
            best_w, best_m = w, m
    return best_w, best_m


def run(heads: list[str] | None = None) -> dict:
    config.ensure_dirs()
    np.random.seed(config.SEED)
    heads = heads or ["logreg", "gboost", "hmm"]
    use_kgw = kgw_available()
    if not use_kgw:
        print("[train] WARNING: KGW unavailable (no CUDA / vendor) — training without "
              "KGW. Run on a GPU node for the full signal.")

    train = load_split("train")
    val = load_split("validation")

    # Feature matrices for the calibrator heads.
    print("[train] building features (train) ...")
    tr_feats = _doc_features(train, use_kgw)
    print("[train] building features (val) ...")
    va_feats = _doc_features(val, use_kgw)
    Xtr = np.vstack(tr_feats)
    ytr = np.concatenate([np.asarray(d.labels) for d in train])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    candidates: list[dict] = []

    for kind in [h for h in heads if h in ("logreg", "gboost")]:
        print(f"[train] head={kind}: fitting on X={Xtr.shape} ...")
        model = _make_model(kind)
        model.fit(Xtr_s, ytr)
        raw_by_doc = [model.predict_proba(scaler.transform(X))[:, 1] for X in va_feats]
        w, m = _best_window(raw_by_doc, val)
        print(f"[train] head={kind}: best window={w} val TPR@0.1%FPR={m['tpr@0.1%fpr']:.4f}")
        candidates.append({
            "scorer": "calibrator", "model_kind": kind, "model": model, "scaler": scaler,
            "smooth_window": w, "use_kgw": use_kgw, "metrics": m,
            "feature_names": feature_names(with_kgw=use_kgw),
        })

    if "hmm" in heads:
        try:
            from hmm_scorer import fit_hmm, posterior_scores

            print("[train] head=hmm: building KGW extra signals ...")
            kgw_tr = {d.document_id: kgw_signal(d.token_ids) for d in train} if use_kgw else {}
            kgw_va = {d.document_id: kgw_signal(d.token_ids) for d in val} if use_kgw else {}
            train_records = [{"document_id": d.document_id, "token_ids": list(d.token_ids),
                              "labels": list(d.labels)} for d in train]
            extra = {"kgw": kgw_tr} if use_kgw else None
            print("[train] head=hmm: fitting forward-backward LLRs ...")
            hmm = fit_hmm(train_records, extra_signals=extra)
            raw_by_doc = []
            for d in val:
                ex = {"kgw": kgw_va[d.document_id]} if use_kgw else None
                raw_by_doc.append(np.asarray(posterior_scores(hmm, list(d.token_ids), ex)))
            w, m = _best_window(raw_by_doc, val)
            print(f"[train] head=hmm: best window={w} val TPR@0.1%FPR={m['tpr@0.1%fpr']:.4f}")
            candidates.append({
                "scorer": "hmm", "hmm": hmm, "smooth_window": w,
                "use_kgw": use_kgw, "metrics": m,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[train] head=hmm skipped ({exc})")

    if not candidates:
        raise RuntimeError("no scorer head produced a result")

    best = max(candidates, key=lambda c: c["metrics"]["tpr@0.1%fpr"])
    tag = best.get("model_kind", best["scorer"])
    print(f"[train] SELECTED head={tag} window={best['smooth_window']} "
          f"val TPR@0.1%FPR={best['metrics']['tpr@0.1%fpr']:.4f}")

    out = config.OUTPUT_DIR / "scorer.pkl"
    with open(out, "wb") as fh:
        pickle.dump(best, fh)
    print(f"[train] saved {out}")

    # Write validation predictions for the eval script / dashboard viewer.
    w = best["smooth_window"]
    val_pred = {}
    if best["scorer"] == "calibrator":
        model, scaler = best["model"], best["scaler"]
        for d, X in zip(val, va_feats):
            raw = model.predict_proba(scaler.transform(X))[:, 1]
            val_pred[d.document_id] = apply_smoothing(list(raw), w)
    else:
        from hmm_scorer import posterior_scores
        for d in val:
            ex = {"kgw": kgw_signal(d.token_ids)} if best["use_kgw"] else None
            raw = posterior_scores(best["hmm"], list(d.token_ids), ex)
            val_pred[d.document_id] = apply_smoothing(list(raw), w)
    write_submission(val_pred, str(config.OUTPUT_DIR / "val_pred.jsonl"))
    return best["metrics"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit scorers and keep the best on val.")
    parser.add_argument("--heads", nargs="+", default=None,
                        choices=["logreg", "gboost", "hmm"],
                        help="Restrict which heads to try (default: all).")
    args = parser.parse_args()
    run(heads=args.heads)
