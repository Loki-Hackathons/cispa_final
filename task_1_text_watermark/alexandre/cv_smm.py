"""5-fold document-level CV harness + final fit/score for the SMM pipeline.

Usage:
  python cv_smm.py --grid                 # run the predefined config grid
  python cv_smm.py --final CONFIG --out submission.jsonl   # fit on 180 docs, score test

Metric: pooled TPR @ 0.1% FPR over held-out documents only (each doc scored
by a model fitted without it). Decision rule for the whole plan: accept a
change only if this metric improves over the current-pipeline baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fit_smm import fit_params
from smm_scorer import default_params, doc_signals, read_jsonl, score_document

DATA = Path(__file__).resolve().parents[2] / "data" / "watermark_localization"
KGW_DIR = Path(__file__).resolve().parent / "output"
SEED = 0
N_FOLDS = 5

# name -> kwargs for fit_params (None = unfitted default params, baseline)
CONFIGS = {
    "baseline": None,
    "gauss_fitpriors": dict(emission_mode="gaussian", fit_priors=True),
    "gauss_neutral": dict(emission_mode="gaussian", fit_priors=True,
                          neutral_invalid=True),
    "binned30_c4": dict(emission_mode="binned", n_bins=30, clip=4.0),
    "binned20_c4": dict(emission_mode="binned", n_bins=20, clip=4.0),
    "binned40_c4": dict(emission_mode="binned", n_bins=40, clip=4.0),
    "binned30_c3": dict(emission_mode="binned", n_bins=30, clip=3.0),
    "binned30_c6": dict(emission_mode="binned", n_bins=30, clip=6.0),
    "binned30_c4_ps05": dict(emission_mode="binned", n_bins=30, clip=4.0,
                             p_span_scale=0.5),
    "binned30_c4_ps2": dict(emission_mode="binned", n_bins=30, clip=4.0,
                            p_span_scale=2.0),
    "binned30_c4_edge_lo": dict(emission_mode="binned", n_bins=30, clip=4.0,
                                edge_prior=np.log(0.005)),
    "binned30_c4_edge_hi": dict(emission_mode="binned", n_bins=30, clip=4.0,
                                edge_prior=np.log(0.08)),
    # combination round (after first grid: binned>gauss, ps2 and edge_lo help)
    "binned30_edge_lo_ps2": dict(emission_mode="binned", n_bins=30, clip=4.0,
                                 p_span_scale=2.0, edge_prior=np.log(0.005)),
    "binned40_edge_lo": dict(emission_mode="binned", n_bins=40, clip=4.0,
                             edge_prior=np.log(0.005)),
    "binned40_edge_lo_ps2": dict(emission_mode="binned", n_bins=40, clip=4.0,
                                 p_span_scale=2.0, edge_prior=np.log(0.005)),
    "binned30_edge_vlo": dict(emission_mode="binned", n_bins=30, clip=4.0,
                              edge_prior=np.log(0.001)),
    # final refinement around binned40_edge_lo_ps2
    "binned40_edge_lo_ps3": dict(emission_mode="binned", n_bins=40, clip=4.0,
                                 p_span_scale=3.0, edge_prior=np.log(0.005)),
    "binned50_edge_lo_ps2": dict(emission_mode="binned", n_bins=50, clip=4.0,
                                 p_span_scale=2.0, edge_prior=np.log(0.005)),
    "binned40_edge_vlo_ps2": dict(emission_mode="binned", n_bins=40, clip=4.0,
                                  p_span_scale=2.0, edge_prior=np.log(0.001)),
    # winner + unigram emission re-enabled (parallel-agent integration, #408)
    "binned50_edge_lo_ps2_uni": dict(emission_mode="binned", n_bins=50, clip=4.0,
                                     p_span_scale=2.0, edge_prior=np.log(0.005),
                                     include_unigram=True),
}


def load_labeled():
    docs = read_jsonl(DATA / "train.jsonl") + read_jsonl(DATA / "validation.jsonl")
    kgw = {}
    for split in ("train", "validation"):
        npz = np.load(KGW_DIR / f"kgw_{split}.npz")
        kgw.update({k: npz[k] for k in npz.files})
    return docs, kgw


def build_cache(docs, kgw):
    cache = {}
    for rec in docs:
        did = str(rec["document_id"])
        extra = {"kgw": kgw[did]} if did in kgw else None
        cache[did] = doc_signals(rec["token_ids"], extra)
    return cache


def tpr_at_fpr(scores, labels, fpr=0.001):
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    clean = np.sort(s[y == 0])[::-1]
    k = max(int(len(clean) * fpr), 1)
    tau = clean[k - 1]
    return float((s[y == 1] >= tau).mean())


def eval_config(name, docs, kgw, cache):
    kwargs = CONFIGS[name]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(docs))
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]

    scores, labels = [], []
    t0 = time.time()
    for fi, fold in enumerate(folds):
        held = set(fold.tolist())
        if kwargs is None:
            params = default_params()
        else:
            fit_docs = [docs[i] for i in range(len(docs)) if i not in held]
            params = fit_params(fit_docs, kgw=kgw, signal_cache=cache, **kwargs)
        for i in fold:
            rec = docs[i]
            did = str(rec["document_id"])
            sc = score_document(rec["token_ids"], params=params,
                                signals=cache[did])
            scores.append(sc)
            labels.append(np.array(rec["labels"]))
    tpr = tpr_at_fpr(scores, labels)
    print(f"{name:24s} CV TPR@0.1%FPR = {tpr:.4f}   ({time.time() - t0:.0f}s)",
          flush=True)
    return tpr


def run_final(config, out_path):
    docs, kgw = load_labeled()
    cache = build_cache(docs, kgw)
    kwargs = CONFIGS[config]
    params = (default_params() if kwargs is None
              else fit_params(docs, kgw=kgw, signal_cache=cache, **kwargs))

    test = read_jsonl(DATA / "test.jsonl")
    kgw_test = np.load(KGW_DIR / "kgw_test.npz")
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(test):
            did = str(rec["document_id"])
            extra = {"kgw": kgw_test[did]} if did in kgw_test.files else None
            sc = score_document(rec["token_ids"], extra=extra, params=params)
            f.write(json.dumps({"document_id": rec["document_id"],
                                "scores": [float(s) for s in sc]}) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(test)} ({(i + 1) / (time.time() - t0):.1f} docs/s)",
                      flush=True)
    print(f"Wrote {len(test)} docs to {out_path} in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--final", default=None, metavar="CONFIG")
    ap.add_argument("--out", default="submission_cv.jsonl")
    args = ap.parse_args()

    if args.final:
        run_final(args.final, args.out)
        return

    docs, kgw = load_labeled()
    print(f"{len(docs)} labeled docs; building signal cache...", flush=True)
    cache = build_cache(docs, kgw)
    names = args.configs if args.configs else (list(CONFIGS) if args.grid
                                               else ["baseline"])
    results = {}
    for name in names:
        results[name] = eval_config(name, docs, kgw, cache)
    best = max(results, key=results.get)
    print(f"\nBest: {best} ({results[best]:.4f})")


if __name__ == "__main__":
    main()
