"""Distribution-shift diagnostic: labeled (held-out CV) vs test score behavior.

No test labels needed. Compares, at matched score thresholds:
- per-token score quantiles
- run-length histograms of contiguous tokens above threshold (detected spans):
  if the test set contains span lengths absent from train+val (e.g. ~200),
  its run-length histogram will show mass away from the canonical lengths.
- fraction of docs with no detection

Usage: python diagnose_shift.py [--test submission_smm_cv.jsonl]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from cv_smm import CONFIGS, DATA, N_FOLDS, SEED, build_cache, load_labeled
from fit_smm import fit_params
from smm_scorer import read_jsonl, score_document

WINNER = "binned50_edge_lo_ps2"
CANONICAL = (31, 47, 63, 95, 159, 320)
TOL = 4  # posterior runs bleed a little around true span edges


def heldout_scores():
    docs, kgw = load_labeled()
    cache = build_cache(docs, kgw)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(docs))
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]
    scores, labels = [], []
    for fold in folds:
        held = set(fold.tolist())
        fit_docs = [docs[i] for i in range(len(docs)) if i not in held]
        params = fit_params(fit_docs, kgw=kgw, signal_cache=cache,
                            **CONFIGS[WINNER])
        for i in fold:
            rec = docs[i]
            sc = score_document(rec["token_ids"], params=params,
                                signals=cache[str(rec["document_id"])])
            scores.append(sc)
            labels.append(np.array(rec["labels"]))
    return scores, labels


def run_lengths(doc_scores, tau):
    """Lengths of contiguous runs of score >= tau, per corpus."""
    lengths = []
    n_docs_hit = 0
    for sc in doc_scores:
        hit = False
        for val, grp in itertools.groupby(sc >= tau):
            L = len(list(grp))
            if val:
                lengths.append(L)
                hit = True
        n_docs_hit += hit
    return np.array(lengths), n_docs_hit


def band_report(lengths, label):
    if len(lengths) == 0:
        print(f"  {label}: no runs")
        return
    canon = np.zeros(len(lengths), dtype=bool)
    for c in CANONICAL:
        canon |= np.abs(lengths - c) <= TOL
    short = lengths < CANONICAL[0] - TOL
    other = ~canon & ~short
    print(f"  {label}: {len(lengths)} runs | canonical+-{TOL}: {canon.mean():.1%}"
          f" | short(<{CANONICAL[0] - TOL}): {short.mean():.1%}"
          f" | non-canonical>=27: {other.mean():.1%}")
    vals, counts = np.unique(lengths[other], return_counts=True)
    top = sorted(zip(counts, vals), reverse=True)[:10]
    print(f"    top non-canonical lengths: {[(int(v), int(c)) for c, v in top]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="submission_smm_cv.jsonl")
    args = ap.parse_args()

    print("Scoring labeled docs held-out (5-fold, winner config)...", flush=True)
    lab_scores, lab_labels = heldout_scores()
    y = np.concatenate(lab_labels)
    s = np.concatenate(lab_scores)

    test_recs = read_jsonl(args.test)
    test_scores = [np.array(r["scores"]) for r in test_recs]
    ts = np.concatenate(test_scores)

    qs = (50, 90, 99, 99.9)
    print("\ntoken score quantiles (labeled held-out vs test):")
    for q in qs:
        print(f"  p{q}: labeled={np.percentile(s, q):.4f}  test={np.percentile(ts, q):.4f}")
    print(f"  max: labeled={s.max():.4f}  test={ts.max():.4f}")
    print(f"docs: labeled n={len(lab_scores)} len_mean={np.mean([len(x) for x in lab_scores]):.0f}"
          f" | test n={len(test_scores)} len_mean={np.mean([len(x) for x in test_scores]):.0f}")

    # threshold tau: pooled labeled clean at 0.1% FPR (the metric's operating point)
    clean = np.sort(s[y == 0])[::-1]
    tau_strict = clean[max(int(len(clean) * 0.001), 1) - 1]
    tau_soft = clean[max(int(len(clean) * 0.01), 1) - 1]
    frac_labeled = float((s >= tau_strict).mean())
    frac_test = float((ts >= tau_strict).mean())
    print(f"\ntau@0.1%FPR(labeled)={tau_strict:.6f}: frac tokens above -> "
          f"labeled={frac_labeled:.4f}  test={frac_test:.4f}")

    for tau, nm in ((tau_strict, "tau strict (0.1% FPR)"), (tau_soft, "tau soft (1% FPR)")):
        print(f"\nrun-length analysis at {nm} = {tau:.6f}:")
        ll, lab_hit = run_lengths(lab_scores, tau)
        tl, test_hit = run_lengths(test_scores, tau)
        band_report(ll, "labeled")
        band_report(tl, "test   ")
        print(f"  docs with >=1 run: labeled {lab_hit}/{len(lab_scores)}"
              f" | test {test_hit}/{len(test_scores)}")

    # long-run check: any test runs far beyond the max canonical length?
    if len(tl):
        long_runs = tl[tl > 320 + TOL]
        print(f"\ntest runs longer than 320+{TOL}: {len(long_runs)}"
              f" (max={long_runs.max() if len(long_runs) else 0})")


if __name__ == "__main__":
    main()
