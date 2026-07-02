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
from smm_scorer import (default_params, doc_exact_signals, doc_signals,
                        read_jsonl, score_document)

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
    # length-prior robustness (Phase B): uniform mixing / extended range
    "b50_ps2_elo_mix10": dict(emission_mode="binned", n_bins=50, clip=4.0,
                              p_span_scale=2.0, edge_prior=np.log(0.005),
                              mix_uniform=0.1),
    "b50_ps2_elo_mix20": dict(emission_mode="binned", n_bins=50, clip=4.0,
                              p_span_scale=2.0, edge_prior=np.log(0.005),
                              mix_uniform=0.2),
    "b50_ps2_elo_mix10_ext": dict(emission_mode="binned", n_bins=50, clip=4.0,
                                  p_span_scale=2.0, edge_prior=np.log(0.005),
                                  mix_uniform=0.1, len_range=(20, 401)),
    "b50_ps2_elo_unif": dict(emission_mode="binned", n_bins=50, clip=4.0,
                             p_span_scale=2.0, edge_prior=np.log(0.005),
                             mix_uniform=1.0),
    # entropy weighting (Phase C) — requires output/entropy_{split}.npz
    "b50_ps2_elo_ent_lin": dict(emission_mode="binned", n_bins=50, clip=4.0,
                                p_span_scale=2.0, edge_prior=np.log(0.005),
                                entropy_kind="linear"),
    "b50_ps2_elo_ent_sqrt": dict(emission_mode="binned", n_bins=50, clip=4.0,
                                 p_span_scale=2.0, edge_prior=np.log(0.005),
                                 entropy_kind="sqrt"),
    # entropy-conditioned LLR tables (learned, replaces multiplicative weights)
    "b30_ps2_elo_entbin3": dict(emission_mode="binned_ent", n_ent_bins=3,
                                n_bins=30, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b50_ps2_elo_entbin3": dict(emission_mode="binned_ent", n_ent_bins=3,
                                n_bins=50, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b30_ps2_elo_entbin5": dict(emission_mode="binned_ent", n_ent_bins=5,
                                n_bins=30, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b50_ps2_elo_entbin5": dict(emission_mode="binned_ent", n_ent_bins=5,
                                n_bins=50, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b20_ps2_elo_entbin7": dict(emission_mode="binned_ent", n_ent_bins=7,
                                n_bins=20, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b30_ps2_elo_entbin7": dict(emission_mode="binned_ent", n_ent_bins=7,
                                n_bins=30, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    # refinement around the entbin5 winner (0.3685 CV, 0.5B-proxy entropy)
    "b40_ps2_elo_entbin4": dict(emission_mode="binned_ent", n_ent_bins=4,
                                n_bins=40, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b40_ps2_elo_entbin5": dict(emission_mode="binned_ent", n_ent_bins=5,
                                n_bins=40, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b50_ps2_elo_entbin4": dict(emission_mode="binned_ent", n_ent_bins=4,
                                n_bins=50, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b50_ps2_elo_entbin6": dict(emission_mode="binned_ent", n_ent_bins=6,
                                n_bins=50, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b60_ps2_elo_entbin5": dict(emission_mode="binned_ent", n_ent_bins=5,
                                n_bins=60, clip=4.0, p_span_scale=2.0,
                                edge_prior=np.log(0.005)),
    "b50_ps3_elo_entbin5": dict(emission_mode="binned_ent", n_ent_bins=5,
                                n_bins=50, clip=4.0, p_span_scale=3.0,
                                edge_prior=np.log(0.005)),
    "b50_ps2_evlo_entbin5": dict(emission_mode="binned_ent", n_ent_bins=5,
                                 n_bins=50, clip=4.0, p_span_scale=2.0,
                                 edge_prior=np.log(0.002)),
    # closed-form Gumbel-max / TextSeal LLR from realized-token probability
    # p_t (7B forward pass, requires output/logp_{split}.npz) - point 1 of
    # docs/task1/attempt1.md §"prochaines etapes". Ensemble with the binned
    # emissions (logsumexp already treats them as alternative hypotheses).
    "binned50_edge_lo_ps2_exact": dict(emission_mode="binned", n_bins=50,
                                       clip=4.0, p_span_scale=2.0,
                                       edge_prior=np.log(0.005),
                                       use_exact=True),
    "binned50_edge_lo_ps2_exactonly": dict(emission_mode="binned", n_bins=50,
                                           clip=4.0, p_span_scale=2.0,
                                           edge_prior=np.log(0.005),
                                           use_exact=True, exact_only=True),
    "binned50_edge_lo_ps2_exact_pmin3": dict(emission_mode="binned", n_bins=50,
                                             clip=4.0, p_span_scale=2.0,
                                             edge_prior=np.log(0.005),
                                             use_exact=True, exact_p_min=1e-3),
    "binned50_edge_lo_ps2_exact_pmin5": dict(emission_mode="binned", n_bins=50,
                                             clip=4.0, p_span_scale=2.0,
                                             edge_prior=np.log(0.005),
                                             use_exact=True, exact_p_min=1e-5),
    "binned50_edge_lo_ps2_exact_clip4": dict(emission_mode="binned", n_bins=50,
                                             clip=4.0, p_span_scale=2.0,
                                             edge_prior=np.log(0.005),
                                             use_exact=True, exact_clip=4.0),
    "binned50_edge_lo_ps2_exact_clip12": dict(emission_mode="binned", n_bins=50,
                                              clip=4.0, p_span_scale=2.0,
                                              edge_prior=np.log(0.005),
                                              use_exact=True, exact_clip=12.0),
    # exact + entropy-binned combo (best of both, once both npz exist)
    "b50_ps2_elo_entbin5_exact": dict(emission_mode="binned_ent", n_ent_bins=5,
                                      n_bins=50, clip=4.0, p_span_scale=2.0,
                                      edge_prior=np.log(0.005),
                                      use_exact=True),
    # boundary-bleed fix: forbid two watermarked spans back-to-back with no
    # clean token between them (audit finding, docs/task1/attempt1.md §17) —
    # lossless w.r.t. ground-truth segmentation, isolated A/B vs each winner
    "binned50_edge_lo_ps2_noadj": dict(emission_mode="binned", n_bins=50,
                                       clip=4.0, p_span_scale=2.0,
                                       edge_prior=np.log(0.005),
                                       forbid_adjacent_spans=True),
    "b50_ps2_elo_entbin5_noadj": dict(emission_mode="binned_ent", n_ent_bins=5,
                                      n_bins=50, clip=4.0, p_span_scale=2.0,
                                      edge_prior=np.log(0.005),
                                      forbid_adjacent_spans=True),
    # isotonic-smoothed binned LLR (audit finding: tail bins fixing the
    # 0.1%FPR threshold are estimated from very few labeled tokens)
    "binned50_edge_lo_ps2_iso": dict(emission_mode="binned", n_bins=50,
                                     clip=4.0, p_span_scale=2.0,
                                     edge_prior=np.log(0.005),
                                     isotonic_llr=True),
    "b50_ps2_elo_entbin5_iso": dict(emission_mode="binned_ent", n_ent_bins=5,
                                    n_bins=50, clip=4.0, p_span_scale=2.0,
                                    edge_prior=np.log(0.005),
                                    isotonic_llr=True),
    "b50_ps2_elo_entbin5_iso_noadj": dict(emission_mode="binned_ent", n_ent_bins=5,
                                          n_bins=50, clip=4.0, p_span_scale=2.0,
                                          edge_prior=np.log(0.005),
                                          isotonic_llr=True,
                                          forbid_adjacent_spans=True),
    # non-uniform mixture weights by scheme prevalence among labeled spans
    # (audit: ~41/36/22% GM/TS/KGW, not 1/n_hyp each)
    "binned50_edge_lo_ps2_mix": dict(emission_mode="binned", n_bins=50,
                                     clip=4.0, p_span_scale=2.0,
                                     edge_prior=np.log(0.005),
                                     mix_from_prevalence=True),
    "b50_ps2_elo_entbin5_mix": dict(emission_mode="binned_ent", n_ent_bins=5,
                                    n_bins=50, clip=4.0, p_span_scale=2.0,
                                    edge_prior=np.log(0.005),
                                    mix_from_prevalence=True),
    "b50_ps2_elo_entbin5_iso_mix": dict(emission_mode="binned_ent", n_ent_bins=5,
                                        n_bins=50, clip=4.0, p_span_scale=2.0,
                                        edge_prior=np.log(0.005),
                                        isotonic_llr=True,
                                        mix_from_prevalence=True),
    # best-of-both: entropy-conditioned + isotonic + exact Gumbel/TextSeal LLR
    # (7B logp_target, output/logp_{split}.npz) - combines every validated
    # win once all three npz sources (kgw, entropy, logp) are 7B-based.
    "b50_ps2_elo_entbin5_iso_exact": dict(emission_mode="binned_ent", n_ent_bins=5,
                                          n_bins=50, clip=4.0, p_span_scale=2.0,
                                          edge_prior=np.log(0.005),
                                          isotonic_llr=True, use_exact=True),
    # closed-form KGW / Unigram LLR from the LM's boosted green-mass at each
    # position (output/{kgw,unigram}_lpg_{split}.npz, 7B forward pass) —
    # replaces the fitted Bernoulli rate that caps KGW at 9% span detection
    # (audit §17). No fitting needed: pure closed-form given gamma/delta.
    "b50_ps2_elo_entbin5_iso_kgwx": dict(emission_mode="binned_ent", n_ent_bins=5,
                                         n_bins=50, clip=4.0, p_span_scale=2.0,
                                         edge_prior=np.log(0.005),
                                         isotonic_llr=True, use_exact_kgw=True),
    "b50_ps2_elo_entbin5_iso_unix": dict(emission_mode="binned_ent", n_ent_bins=5,
                                         n_bins=50, clip=4.0, p_span_scale=2.0,
                                         edge_prior=np.log(0.005),
                                         isotonic_llr=True, use_exact_unigram=True),
    "b50_ps2_elo_entbin5_iso_bothx": dict(emission_mode="binned_ent", n_ent_bins=5,
                                          n_bins=50, clip=4.0, p_span_scale=2.0,
                                          edge_prior=np.log(0.005),
                                          isotonic_llr=True, use_exact_kgw=True,
                                          use_exact_unigram=True),
    "b50_ps2_elo_entbin5_iso_allx": dict(emission_mode="binned_ent", n_ent_bins=5,
                                         n_bins=50, clip=4.0, p_span_scale=2.0,
                                         edge_prior=np.log(0.005),
                                         isotonic_llr=True, use_exact=True,
                                         use_exact_kgw=True, use_exact_unigram=True),
    # down-weighted exact Gumbel/TextSeal mixture weight sweep (rescue the
    # signal without diluting the binned+entropy+isotonic mixture at 1/n_hyp)
    "b50_ps2_elo_entbin5_iso_exw01": dict(emission_mode="binned_ent", n_ent_bins=5,
                                          n_bins=50, clip=4.0, p_span_scale=2.0,
                                          edge_prior=np.log(0.005),
                                          isotonic_llr=True, use_exact=True,
                                          exact_mix_weight=0.1),
    "b50_ps2_elo_entbin5_iso_exw03": dict(emission_mode="binned_ent", n_ent_bins=5,
                                          n_bins=50, clip=4.0, p_span_scale=2.0,
                                          edge_prior=np.log(0.005),
                                          isotonic_llr=True, use_exact=True,
                                          exact_mix_weight=0.3),
    "b50_ps2_elo_entbin5_iso_exw003": dict(emission_mode="binned_ent", n_ent_bins=5,
                                           n_bins=50, clip=4.0, p_span_scale=2.0,
                                           edge_prior=np.log(0.005),
                                           isotonic_llr=True, use_exact=True,
                                           exact_mix_weight=0.03),
}


def load_labeled():
    docs = read_jsonl(DATA / "train.jsonl") + read_jsonl(DATA / "validation.jsonl")
    kgw = {}
    for split in ("train", "validation"):
        npz = np.load(KGW_DIR / f"kgw_{split}.npz")
        kgw.update({k: npz[k] for k in npz.files})
    return docs, kgw


def load_entropy(splits=("train", "validation")):
    """document_id -> entropy array, or None if the npz files are absent."""
    ent = {}
    for split in splits:
        path = KGW_DIR / f"entropy_{split}.npz"
        if not path.exists():
            return None
        npz = np.load(path)
        ent.update({k: npz[k].astype(np.float64) for k in npz.files})
    return ent


def load_p_target(splits=("train", "validation")):
    """document_id -> per-token model probability of the realized token
    (Aaronson/TextSeal closed-form detector input), or None if the
    logp_{split}.npz files are absent. Sentinel: -1.0 (no forward-pass
    context, converted from the +1.0 logp sentinel written by the GPU pass)."""
    pt = {}
    for split in splits:
        path = KGW_DIR / f"logp_{split}.npz"
        if not path.exists():
            return None
        npz = np.load(path)
        for k in npz.files:
            logp = npz[k].astype(np.float64)
            pt[k] = np.where(logp > 0.5, -1.0, np.exp(logp))
    return pt


def load_lpg(splits=("train", "validation")):
    """document_id -> {"kgw": arr, "unigram": arr} of per-token log P(green |
    context) under the boosted distribution, or None if the npz files
    (output/{kgw,unigram}_lpg_{split}.npz) are absent."""
    out = {}
    for stem in ("kgw_lpg", "unigram_lpg"):
        key = stem.split("_")[0]
        for split in splits:
            path = KGW_DIR / f"{stem}_{split}.npz"
            if not path.exists():
                return None
            npz = np.load(path)
            for did in npz.files:
                out.setdefault(did, {})[key] = npz[did].astype(np.float64)
    return out


def build_cache(docs, kgw):
    cache = {}
    for rec in docs:
        did = str(rec["document_id"])
        extra = {"kgw": kgw[did]} if did in kgw else None
        cache[did] = doc_signals(rec["token_ids"], extra)
    return cache


def build_exact_cache(docs):
    return {str(rec["document_id"]): doc_exact_signals(rec["token_ids"])
           for rec in docs}


def tpr_at_fpr(scores, labels, fpr=0.001):
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    clean = np.sort(s[y == 0])[::-1]
    k = max(int(len(clean) * fpr), 1)
    tau = clean[k - 1]
    return float((s[y == 1] >= tau).mean())


def _needs_entropy(kwargs):
    return kwargs is not None and (kwargs.get("entropy_kind") is not None
                                   or kwargs.get("emission_mode") == "binned_ent")


def _needs_exact(kwargs):
    return kwargs is not None and bool(kwargs.get("use_exact"))


def _needs_lpg(kwargs):
    return kwargs is not None and bool(kwargs.get("use_exact_kgw")
                                       or kwargs.get("use_exact_unigram"))


def eval_config(name, docs, kgw, cache, entropy=None, exact_cache=None,
                p_target=None, lpg=None):
    kwargs = CONFIGS[name]
    needs_entropy = _needs_entropy(kwargs)
    needs_exact = _needs_exact(kwargs)
    needs_lpg = _needs_lpg(kwargs)
    if needs_entropy and entropy is None:
        print(f"{name:24s} SKIPPED (entropy npz missing)", flush=True)
        return None
    if needs_exact and p_target is None:
        print(f"{name:24s} SKIPPED (logp npz missing)", flush=True)
        return None
    if needs_lpg and lpg is None:
        print(f"{name:24s} SKIPPED (kgw_lpg/unigram_lpg npz missing)", flush=True)
        return None
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
            params = fit_params(fit_docs, kgw=kgw, signal_cache=cache,
                                entropy_cache=entropy if needs_entropy else None,
                                **kwargs)
        for i in fold:
            rec = docs[i]
            did = str(rec["document_id"])
            sc = score_document(rec["token_ids"], params=params,
                                signals=cache[did],
                                entropy=entropy[did] if needs_entropy else None,
                                exact_signals=exact_cache[did] if needs_exact else None,
                                p_target=p_target[did] if needs_exact else None,
                                lpg=lpg[did] if needs_lpg else None)
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
    needs_entropy = _needs_entropy(kwargs)
    needs_exact = _needs_exact(kwargs)
    needs_lpg = _needs_lpg(kwargs)
    entropy = load_entropy() if needs_entropy else None
    if needs_entropy and entropy is None:
        raise SystemExit("entropy npz missing for labeled splits")
    p_target = load_p_target() if needs_exact else None
    if needs_exact and p_target is None:
        raise SystemExit("logp npz missing for labeled splits")
    exact_cache = build_exact_cache(docs) if needs_exact else None
    params = (default_params() if kwargs is None
              else fit_params(docs, kgw=kgw, signal_cache=cache,
                              entropy_cache=entropy, **kwargs))

    test = read_jsonl(DATA / "test.jsonl")
    kgw_test = np.load(KGW_DIR / "kgw_test.npz")
    ent_test = np.load(KGW_DIR / "entropy_test.npz") if needs_entropy else None
    logp_test = np.load(KGW_DIR / "logp_test.npz") if needs_exact else None
    kgw_lpg_test = np.load(KGW_DIR / "kgw_lpg_test.npz") if needs_lpg else None
    uni_lpg_test = np.load(KGW_DIR / "unigram_lpg_test.npz") if needs_lpg else None
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(test):
            did = str(rec["document_id"])
            extra = {"kgw": kgw_test[did]} if did in kgw_test.files else None
            ent = (ent_test[did].astype(np.float64)
                   if ent_test is not None and did in ent_test.files else None)
            pt = None
            if needs_exact:
                logp = logp_test[did].astype(np.float64)
                pt = np.where(logp > 0.5, -1.0, np.exp(logp))
            exact_sig = (doc_exact_signals(rec["token_ids"])
                        if needs_exact else None)
            lpg_doc = None
            if needs_lpg:
                lpg_doc = {"kgw": kgw_lpg_test[did].astype(np.float64),
                          "unigram": uni_lpg_test[did].astype(np.float64)}
            sc = score_document(rec["token_ids"], extra=extra, params=params,
                                entropy=ent, exact_signals=exact_sig,
                                p_target=pt, lpg=lpg_doc)
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
    entropy = load_entropy()
    p_target = load_p_target()
    lpg = load_lpg()
    exact_cache = build_exact_cache(docs) if p_target is not None else None
    names = args.configs if args.configs else (list(CONFIGS) if args.grid
                                               else ["baseline"])
    results = {}
    for name in names:
        r = eval_config(name, docs, kgw, cache, entropy=entropy,
                        exact_cache=exact_cache, p_target=p_target, lpg=lpg)
        if r is not None:
            results[name] = r
    best = max(results, key=results.get)
    print(f"\nBest: {best} ({results[best]:.4f})")


if __name__ == "__main__":
    main()
