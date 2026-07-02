"""Supervised fitting of SmmParams from labeled documents.

- Priors (span rate, length distribution) estimated by direct counting.
- Continuous emissions (TextSeal, Gumbel-Max): empirical binned LLR fitted on
  H1 pools (tokens of spans assigned to the scheme by window z >= 3) vs the
  H0 pool (all clean tokens). Only valid positions (first n-gram occurrence)
  enter the pools.
- KGW (binary): closed-form Bernoulli LLR with H1 green rate from KGW spans.
"""

from __future__ import annotations

import itertools

import numpy as np

from detectors import H0_MOMENTS
from smm_scorer import (LEN_RANGE, Emission, SmmParams, default_length_prior,
                        doc_signals)

ASSIGN_Z = 3.0
MIN_SPAN_FIT = 20


def iter_spans(labels):
    pos = 0
    for k, grp in itertools.groupby(labels):
        L = len(list(grp))
        yield k, pos, pos + L
        pos += L


def collect_pools(docs, kgw=None, signal_cache=None):
    """Return (h1 pools, h0 pools, span lengths, counters) from labeled docs."""
    from collections import defaultdict
    h1 = defaultdict(list)
    h0 = defaultdict(list)
    span_lengths = []
    n_tokens = n_spans = n_edge = 0
    for rec in docs:
        did = str(rec["document_id"])
        extra = {"kgw": kgw[did]} if kgw is not None and did in kgw else None
        signals = (signal_cache[did] if signal_cache is not None
                   else doc_signals(rec["token_ids"], extra))
        lab = rec["labels"]
        n = len(lab)
        n_tokens += n
        for name, (sig, valid) in signals.items():
            mask = valid & (np.array(lab) == 0)
            h0[name].append(sig[mask])
        for k, s, e in iter_spans(lab):
            if k != 1:
                continue
            n_spans += 1
            span_lengths.append(e - s)
            if s == 0 or e == n:
                n_edge += 1
            if e - s < MIN_SPAN_FIT:
                continue
            # assign span to the scheme with the strongest window z
            best_name, best_z = None, ASSIGN_Z
            for name, (sig, valid) in signals.items():
                mu0, var0 = H0_MOMENTS[name]
                v = valid[s:e]
                if v.sum() < 5:
                    continue
                x = (sig[s:e][v] - mu0) / np.sqrt(var0)
                z = x.sum() / np.sqrt(len(x))
                if z >= best_z:
                    best_name, best_z = name, z
            if best_name is not None:
                sig, valid = signals[best_name]
                h1[best_name].append(sig[s:e][valid[s:e]])
    h1 = {k: (np.concatenate(v) if v else np.array([])) for k, v in h1.items()}
    h0 = {k: (np.concatenate(v) if v else np.array([])) for k, v in h0.items()}
    stats = {"n_tokens": n_tokens, "n_spans": n_spans, "n_edge": n_edge,
             "n_docs": len(docs)}
    return h1, h0, np.array(span_lengths), stats


def fit_length_prior(span_lengths, pseudo=0.05):
    lengths = np.arange(*LEN_RANGE)
    counts = np.array([(span_lengths == L).sum() for L in lengths], dtype=float)
    mass = (counts + pseudo) / (counts.sum() + pseudo * len(lengths))
    return lengths, np.log(mass)


def fit_binned_llr(h0_vals, h1_vals, n_bins=30, clip=4.0, pseudo=1.0) -> Emission:
    edges = np.quantile(h0_vals, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    B = len(edges) - 1
    idx0 = np.clip(np.searchsorted(edges, h0_vals, side="right") - 1, 0, B - 1)
    idx1 = np.clip(np.searchsorted(edges, h1_vals, side="right") - 1, 0, B - 1)
    c0 = np.bincount(idx0, minlength=B).astype(float)
    c1 = np.bincount(idx1, minlength=B).astype(float)
    llr = (np.log(c1 + pseudo) - np.log(c1.sum() + pseudo * B)
           - np.log(c0 + pseudo) + np.log(c0.sum() + pseudo * B))
    return Emission(kind="binned", edges=edges, llr=np.clip(llr, -clip, clip))


def fit_bernoulli_kgw(h1_vals, gamma=0.25) -> Emission:
    p1 = float(np.clip(h1_vals.mean(), gamma + 0.01, 0.95)) if len(h1_vals) else gamma + 0.01
    return Emission(kind="bernoulli",
                    llr_green=float(np.log(p1 / gamma)),
                    llr_red=float(np.log((1 - p1) / (1 - gamma))))


def fit_params(docs, kgw=None, signal_cache=None, emission_mode="binned",
               n_bins=30, clip=4.0, fit_priors=True, shifts=None,
               neutral_invalid=False, p_span_scale=1.0,
               edge_prior=None, include_unigram=False) -> SmmParams:
    h1, h0, span_lengths, stats = collect_pools(docs, kgw, signal_cache)

    if fit_priors:
        lengths, log_len_prior = fit_length_prior(span_lengths)
        log_p_span = np.log(p_span_scale * stats["n_spans"] / stats["n_tokens"])
    else:
        lengths, log_len_prior = default_length_prior()
        log_p_span = np.log(p_span_scale * 0.004)
    if edge_prior is None:
        # fraction of spans truncated by a document boundary
        edge_prior = np.log(max(stats["n_edge"], 1) / stats["n_spans"])

    emissions = {}
    if emission_mode == "gaussian":
        from smm_scorer import DEFAULT_SHIFTS
        shifts = shifts or DEFAULT_SHIFTS
        for name, s in shifts.items():
            emissions[name] = Emission(kind="gaussian", shifts=s,
                                       neutral_invalid=neutral_invalid)
    elif emission_mode == "binned":
        for name in ("textseal", "gumbelmax"):
            if len(h1.get(name, ())) >= 500:
                emissions[name] = fit_binned_llr(h0[name], h1[name], n_bins, clip)
        emissions["kgw"] = fit_bernoulli_kgw(h1.get("kgw", np.array([])))
        if include_unigram:
            # too few labeled Unigram spans to fit; keep the conservative
            # hand-set single hypothesis (see smm_scorer.DEFAULT_SHIFTS)
            emissions["unigram"] = Emission(kind="gaussian", shifts=(0.5,))
    else:
        raise ValueError(emission_mode)

    return SmmParams(lengths=lengths, log_len_prior=log_len_prior,
                     log_p_span=float(log_p_span), edge_prior=float(edge_prior),
                     emissions=emissions)
