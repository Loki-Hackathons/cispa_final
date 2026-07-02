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


def collect_pools(docs, kgw=None, signal_cache=None, entropy_cache=None):
    """Return (h1 pools, h0 pools, span lengths, counters) from labeled docs.

    With entropy_cache, also returns parallel entropy pools h1_ent / h0_ent
    (same masks, aligned element-wise with the signal pools)."""
    from collections import Counter, defaultdict
    h1 = defaultdict(list)
    h0 = defaultdict(list)
    h1_ent = defaultdict(list)
    h0_ent = defaultdict(list)
    span_lengths = []
    scheme_counts = Counter()
    n_tokens = n_spans = n_edge = 0
    for rec in docs:
        did = str(rec["document_id"])
        extra = {"kgw": kgw[did]} if kgw is not None and did in kgw else None
        signals = (signal_cache[did] if signal_cache is not None
                   else doc_signals(rec["token_ids"], extra))
        ent = (entropy_cache[did].astype(np.float64)
               if entropy_cache is not None else None)
        lab = rec["labels"]
        n = len(lab)
        n_tokens += n
        for name, (sig, valid) in signals.items():
            mask = valid & (np.array(lab) == 0)
            h0[name].append(sig[mask])
            if ent is not None:
                h0_ent[name].append(ent[mask])
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
                if ent is not None:
                    h1_ent[best_name].append(ent[s:e][valid[s:e]])
                scheme_counts[best_name] += 1
    cat = lambda d: {k: (np.concatenate(v) if v else np.array([]))
                     for k, v in d.items()}
    h1, h0, h1_ent, h0_ent = cat(h1), cat(h0), cat(h1_ent), cat(h0_ent)
    stats = {"n_tokens": n_tokens, "n_spans": n_spans, "n_edge": n_edge,
             "n_docs": len(docs), "scheme_counts": scheme_counts}
    return h1, h0, np.array(span_lengths), stats, h1_ent, h0_ent


def fit_length_prior(span_lengths, pseudo=0.05, mix_uniform=0.0,
                     len_range=LEN_RANGE):
    """Fitted prior, optionally mixed with a uniform floor over the range.

    mix_uniform=m gives p = (1-m)*fitted + m*uniform: insurance against span
    lengths absent from the labeled data. m=1.0 -> pure uniform (sensitivity).
    """
    lengths = np.arange(*len_range)
    counts = np.array([(span_lengths == L).sum() for L in lengths], dtype=float)
    mass = (counts + pseudo) / (counts.sum() + pseudo * len(lengths))
    if mix_uniform > 0.0:
        mass = (1.0 - mix_uniform) * mass + mix_uniform / len(lengths)
    return lengths, np.log(mass)


def fit_binned_llr(h0_vals, h1_vals, n_bins=30, clip=4.0, pseudo=1.0,
                   isotonic=False) -> Emission:
    """Empirical per-bin LLR. With isotonic=True, additionally enforces a
    non-decreasing LLR across bins (weighted by per-bin sample count): all
    four signals are constructed so higher raw value = more H1 evidence, so
    this is a valid, variance-reducing constraint, not an assumption-free
    smoother. Mainly matters in the tail bins that set the 0.1% FPR
    threshold, which are estimated from very few labeled tokens."""
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
    llr = np.clip(llr, -clip, clip)
    if isotonic:
        from sklearn.isotonic import IsotonicRegression
        weights = c0 + c1 + 2 * pseudo
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        llr = iso.fit_transform(np.arange(B), llr, sample_weight=weights)
    return Emission(kind="binned", edges=edges, llr=llr)


def fit_bernoulli_kgw(h1_vals, gamma=0.25) -> Emission:
    p1 = float(np.clip(h1_vals.mean(), gamma + 0.01, 0.95)) if len(h1_vals) else gamma + 0.01
    return Emission(kind="bernoulli",
                    llr_green=float(np.log(p1 / gamma)),
                    llr_red=float(np.log((1 - p1) / (1 - gamma))))


def _entropy_bin_edges(h0_ent, n_ent_bins):
    edges = np.quantile(h0_ent, np.linspace(0.0, 1.0, n_ent_bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def fit_binned_llr_ent(h0_vals, h1_vals, h0_ent, h1_ent, n_ent_bins=3,
                       n_bins=30, clip=4.0, isotonic=False) -> Emission:
    """Signal LLR tables conditioned on entropy bin (H0-quantile bins).

    Generalizes multiplicative weighting: the fit learns how discriminative
    the signal actually is at each entropy level, on both H0 and H1 sides."""
    ent_edges = _entropy_bin_edges(h0_ent, n_ent_bins)
    E = len(ent_edges) - 1
    e0 = np.clip(np.searchsorted(ent_edges, h0_ent, side="right") - 1, 0, E - 1)
    e1 = np.clip(np.searchsorted(ent_edges, h1_ent, side="right") - 1, 0, E - 1)
    edges_list, llr_list = [], []
    for e in range(E):
        sub = fit_binned_llr(h0_vals[e0 == e], h1_vals[e1 == e],
                             n_bins=n_bins, clip=clip, isotonic=isotonic)
        edges_list.append(sub.edges)
        llr_list.append(sub.llr)
    return Emission(kind="binned_ent", ent_edges=ent_edges,
                    edges_list=edges_list, llr_list=llr_list)


def fit_bernoulli_kgw_ent(h1_vals, h0_ent, h1_ent, n_ent_bins=3,
                          gamma=0.25) -> Emission:
    ent_edges = _entropy_bin_edges(h0_ent, n_ent_bins)
    E = len(ent_edges) - 1
    e1 = np.clip(np.searchsorted(ent_edges, h1_ent, side="right") - 1, 0, E - 1)
    lg = np.zeros(E)
    lr = np.zeros(E)
    for e in range(E):
        vals = h1_vals[e1 == e]
        p1 = (float(np.clip(vals.mean(), gamma + 0.01, 0.95)) if len(vals) >= 50
              else gamma + 0.01)
        lg[e] = np.log(p1 / gamma)
        lr[e] = np.log((1 - p1) / (1 - gamma))
    return Emission(kind="bernoulli_ent", ent_edges=ent_edges,
                    llr_green_arr=lg, llr_red_arr=lr)


def fit_entropy_norm(docs, entropy_cache, lo_q=5.0, hi_q=95.0):
    """Global percentile normalization of entropies, fitted on the fit folds."""
    pool = np.concatenate([entropy_cache[str(d["document_id"])] for d in docs])
    pool = pool[pool >= 0].astype(np.float64)  # drop sentinel positions
    return float(np.percentile(pool, lo_q)), float(np.percentile(pool, hi_q))


def fit_params(docs, kgw=None, signal_cache=None, emission_mode="binned",
               n_bins=30, clip=4.0, fit_priors=True, shifts=None,
               neutral_invalid=False, p_span_scale=1.0,
               edge_prior=None, include_unigram=False,
               mix_uniform=0.0, len_range=LEN_RANGE,
               entropy_kind=None, entropy_cache=None,
               entropy_floor=0.1, n_ent_bins=0,
               use_exact=False, exact_only=False,
               exact_p_min=1e-4, exact_clip=8.0,
               forbid_adjacent_spans=False, isotonic_llr=False,
               mix_from_prevalence=False,
               use_exact_kgw=False, use_exact_unigram=False,
               exact_mix_weight=None) -> SmmParams:
    h1, h0, span_lengths, stats, h1_ent, h0_ent = collect_pools(
        docs, kgw, signal_cache,
        entropy_cache=entropy_cache if (entropy_kind or n_ent_bins) else None)

    if fit_priors:
        lengths, log_len_prior = fit_length_prior(span_lengths,
                                                  mix_uniform=mix_uniform,
                                                  len_range=len_range)
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
                emissions[name] = fit_binned_llr(h0[name], h1[name], n_bins, clip,
                                                 isotonic=isotonic_llr)
        emissions["kgw"] = fit_bernoulli_kgw(h1.get("kgw", np.array([])))
        if include_unigram:
            # too few labeled Unigram spans to fit; keep the conservative
            # hand-set single hypothesis (see smm_scorer.DEFAULT_SHIFTS)
            emissions["unigram"] = Emission(kind="gaussian", shifts=(0.5,))
    elif emission_mode == "binned_ent":
        for name in ("textseal", "gumbelmax"):
            if len(h1.get(name, ())) >= 500:
                emissions[name] = fit_binned_llr_ent(
                    h0[name], h1[name], h0_ent[name], h1_ent[name],
                    n_ent_bins=n_ent_bins or 3, n_bins=n_bins, clip=clip,
                    isotonic=isotonic_llr)
        emissions["kgw"] = fit_bernoulli_kgw_ent(
            h1.get("kgw", np.array([])), h0_ent.get("kgw", np.array([])),
            h1_ent.get("kgw", np.array([])), n_ent_bins=n_ent_bins or 3)
    else:
        raise ValueError(emission_mode)

    if exact_only:
        emissions.pop("textseal", None)
        emissions.pop("gumbelmax", None)
    if use_exact:
        emissions["gumbelmax_exact"] = Emission(kind="gumbel_exact",
                                                p_min=exact_p_min,
                                                exact_clip=exact_clip)
        emissions["textseal_exact"] = Emission(kind="textseal_exact",
                                               p_min=exact_p_min,
                                               exact_clip=exact_clip)
    if use_exact_kgw:
        # closed-form LLR from the LM's own boosted green-mass at this
        # position (output/kgw_lpg_{split}.npz), replacing the single fitted
        # Bernoulli rate — see docs/task1/attempt1.md audit (§17): KGW's
        # global Bernoulli only reaches 9% span detection, the weakest of
        # all four schemes despite decent coverage.
        emissions["kgw"] = Emission(kind="kgw_exact", exact_clip=exact_clip,
                                    boost=1.5)  # KGW delta, watermark_config.yaml
    if use_exact_unigram:
        emissions["unigram"] = Emission(kind="unigram_exact", exact_clip=exact_clip,
                                        boost=1.0)  # Unigram strength

    if (entropy_kind is not None or emission_mode == "binned_ent") \
            and entropy_cache is None:
        raise ValueError("entropy-based config but no entropy_cache given")
    ent_lo, ent_hi = 0.0, 1.0
    if entropy_kind is not None:
        ent_lo, ent_hi = fit_entropy_norm(docs, entropy_cache)

    mix_log_weights = None
    if mix_from_prevalence:
        counts = stats["scheme_counts"]
        # Laplace pseudo-count so schemes with zero assigned spans (e.g.
        # unigram, exact hypotheses not covered by this counter) aren't
        # zeroed out entirely.
        mix_log_weights = {name: counts.get(name, 0) + 1.0 for name in emissions}
    if exact_mix_weight is not None:
        # Down-weight the closed-form exact hypotheses relative to the
        # binned/empirical ones (uniform 1/n_hyp dilutes an already-strong
        # binned+entropy+isotonic mixture — verified empirically: adding
        # gumbelmax_exact/textseal_exact at full weight drops CV from 0.4301
        # to 0.3510, see docs/task1/attempt1.md audit).
        mix_log_weights = mix_log_weights or {name: 1.0 for name in emissions}
        for name in ("gumbelmax_exact", "textseal_exact"):
            if name in emissions:
                mix_log_weights[name] = exact_mix_weight

    return SmmParams(lengths=lengths, log_len_prior=log_len_prior,
                     log_p_span=float(log_p_span), edge_prior=float(edge_prior),
                     emissions=emissions, entropy_kind=entropy_kind,
                     entropy_lo=ent_lo, entropy_hi=ent_hi,
                     entropy_floor=entropy_floor,
                     forbid_adjacent_spans=forbid_adjacent_spans,
                     mix_log_weights=mix_log_weights)
