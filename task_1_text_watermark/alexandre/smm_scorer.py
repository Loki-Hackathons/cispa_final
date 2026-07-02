"""Semi-Markov (segment) forward-backward scorer for Task 1.

Model: a document is a concatenation of segments. Each segment is either a
single clean token (LLR = 0) or a watermarked span of length L with prior
p(L) estimated from train (lengths are quasi-discrete: 31/47/63/95/159/320).
Span emission LLR under a Gaussian shift model on standardized PRF signals:

    LLR(s, e, scheme) = shift * sum(x[s:e]) - (e - s) * shift^2 / 2

combined over schemes/shift hypotheses with logsumexp. Forward-backward over
segmentations gives, for every token, the exact posterior mass of "this token
is a clean segment"; score = log-odds of being watermarked, computed stably
in log domain (no saturation ties).

Merged detections are handled natively: covering a clean gap inside a span
costs -gap * shift^2 / 2 in LLR, so the model prefers two shorter spans.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.special import logsumexp

from detectors import H0_MOMENTS, compute_signals

# empirical span-length prior (train+val counts, canonical lengths)
CANONICAL_LENGTHS = {31: 64, 47: 133, 63: 163, 95: 145, 159: 158, 320: 73}
OTHER_LEN_RANGE = (24, 321)   # non-canonical lengths allowed with small prior
OTHER_LEN_MASS = 0.07         # ~7% of spans are non-canonical
LOG_P_SPAN = np.log(0.004)    # prior prob a span starts at a given token
# per-scheme H1 mean shift hypotheses (std units, estimated on train)
SHIFTS = {
    "textseal": (0.45, 0.65, 0.9),
    "gumbelmax": (0.55, 0.8, 1.1),
    "kgw": (0.6, 0.9, 1.3),  # binary green: p1 in ~{0.5, 0.65, 0.8}
}
EDGE_PRIOR = np.log(0.02)     # prior for truncated span at doc boundary


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_length_prior() -> tuple[np.ndarray, np.ndarray]:
    """Return (lengths, log_prior) arrays."""
    lens, mass = [], []
    total_canon = sum(CANONICAL_LENGTHS.values())
    other = [L for L in range(*OTHER_LEN_RANGE) if L not in CANONICAL_LENGTHS]
    for L, c in CANONICAL_LENGTHS.items():
        lens.append(L)
        mass.append((1 - OTHER_LEN_MASS) * c / total_canon)
    for L in other:
        lens.append(L)
        mass.append(OTHER_LEN_MASS / len(other))
    lens = np.array(lens)
    order = np.argsort(lens)
    return lens[order], np.log(np.array(mass)[order])


LENGTHS, LOG_LEN_PRIOR = build_length_prior()


def standardized_signals(token_ids, extra=None) -> dict[str, np.ndarray]:
    from detectors import _dedup_mask

    sigs = compute_signals(token_ids)
    sigs.pop("unigram", None)  # unreproducible locally, adds noise only
    if extra:
        for name, sig in extra.items():
            sig = np.asarray(sig, dtype=np.float64).copy()
            if name == "kgw":
                # KGW seeds on the 4-gram incl. target: repeated 4-grams
                # re-emit the same mask bit -> no new evidence.
                keep = _dedup_mask(token_ids, 3)
                sig[~keep] = H0_MOMENTS["kgw"][0]
            sigs[name] = sig
    out = {}
    for name, sig in sigs.items():
        mu0, var0 = H0_MOMENTS[name]
        out[name] = (sig - mu0) / np.sqrt(var0)
    return out


def _span_llr_tables(std_sigs: dict[str, np.ndarray],
                     shifts: dict[str, tuple] | None = None):
    """For each start s return vector over LENGTHS of combined span LLR.

    Output: dict-free matrix W[s, k] = logsumexp over (scheme, shift) of
    LLR(s, s+LENGTHS[k]) + uniform hypothesis prior. Invalid (overrun) = -inf.
    """
    shifts = shifts or SHIFTS
    n = len(next(iter(std_sigs.values())))
    K = len(LENGTHS)
    hyp = []
    for name, x in std_sigs.items():
        prefix = np.concatenate([[0.0], np.cumsum(x)])
        for shift in shifts.get(name, (0.6,)):
            llr = np.full((n, K), -np.inf)
            for k, L in enumerate(LENGTHS):
                if L > n:
                    continue
                s = np.arange(0, n - L + 1)
                llr[s, k] = shift * (prefix[s + L] - prefix[s]) - L * shift * shift / 2.0
            hyp.append(llr)
    n_hyp = len(hyp)
    return logsumexp(np.stack(hyp, axis=0), axis=0) - np.log(n_hyp)


def _edge_llr(std_sigs, shifts=None):
    """Combined LLR for truncated spans: prefix [0, e) and suffix [s, n)."""
    shifts = shifts or SHIFTS
    n = len(next(iter(std_sigs.values())))
    pre_h, suf_h = [], []
    for name, x in std_sigs.items():
        prefix = np.concatenate([[0.0], np.cumsum(x)])
        for shift in shifts.get(name, (0.6,)):
            e = np.arange(n + 1)
            pre_h.append(shift * prefix - e * shift * shift / 2.0)
            suf_h.append(shift * (prefix[n] - prefix) - (n - e) * shift * shift / 2.0)
    lp = logsumexp(np.stack(pre_h), axis=0) - np.log(len(pre_h))
    ls = logsumexp(np.stack(suf_h), axis=0) - np.log(len(suf_h))
    return lp, ls  # lp[e]: span [0,e) ; ls[s]: span [s,n)


def score_document(token_ids, extra=None, temperature=40.0):
    """Return per-token scores in [0,1] (monotone in watermark log-odds)."""
    std_sigs = standardized_signals(token_ids, extra)
    n = len(token_ids)
    W = _span_llr_tables(std_sigs)                       # (n, K)
    lp_edge, ls_edge = _edge_llr(std_sigs)               # (n+1,), (n+1,)
    log_p_clean = np.log1p(-np.exp(LOG_P_SPAN))

    span_w = W + (LOG_P_SPAN + LOG_LEN_PRIOR)[None, :]   # full spans
    min_edge = 5

    # forward (target-side accumulation)
    alpha = np.full(n + 1, -np.inf)
    alpha[0] = 0.0
    for e in range(1, n + 1):
        terms = [alpha[e - 1] + log_p_clean]
        starts = e - LENGTHS
        ok = starts >= 0
        if ok.any():
            s_idx = starts[ok]
            terms.append(logsumexp(alpha[s_idx] + span_w[s_idx, np.where(ok)[0]]))
        if e >= min_edge:                                 # truncated prefix [0, e)
            terms.append(alpha[0] + LOG_P_SPAN + EDGE_PRIOR + lp_edge[e])
        if e == n:                                        # truncated suffix [s, n)
            s_ok = np.arange(0, n - min_edge + 1)
            terms.append(logsumexp(alpha[s_ok] + LOG_P_SPAN + EDGE_PRIOR + ls_edge[s_ok]))
        alpha[e] = logsumexp(terms)

    # backward
    beta = np.full(n + 1, -np.inf)
    beta[n] = 0.0
    for s in range(n - 1, -1, -1):
        terms = [beta[s + 1] + log_p_clean]
        ends = s + LENGTHS
        ok = ends <= n
        if ok.any():
            e_idx = ends[ok]
            terms.append(logsumexp(beta[e_idx] + span_w[s, np.where(ok)[0]]))
        if s == 0:
            e_ok = np.arange(min_edge, n + 1)
            terms.append(logsumexp(beta[e_ok] + LOG_P_SPAN + EDGE_PRIOR + lp_edge[e_ok]))
        if s <= n - min_edge:
            terms.append(beta[n] + LOG_P_SPAN + EDGE_PRIOR + ls_edge[s])
        beta[s] = logsumexp(terms)

    log_z = alpha[n]
    # exact clean mass per token: token t is clean segment [t, t+1)
    log_m_clean = alpha[:n] + log_p_clean + beta[1:] - log_z
    log_m_clean = np.minimum(log_m_clean, -1e-12)
    # log-odds wm vs clean, stable at both ends
    log_odds = np.log1p(-np.exp(log_m_clean)) - log_m_clean
    return 1.0 / (1.0 + np.exp(-log_odds / temperature))
