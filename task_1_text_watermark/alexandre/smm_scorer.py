"""Semi-Markov (segment) forward-backward scorer for Task 1.

Model: a document is a concatenation of segments. Each segment is either a
single clean token (LLR = 0) or a watermarked span of length L with prior
p(L). Span emissions are per-token LLR hypotheses combined with logsumexp:

- "gaussian": shift model on standardized signals, l_t = shift*x_t - shift^2/2
  (one hypothesis per shift value, replicating the original hand-tuned model)
- "binned":   empirical LLR table fitted on labeled train spans (fit_smm.py)
- "bernoulli": closed-form binary LLR (KGW green/red)

Forward-backward over segmentations gives, per token, the exact posterior
mass of "this token is a clean segment"; score = watermark log-odds, stable
in log domain (no saturation ties). All parameters live in SmmParams so the
CV harness (cv_smm.py) can inject fold-fitted values; module defaults
reproduce the submitted #262 pipeline exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from detectors import (GUMBEL_NGRAM, H0_MOMENTS, TEXTSEAL_NGRAM, _dedup_mask,
                       compute_signals)
from unigram_scan import REAL_KEY, VOCAB_SIZE, greenlist_mask, token_dedup_mask

KGW_CONTEXT = 3  # ff-anchored_minhash_prf-4 self-salt: 3 context + 1 target

# empirical span-length prior (train+val counts, canonical lengths)
CANONICAL_LENGTHS = {31: 64, 47: 133, 63: 163, 95: 145, 159: 158, 320: 73}
LEN_RANGE = (24, 321)         # span lengths considered by the model
OTHER_LEN_MASS = 0.07         # ~7% of spans are non-canonical
DEFAULT_LOG_P_SPAN = np.log(0.004)
DEFAULT_EDGE_PRIOR = np.log(0.02)
DEFAULT_SHIFTS = {
    "textseal": (0.45, 0.65, 0.9),
    "gumbelmax": (0.55, 0.8, 1.1),
    "kgw": (0.6, 0.9, 1.3),
    # Single conservative hypothesis: signal confirmed present (unlabeled test-set
    # scan vs 20 decoy keys, docs/task1/attempt1.md §9/§14) but extremely rare
    # (~1 clearly-attributable document out of 1320); a wide shift grid would
    # only add FPR risk on the ~1319 docs where it is absent.
    "unigram": (0.5,),
}

_UNIGRAM_MASK: np.ndarray | None = None


def _unigram_signal(token_ids) -> tuple[np.ndarray, np.ndarray]:
    """Greenlist hit (1/0) per token; valid = first occurrence of an eligible
    (id < BASE_VOCAB) token id, matching vendor unidetect() semantics."""
    global _UNIGRAM_MASK
    if _UNIGRAM_MASK is None:
        _UNIGRAM_MASK = greenlist_mask(REAL_KEY, VOCAB_SIZE)
    ids = np.asarray(token_ids)
    sig = np.zeros(len(ids), dtype=np.float64)
    in_vocab = ids < VOCAB_SIZE
    sig[in_vocab] = _UNIGRAM_MASK[ids[in_vocab]].astype(np.float64)
    valid = token_dedup_mask(token_ids)
    return sig, valid


@dataclass
class Emission:
    kind: str                         # "gaussian" | "binned" | "bernoulli"
    shifts: tuple = ()                # gaussian
    neutral_invalid: bool = False     # gaussian: LLR 0 (not -shift^2/2) at invalid pos
    edges: np.ndarray | None = None   # binned: bin edges (len B+1, +-inf ends)
    llr: np.ndarray | None = None     # binned: per-bin LLR (len B)
    llr_green: float = 0.0            # bernoulli
    llr_red: float = 0.0


@dataclass
class SmmParams:
    lengths: np.ndarray
    log_len_prior: np.ndarray
    log_p_span: float
    edge_prior: float
    emissions: dict[str, Emission]
    min_edge: int = 5


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def default_length_prior() -> tuple[np.ndarray, np.ndarray]:
    """Canonical lengths get (1 - OTHER_LEN_MASS) by count; rest uniform."""
    lengths = np.arange(*LEN_RANGE)
    total_canon = sum(CANONICAL_LENGTHS.values())
    n_other = len(lengths) - len(CANONICAL_LENGTHS)
    mass = np.full(len(lengths), OTHER_LEN_MASS / n_other)
    for i, L in enumerate(lengths):
        if L in CANONICAL_LENGTHS:
            mass[i] = (1 - OTHER_LEN_MASS) * CANONICAL_LENGTHS[L] / total_canon
    return lengths, np.log(mass)


def default_params() -> SmmParams:
    lengths, log_prior = default_length_prior()
    emissions = {name: Emission(kind="gaussian", shifts=s)
                 for name, s in DEFAULT_SHIFTS.items()}
    return SmmParams(lengths=lengths, log_len_prior=log_prior,
                     log_p_span=DEFAULT_LOG_P_SPAN, edge_prior=DEFAULT_EDGE_PRIOR,
                     emissions=emissions)


def doc_signals(token_ids, extra=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Raw per-token signals + validity masks (first n-gram occurrence, scored pos)."""
    ids = list(token_ids)
    n = len(ids)
    pos = np.arange(n)
    sigs = compute_signals(ids)
    out = {}
    for name, ngram in (("textseal", TEXTSEAL_NGRAM), ("gumbelmax", GUMBEL_NGRAM)):
        valid = _dedup_mask(ids, ngram) & (pos >= ngram)
        out[name] = (sigs[name], valid)
    out["unigram"] = _unigram_signal(ids)
    if extra and "kgw" in extra:
        sig = np.asarray(extra["kgw"], dtype=np.float64)
        valid = (_dedup_mask(ids, KGW_CONTEXT) & (pos >= KGW_CONTEXT)
                 & ((sig == 0.0) | (sig == 1.0)))
        out["kgw"] = (sig, valid)
    return out


def _token_llr_hypotheses(signals, emissions) -> list[np.ndarray]:
    hyp = []
    for name, (sig, valid) in signals.items():
        em = emissions.get(name)
        if em is None:
            continue
        if em.kind == "gaussian":
            mu0, var0 = H0_MOMENTS[name]
            x = np.where(valid, (sig - mu0) / np.sqrt(var0), 0.0)
            for s in em.shifts:
                l = s * x - 0.5 * s * s
                if em.neutral_invalid:
                    l = np.where(valid, l, 0.0)
                hyp.append(l)
        elif em.kind == "binned":
            idx = np.clip(np.searchsorted(em.edges, sig, side="right") - 1,
                          0, len(em.llr) - 1)
            hyp.append(np.where(valid, em.llr[idx], 0.0))
        elif em.kind == "bernoulli":
            l = np.where(sig == 1.0, em.llr_green, em.llr_red)
            hyp.append(np.where(valid, l, 0.0))
        else:
            raise ValueError(em.kind)
    return hyp


def score_document(token_ids, extra=None, temperature=40.0,
                   params: SmmParams | None = None, signals=None) -> np.ndarray:
    """Per-token scores in [0,1], monotone in watermark log-odds."""
    p = params or default_params()
    if signals is None:
        signals = doc_signals(token_ids, extra)
    n = len(token_ids)
    hyp = _token_llr_hypotheses(signals, p.emissions)
    n_hyp = len(hyp)
    prefixes = [np.concatenate([[0.0], np.cumsum(l)]) for l in hyp]

    # span LLR table W[s, k] over p.lengths, mixed over hypotheses
    K = len(p.lengths)
    tables = []
    for pre in prefixes:
        w = np.full((n, K), -np.inf)
        for k, L in enumerate(p.lengths):
            if L > n:
                continue
            s = np.arange(0, n - L + 1)
            w[s, k] = pre[s + L] - pre[s]
        tables.append(w)
    W = logsumexp(np.stack(tables, axis=0), axis=0) - np.log(n_hyp)

    # truncated-span LLRs: prefix [0, e) and suffix [s, n)
    lp_edge = logsumexp(np.stack([pre for pre in prefixes]), axis=0) - np.log(n_hyp)
    suf = np.stack([pre[n] - pre for pre in prefixes])
    ls_edge = logsumexp(suf, axis=0) - np.log(n_hyp)

    log_p_clean = np.log1p(-np.exp(p.log_p_span))
    span_w = W + (p.log_p_span + p.log_len_prior)[None, :]
    min_edge = p.min_edge

    # forward (target-side accumulation)
    alpha = np.full(n + 1, -np.inf)
    alpha[0] = 0.0
    for e in range(1, n + 1):
        terms = [alpha[e - 1] + log_p_clean]
        starts = e - p.lengths
        ok = starts >= 0
        if ok.any():
            s_idx = starts[ok]
            terms.append(logsumexp(alpha[s_idx] + span_w[s_idx, np.where(ok)[0]]))
        if e >= min_edge:                                 # truncated prefix [0, e)
            terms.append(alpha[0] + p.log_p_span + p.edge_prior + lp_edge[e])
        if e == n:                                        # truncated suffix [s, n)
            s_ok = np.arange(0, n - min_edge + 1)
            terms.append(logsumexp(alpha[s_ok] + p.log_p_span + p.edge_prior
                                   + ls_edge[s_ok]))
        alpha[e] = logsumexp(terms)

    # backward
    beta = np.full(n + 1, -np.inf)
    beta[n] = 0.0
    for s in range(n - 1, -1, -1):
        terms = [beta[s + 1] + log_p_clean]
        ends = s + p.lengths
        ok = ends <= n
        if ok.any():
            e_idx = ends[ok]
            terms.append(logsumexp(beta[e_idx] + span_w[s, np.where(ok)[0]]))
        if s == 0:
            e_ok = np.arange(min_edge, n + 1)
            terms.append(logsumexp(beta[e_ok] + p.log_p_span + p.edge_prior
                                   + lp_edge[e_ok]))
        if s <= n - min_edge:
            terms.append(beta[n] + p.log_p_span + p.edge_prior + ls_edge[s])
        beta[s] = logsumexp(terms)

    log_z = alpha[n]
    # exact clean mass per token: token t is clean segment [t, t+1)
    log_m_clean = alpha[:n] + log_p_clean + beta[1:] - log_z
    log_m_clean = np.minimum(log_m_clean, -1e-12)
    log_odds = np.log1p(-np.exp(log_m_clean)) - log_m_clean
    return 1.0 / (1.0 + np.exp(-log_odds / temperature))
