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

from detectors import (GUMBEL_NGRAM, H0_MOMENTS, TEXTSEAL_ALPHA, TEXTSEAL_NGRAM,
                       _dedup_mask, compute_signals, gumbelmax_r, textseal_r)
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
    kind: str                         # "gaussian" | "binned" | "bernoulli" | *_ent
                                       # | "gumbel_exact" | "textseal_exact"
                                       # | "kgw_exact" | "unigram_exact"
    shifts: tuple = ()                # gaussian
    neutral_invalid: bool = False     # gaussian: LLR 0 (not -shift^2/2) at invalid pos
    edges: np.ndarray | None = None   # binned: bin edges (len B+1, +-inf ends)
    llr: np.ndarray | None = None     # binned: per-bin LLR (len B)
    llr_green: float = 0.0            # bernoulli
    llr_red: float = 0.0
    # entropy-conditioned variants: one sub-table per entropy bin
    ent_edges: np.ndarray | None = None   # entropy bin edges (len E+1, +-inf ends)
    edges_list: list | None = None        # binned_ent: signal edges per ent bin
    llr_list: list | None = None          # binned_ent: LLR table per ent bin
    llr_green_arr: np.ndarray | None = None  # bernoulli_ent (len E)
    llr_red_arr: np.ndarray | None = None
    # closed-form Gumbel-max / TextSeal LLR from the realized-token model
    # probability p_t (Aaronson detector): f(r) = (1/p) r^(1/p - 1) vs H0
    # uniform. p_min floors 1/p blowups on rare tokens; clip bounds the
    # resulting per-token LLR (p_t is an approximation - no known top-p /
    # temperature at generation time).
    p_min: float = 1e-4
    exact_clip: float = 8.0
    boost: float = 1.5                 # kgw_exact / unigram_exact: delta/strength logit shift


@dataclass
class SmmParams:
    lengths: np.ndarray
    log_len_prior: np.ndarray
    log_p_span: float
    edge_prior: float
    emissions: dict[str, Emission]
    min_edge: int = 5
    # entropy weighting (TextSeal 3.2): weight per-token LLRs by the LM's
    # predictive entropy, normalized with percentiles fitted on the fit folds
    entropy_kind: str | None = None   # None | "linear" | "sqrt"
    entropy_lo: float = 0.0
    entropy_hi: float = 1.0
    entropy_floor: float = 0.1
    # structural boundary-bleed fix: forbid a segmentation from placing two
    # watermarked spans back-to-back with zero clean tokens between them.
    # Lossless w.r.t. ground truth: two adjacent label=1 positions are always
    # merged into one span by construction (iter_spans groups consecutive
    # equal labels), so a true segmentation never has this pattern.
    forbid_adjacent_spans: bool = False
    # non-uniform mixture over scheme hypotheses (audit finding: real scheme
    # prevalence among labeled spans is ~41/36/22% GM/TS/KGW, not 1/n_hyp
    # each). name -> raw weight (renormalized internally); None = uniform.
    mix_log_weights: dict[str, float] | None = None


def entropy_weights(entropy: np.ndarray, p: SmmParams) -> np.ndarray:
    """w = floor + (1-floor)*f(normalized H); sentinel (<0) positions get floor."""
    h = np.clip((entropy - p.entropy_lo) / max(p.entropy_hi - p.entropy_lo, 1e-9),
                0.0, 1.0)
    h = np.where(entropy < 0, 0.0, h)
    if p.entropy_kind == "sqrt":
        h = np.sqrt(h)
    elif p.entropy_kind != "linear":
        raise ValueError(p.entropy_kind)
    return p.entropy_floor + (1.0 - p.entropy_floor) * h


def _log_gumbel_density(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """log f(r; p) for the Aaronson Gumbel-max detector: r_v^(1/p_v) argmax
    reparameterization implies r | H1, selected ~ density (1/p) r^(1/p - 1)
    on [0,1]; H0 (no watermark) has r ~ Uniform(0,1), density 1. This is
    already the log-likelihood-ratio (log(f(r)/1))."""
    r = np.clip(r, 1e-12, 1.0 - 1e-12)
    inv_p = 1.0 / p
    return np.log(inv_p) + (inv_p - 1.0) * np.log(r)


def gumbel_exact_llr(r: np.ndarray, p_target: np.ndarray, em: Emission) -> np.ndarray:
    p = np.clip(p_target, em.p_min, 1.0)
    return np.clip(_log_gumbel_density(r, p), -em.exact_clip, em.exact_clip)


def textseal_exact_llr(r_a: np.ndarray, r_b: np.ndarray, p_target: np.ndarray,
                       em: Emission, alpha: float = TEXTSEAL_ALPHA) -> np.ndarray:
    """LLR = log(alpha*f(r_a) + (1-alpha)*f(r_b)); r_b (resp. r_a) stays
    Uniform(0,1) regardless of which key drove the actual sampling step, so
    H0 density of the joint (r_a, r_b) pair is 1 and this is already a LLR."""
    p = np.clip(p_target, em.p_min, 1.0)
    la = np.log(alpha) + _log_gumbel_density(r_a, p)
    lb = np.log(1.0 - alpha) + _log_gumbel_density(r_b, p)
    m = np.maximum(la, lb)
    combined = m + np.log(np.exp(la - m) + np.exp(lb - m))
    return np.clip(combined, -em.exact_clip, em.exact_clip)


def green_exact_llr(sig: np.ndarray, lpg: np.ndarray, boost: float,
                    em: Emission) -> np.ndarray:
    """Closed-form LLR for a red/green watermark (KGW, Unigram) using the
    LM's own per-position green-mass probability instead of a single fitted
    Bernoulli rate.

    lpg = log P(green | context) under the delta/strength-BOOSTED
    distribution (the watermark logit shift applied to green tokens before
    softmax). The correct H0 reference is NOT the marginal gamma: the boost
    only shifts logits, so at confident/low-entropy positions where the
    model's top (unboosted) token is already green, both the boosted AND
    unboosted distributions are ~certain-green, and under H0 the realized
    token is drawn from the UNBOOSTED distribution — which is then just as
    likely to be green there. Using a constant gamma as H0 wrongly credits
    exactly those high-confidence contexts as watermark evidence regardless
    of whether a watermark was active (verified empirically: corr(realized
    green, boosted p_green) = 0.69 even on label=0 tokens).

    Invert the known boost to recover the model's own unboosted p_green,
    g0 = pb / (pb + (1-pb)*e^boost) [from pb = g0*e^boost/(g0*e^boost+1-g0)],
    and compare boosted vs unboosted at the SAME context — cancelling the
    confidence confound. Verified: mean(g0) = 0.2507 on H0 tokens (gamma is
    recovered on average, as expected only when using the correct reference)."""
    pb = np.clip(np.exp(np.clip(lpg, -700.0, 0.0)), 1e-6, 1.0 - 1e-6)
    g0 = np.clip(pb / (pb + (1.0 - pb) * np.exp(boost)), 1e-6, 1.0 - 1e-6)
    llr_green = np.log(pb / g0)
    llr_red = np.log((1.0 - pb) / (1.0 - g0))
    llr = np.where(sig == 1.0, llr_green, llr_red)
    return np.clip(llr, -em.exact_clip, em.exact_clip)


def doc_exact_signals(token_ids) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Raw PRF draws (r) for the closed-form Gumbel/TextSeal LLR, keyed
    separately from doc_signals() (which holds the fused z-score signal used
    for schema assignment / binned emissions)."""
    ids = list(token_ids)
    n = len(ids)
    pos = np.arange(n)
    gm_valid = _dedup_mask(ids, GUMBEL_NGRAM) & (pos >= GUMBEL_NGRAM)
    ts_valid = _dedup_mask(ids, TEXTSEAL_NGRAM) & (pos >= TEXTSEAL_NGRAM)
    r_a, r_b = textseal_r(ids)
    return {
        "gumbelmax_r": (gumbelmax_r(ids), gm_valid),
        "textseal_ra": (r_a, ts_valid),
        "textseal_rb": (r_b, ts_valid),
    }


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


def _token_llr_hypotheses(signals, emissions, entropy=None,
                          exact_signals=None, p_target=None, lpg=None
                          ) -> list[tuple[str, np.ndarray]]:
    """Returns (scheme_name, per-token LLR array) pairs — the name lets
    score_document apply non-uniform mixture weights (SmmParams.mix_log_weights);
    multiple entries can share a name (e.g. several gaussian shifts)."""
    hyp = []
    ent_idx_cache = {}

    def ent_idx(edges):
        key = id(edges)
        if key not in ent_idx_cache:
            ent_idx_cache[key] = np.clip(
                np.searchsorted(edges, entropy, side="right") - 1,
                0, len(edges) - 2)
        return ent_idx_cache[key]

    for name, em in emissions.items():
        if em is None:
            continue
        if em.kind in ("gumbel_exact", "textseal_exact"):
            if exact_signals is None or p_target is None:
                continue
            has_ctx = p_target >= 0.0  # sentinel for "no context" is -1.0
            if em.kind == "gumbel_exact":
                r, valid = exact_signals["gumbelmax_r"]
                l = gumbel_exact_llr(r, p_target, em)
            else:
                r_a, valid = exact_signals["textseal_ra"]
                r_b, _ = exact_signals["textseal_rb"]
                l = textseal_exact_llr(r_a, r_b, p_target, em)
            hyp.append((name, np.where(valid & has_ctx, l, 0.0)))
            continue
        if em.kind in ("kgw_exact", "unigram_exact"):
            if lpg is None or name not in lpg:
                continue
            sig, valid = signals.get(name, (None, None))
            if sig is None:
                continue
            lpg_arr = lpg[name]
            has_ctx = ~np.isnan(lpg_arr)
            l = green_exact_llr(sig, np.where(has_ctx, lpg_arr, -1.0), em.boost, em)
            hyp.append((name, np.where(valid & has_ctx, l, 0.0)))
            continue
        sig, valid = signals.get(name, (None, None))
        if sig is None:
            continue
        if em.kind == "gaussian":
            mu0, var0 = H0_MOMENTS[name]
            x = np.where(valid, (sig - mu0) / np.sqrt(var0), 0.0)
            for s in em.shifts:
                l = s * x - 0.5 * s * s
                if em.neutral_invalid:
                    l = np.where(valid, l, 0.0)
                hyp.append((name, l))
        elif em.kind == "binned":
            idx = np.clip(np.searchsorted(em.edges, sig, side="right") - 1,
                          0, len(em.llr) - 1)
            hyp.append((name, np.where(valid, em.llr[idx], 0.0)))
        elif em.kind == "bernoulli":
            l = np.where(sig == 1.0, em.llr_green, em.llr_red)
            hyp.append((name, np.where(valid, l, 0.0)))
        elif em.kind == "binned_ent":
            ei = ent_idx(em.ent_edges)
            l = np.zeros(len(sig))
            for e in range(len(em.llr_list)):
                m = ei == e
                if not m.any():
                    continue
                idx = np.clip(np.searchsorted(em.edges_list[e], sig[m],
                                              side="right") - 1,
                              0, len(em.llr_list[e]) - 1)
                l[m] = em.llr_list[e][idx]
            hyp.append((name, np.where(valid, l, 0.0)))
        elif em.kind == "bernoulli_ent":
            ei = ent_idx(em.ent_edges)
            l = np.where(sig == 1.0, em.llr_green_arr[ei], em.llr_red_arr[ei])
            hyp.append((name, np.where(valid, l, 0.0)))
        else:
            raise ValueError(em.kind)
    return hyp


def _forward_backward_no_adjacent(n, p, span_w, lp_edge, ls_edge, log_p_clean):
    """Two-phase forward-backward forbidding back-to-back spans.

    Phase 0 ("after clean"): the segment immediately preceding this position
    was a single clean token, or this is the document start — a new span is
    allowed to start here. Phase 1 ("after span"): the preceding segment was
    itself a watermarked span — only a clean token may follow, no new span
    may start immediately. This matches the ground-truth generative process
    exactly (see SmmParams.forbid_adjacent_spans) and removes the
    "hallucinated span glued to a real one" failure mode without discarding
    any true segmentation.
    """
    lengths = p.lengths
    min_edge = p.min_edge
    NEG = -np.inf

    a0 = np.full(n + 1, NEG)  # phase "after clean" (spans may start here)
    a1 = np.full(n + 1, NEG)  # phase "after span" (only clean may follow)
    a0[0] = 0.0

    for e in range(1, n + 1):
        a0[e] = logsumexp([a0[e - 1], a1[e - 1]]) + log_p_clean

        terms = []
        starts = e - lengths
        ok = starts >= 0
        if ok.any():
            s_idx = starts[ok]
            terms.append(logsumexp(a0[s_idx] + span_w[s_idx, np.where(ok)[0]]))
        if e >= min_edge:                                  # prefix-truncated
            terms.append(a0[0] + p.log_p_span + p.edge_prior + lp_edge[e])
        a1[e] = logsumexp(terms) if terms else NEG

    s_ok = np.arange(0, n - min_edge + 1)                   # suffix-truncated
    suffix = logsumexp(a0[s_ok] + p.log_p_span + p.edge_prior + ls_edge[s_ok])
    a1[n] = logsumexp([a1[n], suffix])

    log_z = logsumexp([a0[n], a1[n]])

    b_clean = np.full(n + 1, NEG)  # entering s from "after clean" (spans ok)
    b_span = np.full(n + 1, NEG)   # entering s from "after span" (clean only)
    b_clean[n] = 0.0
    b_span[n] = 0.0

    for s in range(n - 1, -1, -1):
        clean_term = b_clean[s + 1] + log_p_clean
        b_span[s] = clean_term

        terms = [clean_term]
        ends = s + lengths
        ok = ends <= n
        if ok.any():
            e_idx = ends[ok]
            terms.append(logsumexp(b_span[e_idx] + span_w[s, np.where(ok)[0]]))
        if s == 0:
            e_ok = np.arange(min_edge, n + 1)
            terms.append(logsumexp(b_span[e_ok] + p.log_p_span + p.edge_prior
                                   + lp_edge[e_ok]))
        if s <= n - min_edge:
            terms.append(p.log_p_span + p.edge_prior + ls_edge[s])  # b_span[n]=0
        b_clean[s] = logsumexp(terms)

    a_comb = np.logaddexp(a0[:n], a1[:n])
    log_m_clean = a_comb + log_p_clean + b_clean[1:] - log_z
    return log_m_clean


def score_document(token_ids, extra=None, temperature=40.0,
                   params: SmmParams | None = None, signals=None,
                   entropy=None, exact_signals=None,
                   p_target=None, lpg=None) -> np.ndarray:
    """Per-token scores in [0,1], monotone in watermark log-odds.

    p_target: per-token probability the model assigned to the realized token
    (Aaronson/TextSeal closed-form detector), or -1.0 where no context /
    forward-pass logits were available. Requires emissions of kind
    "gumbel_exact" / "textseal_exact" to have any effect.

    lpg: dict {"kgw": arr, "unigram": arr} of per-token log P(green | context)
    under the boosted (watermarked) distribution, NaN where undefined.
    Requires emissions of kind "kgw_exact" / "unigram_exact"."""
    p = params or default_params()
    if signals is None:
        signals = doc_signals(token_ids, extra)
    if exact_signals is None and any(
            em is not None and em.kind in ("gumbel_exact", "textseal_exact")
            for em in p.emissions.values()):
        exact_signals = doc_exact_signals(token_ids)
    n = len(token_ids)
    ent = np.asarray(entropy, dtype=np.float64) if entropy is not None else None
    pt = np.asarray(p_target, dtype=np.float64) if p_target is not None else None
    hyp_named = _token_llr_hypotheses(signals, p.emissions, entropy=ent,
                                      exact_signals=exact_signals, p_target=pt,
                                      lpg=lpg)
    if p.entropy_kind is not None and ent is not None:
        w = entropy_weights(ent, p)
        hyp_named = [(name, l * w) for name, l in hyp_named]
    names = [name for name, _ in hyp_named]
    hyp = [l for _, l in hyp_named]
    n_hyp = len(hyp)
    prefixes = [np.concatenate([[0.0], np.cumsum(l)]) for l in hyp]

    if p.mix_log_weights:
        raw = np.array([p.mix_log_weights.get(name, 1.0) for name in names])
        log_w = np.log(raw) - logsumexp(np.log(raw))
    else:
        log_w = np.full(n_hyp, -np.log(n_hyp))

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
    W = logsumexp(np.stack(tables, axis=0) + log_w[:, None, None], axis=0)

    # truncated-span LLRs: prefix [0, e) and suffix [s, n)
    lp_edge = logsumexp(np.stack(prefixes) + log_w[:, None], axis=0)
    suf = np.stack([pre[n] - pre for pre in prefixes])
    ls_edge = logsumexp(suf + log_w[:, None], axis=0)

    log_p_clean = np.log1p(-np.exp(p.log_p_span))
    span_w = W + (p.log_p_span + p.log_len_prior)[None, :]
    min_edge = p.min_edge

    if p.forbid_adjacent_spans:
        log_m_clean = _forward_backward_no_adjacent(
            n, p, span_w, lp_edge, ls_edge, log_p_clean)
        log_m_clean = np.minimum(log_m_clean, -1e-12)
        log_odds = np.log1p(-np.exp(log_m_clean)) - log_m_clean
        return 1.0 / (1.0 + np.exp(-log_odds / temperature))

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
