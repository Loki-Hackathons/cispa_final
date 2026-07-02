"""Brute-force validation of _forward_backward_no_adjacent against explicit
enumeration of all valid (no-back-to-back-span) segmentations on tiny docs.

Run: python test_no_adjacent.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from smm_scorer import SmmParams, _forward_backward_no_adjacent


def enumerate_segmentations(n, lengths, min_edge, log_p_span, edge_prior,
                            span_w, log_p_clean, lp_edge, ls_edge):
    results = []

    def rec(pos, last_was_span, segs, logw):
        if pos == n:
            results.append((logw, list(segs)))
            return
        segs.append(("clean", pos, pos + 1))
        rec(pos + 1, False, segs, logw + log_p_clean)
        segs.pop()
        if not last_was_span:
            for k, L in enumerate(lengths):
                e = pos + L
                if e <= n:
                    segs.append(("span", pos, e))
                    rec(e, True, segs, logw + span_w[pos, k])
                    segs.pop()
            if pos == 0:
                for e in range(min_edge, n + 1):
                    w = log_p_span + edge_prior + lp_edge[e]
                    segs.append(("prefix_trunc", 0, e))
                    rec(e, True, segs, logw + w)
                    segs.pop()
            if n - pos >= min_edge:
                w = log_p_span + edge_prior + ls_edge[pos]
                segs.append(("suffix_trunc", pos, n))
                rec(n, True, segs, logw + w)
                segs.pop()

    rec(0, False, [], 0.0)
    return results


def brute_force_log_m_clean(n, lengths, min_edge, log_p_span, edge_prior,
                            span_w, log_p_clean, lp_edge, ls_edge):
    segs_all = enumerate_segmentations(n, lengths, min_edge, log_p_span,
                                       edge_prior, span_w, log_p_clean,
                                       lp_edge, ls_edge)
    logws = np.array([lw for lw, _ in segs_all])
    log_z = logsumexp(logws)
    per_token = [[] for _ in range(n)]
    for lw, segs in segs_all:
        for kind, s, e in segs:
            if kind == "clean":
                per_token[s].append(lw)
    log_m_clean = np.array([
        logsumexp(v) - log_z if v else -np.inf for v in per_token])
    return log_m_clean, log_z, len(segs_all)


def run_case(seed, n, lengths, min_edge, p_span, edge_prior_val):
    rng = np.random.default_rng(seed)
    K = len(lengths)
    log_p_span = np.log(p_span)
    log_p_clean = np.log1p(-p_span)
    edge_prior = np.log(edge_prior_val)
    log_len_prior = np.log(rng.dirichlet(np.ones(K)))
    W = rng.normal(0, 1.5, size=(n, K))
    span_w = W + (log_p_span + log_len_prior)[None, :]
    lp_edge = rng.normal(0, 1.0, size=n + 1)
    ls_edge = rng.normal(0, 1.0, size=n + 1)

    p = SmmParams(lengths=np.array(lengths), log_len_prior=log_len_prior,
                 log_p_span=log_p_span, edge_prior=edge_prior, emissions={},
                 min_edge=min_edge)

    log_m_dp = _forward_backward_no_adjacent(n, p, span_w, lp_edge, ls_edge,
                                             log_p_clean)
    log_m_bf, log_z_bf, n_segs = brute_force_log_m_clean(
        n, np.array(lengths), min_edge, log_p_span, edge_prior, span_w,
        log_p_clean, lp_edge, ls_edge)

    ok_vals = np.isfinite(log_m_dp) == np.isfinite(log_m_bf)
    diff = np.abs(np.where(np.isfinite(log_m_bf), log_m_dp - log_m_bf, 0.0))
    max_diff = diff.max()
    status = "OK" if (ok_vals.all() and max_diff < 1e-8) else "MISMATCH"
    print(f"seed={seed} n={n} lengths={lengths} min_edge={min_edge} "
          f"n_segmentations={n_segs}: max|diff|={max_diff:.2e}  {status}")
    if status == "MISMATCH":
        print("  DP :", log_m_dp)
        print("  BF :", log_m_bf)
    return status == "OK"


def main():
    cases = [
        dict(seed=0, n=10, lengths=[3, 4], min_edge=2, p_span=0.1, edge_prior_val=0.3),
        dict(seed=1, n=12, lengths=[3, 5], min_edge=3, p_span=0.15, edge_prior_val=0.2),
        dict(seed=2, n=9, lengths=[2, 3, 4], min_edge=2, p_span=0.08, edge_prior_val=0.4),
        dict(seed=3, n=14, lengths=[4], min_edge=4, p_span=0.05, edge_prior_val=0.1),
        dict(seed=4, n=11, lengths=[3, 4, 5], min_edge=20, p_span=0.1, edge_prior_val=0.3),
    ]
    ok = all(run_case(**c) for c in cases)
    print("\nALL OK" if ok else "\nFAILURES DETECTED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
