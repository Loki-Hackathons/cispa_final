"""Matched-filter span detection for Task 1.

Watermarked spans have quasi-discrete lengths {31, 47, 63, 95, 159, 320}
(93% of train+val spans). For every (start, length, scheme) candidate we
compute the exact-window z-score of the scheme's standardized increments;
greedy non-overlapping selection snaps span boundaries; token scores are
then ranked by (span z, distance to span edge) so that boundary tokens of
detected spans rank below deep tokens — which is what TPR @ 0.1% FPR
(pooled) rewards.

Fallback for truncated spans: suffix candidates [s, n) and prefix
candidates [0, e) of any length >= MIN_LEN.
"""

from __future__ import annotations

import json

import numpy as np

from detectors import H0_MOMENTS, compute_signals

SPAN_LENGTHS = (31, 47, 63, 95, 159, 320)
MIN_LEN = 24
Z_MIN = 2.0          # ignore candidates weaker than this
EDGE_SATURATION = 12  # edge-distance feature saturates here


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def standardized_signals(token_ids: list[int],
                         extra: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    sigs = compute_signals(token_ids)
    if extra:
        sigs.update(extra)
    out = {}
    for name, sig in sigs.items():
        mu0, var0 = H0_MOMENTS[name]
        out[name] = (sig - mu0) / np.sqrt(var0)
    return out


def _window_z(prefix: np.ndarray, s: int, e: int) -> float:
    return (prefix[e] - prefix[s]) / np.sqrt(e - s)


def candidates_for_doc(std_sigs: dict[str, np.ndarray], z_min: float = Z_MIN):
    """Yield (z, start, end, scheme) candidates: discrete lengths + prefix/suffix."""
    n = len(next(iter(std_sigs.values())))
    cands = []
    for name, x in std_sigs.items():
        prefix = np.concatenate([[0.0], np.cumsum(x)])
        for L in SPAN_LENGTHS:
            if L > n:
                continue
            starts = np.arange(0, n - L + 1)
            zs = (prefix[starts + L] - prefix[starts]) / np.sqrt(L)
            for s in np.where(zs >= z_min)[0]:
                cands.append((float(zs[s]), int(s), int(s + L), name))
        # truncated-span fallbacks: suffixes and prefixes of any length
        for s in range(0, n - MIN_LEN + 1):
            z = _window_z(prefix, s, n)
            if z >= z_min:
                cands.append((float(z), int(s), int(n), name))
        for e in range(MIN_LEN, n + 1):
            z = _window_z(prefix, 0, e)
            if z >= z_min:
                cands.append((float(z), 0, int(e), name))
    return cands


def greedy_select(cands, n: int, max_overlap: int = 0):
    """Pick non-overlapping candidates by descending z."""
    cands = sorted(cands, key=lambda c: -c[0])
    taken = np.zeros(n, dtype=bool)
    selected = []
    for z, s, e, name in cands:
        overlap = int(taken[s:e].sum())
        if overlap > max_overlap:
            continue
        taken[s:e] = True
        selected.append((z, s, e, name))
    return selected


def token_features(token_ids: list[int],
                   extra: dict[str, np.ndarray] | None = None):
    """Per-token (span_z, edge_dist) features from greedy matched-filter spans."""
    n = len(token_ids)
    std_sigs = standardized_signals(token_ids, extra)
    spans = greedy_select(candidates_for_doc(std_sigs), n)
    span_z = np.zeros(n)
    edge = np.zeros(n)
    for z, s, e, _ in spans:
        idx = np.arange(s, e)
        d = np.minimum(idx - s + 1, e - idx)
        span_z[s:e] = z
        edge[s:e] = np.minimum(d, EDGE_SATURATION)
    return span_z, edge


def fit_token_ranker(records: list[dict], extra_signals: dict | None = None):
    """Logistic regression on (span_z, edge_dist, z*edge) using train labels."""
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    for rec in records:
        extra = ({k: v[rec["document_id"]] for k, v in extra_signals.items()}
                 if extra_signals else None)
        span_z, edge = token_features(rec["token_ids"], extra)
        X.append(np.column_stack([span_z, edge, span_z * edge]))
        y.append(np.array(rec["labels"]))
    X = np.vstack(X)
    y = np.concatenate(y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def score_doc(model, token_ids: list[int],
              extra: dict[str, np.ndarray] | None = None) -> np.ndarray:
    span_z, edge = token_features(token_ids, extra)
    X = np.column_stack([span_z, edge, span_z * edge])
    # decision_function = unbounded log-odds -> no saturation ties
    logit = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-logit / 8.0))
