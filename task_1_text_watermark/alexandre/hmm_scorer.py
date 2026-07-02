"""HMM-based token scorer for Task 1.

States: clean + one state per watermark scheme. Emission for a watermark
state is the per-token log-likelihood ratio (LLR) of that scheme's signal
under "watermark active" vs H0; the clean state emits 0. Forward-backward
posteriors give P(any watermark active) per token — sharp at span
boundaries, which is what TPR @ 0.1% FPR (pooled) rewards.

LLRs for continuous signals (TextSeal, Gumbel-Max) are nonparametric:
quantile-binned densities estimated from labeled train spans. Binary
signals (KGW, Unigram greenlists) use Bernoulli LLRs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from detectors import compute_signals

N_BINS = 40
_EPS = 1e-6


@dataclass
class BinnedLLR:
    edges: np.ndarray
    llr: np.ndarray  # per bin

    def __call__(self, x: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.llr) - 1)
        return self.llr[idx]


@dataclass
class BernoulliLLR:
    p1: float
    p0: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.5,
                        np.log(self.p1 / self.p0),
                        np.log((1 - self.p1) / (1 - self.p0)))


@dataclass
class HmmModel:
    schemes: list[str]
    llrs: dict = field(default_factory=dict)
    p_enter: float = 0.005   # clean -> given wm state
    p_exit: float = 0.010    # wm state -> clean
    llr_scale: float = 1.0   # temper emissions (guards against overconfident LLRs)


def fit_binned_llr(x1: np.ndarray, x0: np.ndarray, n_bins: int = N_BINS) -> BinnedLLR:
    edges = np.quantile(x0, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    c1, _ = np.histogram(x1, bins=edges)
    c0, _ = np.histogram(x0, bins=edges)
    f1 = (c1 + 1.0) / (c1.sum() + len(c1))
    f0 = (c0 + 1.0) / (c0.sum() + len(c0))
    return BinnedLLR(edges=edges[:-1], llr=np.log(f1 / f0))


def span_iter(labels: list[int]):
    """Yield (label, start, end) for contiguous runs."""
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            yield labels[start], start, i
            start = i


def collect_training_data(records: list[dict], extra_signals: dict | None = None,
                          z_assign: float = 3.0) -> dict:
    """Assign each labeled watermarked span to its most likely scheme by span z,
    and pool per-scheme H1 samples plus global H0 samples."""
    h0 = {}
    h1 = {}
    moments = {"textseal": (1.0, 0.5), "gumbelmax": (1.0, 1.0),
               "kgw": (0.25, 0.1875), "unigram": (0.5, 0.25)}
    for rec in records:
        ids = rec["token_ids"]
        sigs = compute_signals(ids)
        if extra_signals:
            for name, per_doc in extra_signals.items():
                sigs[name] = per_doc[rec["document_id"]]
        labels = rec["labels"]
        for lab, a, b in span_iter(labels):
            seg = {k: v[a:b] for k, v in sigs.items()}
            if lab == 0:
                for k, v in seg.items():
                    h0.setdefault(k, []).append(v)
                continue
            # pick the scheme with the highest span z-score
            best_name, best_z = None, -np.inf
            for k, v in seg.items():
                mu0, var0 = moments[k]
                z = (v.mean() - mu0) / np.sqrt(var0 / len(v))
                if z > best_z:
                    best_name, best_z = k, z
            if best_z >= z_assign:
                h1.setdefault(best_name, []).append(seg[best_name])
    return {"h0": {k: np.concatenate(v) for k, v in h0.items()},
            "h1": {k: np.concatenate(v) for k, v in h1.items()}}


def fit_hmm(records: list[dict], extra_signals: dict | None = None,
            p_enter: float = 0.005, p_exit: float = 0.010,
            llr_scale: float = 1.0) -> HmmModel:
    data = collect_training_data(records, extra_signals)
    model = HmmModel(schemes=[], p_enter=p_enter, p_exit=p_exit, llr_scale=llr_scale)
    for name, x1 in data["h1"].items():
        x0 = data["h0"][name]
        if name in ("kgw", "unigram"):
            model.llrs[name] = BernoulliLLR(p1=float(np.clip(x1.mean(), _EPS, 1 - _EPS)),
                                            p0=float(np.clip(x0.mean(), _EPS, 1 - _EPS)))
        else:
            model.llrs[name] = fit_binned_llr(x1, x0)
        model.schemes.append(name)
        print(f"  fitted {name}: {len(x1)} H1 tokens")
    return model


def _log_transitions(model: HmmModel) -> np.ndarray:
    k = len(model.schemes)
    T = np.zeros((k + 1, k + 1))
    T[0, 0] = 1.0 - k * model.p_enter
    T[0, 1:] = model.p_enter
    for s in range(1, k + 1):
        T[s, 0] = model.p_exit
        T[s, s] = 1.0 - model.p_exit
    return np.log(T + 1e-300)


def posterior_scores(model: HmmModel, token_ids: list[int],
                     extra: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """P(any watermark state) per token via forward-backward."""
    sigs = compute_signals(token_ids)
    if extra:
        sigs.update(extra)
    n = len(token_ids)
    k = len(model.schemes)

    log_emit = np.zeros((n, k + 1))
    for j, name in enumerate(model.schemes, start=1):
        log_emit[:, j] = model.llrs[name](sigs[name]) * model.llr_scale

    logT = _log_transitions(model)
    log_pi = np.log(np.array([0.6] + [0.4 / k] * k))

    # forward-backward in log space
    fwd = np.zeros((n, k + 1))
    fwd[0] = log_pi + log_emit[0]
    for t in range(1, n):
        m = fwd[t - 1][:, None] + logT
        fwd[t] = log_emit[t] + np.logaddexp.reduce(m, axis=0)
    bwd = np.zeros((n, k + 1))
    for t in range(n - 2, -1, -1):
        m = logT + (log_emit[t + 1] + bwd[t + 1])[None, :]
        bwd[t] = np.logaddexp.reduce(m, axis=1)

    log_post = fwd + bwd
    log_post -= np.logaddexp.reduce(log_post, axis=1, keepdims=True)
    p_clean = np.exp(log_post[:, 0])
    return 1.0 - p_clean


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
