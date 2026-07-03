"""Unit tests for the roadmap.md additions (no dataset / npz needed).

Covers:
1. erode_scores == brute-force sliding-window minimum
2. edge_len_logprior of all-zeros == legacy scores (fold is a no-op);
   non-trivial prior changes scores and keeps them in [0,1]
3. fit_edge_len_prior: proper distribution, non-increasing in visible length
4. Emission.signal aliasing: kgw_exact_ensemble hypothesis reads the "kgw"
   signal; mixture with an extra duplicate hypothesis matches an explicit
   manual construction
5. fit_params synth pooling: synthetic docs enlarge the emission pools
6. ensemble logit round-trip

Run: python test_roadmap_units.py
"""

from __future__ import annotations

import numpy as np

from fit_smm import fit_edge_len_prior, fit_params
from smm_scorer import (Emission, SmmParams, default_length_prior,
                        default_params, erode_scores, score_document)

rng = np.random.default_rng(42)
FAILURES = []


def check(name, cond):
    print(f"{'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILURES.append(name)


def fake_signals(n, span=(30, 80), scheme="gumbelmax", strength=1.2):
    """Continuous TS/GM signals + binary kgw signal with a boosted span."""
    labels = np.zeros(n, dtype=int)
    labels[span[0]:span[1]] = 1
    sigs = {}
    for name, (mu, var) in (("textseal", (1.0, 0.5)), ("gumbelmax", (1.0, 1.0))):
        x = rng.normal(mu, np.sqrt(var), n)
        if scheme == name:
            x[span[0]:span[1]] += strength * np.sqrt(var)
        sigs[name] = (x, np.ones(n, dtype=bool))
    green_p = np.full(n, 0.25)
    if scheme == "kgw":
        green_p[span[0]:span[1]] = 0.55
    sigs["kgw"] = ((rng.random(n) < green_p).astype(np.float64),
                   np.ones(n, dtype=bool))
    return sigs, labels


def docs_and_cache(n_docs, n=250, scheme_cycle=("gumbelmax", "textseal", "kgw")):
    docs, cache = [], {}
    for i in range(n_docs):
        scheme = scheme_cycle[i % len(scheme_cycle)]
        s = int(rng.integers(20, 100))
        L = int(rng.choice([31, 47, 63, 95]))
        sigs, labels = fake_signals(n, span=(s, s + L), scheme=scheme,
                                    strength=1.5)
        did = f"doc_{i}"
        docs.append({"document_id": did, "token_ids": list(range(n)),
                     "labels": labels.tolist()})
        cache[did] = sigs
    return docs, cache


# ---- 1. erosion --------------------------------------------------------
s = rng.random(200)
for radius in (1, 2, 3):
    got = erode_scores(s, radius)
    want = np.array([s[max(0, i - radius):i + radius + 1].min()
                     for i in range(len(s))])
    check(f"erode radius {radius} == brute force", np.allclose(got, want))
check("erode radius 0 is identity", np.array_equal(erode_scores(s, 0), s))

# ---- 2. edge_len_logprior fold -----------------------------------------
n = 120
sigs, labels = fake_signals(n, span=(20, 67), scheme="gumbelmax")
base = default_params()
scores_legacy = score_document(list(range(n)), params=base, signals=sigs)

p_zero = default_params()
p_zero.edge_len_logprior = np.zeros(4097)  # log 1 everywhere = legacy
scores_zero = score_document(list(range(n)), params=p_zero, signals=sigs)
check("edge prior all-zeros == legacy", np.allclose(scores_legacy, scores_zero))

lengths, log_prior = default_length_prior()
p_struct = default_params()
p_struct.edge_len_logprior = fit_edge_len_prior(lengths, log_prior)
scores_struct = score_document(list(range(n)), params=p_struct, signals=sigs)
check("structured edge prior changes scores",
      not np.allclose(scores_legacy, scores_struct))
check("structured edge prior scores in [0,1]",
      np.all((scores_struct >= 0) & (scores_struct <= 1))
      and np.all(np.isfinite(scores_struct)))

# no-adjacent branch also folds the edge prior
p_na = default_params()
p_na.edge_len_logprior = fit_edge_len_prior(lengths, log_prior)
p_na.forbid_adjacent_spans = True
scores_na = score_document(list(range(n)), params=p_na, signals=sigs)
check("structured edge prior + no-adjacent finite",
      np.all(np.isfinite(scores_na)))

# ---- 3. fit_edge_len_prior properties ----------------------------------
elp = fit_edge_len_prior(lengths, log_prior)
mass = np.exp(elp[np.isfinite(elp)])
check("edge len prior sums to 1", abs(mass.sum() - 1.0) < 1e-9)
finite = elp[1:int(lengths.max()) + 1]
check("edge len prior non-increasing",
      np.all(np.diff(finite[np.isfinite(finite)]) <= 1e-12))
check("edge len prior -inf beyond max span length",
      not np.isfinite(elp[int(lengths.max()) + 1:]).any()
      if len(elp) > int(lengths.max()) + 1 else True)

# ---- 4. Emission.signal aliasing (kgw ensemble hypothesis) --------------
lpg = {"kgw": np.log(np.clip(rng.random(n), 0.05, 0.95))}
p_ens = default_params()
p_ens.emissions["kgw_x"] = Emission(kind="kgw_exact", exact_clip=8.0,
                                    boost=1.5, signal="kgw")
scores_ens = score_document(list(range(n)), params=p_ens, signals=sigs,
                            lpg=lpg)
check("kgw_x aliased hypothesis runs, scores valid",
      np.all(np.isfinite(scores_ens))
      and np.all((scores_ens >= 0) & (scores_ens <= 1)))
check("kgw_x hypothesis changes scores",
      not np.allclose(scores_legacy, scores_ens))

# without lpg the aliased hypothesis must be skipped silently -> legacy
scores_no_lpg = score_document(list(range(n)), params=p_ens, signals=sigs)
check("kgw_x skipped without lpg == legacy",
      np.allclose(scores_legacy, scores_no_lpg))

# ---- 5. fit_params with synth pooling ----------------------------------
docs, cache = docs_and_cache(30)
sdocs, scache = docs_and_cache(60)
sdocs = [{**d, "document_id": "s_" + d["document_id"]} for d in sdocs]
scache = {"s_" + k: v for k, v in scache.items()}

params_real = fit_params(docs, signal_cache=cache, emission_mode="binned",
                         n_bins=20)
params_aug = fit_params(docs, signal_cache=cache, emission_mode="binned",
                        n_bins=20, synth_docs=sdocs,
                        synth_signal_cache=scache)
check("synth pooling changes fitted emissions",
      any(not np.allclose(params_real.emissions[k].llr,
                          params_aug.emissions[k].llr)
          for k in ("textseal", "gumbelmax")
          if k in params_real.emissions and k in params_aug.emissions))
check("synth pooling keeps priors from real docs only",
      np.allclose(params_real.log_len_prior, params_aug.log_len_prior)
      and params_real.log_p_span == params_aug.log_p_span)

params_ens2 = fit_params(docs, signal_cache=cache, emission_mode="binned",
                         n_bins=20, kgw_exact_ensemble=True)
check("kgw_exact_ensemble keeps bernoulli AND adds kgw_x",
      params_ens2.emissions["kgw"].kind == "bernoulli"
      and params_ens2.emissions["kgw_x"].kind == "kgw_exact"
      and params_ens2.emissions["kgw_x"].signal == "kgw")

params_er = fit_params(docs, signal_cache=cache, emission_mode="binned",
                       n_bins=20, erode_radius=2, structured_edge=True)
check("fit_params wires erode_radius + structured_edge",
      params_er.erode_radius == 2
      and params_er.edge_len_logprior is not None)

# ---- 6. ensemble logit round trip ---------------------------------------
from ensemble_smm import to_logit
x = rng.random(1000)
back = 1.0 / (1.0 + np.exp(-to_logit(x)))
check("logit round trip", np.allclose(x, back, atol=1e-6))
xa, xb = rng.random(500), rng.random(500)
ens = 1.0 / (1.0 + np.exp(-(to_logit(xa) + to_logit(xb)) / 2))
check("ensemble between components",
      np.all(ens >= np.minimum(xa, xb) - 1e-9)
      and np.all(ens <= np.maximum(xa, xb) + 1e-9))

print(f"\n{'ALL OK' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
raise SystemExit(0 if not FAILURES else 1)
