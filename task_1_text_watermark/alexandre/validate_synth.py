"""Realism check for gen_synth.py output — MANDATORY before use_synth CV runs.

Compares, per scheme, the H1 signal pools of the synthetic docs against the
labeled (train+val) pools, and the H0 pools likewise. If the organizers used
different sampling params (temperature/top-p) the H1 distributions will
mismatch — in that case regenerate with other --temperature/--top-p instead
of feeding CV a biased fit pool.

Checks:
- two-sample KS statistic per scheme (H1 and H0 pools)
- mean/std side by side
- KGW green rate (H1) overall and per entropy tercile

Rule of thumb printed at the end: KS < 0.05 on every H1 pool -> safe to use;
0.05-0.10 -> use with caution (run the *_synth config but distrust small CV
deltas); > 0.10 -> do NOT use, regenerate.

Usage:
    python validate_synth.py            # expects output/synth.jsonl etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cv_smm import build_cache, load_entropy, load_labeled, load_synth
from fit_smm import collect_pools

OUT = Path(__file__).resolve().parent / "output"


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def describe(name, real, synth):
    ks = ks_stat(real, synth)
    print(f"  {name:10s} n_real={len(real):7d} n_synth={len(synth):7d}  "
          f"mean {np.mean(real) if len(real) else float('nan'):.4f} vs "
          f"{np.mean(synth) if len(synth) else float('nan'):.4f}  "
          f"std {np.std(real) if len(real) else float('nan'):.4f} vs "
          f"{np.std(synth) if len(synth) else float('nan'):.4f}  "
          f"KS={ks:.4f}")
    return ks


def main() -> None:
    docs, kgw = load_labeled()
    cache = build_cache(docs, kgw)
    entropy = load_entropy()
    synth = load_synth(need_entropy=entropy is not None)
    if synth is None:
        raise SystemExit("synth files missing under output/ - run gen_synth.py first")
    sdocs, scache, sent = synth

    h1, h0, _, stats, h1_ent, h0_ent = collect_pools(
        docs, kgw, cache, entropy_cache=entropy)
    sh1, sh0, _, sstats, sh1_ent, sh0_ent = collect_pools(
        sdocs, None, scache, entropy_cache=sent)

    print(f"real: {stats['n_spans']} spans / {stats['n_docs']} docs, "
          f"scheme_counts={dict(stats['scheme_counts'])}")
    print(f"synth: {sstats['n_spans']} spans / {sstats['n_docs']} docs, "
          f"scheme_counts={dict(sstats['scheme_counts'])}")

    worst = 0.0
    print("\nH1 pools (watermarked spans, assigned by window z):")
    for name in ("gumbelmax", "textseal", "kgw"):
        ks = describe(name, h1.get(name, np.array([])),
                      sh1.get(name, np.array([])))
        if np.isfinite(ks):
            worst = max(worst, ks)

    print("\nH0 pools (clean tokens):")
    for name in ("gumbelmax", "textseal", "kgw"):
        describe(name, h0.get(name, np.array([])), sh0.get(name, np.array([])))

    if entropy is not None and sent is not None:
        print("\nKGW H1 green rate by entropy tercile (real vs synth):")
        re_v, re_e = h1.get("kgw", np.array([])), h1_ent.get("kgw", np.array([]))
        sy_v, sy_e = sh1.get("kgw", np.array([])), sh1_ent.get("kgw", np.array([]))
        if len(re_v) and len(sy_v):
            qs = np.quantile(re_e, [0, 1 / 3, 2 / 3, 1.0])
            for lo, hi in zip(qs[:-1], qs[1:]):
                rm = re_v[(re_e >= lo) & (re_e <= hi)]
                sm = sy_v[(sy_e >= lo) & (sy_e <= hi)]
                print(f"  ent [{lo:6.2f},{hi:6.2f}]  real green="
                      f"{rm.mean() if len(rm) else float('nan'):.4f} "
                      f"(n={len(rm)})  synth green="
                      f"{sm.mean() if len(sm) else float('nan'):.4f} (n={len(sm)})")

    print(f"\nWorst H1 KS = {worst:.4f}")
    if worst < 0.05:
        print("VERDICT: pools match - safe to enable use_synth configs.")
    elif worst < 0.10:
        print("VERDICT: marginal - usable, but distrust small CV deltas; "
              "consider regenerating with adjusted --temperature/--top-p.")
    else:
        print("VERDICT: MISMATCH - do NOT use; regenerate with different "
              "sampling params (the organizers' temperature/top-p differ).")


if __name__ == "__main__":
    main()
