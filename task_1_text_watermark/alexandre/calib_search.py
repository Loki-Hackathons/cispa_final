"""Point 1: select the best temperature/top-p calibration for the exact
Gumbel-Max / TextSeal LLR, then check its gain inside the full pipeline.

Stage 1 isolates the exact LLR alone (emission_mode="binned", exact_only=True
drops the binned textseal/gumbelmax tables, leaving only gumbel_exact /
textseal_exact + the kgw Bernoulli floor) so the CV TPR directly measures
that one signal's quality under each (T, top_p) candidate - no dilution by
the strong binned+entropy+isotonic baseline.

Stage 2 takes the winning tag and re-tests it embedded in the full best
pipeline (b50_ps2_elo_entbin5_iso + use_exact), across a small
exact_mix_weight sweep, to see the true end-to-end delta over the
0.4301 baseline (docs/task1/attempt1.md, no-exact CV score).

Requires calib_T*_p*_{train,validation}.npz in output/ (calib_pass.py).
"""

from __future__ import annotations

import numpy as np

from cv_smm import (build_cache, build_exact_cache, eval_config, load_entropy,
                    load_labeled, load_p_target_tag)

ISOLATE_CONFIG = "binned50_edge_lo_ps2_exactonly"
FULL_CONFIG = "b50_ps2_elo_entbin5_iso_exact"

TAGS = [f"T{T}_p{P}" for T in (0.8, 0.9, 1.0) for P in (0.9, 0.95, 1.0)]


def main():
    docs, kgw = load_labeled()
    print(f"{len(docs)} labeled docs; building caches...", flush=True)
    cache = build_cache(docs, kgw)
    exact_cache = build_exact_cache(docs)
    entropy = load_entropy()

    print("\n=== Stage 1: isolate exact-LLR quality per (T, top_p) ===", flush=True)
    results = {}
    for tag in TAGS:
        pt = load_p_target_tag(tag)
        if pt is None:
            print(f"{tag:16s} MISSING npz", flush=True)
            continue
        r = eval_config(ISOLATE_CONFIG, docs, kgw, cache, exact_cache=exact_cache,
                        p_target=pt)
        if r is not None:
            results[tag] = r
    if not results:
        print("No calib npz found - run calib_pass.py first.", flush=True)
        return
    best_tag = max(results, key=results.get)
    print(f"\nBest calibration tag: {best_tag} (isolated CV TPR={results[best_tag]:.4f})",
          flush=True)
    print("(uncalibrated raw p_target for reference:)", flush=True)
    from cv_smm import load_p_target
    pt_raw = load_p_target()
    if pt_raw is not None:
        eval_config(ISOLATE_CONFIG, docs, kgw, cache, exact_cache=exact_cache,
                   p_target=pt_raw)

    print(f"\n=== Stage 2: {best_tag} inside {FULL_CONFIG} (vs no-exact baseline) ===",
          flush=True)
    eval_config("b50_ps2_elo_entbin5_iso", docs, kgw, cache, entropy=entropy)
    pt_best = load_p_target_tag(best_tag)
    eval_config(FULL_CONFIG, docs, kgw, cache, entropy=entropy,
               exact_cache=exact_cache, p_target=pt_best)


if __name__ == "__main__":
    main()
