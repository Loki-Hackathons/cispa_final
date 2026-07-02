# EXPERIMENT LOG — Text Watermark Localization (Melissa)

> **Append-only.** Never edit or delete past entries. One block per run/experiment.
> Copy the template, fill it in, add it at the **bottom**.

Metric of record: **TPR @ 0.1 % FPR**, pooled over all tokens (see `src/evaluate.py`).
Also report AUC and TPR@1% as sanity checks.

---

## Template (copy me)

```
### EXP-<n> — <short name>
- date/time     : YYYY-MM-DD HH:MM (TZ)
- host          : local | jureca (job <id>)
- script/cmd    : python -m src.<...> ...
- method        : <what this run does>
- key params    : <detectors on, calibrator, smoothing window, seed, ...>
- data          : train=90 / val=90 (which split evaluated)
- results (val) : TPR@0.1%FPR=<> | AUC=<> | TPR@1%FPR=<>
- submitted     : no | yes (leaderboard public=<> private=<>)
- observations  : <what happened, what surprised you>
- errors        : <none | ...>
- next          : <the single next improvement to try>
```

---

## History

### EXP-0 — scaffolding (no run yet)
- date/time     : 2026-07-02
- host          : local (Windows, edit only)
- script/cmd    : n/a (created pipeline code + docs)
- method        : Built full `src/` pipeline: config, load_data, PRF + 4 detectors,
                  entropy, features, key-free baseline, evaluate (pooled TPR@0.1%FPR),
                  postprocess (span smoothing), calibrator trainer, predict.
- key params    : none run
- data          : n/a
- results (val) : none (not executed — needs JURECA + dataset + YAML keys)
- submitted     : no
- observations  : Code validated with `py_compile` only. Cannot download dataset / use
                  GPU / read watermark YAML from the local Windows machine.
- errors        : none (syntax-checked)
- next          : Run EXP-1 on JURECA — key-free baseline → first val TPR@0.1%FPR + valid submit.

### EXP-0b — local logic verification (no dataset)
- date/time     : 2026-07-02
- host          : local (Windows)
- script/cmd    : python -m py_compile src/*.py src/detectors/*.py ; python -c "<smoke test>"
- method        : Syntax-checked all 17 modules; ran pure-Python detectors + postprocess
                  on synthetic tokens (numpy/torch/dataset absent locally).
- key params    : WatermarkConfig(vocab=2000, keys={gumbel,unigram,kgw,textseal})
- data          : synthetic (200 random tokens; 1000-token synthetic metric check)
- results (val) : n/a — sanity only: PRF mean=0.498 (uniform ✓); Gumbel mean=1.027
                  (Exp(1) null ✓); Unigram green frac=0.54 (~γ=0.5 ✓); KGW→zeros without
                  GPU (graceful ✓); Gaussian smoothing monotone over a ramp ✓.
- submitted     : no
- observations  : Core signal math is correct. Full run blocked only by missing
                  GPU/dataset/keys locally — all handled on JURECA.
- errors        : none (numpy-dependent modules not exercised locally, by design)
- next          : EXP-1 on JURECA — key-free baseline val score + first valid submit.
