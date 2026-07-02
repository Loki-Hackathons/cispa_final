# SOLUTION PLAN — Text Watermark Localization

Goal: maximize **TPR @ 0.1 % FPR** (pooled over all tokens) with a reproducible pipeline
living entirely in `melissa/`. Strategy = build a valid submission fast, then improve the
per-token signal with detector fusion + calibration + span smoothing.

Design principle: the metric only cares about **ranking** and about clean tokens having
**reliably low** scores at a 0.1 % FPR budget. So: strong separation + good calibration +
low-variance clean scores beat raw detector outputs.

---

## Phase 0 — Environment & data (runs on JURECA)

- `src/config.py` centralizes: dataset name/path, tokenizer `Qwen/Qwen2.5-7B-Instruct`,
  path to the **watermark YAML** (keys/detector params), output dir, task id
  `30-watermark-localization`.
- `src/load_data.py`: load HF dataset (or local cache), expose train/val/test with
  `document_id`, `token_ids`, `token_pieces`, `text`, `labels?`. Validate index alignment.

## Phase 1 — Baseline (fast, valid, key-free)

`src/baseline.py`: a statistical, **key-free** baseline that runs even without the YAML/GPU,
so we get a valid `.jsonl` early and exercise the submit pipeline + cooldowns.
- Features that need no secret key: local token-id rarity, repetition/n-gram novelty,
  piece-length, whitespace/newline structure, running local entropy proxy.
- Map to `[0,1]` via a simple logistic fit on train labels.
- Purpose: pipeline correctness + a non-trivial ranking floor. **Not** the final method.

## Phase 2 — Real detector signals (the main lift)

`src/detectors/`:
- `prf.py` — shared PRF hash (Aaronson/Fernandez form) mapping `(token, context, key)→[0,1]`.
- `gumbel.py` — Gumbel-Max score `-ln(1-R_t)` per token.
- `textseal.py` — dual-key early fusion `(1-α)s⁽¹⁾+α s⁽²⁾` (α from YAML, default 0.1).
- `unigram.py` — fixed key-seeded green list; per-token membership + local green fraction.
- `kgw.py` — context-seeded green list on **CUDA Philox** (`torch.randperm` on GPU gen).
- `entropy.py` — per-token entropy `Hᵢ` from a proxy LM (or cheap approximation).

Each detector reads its key/params from the YAML via `config.py`. All produce a per-token
float aligned to `token_ids`.

## Phase 3 — Feature fusion + calibration

`src/features.py`: stack per token → `[gumbel, textseal, unigram_mem, unigram_frac,
kgw_mem, kgw_frac, entropy, local context means, position]`.
`src/train_calibrator.py`: train a lightweight model on **train** tokens, select on **val**:
- start Logistic Regression (interpretable, robust on 90 docs),
- then gradient boosting / small MLP if val TPR@0.1%FPR improves.
Handle dedup of `(context, token)` tuples so correlated repeats don't dominate.

## Phase 4 — Local evaluation harness

`src/evaluate.py`: implement the **exact** pooled **TPR @ 0.1 % FPR** (pool all val tokens,
sort by score, find threshold at 0.1 % FPR, report TPR). Also report AUC + TPR@1% as
sanity. This is the number every experiment reports.

## Phase 5 — Post-processing (span logic)

`src/postprocess.py`:
- moving-average / Gaussian smoothing of scores over neighboring tokens (regions are
  contiguous),
- penalize isolated high spikes surrounded by clean tokens,
- optional density boost for tokens inside a high-mean local window (TextSeal region idea).
Tune smoothing window + weights on **val** only.

## Phase 6 — Prediction + submission

`src/predict.py`: run best pipeline on **test**, emit `melissa/outputs/submission.jsonl`
in the exact format, and **validate** (one line/doc, each test doc once, scores length ==
token count, all finite in `[0,1]`). Submit via `shared/submit.py --task-id
30-watermark-localization`. Only submit when **val** improves (respect ~5 min cooldown).

## Phase 7 — Finalize

- Pick best method by val TPR@0.1%FPR (with overfit check).
- Write `FINAL_REPORT.md`, freeze the reproduce command, update `TASK_TRACKER.md`.

---

## Ordered improvement backlog (each: hypothesize → implement → eval on val → log)

1. Key-free statistical baseline (Phase 1) — establish floor + valid submit.
2. Add Gumbel + TextSeal PRF features.
3. Add Unigram green features.
4. Add KGW green features **with CUDA Philox** (expect a big jump on KGW docs).
5. Add entropy weighting/feature.
6. Logistic calibrator → gradient boosting / MLP.
7. Span smoothing + isolated-spike penalty.
8. Threshold/temperature calibration targeting the 0.1 % FPR regime.
9. Dedup correlated `(context, token)` tuples; per-detector null normalization.

## Reproducibility

- Fixed seeds; deterministic ops where possible (KGW needs CUDA gen — document the seed).
- All artifacts under `melissa/outputs/`; all runs appended to `EXPERIMENT_LOG.md`.
- `requirements.txt` pins the stack; SLURM launchers in `melissa/scripts/`.
