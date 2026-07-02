# Melissa — Task 1: Text Watermark Localization

Per-token watermark detection for the CISPA Grand Finals. For each token of a document,
output a score `∈ [0,1]` = confidence it was generated while a watermark was active.
Metric: **TPR @ 0.1 % FPR** (pooled over all tokens). Submission: `.jsonl`.

> **Scope rule:** everything here stays inside `melissa/`. See `AGENT_INSTRUCTIONS.md`.

## Documents

| File | Purpose |
|------|---------|
| `TASK_UNDERSTANDING.md` | What the task is, dataset, metric, TextSeal ideas that transfer |
| `SOLUTION_PLAN.md` | Phased strategy + improvement backlog |
| `AGENT_INSTRUCTIONS.md` | Permanent rules + resume workflow |
| `TASK_TRACKER.md` | Living state (todo / done / decisions / risks) |
| `EXPERIMENT_LOG.md` | Append-only log of every run + result |
| `FINAL_REPORT.md` | Final method + how to reproduce (written at the end) |

## Code (`src/`)

```
config.py            paths, tokenizer, YAML-of-keys, seed, task id
load_data.py         HF dataset -> Document objects (token_ids authoritative)
detectors/
  prf.py             shared PRF hash  R_v = PRF(token, context, key)
  gumbel.py          Gumbel-Max score  -ln(1 - R_t)
  textseal.py        dual-key early fusion (alpha)
  unigram.py         fixed green-list membership + local fraction
  kgw.py             context green-list (CUDA Philox!) + z-score
  entropy.py         proxy-LM per-token entropy (TextSeal weighting)
features.py          stack detectors + key-free stats + local context
pipeline.py          build matrices, scatter scores, write + validate .jsonl
baseline.py          Phase-1 key-free logistic baseline (runs anywhere)
train_calibrator.py  Phase-3 detector-fusion calibrator (logreg / gboost)
postprocess.py       span smoothing + isolated-spike penalty
evaluate.py          exact pooled TPR@0.1%FPR (+ AUC, TPR@1%)
predict.py           test submission + format validation
```

## One-shot: run everything + submit (recommended)

A single script does the **whole pipeline and submits** (data check → train → eval →
generate + validate `.jsonl` → leaderboard submit):

```bash
# GPU job (KGW needs CUDA Philox):
sbatch task_1_text_watermark/melissa/scripts/run_all_and_submit.sh
# or interactively on a GPU node:
bash  task_1_text_watermark/melissa/scripts/run_all_and_submit.sh
```

Options (env vars): `MODEL=gboost|logreg`, `SUBMIT=0` (generate only, no submit),
`WML_WATERMARK_YAML=/path/to/keys.yaml`. Needs `.env` (CISPA_BASE_URL / CISPA_API_KEY)
at the repo root for the submit step.

## Quick start step-by-step (on JURECA, from repo root)

```bash
cd task_1_text_watermark/melissa
pip install -r requirements.txt

# 1. sanity-check the dataset
python -m src.load_data --check

# 2. key-free baseline -> first valid submission + val score
python -m src.baseline
python -m src.evaluate --pred outputs/baseline_val_pred.jsonl --split validation

# 3. real detectors: point at the dataset YAML (keys) and a GPU (for KGW Philox)
export WML_WATERMARK_YAML=/p/scratch/training2625/ansart1/loki/<keys>.yaml
python -m src.train_calibrator --model logreg      # then --model gboost
python -m src.predict --model outputs/calibrator_logreg.pkl

# 4. submit (respect ~5 min cooldown)
python ../../shared/submit.py outputs/submission.jsonl \
    --task-id 30-watermark-localization --action submit
```

## Environment variables

| Var | Meaning | Default |
|-----|---------|---------|
| `WML_DATASET` | HF id or local dataset path | `SprintML/watermark_localization` |
| `WML_TOKENIZER` | tokenizer id | `Qwen/Qwen2.5-7B-Instruct` |
| `WML_WATERMARK_YAML` | dataset YAML with keys/params | *(empty → key-free)* |
| `WML_ENTROPY_MODEL` | proxy LM for entropy | `Qwen/Qwen2.5-0.5B` |
| `WML_OUTPUT_DIR` / `WML_CACHE_DIR` | artifacts / HF cache | `melissa/outputs`, `melissa/data_cache` |
| `WML_SEED` | random seed | `1234` |

## Non-negotiables

- Never retokenize `text`; use `token_ids`.
- KGW greenlists **must** be recomputed on a CUDA generator (Philox) — CPU fallback is
  only for local smoke tests and will not match the data.
- Tune calibrator/thresholds on **val** (90 docs), not train.
