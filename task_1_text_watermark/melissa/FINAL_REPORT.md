# FINAL REPORT — Text Watermark Localization (Melissa)

Status at time of writing: **complete, reproducible pipeline built and syntax/logic-verified
locally.** Full training + leaderboard numbers require a JURECA GPU run with the dataset YAML
(keys) — see "How to reproduce". This report is updated once cluster results land.

## 1. Task

Per-token watermark localization: for each token of a document, output a score `∈ [0,1]` =
confidence the token was generated while a watermark was active. Watermark families:
TextSeal, Gumbel-Max, Unigram, KGW. Metric: **TPR @ 0.1 % FPR**, pooled over all tokens
(a ranking metric). Submission: `.jsonl`, one object per test document, task id
`30-watermark-localization`.

## 2. Dataset

`SprintML/watermark_localization` — 1,500 docs (train 90 labeled, val 90 labeled, test 1320
hidden). Tokenizer `Qwen/Qwen2.5-7B-Instruct`. `token_ids` authoritative (never retokenize
`text`). A YAML shipped with the data provides the real watermark keys + detector params.

## 3. Method

A two-stage design driven by the metric (clean tokens must be *reliably* low at 0.1 % FPR):

1. **Per-token detector signals** (`src/detectors/`), each recomputed from the real keys:
   - Gumbel-Max score `-ln(1 - R_t)` with `R_t = PRF(token, context, key)`;
   - TextSeal dual-key early fusion `(1-α)s⁽¹⁾ + α s⁽²⁾`;
   - Unigram green-list membership + local green fraction;
   - KGW green membership + local fraction + running z-score, greenlists rebuilt on a
     **CUDA (Philox)** generator (critical — CPU makes ~1/3 of KGW tokens read clean);
   - proxy-LM per-token entropy (TextSeal weighting: low entropy ⇒ near-null signal).
2. **Fusion + calibration + span logic**: stack detector signals with key-free statistics
   and local-context rolling means (`src/features.py`), train a lightweight calibrator
   (`LogisticRegression` → `HistGradientBoostingClassifier`) on train, select on val with the
   exact pooled metric, then smooth scores over neighbours and damp isolated spikes
   (`src/postprocess.py`) because watermarked regions are contiguous.

A **key-free baseline** (`src/baseline.py`) runs without the YAML/GPU to give a valid
submission early and a ranking floor.

## 4. Experiments

See `EXPERIMENT_LOG.md` (append-only). Local verification done:
- all modules `py_compile` clean;
- PRF is uniform (mean ≈ 0.5); Gumbel score mean ≈ 1.0 (correct Exp(1) null); Unigram green
  fraction ≈ γ; KGW degrades gracefully without a GPU; smoothing behaves.

Cluster experiments (to be filled): baseline val TPR@0.1%FPR → +Gumbel/TextSeal → +Unigram →
+KGW(CUDA) → +entropy → calibrator → smoothing.

## 5. Best method

To be confirmed on val. Expected best = full detector fusion (esp. correct CUDA KGW) +
gradient-boosting calibrator + span smoothing.

## 6. Limitations / risks

- KGW CUDA-Philox reproduction is the highest-risk correctness item; validate on GPU.
- Only 90+90 labeled docs → calibrator/threshold overfitting; select strictly on val.
- Detector keys/param layout in the YAML is read defensively (`config.WatermarkConfig`);
  confirm field names on first cluster load.
- Local machine can't run the full pipeline (no GPU/dataset/keys); code is written to run on
  JURECA.

## 7. How to reproduce

```bash
cd task_1_text_watermark/melissa
pip install -r requirements.txt
export WML_WATERMARK_YAML=/p/scratch/training2625/ansart1/loki/<keys>.yaml   # real keys
python -m src.load_data --check
python -m src.train_calibrator --model gboost
python -m src.evaluate --pred outputs/val_pred.jsonl --split validation
python -m src.predict --model outputs/calibrator_gboost.pkl
python ../../shared/submit.py outputs/submission.jsonl \
    --task-id 30-watermark-localization --action submit
# or submit the whole job:  sbatch scripts/run_full.sh
```

## 8. Important files

- Strategy/understanding: `TASK_UNDERSTANDING.md`, `SOLUTION_PLAN.md`.
- Code entry points: `src/baseline.py`, `src/train_calibrator.py`, `src/predict.py`,
  `src/evaluate.py`.
- Detectors: `src/detectors/{prf,gumbel,textseal,unigram,kgw,entropy}.py`.
- Launchers: `scripts/run_baseline.sh`, `scripts/run_full.sh`.
- Tracking: `TASK_TRACKER.md`, `EXPERIMENT_LOG.md`.
