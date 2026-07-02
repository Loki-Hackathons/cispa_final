# TASK TRACKER — Text Watermark Localization (Melissa)

> Living state of the solution. Update at the start/end of every work session.
> History of *runs* goes in `EXPERIMENT_LOG.md` (append-only); this file is the *current* view.

Last updated: 2026-07-02

## Legend
`[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

## Milestones

- [x] Read spec + TextSeal paper, confirm task = per-token watermark score, metric TPR@0.1%FPR
- [x] Create tracking/doc files (AGENT_INSTRUCTIONS, TASK_UNDERSTANDING, SOLUTION_PLAN, this, EXPERIMENT_LOG)
- [x] Scaffold `src/` pipeline (config, load_data, detectors, features, baseline, evaluate, postprocess, predict)
- [x] Key-free statistical baseline producing valid `.jsonl` (code ready)
- [x] Local metric harness `evaluate.py` (exact pooled TPR@0.1%FPR)
- [x] Calibrator trainer (logistic → gradient boosting) code ready
- [x] Post-processing (span smoothing) code ready
- [x] SLURM launch scripts + requirements + README
- [ ] **Run on JURECA**: download data, run baseline, get first val score (needs cluster + TOTP)
- [ ] Wire detector keys from the dataset YAML (`config.py: WATERMARK_YAML`)
- [ ] KGW with CUDA Philox validated on GPU
- [ ] Feature fusion + calibrator trained, val TPR@0.1%FPR recorded
- [ ] Span smoothing tuned on val
- [ ] First leaderboard submission via `shared/submit.py --task-id 30-watermark-localization`
- [ ] Final method chosen + `FINAL_REPORT.md`

## Decisions

- Optimize the **ranking** metric, not MSE — clean tokens must be reliably low at 0.1 % FPR.
- Baseline is intentionally **key-free** so it runs before the YAML/GPU are wired → fast valid submit.
- `token_ids` authoritative; never retokenize `text`.
- Calibrate on **val** (90 docs), never train — overfit risk.

## Open problems / risks

- **KGW CUDA Philox**: must recompute greenlists on GPU or ~1/3 KGW tokens read clean. Highest-risk item.
- Don't yet have the dataset **YAML** (keys/params/repos) locally — path set in `config.py`, fill on cluster.
- Cannot execute on JURECA from here (TOTP + no GPU/data locally). Code is written to run there.
- Exact HF column names assumed from spec (`document_id/text/token_ids/token_pieces/labels`); verify on first load.

## Next actions

1. On JURECA: `pip install -r melissa/requirements.txt`, run `python -m src.load_data --check`.
2. Run key-free baseline → `evaluate.py` on val → log score → submit once to test API.
3. Point `config.py:WATERMARK_YAML` at the dataset YAML; enable detectors incrementally
   (Gumbel → TextSeal → Unigram → KGW-CUDA), re-eval on val after each.
4. Train calibrator, tune smoothing, submit when val improves.
