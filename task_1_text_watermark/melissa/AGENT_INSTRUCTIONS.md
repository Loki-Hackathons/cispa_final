# AGENT INSTRUCTIONS — Melissa / Task 1 Text Watermark Localization

Permanent rules for any agent (or human) working on this solution.

## Scope — READ FIRST

- **Work only inside `task_1_text_watermark/melissa/`.**
- You **may read** anything in the repo (docs, shared/, other tasks) but **never create, edit, move or delete files outside `melissa/`.**
- Every path you write to must start with `.../task_1_text_watermark/melissa/`.
- Do not touch other participants' folders (`attempt1/`, other tasks).

## Task in one line

For every token of a document, output a score in `[0, 1]` = confidence that the
token was generated **while a watermark was active**. Metric: **TPR @ 0.1 % FPR**,
pooled over all tokens. Submission: `.jsonl`, one object per test document.

## Golden rules (from the official spec)

1. **`token_ids` are authoritative.** Never retokenize `text` (no round-trip guarantee).
2. **KGW greenlists use CUDA Philox** (`torch.randperm` on a GPU generator). Recompute
   on GPU or ~1/3 of KGW tokens look clean. This is the single biggest correctness trap.
3. Optimize the **ranking metric** (TPR@0.1%FPR), *not* MSE against 0/1 labels.
4. Watermarked regions are **contiguous spans** → smoothing / span logic helps.
5. Raw detector statistics are **noisy** → fuse + calibrate, do not submit raw scores.
6. Only 90 train + 90 val docs → **tune thresholds/calibrators on val**, watch overfit.
7. Keys/detector params come from a **YAML shipped with the dataset** — this is not
   blind detection. Point `src/config.py` at that YAML on the cluster.

## Workflow every time you resume

1. Read `TASK_TRACKER.md` (current state, next actions).
2. Read the tail of `EXPERIMENT_LOG.md` (never overwrite history; append only).
3. Read `SOLUTION_PLAN.md` to know which phase you are in.
4. Do the smallest next useful step, measure on val, log it, update the tracker.

## Environment

- Real compute = **JURECA** (1× A100 is enough). Dataset + keys live in team scratch
  `/p/scratch/training2625/ansart1/loki/` after `hackathon_setup.sh`.
- Local Windows machine = editing + syntax checks only (no GPU, no dataset, no keys).
- Submission API: `--task-id 30-watermark-localization`, `.env` has `CISPA_BASE_URL` /
  `CISPA_API_KEY`. Use `shared/submit.py`. ~5 min submit cooldown.

## Files that must stay updated

- `TASK_TRACKER.md` — living state.
- `EXPERIMENT_LOG.md` — append one block per run/experiment.
- `FINAL_REPORT.md` — final method + how to reproduce.
