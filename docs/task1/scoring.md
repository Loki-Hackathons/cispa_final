# Task 1 — Scoring (TPR @ 0.1% FPR)

Canonical spec: [`docs/subject/task_1.md`](../subject/task_1.md). Local eval script: [`shared/task1_eval.py`](../../shared/task1_eval.py).

## What you submit

One continuous score in `[0, 1]` per token, per test document (`.jsonl`):

```json
{"document_id": "1", "scores": [0.02, 0.04, 0.91, 0.87, ...]}
```

`scores[i]` = your confidence that `token_ids[i]` was generated **while watermarking was active**. No labels, text, token IDs, or watermark-family predictions.

## Ground truth

Organizers hold **binary** labels per token (train/val only):

| Label | Meaning |
|---|---|
| **0** | Non-watermarked (human, clean LLM, or unknown watermark scheme) |
| **1** | Watermark active during generation |

Labels come from the generation pipeline (when watermarking was switched on/off), **not** from re-running a detector. They are hard 0/1, not probabilities.

Test set (1,320 docs): labels are **hidden**. You only get an aggregated leaderboard score back.

## This is NOT regression

The metric does **not** measure how close your scores are to 0/1 (no MSE, log-loss, or Brier score). It is a **ranking / detection metric** at an extremely low false-positive rate.

## Pooling

Scores are **not** averaged per document, then averaged globally.

All tokens from all documents in a leaderboard subset (public 30% or private 70%) are **pooled** into one list:

```
(token₁, label₁, score₁), (token₂, label₂, score₂), …
```

Long documents weigh more than short ones. The metric reflects performance over **all tokens**, not “fraction of documents correctly classified.”

## TPR and FPR

Fix a threshold τ. Predict “watermarked” when `score ≥ τ`.

| Metric | Definition |
|---|---|
| **TPR** (True Positive Rate, recall) | Among tokens with label **1**, what fraction have `score ≥ τ`? |
| **FPR** (False Positive Rate) | Among tokens with label **0**, what fraction have `score ≥ τ`? |

\[
\text{TPR}(\tau) = \frac{\#\{i : y_i=1 \land s_i \geq \tau\}}{\#\{i : y_i=1\}}
\qquad
\text{FPR}(\tau) = \frac{\#\{i : y_i=0 \land s_i \geq \tau\}}{\#\{i : y_i=0\}}
\]

## Official metric: TPR @ 0.1% FPR

**Leaderboard score** = TPR at the operating point where **FPR = 0.1%** (0.001).

That means: among all truly clean tokens (label=0), at most **1 in 1,000** may be flagged as watermarked.

### Procedure

1. Pool all tokens in the leaderboard subset (labels hidden on test; you simulate on val).
2. Sort tokens by your score **descending** (most “watermarked” first).
3. Walk thresholds τ (in practice: one per distinct score, or via ROC curve).
4. Find the **smallest τ** (strictest) such that **FPR ≤ 0.001**.
5. Read **TPR** at that τ → that is your score.

In plain language: *“How many real watermarked tokens do you catch if you are only allowed to mis-flag 0.1% of clean tokens?”*

### Numeric example

- 10,000 clean tokens (label=0), 2,000 watermarked (label=1).
- FPR = 0.1% → budget of **10** false positives on clean tokens.
- Descend the ranked list until ~10 clean tokens would be flagged.
- Suppose 800 of the 2,000 watermarked tokens rank above that cutoff:

\[
\text{TPR} = 800 / 2000 = 0.40 \quad\Rightarrow\quad \text{leaderboard score} = 0.40
\]

## Continuous scores vs “≥ 0.5”

Slides say “score ≥ 0.5 → watermarked.” That is a **pedagogical shortcut**, not the actual leaderboard threshold.

The real threshold τ is **derived from your score distribution** to hit FPR = 0.1%. It may be 0.05, 0.73, 0.99, etc.

What matters is **relative ranking**:

- Monotonic remapping (e.g. `score * 0.5 + 0.25`) does **not** change TPR@FPR.
- Pushing watermarked tokens above almost all clean tokens matters far more than hitting 0.5 on average.

## What the metric rewards

At FPR = 0.1%, the false-positive budget is tiny:

- One clean token scored too high “costs” a slot in your ~1 FP per 1,000 clean tokens budget.
- Better to miss weak watermarked tokens than to over-score human or clean LLM text.
- Contiguous watermarked **spans** help; isolated high-score spikes hurt.

Good tactics: calibrate on **validation** (not train), temporal smoothing over neighbors, penalize isolated peaks, optimize TPR@0.1%FPR directly — not accuracy at τ=0.5.

## Public vs private leaderboard

Test documents are split **deterministically** by hash of `document_id`:

| Subset | Share | When visible |
|---|---|---|
| **Public** | 30% of test docs | During hackathon |
| **Private** | 70% of test docs | Final ranking |

All tokens of a document stay in the same subset. Final placement uses the **private** score.

## API behavior

- Each submit returns the score for **that** submission.
- Leaderboard keeps the **best score per team** (a worse submit does not overwrite a better one).
- No per-token feedback on test — only the pooled TPR@0.1%FPR.
- Cooldown between submits (~5 min; see hackathon API skill).

## Simulate locally (train / val)

Pool all tokens from labeled JSONL, then compute TPR@0.1%FPR:

```bash
cd cispa_final
python shared/task1_eval.py \
  --dataset path/to/val.jsonl \
  --predictions output/val_scores.jsonl \
  --method "your approach name"
```

Implementation (`shared/task1_eval.py`): `sklearn.metrics.roc_curve` on pooled labels/scores, then `np.interp(0.001, fpr, tpr)`.

Requirements: predictions must cover val docs; `len(scores) == len(labels)` per document; both classes (0 and 1) must appear in the pool.

## One-line summary

Submit a **score per token**; organizers compare to hidden **0/1** labels; leaderboard = **recall on watermarked tokens at the point where only 0.1% of clean tokens are false alarms**, computed on **all pooled tokens** in the subset.
