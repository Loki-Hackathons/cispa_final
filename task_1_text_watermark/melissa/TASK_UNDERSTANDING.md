# TASK UNDERSTANDING — Text Watermark Localization (Task 1)

Sources used: `docs/subject/task_1.md`, `docs/Task 1 Text Watermark Localization.md`,
the CISPA slide deck (`Tasks-Final.pdf`), and the **TextSeal** paper
(`TextSeal_..._Protection.pdf`).

---

## 1. What the task asks

Given a tokenized document, predict for **each token** a confidence score `∈ [0, 1]`
that the token was produced **while a watermarking process was active** during
generation.

- Ground truth is **binary per token**: `1` = watermark was active, `0` = not.
- The label comes from the **generation pipeline** (the watermark switch state), **not**
  from re-running a detector. A detector can flag a clean token by chance (false positive)
  or miss a real watermarked token (false negative). So the target is the *true hidden
  state*, and detector outputs are only noisy evidence for it.
- We do **not** need to say *which* watermark family a token belongs to — only
  watermarked vs. not.

## 2. What "localized watermark" means here

A single document is one continuation that can **mix**:
- clean model-generated text,
- human-written text,
- one or more **watermarked spans**, possibly from **different schemes**.

So the watermark is *localized*: only parts of the document are watermarked, and we must
find **where**. This is exactly the setting TextSeal §3.3 studies (watermarked segments
diluted inside a larger document). The difference: TextSeal returns *document-level*
p-values / region intervals; here we must return a **per-token score**.

Watermark families present: **TextSeal, Gumbel-Max, Unigram, KGW (red-green list).**

## 3. Exactly what we predict + output format

- Per-token score in `[0, 1]`. Length must equal the document's token count.
- Submission = `.jsonl`, **one line per test document**:
  ```json
  {"document_id": "1", "scores": [0.02, 0.04, 0.91, 0.87]}
  ```
- Rules: exactly `document_id` + `scores`; each test doc exactly once; `scores` numeric,
  finite, in `[0,1]`; no hard labels / text / token ids / family predictions.

## 4. Dataset structure (`SprintML/watermark_localization`)

1,500 documents. Tokenizer: **`Qwen/Qwen2.5-7B-Instruct`**.

| Split | Docs | Labels |
|-------|------|--------|
| train | 90   | yes    |
| val   | 90   | yes    |
| test  | 1320 | hidden |

Train/val record:
```json
{
  "document_id": "train_1",
  "text": "Paul's statement",
  "token_ids": [25300, 594, 5114],
  "token_pieces": ["Paul", "'s", " statement"],
  "labels": [0, 0, 1]
}
```
Test records omit `labels`. `token_ids` / `token_pieces` / `labels` are index-aligned and
equal length. A **YAML** (shipped with the data) gives the tokenizer, watermark **keys**,
detector params, and the official watermark repos + commits → keys and detector code are
**available** (white-box, not blind).

Token notes: `Ġ` = leading space, `Ċ` = newline. Every token needs a score, including
chat-template markers.

## 5. Metric

**TPR @ 0.1 % FPR**, computed by **pooling all tokens across all documents** (not per-doc
average). It is a **ranking metric**: only the ordering of scores matters, so we optimize
separation of watermarked vs clean at a very strict false-positive budget (0.1 %). The
absolute value / 0.5 threshold is conceptual only.

Leaderboard: 30 % public / 70 % private, split deterministically by `document_id` (a
document's tokens never span both).

## 6. Main difficulties

1. **KGW CUDA Philox trap** — greenlists must be recomputed with `torch.randperm` on a GPU
   generator, else ~1/3 of KGW tokens read as clean. Biggest correctness risk.
2. **Very strict FPR (0.1 %)** — one noisy clean token scored high can hurt a lot; we need
   clean tokens' scores to be *reliably* low. Calibration matters more than raw AUC.
3. **Noisy per-token detector stats** — single-token PRF/greenlist evidence is weak;
   must be fused across the 4 detectors and smoothed over neighbors.
4. **Mixed schemes + human text** in one doc — no single detector fires everywhere.
5. **Tiny labeled set** (90+90) — overfitting a calibrator/thresholds is easy.
6. **Repeated n-gram / (context, token) tuples** produce correlated detector scores that
   break the independence assumption (TextSeal §4.1, App. A.4) — dedup when aggregating.

## 7. TextSeal ideas that transfer to this task

| TextSeal idea | How we use it for per-token localization |
|---|---|
| **Gumbel-Max score** `s_t = -ln(1 - R_t)`, `R_t = PRF(x_t, context, K)` | Per-token feature. High `s_t` ⇒ likely watermarked. |
| **Entropy weighting** `wᵢ = f(Hᵢ)`, concave `√Ĥ` best (Fig. 9) | Low-entropy tokens carry ~no watermark signal (their score ≈ null). Down-weight / expect near-null there; use entropy as a feature so the calibrator learns this. |
| **Dual-key routing** score `sᵢ = (1-α)s⁽¹⁾ + α s⁽²⁾` (α≈0.1) | TextSeal docs are dual-key; compute both keys' PRF and early-fuse per token. |
| **Moment-matched Gamma p-value** under H0 | Convert raw score sums to calibrated per-token / per-window p-values → comparable across detectors. |
| **Localized region search** (dyadic windows, §3.3) | Watermarked regions are contiguous → smooth token scores over local windows; boost tokens inside a high-density span. |
| **Deduplicate (context, token) tuples** (§4.1) | Avoid double-counting correlated signals when building window statistics. |
| **KGW green-list count / z-test** (Kirchenbauer) | Per-token green membership + local green fraction as features (with CUDA Philox!). |

## 8. Signal model per detector (per token `t`, context = k previous tokens)

- **Gumbel-Max / TextSeal:** recompute `R_t = PRF(token_id, context, key)`; feature
  `-ln(1 - R_t)` (exp(1) mean 1 under H0, larger when watermarked). TextSeal: fuse two keys.
- **Unigram:** fixed key-seeded green list over the whole vocab; feature = token ∈ green (0/1)
  + local green fraction over a window.
- **KGW:** context-seeded green list (**CUDA Philox**); feature = token ∈ green + local
  green fraction / running z-score.
- **Entropy `Hᵢ`:** from a proxy LM forward pass (or approximated) — gates how much any
  score should count.

These stacked features + local context are fed to a small calibrator trained on train,
tuned on val for TPR@0.1%FPR, then smoothed across neighbors.
