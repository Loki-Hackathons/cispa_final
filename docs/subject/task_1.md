# Task 1 — Text Watermark Localization

> **Note:** the CISPA slide deck mislabels this "Task 3" on some slides. This is **Task 1**. Task 2 = MGI (`docs/Task 2 Description.md`), Task 3 = FL gradient reconstruction (`docs/Task 3 Description.md`). Source spec: `docs/Task 1 Text Watermark Localization.md`.

Developed by: Maitri Shah and Louis Kerner.

## Goal

For every token in a document, predict a confidence score `[0, 1]` that the token was generated **while a watermarking process was active**.

- Label **1** = watermarked, **0** = non-watermarked.
- Ground truth comes from the generation pipeline (when watermarking was switched on), **not** from re-running a detector. A token can pass a detector's statistical test by chance without being watermarked, and vice versa.
- You do not need to identify which watermark family produced a token — only whether the token is in an active watermark region.

## Challenge

Each document is one tokenized continuation that may mix:
- clean model-generated text
- human-written text
- one or more watermarked spans (possibly different schemes per span)

Schemes present in the dataset: **TextSeal, Gumbel-Max, Unigram, KGW** (red-green list).

A YAML file provides the exact tokenizer, watermark keys, detector parameters, and the official watermark repos + commits — so the real secret keys and detector code are available, this is not blind/key-less detection. The difficulty is that raw per-token detector statistics are noisy (false positives from chance, false negatives from weak signal), so they must be combined/calibrated into a localization signal, not used raw.

## Data

Hugging Face: `SprintML/watermark_localization`. 1,500 documents total:

| Split | Docs | Labels |
|---|---|---|
| Training | 90 | yes |
| Validation | 90 | yes |
| Test | 1,320 | **no** (hidden) |

Record schema (train/val):

```json
{
  "document_id": "train_1",
  "text": "Paul's statement",
  "token_ids": [25300, 594, 5114],
  "token_pieces": ["Paul", "'s", " statement"],
  "labels": [0, 0, 1]
}
```

Test records omit `labels`. `token_ids`, `token_pieces`, `labels` are always the same length and index-aligned.

Tokenizer: `Qwen/Qwen2.5-7B-Instruct`. Use `token_ids` as authoritative — do not retokenize decoded `text` (not guaranteed to round-trip). `Ġ` = preceding space, `Ċ` = newline. All tokens (including chat markers) require a prediction.

**KGW gotcha:** greenlists were generated with `torch.randperm` on a CUDA (Philox) generator. Recomputing on CPU gives effectively random greenlists — ~1/3 of KGW-watermarked tokens will look unwatermarked if you don't replicate this.

## Submission format

One JSON object per test document, one line per document (`.jsonl`):

```json
{"document_id": "1", "scores": [0.02, 0.04, 0.91, 0.87]}
```

Rules: exactly `document_id` + `scores`; every test document exactly once; `scores` length must equal that document's token count; values numeric, finite, in `[0, 1]`; no hard labels/text/token ids/watermark-family predictions.

## Metric

**TPR @ 0.1% FPR**, computed by pooling all tokens across all documents in a leaderboard subset (not averaged per-document). Score ≥ 0.5 conceptually means "watermarked" but the metric itself only cares about ranking — see Q&A below.

Leaderboard split: 30% public / 70% private, assigned deterministically per `document_id` (all tokens of a document stay together).

## References

1. Kirchenbauer et al., "A Watermark for Large Language Models" (KGW), ICML 2023.
2. Sander et al., "TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection", arXiv 2026.
3. Aaronson & Kirchner, "Watermarking GPT Outputs" (Gumbel-Max), 2023.
4. Zhao et al., "Provable Robust Watermarking for AI-Generated Text" (Unigram), ICLR 2024.
