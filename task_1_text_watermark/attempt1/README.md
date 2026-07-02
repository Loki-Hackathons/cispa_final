# Task 1 — Text Watermark Localization (attempt 1)

**Owners:** Alexandre (ansart1), Bastian, Melissa

Spec: [Task 1 Text Watermark Localization.md](../../docs/Task%201%20Text%20Watermark%20Localization.md)

Metric: **TPR @ 0.1% FPR** · Submit `.jsonl` · API task id `30-watermark-localization`

## Layout

| Path | Role |
|------|------|
| [`../watermark_config.yaml`](../watermark_config.yaml) | Keys, detector params, pinned repo commits |
| [`../vendor/`](../vendor/) | Git submodules (TextSeal+Gumbel, KGW, Unigram) |
| `main.py` | Inference pipeline (to be added) |
| `output/` | `submission.jsonl`, val predictions |
| `logs/` | SLURM / run logs |

## Setup

```bash
# From cispa_final root — pin detector submodules
bash scripts/task1/sync_watermark_repos.sh

# Dataset lives on scratch after hackathon_setup.sh, e.g.:
# /p/scratch/training2625/ansart1/loki/<watermark-dataset>/
# Copy task_template.py from that folder into this attempt if needed.
```

Local eval before API submit (when labels available):

```bash
python shared/task1_eval.py \
  --dataset /path/to/validation.jsonl \
  --predictions output/val_scores.jsonl \
  --method "baseline 4-detector ensemble"
```

Submit:

```bash
python shared/submit.py output/submission.jsonl \
  --task-id 30-watermark-localization --action submit \
  --method "..."
```

## Detector map (4 families → 3 repos)

- **TextSeal + Gumbel-Max** → `vendor/textseal/`
- **KGW** → `vendor/lm-watermarking/` (CUDA Philox required)
- **Unigram** → `vendor/unigram-watermark/`
