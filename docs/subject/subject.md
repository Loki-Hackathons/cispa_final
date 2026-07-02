# CISPA Grand Finals — Subject

Cluster setup: [Hackathon_Setup Finale.md](../Hackathon_Setup%20Finale.md) + [scripts/cluster/README.md](../../scripts/cluster/README.md)

## Project / systems

| Resource | Value |
|----------|-------|
| JuDoor / SLURM project | **`training2625`** |
| Login (data access) | **`judac.fz-juelich.de`** — JUDAC, global filesystem, no GPU |
| Login (GPU compute) | **`jureca.fz-juelich.de`** — JURECA + SLURM (when granted) |
| Team scratch (after owner setup) | `/p/scratch/training2625/ansart1/loki/` |
| SLURM reservation | `cispahack` |
| Leaderboard (all tasks) | http://35.192.205.84/leaderboard_page |

Datasets are downloaded by `hackathon_setup.sh` into the team scratch folder.

## Team assignments

| Task | People | Directory |
|------|--------|-----------|
| **1** Text Watermark Localization | **Alexandre**, **Bastian**, **Melissa** | `task_1_text_watermark/attempt1/` |
| **2** MGI | **Florian** | `task_2_mgi/attempt1/` |
| 3 FL gradient reconstruction | TBD | `task_3_fl_reconstruction/attempt1/` |

## Tasks

| # | Name | Spec | Metric | Submission |
|---|------|------|--------|------------|
| 1 | Text Watermark Localization | [Task 1 Text Watermark Localization.md](../Task%201%20Text%20Watermark%20Localization.md) | TPR @ 0.1% FPR | `.jsonl` |
| 2 | Member vs Generated Inference (MGI) | [Task 2 Description.md](../Task%202%20Description.md) | DetectorScore × (1−MSE) | `.npz` |
| 3 | FL Data Reconstruction from Gradients | [Task 3 Description.md](../Task%203%20Description.md) | Mean SSIM | `.pt` |

### Task 1 — Text Watermark Localization

Full spec: [Task 1 Text Watermark Localization.md](../Task%201%20Text%20Watermark%20Localization.md)

- Token-level confidence: was watermarking **active** during generation (not raw detector score).
- Watermarks: TextSeal, Gumbel-Max, Unigram, KGW. Tokenizer: `Qwen/Qwen2.5-7B-Instruct`.
- Dataset: https://huggingface.co/datasets/SprintML/watermark_localization
- **KGW note:** greenlists use CUDA Philox — recompute on GPU, not CPU.

### Task 2 — MGI (attack detector)

Full spec: [Task 2 Description.md](../Task%202%20Description.md)

- Goal: fool hidden 3-class detector (M/N/G) on 1800 modified images while preserving quality.
- 6 misclassification directions × 300 images each (IDs 0000–1799).
- Dataset: https://huggingface.co/datasets/SprintML/MGI

### Task 3 — FL gradient reconstruction

Full spec: [Task 3 Description.md](../Task%203%20Description.md)

- Reconstruct 1536 private images (12 models × 128 images) from gradients + white-box models.
- Output: float32 tensors (128, 3, 64, 64) per model, values in [0, 1].
- Cooldown: 5 min between submits (2 min after error).

## API / Leaderboard

- Base URL: TBD (organizer API key per team)
- Leaderboard: http://35.192.205.84/leaderboard_page
- API keys: cluster `.env` only — never commit

## Task directories (repo)

```
task_1_text_watermark/attempt1/   ← Alexandre, Bastian, Melissa
task_2_mgi/attempt1/              ← Florian
task_3_fl_reconstruction/attempt1/
```

Official `task_template.py` / `main.py` land in scratch after `hackathon_setup.sh`.
