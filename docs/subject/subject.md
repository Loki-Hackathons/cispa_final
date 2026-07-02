# CISPA Grand Finals — Subject

Source docs: `docs/Task 1 Text Watermark Localization.md`, `docs/Task 2 Description.md`, `docs/Task 3 Description.md`

Cluster setup: `docs/Hackathon_Setup Finale.md` + `scripts/cluster/README.md`

## Project / paths

| Resource | Path / value |
|----------|----------------|
| JuDoor / SLURM project | **`training2625`** (finals — not regional `training2557`) |
| Team scratch (after owner setup) | `/p/scratch/training2625/ansart1/loki/` |
| SLURM reservation | `cispahack` |
| Leaderboard (all tasks) | http://35.192.205.84/leaderboard_page |

Datasets are downloaded by `hackathon_setup.sh` into the team scratch folder.

## Tasks

| # | Name | Metric | Submission |
|---|------|--------|------------|
| 1 | Text Watermark Localization | TPR @ 0.1% FPR | `.jsonl` (token scores per document) |
| 2 | Member vs Generated Inference (MGI) | DetectorScore × (1−MSE), mean over 1800 images | `.npz` (256×256×3 images, IDs 0000–1799) |
| 3 | FL Data Reconstruction from Gradients | Mean SSIM (paired), scaled to [0,1] | `.pt` dict `{model_id: Tensor(128,3,64,64)}` |

### Task 1 — Text Watermark Localization

- Token-level confidence: was watermarking **active** during generation (not raw detector score).
- Watermarks: TextSeal, Gumbel-Max, Unigram, KGW. Tokenizer: `Qwen/Qwen2.5-7B-Instruct`.
- Dataset: https://huggingface.co/datasets/SprintML/watermark_localization
- **KGW note:** greenlists use CUDA Philox — recompute on GPU, not CPU.
- Cooldowns: submit ~5 min (verify at finals).

### Task 2 — MGI (attack detector)

- Goal: fool hidden 3-class detector (M/N/G) on 1800 modified images while preserving quality.
- 6 misclassification directions × 300 images each (IDs 0000–1799).
- RAR generative model + 900 reference images in Dataset.zip.
- Dataset: https://huggingface.co/datasets/SprintML/MGI

### Task 3 — FL gradient reconstruction

- Reconstruct 1536 private images (12 models × 128 images) from gradients + white-box models.
- Output: float32 tensors (128, 3, 64, 64) per model, values in [0, 1].
- Cooldown: 5 min between submits (2 min after error).

## API / Leaderboard

- Base URL: TBD (organizer API key per team)
- Leaderboard: http://35.192.205.84/leaderboard_page
- API keys: cluster `.env` only — never commit

## Task directories (repo)

```
task_1_text_watermark/attempt1/
task_2_mgi/attempt1/
task_3_fl_reconstruction/attempt1/
```

Official `task_template.py` / `main.py` land in scratch after `hackathon_setup.sh`.

## Raw spec files

| Task | Doc |
|------|-----|
| 1 | `docs/Task 1 Text Watermark Localization.md` |
| 2 | `docs/Task 2 Description.md` |
| 3 | `docs/Task 3 Description.md` |
