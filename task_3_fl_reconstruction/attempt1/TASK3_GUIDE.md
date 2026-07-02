# Task 3 — FL Data Reconstruction from Gradients

**Operational guide for Team Loki (CISPA Grand Finals)**  
**Owner:** Bastian Paoli (`paoli1`) — Task 3 lead  
**Last updated:** 2026-07-02 (post cluster setup)

> **Security:** The API token lives in **`.env`** only (gitignored). This guide is safe to commit.  
> Never put secrets in `.md` files or push `.env` to GitHub.

---

## 1. Team & API access

Organizer message:

> We are glad to have you on board for the hackathon championship.
>
> The following is your team's API key for accessing the API to submit your solutions to the score board. You need this key, otherwise, you cannot submit your solutions. Please paste this in your scripts for submitting the solutions everywhere and do not share with other teams.

| Field | Value |
|-------|-------|
| **Team name** | Loki |
| **API token** | in `.env` → `CISPA_API_KEY` (see below) |
| **Team number (venue table)** | 99 |

### Setup `.env` (cluster)

```bash
cd /p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
# or in repo: task_3_fl_reconstruction/attempt1/

cp .env.example .env
# Ask Task 3 lead (paoli1) for CISPA_API_KEY if missing, then edit .env
```

Load into your shell before submit:

```bash
set -a
source .env
set +a
echo "API key set: ${CISPA_API_KEY:0:8}..."   # prints prefix only
```

**Usage in `task_template.py` (read from env, do not hardcode):**

```python
import os
API_KEY = os.environ["CISPA_API_KEY"]
SUBMIT = True
FILE_PATH = os.environ.get("TASK3_SUBMISSION_PATH", "submission.pt")
```

**Leaderboard:** http://35.192.205.84/leaderboard_page

**Submission cooldown:** 5 minutes between successful submits; 2 minutes after a failed submit (format/validation error).

---

## 2. Objective (official spec)

You play the role of a **malicious FL server** that received client gradients.

| Given | Not given |
|-------|-----------|
| 12 global models (white-box `state_dict`) | Private training images |
| Per-model gradient dict (batch of 128) | Labels |

**Goal:** Reconstruct **1536 images** total (12 models × 128 images each) as accurately as possible.

**Metric:** Mean **SSIM** after **optimal one-to-one matching** per model (Hungarian-style pairing). Each ground-truth image is matched at most once — **duplicating** a good reconstruction does not help.

**Leaderboard:** Scores shown on **30%** of data during the hackathon; final ranking uses the hidden **70%** (avoid overfitting hyperparameters to leaderboard feedback alone).

**References (organizers):**

- Boenisch et al. — *When the Curious Abandon Honesty* (trap weights, analytic extraction) — `docs/2112.02918v2.pdf`
- Zhu et al. — DLG (optimization-based)
- Geiping et al. — Inverting Gradients (optimization + TV)

---

## 3. Cluster setup (Team Loki)

| Role | Jülich user | Responsibility |
|------|-------------|----------------|
| **Owner (lead)** | `dougnon1` | Ran `hackathon_setup.sh` once |
| **Teammate (Task 3)** | `paoli1` | `teammate.sh` + FL reconstruction |

**Teammate shell (every new session):**

```bash
ssh juelich   # or ssh paoli1@jureca.fz-juelich.de
jutil env activate -p training2625
cd /p/scratch/training2625/dougnon1
source teammate.sh   # OWNER="dougnon1", TEAM_FOLDER="Loki"
```

**Correct working directory (has `.venv`):**

```text
/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction/
```

> Do **not** use `/p/scratch/training2625/dougnon1/FL_Data_Reconstruction/` (duplicate HF tree **without** `.venv`).

**Activate environment:**

```bash
cd /p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
source .venv/bin/activate
module load GCC CUDA PyTorch torchvision   # before GPU jobs
```

**Team repo (code):**

```bash
/p/scratch/training2625/dougnon1/Loki/cispa_final/
# or home: /p/home/jusers/paoli1/jureca/code/cispa_final
```

**Logs / outputs:** `output/` under the dataset folder.

---

## 4. On-disk layout

```text
/p/scratch/training2625/dougnon1/Loki/
├── cispa_final/              # team GitHub repo
├── FL_Data_Reconstruction/   # Task 3 (THIS TASK)
│   ├── .venv/
│   ├── models/               # model1.pt … model12.pt
│   ├── gradients/            # model1.pt … model12.pt
│   ├── task_template.py      # loader + submit example
│   ├── main.py
│   ├── output/
│   ├── requirements.txt
│   └── pyproject.toml
├── MGI/                      # Task 2
└── watermark_localization/     # Task 1
```

---

## 5. Data file formats (verified on cluster)

### 5.1 Gradient file — `gradients/model{i}.pt`

```python
{
  "gradients": {param_name: Tensor, ...},  # mirrors model.named_parameters()
  "family": "mlp" | "cnn" | "vit",
  "activation": "relu" | "tanh" | "sigmoid" | "gelu",
  "feature_shape": (C, H, W),              # per-image shape at model input
  "batch_size": 128,
  "model": ...                             # optional extra metadata
}
```

### 5.2 Model file — `models/model{i}.pt`

Plain `state_dict`: `{param_name: weight Tensor, ...}`

### 5.3 Submission file — `submission.pt`

```python
{
  "model1":  Tensor(128, 3, 64, 64),   # float32, values in [0, 1]
  "model2":  Tensor(128, 3, 64, 64),
  ...
  "model12": Tensor(128, 3, 64, 64),
}
```

- Keys must be **exactly** `"model1"` … `"model12"`.
- Final images are always **(128, 3, 64, 64)** for submission, even if `feature_shape` differs internally (resize/interpolate + channel handling as needed).
- Order within the 128 slots does not matter (optimal matching).

### 5.4 Quick inspect

```bash
cd /p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
source .venv/bin/activate
python task_template.py
```

---

## 6. Model inventory (measured 2026-07-02)

| Model | Family | Activation | `feature_shape` | Key layers | Tier |
|-------|--------|------------|-----------------|------------|------|
| **1** | mlp | sigmoid | (3, 64, 64) | `net.0`, `net.2`, `net.5` | T1/T3 |
| **2** | cnn | tanh | (8, 16, 16) | `conv`, `fc1`, `head` | T2/T3 |
| **3** | cnn | relu | (8, 8, 8) | `conv`, `fc1`, `head` | T2 |
| **4** | mlp | tanh | (3, 64, 64) | `net.0`, `net.2`, `net.5` | T1/T3 |
| **5** | mlp | relu | (3, 64, 64) | `net.0`, `net.2`, `net.5` | **T1** |
| **6** | cnn | relu | (8, 64, 64) | `conv`, `fc1`, `head` | T2 |
| **7** | cnn | relu | (8, 16, 16) | `conv`, `fc1`, `head` | T2 |
| **8** | mlp | relu | (3, 64, 64) | `net.0`, `net.2`, `net.5` | **T1** |
| **9** | vit | gelu | (3, 64, 64) | ViT blocks + `head` | **T3/T4** |
| **10** | cnn | relu | (8, 64, 64) | `conv`, `fc1`, `head` | T2 |
| **11** | vit | gelu | (3, 64, 64) | ViT blocks (deeper) + `head` | **T3/T4** |
| **12** | cnn | sigmoid | (8, 64, 64) | `conv`, `fc1`, `head` | T2/T3 |

### Layer shapes — MLP family (models 1, 4, 5, 8)

| Parameter | Shape |
|-----------|-------|
| `net.0.weight` | (1024, 12288) |
| `net.0.bias` | (1024,) |
| `net.2.weight` | (1024, 1024) |
| `net.2.bias` | (1024,) |
| `net.5.weight` | (200, 1024) |
| `net.5.bias` | (200,) |

`12288 = 3 × 64 × 64` → first layer sees flattened RGB 64×64 images.

### Layer shapes — CNN family (models 2, 3, 6, 7, 10, 12)

Typical keys: `conv.weight`, `conv.bias`, `fc1.weight`, `fc1.bias`, `head.weight`, `head.bias`.

Input feature maps vary: `(8, 64, 64)`, `(8, 16, 16)`, `(8, 8, 8)`.

### ViT family (models 9, 11)

Full transformer stack; extraction via analytic FC formula does not apply directly → **gradient-matching optimization** (Geiping) on pixels.

---

## 7. Solution strategy (tiered pipeline)

### Tier 1 — Analytic extraction (MLP + relu/tanh on first linear layer)

Core identity (Boenisch et al., Eq. 5–6) for neuron \(i\) with active ReLU:

\[
\frac{\partial L}{\partial w_i^T} = \frac{\partial L}{\partial b_i}\, x^T
\qquad\Rightarrow\qquad
x^T = \left(\frac{\partial L}{\partial b_i}\right)^{-1} \frac{\partial L}{\partial w_i^T}
\]

**Implementation (first layer `net.0` for MLP):**

```python
gW = grad["gradients"]["net.0.weight"]   # (1024, 12288)
gb = grad["gradients"]["net.0.bias"]   # (1024,)
valid = gb.abs() > 1e-8
X = gW[valid] / gb[valid].unsqueeze(1)   # (n_active, 12288)
imgs = X.view(-1, 3, 64, 64).clamp(0, 1)
```

Each row ≈ one image candidate (or blur if multiple images activated the neuron).

**Priority models:** 5, 8 (relu); then 4 (tanh — try same formula, may need scaling); model 1 (sigmoid — weaker isolation, likely needs T3 fallback).

Also try `net.2` gradients for extra candidates.

### Tier 2 — CNN (extract from `fc1`, upscale to 64×64)

1. Analytic extraction on `fc1.weight` / `fc1.bias` → feature-space images.
2. Map `(8, H, W)` → RGB `(3, 64, 64)` via interpolation + channel projection if needed.
3. If conv is approximately identity-preserving, features may already look like downsampled images.

**Models:** 3, 6, 7, 10 (relu); 2, 12 (tanh/sigmoid — harder).

### Tier 3 — Gradient matching (Geiping / DLG family)

When analytic path fails (ViT, sigmoid/gelu, heavy mixing):

\[
\hat{x}^\star = \arg\min_{\hat{x}} \; \Big[1 - \cos\big(\nabla_W L(\hat{x}, \hat{y}), G\big)\Big] + \alpha\,\mathrm{TV}(\hat{x})
\]

- Infer labels via **iDLG** (sign of last-layer gradient).
- Warm-start from Tier 1/2 candidates when available.
- Run on GPU (`sbatch`); batch reconstruct groups of images if memory allows.

**Models:** 9, 11 (ViT); fallback for 1, 2, 4, 12.

### Tier 4 — Fill 128 distinct slots

1. Score candidates with **gradient re-simulation** (no ground truth needed).
2. Deduplicate (cluster by L2/cosine on flattened images).
3. Keep top 128 **distinct** reconstructions.
4. Fill remaining slots with diverse priors (not copies of the same image).

---

## 8. Local validator (no labels required)

For each candidate image \(\hat{x}\):

1. Forward + backward through the **known** model.
2. Compare \(\hat{G}\) to provided \(G\) (cosine or L2).

Use this to rank rows, tune TV weight / learning rate, and decide whether a leaderboard submit is worthwhile — **do not burn 5-minute cooldown on worse submissions**.

---

## 9. Submission workflow

```bash
cd /p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
source .venv/bin/activate

# 1. Build submission dict → submission.pt
python run.py --out submission.pt

# 2. Load secrets, then submit
set -a && source .env && set +a
python task_template.py   # SUBMIT=True; reads CISPA_API_KEY from env
```

Or integrate submit block from `task_template.py` into `run.py`.

**First submit:** use random/placeholder only to validate API pipeline, then switch to real reconstructions.

---

## 10. GPU / SLURM

Login node (`jrlogin08`) is for editing and light CPU tests. **GPU jobs** via SLURM:

```bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:1
```

Use **1 GPU per optimization job**; parallelize hard models (9, 11) across jobs. Coordinate with team in `docs/notes-communes.md`.

Interactive debug:

```bash
salloc -p dc-gpu-devel -t 30 -N 1 -A training2625 --gres=gpu:1
```

### 10.1 Parallel per-model workflow (recommended under GPU congestion)

The optimization now covers **all three families** (MLP, CNN, ViT) and, per
model, keeps whichever of {analytic, optimized} reproduces the **observed
gradient** best (`run.py` prints `[select] ...`). This selection uses the real
leaked gradient, never the leaderboard, so it cannot cause public-split
overfitting.

To grab GPU slots as they free (backfill), submit **one array task per model**:

```bash
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
mkdir -p output/parts

sbatch slurm_array.sh          # 12 independent tasks -> output/parts/model{i}.pt
# throttle concurrency if needed: sbatch --array=1-12%4 slurm_array.sh

squeue -u $USER                # watch tasks trickle through as GPUs free
```

Each task writes `output/parts/model{i}.pt`. Assemble + submit **incrementally**
on the login node as parts land (no GPU needed, instant):

```bash
python merge.py --parts output/parts --base submission.pt --out submission.pt
python submit.py --check submission.pt      # local validate + quality report

cp submission.pt $TASK3_DATA_ROOT/submission.pt
cd $TASK3_DATA_ROOT && python task_template.py   # SUBMIT=True (5-min cooldown)
```

**Before trusting ViT (9, 11):** confirm the architecture matches the rebuilt
forward (timm naming). The builder fast-fails on mismatch and keeps noise, so a
wrong guess never wastes a slot — but you can verify keys first:

```bash
python inspect_models.py --keys 9 11
```

**Steps:** MLP/CNN use 4000, ViT 6000 (set in `slurm_array.sh`). These are
principled Geiping defaults; do **not** tune them against leaderboard feedback.

---

## 11. Pitfalls

| Pitfall | Consequence | Mitigation |
|---------|-------------|------------|
| Wrong folder (no `.venv`) | cannot run | use `Loki/FL_Data_Reconstruction` |
| Duplicate images in submission | wasted SSIM slots | dedup + 128 unique outputs |
| Divide by `grad_b ≈ 0` | NaN explosions | filter `\|gb\| > eps` |
| Overfit to 30% leaderboard | bad final 70% | tune on gradient re-sim, not LB |
| Cooldown spam | lose submit windows | submit only when local score improves |
| Ignore `feature_shape` | wrong reshape | always read from gradient file |
| Submit wrong dtype/shape | rejected submit | `float32`, `(128,3,64,64)`, keys `model1`…`model12` |
| API key in public git | leak to other teams | use `.env` only (gitignored); never in `.md` |

---

## 12. Code layout (`cispa_final/task_3_fl_reconstruction/attempt1/`)

```text
attempt1/
├── TASK3_GUIDE.md      # this file
├── .env / .env.example # secrets (API key) — .env gitignored
├── config.py           # paths, constants (TASK3_DATA_ROOT overridable)
├── utils.py            # IO, validate, image mapping, quality score, dedup
├── extract.py          # introspect + analytic extraction (eq. 6), MLP & CNN
├── rebuild.py          # MLP/CNN/ViT rebuild, iDLG labels, gradient re-sim, Geiping opt
├── run.py              # orchestrate models → submission.pt (or per-model parts)
├── merge.py            # assemble submission.pt from output/parts (array jobs)
├── submit.py           # strict validator + upload wiring (reads .env)
├── inspect_models.py   # shape table, --keys full dump, analytic preview grids
├── slurm_run.sh        # GPU job: analytic + optimize MLP into submission.pt
└── slurm_array.sh      # GPU array (1 task/model) -> output/parts (backfill)
```

Data stays in scratch; code reads `TASK3_DATA_ROOT` (defaults to the team path).

### Quick start on cluster

```bash
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
source /p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction/.venv/bin/activate

python inspect_models.py                 # 1. confirm shapes match our assumptions
python inspect_models.py --preview 5 8   # 2. eyeball analytic reconstructions
python run.py                            # 3. build submission.pt (analytic, all 12)
python submit.py --check submission.pt   # 4. validate format before uploading
# 5. optimize hard models on GPU: sbatch slurm_run.sh
```

> **Verify on first run:** CNN `feature_shape` vs `fc1` input size, and the
> channel/spatial mapping in `utils.features_to_image`. `inspect_models.py`
> prints the real shapes so you can adjust `extract.extract_cnn` if the conv
> output layout differs from the square-root assumption.

---

## 13. Priority order (score per unit time)

1. **Models 5 & 8** (mlp + relu) — analytic `net.0`, expect many free points.
2. **Models 3, 6, 7, 10** (cnn + relu) — `fc1` analytic + upscale.
3. **Models 4, 1** (mlp + tanh/sigmoid) — analytic attempt + T3 fallback.
4. **Models 2, 12** (cnn + tanh/sigmoid).
5. **Models 9, 11** (ViT + gelu) — GPU optimization last, highest cost.

---

## 14. Session checklist

- [ ] `ssh juelich` + TOTP
- [ ] `jutil env activate -p training2625`
- [ ] `source teammate.sh` (if new shell)
- [ ] `cd .../Loki/FL_Data_Reconstruction && source .venv/bin/activate`
- [ ] `tmux attach -t hackathon || tmux new -s hackathon`
- [ ] Run `python task_template.py` smoke test
- [ ] Implement `analytic.py` on model5
- [ ] First real `submission.pt` + API test
- [ ] Log progress in `docs/notes-communes.md`

---

## 15b. v2 analytic pipeline (2026-07-02) — run this first

New modules that fix the two biggest score leaks (score 0.24 → target >0.5). All
CPU, seconds per model, no leaderboard needed to validate.

| File | What it does |
|------|--------------|
| `channels.py` | Detects the CNN conv **transmit filters** (Boenisch Appendix B) and inverts them **channel-by-channel** to RGB, instead of averaging all 8 feature channels (which mixed image + noise channels). Falls back to averaging when no transmit structure exists — never regresses. |
| `separation.py` | **Isolated-image recovery by clustering** the analytic rows. Rows that reconstruct the *same* private image are near-identical (single-image neurons); mixture rows are lone outliers. Tight/populous clusters = high-confidence reconstructions, averaged to denoise. Also `effective_rank()` to tell ReLU-leaky from smooth-dense models. `diversify_fill()` fills leftover slots with augmented variants of real reconstructions (beats white noise ≈ 0 SSIM). |
| `reconstruct_v2.py` | Orchestrates per family + `--diagnose` mode (recoverability table + preview PNGs). MLP-ReLU rows are **clamped** (true [0,1] scale) instead of min/max-stretched → better SSIM luminance term. |
| `selftest_v2.py` | Synthetic trap-weight MLP+CNN end-to-end check. Passes at MLP 1.00 / CNN 0.99 nearest-match SSIM. |

**Step 0 — environment (copy-paste exactly; never use `...` paths from chat):**

```bash
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
source setup_cluster.sh
```

This sets `TASK3_DATA_ROOT`, activates `.venv`, and `cd`s into `attempt1`. If you
already ran `export TASK3_DATA_ROOT=.../FL_Data_Reconstruction` by mistake, open a
new shell or run `unset TASK3_DATA_ROOT` then `source setup_cluster.sh` again.

**Step 1 — diagnose (no GPU), look at the previews:**

```bash
python channels.py                       # which CNN models are transmit-trapped?
python reconstruct_v2.py --diagnose      # per-model: valid rows / clusters / conf>.5 / transmit
#  -> eyeball output/v2_diag/v2_model*.png : recognizable = recoverable
```

Read the table: **high `clusters` + high `conf>.5`** = strongly recoverable (submit as-is);
low clusters = smooth/defended model (needs `mlp_reconstruct.py --refine` on GPU or stays prior-filled).

**Step 2 — build + validate + submit:**

```bash
python reconstruct_v2.py --out submission_v2.pt          # all 12 (analytic++)
python submit.py --check submission_v2.pt                # strict format check
cp submission_v2.pt $TASK3_DATA_ROOT/submission.pt
cd $TASK3_DATA_ROOT && python task_template.py           # SUBMIT=True (5-min cooldown)
```

**Step 3 — hard models (smooth sigmoid/tanh 1,4,2,12 and ViT 9,11), GPU:**
warm-start exact-forward gradient matching from the v2 base, keep only if it beats
the analytic re-sim (the anti-regression guard in `run.py`/`mlp_reconstruct.py`):

```bash
python mlp_reconstruct.py --base submission_v2.pt --out sub_refine.pt --models 1 4 --refine --steps 4000
```

**Do NOT tune steps/thresholds against leaderboard feedback** — validate with the
`--diagnose` previews and gradient re-sim only (public split is 30%; final is the hidden 70%).

## 15. Contacts

- JSC support: sc@fz-juelich.de
- Project advisor: a.herten@fz-juelich.de
- SLURM account: `training2625`, reservation: `cispahack`
