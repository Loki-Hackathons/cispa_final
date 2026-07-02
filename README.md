# CISPA Finals — Team Loki

European Championship in Trustworthy AI — Grand Finals preparation repo.

**Team:** Alexandre Ansart, Bastian Paoli, Florian Dougnon-Greder, Melissa Abider

## Quick start (cluster)

First-time setup: see [Setup](#setup-first-time) below.

```bash
ssh -i ~/.ssh/id_ed25519 -o Ciphers=aes256-ctr -o MACs=hmac-sha2-256-etm@openssh.com <user>@jureca.fz-juelich.de
jutil env activate -p training2625
cd /p/home/jusers/<user>/jureca/code/cispa_final
module load GCC CUDA PyTorch torchvision
source .venv/bin/activate
tmux new -s hackathon
```

Full checklist: [docs/hackathon-start-guide.md](docs/hackathon-start-guide.md)

## Quick start (local)

For mock dashboard, shared scripts, and CPU-only prototyping on your laptop. GPU jobs stay on JURECA.

**Windows (PowerShell):**

```powershell
cd cispa_final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r shared\requirements.txt
```

**Linux / macOS:**

```bash
cd cispa_final
python3 -m venv .venv
source .venv/bin/activate
pip install -r shared/requirements.txt
```

Verify:

```bash
python -c "import torch, timm, wandb, torchattacks; print('OK', torch.__version__)"
```

Deactivate when done: `deactivate`

> `.venv/` is gitignored. Local PyTorch installs as CPU-only by default; CUDA builds are handled on the cluster via modules + `uv`.

### Task 1 dataset (local)

Clone into `data/` (gitignored, see [data/README.md](data/README.md)):

```powershell
cd data
git lfs install
git clone https://huggingface.co/datasets/SprintML/watermark_localization
```

Browse **all labeled docs** (train + val, ground truth highlighted):

```powershell
python scripts/task1/view_dataset.py
# → http://127.0.0.1:8765
```

Cluster copy: `/p/scratch/training2625/ansart1/loki/watermark_localization/`. One-shot owner setup: `scripts/cluster/setup_task1_only.sh`.

## Repo map

```
cispa_final/
├── AGENTS.md                  # Cursor / agent instructions
├── docs/
│   ├── subject/subject.md     # Official task specs (fill when released)
│   ├── notes-communes.md      # Live team scratchpad
│   ├── cluster-guide.md       # JURECA setup
│   └── hackathon-start-guide.md
├── dashboard/                 # FastAPI backend (mock | live)
├── client/                    # React browser UI
├── scripts/
│   ├── task1/view_dataset.py  # Local GT browser (train/val)
│   └── cluster/setup_task1_only.sh
├── data/                      # HF dataset clones (gitignored) — see data/README.md
├── slurm/templates/           # 1/2/4 GPU job templates
├── shared/
│   ├── submit.py              # API submission
│   ├── analyze.py             # Result analysis
│   ├── dashboard.py           # Terminal TUI (fallback)
│   ├── team_state.py          # Shared state JSON (tasks + jobs progress)
│   ├── job_progress.py        # Report Slurm progress → dashboard
│   ├── wandb_utils.py         # W&B helper
│   └── requirements.txt
└── task_N_<name>/attempt1/     # Created when subject drops
```

## Common commands

Activate the venv first (`source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`).

Fill `CISPA_BASE_URL` + `CISPA_API_KEY` in `.env` (organizer team token). API URL and `--task-id` values: [docs/subject/subject.md](docs/subject/subject.md#api--leaderboard).

```bash
# Submit to leaderboard
python shared/submit.py output/submission.npz --task-id <TASK_ID> --action submit

# Analyze via API
python shared/analyze.py output/submission.npz --task-id <TASK_ID> --mode api --dataset <dataset.pt>

# SLURM
sbatch slurm/templates/2gpu_devel.sh
squeue -A training2625
```

## Team dashboard (browser)

Ops UI: SLURM queue, cooldowns, scores, Task 1 token viewer, failed jobs. Two modes:

| Mode | Where | Config |
|------|-------|--------|
| **mock** | Laptop | Default — no `config_local.py` |
| **live** | JURECA | `dashboard/config_local.py` with `MODE = "live"` |

**Do not** edit `dashboard/config.py` on the cluster — use gitignored `config_local.py` (see `dashboard/config_local.py.example`).

### Laptop (mock)

```powershell
cd cispa_final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r dashboard\requirements.txt
.\scripts\build_dashboard.ps1          # builds client/dist (once per UI change)
.\scripts\run_dashboard.ps1            # or: python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080 — yellow “Mock data” banner is expected.

Details: [docs/dashboard-mock-test.md](docs/dashboard-mock-test.md)

### JURECA (live)

**One-time venv** (login node — no GPU needed for the dashboard):

```bash
jutil env activate -p training2625
cd /p/home/jusers/ansart1/jureca/code/cispa_final
python3 -m venv .venv
source .venv/bin/activate
pip install -r dashboard/requirements.txt
cp dashboard/config_local.py.example dashboard/config_local.py   # MODE=live, gitignored
```

**Each session** (keep alive in tmux):

```bash
jutil env activate -p training2625
cd /p/home/jusers/ansart1/jureca/code/cispa_final
source .venv/bin/activate
tmux new -s dash    # or: tmux attach -t dash
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
```

**Build the React UI on your laptop** (npm is not on JURECA login nodes), then copy `client/dist` once:

```powershell
cd cispa_final
.\scripts\build_dashboard.ps1
scp -i $env:USERPROFILE\.ssh\id_ed25519 -r client/dist ansart1@jureca.fz-juelich.de:/p/home/jusers/ansart1/jureca/code/cispa_final/client/
```

**SSH tunnel from laptop** (leave this terminal open):

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -L 8080:127.0.0.1:8080 ansart1@jureca.fz-juelich.de
```

Open http://localhost:8080 — header should show **live** (no mock banner).

**Verify API:**

```bash
curl -s http://127.0.0.1:8080/api/health
# {"ok":true,"mode":"live"}
```

**Sync repo after git pull** (cluster only — never commit/push from JURECA):

```bash
bash scripts/cluster/sync_repo_from_github.sh
```

**Leaderboard panel:** shows your best scores from `team_state.json` after `shared/submit.py` runs. Full rankings: http://35.192.205.84/leaderboard_page (no JSON poll — avoids 404 spam).

**Terminal fallback:** `python shared/dashboard.py` (Rich TUI, same config).

---

## Common commands

### Cluster (JURECA)

Python **3.12** via `uv` (see [docs/cluster-guide.md](docs/cluster-guide.md)):

```bash
cd /p/home/jusers/<user>/jureca/code/cispa_final
module load GCC CUDA PyTorch torchvision
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -r shared/requirements.txt
cp .env.example .env   # fill in on cluster, never commit
wandb login
```

Reactivate in every new shell (after `module load …`):

```bash
source .venv/bin/activate
```

### Local (laptop)

Same venv path (`.venv/`), standard library `venv` + `pip`:

| OS | Create + install | Activate (each session) |
|----|------------------|-------------------------|
| Windows | `python -m venv .venv` then `pip install -r shared\requirements.txt` | `.\.venv\Scripts\Activate.ps1` |
| Linux / macOS | `python3 -m venv .venv` then `pip install -r shared/requirements.txt` | `source .venv/bin/activate` |

Optional: copy `.env.example` to `.env` for local API tests. Never commit `.env`.

## Docs

- [Hackathon start guide](docs/hackathon-start-guide.md) — day-0 checklist
- [Cluster guide](docs/cluster-guide.md) — JURECA, SLURM, tmux, ACLs
- [Shared notes](docs/notes-communes.md) — assignments, jobs, decisions

### Required reading (finals — organizer-assigned)

These three papers were **explicitly assigned** before the Grand Finals. Read them for task inspiration, threat models, and method patterns — not as a substitute for `docs/subject/subject.md` when the subject drops.

| Paper | Local doc | Core idea |
|-------|-----------|-----------|
| **MGI: Member vs Generated Inference** (Zhao et al., CISPA) | [docs/mgi-member-vs-generated-inference.md](docs/mgi-member-vs-generated-inference.md) | Distinguish *training members* from *model-generated* outputs in image generative models (IARs, diffusion). Standard MIA and attribution fail because likelihood scores overlap. **DCB** (Data Circuit Breaker): 3-stage cascade — (1) autoencoder self-consistency (reconstruction + VQ quantization error) filters generated samples; (2) MIA on remaining natural samples; (3) cross-generator CPD for model-derivative / data-circuit settings. Robust to verbatim memorization. |
| **TextSeal** (Sander et al., Meta) | [docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md](docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md) | Distortion-free LLM watermark (Gumbel-max + dual-key routing for diversity). **Entropy-weighted detection** + **multi-region localization** in mixed human/AI documents. **Radioactive**: watermark survives distillation — detect unauthorized use of model outputs. Zero inference overhead; compatible with speculative decoding. |
| **When the Curious Abandon Honesty** (Boenisch et al.) | [docs/when_the_curious_abandon_honesty_federated_learning_is_not_private.md](docs/when_the_curious_abandon_honesty_federated_learning_is_not_private.md) | FL is not private by default: gradients leak training data. **Trap weights** — active server re-initializes FC-layer weights so ReLU activations isolate single batch items; **perfect reconstruction** from gradient projection (no optimization). Scales to ImageNet batches of 100. Defenses: DP (CDP fails vs malicious server), large local batches, leaky ReLU, lossy layers. |

**When to consult:** image provenance / MGI / data circuits → MGI paper; text watermarking / provenance / distillation detection → TextSeal; gradient leakage / FL privacy / weight manipulation attacks → trap-weights paper.

Claude Project setup: [docs/claude-project-instructions.md](docs/claude-project-instructions.md) — copy-paste system instructions + knowledge upload list.
