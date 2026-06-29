# CISPA Finals — Team Loki

European Championship in Trustworthy AI — Grand Finals preparation repo.

**Team:** Alexandre Ansart, Bastian Paoli, Florian Dougnon-Greder, Melissa Abider

## Quick start (cluster)

First-time setup: see [Setup](#setup-first-time) below.

```bash
ssh -i ~/.ssh/id_ed25519 -o Ciphers=aes256-ctr -o MACs=hmac-sha2-256-etm@openssh.com <user>@jureca.fz-juelich.de
jutil env activate -p training2557
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
├── scripts/build_dashboard.sh
├── slurm/templates/           # 1/2/4 GPU job templates
├── shared/
│   ├── submit.py              # API submission
│   ├── analyze.py             # Result analysis
│   ├── dashboard.py           # Terminal TUI (fallback)
│   ├── team_state.py          # Shared state JSON
│   ├── wandb_utils.py         # W&B helper
│   └── requirements.txt
└── task_N_<name>/attempt1/     # Created when subject drops
```

## Common commands

Activate the venv first (`source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`).

```bash
# Submit to leaderboard
python shared/submit.py output/submission.npz --task-id <TASK_ID> --action submit

# Analyze via API
python shared/analyze.py output/submission.npz --task-id <TASK_ID> --mode api --dataset <dataset.pt>

# SLURM
sbatch slurm/templates/2gpu_devel.sh
squeue -A training2557
```

### Team dashboard (browser)

Edit `dashboard/config.py` → set `MODE = "mock"` (local) or `MODE = "live"` (cluster).

```bash
# Local dev (mock data)
bash scripts/build_dashboard.sh          # once: builds client/
uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
# → http://127.0.0.1:8080

# Dev with hot reload (two terminals)
# T1: uvicorn dashboard.server:app --reload --port 8080
# T2: cd client && npm run dev   → http://localhost:5173

# On cluster (live data) — set MODE = "live" in config.py, run in tmux
uvicorn dashboard.server:app --host 127.0.0.1 --port 8080

# On laptop — SSH tunnel to see cluster dashboard in browser
ssh -L 8080:localhost:8080 <user>@jureca.fz-juelich.de
# → http://localhost:8080
```

Windows shortcut: `.\scripts\run_dashboard.ps1` — see [docs/dashboard-mock-test.md](docs/dashboard-mock-test.md).

Terminal fallback: `python shared/dashboard.py` (uses same `dashboard/config.py`).

## Setup (first time)

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
