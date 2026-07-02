# Hackathon Start Guide

Checklist to run before and at the start of the CISPA finals. Do Phase A now; Phase B when the subject drops.

## Phase A — Before hackathon (test now)

### 1. SSH and project

**JUDAC** (data access — available first):

```bash
ssh -i ~/.ssh/id_ed25519 -o Ciphers=aes256-ctr -o MACs=hmac-sha2-256-etm@openssh.com ansart1@judac.fz-juelich.de
jutil env activate -p training2625
```

**JURECA** (GPU — when separately granted): replace host with `jureca.fz-juelich.de`.

On JuDoor → **Systems** → upload SSH key per system (judac, then jureca).

- [ ] MFA works
- [ ] `echo $PROJECT` returns a path

### 2. Clone repo on cluster

```bash
cd /p/home/jusers/<user>/jureca/code
git clone https://github.com/Loki-Hackathons/cispa_final.git
cd cispa_final
git pull
```

### 3. Python environment

```bash
module load GCC CUDA PyTorch torchvision
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -r shared/requirements.txt
```

Verify imports:

```bash
python -c "import torch, torchattacks, timm, optuna, wandb, rich; print('OK')"
```

Reactivate the venv in every new shell on the cluster:

```bash
module load GCC CUDA PyTorch torchvision
source .venv/bin/activate
```



### 3b. Local environment (laptop — optional)

For mock dashboard and CPU-only scripting without JURECA. Full commands: [README.md](../README.md#quick-start-local).

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

- [ ] `python -c "import torch, timm, wandb; print('OK')"` succeeds
- [ ] Prompt shows `(.venv)` when activated
- [ ] `deactivate` exits the venv

> Dashboard mock test can use `dashboard/requirements.txt` only (lighter, no torch). Use `shared/requirements.txt` when you need the full hackathon stack locally.



### 4. Environment variables

```bash
cp .env.example .env
# Edit .env with real CISPA_BASE_URL and CISPA_API_KEY when available
source .env  # or: export $(grep -v '^#' .env | xargs)
```



### 5. W&B setup (can test before hackathon)

1. Create project `cispa-finals` on [wandb.ai](https://wandb.ai)
2. Invite all teammates
3. `wandb login` on cluster
4. Smoke test:

```bash
python -c "
from shared.wandb_utils import init
import wandb
run = init('smoke-test', {'ok': True})
wandb.log({'test': 1})
run.finish()
print('W&B OK')
"
```

- [ ] All teammates see the smoke-test run on the dashboard



### 6. GPU smoke test (SLURM)

```bash
mkdir -p logs
sbatch slurm/templates/1gpu_devel.sh
squeue -u $USER
# When done:
cat logs/slurm_*.out
```

Expected output: `CUDA available: True`, `Smoke test passed.`

### 7. Team folder ACLs

Share the repo folder with all judoor users (see `docs/cluster-guide.md`).

### 8. tmux

```bash
tmux new -s hackathon
```

Keep coding inside tmux. Detach: `Ctrl+B` then `D`. Reattach: `tmux attach -t hackathon`.

### 9. Dashboard smoke test (mock — test on your laptop now)

Full guide: **[dashboard-mock-test.md](dashboard-mock-test.md)**

1. Confirm `MODE = "mock"` in `dashboard/config.py`
2. Run:

**Windows (PowerShell):**

```powershell
.\scripts\run_dashboard.ps1
# Open http://127.0.0.1:8080
```

**Linux / macOS / cluster:**

```bash
python -m pip install -r dashboard/requirements.txt
bash scripts/build_dashboard.sh
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
```

1. Check API: `Invoke-RestMethod http://127.0.0.1:8080/api/health` (PowerShell) or `curl http://127.0.0.1:8080/api/health`

**On cluster (live)** — set `MODE = "live"` in `dashboard/config.py`, then:

```bash
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
# From laptop: ssh -L 8080:localhost:8080 <user>@jureca.fz-juelich.de
```

Run the server in a dedicated tmux pane during the hackathon. Fallback TUI: `python shared/dashboard.py`.

---



## Phase B — When subject is released (first 30 min)

1. **Subject** — copy official specs into `docs/subject/subject.md` (+ raw `.txt` files in `docs/subject/`)
2. **Notes** — update `docs/notes-communes.md`: task assignments, API URLs, leaderboard
3. **AGENTS.md** — fill the "Current tasks" table
4. **Task dirs** — create `task_N_<name>/attempt1/` with official `task_template.py`
5. **Datasets** — verify paths under `/p/project1/training2557/common/`
6. **GPU budget** — agree who requests 1/2/4 GPUs in `notes-communes.md`
7. **API test** — if endpoints are live:

```bash
python shared/submit.py <file> --task-id <TASK_ID> --action submit
python shared/analyze.py <file> --task-id <TASK_ID> --mode api --dataset <path>
```

1. **SLURM** — confirm `squeue -A training2557` shows expected jobs only

---



## Phase C — Sanity checklist


| Check                                                      | Done |
| ---------------------------------------------------------- | ---- |
| SSH + tmux works for all 4 teammates                       |      |
| Shared venv imports all packages                           |      |
| W&B dashboard shows smoke-test from everyone               |      |
| 1-GPU SLURM job completes in <5 min                        |      |
| `docs/subject/subject.md` filled in                        |      |
| `notes-communes.md` assignments + GPU table updated        |      |
| ACLs: everyone can write to shared folder                  |      |
| API keys in cluster `.env`, not in git                     |      |
| Browser dashboard: next actions, progress, failures (mock) |      |
| Live dashboard on cluster + SSH tunnel tested              |      |


---



## Tool quick reference


| Tool                  | Install                                  | When to use                                                                                          |
| --------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **W&B**               | in `requirements.txt`                    | Training tasks — loss curves, team visibility                                                        |
| **torchattacks**      | in `requirements.txt`                    | Adversarial attack prototyping (PGD, MI-FGSM)                                                        |
| **timm**              | in `requirements.txt`                    | Pretrained CV models (ConvNeXt, EfficientNet)                                                        |
| **Optuna**            | in `requirements.txt`                    | Threshold tuning (`shared/tune_thresholds.py`)                                                       |
| **Browser dashboard** | `python -m uvicorn dashboard.server:app` | Mock/live via `MODE` in `dashboard/config.py` — see [dashboard-mock-test.md](dashboard-mock-test.md) |
| **TUI dashboard**     | `python shared/dashboard.py`             | Terminal fallback                                                                                    |
| **tmux**              | system                                   | Survive SSH disconnects                                                                              |




## Where to put the subject


| What                | Where                                     |
| ------------------- | ----------------------------------------- |
| Task descriptions   | `docs/subject/subject.md`                 |
| Raw organizer files | `docs/subject/task_*.txt`                 |
| Live team notes     | `docs/notes-communes.md`                  |
| Agent context       | `AGENTS.md` → Current tasks section       |
| Starter code        | `task_N_<name>/attempt1/task_template.py` |


