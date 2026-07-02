# JSC Cluster Guide — CISPA Grand Finals

Quick reference for Team Loki on **JUDAC** (data) and **JURECA** (GPU compute).

## Systems (training2625)

| System | Host | Purpose | Status |
|--------|------|---------|--------|
| **JUDAC** | `judac.fz-juelich.de` | Login node, global filesystem, data access only — **no GPU** | Granted (2026-07-02) |
| **JURECA** | `jureca.fz-juelich.de` | GPU compute via SLURM (A100) | Pending separate grant |

On JuDoor, each system has its own entry under **Systems** — sign the User Agreement and upload your SSH key **per system**.

## SSH connection

**JUDAC** (available now):

```bash
ssh -i ~/.ssh/id_ed25519 \
  -o Ciphers=aes256-ctr \
  -o MACs=hmac-sha2-256-etm@openssh.com \
  ansart1@judac.fz-juelich.de
```

**JURECA** (when granted — GPU jobs):

```bash
ssh -i ~/.ssh/id_ed25519 \
  -o Ciphers=aes256-ctr \
  -o MACs=hmac-sha2-256-etm@openssh.com \
  ansart1@jureca.fz-juelich.de
```

Replace `ansart1` with your judoor username. MFA (TOTP) is required on both.

**Windows (PowerShell):** use `$env:USERPROFILE\.ssh\id_ed25519` instead of `~/.ssh/id_ed25519`.

### Agent SSH on Windows (ControlMaster via WSL)

Native Windows OpenSSH prompts TOTP on every connection. Cursor agents use **WSL Ubuntu 26.04** + SSH multiplexing instead.

**Phase 1 — connect (TOTP once):**

```powershell
wsl -d Ubuntu-26.04 -- bash -lc "ssh -O check jureca"
cd cispa_final
$env:TOTP_CODE="<6-digit code>"; .\scripts\jureca-connect.ps1
```

**Phase 2 — run commands (no TOTP, master stays alive on failure):**

```powershell
wsl -d Ubuntu-26.04 -- bash -lc "ssh -o ControlMaster=no jureca 'squeue -u ansart1'"
wsl -d Ubuntu-26.04 -- bash -lc "bash ~/.local/bin/jureca-run.sh 'hostname'"
wsl -d Ubuntu-26.04 -- bash -lc "scp -o ControlMaster=no jureca:/remote/path ./local/"
```

Full agent rules (quoting, when to re-TOTP): root [`AGENTS.md`](../../AGENTS.md#agent-ssh-policy-important).

## Project activation

```bash
jutil env activate -p training2625
```

## Module loads

Required for PyTorch jobs:

```bash
module load GCC
module load CUDA
module load PyTorch
module load torchvision
```

Add these to every SLURM script (see `slurm/templates/`).

## Python environment

We use `uv` for virtual environments:

```bash
# One-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
exec bash

# Per-repo setup
cd /p/home/jusers/<user>/judac/code/cispa_final
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -r shared/requirements.txt
```

**Every new shell** — load modules, then activate:

```bash
module load GCC CUDA PyTorch torchvision
source .venv/bin/activate
```

Deactivate: `deactivate`

For a local laptop venv (Windows / macOS, no `uv`), see [README.md](../README.md#quick-start-local).

Optional uv cache config (add to `~/.bashrc` on cluster):

```bash
export UV_PYTHON_INSTALL_DIR=$PROJECT/user_dirs/$USER/uv/python
export UV_CACHE_DIR=$PROJECT/user_dirs/$USER/uv/cache
export UV_TOOL_DIR=$PROJECT/user_dirs/$USER/uv/tools
```

## Interactive GPU session

For debugging without SLURM:

```bash
salloc -p dc-gpu-devel -t 20 -N 1 -A training2625
srun --pty bash -i
module load GCC CUDA PyTorch torchvision
source .venv/bin/activate
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## SLURM job submission

```bash
# Copy and edit a template
cp slurm/templates/2gpu_devel.sh task_1/attempt1/run.sh
# Edit SCRIPT and ARGS at the bottom, then:
sbatch task_1/attempt1/run.sh

# Monitor
squeue -A training2625          # all team jobs
squeue -u ansart1               # your jobs
scancel <job_id>                # cancel a job
```

Log each submission in `slurm/submitted.log` and update `docs/notes-communes.md`.

## GPU allocation model

- Each compute node has **4× NVIDIA A100 40GB**
- Request GPUs per job: `--gres=gpu:1`, `gpu:2`, or `gpu:4`
- Jobs are independent — coordinate who takes how many GPUs before submitting
- If you finish early, `scancel` your job to free quota for teammates

### Dashboard ETA (live mode)

| Job state | What the dashboard shows | Is it real? |
|-----------|--------------------------|-------------|
| **RUNNING** + `job_progress` | **Reported** or **Extrapolated** from `step/total_steps` | **Yes** — if scripts call `shared/job_progress.py` |
| **RUNNING** (no progress) | `TIME_LIMIT − elapsed` ("Limit left") | **No** — worst-case until Slurm kills the job |
| **RUNNING** (stale heartbeat) | "Stalled?" | Progress not updated in >120s |
| **PENDING** | `START_TIME − now` ("Est. start") if set | **Approximate** |

**Job progress protocol:** every GPU job >10 min must call `bind_job()` + `report()` — see [dashboard-roadmap.md](dashboard-roadmap.md) and skill `job-progress`.

Queue order: **running jobs first**, then **pending by Slurm priority** (position `#1`, `#2`, …).

**Ops panels:** next actions, failed jobs (`sacct`), per-teammate summary, cluster GPU idle (`sinfo`), leaderboard (when `LEADERBOARD_URL` set in `dashboard/config.py`), copy-paste command chips.

### Multi-GPU within one job

SLURM gives you N GPUs; your code must use them:

```python
# Training: DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Independent work: manual split across GPUs
torch.cuda.set_device(gpu_id)
```

### Known JURECA quirk

GPU devices may not be cgroup-constrained by Slurm. Use `CUDA_VISIBLE_DEVICES` explicitly if needed:

```bash
env CUDA_VISIBLE_DEVICES="0" srun --overlap python script.py &
env CUDA_VISIBLE_DEVICES="1" srun --overlap python script.py &
wait
```

## Data paths

Team scratch (after `hackathon_setup.sh`):

```
/p/scratch/training2625/<owner>/loki/
```

Shared project data (if available):

```
/p/project1/training2625/common/
```

Team state file (API cooldowns, scores):

```
/p/project1/training2625/common/team_state.json
```

## Team folder sharing (ACLs)

Grant write access to all teammates on a shared folder:

```bash
# Save as set_acls.sh, then:
chmod +x set_acls.sh
./set_acls.sh ansart1,dougnon1,paoli1,abider1 /path/to/shared/folder
```

Script template (from organizer docs):

```bash
#!/bin/bash
set -euo pipefail
USERS="$1"    # comma-separated judoor usernames
FOLDER="$2"
IFS=',' read -ra USER_ARRAY <<< "$USERS"
setfacl -R -b "$FOLDER"
chmod -R u=rwX,g=rX,o=rX "$FOLDER"
for user in "${USER_ARRAY[@]}"; do
  setfacl -R -m "u:${user}:rwX" "$FOLDER"
  setfacl -R -m "d:u:${user}:rwX" "$FOLDER"
done
```

## tmux (required for 24h SSH)

Always work inside tmux so disconnects don't kill your session:

```bash
tmux new -s hackathon        # start
tmux attach -t hackathon     # reconnect
# Ctrl+B then D to detach without killing
```

Suggested layout: one pane for coding, one for the browser dashboard server.

## Team dashboard

See **[dashboard-mock-test.md](dashboard-mock-test.md)** for full mock testing (Windows + Linux).

### Browser UI (recommended)

Edit `dashboard/config.py` — set `MODE = "mock"` (local) or `MODE = "live"` (cluster).

**Windows (PowerShell):**

```powershell
.\scripts\run_dashboard.ps1
# Open http://127.0.0.1:8080
```

**Linux / cluster:**

```bash
python -m pip install -r dashboard/requirements.txt
bash scripts/build_dashboard.sh
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
```

From your laptop (live mode on cluster):

```powershell
# PowerShell
ssh -L 8080:localhost:8080 <user>@jureca.fz-juelich.de
# Open http://localhost:8080
```

| Mode | Setting in `dashboard/config.py` | Data source |
|------|----------------------------------|-------------|
| Mock | `MODE = "mock"` | `dashboard/fixtures/mock_status.json` |
| Live | `MODE = "live"` | `squeue` + `team_state.json` |

### Terminal fallback

```bash
python shared/dashboard.py
```

Shows SLURM queue, API cooldowns, last scores. Uses `MODE` from `dashboard/config.py`.
