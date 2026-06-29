# Dashboard — mock mode testing guide

Test the browser dashboard locally **without JURECA**, using fake data from `dashboard/fixtures/mock_status.json`.

## Prerequisites

| Tool | Check | Install |
|------|-------|---------|
| Python 3.10+ | `python --version` | [python.org](https://www.python.org/downloads/) |
| Node.js 18+ | `node --version` | [nodejs.org](https://nodejs.org/) |
| npm | `npm --version` | Included with Node.js |

No GPU, no SSH, no `squeue` required for mock mode.

**Python environment:** mock mode only needs `dashboard/requirements.txt` (FastAPI, no torch). For the full hackathon stack (torch, timm, wandb, …), create and activate a venv first — see [README.md](../README.md#quick-start-local).

## Configuration

Edit [`dashboard/config.py`](../dashboard/config.py):

```python
MODE = "mock"   # must be "mock" for local testing
PORT = 8080
```

All settings are hardcoded in that file (no environment variables).

---

## Windows (PowerShell)

Run from the repo root `cispa_final/`:

```powershell
# One-shot: install deps, build client, start server
.\scripts\run_dashboard.ps1
```

Or step by step:

```powershell
# 1. Dashboard Python deps only (fast — no torch)
python -m pip install -r dashboard/requirements.txt

# 2. Build React client (first time, or after UI changes)
.\scripts\build_dashboard.ps1

# 3. Start server
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
```

Open **http://127.0.0.1:8080** in your browser.

Verify API:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/health
Invoke-RestMethod http://127.0.0.1:8080/api/status
```

Expected: yellow **mock** banner, 2 SLURM jobs, 3 tasks with cooldowns.

### Common Windows errors

| Error | Fix |
|-------|-----|
| `export` not recognized | Don't use bash `export` — edit `dashboard/config.py` instead |
| `uvicorn` not recognized | Use `python -m uvicorn ...` |
| `bash` not found | Use `.\scripts\build_dashboard.ps1` instead of `.sh` |
| Port 8080 in use | Change `PORT` in `config.py` and use that port in the URL |

---

## Linux / macOS / JURECA login node

```bash
cd cispa_final
python -m pip install -r dashboard/requirements.txt
bash scripts/build_dashboard.sh
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
```

---

## Dev mode (hot reload)

Two terminals — UI changes reload without rebuilding:

```powershell
# Terminal 1 — API
python -m pip install -r dashboard/requirements.txt
python -m uvicorn dashboard.server:app --reload --host 127.0.0.1 --port 8080

# Terminal 2 — React dev server (proxies /api to :8080)
cd client
npm install
npm run dev
```

Open **http://localhost:5173** (not 8080).

---

## Terminal fallback (TUI)

Same config file, no browser:

```powershell
python -m pip install rich
python shared/dashboard.py
```

---

## Switching to live mode (hackathon day)

On JURECA, edit `dashboard/config.py`:

```python
MODE = "live"
```

Then start the server in tmux and tunnel from your laptop:

```bash
# On cluster
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
```

```powershell
# On Windows laptop
ssh -L 8080:localhost:8080 <user>@jureca.fz-juelich.de
# Open http://localhost:8080
```

See [cluster-guide.md](cluster-guide.md) for full cluster setup.

---

## What you should see in mock mode

- **Header:** yellow mock banner, mode badge `mock`
- **GPU summary:** 6 GPUs, 2 jobs
- **SLURM table:** jobs `123456`, `123457` with progress bars
- **Cooldowns:** task_1 query blocked ~8 min, task_2 submit blocked
- **Scores:** task_1 = 0.218, task_2 W&B link, etc.
- Cooldowns and job elapsed times tick down on each refresh (~5 s)
