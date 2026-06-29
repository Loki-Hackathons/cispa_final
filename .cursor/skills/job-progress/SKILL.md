---
name: job-progress
description: Report Slurm job progress into team_state for the LOKI dashboard. Use when writing training loops, attack scripts, or sbatch entrypoints expected to run longer than 10 minutes.
---

# Job progress protocol

## When to use

- Writing `main.py` or any script launched via `sbatch`
- Long loops: epochs, PGD attacks, data generation
- Agent needs to expose progress bars / ETA on the team dashboard

## Rules

1. Call `bind_job()` once at script start (uses `$SLURM_JOB_ID`).
2. Call `report()` every epoch or at least every 5 minutes in long loops.
3. Call `complete()` on success or `fail()` on error before exit.
4. Use SLURM job name: `task{N}_att{M}_{user}` (e.g. `task1_att3_ansart1`).

Jobs under 10 minutes: optional but still recommended.

## API

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from job_progress import bind_job, report, complete, fail

bind_job("task_1", attempt=3, owner="ansart1", total_steps=50, unit="epochs")

for epoch in range(50):
    # ... work ...
    report(epoch + 1, 50, phase="train", message=f"epoch {epoch + 1}/50")

complete("saved checkpoint")
```

Attack / item loops: use **items** as steps, not epochs:

```python
bind_job("task_1", attempt=3, total_steps=1000, unit="images")
for i, img in enumerate(images, start=1):
    # ...
    if i % 50 == 0:
        report(i, len(images), phase="attack", message=f"image {i}")
```

## SLURM template snippet

Add after `cd` and `source .venv/bin/activate` in `run.sh`:

```python
python -u main.py "$@"
```

Inside `main.py` at the top:

```python
bind_job("task_N", attempt=1, owner=os.environ.get("USER"))
```

## Where data goes

`/p/project1/training2557/common/team_state.json` → `jobs.<SLURM_JOB_ID>`

Local dev: `cispa_final/team_state.json` (or `CISPA_TEAM_STATE` env).

## Dashboard behavior

| Field | UI |
|-------|-----|
| `step` / `total_steps` | Progress bar |
| `eta_seconds` | "Reported" ETA |
| missing heartbeat >120s | "Stalled?" warning + next action |
| extrapolation | "Extrapolated" ETA from elapsed + progress |

## See also

- [docs/dashboard-roadmap.md](../../docs/dashboard-roadmap.md)
- [docs/cluster-guide.md](../../docs/cluster-guide.md) — ETA semantics
- skill `jureca-slurm` — job naming and logs
