# Shared Notes — CISPA Finals

Live scratchpad for the team. Update this during the hackathon.

## Subject / Tasks

<!-- Fill from docs/subject/subject.md when released -->

## Team assignments

| Person | judoor user | Task | Directory |
|--------|-------------|------|-----------|
| Alexandre | ansart1 | TBD | `task_*/attempt1/` |
| Bastian | TBD | TBD | `task_*/attempt1/` |
| Florian | dougnon1 | TBD | `task_*/attempt1/` |
| Melissa | TBD | TBD | `task_*/attempt1/` |

## GPU budget

Coordinate before `sbatch`. Total team quota is shared under `training2557`.

| Person | GPUs requested | Job name | Status |
|--------|----------------|----------|--------|
| | | | |

## Active SLURM jobs

Update before each `sbatch`. Also append to `slurm/submitted.log`.

| User | Job ID | Task | Script | GPUs | Started | ETA |
|------|--------|------|--------|------|---------|-----|
| | | | | | | |

Check live: `squeue -A training2557`

## API / Leaderboard

- Base URL: TBD
- Leaderboard: TBD
- API keys: in cluster `.env` only (never commit)

Cooldowns (regional reference — verify at finals):
- Submit: ~5 min
- Query/logits: ~15 min

Run `python shared/dashboard.py` or the browser dashboard for live cooldown status.

Dashboard also shows: **next actions**, **job progress/ETA** (`job_progress.py`), **failed jobs**, **cluster GPUs**, **leaderboard** (when configured).

## Findings & decisions

<!-- Running log — newest at top -->

| Time | Person | Decision |
|------|--------|----------|
| | | |
