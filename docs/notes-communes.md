# Shared Notes — CISPA Finals

Live scratchpad for the team. Update during the hackathon.

## Setup status (2026-07-02)

| Step | Owner | Status |
|------|-------|--------|
| JuDoor account | all | ansart1 ✅ MFA ✅ |
| Join project **training2625** | all | ⏳ pending PI approval |
| JURECA system + SSH key on JuDoor | all | ⏳ blocked until project |
| SSH to jureca | ansart1 | ❌ publickey (no system yet) |
| Owner `hackathon_setup.sh` | ansart1 | ⏳ after SSH |
| Teammate `teammate.sh` | others | ⏳ after owner |
| Local dashboard mock | ansart1 | ✅ `MODE=mock`, health OK |
| API key from organizers | team | ⏳ contact if missing |

**Project ID finals:** `training2625` (see `docs/Hackathon_Setup Finale.md`). Do **not** use regional `training2557`.

Bootstrap guide: `scripts/cluster/README.md`

## Subject / Tasks

See `docs/subject/subject.md`. Leaderboard: http://35.192.205.84/leaderboard_page

| Task | Doc | Scratch folder (after setup) |
|------|-----|------------------------------|
| 1 Text Watermark | `docs/Task 1 Text Watermark Localization.md` | `.../loki/<watermark-dataset>/` |
| 2 MGI | `docs/Task 2 Description.md` | `.../loki/<mgi-dataset>/` |
| 3 FL Reconstruction | `docs/Task 3 Description.md` | `.../loki/<fl-dataset>/` |

## Team assignments

| Person | judoor user | Role | Task | Directory |
|--------|-------------|------|------|-----------|
| Alexandre | ansart1 | **Owner** (runs setup) | TBD | `task_*/attempt1/` |
| Bastian | paoli1? | Teammate | TBD | `task_*/attempt1/` |
| Florian | dougnon1 | Teammate | TBD | `task_*/attempt1/` |
| Melissa | abider1? | Teammate | TBD | `task_*/attempt1/` |

Confirm judoor usernames for Bastian/Melissa if different.

## GPU budget

Coordinate before `sbatch`. Account `training2625`, reservation `cispahack`.

| Person | GPUs requested | Job name | Status |
|--------|----------------|----------|--------|
| | | | |

## Active SLURM jobs

Update before each `sbatch`. Also append to `slurm/submitted.log`.

| User | Job ID | Task | Script | GPUs | Started | ETA |
|------|--------|------|--------|------|---------|-----|
| | | | | | | |

Check live: `squeue -A training2625`

## API / Leaderboard

- Base URL: TBD
- Leaderboard: http://35.192.205.84/leaderboard_page
- API keys: in cluster `.env` only (never commit)

Cooldowns (from task specs — verify at finals):
- Submit: ~5 min (Task 3: 2 min after error)
- Query/logits: ~15 min (regional reference)

Run browser dashboard or `python shared/dashboard.py` for cooldown status.

## Findings & decisions

| Time | Person | Decision |
|------|--------|----------|
| 2026-07-02 | Alexandre | Finals project = training2625; SSH blocked until JuDoor project + JURECA system assigned |
