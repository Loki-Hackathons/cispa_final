# Shared Notes — CISPA Finals

Live scratchpad for the team. Update during the hackathon.

## Setup status (2026-07-02)

| Step | Owner | Status |
|------|-------|--------|
| JuDoor account | all | ansart1 ✅ MFA ✅ |
| Project **training2625** | ansart1 | ✅ approved |
| **JUDAC** system (data access) | ansart1 | ✅ granted — `judac.fz-juelich.de` |
| JUDAC: User Agreement + SSH key on JuDoor | ansart1 | ⏳ do on JuDoor → Systems → judac |
| SSH to judac | ansart1 | ⏳ after SSH key on JUDAC (~15 min) |
| **JURECA** system (GPU / SLURM) | all | ⏳ pending separate grant |
| Owner `hackathon_setup.sh` | ansart1 | ⏳ after SSH to judac |
| Teammate `teammate.sh` | others | ⏳ after owner |
| Local dashboard mock | ansart1 | ✅ `MODE=mock`, health OK |
| API key from organizers | team | ✅ Loki — cluster `.env` only |

**JUDAC** = data access + global filesystem only (no GPU). **JURECA** = GPU jobs via SLURM — expect a separate JuDoor email when granted.

Bootstrap guide: `scripts/cluster/README.md` · Cluster ref: `docs/cluster-guide.md`

## Subject / Tasks

See `docs/subject/subject.md`. Leaderboard: http://35.192.205.84/leaderboard_page

| Task | Doc | Scratch folder (after setup) |
|------|-----|------------------------------|
| 1 Text Watermark | [Task 1 Text Watermark Localization.md](Task%201%20Text%20Watermark%20Localization.md) | `.../loki/<watermark-dataset>/` |
| 2 MGI | [Task 2 Description.md](Task%202%20Description.md) | `.../loki/<mgi-dataset>/` |
| 3 FL Reconstruction | [Task 3 Description.md](Task%203%20Description.md) | `.../loki/<fl-dataset>/` |

## Team assignments

| Person | judoor user | Role | Task | Directory |
|--------|-------------|------|------|-----------|
| Alexandre | ansart1 | **Owner** | **Task 1 — Text Watermark** | `task_1_text_watermark/alexandre/` |
| Melissa | abider1? | Teammate | **Task 1 — Text Watermark** | `task_1_text_watermark/melissa/` |
| Florian | dougnon1 | **Owner** | **Task 2 — MGI** | `task_2_mgi/attempt1/` |
| Bastian | paoli1 | **Owner** | **Task 3 — FL Reconstruction** | `task_3_fl_reconstruction/attempt1/` |

Confirm judoor username for Melissa if different.

## GPU budget

Coordinate before `sbatch`. Account `training2625`, reservation `cispahack`. Requires **JURECA** access.

| Person | GPUs requested | Job name | Status |
|--------|----------------|----------|--------|
| | | | |

## Active SLURM jobs

Update before each `sbatch`. Also append to `slurm/submitted.log`.

| User | Job ID | Task | Script | GPUs | Started | ETA |
|------|--------|------|--------|------|---------|-----|
| | | | | | | |

Check live: `squeue -A training2625` (JURECA only)

## API / Leaderboard

URL, `--task-id`, `.env` fields: `docs/subject/subject.md` § API. Leaderboard UI: http://35.192.205.84/leaderboard_page

Run browser dashboard or `python shared/dashboard.py` for cooldown status.

## Findings & decisions

| Time | Person | Decision |
|------|--------|----------|
| 2026-07-02 | Alexandre | JUDAC granted (`judac.fz-juelich.de`); JURECA still pending for GPU |
| 2026-07-02 | Team | Task 1: Alexandre + Melissa · Task 2: Florian · Task 3: Bastian |
| 2026-07-02 | Alexandre | Corrected assignments: Bastian → Task 3 (not Task 1) |
| 2026-07-02 | Alexandre | ansart1 → Task 1 (Text Watermark Localization) |
| 2026-07-02 | Alexandre | Task 1 vendor submodules pinned (textseal, lm-watermarking, unigram) + `watermark_config.yaml` in repo |

## JSC contacts (training2625)

- Project advisor: Dr. Andreas Herten — a.herten@fz-juelich.de, +49 2461/61-1825
- User Services: user-services.jsc@fz-juelich.de, +49 2461/61-5642
- Supercomputer support: sc@fz-juelich.de, +49 2461-61-2828
