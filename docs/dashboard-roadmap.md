# Dashboard & Ops Roadmap — LOKI CISPA Finals

Complete plan: real ETA feasibility, team conventions (skills / AGENTS), and dashboard evolution.

**Status:** planning document — implement in phases before / during hackathon.

---

## 1. ETA: what is real today?

### 1.1 What Slurm gives (no code change)

| Source | Field | Meaning | Reliable as “finish time”? |
|--------|-------|---------|----------------------------|
| `squeue` RUNNING | `TIME_LIMIT − TimeUsed` | Worst case until Slurm kills the job | **No** — job may finish in 10 min with a 2 h limit |
| `squeue` PENDING | `START_TIME` | Scheduler estimate for start | **Sometimes** — often `N/A` when cluster is busy |
| `squeue` | Priority, reason | Queue position hints | N/A for duration |

**Conclusion:** Slurm alone cannot produce a “real” ETA based on training progress. The dashboard correctly labels `time_limit_remaining` vs `scheduled_start` today.

### 1.2 What we do *not* have unless jobs report it

- Epoch / step index
- Items processed vs total (attacks, images, batches)
- Phase (`load_data`, `train`, `attack`, `export`)
- Per-phase ETA

Regional scripts print `Epoch 12/50` to stdout (tqdm / print) but **nothing writes that to a shared file** the dashboard reads.

### 1.3 Three ways to get a better ETA (ranked)

| Approach | Accuracy | Effort | Hackathon fit |
|----------|----------|--------|---------------|
| **A. Job progress protocol** (recommended) | High if authors call one helper | Low per script (~5 lines) | **Best** |
| **B. Log tail + regex** on `logs/slurm_%j.out` | Medium, breaks on format change | Medium (dashboard only) | Fallback / bonus |
| **C. W&B API** | Good for training tasks | High (auth, polling, not all tasks use W&B) | Parallel tool, not dashboard core |

**Recommendation:** adopt **A** as team standard; optionally add **B** later as best-effort for jobs that forget to report.

### 1.4 How “real ETA” would be computed (protocol A)

Inside the job, each report includes:

```text
step, total_steps, phase, optional message
```

Dashboard (or helper) computes:

```text
if step > 0 and total_steps > 0:
    progress_pct = step / total_steps
    elapsed_job = squeue TimeUsed (or now - job_start from first heartbeat)
    eta_seconds = elapsed_job / progress_pct - elapsed_job   # linear extrapolation
```

Caveats (acceptable for hackathon):

- Linear extrapolation assumes steady work per step (epochs OK; early stopping skews it).
- Attack loops with variable cost per image: report **items done / total items**, not epochs.
- Stale heartbeat (>2 min): show “stalled?” and fall back to limit remaining.

---

## 2. Standardization: is it worth it?

**Yes.** The dashboard is only as good as the data the team emits. Five people, 24 h, no shared conventions → everyone logs differently → dashboard stays a pretty `squeue` viewer.

**Principle:** one small Python helper + one JSON file + skills/AGENTS so agents and humans do the same thing without thinking.

### 2.1 Core artifact: extend `team_state.json`

Today: `tasks` only (cooldowns, scores).

Add top-level `jobs` keyed by Slurm `job_id` (from `$SLURM_JOB_ID`):

```json
{
  "tasks": { "...": "unchanged" },
  "jobs": {
    "1234567": {
      "job_id": "1234567",
      "job_name": "task1_att3_ansart1",
      "owner": "ansart1",
      "task_id": "task_1",
      "attempt": 3,
      "phase": "attack",
      "step": 240,
      "total_steps": 1000,
      "unit": "images",
      "message": "PGD batch 24",
      "eta_seconds": 420,
      "updated_at": "2026-06-29T21:40:00"
    }
  }
}
```

- **Progress bar:** `step / total_steps`
- **ETA:** prefer job-reported `eta_seconds` if set; else dashboard extrapolates
- **Merge:** live provider joins `squeue` row with `jobs[job_id]`; stale entries removed when job leaves queue (or marked completed via `sacct`)

### 2.2 New shared module: `shared/job_progress.py`

Minimal API (atomic read-modify-write via existing `team_state` patterns):

```python
from job_progress import bind_job, report, complete, fail

bind_job(task_id="task_1", attempt=3, owner="ansart1")  # once at start; uses SLURM_JOB_ID
report(step=12, total_steps=50, phase="train", message="epoch 12", eta_seconds=1800)
complete(message="wrote submission.npz")
fail(message="CUDA OOM")
```

**Rules:**

- Call `bind_job()` at top of every `main.py` / long script under sbatch.
- Call `report()` at least every epoch or every N minutes on long loops.
- `complete()` / `fail()` in `atexit` or explicit before exit.

### 2.3 SLURM naming convention (enables everything else)

Standardize `#SBATCH --job-name`:

```text
task{N}_att{M}_{user}     e.g. task1_att3_ansart1
```

Benefits: dashboard parses task/attempt/owner from name; `notes-communes.md` matches `squeue`; agents generate consistent scripts.

### 2.4 Other conventions worth standardizing

| Convention | Where enforced | Enables |
|------------|----------------|---------|
| `--owner` on every `submit.py` / `analyze.py` | skill `hackathon-api` + AGENTS | Per-person cooldown view |
| `score_history[]` on submit (last 5 scores) | `submit.py` + skill | Delta ↑↓ in dashboard |
| Logs always `logs/slurm_%j.out` | `slurm/templates/` | Log-tail fallback, failure links |
| Append `slurm/submitted.log` on sbatch | skill `jureca-slurm` | Audit trail |
| W&B: `wandb_url` saved to `team_state` on init | `wandb_utils.py` | Link only (graphs stay on W&B) |
| Task dirs `task_N_<slug>/attemptM/` | AGENTS + subject doc | Copy-paste commands in UI |
| Exit handler on GPU jobs | skill `job-progress` | Failed jobs visible without `sacct` poll |

### 2.5 Skills & AGENTS updates (planned)

| Item | Action |
|------|--------|
| **New:** `.cursor/skills/job-progress/SKILL.md` | When writing training/attack loops, sbatch entrypoints |
| **Update:** `jureca-slurm` | job-name format, `bind_job` at start, log paths |
| **Update:** `hackathon-api` | `--owner`, `score_history`, link to cooldown dashboard |
| **New (optional):** `.cursor/skills/dashboard-ops/SKILL.md` | Running dashboard live, tunnel, interpreting ETA kinds |
| **Update:** `AGENTS.md` | Doc map row, “Job progress protocol” section, verification checklist |
| **Update:** `docs/cluster-guide.md` | ETA semantics + progress protocol pointer |

**AGENTS.md rule (proposed):**

> Any Slurm job expected to run **>10 minutes** must call `job_progress.bind_job()` at start and `report()` at least once per epoch or every 5 minutes.

---

## 3. Dashboard roadmap (phases)

### Phase 0 — Done

- [x] Browser UI (mock + live)
- [x] Queue order, priority, positions, GPU node packs
- [x] Cooldowns + scores from `team_state`
- [x] ETA kinds documented (limit vs scheduled start)

**Status:** Phases 0–3 implemented (mock + live). Phase 4 optional polish remains.

### Phase 1 — Ops coordinator — **done**

| # | Feature | Status |
|---|---------|--------|
| 1.1 | Next actions | done |
| 1.2 | Browser notifications | done (toggle in header) |
| 1.3 | Failed jobs (`sacct`) | done |
| 1.4 | Score delta | done |
| 1.5 | Per-owner row | done |

### Phase 2 — Real progress & ETA — **done**

| # | Feature | Status |
|---|---------|--------|
| 2.1 | `shared/job_progress.py` | done |
| 2.2 | Merge `jobs` + `squeue` | done |
| 2.3 | Progress bar + ETA kinds | done |
| 2.4 | Stale heartbeat | done |
| 2.5 | Mock fixture | done |

### Phase 3 — Cluster awareness — **done**

| # | Feature | Status |
|---|---------|--------|
| 3.1 | `sinfo` GPU summary | done |
| 3.2 | Leaderboard poll | done (config `LEADERBOARD_URL`) |
| 3.3 | Command chips | done |

### Phase 4 — Polish (only if time)

- Sound on cooldown open (opt-in)
- Discord webhook on failure / submit ready (config flag)
- TUI parity with web next-actions

---

## 4. Implementation order (recommended)

```text
Before hackathon (team)
├── Agree job-name convention + job-progress protocol (15 min standup)
├── Phase 1.3 + 1.4 backend (failed jobs, score_history)
├── Phase 2.1 job_progress.py + skills + AGENTS.md
└── Smoke: one template job calls bind_job/report; dashboard shows bar

Day 0 (first hour)
├── Phase 1.1 next actions live
├── Dashboard in tmux + SSH tunnel for remote
└── Everyone adds bind_job to new scripts (agents use skill)

During hackathon
├── Phase 1.2 notifications (if useful)
├── Phase 3.1 cluster busy indicator
└── Phase 3.2 leaderboard if endpoint ready

Do NOT during hackathon
├── UI redesign
├── W&B embedded charts
└── Dashboard-side submit buttons (write conflicts)
```

---

## 5. Verification checklist

| Check | Command / signal |
|-------|------------------|
| Progress written | `cat .../team_state.json` has `jobs.<id>.step` updating |
| Dashboard ETA | UI shows “Reported” or “Extrapolated”, not only “Limit left” |
| Failed job surfaced | Kill a test job; dashboard shows failure within 1 refresh |
| Next action | Cooldown expiry generates visible suggestion |
| Agents follow protocol | New `main.py` includes `bind_job` (grep / review) |

---

## 6. Open decisions (team)

1. **Stale job entries:** delete from `team_state.jobs` when job leaves `squeue`, or move to `jobs_completed`?
2. **Log parsing:** invest in regex fallback or rely 100% on `job_progress`?
3. **Leaderboard:** official URL in `docs/subject/subject.md` when released — add to dashboard or separate tab?
4. **Notifications:** browser-only OK, or webhook to Discord?

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Real ETA from Slurm alone? | **No** — only limit remaining and approximate start time. |
| Real ETA from progress? | **Yes**, if jobs call `job_progress.report()` (5 lines per loop). |
| Skill + AGENTS worth it? | **Yes** — cheap adoption, unlocks progress bars, better ETA, next actions, failure visibility. |
| Best next dashboard work? | **Phase 1** (coordinator) + **Phase 2** (progress protocol), not more UI. |

Related docs: [cluster-guide.md](cluster-guide.md) (ETA semantics), [hackathon-start-guide.md](hackathon-start-guide.md) (dashboard smoke test), [notes-communes.md](notes-communes.md) (live assignments).
