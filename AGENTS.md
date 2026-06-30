# AGENTS.md

Guidance for Claude Code and Cursor on this repository. **Read the full file before writing code.**

> Workspace copy with parent paths: `../AGENTS.md` (includes `CISPA_Regional/` references).

---

## Behavioral guidelines

See full guidelines in this file — think before coding, simplicity first, surgical changes, goal-driven execution.

**Repository verification:** After Python changes, run `python -m py_compile` on touched modules; smoke-test on cluster when relevant.

---

## Project overview

**CISPA European Championship in Trustworthy AI — Grand Finals**

- **Team:** Loki (Alexandre Ansart, Bastian Paoli, Florian Dougnon-Greder, Melissa Abider)
- **Format:** 24-hour hackathon
- **Compute:** JURECA (`jureca.fz-juelich.de`), account `training2557`

---

## Documentation — read before you act

**Primary rule:** Before writing code or giving cluster/API advice, read the relevant docs below. Do not guess when the answer is documented.

### Read order (new session)

1. **`docs/subject/subject.md`** — task specs, API URLs, metrics
2. **`docs/notes-communes.md`** — assignments, GPU budget, active jobs
3. **This file** — conventions and shared utilities
4. **`task_N_<name>/`** or **`../CISPA_Regional/`** (if available) — prior solutions

### Documentation map (paths relative to repo root)

| File | When to read |
|------|----------------|
| [docs/subject/subject.md](docs/subject/subject.md) | Task rules, metrics, API, datasets |
| [docs/notes-communes.md](docs/notes-communes.md) | Live team state, SLURM jobs, decisions |
| [docs/hackathon-start-guide.md](docs/hackathon-start-guide.md) | Day-0 checklist, W&B, smoke tests |
| [docs/dashboard-mock-test.md](docs/dashboard-mock-test.md) | Test browser dashboard locally (Windows PowerShell) |
| [docs/dashboard-roadmap.md](docs/dashboard-roadmap.md) | ETA/progress protocol, ops features, implementation phases |
| [docs/cluster-guide.md](docs/cluster-guide.md) | SSH, venv, SLURM, tmux, ACLs, multi-GPU |
| [README.md](README.md) | Repo map, quick commands |
| [docs/recherche_preparation_hackathon.md](docs/recherche_preparation_hackathon.md) | PGD / adversarial robustness theory |
| [docs/mgi-member-vs-generated-inference.md](docs/mgi-member-vs-generated-inference.md) | **Finals required reading** — MGI, DCB, image provenance |
| [docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md](docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md) | **Finals required reading** — LLM watermark, localization, radioactivity |
| [docs/when_the_curious_abandon_honesty_federated_learning_is_not_private.md](docs/when_the_curious_abandon_honesty_federated_learning_is_not_private.md) | **Finals required reading** — FL gradient leakage, trap weights |
| [.env.example](.env.example) | `CISPA_BASE_URL`, `CISPA_API_KEY`, W&B |
| [slurm/templates/](slurm/templates/) | SLURM headers (1/2/4 GPU) |
| [slurm/submitted.log](slurm/submitted.log) | Job submission history |

**Regional reference** (parent workspace, not in this git repo): `../CISPA_Regional/` — Task 1 Lock & Ram, Task 2 cascade, Task 3 forensics.

### Cursor skills — read before acting

| Skill | Path |
|-------|------|
| jureca-slurm | [.cursor/skills/jureca-slurm/SKILL.md](.cursor/skills/jureca-slurm/SKILL.md) |
| hackathon-api | [.cursor/skills/hackathon-api/SKILL.md](.cursor/skills/hackathon-api/SKILL.md) |
| job-progress | [.cursor/skills/job-progress/SKILL.md](.cursor/skills/job-progress/SKILL.md) |

### Job progress protocol (dashboard ETA)

Any Slurm job expected to run **>10 minutes** must:

1. Call `bind_job(task_id, attempt=..., owner=...)` at script start.
2. Call `report(step, total_steps, phase, ...)` every epoch or every 5 minutes.
3. Call `complete()` or `fail()` before exit.
4. Use `#SBATCH --job-name=task{N}_att{M}_{user}`.

Module: [shared/job_progress.py](shared/job_progress.py). Skill: `job-progress`.

### Shared code — prefer over reimplementing

| File | Purpose |
|------|---------|
| [shared/submit.py](shared/submit.py) | API submit + logits |
| [shared/analyze.py](shared/analyze.py) | Local/API analysis |
| [shared/team_state.py](shared/team_state.py) | Shared cooldown/score JSON |
| [shared/job_progress.py](shared/job_progress.py) | Slurm job progress → dashboard |
| [shared/dashboard.py](shared/dashboard.py) | Terminal TUI fallback |
| [dashboard/server.py](dashboard/server.py) | Browser dashboard API (`MODE` in `dashboard/config.py`) |
| [client/](client/) | React browser UI |
| [shared/wandb_utils.py](shared/wandb_utils.py) | W&B init |
| [shared/tune_thresholds.py](shared/tune_thresholds.py) | Optuna template |
| [shared/smoke_test_gpu.py](shared/smoke_test_gpu.py) | GPU smoke test |
| [shared/requirements.txt](shared/requirements.txt) | Dependencies |

### Decision guide

| User asks about… | Read first |
|------------------|------------|
| Task rules, metric, format | `docs/subject/subject.md` |
| Team coordination, GPUs | `docs/notes-communes.md` |
| Cluster setup | `docs/cluster-guide.md` |
| Pre-hackathon checklist | `docs/hackathon-start-guide.md` |
| Dashboard mock test (Windows) | `docs/dashboard-mock-test.md` |
| SLURM jobs | `docs/cluster-guide.md` + skills `jureca-slurm`, `job-progress` |
| Submit / cooldowns | skill `hackathon-api` + `shared/submit.py` |
| Job progress / ETA | skill `job-progress` + `shared/job_progress.py` |
| Adversarial theory | `docs/recherche_preparation_hackathon.md` |
| Image member vs generated / data circuits / autoencoder signals | `docs/mgi-member-vs-generated-inference.md` |
| Text watermarking / provenance / distillation detection | `docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md` |
| Federated learning privacy / gradient reconstruction | `docs/when_the_curious_abandon_honesty_federated_learning_is_not_private.md` |

When the subject is not released, use `../CISPA_Regional/` only as **pattern reference**.

### Finals required reading (organizer-assigned)

Three papers were assigned before the Grand Finals. Use them for **inspiration and threat-model context** — always defer to `docs/subject/subject.md` for actual task specs.

1. **MGI (Member vs Generated Inference)** — [`docs/mgi-member-vs-generated-inference.md`](docs/mgi-member-vs-generated-inference.md)
   - Task: given image + generative model, classify sample as training member vs model output (harder than MIA).
   - Key failure mode: CPD / likelihood scores high for both members and generated samples.
   - **DCB pipeline:** Stage 1 = autoencoder score \(L_A\) (double reconstruction ratio + VQ quantization error) separates generated from natural; Stage 2 = standard MIA (ICAS) on non-generated; Stage 3 = cross-generator \(\phi(x,c)\) KDE for derivative models (\(M_2\) trained on \(M_1\) outputs).
   - Models evaluated: VAR, RAR, LlamaGen, Stable Diffusion 1.4/2.1. Metric: TPR@1%FPR.
   - Hackathon hooks: multi-signal cascades, threshold tuning (`tune_thresholds.py`), white-box model access (encoder + generator).

2. **TextSeal** — [`docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md`](docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md)
   - Distortion-free Gumbel-max watermark with **dual-key routing** (\(\alpha\)) for output diversity.
   - Detection: entropy-weighted scores, moment-matched Gamma \(p\)-values, **geometric cover search** for localized regions in diluted documents.
   - **Radioactivity:** watermark bias transfers through distillation; detect via teacher-forcing + PRF scoring.
   - Hackathon hooks: statistical detection tests, FPR control, segment-level attribution in mixed content.

3. **Trap weights / FL is not private** — [`docs/when_the_curious_abandon_honesty_federated_learning_is_not_private.md`](docs/when_the_curious_abandon_honesty_federated_learning_is_not_private.md)
   - Passive: FC-layer gradients contain scaled inputs; ReLU zeros leak individual batch items.
   - Active: server sends **trap weights** (adversarial FC init) → perfect extraction by projecting gradients to input space — no iterative optimization.
   - Scales to ImageNet \(B=100\), FedAvg, CNNs (with extensions).
   - Defenses: DP (users must add noise locally), leaky ReLU, compression/dropout, TEE.
   - Hackathon hooks: gradient-based attacks/defenses, weight inspection, batch-size effects.

### Tools & platforms

| Tool | When to use |
|------|-------------|
| **SLURM + JURECA** | Any GPU-heavy job (attacks, training, data gen) |
| **tmux** | All cluster work (24h SSH) |
| **submit.py / analyze.py** | Leaderboard submissions and API feedback |
| **dashboard** (web + TUI) | Live ops: queue, cooldowns, next actions, progress, failures |
| **W&B** (`wandb_utils.py`) | Model **training** only — loss curves, compare runs in browser |
| **torchattacks** | Adversarial attack prototyping (PGD, FGSM) |
| **timm** | Pretrained image models |
| **Optuna** (`tune_thresholds.py`) | Threshold search when metric penalizes FPR |
| **uv** | Virtualenv on cluster |

**Regional GPU usage:** Task 1 = 1–2 GPU attack jobs (~30 min); Task 2 = **4 GPU training** + synthetic data; Task 3 = GPU training. W&B was not used but would have helped Tasks 2/3.

**Browser dashboard:** edit `dashboard/config.py` (`MODE = "mock"` or `"live"`), then `uvicorn dashboard.server:app --host 127.0.0.1 --port 8080`. Access via SSH tunnel.

```bash
uvicorn dashboard.server:app --host 127.0.0.1 --port 8080
ssh -L 8080:localhost:8080 <user>@jureca.fz-juelich.de
```

---

### Current tasks

> Update when subject is released — also edit `docs/subject/subject.md`.

| Task | Owner | Directory | Metric |
|------|-------|-----------|--------|
| TBD  | TBD   | `task_N_<name>/attempt1/` | TBD |

### Cluster, SLURM, API

See [docs/cluster-guide.md](docs/cluster-guide.md) and [README.md](README.md).

```bash
python shared/submit.py output/submission.npz --task-id <TASK_ID> --action submit
python shared/analyze.py output/submission.npz --mode api --task-id <TASK_ID> --dataset <path>
python shared/dashboard.py
```

Templates: `slurm/templates/`. Coordinate GPUs in `docs/notes-communes.md`.
