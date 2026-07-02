"""SLURM queue polling via squeue."""

from __future__ import annotations

import subprocess

from dashboard.models import GpuSummary, SlurmJob
from dashboard.services.queue_plan import normalize_state
from dashboard.services.time_parse import parse_slurm_time


def fetch_slurm_jobs(account: str) -> tuple[list[SlurmJob], GpuSummary, list[str]]:
    warnings: list[str] = []
    jobs: list[SlurmJob] = []
    total_gpus = 0

    fmt = "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %b %.10Q %.20S %.30r"
    try:
        result = subprocess.run(
            [
                "squeue",
                "-A",
                account,
                f"--format={fmt}",
                "--noheader",
                "--sort=-t,-p,-S",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        warnings.append("squeue not found (not on cluster?)")
        return jobs, GpuSummary(used=0, team_jobs=0), warnings
    except subprocess.TimeoutExpired:
        warnings.append("squeue timed out")
        return jobs, GpuSummary(used=0, team_jobs=0), warnings

    if result.returncode != 0:
        warnings.append(f"squeue failed: {result.stderr.strip() or 'unknown error'}")
        return jobs, GpuSummary(used=0, team_jobs=0), warnings

    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 10)
        if len(parts) < 7:
            continue

        job_id, partition, name, user, state, elapsed_raw, limit_raw = parts[:7]
        gres = parts[7] if len(parts) > 7 else ""
        priority_raw = parts[8] if len(parts) > 8 else ""
        start_time = parts[9] if len(parts) > 9 else None
        reason = parts[10] if len(parts) > 10 else None

        gpus: int | None = None
        if "gpu:" in gres:
            try:
                gpus = int(gres.split("gpu:")[1].split(",")[0])
                if normalize_state(state) == "RUNNING":
                    total_gpus += gpus
            except ValueError:
                gpus = None

        priority: int | None = None
        if priority_raw and priority_raw not in ("N/A", "N/A(N/A)"):
            try:
                priority = int(float(priority_raw))
            except ValueError:
                priority = None

        elapsed = parse_slurm_time(elapsed_raw)
        time_limit = parse_slurm_time(limit_raw)

        jobs.append(
            SlurmJob(
                job_id=job_id,
                user=user,
                name=name[:30],
                gpus=gpus,
                state=state,
                elapsed_seconds=elapsed,
                time_limit_seconds=time_limit,
                partition=partition,
                priority=priority,
                start_time=start_time,
                reason=reason,
            )
        )

    return jobs, GpuSummary(used=total_gpus, team_jobs=len(jobs)), warnings
