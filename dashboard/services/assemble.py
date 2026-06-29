"""Assemble full dashboard status for live and mock providers."""

from __future__ import annotations

import os
from datetime import datetime

from dashboard.config import DashboardConfig
from dashboard.models import DashboardStatus, GpuSummary, SlurmJob
from dashboard.services.actions import build_next_actions
from dashboard.services.commands import (
    attach_failed_commands,
    attach_job_commands,
    attach_task_commands,
)
from dashboard.services.cooldowns import fetch_task_statuses
from dashboard.services.job_merge import merge_job_progress
from dashboard.services.leaderboard import fetch_leaderboard
from dashboard.services.owners import build_owner_summaries
from dashboard.services.queue_plan import enrich_jobs, normalize_state
from dashboard.services.sacct import fetch_failed_jobs
from dashboard.services.sinfo import fetch_cluster_status


def assemble_status(
    config: DashboardConfig,
    *,
    mode: str,
    raw_jobs: list[SlurmJob],
    gpu_summary: GpuSummary,
    jobs_state: dict,
    tasks_override: list[TaskStatus] | None = None,
    failed_jobs_data: list | None = None,
    cluster_data=None,
    leaderboard_data: list | None = None,
    extra_warnings: list[str] | None = None,
) -> DashboardStatus:
    warnings = list(extra_warnings or [])
    now = datetime.now()

    jobs, gpu_schedule = enrich_jobs(raw_jobs, config.gpus_per_node, now)
    jobs = merge_job_progress(
        jobs,
        jobs_state,
        stale_seconds=config.progress_stale_seconds,
        now=now,
    )
    jobs = attach_job_commands(jobs)

    if tasks_override is not None:
        tasks = attach_task_commands(tasks_override)
    else:
        tasks = attach_task_commands(
            fetch_task_statuses(config.submit_cooldown, config.query_cooldown)
        )

    team_pending = sum(1 for j in jobs if normalize_state(j.state) == "PENDING")

    if failed_jobs_data is None:
        failed, w = fetch_failed_jobs(config.slurm_account, config.sacct_hours)
        warnings.extend(w)
    else:
        failed = failed_jobs_data

    failed = attach_failed_commands(failed)

    if cluster_data is None and mode == "live":
        cluster, w = fetch_cluster_status(
            config.slurm_partition, config.gpus_per_node, team_pending
        )
        warnings.extend(w)
    else:
        cluster = cluster_data

    if leaderboard_data is None and mode == "live" and config.leaderboard_url:
        api_key = os.environ.get("CISPA_API_KEY")
        leaderboard, w = fetch_leaderboard(
            config.leaderboard_url,
            config.leaderboard_task_ids,
            api_key=api_key,
        )
        warnings.extend(w)
    else:
        leaderboard = leaderboard_data or []

    next_actions = build_next_actions(tasks, jobs, gpu_schedule, failed, config)
    owners = build_owner_summaries(tasks, jobs)

    return DashboardStatus(
        mode=mode,
        timestamp=now.isoformat(timespec="seconds"),
        refresh_seconds=config.refresh_seconds,
        account=config.slurm_account,
        gpu_summary=gpu_summary,
        gpu_schedule=gpu_schedule,
        slurm_jobs=jobs,
        tasks=tasks,
        next_actions=next_actions,
        failed_jobs=failed,
        owners=owners,
        cluster=cluster,
        leaderboard=leaderboard,
        warnings=warnings,
    )
