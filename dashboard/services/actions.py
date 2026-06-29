"""Build next-action suggestions for the team."""

from dashboard.config import DashboardConfig
from dashboard.models import FailedJob, GpuSchedule, NextAction, SlurmJob, TaskStatus


def build_next_actions(
    tasks: list[TaskStatus],
    jobs: list[SlurmJob],
    schedule: GpuSchedule,
    failed_jobs: list[FailedJob],
    config: DashboardConfig,
) -> list[NextAction]:
    actions: list[NextAction] = []

    for task in tasks:
        owner = task.owner or "team"
        if task.submit_ready:
            actions.append(
                NextAction(
                    kind="submit_ready",
                    priority=10,
                    message=f"{owner}: submit {task.task_id} (gate open)",
                    owner=task.owner,
                    task_id=task.task_id,
                )
            )
        elif 0 < task.submit_cooldown_seconds <= config.cooldown_soon_seconds:
            actions.append(
                NextAction(
                    kind="cooldown_soon",
                    priority=30,
                    message=f"{owner}: submit {task.task_id} in {task.submit_cooldown_seconds}s",
                    owner=task.owner,
                    task_id=task.task_id,
                )
            )

        if task.query_ready:
            actions.append(
                NextAction(
                    kind="query_ready",
                    priority=15,
                    message=f"{owner}: run analyze / logits on {task.task_id}",
                    owner=task.owner,
                    task_id=task.task_id,
                )
            )
        elif 0 < task.query_cooldown_seconds <= config.cooldown_soon_seconds:
            actions.append(
                NextAction(
                    kind="cooldown_soon",
                    priority=35,
                    message=f"{owner}: query {task.task_id} in {task.query_cooldown_seconds}s",
                    owner=task.owner,
                    task_id=task.task_id,
                )
            )

    for pack in schedule.pending_packs:
        if len(pack.job_ids) > 1:
            actions.append(
                NextAction(
                    kind="node_pack",
                    priority=50,
                    message=f"Pending jobs can share a node: {' + '.join(pack.job_ids)}",
                    job_id=pack.job_ids[0],
                )
            )

    for job in jobs:
        if job.state == "RUNNING" and job.progress and job.progress.heartbeat_stale:
            actions.append(
                NextAction(
                    kind="job_stalled",
                    priority=20,
                    message=f"Job {job.job_id} ({job.user}): no progress heartbeat >{config.progress_stale_seconds}s",
                    owner=job.user,
                    job_id=job.job_id,
                )
            )

    for failed in failed_jobs[:5]:
        actions.append(
            NextAction(
                kind="job_failed",
                priority=5,
                message=f"{failed.user}: job {failed.job_id} {failed.state} — check {failed.log_err}",
                owner=failed.user,
                job_id=failed.job_id,
            )
        )

    actions.sort(key=lambda a: (a.priority, a.message))
    return actions[:12]
