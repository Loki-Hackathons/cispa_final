"""Per-owner rollup for dashboard."""

from dashboard.models import OwnerSummary, SlurmJob, TaskStatus
from dashboard.services.queue_plan import normalize_state


def build_owner_summaries(
    tasks: list[TaskStatus],
    jobs: list[SlurmJob],
) -> list[OwnerSummary]:
    owners: dict[str, OwnerSummary] = {}

    def get(owner: str) -> OwnerSummary:
        if owner not in owners:
            owners[owner] = OwnerSummary(owner=owner)
        return owners[owner]

    for task in tasks:
        owner = task.owner or "unknown"
        row = get(owner)
        if task.task_id not in row.task_ids:
            row.task_ids.append(task.task_id)
        if task.submit_ready and task.task_id not in row.submit_ready_tasks:
            row.submit_ready_tasks.append(task.task_id)
        if task.query_ready and task.task_id not in row.query_ready_tasks:
            row.query_ready_tasks.append(task.task_id)

    for job in jobs:
        owner = job.user or "unknown"
        row = get(owner)
        if normalize_state(job.state) == "RUNNING":
            row.running_jobs += 1
        elif normalize_state(job.state) == "PENDING":
            row.pending_jobs += 1

    return sorted(owners.values(), key=lambda o: o.owner)
