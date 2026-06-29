"""Mock dashboard data with simulated cooldown decay and job progress."""

import copy
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from dashboard.config import DashboardConfig
from dashboard.models import ClusterStatus, DashboardStatus, FailedJob, GpuSummary, LeaderboardRow, SlurmJob, TaskStatus
from dashboard.providers.base import StatusProvider
from dashboard.services.assemble import assemble_status

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "mock_status.json"


def _score_delta(history: list[float]) -> float | None:
    if len(history) < 2:
        return None
    return round(history[-1] - history[-2], 6)


class MockProvider(StatusProvider):
    def __init__(self, config: DashboardConfig):
        self._config = config
        self._base = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self._last_fetch = time.monotonic()

    def get_status(self) -> DashboardStatus:
        delta = int(time.monotonic() - self._last_fetch)
        self._last_fetch = time.monotonic()

        data = copy.deepcopy(self._base)

        raw_jobs: list[SlurmJob] = []
        for job in data.get("slurm_jobs", []):
            if job.get("state") == "RUNNING" and job.get("elapsed_seconds") is not None:
                job["elapsed_seconds"] += delta
            raw_jobs.append(SlurmJob.model_validate(job))

        jobs_state = data.get("jobs_state", {})
        now = datetime.now()
        for info in jobs_state.values():
            if info.get("status") != "running":
                continue
            step = info.get("step", 0)
            total = info.get("total_steps", 0)
            if total and step < total:
                info["step"] = min(total, step + max(1, delta // 30))
            info["updated_at"] = (now - timedelta(seconds=5)).isoformat(timespec="seconds")
            if info.get("eta_seconds") is not None:
                info["eta_seconds"] = max(0, int(info["eta_seconds"]) - delta)

        tasks: list[TaskStatus] = []
        for task in data.get("tasks", []):
            submit = max(0, task.get("submit_cooldown_seconds", 0) - delta)
            query = max(0, task.get("query_cooldown_seconds", 0) - delta)
            history = [float(x) for x in (task.get("score_history") or [])]
            tasks.append(
                TaskStatus(
                    task_id=task["task_id"],
                    owner=task.get("owner"),
                    last_score=task.get("last_score"),
                    score_delta=_score_delta(history),
                    score_history=history,
                    attempt=task.get("attempt"),
                    submit_cooldown_seconds=submit,
                    query_cooldown_seconds=query,
                    submit_ready=submit <= 0,
                    query_ready=query <= 0,
                    updated_at=task.get("updated_at"),
                    wandb_url=task.get("wandb_url"),
                )
            )

        failed = [FailedJob.model_validate(f) for f in data.get("failed_jobs", [])]
        cluster = ClusterStatus.model_validate(data["cluster"]) if data.get("cluster") else None
        leaderboard = [LeaderboardRow.model_validate(r) for r in data.get("leaderboard", [])]

        return assemble_status(
            self._config,
            mode="mock",
            raw_jobs=raw_jobs,
            gpu_summary=GpuSummary.model_validate(data["gpu_summary"]),
            jobs_state=jobs_state,
            tasks_override=tasks,
            failed_jobs_data=failed,
            cluster_data=cluster,
            leaderboard_data=leaderboard,
            extra_warnings=data.get("warnings", []),
        )
