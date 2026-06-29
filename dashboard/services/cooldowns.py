"""Task cooldown and score enrichment from team_state."""

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from team_state import cooldown_remaining, load_state  # noqa: E402

from dashboard.models import TaskStatus


def _score_delta(history: list[float]) -> float | None:
    if len(history) < 2:
        return None
    return round(history[-1] - history[-2], 6)


def fetch_task_statuses(submit_cooldown: int, query_cooldown: int) -> list[TaskStatus]:
    state = load_state()
    tasks: list[TaskStatus] = []

    for task_id, info in state.get("tasks", {}).items():
        submit_left = cooldown_remaining(info.get("last_submit_ts"), submit_cooldown)
        query_left = cooldown_remaining(info.get("last_query_ts"), query_cooldown)
        history = info.get("score_history") or []
        if info.get("last_score") is not None and not history:
            history = [float(info["last_score"])]
        tasks.append(
            TaskStatus(
                task_id=task_id,
                owner=info.get("owner"),
                last_score=info.get("last_score"),
                score_delta=_score_delta(history),
                score_history=[float(x) for x in history],
                attempt=info.get("attempt"),
                submit_cooldown_seconds=submit_left,
                query_cooldown_seconds=query_left,
                submit_ready=submit_left <= 0,
                query_ready=query_left <= 0,
                updated_at=info.get("updated_at"),
                wandb_url=info.get("wandb_url"),
            )
        )

    return tasks


def load_jobs_state() -> dict:
    return load_state().get("jobs", {})
