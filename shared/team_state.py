"""Read/write shared team state on the cluster."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Cluster path when on JURECA (finals team scratch); local fallback for dev
_CLUSTER_PATH = Path("/p/scratch/training2625/ansart1/loki/team_state.json")
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "team_state.json"


def state_path() -> Path:
    env = os.environ.get("CISPA_TEAM_STATE")
    if env:
        return Path(env)
    if _CLUSTER_PATH.parent.exists():
        return _CLUSTER_PATH
    return _LOCAL_PATH


def load_state() -> dict:
    path = state_path()
    if not path.exists():
        return {"tasks": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def update_task(task_id: str, owner: str | None = None, **fields) -> dict:
    """Atomic read-modify-write for one task entry."""
    state = load_state()
    tasks = state.setdefault("tasks", {})
    task = tasks.setdefault(task_id, {})
    if owner is not None:
        task["owner"] = owner
    task.update(fields)
    task["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)
    return task


def append_score(task_id: str, score: float | None, *, max_history: int = 5) -> list[float]:
    """Append score to history; returns full history list."""
    if score is None:
        state = load_state()
        return state.get("tasks", {}).get(task_id, {}).get("score_history", [])

    state = load_state()
    tasks = state.setdefault("tasks", {})
    task = tasks.setdefault(task_id, {})
    history: list[float] = list(task.get("score_history", []))
    history.append(float(score))
    task["score_history"] = history[-max_history:]
    task["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)
    return task["score_history"]


def cooldown_remaining(last_ts: str | None, cooldown_seconds: int) -> int:
    """Seconds until cooldown expires. 0 if ready."""
    if not last_ts:
        return 0
    try:
        last = datetime.fromisoformat(last_ts)
    except ValueError:
        return 0
    elapsed = (datetime.now() - last).total_seconds()
    return max(0, int(cooldown_seconds - elapsed))
