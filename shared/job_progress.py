"""Report Slurm job progress into team_state.json for the dashboard."""

from __future__ import annotations

import os
from datetime import datetime

from team_state import load_state, save_state

RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


def _slurm_job_id() -> str | None:
    jid = os.environ.get("SLURM_JOB_ID")
    return jid if jid and jid != "0" else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_job(job_id: str, fields: dict) -> None:
    state = load_state()
    jobs = state.setdefault("jobs", {})
    entry = jobs.setdefault(job_id, {"job_id": job_id})
    entry.update(fields)
    entry["updated_at"] = _now()
    save_state(state)


def bind_job(
    task_id: str,
    *,
    attempt: int | None = None,
    owner: str | None = None,
    job_name: str | None = None,
    total_steps: int | None = None,
    unit: str | None = None,
) -> str | None:
    """Register job at script start. Uses SLURM_JOB_ID from the environment."""
    job_id = _slurm_job_id()
    if not job_id:
        return None

    _write_job(
        job_id,
        {
            "job_name": job_name or os.environ.get("SLURM_JOB_NAME", ""),
            "owner": owner or os.environ.get("USER"),
            "task_id": task_id,
            "attempt": attempt,
            "phase": "init",
            "step": 0,
            "total_steps": total_steps or 0,
            "unit": unit,
            "message": "job started",
            "status": RUNNING,
            "eta_seconds": None,
        },
    )
    return job_id


def report(
    step: int,
    total_steps: int,
    phase: str,
    *,
    message: str | None = None,
    unit: str | None = None,
    eta_seconds: int | None = None,
) -> None:
    """Heartbeat during long loops (epoch, batch, image index, etc.)."""
    job_id = _slurm_job_id()
    if not job_id:
        return

    fields: dict = {
        "phase": phase,
        "step": step,
        "total_steps": total_steps,
        "status": RUNNING,
        "eta_seconds": eta_seconds,
    }
    if message is not None:
        fields["message"] = message
    if unit is not None:
        fields["unit"] = unit
    _write_job(job_id, fields)


def complete(message: str = "finished") -> None:
    job_id = _slurm_job_id()
    if not job_id:
        return
    _write_job(job_id, {"status": COMPLETED, "message": message, "eta_seconds": 0})


def fail(message: str = "failed") -> None:
    job_id = _slurm_job_id()
    if not job_id:
        return
    _write_job(job_id, {"status": FAILED, "message": message, "eta_seconds": None})
