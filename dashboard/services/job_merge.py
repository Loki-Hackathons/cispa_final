"""Merge squeue jobs with team_state job progress and compute progress ETA."""

from __future__ import annotations

from datetime import datetime

from dashboard.models import JobProgress, SlurmJob


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _extrapolate_eta(elapsed_seconds: int | None, step: int, total_steps: int) -> int | None:
    if not elapsed_seconds or step <= 0 or total_steps <= 0 or step >= total_steps:
        return None
    pct = step / total_steps
    if pct <= 0:
        return None
    total_est = elapsed_seconds / pct
    return max(0, int(total_est - elapsed_seconds))


def build_progress(
    raw: dict | None,
    *,
    stale_seconds: int,
    elapsed_seconds: int | None,
    now: datetime | None = None,
) -> JobProgress | None:
    if not raw:
        return None

    ref = now or datetime.now()
    updated = _parse_ts(raw.get("updated_at"))
    stale = False
    if updated:
        stale = (ref - updated).total_seconds() > stale_seconds

    step = raw.get("step")
    total = raw.get("total_steps")
    pct = None
    if isinstance(step, int) and isinstance(total, int) and total > 0:
        pct = min(100.0, 100.0 * step / total)

    return JobProgress(
        step=step if isinstance(step, int) else None,
        total_steps=total if isinstance(total, int) else None,
        unit=raw.get("unit"),
        phase=raw.get("phase"),
        message=raw.get("message"),
        progress_pct=pct,
        heartbeat_stale=stale,
        task_id=raw.get("task_id"),
        attempt=raw.get("attempt"),
        status=raw.get("status"),
    )


def apply_progress_eta(
    job: SlurmJob,
    progress: JobProgress | None,
    progress_raw: dict | None,
    *,
    stale_seconds: int,
) -> SlurmJob:
    """Prefer reported ETA, then extrapolated, keep slurm eta as fallback."""
    if not progress or job.state != "RUNNING":
        return job

    reported_eta = progress_raw.get("eta_seconds") if progress_raw else None
    if isinstance(reported_eta, int) and reported_eta >= 0 and not progress.heartbeat_stale:
        return job.model_copy(
            update={"eta_seconds": reported_eta, "eta_kind": "reported"}
        )

    if (
        progress.step
        and progress.total_steps
        and progress.step > 0
        and not progress.heartbeat_stale
    ):
        eta = _extrapolate_eta(job.elapsed_seconds, progress.step, progress.total_steps)
        if eta is not None:
            return job.model_copy(update={"eta_seconds": eta, "eta_kind": "extrapolated"})

    if progress.heartbeat_stale and job.eta_kind in (None, "extrapolated", "reported"):
        return job.model_copy(update={"eta_kind": "stale_progress"})

    return job


def merge_job_progress(
    jobs: list[SlurmJob],
    jobs_state: dict,
    *,
    stale_seconds: int,
    now: datetime | None = None,
) -> list[SlurmJob]:
    merged: list[SlurmJob] = []
    for job in jobs:
        raw = jobs_state.get(job.job_id)
        progress = build_progress(
            raw, stale_seconds=stale_seconds, elapsed_seconds=job.elapsed_seconds, now=now
        )
        updated = apply_progress_eta(job, progress, raw, stale_seconds=stale_seconds)
        merged.append(updated.model_copy(update={"progress": progress}))
    return merged
