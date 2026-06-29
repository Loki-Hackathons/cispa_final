"""Queue ordering, ETA semantics, and GPU node packing hints."""

from __future__ import annotations

from datetime import datetime
from itertools import combinations

from dashboard.models import GpuSchedule, NodePack, SlurmJob
from dashboard.services.time_parse import parse_slurm_datetime, parse_slurm_time

RUNNING_STATES = frozenset({"R", "RUNNING"})
PENDING_STATES = frozenset({"PD", "PENDING"})


def normalize_state(state: str) -> str:
    mapping = {
        "R": "RUNNING",
        "PD": "PENDING",
        "CG": "COMPLETING",
        "CF": "CONFIGURING",
        "S": "SUSPENDED",
    }
    return mapping.get(state, state)


def compute_eta(
    state: str,
    elapsed_seconds: int | None,
    time_limit_seconds: int | None,
    start_time: str | None,
    now: datetime | None = None,
) -> tuple[int | None, str | None]:
    """Return (eta_seconds, eta_kind). See cluster-guide for semantics."""
    norm = normalize_state(state)
    if norm == "RUNNING":
        if elapsed_seconds is not None and time_limit_seconds is not None:
            return max(0, time_limit_seconds - elapsed_seconds), "time_limit_remaining"
        return None, None

    if norm == "PENDING":
        start = parse_slurm_datetime(start_time)
        ref = now or datetime.now()
        if start and start > ref:
            return int((start - ref).total_seconds()), "scheduled_start"
        return None, None

    return None, None


def order_jobs(jobs: list[SlurmJob]) -> list[SlurmJob]:
    """Running first, then pending by Slurm priority (desc), then others."""
    running = [j for j in jobs if normalize_state(j.state) == "RUNNING"]
    pending = [j for j in jobs if normalize_state(j.state) == "PENDING"]
    other = [
        j
        for j in jobs
        if normalize_state(j.state) not in ("RUNNING", "PENDING")
    ]

    running.sort(key=lambda j: (-(j.elapsed_seconds or 0), j.job_id))
    pending.sort(key=lambda j: (-(j.priority or 0), j.job_id))
    other.sort(key=lambda j: j.job_id)

    ordered: list[SlurmJob] = []
    for job in running:
        ordered.append(job.model_copy(update={"queue_position": None}))
    for index, job in enumerate(pending, start=1):
        ordered.append(job.model_copy(update={"queue_position": index}))
    for job in other:
        ordered.append(job.model_copy(update={"queue_position": None}))

    return ordered


def _pending_jobs(jobs: list[SlurmJob]) -> list[SlurmJob]:
    return [
        j
        for j in jobs
        if normalize_state(j.state) == "PENDING" and j.gpus and j.gpus > 0
    ]


def _pack_pending_nodes(
    pending: list[SlurmJob],
    gpus_per_node: int,
) -> list[NodePack]:
    """Greedy first-fit decreasing bin packing for pending jobs."""
    if not pending:
        return []

    remaining = sorted(pending, key=lambda j: (-(j.gpus or 0), j.job_id))
    packs: list[NodePack] = []
    node_index = 1

    while remaining:
        pack_jobs: list[SlurmJob] = []
        pack_gpus = 0
        next_remaining: list[SlurmJob] = []

        for job in remaining:
            gpus = job.gpus or 0
            if not pack_jobs:
                pack_jobs = [job]
                pack_gpus = gpus
                continue
            if pack_gpus + gpus <= gpus_per_node:
                pack_jobs.append(job)
                pack_gpus += gpus
            else:
                next_remaining.append(job)

        packs.append(
            NodePack(
                label=f"Node pack {node_index}",
                job_ids=[j.job_id for j in pack_jobs],
                gpu_total=pack_gpus,
                gpus_per_node=gpus_per_node,
            )
        )
        node_index += 1
        remaining = next_remaining

    return packs


def _compatible_job_ids(job: SlurmJob, pending: list[SlurmJob], gpus_per_node: int) -> list[str]:
    """Other pending jobs that can share a node with this job (sum GPUs <= capacity)."""
    gpus = job.gpus or 0
    if gpus <= 0 or gpus > gpus_per_node:
        return []

    compatible: list[str] = []
    for other in pending:
        if other.job_id == job.job_id:
            continue
        other_gpus = other.gpus or 0
        if other_gpus > 0 and gpus + other_gpus <= gpus_per_node:
            compatible.append(other.job_id)

    for size in range(3, len(pending) + 1):
        for group in combinations(pending, size):
            if job not in group:
                continue
            total = sum(j.gpus or 0 for j in group)
            if total <= gpus_per_node:
                for other in group:
                    if other.job_id != job.job_id and other.job_id not in compatible:
                        compatible.append(other.job_id)

    return sorted(compatible)


def enrich_jobs(
    jobs: list[SlurmJob],
    gpus_per_node: int,
    now: datetime | None = None,
) -> tuple[list[SlurmJob], GpuSchedule]:
    ref = now or datetime.now()
    pending = _pending_jobs(jobs)
    running_gpus = sum(j.gpus or 0 for j in jobs if normalize_state(j.state) == "RUNNING")
    pending_gpus = sum(j.gpus or 0 for j in pending)

    compatibility: dict[str, list[str]] = {
        j.job_id: _compatible_job_ids(j, pending, gpus_per_node) for j in pending
    }

    enriched: list[SlurmJob] = []
    for job in jobs:
        eta_seconds, eta_kind = compute_eta(
            job.state,
            job.elapsed_seconds,
            job.time_limit_seconds,
            job.start_time,
            ref,
        )
        enriched.append(
            job.model_copy(
                update={
                    "state": normalize_state(job.state),
                    "eta_seconds": eta_seconds,
                    "eta_kind": eta_kind,
                    "compatible_with": compatibility.get(job.job_id, []),
                }
            )
        )

    ordered = order_jobs(enriched)
    schedule = GpuSchedule(
        gpus_per_node=gpus_per_node,
        running_gpus=running_gpus,
        pending_gpus=pending_gpus,
        pending_packs=_pack_pending_nodes(pending, gpus_per_node),
    )
    return ordered, schedule
