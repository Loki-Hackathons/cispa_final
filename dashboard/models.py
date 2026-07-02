"""Pydantic schemas for dashboard API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GpuSummary(BaseModel):
    used: int = 0
    team_jobs: int = 0


class NodePack(BaseModel):
    label: str
    job_ids: list[str] = Field(default_factory=list)
    gpu_total: int
    gpus_per_node: int


class GpuSchedule(BaseModel):
    gpus_per_node: int = 4
    running_gpus: int = 0
    pending_gpus: int = 0
    pending_packs: list[NodePack] = Field(default_factory=list)


class JobProgress(BaseModel):
    step: int | None = None
    total_steps: int | None = None
    unit: str | None = None
    phase: str | None = None
    message: str | None = None
    progress_pct: float | None = None
    heartbeat_stale: bool = False
    task_id: str | None = None
    attempt: int | None = None
    status: str | None = None


class CommandChip(BaseModel):
    label: str
    command: str


class SlurmJob(BaseModel):
    job_id: str
    user: str
    name: str
    gpus: int | None = None
    state: str
    elapsed_seconds: int | None = None
    time_limit_seconds: int | None = None
    eta_seconds: int | None = None
    eta_kind: str | None = None
    partition: str | None = None
    queue_position: int | None = None
    priority: int | None = None
    reason: str | None = None
    start_time: str | None = None
    compatible_with: list[str] = Field(default_factory=list)
    progress: JobProgress | None = None
    commands: list[CommandChip] = Field(default_factory=list)


class TaskStatus(BaseModel):
    task_id: str
    owner: str | None = None
    last_score: float | None = None
    score_delta: float | None = None
    score_history: list[float] = Field(default_factory=list)
    attempt: int | None = None
    submit_cooldown_seconds: int = 0
    query_cooldown_seconds: int = 0
    submit_ready: bool = True
    query_ready: bool = True
    updated_at: str | None = None
    wandb_url: str | None = None
    commands: list[CommandChip] = Field(default_factory=list)


class NextAction(BaseModel):
    kind: str
    priority: int
    message: str
    owner: str | None = None
    task_id: str | None = None
    job_id: str | None = None


class FailedJob(BaseModel):
    job_id: str
    user: str
    name: str
    state: str
    exit_code: str | None = None
    ended_at: str | None = None
    log_err: str | None = None
    commands: list[CommandChip] = Field(default_factory=list)


class OwnerSummary(BaseModel):
    owner: str
    running_jobs: int = 0
    pending_jobs: int = 0
    task_ids: list[str] = Field(default_factory=list)
    submit_ready_tasks: list[str] = Field(default_factory=list)
    query_ready_tasks: list[str] = Field(default_factory=list)


class ClusterStatus(BaseModel):
    partition: str
    nodes_alloc: int | None = None
    nodes_idle: int | None = None
    nodes_total: int | None = None
    gpus_alloc: int | None = None
    gpus_idle: int | None = None
    gpus_total: int | None = None
    team_pending: int = 0
    note: str | None = None


class LeaderboardRow(BaseModel):
    task_id: str
    team_rank: int | None = None
    team_score: float | None = None
    leader_score: float | None = None
    gap: float | None = None
    updated_at: str | None = None


class DashboardStatus(BaseModel):
    mode: str
    timestamp: str
    refresh_seconds: int
    account: str
    gpu_summary: GpuSummary
    gpu_schedule: GpuSchedule = Field(default_factory=GpuSchedule)
    slurm_jobs: list[SlurmJob] = Field(default_factory=list)
    tasks: list[TaskStatus] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list)
    failed_jobs: list[FailedJob] = Field(default_factory=list)
    owners: list[OwnerSummary] = Field(default_factory=list)
    cluster: ClusterStatus | None = None
    leaderboard: list[LeaderboardRow] = Field(default_factory=list)
    leaderboard_page_url: str = ""
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    ok: bool = True
    mode: str
