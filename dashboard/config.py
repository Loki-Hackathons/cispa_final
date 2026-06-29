"""Dashboard configuration — edit constants here before running."""

from dataclasses import dataclass

# mock = local dev with fixture data | live = JURECA (squeue + team_state.json)
MODE = "mock"

HOST = "127.0.0.1"
PORT = 8080
REFRESH_SECONDS = 5
SLURM_ACCOUNT = "training2557"
SUBMIT_COOLDOWN = 300  # seconds (5 min)
QUERY_COOLDOWN = 900   # seconds (15 min)
GPUS_PER_NODE = 4  # JURECA dc-gpu nodes (4x A100)
SLURM_PARTITION = "dc-gpu-devel"
PROGRESS_STALE_SECONDS = 120
SACCT_HOURS = 6
COOLDOWN_SOON_SECONDS = 60

# Leaderboard: set URL when subject is released (empty = disabled in live)
LEADERBOARD_URL = ""
LEADERBOARD_TASK_IDS: list[str] = []  # e.g. ["task_1", "task_2", "task_3"]


@dataclass(frozen=True)
class DashboardConfig:
    mode: str  # mock | live
    host: str
    port: int
    refresh_seconds: int
    slurm_account: str
    submit_cooldown: int
    query_cooldown: int
    gpus_per_node: int
    slurm_partition: str
    progress_stale_seconds: int
    sacct_hours: int
    cooldown_soon_seconds: int
    leaderboard_url: str
    leaderboard_task_ids: tuple[str, ...]


def load_config() -> DashboardConfig:
    mode = MODE.lower()
    if mode not in ("mock", "live"):
        mode = "mock"
    return DashboardConfig(
        mode=mode,
        host=HOST,
        port=PORT,
        refresh_seconds=REFRESH_SECONDS,
        slurm_account=SLURM_ACCOUNT,
        submit_cooldown=SUBMIT_COOLDOWN,
        query_cooldown=QUERY_COOLDOWN,
        gpus_per_node=GPUS_PER_NODE,
        slurm_partition=SLURM_PARTITION,
        progress_stale_seconds=PROGRESS_STALE_SECONDS,
        sacct_hours=SACCT_HOURS,
        cooldown_soon_seconds=COOLDOWN_SOON_SECONDS,
        leaderboard_url=LEADERBOARD_URL.strip(),
        leaderboard_task_ids=tuple(LEADERBOARD_TASK_IDS),
    )
