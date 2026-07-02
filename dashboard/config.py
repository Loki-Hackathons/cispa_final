"""Dashboard configuration — shared defaults in git; machine overrides in config_local.py."""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

# mock = local dev with fixture data | live = JURECA (squeue + team_state.json)
MODE = "mock"

HOST = "127.0.0.1"
PORT = 8080
REFRESH_SECONDS = 5
SLURM_ACCOUNT = "training2625"  # CISPA Grand Finals (regional was training2557)
SUBMIT_COOLDOWN = 300  # seconds (5 min)
QUERY_COOLDOWN = 900   # seconds (15 min)
GPUS_PER_NODE = 4  # JURECA dc-gpu nodes (4x A100)
SLURM_PARTITION = "dc-gpu-devel"
PROGRESS_STALE_SECONDS = 120
SACCT_HOURS = 6
COOLDOWN_SOON_SECONDS = 60

# Human-readable leaderboard (HTML — open in browser)
LEADERBOARD_PAGE_URL = "http://35.192.205.84/leaderboard_page"
# Optional JSON poll base (CISPA_BASE_URL). Leave empty — no public /leaderboard/* API today.
LEADERBOARD_POLL_URL = ""
# API task ids for team_state / submit.py (not task_1, task_2, …)
LEADERBOARD_TASK_IDS: list[str] = [
    "30-watermark-localization",
    "29-mgi",
    "21-fl-audit",
]

_OVERRIDE_KEYS = (
    "MODE",
    "HOST",
    "PORT",
    "REFRESH_SECONDS",
    "SLURM_ACCOUNT",
    "SUBMIT_COOLDOWN",
    "QUERY_COOLDOWN",
    "GPUS_PER_NODE",
    "SLURM_PARTITION",
    "PROGRESS_STALE_SECONDS",
    "SACCT_HOURS",
    "COOLDOWN_SOON_SECONDS",
    "LEADERBOARD_PAGE_URL",
    "LEADERBOARD_POLL_URL",
    "LEADERBOARD_TASK_IDS",
)


def _apply_local_overrides() -> None:
    local_path = Path(__file__).with_name("config_local.py")
    if not local_path.is_file():
        return
    spec = importlib.util.spec_from_file_location("dashboard.config_local", local_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g = globals()
    for key in _OVERRIDE_KEYS:
        if hasattr(module, key):
            g[key] = getattr(module, key)


_apply_local_overrides()

# Env wins for mode only (handy on cluster without editing files).
if os.environ.get("DASHBOARD_MODE"):
    MODE = os.environ["DASHBOARD_MODE"]


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
    leaderboard_page_url: str
    leaderboard_poll_url: str
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
        leaderboard_page_url=LEADERBOARD_PAGE_URL.strip(),
        leaderboard_poll_url=LEADERBOARD_POLL_URL.strip(),
        leaderboard_task_ids=tuple(LEADERBOARD_TASK_IDS),
    )
