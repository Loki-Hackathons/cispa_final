"""Optional leaderboard data — team_state in live mode; HTTP poll if configured."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import requests

from dashboard.models import LeaderboardRow

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from team_state import load_state  # noqa: E402

# API task ids (same as shared/submit.py --task-id)
API_TASK_IDS = (
    "30-watermark-localization",
    "29-mgi",
    "21-fl-audit",
)

TASK_DISPLAY: dict[str, str] = {
    "30-watermark-localization": "task_1",
    "29-mgi": "task_2",
    "21-fl-audit": "task_3",
}


def leaderboard_from_team_state(task_ids: tuple[str, ...]) -> list[LeaderboardRow]:
    """Best scores from team_state after API submits (no rank/gap without organizer JSON API)."""
    state = load_state()
    tasks = state.get("tasks", {})
    rows: list[LeaderboardRow] = []

    for task_id in task_ids:
        info = tasks.get(task_id, {})
        rows.append(
            LeaderboardRow(
                task_id=TASK_DISPLAY.get(task_id, task_id),
                team_score=float(info["last_score"]) if info.get("last_score") is not None else None,
                updated_at=info.get("updated_at"),
            )
        )

    return rows


def fetch_leaderboard(
    api_base_url: str,
    task_ids: tuple[str, ...],
    api_key: str | None = None,
) -> tuple[list[LeaderboardRow], list[str]]:
    """Optional poll — only if organizer exposes GET {base}/leaderboard/{task_id}."""
    warnings: list[str] = []
    if not api_base_url:
        return [], warnings

    rows: list[LeaderboardRow] = []
    headers = {"X-API-Key": api_key} if api_key else {}

    for task_id in task_ids:
        url = f"{api_base_url.rstrip('/')}/leaderboard/{task_id}"
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            warnings.append(f"leaderboard {task_id}: {exc}")
            rows.append(LeaderboardRow(task_id=TASK_DISPLAY.get(task_id, task_id)))
            continue

        team_score = data.get("team_score") or data.get("score")
        team_rank = data.get("team_rank") or data.get("rank")
        leader_score = data.get("leader_score") or data.get("top_score")
        gap = None
        if team_score is not None and leader_score is not None:
            try:
                gap = float(leader_score) - float(team_score)
            except (TypeError, ValueError):
                gap = None

        rows.append(
            LeaderboardRow(
                task_id=TASK_DISPLAY.get(task_id, task_id),
                team_rank=int(team_rank) if team_rank is not None else None,
                team_score=float(team_score) if team_score is not None else None,
                leader_score=float(leader_score) if leader_score is not None else None,
                gap=gap,
                updated_at=datetime.now().isoformat(timespec="seconds"),
            )
        )

    return rows, warnings
