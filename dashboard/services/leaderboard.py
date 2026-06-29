"""Optional leaderboard polling."""

from datetime import datetime

import requests

from dashboard.models import LeaderboardRow


def fetch_leaderboard(
    base_url: str,
    task_ids: tuple[str, ...],
    api_key: str | None = None,
) -> tuple[list[LeaderboardRow], list[str]]:
    warnings: list[str] = []
    if not base_url:
        return [], warnings

    rows: list[LeaderboardRow] = []
    headers = {"X-API-Key": api_key} if api_key else {}

    for task_id in task_ids:
        url = f"{base_url.rstrip('/')}/leaderboard/{task_id}"
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            warnings.append(f"leaderboard {task_id}: {exc}")
            rows.append(LeaderboardRow(task_id=task_id))
            continue

        # Flexible parsing — adapt when subject is released
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
                task_id=task_id,
                team_rank=int(team_rank) if team_rank is not None else None,
                team_score=float(team_score) if team_score is not None else None,
                leader_score=float(leader_score) if leader_score is not None else None,
                gap=gap,
                updated_at=datetime.now().isoformat(timespec="seconds"),
            )
        )

    return rows, warnings
