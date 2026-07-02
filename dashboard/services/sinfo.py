"""Cluster GPU availability via sinfo."""

from __future__ import annotations

import subprocess

from dashboard.models import ClusterStatus


def fetch_cluster_status(
    partition: str,
    gpus_per_node: int,
    team_pending: int,
) -> tuple[ClusterStatus | None, list[str]]:
    warnings: list[str] = []

    try:
        result = subprocess.run(
            ["sinfo", "-p", partition, "-h", "-o", "%A"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        warnings.append("sinfo not found (not on cluster?)")
        return None, warnings
    except subprocess.TimeoutExpired:
        warnings.append("sinfo timed out")
        return None, warnings

    if result.returncode != 0:
        warnings.append(f"sinfo failed: {result.stderr.strip() or 'unknown error'}")
        return None, warnings

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    # Format: alloc/idle/other/total
    parts = line.split("/")
    if len(parts) < 4:
        return ClusterStatus(
            partition=partition,
            team_pending=team_pending,
            note="Could not parse sinfo output",
        ), warnings

    try:
        alloc, idle, _other, total = (int(p) for p in parts[:4])
    except ValueError:
        return ClusterStatus(
            partition=partition,
            team_pending=team_pending,
            note=f"Unexpected sinfo: {line}",
        ), warnings

    return ClusterStatus(
        partition=partition,
        nodes_alloc=alloc,
        nodes_idle=idle,
        nodes_total=total,
        gpus_alloc=alloc * gpus_per_node,
        gpus_idle=idle * gpus_per_node,
        gpus_total=total * gpus_per_node,
        team_pending=team_pending,
    ), warnings
