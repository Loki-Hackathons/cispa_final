"""Parse SLURM time strings to seconds."""

from __future__ import annotations

from datetime import datetime


def parse_slurm_time(value: str | None) -> int | None:
    """Parse SLURM TIME/TIMELIMIT: MM:SS, HH:MM:SS, D-HH:MM:SS, or UNLIMITED."""
    if not value or value in ("", "N/A", "NOT_SET"):
        return None
    if value.upper() in ("UNLIMITED", "INVALID"):
        return None

    try:
        if "-" in value:
            days_part, time_part = value.split("-", 1)
            days = int(days_part)
        else:
            days = 0
            time_part = value

        parts = time_part.split(":")
        if len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            return None

        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return None


def parse_slurm_datetime(value: str | None) -> datetime | None:
    """Parse squeue START_TIME (e.g. 2024-06-29T14:30:00)."""
    if not value or value.upper() in ("N/A", "NONE", "NOT_SET"):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None
