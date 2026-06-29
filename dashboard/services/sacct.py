"""Failed SLURM jobs via sacct."""

import subprocess

from dashboard.models import FailedJob


def fetch_failed_jobs(account: str, hours: int = 6) -> tuple[list[FailedJob], list[str]]:
    warnings: list[str] = []
    failed: list[FailedJob] = []
    start = f"now-{hours}hours"
    fmt = "%i|%u|%j|%T|%f|%End"

    try:
        result = subprocess.run(
            [
                "sacct",
                "-A",
                account,
                "-S",
                start,
                "--state=FAILED,TIMEOUT,CANCELLED",
                "-n",
                "-P",
                f"--format={fmt}",
                "-X",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        warnings.append("sacct not found (not on cluster?)")
        return failed, warnings
    except subprocess.TimeoutExpired:
        warnings.append("sacct timed out")
        return failed, warnings

    if result.returncode != 0:
        warnings.append(f"sacct failed: {result.stderr.strip() or 'unknown error'}")
        return failed, warnings

    seen: set[str] = set()
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        job_id, user, name, state, exit_code, ended = parts[:6]
        base_id = job_id.split(".")[0]
        if base_id in seen:
            continue
        seen.add(base_id)
        failed.append(
            FailedJob(
                job_id=base_id,
                user=user,
                name=name[:40],
                state=state,
                exit_code=exit_code or None,
                ended_at=ended or None,
                log_err=f"logs/slurm_{base_id}.err",
            )
        )

    failed.sort(key=lambda j: j.ended_at or "", reverse=True)
    return failed[:10], warnings
