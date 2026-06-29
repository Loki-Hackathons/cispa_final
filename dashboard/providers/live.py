"""Live dashboard data from squeue and team_state."""

from dashboard.config import DashboardConfig
from dashboard.providers.base import StatusProvider
from dashboard.services.assemble import assemble_status
from dashboard.services.cooldowns import load_jobs_state
from dashboard.services.slurm import fetch_slurm_jobs


class LiveProvider(StatusProvider):
    def __init__(self, config: DashboardConfig):
        self._config = config

    def get_status(self):
        raw_jobs, gpu_summary, warnings = fetch_slurm_jobs(self._config.slurm_account)
        jobs_state = load_jobs_state()
        return assemble_status(
            self._config,
            mode="live",
            raw_jobs=raw_jobs,
            gpu_summary=gpu_summary,
            jobs_state=jobs_state,
            extra_warnings=warnings,
        )
