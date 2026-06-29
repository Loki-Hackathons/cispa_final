"""Rich TUI dashboard — delegates to dashboard services."""

import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.config import load_config
from dashboard.providers.live import LiveProvider
from dashboard.providers.mock import MockProvider

console = Console()


def _fmt_cooldown(seconds: int) -> str:
    if seconds <= 0:
        return "[green]ready[/green]"
    return f"[yellow]{seconds // 60}m {seconds % 60}s left[/yellow]"


def _fmt_duration(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def slurm_table(status) -> Table:
    table = Table(title="SLURM Queue", expand=True)
    table.add_column("Pos")
    table.add_column("JobID", style="cyan")
    table.add_column("User")
    table.add_column("Name")
    table.add_column("GPUs")
    table.add_column("State")
    table.add_column("Pri")
    table.add_column("Elapsed")
    table.add_column("Time")

    for job in status.slurm_jobs:
        pos = "active" if job.state == "RUNNING" else (
            f"#{job.queue_position}" if job.queue_position else "-"
        )
        eta_label = "-"
        if job.eta_seconds is not None:
            if job.eta_kind == "scheduled_start":
                eta_label = f"start {_fmt_duration(job.eta_seconds)}"
            elif job.eta_kind == "time_limit_remaining":
                eta_label = f"limit {_fmt_duration(job.eta_seconds)}"
            else:
                eta_label = _fmt_duration(job.eta_seconds)

        table.add_row(
            pos,
            job.job_id,
            job.user,
            job.name[:20],
            str(job.gpus or "?"),
            job.state,
            str(job.priority or "-"),
            _fmt_duration(job.elapsed_seconds),
            eta_label,
        )

    if not status.slurm_jobs:
        table.add_row("-", "-", "-", "(no jobs)", "-", "-", "-", "-", "-")

    packs = status.gpu_schedule.pending_packs
    if packs:
        pack_lines = [
            f"{p.label}: {' + '.join(p.job_ids)} ({p.gpu_total}/{p.gpus_per_node} GPUs)"
            for p in packs
        ]
        table.caption = (
            f"Team GPUs running: ~{status.gpu_summary.used} | "
            f"Pending packs: {'; '.join(pack_lines)}"
        )
    else:
        table.caption = f"Team GPUs in use: ~{status.gpu_summary.used}"
    return table


def cooldown_table(status) -> Table:
    table = Table(title="API Cooldowns", expand=True)
    table.add_column("Task")
    table.add_column("Owner")
    table.add_column("Submit")
    table.add_column("Query")
    table.add_column("Last score")

    if not status.tasks:
        table.add_row("(none)", "-", "-", "-", "-")
    for task in status.tasks:
        score_str = f"{task.last_score:.4f}" if task.last_score is not None else "-"
        table.add_row(
            task.task_id,
            task.owner or "-",
            _fmt_cooldown(task.submit_cooldown_seconds),
            _fmt_cooldown(task.query_cooldown_seconds),
            score_str,
        )
    return table


def scores_table(status) -> Table:
    table = Table(title="Last Scores", expand=True)
    table.add_column("Task")
    table.add_column("Score")
    table.add_column("Attempt")
    table.add_column("Updated")

    for task in status.tasks:
        table.add_row(
            task.task_id,
            f"{task.last_score:.6f}" if task.last_score is not None else "-",
            str(task.attempt or "-"),
            (task.updated_at or "-")[:19],
        )
    if not status.tasks:
        table.add_row("(none)", "-", "-", "-")
    return table


def build_layout(status) -> Layout:
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))
    layout["body"].split_row(Layout(name="slurm", ratio=1), Layout(name="right", ratio=1))
    layout["right"].split_column(Layout(name="cooldowns"), Layout(name="scores"))

    now = datetime.now().strftime("%H:%M:%S")
    mode_label = status.mode.upper()
    layout["header"].update(
        Panel(
            f"[bold]LOKI — CISPA Finals[/bold]  |  {now}  |  mode: {mode_label}  |  refresh: {status.refresh_seconds}s",
            style="bold white on blue",
        )
    )
    layout["slurm"].update(Panel(slurm_table(status)))
    layout["cooldowns"].update(Panel(cooldown_table(status)))
    layout["scores"].update(Panel(scores_table(status)))
    return layout


def main() -> None:
    config = load_config()
    refresh_sec = config.refresh_seconds
    provider = MockProvider(config) if config.mode == "mock" else LiveProvider(config)
    console.print(f"[bold]Starting TUI dashboard (mode={config.mode}). Ctrl+C to quit.[/bold]")
    console.print("[dim]For browser UI: uvicorn dashboard.server:app --host 127.0.0.1 --port 8080[/dim]")

    status = provider.get_status()
    with Live(build_layout(status), refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(refresh_sec)
                status = provider.get_status()
                live.update(build_layout(status))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
