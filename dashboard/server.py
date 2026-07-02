"""FastAPI server for browser dashboard."""

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.config import load_config
from dashboard.models import DashboardStatus, HealthResponse
from dashboard.providers.live import LiveProvider
from dashboard.providers.mock import MockProvider

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "shared"))
from history import read_events  # noqa: E402

config = load_config()
provider = MockProvider(config) if config.mode == "mock" else LiveProvider(config)

app = FastAPI(title="LOKI CISPA Dashboard", version="1.0.0")

_CLIENT_DIST = _ROOT / "client" / "dist"
_MOCK_HISTORY = Path(__file__).resolve().parent / "fixtures" / "mock_history.jsonl"


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, mode=config.mode)


@app.get("/api/status", response_model=DashboardStatus)
def status() -> DashboardStatus:
    return provider.get_status()


@app.get("/api/history")
def history(limit: int = 100, task_id: str | None = None, kind: str | None = None) -> list[dict]:
    if config.mode == "mock":
        events = [json.loads(line) for line in _MOCK_HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]
        events.reverse()
        if task_id:
            events = [e for e in events if e.get("task_id") == task_id]
        if kind:
            events = [e for e in events if e.get("kind") == kind]
        return events[:limit]
    return read_events(limit=limit, task_id=task_id, kind=kind)


def _mount_static() -> None:
    if not _CLIENT_DIST.is_dir():
        return

    assets = _CLIENT_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        index = _CLIENT_DIST / "index.html"
        if full_path.startswith("api/"):
            return {"detail": "Not Found"}
        file_path = _CLIENT_DIST / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(index)


_mount_static()
