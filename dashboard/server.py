"""FastAPI server for browser dashboard."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.config import load_config
from dashboard.models import DashboardStatus, HealthResponse
from dashboard.providers.live import LiveProvider
from dashboard.providers.mock import MockProvider

config = load_config()
provider = MockProvider(config) if config.mode == "mock" else LiveProvider(config)

app = FastAPI(title="LOKI CISPA Dashboard", version="1.0.0")

_CLIENT_DIST = Path(__file__).resolve().parent.parent / "client" / "dist"


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, mode=config.mode)


@app.get("/api/status", response_model=DashboardStatus)
def status() -> DashboardStatus:
    return provider.get_status()


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
