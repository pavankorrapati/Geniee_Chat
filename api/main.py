from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"

app = FastAPI(
    title="Geniee AI",
    description="Local Geniee 27M language model API",
    version="2.0.0",
)

app.include_router(router, prefix="/api")

# Serve the UI and its assets from the same FastAPI process.
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    index_file = UI_DIR / "index.html"
    if not index_file.exists():
        return {"name": "Geniee AI", "status": "running"}
    return FileResponse(index_file)


@app.get("/api", include_in_schema=False)
def api_info():
    return {
        "name": "Geniee AI",
        "version": "2.0.0",
        "status": "running",
        "model": "Geniee 27M",
        "docs": "/docs",
        "redoc": "/redoc",
    }
