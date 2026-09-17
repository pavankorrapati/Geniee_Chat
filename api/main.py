from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UI_DIR = PROJECT_ROOT / "ui"


# ================================================================
# FASTAPI APPLICATION
# ================================================================

app = FastAPI(
    title="Geniee AI",
    description="Local Geniee 27M language model API",
    version="1.0.0",
)


# ================================================================
# API ROUTES
# ================================================================

app.include_router(
    router,
    prefix="/api",
)


# ================================================================
# UI
# ================================================================

if UI_DIR.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=UI_DIR,
            html=True,
        ),
        name="ui",
    )


# ================================================================
# ROOT INFORMATION
# ================================================================

@app.get("/api")
def api_info():
    return {
        "name": "Geniee AI",
        "version": "1.0.0",
        "status": "running",
        "model": "Geniee 27M",
    }