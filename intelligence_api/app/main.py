from __future__ import annotations
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from .db import init_db
from .logging_mw import configure_logging, TraceMiddleware
from .config import CLIPS_DIR, ANNOTATED_DIR
from .routers import ingest, analytics, health, dashboard, uploads
from .bootstrap import bootstrap_from_jsonl

configure_logging()
log = logging.getLogger("api")

app = FastAPI(title="Purplle Intelligence API", version="1.0.0")
app.add_middleware(TraceMiddleware)

app.include_router(ingest.router)
app.include_router(analytics.router)
app.include_router(health.router)
app.include_router(dashboard.router)
app.include_router(uploads.router)

# Serve original CCTV clips for the dashboard (read-only).
if CLIPS_DIR.exists():
    app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
if ANNOTATED_DIR.exists():
    app.mount("/annotated", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(Exception)
async def _unhandled(req: Request, exc: Exception):
    log.exception("unhandled %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "an unexpected error occurred"},
    )


@app.exception_handler(RequestValidationError)
async def _validation(req: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "validation", "details": exc.errors()})


@app.on_event("startup")
def _startup():
    init_db()
    try:
        bootstrap_from_jsonl()
    except Exception as ex:
        log.warning("bootstrap failed: %s", ex)


# Also init at import so TestClient (which doesn't fire startup unless used
# as a context manager) and ad-hoc imports see the schema.
init_db()
try:
    bootstrap_from_jsonl()
except Exception as ex:
    log.warning("bootstrap-on-import failed: %s", ex)
