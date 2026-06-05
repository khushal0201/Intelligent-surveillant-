from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from .db import init_db
from .logging_mw import configure_logging, TraceMiddleware
from .config import CLIPS_DIR, ANNOTATED_DIR, STORES
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


# ---------------------------------------------------------------------------
# Range-aware mp4 streaming
# Starlette's StaticFiles ignores HTTP Range, which breaks HTML5 <video> seek.
# This handler returns 206 Partial Content for Range requests so the browser
# scrubber works end-to-end.
# ---------------------------------------------------------------------------
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _stream_video(file_path: Path, request: Request) -> StreamingResponse | FileResponse:
    if not file_path.is_file():
        raise HTTPException(404, f"not found: {file_path.name}")
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")
    if not range_header:
        return FileResponse(
            str(file_path),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )
    m = _RANGE_RE.fullmatch(range_header.strip())
    if not m:
        raise HTTPException(416, "invalid range")
    start_s, end_s = m.group(1), m.group(2)
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else file_size - 1
    if start >= file_size or end >= file_size or start > end:
        return JSONResponse(
            status_code=416,
            content={"error": "range_not_satisfiable"},
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    chunk = end - start + 1

    def _iter():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = chunk
            while remaining > 0:
                buf = f.read(min(1024 * 1024, remaining))
                if not buf:
                    break
                remaining -= len(buf)
                yield buf

    return StreamingResponse(
        _iter(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(chunk),
        },
    )


def _safe_join(base: Path, name: str) -> Path:
    p = (base / name).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise HTTPException(400, "invalid path")
    return p


@app.get("/annotated/{fname}", include_in_schema=False)
def annotated_legacy(fname: str, request: Request):
    return _stream_video(_safe_join(Path(ANNOTATED_DIR), fname), request)


def _make_annotated_route(sid: str, ann_dir: Path):
    @app.get(f"/annotated_{sid}/{{fname}}", include_in_schema=False)
    def _ann(fname: str, request: Request, _ad: Path = ann_dir):
        return _stream_video(_safe_join(_ad, fname), request)
    return _ann


def _make_raw_route(sid: str, raw_dir: Path):
    @app.get(f"/raw/{sid}/{{fname}}", include_in_schema=False)
    def _raw(fname: str, request: Request, _rd: Path = raw_dir):
        return _stream_video(_safe_join(_rd, fname), request)
    return _raw


# Serve original CCTV clips for the dashboard (read-only).
if CLIPS_DIR.exists():
    app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")

# Per-store annotated mp4 dirs + raw footage dirs — registered as
# Range-aware routes (above) so HTML5 <video> seeking works.
for _sid, _cfg in STORES.items():
    _ann = Path(_cfg["annotated_dir"])
    _ann.mkdir(parents=True, exist_ok=True)
    _make_annotated_route(_sid, _ann)
    _raw = Path(_cfg["footage_dir"])
    if _raw.exists():
        _make_raw_route(_sid, _raw)

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
