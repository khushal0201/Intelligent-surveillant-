"""Upload a clip → run detection_pipeline → ingest events. Tracked by job_id.

Streams pipeline stdout via SSE (`/uploads/{job_id}/stream`) and continuously
writes a preview JPG (`/uploads/{job_id}/preview.jpg`) so the UI can show
live frame-by-frame predictions while the pipeline runs.
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
import shutil
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from sqlalchemy.orm import Session
from .. import models
from ..db import session_scope
from ..config import (
    UPLOAD_DIR, PIPELINE_DIR, PYTHON_EXE, STORE_ID, CAMERA_CLIPS, ANNOTATED_DIR,
    STORES, CAMERA_TO_STORE,
)

router = APIRouter(prefix="/uploads", tags=["uploads"])
log = logging.getLogger("uploads")

JOBS: dict[str, dict] = {}
JOB_LOGS: dict[str, deque] = {}
LOG_MAX = 800

_PROG_RE = re.compile(r"(\d+)it \[")


def _resolve_store_layout(camera_id: str) -> tuple[str, str]:
    sid = CAMERA_TO_STORE.get(camera_id)
    if sid and sid in STORES:
        return sid, STORES[sid]["layout"]
    return STORE_ID, str(PIPELINE_DIR / "configs" / "store_layout.json")


def _run_pipeline(job_id: str, video_path: Path, camera_id: str) -> None:
    job = JOBS[job_id]
    logs = JOB_LOGS.setdefault(job_id, deque(maxlen=LOG_MAX))
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat() + "Z"

    out_jsonl = video_path.with_suffix(".events.jsonl")
    preview_jpg = video_path.parent / "preview.jpg"
    annot_dir = video_path.parent / "annotated"
    job["preview_path"] = str(preview_jpg)
    job["annotated_dir"] = str(annot_dir)

    store_id, layout_path = _resolve_store_layout(camera_id)
    job["store_id"] = store_id

    cmd = [
        PYTHON_EXE,
        "-u",
        str(PIPELINE_DIR / "run.py"),
        "--footage-dir", str(video_path.parent),
        "--layout", layout_path,
        "--schema", str(PIPELINE_DIR / "schema" / "event_schema.json"),
        "--out", str(out_jsonl),
        "--detector-backend", "onnx",
        "--onnx-model-path", str(PIPELINE_DIR / "models" / "yolov8n.onnx"),
        "--staff-mode", "openclip",
        "--conf-thresh", "0.35",
        "--stride", "5",
        "--cameras", camera_id,
        "--save-video-dir", str(annot_dir),
        "--preview-jpg-path", str(preview_jpg),
        "--preview-every-n", "2",
    ]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PIPELINE_DIR.parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        job["pid"] = proc.pid
        for line in proc.stdout or []:
            line = line.rstrip("\n")
            logs.append(line)
            m = _PROG_RE.search(line)
            if m:
                try:
                    job["frames_processed"] = int(m.group(1))
                except ValueError:
                    pass
            if line.startswith("[run] "):
                job["current_step"] = line[6:]
        rc = proc.wait()
        job["return_code"] = rc
        if rc != 0:
            job["status"] = "failed"
            job["error"] = f"pipeline exit {rc}"
            return
    except FileNotFoundError as ex:
        job["status"] = "failed"
        job["error"] = f"python/pipeline not found: {ex}"
        return
    except Exception as ex:
        job["status"] = "failed"
        job["error"] = str(ex)
        return

    if not out_jsonl.exists():
        job["status"] = "failed"
        job["error"] = "no events emitted"
        return

    accepted = duplicates = failed = 0
    with session_scope() as db, open(out_jsonl, "r", encoding="utf-8") as f:
        batch: list[dict] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                batch.append(json.loads(line))
            except Exception:
                failed += 1
                continue
            if len(batch) >= 250:
                a, d, fl = _ingest_dicts(db, batch)
                accepted += a; duplicates += d; failed += fl
                batch = []
        if batch:
            a, d, fl = _ingest_dicts(db, batch)
            accepted += a; duplicates += d; failed += fl

    annot_src = annot_dir / f"{camera_id}.mp4"
    if annot_src.exists():
        try:
            target_dir = (Path(STORES[store_id]["annotated_dir"])
                          if store_id in STORES else ANNOTATED_DIR)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(annot_src, target_dir / f"{camera_id}.mp4")
            job["annotated_url"] = (
                f"/annotated_{store_id}/{camera_id}.mp4" if store_id in STORES
                else f"/annotated/{camera_id}.mp4"
            )
        except Exception as ex:
            logs.append(f"[copy-annot-failed] {ex}")

    job["status"] = "done"
    job["accepted"] = accepted
    job["duplicates"] = duplicates
    job["failed"] = failed
    job["events_path"] = str(out_jsonl)
    job["finished_at"] = datetime.utcnow().isoformat() + "Z"


def _ingest_dicts(db: Session, rows: list[dict]) -> tuple[int, int, int]:
    accepted = duplicates = failed = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ids = [r.get("event_id") for r in rows if r.get("event_id")]
    existing = {
        r[0] for r in db.execute(
            models.Event.__table__.select().with_only_columns(models.Event.event_id)
            .where(models.Event.event_id.in_(ids))
        ).all()
    }
    for r in rows:
        eid = r.get("event_id")
        if not eid:
            failed += 1; continue
        if eid in existing:
            duplicates += 1; continue
        try:
            ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            db.add(models.Event(
                event_id=eid,
                store_id=r["store_id"],
                camera_id=r["camera_id"],
                visitor_id=r["visitor_id"],
                event_type=r["event_type"],
                timestamp=ts,
                zone_id=r.get("zone_id"),
                dwell_ms=int(r.get("dwell_ms") or 0),
                is_staff=bool(r.get("is_staff", False)),
                confidence=float(r.get("confidence") or 0.0),
                meta=r.get("metadata") or {},
                created_at=now,
            ))
            accepted += 1
            existing.add(eid)
        except Exception:
            failed += 1
    return accepted, duplicates, failed


@router.post("")
async def upload_clip(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
):
    valid_cams = set(CAMERA_CLIPS) | set(CAMERA_TO_STORE)
    if camera_id not in valid_cams:
        raise HTTPException(400, f"unknown camera_id; expected one of {sorted(valid_cams)}")
    job_id = uuid.uuid4().hex[:12]
    sid = CAMERA_TO_STORE.get(camera_id)
    if sid and sid in STORES:
        clip_name = STORES[sid]["cameras"].get(camera_id, f"{camera_id}.mp4")
    else:
        clip_name = CAMERA_CLIPS.get(camera_id, f"{camera_id}.mp4")
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / clip_name
    with open(target, "wb") as out:
        shutil.copyfileobj(file.file, out)

    JOBS[job_id] = {
        "job_id": job_id,
        "camera_id": camera_id,
        "video_path": str(target),
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "frames_processed": 0,
    }
    JOB_LOGS[job_id] = deque(maxlen=LOG_MAX)

    threading.Thread(
        target=_run_pipeline, args=(job_id, target, camera_id), daemon=True,
    ).start()
    log.info("upload_queued", extra={"event_count": 1})
    return JOBS[job_id]


@router.get("/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job_id")
    return JOBS[job_id]


@router.get("")
def list_jobs():
    return {"jobs": list(JOBS.values())}


# 1x1 transparent PNG returned when no preview frame exists yet
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


@router.get("/{job_id}/preview.jpg")
def job_preview(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job_id")
    p = Path(job.get("preview_path") or "")
    if not p.exists():
        return Response(content=_PLACEHOLDER_PNG, media_type="image/png",
                        headers={"Cache-Control": "no-store"})
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/{job_id}/stream")
async def job_stream(job_id: str):
    """Server-Sent Events: pipeline stdout + status updates."""
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job_id")

    async def gen():
        sent = 0
        last_status = None
        while True:
            job = JOBS.get(job_id)
            logs = JOB_LOGS.get(job_id, deque())
            if job:
                cur = (job.get("status"), job.get("frames_processed", 0),
                       job.get("current_step"))
                if cur != last_status:
                    last_status = cur
                    payload = json.dumps({
                        "type": "status",
                        "status": job.get("status"),
                        "frames_processed": job.get("frames_processed", 0),
                        "current_step": job.get("current_step"),
                        "accepted": job.get("accepted"),
                        "error": job.get("error"),
                        "annotated_url": job.get("annotated_url"),
                    })
                    yield f"event: status\ndata: {payload}\n\n"
            buf = list(logs)
            for line in buf[sent:]:
                yield f"event: log\ndata: {json.dumps(line)}\n\n"
            sent = len(buf)
            if job and job.get("status") in {"done", "failed"}:
                yield f"event: done\ndata: {json.dumps(job)}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})
