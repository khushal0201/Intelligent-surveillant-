"""Upload a clip → run detection_pipeline → ingest events. Tracked by job_id."""
from __future__ import annotations
import asyncio
import json
import logging
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from .. import models, schemas
from ..db import get_db, session_scope
from ..config import (
    UPLOAD_DIR, PIPELINE_DIR, PYTHON_EXE, STORE_ID, CAMERA_CLIPS, ANNOTATED_DIR,
)
from .ingest import ingest as ingest_endpoint

router = APIRouter(prefix="/uploads", tags=["uploads"])
log = logging.getLogger("uploads")

JOBS: dict[str, dict] = {}


def _run_pipeline(job_id: str, video_path: Path, camera_id: str) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    out_jsonl = video_path.with_suffix(".events.jsonl")
    cmd = [
        PYTHON_EXE,
        str(PIPELINE_DIR / "run.py"),
        "--footage-dir", str(video_path.parent),
        "--layout", str(PIPELINE_DIR / "configs" / "store_layout.json"),
        "--schema", str(PIPELINE_DIR / "schema" / "event_schema.json"),
        "--out", str(out_jsonl),
        "--detector-backend", "onnx",
        "--onnx-model-path", str(PIPELINE_DIR / "models" / "yolov8n.onnx"),
        "--staff-mode", "openclip",
        "--staff-ref-dir", str(PIPELINE_DIR / "references" / "staff"),
        "--customer-ref-dir", str(PIPELINE_DIR / "references" / "customers"),
        "--conf-thresh", "0.35",
        "--stride", "5",
        "--cameras", camera_id,
        "--save-video-dir", str(ANNOTATED_DIR),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(PIPELINE_DIR.parent),
            capture_output=True, text=True, timeout=60 * 30,
        )
        job["stdout_tail"] = (proc.stdout or "")[-2000:]
        job["stderr_tail"] = (proc.stderr or "")[-2000:]
        if proc.returncode != 0:
            job["status"] = "failed"
            job["error"] = f"pipeline exit {proc.returncode}"
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

    accepted = 0
    duplicates = 0
    failed = 0
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

    job["status"] = "done"
    job["accepted"] = accepted
    job["duplicates"] = duplicates
    job["failed"] = failed
    job["events_path"] = str(out_jsonl)


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
            failed += 1
            continue
        if eid in existing:
            duplicates += 1
            continue
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
    if camera_id not in CAMERA_CLIPS:
        raise HTTPException(400, f"unknown camera_id; expected one of {list(CAMERA_CLIPS)}")
    job_id = uuid.uuid4().hex[:12]
    # Pipeline resolves source_clip by name inside --footage-dir; create a
    # per-job dir and place the uploaded file under the expected name.
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / CAMERA_CLIPS[camera_id]
    with open(target, "wb") as out:
        shutil.copyfileobj(file.file, out)

    JOBS[job_id] = {
        "job_id": job_id,
        "camera_id": camera_id,
        "video_path": str(target),
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

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
