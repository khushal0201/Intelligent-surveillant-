"""Dashboard helper endpoints: list cameras, fetch events per camera with
timestamps relative to that camera's first event so the player can sync.
"""
from __future__ import annotations
import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from .. import models
from ..db import get_db
from ..config import CAMERA_CLIPS, STORE_ID, ANNOTATED_DIR

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# When set (e.g. on Hugging Face Space), redirect video URLs to a remote host
# (GitHub raw, S3, etc.) so the small Space container does not need to ship
# the heavy mp4 assets. Trailing slash optional.
VIDEO_BASE_URL = (os.environ.get("VIDEO_BASE_URL") or "").rstrip("/")


def _annotated_url(cam_id: str) -> Optional[str]:
    """Public URL for an annotated mp4. Local path if the file exists,
    otherwise fall back to remote VIDEO_BASE_URL when configured."""
    if (ANNOTATED_DIR / f"{cam_id}.mp4").exists():
        return f"/annotated/{cam_id}.mp4"
    if VIDEO_BASE_URL:
        return f"{VIDEO_BASE_URL}/detection_pipeline/out/annotated/{cam_id}.mp4"
    return None


def _clip_url(fname: str) -> str:
    if VIDEO_BASE_URL:
        # Original raw clips live under resources/CCTV Footage*/
        return f"{VIDEO_BASE_URL}/resources/CCTV%20Footage-20260529T160731Z-3-00144614ea/CCTV%20Footage/{fname}"
    return f"/clips/{fname}"


@router.get("/cameras")
def cameras(db: Session = Depends(get_db)):
    """Return camera list, clip URL, and the first/last event ts per camera
    so the front-end can compute video offsets."""
    out = []
    for cam_id, fname in CAMERA_CLIPS.items():
        first, last = db.execute(
            select(func.min(models.Event.timestamp), func.max(models.Event.timestamp))
            .where(models.Event.camera_id == cam_id)
        ).one()
        out.append({
            "camera_id": cam_id,
            "clip_url": _clip_url(fname),
            "annotated_url": _annotated_url(cam_id),
            "first_event_ts": first.isoformat() + "Z" if first else None,
            "last_event_ts":  last.isoformat() + "Z" if last else None,
        })
    return {"store_id": STORE_ID, "cameras": out}


@router.get("/events")
def events(
    camera_id: Optional[str] = None,
    store_id: Optional[str] = None,
    since: Optional[datetime] = Query(None),
    limit: int = Query(2000, le=10000),
    db: Session = Depends(get_db),
):
    """Return events for sync-with-video. Each row includes offset_ms from
    the camera's first event timestamp so the player can show events at the
    right moment."""
    q = select(models.Event)
    if camera_id:
        q = q.where(models.Event.camera_id == camera_id)
    if store_id:
        q = q.where(models.Event.store_id == store_id)
    if since:
        ts = since.replace(tzinfo=None) if since.tzinfo else since
        q = q.where(models.Event.timestamp >= ts)
    q = q.order_by(models.Event.timestamp).limit(limit)
    rows = db.execute(q).scalars().all()

    # offset reference per camera = the camera's earliest event in DB
    cam_first: dict[str, datetime] = {}
    for r in rows:
        cam_first.setdefault(r.camera_id, r.timestamp)
    if camera_id and camera_id not in cam_first:
        first_db = db.execute(
            select(func.min(models.Event.timestamp)).where(models.Event.camera_id == camera_id)
        ).scalar()
        if first_db:
            cam_first[camera_id] = first_db

    out = []
    for r in rows:
        base = cam_first.get(r.camera_id)
        offset_ms = int((r.timestamp - base).total_seconds() * 1000) if base else 0
        out.append({
            "event_id": r.event_id,
            "store_id": r.store_id,
            "camera_id": r.camera_id,
            "visitor_id": r.visitor_id,
            "event_type": r.event_type,
            "timestamp": r.timestamp.isoformat() + "Z",
            "offset_ms": offset_ms,
            "zone_id": r.zone_id,
            "dwell_ms": r.dwell_ms,
            "is_staff": r.is_staff,
            "confidence": r.confidence,
            "metadata": r.meta,
        })
    return {"count": len(out), "events": out}
