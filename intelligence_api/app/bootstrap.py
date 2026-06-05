"""Bootstrap: load existing JSONL files into the DB on startup."""
from __future__ import annotations
import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select, func
from .db import session_scope
from . import models
from .config import DEFAULT_EVENTS_GLOB

log = logging.getLogger("bootstrap")


def bootstrap_from_jsonl() -> int:
    paths = []
    for pattern in DEFAULT_EVENTS_GLOB.split(";"):
        pattern = pattern.strip()
        if not pattern:
            continue
        if Path(pattern).exists():
            paths.append(pattern)
        else:
            paths.extend(glob.glob(pattern))

    inserted = 0
    with session_scope() as db:
        existing_ids = set(
            db.execute(select(models.Event.event_id)).scalars().all()
        )
        log.info("bootstrap_start", extra={"event_count": len(existing_ids)})
        seen: set[str] = set(existing_ids)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        eid = r.get("event_id")
                        if not eid or eid in seen:
                            continue
                        seen.add(eid)
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
                            inserted += 1
                        except Exception as ex:
                            log.warning("bootstrap_skip_row %s", ex)
            except FileNotFoundError:
                continue
    log.info("bootstrap_done", extra={"event_count": inserted})
    return inserted
