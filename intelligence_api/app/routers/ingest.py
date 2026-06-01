from __future__ import annotations
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from .. import models, schemas
from ..db import get_db

router = APIRouter(prefix="/events", tags=["ingest"])
log = logging.getLogger("ingest")


@router.post("/ingest", response_model=schemas.IngestResponse)
def ingest(req: schemas.IngestRequest, db: Session = Depends(get_db)):
    if len(req.events) == 0:
        raise HTTPException(400, "events array is empty")
    if len(req.events) > 500:
        raise HTTPException(413, "batch exceeds 500 events")

    accepted = 0
    duplicates = 0
    failed: list[schemas.IngestFailure] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    try:
        existing = {
            r[0] for r in db.execute(
                models.Event.__table__.select().with_only_columns(models.Event.event_id).where(
                    models.Event.event_id.in_([e.event_id for e in req.events])
                )
            ).all()
        }
    except OperationalError as ex:
        log.error("db unavailable: %s", ex)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "database unavailable")

    for i, e in enumerate(req.events):
        if e.event_id in existing:
            duplicates += 1
            continue
        try:
            ts = e.timestamp
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            row = models.Event(
                event_id=e.event_id,
                store_id=e.store_id,
                camera_id=e.camera_id,
                visitor_id=e.visitor_id,
                event_type=e.event_type,
                timestamp=ts,
                zone_id=e.zone_id,
                dwell_ms=e.dwell_ms,
                is_staff=e.is_staff,
                confidence=e.confidence,
                meta=e.metadata.model_dump(),
                created_at=now,
            )
            db.add(row)
            db.flush()
            accepted += 1
            existing.add(e.event_id)
        except IntegrityError:
            db.rollback()
            duplicates += 1
        except Exception as ex:
            db.rollback()
            failed.append(schemas.IngestFailure(index=i, reason=str(ex)[:200]))

    try:
        db.commit()
    except OperationalError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "database unavailable")

    log.info("ingest_done", extra={"event_count": len(req.events)})
    return schemas.IngestResponse(accepted=accepted, duplicates=duplicates, failed=failed)
