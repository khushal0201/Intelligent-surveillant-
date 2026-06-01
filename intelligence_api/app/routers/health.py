from __future__ import annotations
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, distinct
from sqlalchemy.orm import Session
from .. import models, schemas
from ..db import get_db

router = APIRouter(tags=["health"])

STALE_AFTER_SEC = 600


@router.get("/health", response_model=schemas.HealthResponse)
def health(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(select(1)).scalar()
    except Exception:
        db_ok = False

    stores: list[schemas.StoreFeedHealth] = []
    if db_ok:
        ids = [r[0] for r in db.execute(select(distinct(models.Event.store_id))).all()]
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for sid in ids:
            last = db.execute(
                select(func.max(models.Event.timestamp)).where(models.Event.store_id == sid)
            ).scalar()
            lag = (now - last).total_seconds() if last else None
            stores.append(schemas.StoreFeedHealth(
                store_id=sid, last_event_at=last,
                lag_seconds=lag,
                stale_feed=bool(lag is not None and lag > STALE_AFTER_SEC),
            ))
    return schemas.HealthResponse(
        status="ok" if db_ok else "degraded",
        db_ok=db_ok, stores=stores,
    )
