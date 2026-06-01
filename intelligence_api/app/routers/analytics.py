from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import schemas
from ..db import get_db
from ..services import analytics

router = APIRouter(tags=["analytics"])


@router.get("/stores/{store_id}/metrics", response_model=schemas.MetricsResponse)
def metrics(store_id: str, db: Session = Depends(get_db)):
    return analytics.compute_metrics(db, store_id)


@router.get("/stores/{store_id}/funnel", response_model=schemas.FunnelResponse)
def funnel(store_id: str, db: Session = Depends(get_db)):
    return analytics.compute_funnel(db, store_id)


@router.get("/stores/{store_id}/heatmap", response_model=schemas.HeatmapResponse)
def heatmap(store_id: str, db: Session = Depends(get_db)):
    return analytics.compute_heatmap(db, store_id)


@router.get("/stores/{store_id}/anomalies", response_model=schemas.AnomaliesResponse)
def anomalies(store_id: str, db: Session = Depends(get_db)):
    return analytics.compute_anomalies(db, store_id)
