from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Literal, Any
from pydantic import BaseModel, Field, ConfigDict

EventType = Literal[
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
]


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone:    Optional[str] = None
    session_seq: int = 0
    model_config = ConfigDict(extra="allow")


class EventIn(BaseModel):
    event_id:   str
    store_id:   str
    camera_id:  str
    visitor_id: str
    event_type: EventType
    timestamp:  datetime
    zone_id:    Optional[str] = None
    dwell_ms:   int = 0
    is_staff:   bool = False
    confidence: float = 0.0
    metadata:   EventMetadata


class IngestRequest(BaseModel):
    events: List[EventIn] = Field(..., max_length=500)


class IngestFailure(BaseModel):
    index: int
    reason: str


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    failed: List[IngestFailure]


class ZoneDwell(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visits: int


class MetricsResponse(BaseModel):
    store_id: str
    window_start: datetime
    window_end: datetime
    unique_visitors: int
    sessions: int
    purchases: int
    conversion_rate: float
    avg_dwell_per_zone: List[ZoneDwell]
    queue_depth_now: Optional[int]
    queue_depth_avg: float
    abandonment_rate: float


class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    stages: List[FunnelStage]


class HeatmapEntry(BaseModel):
    zone_id: str
    visits: int
    avg_dwell_ms: float
    intensity: float  # 0..100


class HeatmapResponse(BaseModel):
    store_id: str
    sessions_in_window: int
    data_confidence: Literal["low", "ok"]
    zones: List[HeatmapEntry]


class Anomaly(BaseModel):
    type: Literal["QUEUE_SPIKE", "CONVERSION_DROP", "DEAD_ZONE"]
    severity: Literal["INFO", "WARN", "CRITICAL"]
    message: str
    suggested_action: str
    detected_at: datetime
    context: dict[str, Any] = Field(default_factory=dict)


class AnomaliesResponse(BaseModel):
    store_id: str
    anomalies: List[Anomaly]


class StoreFeedHealth(BaseModel):
    store_id: str
    last_event_at: Optional[datetime]
    lag_seconds: Optional[float]
    stale_feed: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db_ok: bool
    stores: List[StoreFeedHealth]
