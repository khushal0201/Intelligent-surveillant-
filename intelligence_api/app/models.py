from __future__ import annotations
from sqlalchemy import (
    Column, String, Integer, Boolean, Float, DateTime, JSON, Index
)
from .db import Base


class Event(Base):
    __tablename__ = "events"

    event_id   = Column(String, primary_key=True)
    store_id   = Column(String, nullable=False, index=True)
    camera_id  = Column(String, nullable=False, index=True)
    visitor_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    timestamp  = Column(DateTime, nullable=False, index=True)
    zone_id    = Column(String, nullable=True, index=True)
    dwell_ms   = Column(Integer, nullable=False, default=0)
    is_staff   = Column(Boolean, nullable=False, default=False, index=True)
    confidence = Column(Float, nullable=False, default=0.0)
    meta       = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_store_visitor", "store_id", "visitor_id"),
    )


class Purchase(Base):
    """POS rows correlated to visitor sessions (optional)."""
    __tablename__ = "purchases"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    store_id    = Column(String, nullable=False, index=True)
    visitor_id  = Column(String, nullable=True, index=True)
    txn_id      = Column(String, nullable=False, unique=True)
    amount      = Column(Float, nullable=False, default=0.0)
    timestamp   = Column(DateTime, nullable=False, index=True)
