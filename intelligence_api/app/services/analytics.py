"""Analytics computations on the events table."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from .. import models, schemas


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _today_window(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    n = now or _utcnow()
    start = datetime(n.year, n.month, n.day)
    return start, n


def _window_for_data(db: Session, store_id: str) -> Tuple[datetime, datetime]:
    """Use the actual data window (max event ts) for the given store, so
    short demo clips do not produce empty 'today' windows."""
    last = db.execute(
        select(func.max(models.Event.timestamp)).where(models.Event.store_id == store_id)
    ).scalar()
    if last is None:
        return _today_window()
    end = last
    start = datetime(end.year, end.month, end.day)
    return start, end


# ---------------- session derivation ----------------

def _sessions(db: Session, store_id: str, t0: datetime, t1: datetime,
              include_staff: bool = False) -> dict[str, dict]:
    """Build sessions from events. A session = one visitor's events between
    ENTRY (or first event) and EXIT (or last). Re-entries don't double-count."""
    q = select(models.Event).where(
        models.Event.store_id == store_id,
        models.Event.timestamp >= t0,
        models.Event.timestamp <= t1,
    ).order_by(models.Event.timestamp)
    if not include_staff:
        q = q.where(models.Event.is_staff == False)  # noqa: E712
    rows = db.execute(q).scalars().all()

    sessions: dict[str, dict] = {}
    for e in rows:
        s = sessions.setdefault(e.visitor_id, {
            "visitor_id": e.visitor_id,
            "first_ts": e.timestamp,
            "last_ts": e.timestamp,
            "zones": set(),
            "zone_dwells": defaultdict(int),
            "queued": False,
            "abandoned": False,
            "billed": False,
            "events": 0,
        })
        s["last_ts"] = e.timestamp
        s["events"] += 1
        if e.event_type in ("ZONE_ENTER", "ZONE_DWELL", "ZONE_EXIT") and e.zone_id:
            s["zones"].add(e.zone_id)
            if e.event_type == "ZONE_EXIT" and e.dwell_ms:
                s["zone_dwells"][e.zone_id] += e.dwell_ms
        elif e.event_type == "BILLING_QUEUE_JOIN":
            s["queued"] = True
        elif e.event_type == "BILLING_QUEUE_ABANDON":
            s["abandoned"] = True
    return sessions


# ---------------- metrics ----------------

def compute_metrics(db: Session, store_id: str) -> schemas.MetricsResponse:
    t0, t1 = _window_for_data(db, store_id)
    sessions = _sessions(db, store_id, t0, t1)

    purchases = db.execute(
        select(func.count(models.Purchase.id)).where(
            models.Purchase.store_id == store_id,
            models.Purchase.timestamp >= t0,
            models.Purchase.timestamp <= t1,
        )
    ).scalar() or 0

    unique_visitors = len(sessions)
    sess_count = unique_visitors  # 1 visitor = 1 session in our model
    conv = (purchases / sess_count) if sess_count > 0 else 0.0

    # Average dwell per zone across sessions that visited that zone.
    zone_totals: dict[str, list[int]] = defaultdict(list)
    for s in sessions.values():
        for z, d in s["zone_dwells"].items():
            if d > 0:
                zone_totals[z].append(d)
    avg_dwell = [
        schemas.ZoneDwell(
            zone_id=z, avg_dwell_ms=sum(v) / len(v), visits=len(v)
        )
        for z, v in sorted(zone_totals.items(), key=lambda kv: -sum(kv[1]))
    ]

    # Queue depth: latest BILLING_* event's metadata.queue_depth.
    last_q = db.execute(
        select(models.Event).where(
            models.Event.store_id == store_id,
            models.Event.event_type.in_(["BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"]),
        ).order_by(models.Event.timestamp.desc()).limit(1)
    ).scalar_one_or_none()
    queue_now = (last_q.meta or {}).get("queue_depth") if last_q else None

    q_rows = db.execute(
        select(models.Event.meta).where(
            models.Event.store_id == store_id,
            models.Event.event_type == "BILLING_QUEUE_JOIN",
            models.Event.timestamp >= t0,
            models.Event.timestamp <= t1,
        )
    ).scalars().all()
    q_depths = [m.get("queue_depth") for m in q_rows if m and m.get("queue_depth") is not None]
    queue_avg = (sum(q_depths) / len(q_depths)) if q_depths else 0.0

    queued_sessions = [s for s in sessions.values() if s["queued"]]
    abandoned = [s for s in queued_sessions if s["abandoned"]]
    abandon_rate = (len(abandoned) / len(queued_sessions)) if queued_sessions else 0.0

    return schemas.MetricsResponse(
        store_id=store_id,
        window_start=t0,
        window_end=t1,
        unique_visitors=unique_visitors,
        sessions=sess_count,
        purchases=purchases,
        conversion_rate=round(conv, 4),
        avg_dwell_per_zone=avg_dwell,
        queue_depth_now=queue_now,
        queue_depth_avg=round(queue_avg, 2),
        abandonment_rate=round(abandon_rate, 4),
    )


# ---------------- funnel ----------------

def compute_funnel(db: Session, store_id: str) -> schemas.FunnelResponse:
    t0, t1 = _window_for_data(db, store_id)
    sessions = _sessions(db, store_id, t0, t1)

    purchases_by_visitor = {
        r[0] for r in db.execute(
            select(models.Purchase.visitor_id).where(
                models.Purchase.store_id == store_id,
                models.Purchase.timestamp >= t0,
                models.Purchase.timestamp <= t1,
                models.Purchase.visitor_id.isnot(None),
            )
        ).all()
    }

    entry = len(sessions)
    zone_visit = sum(1 for s in sessions.values() if s["zones"])
    queued = sum(1 for s in sessions.values() if s["queued"])
    purchased = sum(1 for vid in sessions if vid in purchases_by_visitor)

    raw = [("ENTRY", entry), ("ZONE_VISIT", zone_visit),
           ("BILLING_QUEUE", queued), ("PURCHASE", purchased)]
    stages: List[schemas.FunnelStage] = []
    prev = None
    for name, count in raw:
        drop = 0.0 if prev in (None, 0) else round((1 - count / prev) * 100, 2)
        stages.append(schemas.FunnelStage(stage=name, count=count, drop_off_pct=drop))
        prev = count
    return schemas.FunnelResponse(store_id=store_id, stages=stages)


# ---------------- heatmap ----------------

def compute_heatmap(db: Session, store_id: str) -> schemas.HeatmapResponse:
    t0, t1 = _window_for_data(db, store_id)
    sessions = _sessions(db, store_id, t0, t1)
    zone_visits: dict[str, int] = defaultdict(int)
    zone_dwells: dict[str, list[int]] = defaultdict(list)
    for s in sessions.values():
        for z in s["zones"]:
            zone_visits[z] += 1
            if s["zone_dwells"].get(z):
                zone_dwells[z].append(s["zone_dwells"][z])

    max_v = max(zone_visits.values(), default=1)
    entries: List[schemas.HeatmapEntry] = []
    for z, v in zone_visits.items():
        avg_d = (sum(zone_dwells[z]) / len(zone_dwells[z])) if zone_dwells[z] else 0.0
        entries.append(schemas.HeatmapEntry(
            zone_id=z, visits=v, avg_dwell_ms=avg_d,
            intensity=round(v / max_v * 100, 2),
        ))
    entries.sort(key=lambda e: -e.intensity)
    confidence = "low" if len(sessions) < 20 else "ok"
    return schemas.HeatmapResponse(
        store_id=store_id,
        sessions_in_window=len(sessions),
        data_confidence=confidence,
        zones=entries,
    )


# ---------------- anomalies ----------------

def compute_anomalies(db: Session, store_id: str) -> schemas.AnomaliesResponse:
    out: List[schemas.Anomaly] = []
    t1 = db.execute(
        select(func.max(models.Event.timestamp)).where(models.Event.store_id == store_id)
    ).scalar()
    if t1 is None:
        return schemas.AnomaliesResponse(store_id=store_id, anomalies=[])

    # Queue spike: avg queue_depth in last 5 min > 1.5x avg over last 30 min.
    win_short = t1 - timedelta(minutes=5)
    win_long = t1 - timedelta(minutes=30)
    q_short = db.execute(select(models.Event.meta).where(
        models.Event.store_id == store_id,
        models.Event.event_type == "BILLING_QUEUE_JOIN",
        models.Event.timestamp >= win_short,
    )).scalars().all()
    q_long = db.execute(select(models.Event.meta).where(
        models.Event.store_id == store_id,
        models.Event.event_type == "BILLING_QUEUE_JOIN",
        models.Event.timestamp >= win_long,
    )).scalars().all()

    def _avg(rows):
        vs = [m.get("queue_depth") for m in rows if m and m.get("queue_depth") is not None]
        return (sum(vs) / len(vs)) if vs else 0.0

    s_avg, l_avg = _avg(q_short), _avg(q_long)
    if s_avg > 0 and s_avg > 1.5 * max(l_avg, 1.0):
        sev = "CRITICAL" if s_avg > 5 else "WARN"
        out.append(schemas.Anomaly(
            type="QUEUE_SPIKE", severity=sev,
            message=f"Billing queue averaging {s_avg:.1f} (vs {l_avg:.1f}/30m)",
            suggested_action="Open additional billing counter or radio backup staff",
            detected_at=t1,
            context={"avg_5m": s_avg, "avg_30m": l_avg},
        ))

    # Conversion drop vs 7-day baseline.
    today_start = datetime(t1.year, t1.month, t1.day)
    today_metrics = compute_metrics(db, store_id)
    base_t0 = today_start - timedelta(days=7)
    base_sessions = _sessions(db, store_id, base_t0, today_start)
    base_purch = db.execute(select(func.count(models.Purchase.id)).where(
        models.Purchase.store_id == store_id,
        models.Purchase.timestamp >= base_t0,
        models.Purchase.timestamp < today_start,
    )).scalar() or 0
    base_conv = (base_purch / len(base_sessions)) if base_sessions else 0.0
    if base_conv > 0 and today_metrics.conversion_rate < 0.7 * base_conv:
        out.append(schemas.Anomaly(
            type="CONVERSION_DROP", severity="WARN",
            message=f"Conversion {today_metrics.conversion_rate:.2%} vs 7d avg {base_conv:.2%}",
            suggested_action="Review staff coverage and recent promotions",
            detected_at=t1,
            context={"today": today_metrics.conversion_rate, "baseline": base_conv},
        ))

    # Dead zones: zones with no events in last 30 min but had visits before.
    active_recent = {r[0] for r in db.execute(select(models.Event.zone_id).where(
        models.Event.store_id == store_id,
        models.Event.timestamp >= t1 - timedelta(minutes=30),
        models.Event.zone_id.isnot(None),
    )).all()}
    all_zones = {r[0] for r in db.execute(select(models.Event.zone_id).where(
        models.Event.store_id == store_id,
        models.Event.zone_id.isnot(None),
    )).all()}
    dead = sorted(all_zones - active_recent)
    for z in dead:
        out.append(schemas.Anomaly(
            type="DEAD_ZONE", severity="INFO",
            message=f"No visits to {z} in last 30 minutes",
            suggested_action=f"Check shelf stocking and signage at {z}",
            detected_at=t1,
            context={"zone_id": z},
        ))

    return schemas.AnomaliesResponse(store_id=store_id, anomalies=out)


# ---------------- demographics ----------------

_AGE_BUCKET_ORDER = ["child", "teen", "20s", "30s", "40s", "50s", "60+"]


def compute_demographics(db: Session, store_id: str) -> schemas.DemographicsResponse:
    """Aggregate gender + age buckets per non-staff visitor.

    Each visitor contributes once. We pick the most-recent non-null tag the
    pipeline stamped into ``metadata`` (final-pass values overwrite live ones
    because they're written last in the JSONL bootstrap order).
    """
    rows = db.execute(
        select(models.Event.visitor_id, models.Event.timestamp, models.Event.meta)
        .where(models.Event.store_id == store_id,
               models.Event.is_staff == False)  # noqa: E712
        .order_by(models.Event.timestamp)
    ).all()

    per_visitor_gender: dict[str, str] = {}
    per_visitor_age: dict[str, str] = {}
    all_visitors: set[str] = set()
    for vid, _ts, meta in rows:
        all_visitors.add(vid)
        if not meta:
            continue
        g = meta.get("gender")
        a = meta.get("age_bucket")
        if g:
            per_visitor_gender[vid] = g
        if a:
            per_visitor_age[vid] = a

    gender_counts: dict[str, int] = defaultdict(int)
    for g in per_visitor_gender.values():
        gender_counts[g] += 1

    age_counts: dict[str, int] = {b: 0 for b in _AGE_BUCKET_ORDER}
    for a in per_visitor_age.values():
        if a in age_counts:
            age_counts[a] += 1
        else:
            age_counts[a] = age_counts.get(a, 0) + 1

    classified = len(set(per_visitor_gender) | set(per_visitor_age))
    return schemas.DemographicsResponse(
        store_id=store_id,
        total_visitors=len(all_visitors),
        classified_visitors=classified,
        gender=dict(gender_counts),
        age_buckets=age_counts,
        age_bucket_order=_AGE_BUCKET_ORDER,
    )
