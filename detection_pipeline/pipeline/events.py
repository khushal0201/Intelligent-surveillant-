"""Per-camera event state machine.

Tracks each (camera_id, visitor_id) and emits the schema events. The Pipeline
combines per-camera emitters with a global Re-ID gallery and POS correlator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
import json
import uuid


DWELL_INTERVAL_MS = 30_000


def _new_event_id() -> str:
    return str(uuid.uuid4())


def _ts_iso(start_utc: datetime, offset_ms: int) -> str:
    t = start_utc + timedelta(milliseconds=offset_ms)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class VisitorState:
    visitor_id: str
    current_zone: Optional[str] = None
    zone_enter_ms: Optional[int] = None
    last_dwell_emit_ms: Optional[int] = None
    inside_store: bool = False
    last_entry_side: Optional[float] = None
    session_seq: int = 0
    last_seen_ms: int = 0
    pending_billing_exit_ms: Optional[int] = None  # for ABANDON correlation


class CameraEventEmitter:
    def __init__(self, store_id: str, camera_id: str, role: str,
                 sku_zone_map: dict, clip_start_utc: datetime,
                 sink, schema_validator=None):
        self.store_id = store_id
        self.camera_id = camera_id
        self.role = role  # entry / floor / billing
        self.sku_zone_map = sku_zone_map
        self.clip_start_utc = clip_start_utc
        self.sink = sink  # callable(event_dict)
        self.schema_validator = schema_validator
        self.visitors: dict[str, VisitorState] = {}

    # ------------------------------------------------------------------
    def _emit(self, *, visitor_id: str, event_type: str, ts_ms: int,
              zone_id: Optional[str], dwell_ms: int, is_staff: bool,
              confidence: float, queue_depth: Optional[int] = None,
              demographics: Optional[dict] = None) -> None:
        st = self._state(visitor_id)
        st.session_seq += 1
        sku_zone = self.sku_zone_map.get(zone_id) if zone_id else None
        meta = {
            "queue_depth": queue_depth,
            "sku_zone":    sku_zone,
            "session_seq": st.session_seq,
        }
        if demographics:
            meta["gender"] = demographics.get("gender")
            meta["age_bucket"] = demographics.get("age_bucket")
        evt = {
            "event_id":   _new_event_id(),
            "store_id":   self.store_id,
            "camera_id":  self.camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp":  _ts_iso(self.clip_start_utc, ts_ms),
            "zone_id":    zone_id,
            "dwell_ms":   int(dwell_ms),
            "is_staff":   bool(is_staff),
            "confidence": round(float(confidence), 4),
            "metadata":   meta,
        }
        if self.schema_validator is not None:
            try:
                self.schema_validator(evt)
            except Exception as e:  # noqa: BLE001
                print(f"[schema-warning] {e}: {json.dumps(evt)[:200]}")
        self.sink(evt)

    def _state(self, visitor_id: str) -> VisitorState:
        st = self.visitors.get(visitor_id)
        if st is None:
            st = VisitorState(visitor_id=visitor_id)
            self.visitors[visitor_id] = st
        return st

    # ------------------------------------------------------------------
    def on_entry_line(self, visitor_id: str, ts_ms: int, side: float,
                      is_staff: bool, confidence: float,
                      is_reentry: bool,
                      demographics: Optional[dict] = None) -> None:
        st = self._state(visitor_id)
        prev = st.last_entry_side
        st.last_entry_side = side
        st.last_seen_ms = ts_ms
        if prev is None:
            # first observation; treat positive side as inbound -> ENTRY
            if side > 0 and not st.inside_store:
                st.inside_store = True
                etype = "REENTRY" if is_reentry else "ENTRY"
                self._emit(visitor_id=visitor_id, event_type=etype,
                           ts_ms=ts_ms, zone_id=None, dwell_ms=0,
                           is_staff=is_staff, confidence=confidence,
                           demographics=demographics)
            return
        if prev <= 0 and side > 0 and not st.inside_store:
            st.inside_store = True
            etype = "REENTRY" if is_reentry else "ENTRY"
            self._emit(visitor_id=visitor_id, event_type=etype,
                       ts_ms=ts_ms, zone_id=None, dwell_ms=0,
                       is_staff=is_staff, confidence=confidence,
                       demographics=demographics)
        elif prev >= 0 and side < 0 and st.inside_store:
            st.inside_store = False
            self._emit(visitor_id=visitor_id, event_type="EXIT",
                       ts_ms=ts_ms, zone_id=None, dwell_ms=0,
                       is_staff=is_staff, confidence=confidence,
                       demographics=demographics)

    def on_zone_observation(self, visitor_id: str, ts_ms: int,
                            zone: Optional[str], is_staff: bool,
                            confidence: float,
                            queue_depth: Optional[int],
                            demographics: Optional[dict] = None) -> None:
        st = self._state(visitor_id)
        st.last_seen_ms = ts_ms

        # Zone change?
        if zone != st.current_zone:
            if st.current_zone is not None and st.zone_enter_ms is not None:
                dwell = ts_ms - st.zone_enter_ms
                self._emit(visitor_id=visitor_id, event_type="ZONE_EXIT",
                           ts_ms=ts_ms, zone_id=st.current_zone,
                           dwell_ms=dwell, is_staff=is_staff,
                           confidence=confidence,
                           demographics=demographics)
                if st.current_zone == "BILLING" and not is_staff:
                    st.pending_billing_exit_ms = ts_ms
            if zone is not None:
                self._emit(visitor_id=visitor_id, event_type="ZONE_ENTER",
                           ts_ms=ts_ms, zone_id=zone, dwell_ms=0,
                           is_staff=is_staff, confidence=confidence,
                           demographics=demographics)
                if zone == "BILLING" and queue_depth is not None and queue_depth > 0 and not is_staff:
                    self._emit(visitor_id=visitor_id,
                               event_type="BILLING_QUEUE_JOIN",
                               ts_ms=ts_ms, zone_id="BILLING", dwell_ms=0,
                               is_staff=is_staff, confidence=confidence,
                               queue_depth=queue_depth,
                               demographics=demographics)
            st.current_zone = zone
            st.zone_enter_ms = ts_ms if zone else None
            st.last_dwell_emit_ms = ts_ms if zone else None
            return

        # Same zone -> maybe DWELL
        if zone is None or st.zone_enter_ms is None:
            return
        elapsed = ts_ms - st.zone_enter_ms
        if elapsed < DWELL_INTERVAL_MS:
            return
        last = st.last_dwell_emit_ms or st.zone_enter_ms
        if ts_ms - last >= DWELL_INTERVAL_MS:
            self._emit(visitor_id=visitor_id, event_type="ZONE_DWELL",
                       ts_ms=ts_ms, zone_id=zone, dwell_ms=elapsed,
                       is_staff=is_staff, confidence=confidence,
                       demographics=demographics)
            st.last_dwell_emit_ms = ts_ms

    def consume_pending_billing_exits(self, pos_correlator,
                                      is_staff_lookup: dict[str, bool]) -> None:
        """Called at end of clip. Emits BILLING_QUEUE_ABANDON for visitors who
        left BILLING and had no nearby POS transaction."""
        for vid, st in self.visitors.items():
            if st.pending_billing_exit_ms is None:
                continue
            if is_staff_lookup.get(vid, False):
                st.pending_billing_exit_ms = None
                continue
            ts_iso = _ts_iso(self.clip_start_utc, st.pending_billing_exit_ms)
            t = datetime.strptime(ts_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc)
            if pos_correlator is None or not pos_correlator.had_transaction(t):
                self._emit(visitor_id=vid,
                           event_type="BILLING_QUEUE_ABANDON",
                           ts_ms=st.pending_billing_exit_ms,
                           zone_id="BILLING", dwell_ms=0,
                           is_staff=is_staff_lookup.get(vid, False),
                           confidence=0.5)
            st.pending_billing_exit_ms = None
