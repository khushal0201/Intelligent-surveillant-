# PROMPT: tests for analytics endpoints — staff are excluded from metrics, a
# re-entering visitor is not double-counted in the funnel, and the heatmap
# flags low confidence below 20 sessions.
# CHANGES MADE: seed events directly through /events/ingest then assert
# the computed metrics/funnel/heatmap.
import uuid

def _e(visitor, evtype, **over):
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_MAKEUP_02",
        "visitor_id": visitor,
        "event_type": evtype,
        "timestamp": over.pop("ts", "2026-04-10T14:39:50Z"),
        "zone_id": over.pop("zone", None),
        "dwell_ms": over.pop("dwell", 0),
        "is_staff": over.pop("is_staff", False),
        "confidence": 0.9,
        "metadata": {
            "queue_depth": over.pop("queue_depth", None),
            "sku_zone": over.pop("sku_zone", None),
            "session_seq": 1,
        },
    }


def _ingest(client, events):
    r = client.post("/events/ingest", json={"events": events})
    assert r.status_code == 200, r.text


def test_metrics_excludes_staff(client):
    _ingest(client, [
        _e("VIS_aaaaaa", "ENTRY"),
        _e("VIS_aaaaaa", "ZONE_ENTER", zone="LOREAL"),
        _e("VIS_aaaaaa", "ZONE_EXIT",  zone="LOREAL", dwell=4000),
        _e("VIS_bbbbbb", "ENTRY", is_staff=True),
        _e("VIS_bbbbbb", "ZONE_ENTER", zone="LOREAL", is_staff=True),
    ])
    m = client.get("/stores/ST1008/metrics").json()
    assert m["unique_visitors"] == 1
    assert m["sessions"] == 1
    assert any(z["zone_id"] == "LOREAL" for z in m["avg_dwell_per_zone"])


def test_funnel_no_double_count_on_reentry(client):
    _ingest(client, [
        _e("VIS_111111", "ENTRY"),
        _e("VIS_111111", "EXIT"),
        _e("VIS_111111", "REENTRY"),
        _e("VIS_111111", "ZONE_ENTER", zone="LOREAL"),
    ])
    f = client.get("/stores/ST1008/funnel").json()
    entry = next(s for s in f["stages"] if s["stage"] == "ENTRY")
    assert entry["count"] == 1


def test_heatmap_low_confidence(client):
    _ingest(client, [
        _e("VIS_222222", "ZONE_ENTER", zone="LOREAL"),
        _e("VIS_222222", "ZONE_EXIT", zone="LOREAL", dwell=2000),
    ])
    h = client.get("/stores/ST1008/heatmap").json()
    assert h["data_confidence"] == "low"
    assert h["zones"][0]["zone_id"] == "LOREAL"
    assert h["zones"][0]["intensity"] == 100.0


def test_health_reports_stale(client):
    # No events → no stores → status ok, empty list.
    r = client.get("/health").json()
    assert r["status"] == "ok"
    _ingest(client, [_e("VIS_333333", "ENTRY")])
    r2 = client.get("/health").json()
    # Old timestamp (2026 in past) → stale_feed True (lag > 600s).
    assert r2["stores"][0]["stale_feed"] is True


def test_anomalies_dead_zone(client):
    # Old visit to LOREAL, then a much later (>30 min) visit elsewhere → LOREAL is dead.
    _ingest(client, [
        _e("VIS_444444", "ZONE_ENTER", zone="LOREAL", ts="2026-04-10T14:00:00Z"),
        _e("VIS_555555", "ZONE_ENTER", zone="MAYBELLINE", ts="2026-04-10T15:30:00Z"),
    ])
    a = client.get("/stores/ST1008/anomalies").json()
    types = {x["type"] for x in a["anomalies"]}
    assert "DEAD_ZONE" in types
