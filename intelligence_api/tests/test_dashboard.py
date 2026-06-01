# PROMPT: smoke-test dashboard helper endpoints return the camera list and
# events-with-offsets, ensuring the front-end contract is stable.
# CHANGES MADE: ingest one event then call /dashboard/cameras and
# /dashboard/events.
import uuid


def _ev():
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_MAKEUP_02",
        "visitor_id": "VIS_aaaaaa",
        "event_type": "ZONE_ENTER",
        "timestamp": "2026-04-10T14:39:50Z",
        "zone_id": "LOREAL",
        "dwell_ms": 0, "is_staff": False, "confidence": 0.9,
        "metadata": {"queue_depth": None, "sku_zone": "MAKEUP", "session_seq": 1},
    }


def test_cameras_endpoint(client):
    r = client.get("/dashboard/cameras").json()
    assert r["store_id"] == "ST1008"
    assert len(r["cameras"]) == 5
    assert all("clip_url" in c for c in r["cameras"])


def test_events_endpoint_offsets(client):
    client.post("/events/ingest", json={"events": [_ev()]})
    r = client.get("/dashboard/events?camera_id=CAM_MAKEUP_02").json()
    assert r["count"] == 1
    assert r["events"][0]["offset_ms"] == 0
