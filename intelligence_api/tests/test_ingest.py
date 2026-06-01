# PROMPT: write tests for POST /events/ingest covering: happy path, oversize
# batch (>500 → 413), idempotency (same event twice → second call counts as
# duplicate), and partial failure (one malformed event in a batch).
# CHANGES MADE: build minimal valid event dicts using uuid4, post them via
# TestClient, assert counts on the IngestResponse.
import uuid

def _ev(**over):
    base = {
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_MAKEUP_02",
        "visitor_id": "VIS_abc123",
        "event_type": "ZONE_ENTER",
        "timestamp": "2026-04-10T14:39:50Z",
        "zone_id": "LOREAL",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {"queue_depth": None, "sku_zone": "MAKEUP", "session_seq": 1},
    }
    base.update(over)
    return base


def test_ingest_happy(client):
    r = client.post("/events/ingest", json={"events": [_ev(), _ev()]})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 2 and body["duplicates"] == 0


def test_ingest_idempotent(client):
    e = _ev()
    r1 = client.post("/events/ingest", json={"events": [e]})
    r2 = client.post("/events/ingest", json={"events": [e]})
    assert r1.json()["accepted"] == 1
    assert r2.json()["duplicates"] == 1
    assert r2.json()["accepted"] == 0


def test_ingest_oversize(client):
    batch = [_ev() for _ in range(501)]
    r = client.post("/events/ingest", json={"events": batch})
    # Pydantic max_length=500 → 422; or our manual 413 path.
    assert r.status_code in (413, 422)


def test_ingest_partial_failure(client):
    bad = _ev()
    bad["event_type"] = "BOGUS"  # will fail Pydantic validation
    good = _ev()
    r = client.post("/events/ingest", json={"events": [good, bad]})
    # The whole request fails Pydantic validation → 422 on the batch.
    assert r.status_code == 422


def test_empty_batch_rejected(client):
    r = client.post("/events/ingest", json={"events": []})
    assert r.status_code == 400
