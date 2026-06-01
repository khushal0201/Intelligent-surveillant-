# DESIGN.md

## Architecture

```
              ┌────────────────────┐
   CCTV mp4 → │ detection_pipeline │ → JSONL (schema-validated)
              └────────────────────┘            │
                                                ▼
                                  ┌──────────────────────────┐
                                  │  POST /events/ingest     │
                                  │  (idempotent, batch ≤500)│
                                  └──────────────────────────┘
                                                │
                                                ▼
                                       ┌──────────────┐
                                       │ events table │   SQLite (default)
                                       │ purchases    │   PostgreSQL via env
                                       └──────────────┘
                                                │
       ┌──────────────────────────┬─────────────┼────────────────┬─────────────────┐
       ▼                          ▼             ▼                ▼                 ▼
   /metrics                  /funnel        /heatmap         /anomalies         /health
   (sessions, dwell)         (drop-off)     (intensity)      (3 detectors)      (stale-feed)

                           ┌──────────────────────────┐
                           │ Dashboard (vanilla JS)   │
                           │ • 5 video tiles          │
                           │ • events synced via      │
                           │   offset_ms              │
                           │ • upload → /uploads      │
                           └──────────────────────────┘
```

### Request flow

1. Detection pipeline writes one event per state transition. The same event is
   re-emitted across restarts because `event_id` is a deterministic UUID; the
   ingest endpoint deduplicates on the primary key.
2. Analytics derive **sessions** lazily from events (visitor_id within window).
   Re-entries don't double-count entries because we group by visitor_id.
3. Anomalies are computed on demand from the latest event timestamp, not
   wall-clock — important for replaying recorded footage.

### Sessions

A session = all events for one `visitor_id` inside a window. ENTRY/EXIT pairs
delineate segments but the visitor remains a single session for analytics. We
deliberately accept that this conflates true returning visits within the same
day; the brief specifies "re-entries don't double-count".

### Idempotency

`event_id` is the primary key. Ingest does a single `SELECT … WHERE id IN (…)`
to find existing IDs, then bulk-inserts the rest. `IntegrityError` is caught
per-row as a defence-in-depth (race between concurrent ingests).

### Failure modes
| Mode | Behaviour |
|------|-----------|
| DB unavailable | 503, structured error, no stack trace |
| Pydantic validation | 422 with field-level errors |
| Pipeline subprocess fails on upload | job state `failed` with stderr tail |
| Empty store | endpoints return zeros (not 404) — easier for dashboards |

### Logging

`TraceMiddleware` assigns a 12-char trace id, sets it in a `ContextVar`, and
emits one JSON line per request with `endpoint, method, latency_ms,
status_code, store_id, trace_id`. The same trace id is echoed in the
`x-trace-id` response header.

## AI-Assisted Decisions

1. **Sessions from events, not a separate table.** AI suggested an explicit
   sessions table with start/end columns. Rejected because the JSONL is the
   source of truth and a derived projection is simpler to keep correct under
   re-ingestion. We pay a small per-query cost in exchange for zero migration
   when event semantics change.
2. **Sync SQLAlchemy + FastAPI threadpool.** AI initially scaffolded async
   SQLAlchemy. Rejected for SQLite (one writer) and to keep the test surface
   small; sync sessions are sufficient at expected QPS and easier for new
   contributors to reason about.
3. **Vanilla JS dashboard.** AI proposed React + Vite. Rejected because we
   wanted the API container to ship the dashboard with no build step. The
   trade-off is no component reuse, but the surface is small (≈200 LoC).
