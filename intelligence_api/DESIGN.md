# DESIGN.md

A plain-language tour of how the system fits together. The goal is for a new
engineer to read this once and be able to add a feature without surprises.

---

## 1. End-to-end picture

```
              ┌──────────────────────┐
   CCTV mp4 → │  detection_pipeline  │ → events_*.jsonl
              │  (offline, CPU only) │   (schema-validated)
              └──────────────────────┘             │
                                                   ▼
                                  ┌────────────────────────────────┐
                                  │  bootstrap_from_jsonl()        │
                                  │  POST /events/ingest (batches) │
                                  └────────────────────────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │ events table     │  SQLite default
                                          │ purchases table  │  Postgres via env
                                          └──────────────────┘
                                                   │
       ┌──────────────┬──────────────┬─────────────┼──────────────┬──────────────┐
       ▼              ▼              ▼             ▼              ▼              ▼
   /metrics       /funnel        /heatmap     /anomalies       /health       /dashboard/*
   (sessions,    (drop-off      (intensity,   (3 detectors)    (stale-feed   (camera list,
    dwell)       per stage)     low-conf)                       per store)    sync events)

                           ┌─────────────────────────────────┐
                           │  Tailwind dashboard (no build)  │
                           │ • 5 annotated H.264 mp4 tiles   │
                           │ • events synced via offset_ms   │
                           │ • live event toasts on the      │
                           │   playing video                 │
                           │ • upload form → /uploads        │
                           └─────────────────────────────────┘
```

The pipeline runs **offline** on the laptop. Its only output the API
consumes is `detection_pipeline/out/events_full.jsonl`. That file is the
contract — any pipeline (the OpenCLIP one we ship, a future YOLO-World one,
or human-labelled events) can populate it.

---

## 2. Why the boundary is a JSONL file

Putting a file between the two halves was deliberate:

- **Replayable.** Wipe `intel.db`, restart the API, and the same events
  come back. Dev loop is one delete.
- **Inspectable.** During the staff-classifier flicker bug, being able to
  `grep visitor_id` in plain JSON saved hours over a binary format.
- **Schema-checkable on both sides.** `detection_pipeline/schema/event_schema.json`
  is the single source of truth, validated at write time in the pipeline
  and mirrored by Pydantic v2 on the API. A change has to update both —
  that's a feature, not a bug.

Cost: events lag actual frames by however long the pipeline takes
(offline, not realtime). Acceptable for a retail-analytics dashboard.

---

## 3. Sessions are derived, not stored

A session = all events for one `visitor_id` inside the analytics window.
We do **not** store sessions; we recompute them per request.

The JSONL is the source of truth. If we materialised sessions we would
have to re-materialise on every replay. ENTRY/EXIT pairs delineate visit
segments, but the visitor remains a single session — the brief explicitly
asks for re-entries not to double-count. We accept that this conflates
true returning visits within the same day; that is the intended semantic.

---

## 4. Idempotency

`event_id` is the table primary key (a deterministic UUID computed from
`visitor_id + event_type + timestamp_bucket`). Ingest does one
`SELECT event_id WHERE event_id IN (…)` to find duplicates, then
bulk-inserts the rest. The pipeline can re-emit an event from a re-run
and the API silently dedupes. We catch `IntegrityError` per row as
defence-in-depth against concurrent ingest writers.

---

## 5. Live dashboard sync

The trickiest part of the UI: making the event list line up with the
playing video.

- `GET /dashboard/cameras` → camera list with `clip_url`, `annotated_url`
  (an H.264 transcode), and the **first event timestamp** in DB.
- `GET /dashboard/events?camera_id=X` → events with `offset_ms`, computed
  as `event.timestamp - first_event_ts_for_camera`.

The front-end listens to `<video> timeupdate` and:
1. highlights any event in the list whose `offset_ms` is within ±1500 ms
   of `currentTime * 1000`,
2. pushes a transient toast on the video for the most recent fired event
   (deduped by `event_id`),
3. paints a tiny mark on the timeline strip per event, color-coded by
   staff/customer.

Because both halves use the camera's earliest event as the zero point,
they drift by at most one stride (≈200 ms) — invisible to the viewer.

---

## 6. Browser-friendly video

OpenCV's `mp4v` fourcc is MPEG-4 Part 2, which Chrome and Safari refuse
to play. The pipeline first writes `mp4v` (the codec cv2 ships under an
Apache-friendly license) then post-processes via `imageio-ffmpeg`'s
bundled `libx264` to produce yuv420p H.264 with `+faststart`. The
transcode hook lives in `detection_pipeline/transcode_to_h264.py` and is
invoked automatically at the end of every `run.py` per-camera write.

---

## 7. Failure modes

| Mode | Behaviour |
|------|-----------|
| DB unavailable | 503, structured JSON error, no stack trace leaked |
| Pydantic validation fails | 422 with field-level messages |
| Pipeline subprocess fails on `/uploads` | job state `failed` with stderr tail |
| Empty store | endpoints return zeros (not 404) so dashboards don't break |
| Stale feed (no new events for N min) | `/health` flags `stale_feed: true` |
| Annotated mp4 missing | `/dashboard/cameras` returns `annotated_url=null` and the front-end falls back to the raw clip — or to `VIDEO_BASE_URL` on HF Space |

---

## 8. Logging

`TraceMiddleware` assigns a 12-char hex trace id, stashes it in a
`ContextVar`, and emits one JSON line per request:

```json
{"ts":"…","level":"INFO","logger":"api","msg":"request",
 "trace_id":"d23000806163","store_id":"ST1008",
 "endpoint":"/stores/ST1008/metrics","latency_ms":68.46,
 "status_code":200,"method":"GET"}
```

The same trace id is echoed in the `x-trace-id` response header so the
front-end can quote it in bug reports.

---

## 9. Deployment shape

Two containers, one Dockerfile each:

- `intelligence_api/Dockerfile` — local/Compose build, ships annotated
  mp4s and the pipeline alongside the API.
- `Dockerfile` (repo root) — Hugging Face Space build, **API only**. Heavy
  assets (videos, model weights) are excluded; the front-end resolves
  video URLs against `VIDEO_BASE_URL` (pointed at GitHub raw on the
  Space) so the Space stays well under HF's 10 MiB-per-file ceiling.

---

## AI-Assisted Decisions

Three places where an LLM materially shaped the design, and how the
exchange landed.

### A. Sessions: derived view vs. stored table

- **AI suggested.** A first-class `sessions` table with `start_at`,
  `end_at`, `visitor_id`, computed by an event-driven background job,
  plus a session-level cache layer.
- **We chose.** No sessions table — recompute on each `/metrics` request.
- **Why.** Re-ingestion is a normal operation (replays, schema fixes, new
  detection model) and any materialised view would have to be
  invalidated. Keeping the JSONL as the single source of truth and
  computing sessions lazily means we have **no migration to write** when
  event semantics change. Performance is fine because the SQLite query
  plan uses the `(visitor_id, timestamp)` composite index. We agreed
  with the AI's reasoning about scale, but rejected the suggestion at
  our actual scale (≤10⁵ events/day per store).

### B. Async SQLAlchemy vs. sync + threadpool

- **AI suggested.** `async def` everywhere, `AsyncSession`, `aiosqlite`.
- **We chose.** Sync SQLAlchemy 2.0, FastAPI's default threadpool offload
  for IO-bound routes.
- **Why.** SQLite is single-writer; async around a single mutex buys us
  nothing. Sync sessions also keep the test surface minimal — pytest +
  `TestClient` "just works" without an event loop. The override here was
  intentional: AI was optimising for an idealised concurrent workload
  that doesn't match SQLite's actual concurrency model.

### C. Dashboard framework: React/Next vs. plain Tailwind + JS

- **AI suggested.** React + Vite + tRPC for end-to-end type safety;
  later in the project, Next.js App Router + shadcn/ui for
  "production-ready components".
- **We chose.** Plain HTML + Tailwind via CDN + a single `app.js` ES
  module, served straight from FastAPI's static mount. No build step,
  no `node_modules`.
- **Why.** Single-container deploy is a hard requirement (HF Space,
  Docker Compose). A bundler would have introduced a Node toolchain
  into the Python container or a second build stage, neither of which
  buy us anything for ~600 LoC of UI. We **did** take the AI's
  suggestion to use Tailwind CDN + Inter / JetBrains Mono — that part
  was clearly the right call and lifted the look beyond what we'd have
  written by hand.
