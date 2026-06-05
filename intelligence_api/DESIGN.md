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
- **Bootstrap is small and obvious.** One file per store, list them in
  `INTEL_BOOTSTRAP_GLOB`, done. Smoke runs and old experiments are kept
  out of the default glob so a fresh restart never re-ingests stale
  data; if you want them, set the env var.

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

## 10. Staff classifier — the part the user actually sees

Staff vs customer is the most user-visible label in the dashboard, so
it's worth describing how the math actually decides. The classifier
lives in `detection_pipeline/pipeline/staff.py` (`OpenCLIPPrototypeStaff`).

For each person crop the model produces an L2-normalised CLIP embedding
`emb`. Then it computes two raw scores:

```
s_text = top-K-mean(cosine(emb, staff_prompt_embeddings),  k=3)
c_text = top-K-mean(cosine(emb, customer_prompt_embeddings), k=1)

s_ref  = cosine(emb, mean(staff_reference_anchors))
c_ref  = cosine(emb, mean(customer_reference_anchors))

s = a · s_text + (1-a) · s_ref         # a = text_weight, default 0.80
c = a · c_text + (1-a) · c_ref
```

Three design choices in there are non-obvious:

1. **Asymmetric top-K** (staff k=3, customer k=1). Staff prompts are
   homogeneous — they all describe the same uniform — so a 3-of-N
   average is stable and noise-resistant. Customer prompts are
   intentionally diverse (kurtis, dresses, backpacks, grey t-shirts,
   pale tops, men in jeans, …) — for any specific person only one or
   two prompts truly match. Averaging top-3 dilutes the winner with
   weak matches; top-1 (winner-take-all) lets the most-specific prompt
   drive the customer score. This was the single biggest accuracy
   bump in the project.
2. **Per-store prompt gating.** The factory takes a `store_id` and
   only loads pink-uniform prompts for `ST2009`. If we'd loaded both
   pink and black prompts globally, every pink-clothed customer in
   Store 1 would have been pulled toward STAFF.
3. **Reference anchors with adjustable weight.** Two real Store 1 staff
   photos in `references/staff/` ground the classifier in actual store
   appearance, blended at `(1 - text_weight)`. Tuning that one knob
   trades CLIP's broad text knowledge against this store's specific
   uniform.

Decisions on top of `(s, c)`:

| Function | Rule | When |
|---|---|---|
| `live_vote()` (overlay) | needs `LIVE_MIN_OBS=3` frames; flips into STAFF if `s > c + LIVE_FLIP_TO_STAFF`, out of STAFF if `c > s + LIVE_FLIP_TO_CUST` (asymmetric / hysteresis) | per frame on the dashboard |
| `finalize()` (event row) | `is_staff = s > c + margin` over a per-visitor mean of all crops | once per visitor at end-of-track |

The asymmetric live thresholds exist because raw `(s, c)` wobbles by
~0.01 frame-to-frame; without hysteresis the overlay flips rapidly.
The final-pass `margin` is set conservatively (default 0.0) and pools
embeddings over the whole track, so a few flickering frames don't
change the canonical label written into events.

> **Geometry override.** For the Store 2 billing camera we additionally
> tag any track that spent ≥80% of its time in `BILLING_STAFF_SIDE`
> as STAFF regardless of the CLIP score. Geometry is an unambiguous
> signal there (customers physically can't stand inside the till
> island), so we use it.

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

### D. Staff classifier — symmetric averaging vs asymmetric top-K

- **AI suggested.** Use a symmetric top-K mean (k=3 for both staff and
  customer prompts) so "the score is robust to a single noisy prompt".
- **We chose.** k=3 for staff, k=1 (winner-take-all) for customer.
- **Why.** Symmetric top-K assumes both prompt pools have the same
  *internal coherence*. They don't. Staff prompts all describe the same
  black uniform — averaging the top 3 stabilises the score. Customer
  prompts are intentionally diverse (women in kurtis, men in grey
  t-shirts, dresses, backpacks, pale tops) because customers *are*
  diverse. For any specific person, only the one or two prompts that
  match their actual appearance are informative; the rest are noise.
  Averaging top-3 dilutes the winner. Switching customer to k=1
  (single best prompt drives the score) was the largest single
  accuracy improvement in the staff classifier — the AI's symmetric
  default was correct *in general* but wrong *for asymmetric pool
  semantics*. The lesson: top-K should match the structure of the
  data, not be set by ritual.

### E. Multi-store: copy the pipeline twice vs. parameterise

- **AI suggested.** Fork the detection pipeline per store and run two
  separate processes with different code paths.
- **We chose.** One pipeline binary + a `store_id` plumbed from the
  layout JSON through the classifier factory. Per-store behaviour
  (pink uniform prompts for Store 2, billing-zone geometry override
  for Store 2) is gated on that single id.
- **Why.** Copy-and-paste means every prompt fix and every threshold
  tweak has to be applied twice — and won't be. One binary forces the
  shared pieces (detector, tracker, Re-ID, age/gender heads, schema)
  to stay shared, and limits per-store divergence to a small,
  inspectable surface. Adding Store 3 will be a new layout JSON + one
  prompt list + one entry in `STORES`, no code changes.
