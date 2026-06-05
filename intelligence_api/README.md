# Purplle Intelligence API

FastAPI service that ingests behavioural events emitted by `detection_pipeline/`,
stores them in SQLite/Postgres, and exposes analytics + a live multi-store
dashboard with synced annotated CCTV clips and an upload form.

## Highlights

- **Two stores in one DB.** Each event carries a `store_id`; analytics,
  funnel, heatmap, anomalies and the dashboard all filter per store.
  Adding a third store is a config-file change, not a code change.
- **Replayable, file-based ingest.** The boundary between detection and
  analytics is a JSON Lines file validated against a single JSON Schema.
  Wipe the DB, restart, re-bootstrap — same events back.
- **Idempotent.** Events keyed by a deterministic UUID
  (`visitor_id + event_type + timestamp_bucket`); replays / re-runs never
  produce duplicates.
- **Annotated video sync.** The dashboard plays the pipeline-annotated
  H.264 mp4 (boxes, IDs, zones) and toasts events on the playing video,
  synced to ±1 stride (~200 ms) using each camera's earliest event as
  the zero point.
- **Single-container deploy.** No Node toolchain, no bundler. Tailwind
  via CDN, FastAPI serves `app/static/` directly. Same image runs locally
  and on a Hugging Face Space (with videos resolved against
  `VIDEO_BASE_URL` to keep the Space under the 10 MiB-per-file ceiling).

## Quick start

```powershell
# 1. install
& "C:\venvs\purplle_venv\Scripts\python.exe" -m pip install -r intelligence_api\requirements.txt

# 2. (optional) generate events for both stores back-to-back
cd detection_pipeline
$py = "C:\venvs\purplle_venv\Scripts\python.exe"

& $py run.py --footage-dir "..\updated_resources\Store 1-20260602T101818Z-3-001ec38db8\Store 1" `
             --layout configs\store1_layout.json --schema schema\event_schema.json `
             --out out\events_store1_full.jsonl `
             --detector-backend onnx --onnx-model-path models\yolov8n.onnx `
             --staff-mode openclip `
             --staff-ref-dir references\staff --customer-ref-dir references\customers `
             --conf-thresh 0.55 --stride 2 --max-frames 0 `
             --save-video-dir out\annotated_store1_full

& $py run.py --footage-dir "..\updated_resources\Store 2-20260602T101819Z-3-001099f208\Store 2" `
             --layout configs\store2_layout.json --schema schema\event_schema.json `
             --out out\events_store2_full.jsonl `
             --detector-backend onnx --onnx-model-path models\yolov8n.onnx `
             --staff-mode openclip `
             --staff-ref-dir references\staff --customer-ref-dir references\customers `
             --conf-thresh 0.55 --stride 2 --max-frames 0 `
             --save-video-dir out\annotated_store2_full
cd ..

# 3. run the API + dashboard
& "C:\venvs\purplle_venv\Scripts\python.exe" -m uvicorn intelligence_api.app.main:app --reload --host 0.0.0.0 --port 8000

# 4. open
start http://localhost:8000/
```

Or via Docker: `docker compose up --build`.

> **Tip.** The API bootstraps from `events_store1_full.jsonl` and
> `events_store2_full.jsonl` by default. To load a different set of files
> without touching code, set `INTEL_BOOTSTRAP_GLOB` to a semicolon-separated
> list of paths or globs. To reload from scratch, delete
> `intelligence_api/intel.db` before restarting (the bootstrap is
> idempotent but only re-scans on startup).

## Endpoints

| Method | Path                              | Notes |
|--------|-----------------------------------|-------|
| POST   | `/events/ingest`                  | batch ≤500, idempotent by `event_id` |
| GET    | `/stores/{id}/metrics`            | unique visitors, conversion, dwell, queue, abandonment (excludes staff) |
| GET    | `/stores/{id}/funnel`             | Entry → Zone → Billing Queue → Purchase |
| GET    | `/stores/{id}/heatmap`            | per-zone visits + intensity, low-confidence flag |
| GET    | `/stores/{id}/anomalies`          | queue spike, conversion drop, dead zone |
| GET    | `/health`                         | service + STALE_FEED per store |
| GET    | `/dashboard/cameras?store_id=X`   | per-store camera list with `clip_url`, `annotated_url`, first/last event ts |
| GET    | `/dashboard/events?camera_id=...` | events with `offset_ms` for video sync |
| POST   | `/uploads`                        | multipart `file` + `camera_id` → routes to the right store, runs detection pipeline, ingests |

OpenAPI/Swagger UI at `/docs`.

## Tests

```powershell
& "C:\venvs\purplle_venv\Scripts\python.exe" -m pytest --cov=intelligence_api.app intelligence_api\tests
```

Coverage target ≥70%; current suite covers ingest idempotency, analytics
math, dashboard sync offsets, and the upload→pipeline→ingest round-trip.

## Layout

```
intelligence_api/
  app/
    main.py             FastAPI bootstrap, static + clip mounts
    config.py           per-store registry (cameras, layout, footage, annotated_dir)
    db.py models.py     SQLAlchemy 2.0 (sync)
    schemas.py          Pydantic v2 (mirrors the JSON Schema)
    bootstrap.py        loads existing JSONL on first start
    logging_mw.py       JSON logs + trace_id middleware
    routers/            ingest, analytics, health, dashboard, uploads
    services/           analytics computations
    static/             dashboard (HTML + vanilla JS)
  tests/                pytest suite
  Dockerfile
```

See `DESIGN.md` and `CHOICES.md` for architecture and trade-offs.
