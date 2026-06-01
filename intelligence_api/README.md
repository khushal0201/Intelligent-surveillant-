# Purplle Intelligence API

FastAPI service that ingests behavioural events emitted by `detection_pipeline/`,
stores them in SQLite/Postgres, and exposes analytics + a live dashboard with
five synced CCTV clips and an upload form.

## Quick start (5 commands)

```powershell
# 1. clone & enter
cd "Purplle"

# 2. install
& "C:\venvs\purplle_venv\Scripts\python.exe" -m pip install -r intelligence_api\requirements.txt

# 3. (optional) generate events
& "C:\venvs\purplle_venv\Scripts\python.exe" detection_pipeline\run.py --footage-dir "resources\CCTV Footage-20260529T160731Z-3-00144614ea\CCTV Footage" --layout detection_pipeline\configs\store_layout.json --schema detection_pipeline\schema\event_schema.json --out detection_pipeline\out\events.jsonl --detector-backend onnx --onnx-model-path detection_pipeline\models\yolov8n.onnx --staff-mode openclip --staff-ref-dir detection_pipeline\references\staff --customer-ref-dir detection_pipeline\references\customers --conf-thresh 0.35 --stride 5

# 4. run the API + dashboard
& "C:\venvs\purplle_venv\Scripts\python.exe" -m uvicorn intelligence_api.app.main:app --reload --host 0.0.0.0 --port 8000

# 5. open
start http://localhost:8000/
```

Or via Docker: `docker compose up --build`.

## Endpoints

| Method | Path                              | Notes |
|--------|-----------------------------------|-------|
| POST   | `/events/ingest`                  | batch ≤500, idempotent by `event_id` |
| GET    | `/stores/{id}/metrics`            | unique visitors, conversion, dwell, queue, abandonment (excludes staff) |
| GET    | `/stores/{id}/funnel`             | Entry → Zone → Billing Queue → Purchase |
| GET    | `/stores/{id}/heatmap`            | per-zone visits + intensity, low-confidence flag |
| GET    | `/stores/{id}/anomalies`          | queue spike, conversion drop, dead zone |
| GET    | `/health`                         | service + STALE_FEED per store |
| GET    | `/dashboard/cameras`              | dashboard helper: clip URLs + first/last event ts |
| GET    | `/dashboard/events?camera_id=...` | events with `offset_ms` for video sync |
| POST   | `/uploads`                        | multipart `file` + `camera_id` → runs detection pipeline + ingests |

## Tests

```powershell
& "C:\venvs\purplle_venv\Scripts\python.exe" -m pytest --cov=intelligence_api.app intelligence_api\tests
```

## Layout

```
intelligence_api/
  app/
    main.py             FastAPI bootstrap, static + clip mounts
    db.py models.py     SQLAlchemy 2.0 (sync)
    schemas.py          Pydantic v2
    bootstrap.py        loads existing JSONL on first start
    logging_mw.py       JSON logs + trace_id middleware
    routers/            ingest, analytics, health, dashboard, uploads
    services/           analytics computations
    static/             dashboard (HTML + vanilla JS)
  tests/                pytest suite (>70% coverage target)
  Dockerfile
```

See `DESIGN.md` and `CHOICES.md` for architecture and trade-offs.
