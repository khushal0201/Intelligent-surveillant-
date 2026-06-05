---
title: Intelligent Surveillant
emoji: 🛍️
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Intelligent Surveillant

Offline-first retail computer-vision pipeline + intelligence API for store **ST1008**
(Purplle Tech Challenge submission).

The Hugging Face Space hosts only the **API + dashboard**. Annotated H.264 mp4s
are streamed directly from the GitHub repo's raw URLs (configured via the
`VIDEO_BASE_URL` env var) so the container stays small.

- Detection pipeline (YOLOv8n ONNX + OpenCLIP staff classifier + MobileNetV3 reid)
  ran offline on the Microsoft device — its pre-computed events live at
  `detection_pipeline/out/events_store1_full.jsonl` (Store 1, ST1008) and
  `detection_pipeline/out/events_store2_full.jsonl` (Store 2, ST2009) and are
  auto-bootstrapped into the in-container SQLite on startup. Both files
  conform to `detection_pipeline/schema/event_schema.json` (the same schema
  as the provided `updated_resources/sample_events*.jsonl`).
- Dashboard, funnel, heatmap, anomalies and live event-stream UI (Tailwind CSS)
  are served from FastAPI.

Source repo: <https://github.com/khushal0201/Intelligent-surveillant->

## Where to find the canonical event log

The detection pipeline emits a single line of JSON per event, exactly matching
the format in `updated_resources/sample_eventsbe42122.jsonl`. The committed
canonical outputs are:

| Store      | Path                                                  | Cameras                                                      |
| ---------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| ST1008 (Store 1) | `detection_pipeline/out/events_store1_full.jsonl` | `CAM_ENTRY_03`, `CAM_SKINCARE_01`, `CAM_MAKEUP_02`, `CAM_BILLING_05` |
| ST2009 (Store 2) | `detection_pipeline/out/events_store2_full.jsonl` | `CAM_ENTRY_01`, `CAM_ENTRY_02`, `CAM_ZONE_03`, `CAM_BILLING_06`     |

Schema: `detection_pipeline/schema/event_schema.json` (Draft-7, validated on
every emit). Each row carries `store_id`, `camera_id`, `visitor_id`,
`event_type`, `timestamp`, `zone_id`, `dwell_ms`, `is_staff`, `confidence`,
and a `metadata` block with `gender`, `age_bucket`, `sku_zone`, `queue_depth`,
and `session_seq`.

Annotated H.264 review videos sit alongside the events:

- `detection_pipeline/out/annotated_store1_full/CAM_*.mp4`
- `detection_pipeline/out/annotated_store2_full/CAM_*.mp4`

Everything else (smoke runs, per-camera ablation files, calibration jpegs,
older annotated dirs) has been pruned — only the two `*_full.jsonl` files and
the two `annotated_*_full/` dirs are retained.