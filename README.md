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
  `detection_pipeline/out/events_full.jsonl` and are auto-bootstrapped into the
  in-container SQLite on startup.
- Dashboard, funnel, heatmap, anomalies and live event-stream UI (Tailwind CSS)
  are served from FastAPI.

Source repo: <https://github.com/khushal0201/Intelligent-surveillant->
