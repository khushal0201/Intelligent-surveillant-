# CHOICES.md

Three major decisions, the options considered, what the AI suggested first,
and what we shipped.

---

## 1. Person detection model

**Options considered**
- YOLOv8n ONNX (CPU, ~30 ms / frame at 640px)
- YOLOv8s/m PyTorch (~3× slower on CPU)
- DETR / RT-DETR (transformer; slower on CPU)
- MediaPipe person segmentation (fast but coarse)

**AI suggestion:** YOLOv8m for accuracy.

**Final:** YOLOv8n + ONNX Runtime.

**Reason:** This runs on a Windows laptop with no GPU. n is the only variant
that gives us the ≥5 fps stride budget required for live-ish dashboards. We
accept slightly lower recall on small/occluded persons; the Re-ID stage
recovers most identity continuity.

---

## 2. Event schema + ingest contract

**Options considered**
- Avro / Protobuf with a registry
- JSON Lines + JSON Schema (draft-07)
- Raw rows directly into Postgres

**AI suggestion:** Protobuf with a schema registry for forward compatibility.

**Final:** JSON Lines + Draft-07 JSON Schema, with the same schema enforced
at write-time in the pipeline and at read-time in the API (`Pydantic v2`
mirrors the schema).

**Reason:** The pipeline emits files first, ingests later. JSONL files are
trivially diffable, replayable, and human-readable; Protobuf would have made
debugging the CLIP staff classifier dramatically harder. Draft-07 keeps us
compatible with `jsonschema 4.x`. Idempotent ingest via the `event_id` UUID
means we can replay the same file safely.

---

## 3. API + dashboard architecture

**Options considered**
- FastAPI + SQLAlchemy + vanilla JS dashboard (single container)
- Django + DRF + React SPA (two containers + build step)
- Node/Express + Next.js
- Streamlit one-file app

**AI suggestion:** Next.js + tRPC for end-to-end type safety.

**Final:** FastAPI sync + vanilla JS dashboard served from `app/static/`.

**Reason:** We needed to ship a single container that anyone can run with one
`docker compose up`. Streamlit was a close second, but the requirement to
expose a JSON API to external systems (POS, alerting) and embed five
`<video>` players with per-camera offset sync ruled it out. FastAPI gives us
OpenAPI docs at `/docs` for free, and the dashboard is small enough that a
build step would have been ceremony.
