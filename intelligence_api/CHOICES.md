# CHOICES.md

Three model / architecture decisions, the options considered, what the AI
suggested first, and what we shipped (with reasons we agreed or overrode).

---

## 1. Person-detection model

**Options considered**

| Option | Pros | Cons |
|---|---|---|
| YOLOv8n + ONNX Runtime (CPU) | ~30 ms / frame at 640 px, no GPU needed | Slightly lower recall on small / occluded persons |
| YOLOv8s / m PyTorch | Better mAP | ~3× slower on CPU; can't sustain ≥5 fps stride |
| RT-DETR / DETR | Strong on crowded scenes | Transformer attention is too slow on CPU |
| MediaPipe person segmentation | Very fast | Coarse boxes, no class confidence we trust |
| Ultralytics YOLO-World (open-vocab) | Fewer false positives on store fixtures (mannequins) | 4–6× slower than YOLOv8n; gain wasn't worth it for our 5 cameras |

**AI suggestion.** YOLOv8m for "balanced accuracy", later YOLO-World for
"open-vocabulary robustness".

**We shipped.** YOLOv8n exported to ONNX, run with `onnxruntime` on CPU.
Stride 5 (one inference per ~200 ms), dense streaming reuses the previous
inference's tracks for the carry-over frames so saved videos play at the
source FPS rather than the inference rate.

**Why we overrode.** This runs on a Windows laptop with no GPU. n is the
only variant that hits the latency budget while leaving headroom for the
Re-ID and CLIP staff classifier downstream. We accept slightly lower
recall on tiny/occluded persons; the MobileNetV3 Re-ID stage recovers most
identity continuity on short occlusions, and the staff classifier votes
over multiple frames so a missed detection doesn't change the label.

We did partially adopt the AI's open-vocab nudge — but at the
**classification** step (CLIP), not the detection step.

---

## 2. Event schema + ingest contract

**Options considered**

| Option | Pros | Cons |
|---|---|---|
| Avro / Protobuf with a schema registry | Forward compatibility, tiny on the wire | Heavy tooling for a single-team project; opaque to grep / diff |
| **JSON Lines + Draft-07 JSON Schema** | Diffable, replayable, human-readable; one schema, two enforcers (pipeline + API) | More verbose on disk |
| Raw rows directly into Postgres from the pipeline | One fewer hop | Couples the pipeline to a live DB; no offline replay; ingest is no longer idempotent |

**AI suggestion.** Protobuf with a schema registry "for forward
compatibility and binary efficiency".

**We shipped.** JSONL + JSON Schema (Draft-07), validated at write-time
in the pipeline and at read-time in the API via Pydantic v2 (Pydantic
mirrors the schema; the schema file is the canonical contract).

**Why we overrode.**
- The pipeline emits files first and ingests them later — Protobuf would
  have made debugging the CLIP staff classifier dramatically harder
  (every misclassification was investigated by reading the raw event
  stream).
- Schema evolution at our pace (a half-dozen field additions over the
  project) does not justify a registry service.
- Idempotent ingest via the deterministic `event_id` UUID means we can
  replay the same file safely across re-runs.

We agreed with the AI on **using a schema** (vs. ad-hoc dicts) — we just
chose JSON Schema instead of Protobuf.

---

## 3. API + dashboard architecture

**Options considered**

| Option | Pros | Cons |
|---|---|---|
| **FastAPI sync + Tailwind dashboard, single container** | One `docker compose up`; OpenAPI at `/docs` for free; ships on HF Space unchanged | Vanilla JS scales worse past ~1k LoC |
| Django + DRF + React SPA (two containers) | Mature admin, batteries included | Two build pipelines; heavy for the surface we need |
| Node / Express + Next.js | Same language front-to-back | Loses the Python ML ecosystem (CLIP, ONNX) |
| Streamlit one-file app | Fastest to a working demo | No JSON API for POS / alerting; can't embed 5 synchronised `<video>` tags cleanly |

**AI suggestion.** Next.js + tRPC (round 1), shadcn/ui + App Router
(round 2) for "end-to-end type safety and production-grade UX".

**We shipped.** FastAPI + SQLAlchemy 2.0 (sync) + a Tailwind-CDN
dashboard served from `app/static/`. Single Dockerfile, single port.

**Why we overrode.**
- **Single container** is a hard requirement — the HF Space build copies
  the API + a 118 KB JSONL and that's it. A Node toolchain would have
  required either a second image or a multi-stage build for ~600 LoC of
  UI.
- The dashboard talks to ~10 endpoints. Type safety from tRPC would
  have meant carrying TypeScript and a build step into a Python repo.
  We get the same guarantee informally by generating client types from
  `/openapi.json` when we need them.
- Tailwind via CDN gave us 90 % of the React+shadcn polish at 0 % of the
  build complexity. Inter + JetBrains Mono, glass panels with
  `backdrop-blur`, animated KPI tiles, brand purple/fuchsia gradient —
  all in 3 files (`index.html`, `style.css`, `app.js`).

This was the AI suggestion we **partially** adopted: yes to modern
typography + utility CSS, no to the framework + bundler.

---

## VLM usage in the pipeline (staff vs. customer)

The brief invites us to use a VLM for parts of the pipeline. We used one
for the staff/customer classifier — historically the most thankless part
of retail CV.

**Options considered**

| Option | Training data needed | Latency / crop | Robustness |
|---|---|---|---|
| MobileNetV3 backbone + 2-layer MLP head | 200–500 hand-labelled crops per class | ~6 ms | Brittle to uniform changes; needs re-training |
| Color-histogram heuristic on the torso ROI | None | <1 ms | Falls apart on dark customer outfits |
| **OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) prototype classifier** | None — text prompts + a handful of reference images | ~110 ms | Robust; uniform changes are a prompt edit |
| BLIP-2 / LLaVA caption + keyword match | None | ~500 ms+ | Overkill, latency too high |

**AI suggestion.** Train a small MLP head on top of MobileNetV3 features,
using a few hand-labelled crops per class.

**We shipped.** OpenCLIP zero-shot prototype classifier with:
- text prompts (e.g. "a person wearing a Purplle staff uniform / lanyard"
  for the staff class, "a customer browsing in a beauty store" for the
  customer class),
- a small set of reference image crops per class (`detection_pipeline/references/{staff,customer}/`),
- a weighted blend (text 0.7, visual 0.3 — text dominates because the
  visual reference set is small),
- a `live_vote()` with a 3-frame minimum and a hysteresis margin
  (`STAFF→CUST` requires `s > c + 0.02`; `STAFF` is sticky if
  `s > c − 0.02`) to remove early-frame oscillation,
- a per-visitor `finalize()` that pools embeddings over the whole track
  for the canonical label written into events.

**Why we agreed in spirit but chose differently.** The AI was right
that we needed *some* learned representation, not a hand-rolled
heuristic. We just took the **zero-shot** path: no labels, no training
loop, no model maintenance when uniforms change. The cost is ≈110 ms /
crop on CPU, which is fine because classification runs once per visitor
track (not per frame).

VLM, used surgically — at the **classification** step, where prior
knowledge of "what staff look like" is high-leverage — rather than
trying to do detection or tracking with a VLM (where it would have been
the wrong tool).
