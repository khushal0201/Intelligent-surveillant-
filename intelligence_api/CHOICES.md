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
- text prompts split into two pools (~30 staff prompts + ~50 customer
  prompts), each carrying explicit visual cues — uniform colour,
  silhouette, footwear, accessories — rather than abstract role labels;
- a small set of reference image crops per class
  (`detection_pipeline/references/{staff,customers}/`) that ground the
  classifier in this store's actual appearance;
- **per-store prompt gating** — Store 2's pink-uniform prompts are only
  loaded when `store_id == "ST2009"`; loading them globally would have
  pulled every pink-clad customer in Store 1 toward STAFF;
- **asymmetric top-K text scoring** — staff k=3 (homogeneous prompts,
  averaging stabilises), customer k=1 (diverse prompts, winner-take-all
  lets the most-specific match drive the score). This was the largest
  single accuracy bump;
- a tunable text-vs-reference weight (`text_weight`, default 0.80) —
  lowering it lets the few real reference photos dominate when text
  prompts disagree on edge cases;
- a `live_vote()` with a 3-frame minimum and asymmetric hysteresis
  (`LIVE_FLIP_TO_STAFF` / `LIVE_FLIP_TO_CUST`) that removes the
  early-frame oscillation from a noisy raw score;
- a per-visitor `finalize()` that pools embeddings over the whole
  track for the canonical label written into events;
- an additional **geometry override** for the Store 2 billing camera
  (≥80% time inside the staff-side polygon ⇒ STAFF) because, there,
  geometry is unambiguous and the CLIP score is unnecessary.

**Why we agreed in spirit but chose differently.** The AI was right
that we needed *some* learned representation, not a hand-rolled
heuristic. We just took the **zero-shot** path: no labels, no training
loop, no model maintenance when uniforms change. The cost is ≈110 ms /
crop on CPU, which is fine because classification runs once per visitor
track (not per frame) and most of the work is the encoder pass, which
we batch.

**The tuning journey is the interesting part.** The first version of the
classifier produced this failure mode: real Store 1 staff (women in
all-black uniform behind the counter) classified as customers, while
some customers in dark clothing classified as staff. The fix was *not*
"add more prompts" or "lower the threshold". It was a sequence of
narrower, principled changes:

1. Recognised that the customer prompt pool had ~50 diverse prompts
   while the staff pool had ~22 narrow ones. Symmetric top-3 averaging
   over diverse customer prompts was diluting the winner.
2. Switched customer to k=1 (winner-take-all). Staff stayed at k=3.
3. Pushed `text_weight` down to give the actual reference photos more
   say in the score, then back up to 0.80 once the prompt pools were
   balanced.
4. Removed pose-based staff prompts ("phone in hand", "scanning
   barcode") — every customer also holds a phone, so pose-based prompts
   were false-positive engines.
5. Added explicit customer prompts for the failure modes we kept
   seeing: backpack on shoulder, open-toe footwear, pale top + jeans,
   phone-at-face. Each prompt was written from a real misclassified
   crop, not from imagination.
6. Per-store gating, so Store 1 prompts never describe pink uniforms
   and vice versa.

VLM, used surgically — at the **classification** step, where prior
knowledge of "what staff look like" is high-leverage — rather than
trying to do detection or tracking with a VLM (where it would have been
the wrong tool).

---

## Demographics: same VLM, separate heads

The age and gender labels on the dashboard piggyback on the same
OpenCLIP encoder pass:

- **Gender:** mean-pooled female and male prompt sets (long hair /
  ponytail / narrow shoulders / no Adam's apple vs. facial hair /
  V-shape silhouette).
- **Age:** seven buckets (`child / teen / 20s / 30s / 40s / 50s / 60+`)
  with a per-bucket prompt list and a **per-bucket top-K (k=2)** mean
  cosine, then softmax over buckets. The 20s and 30s buckets carry many
  generic prompts (since most shoppers fall there), while child / 40s+ /
  60+ require explicit cues (relative-height / wrinkles / grey hair) so
  the system stays conservative on the long tails.

Reusing the same image embedding for staff/customer + age + gender
keeps total inference cost flat (~110 ms / crop) for three labels.

---

## Multi-store support

**Options considered**

| Option | Pros | Cons |
|---|---|---|
| Fork the pipeline per store | Independent code paths; per-store changes can't break the other store | Drift between forks; every fix has to be applied N times |
| **One pipeline binary + `store_id` parameter** | Shared detector / tracker / Re-ID / schema; per-store behaviour is a small, inspectable surface | Have to think carefully about which knobs are per-store vs global |
| Multi-tenant DB only, single global pipeline | Simplest API side | No way to encode per-store visual differences (uniform colour, billing geometry) |

**AI suggestion.** Two separate processes with separate config files.

**We shipped.** One detection pipeline + one API, each parameterised on
`store_id`. Per-store layout JSONs (`store1_layout.json`,
`store2_layout.json`) supply zone polygons, camera mappings, and the
store id; the staff classifier factory routes prompt-pool selection on
that id; the API's `STORES` registry maps it to the camera list,
footage folder, layout path, and annotated-video folder.

**Why.** Forking the pipeline would have meant every staff-prompt fix,
every threshold tweak, every schema change had to be applied twice —
and inevitably wouldn't be. Keeping it one binary forces the shared
pieces to stay shared, and limits per-store divergence to a clearly
labelled set of branches in the code. Adding Store 3 will be a layout
JSON + one prompt list + one `STORES` entry — no code change.
