FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs needed by opencv (used by transcoder) — but the Space itself does
# not transcode, it only serves the API. Kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Only the API requirements — the heavy detection_pipeline deps (torch,
# open_clip, onnxruntime) are NOT needed inside the Space, since pre-computed
# events_full.jsonl is shipped with the repo and bootstrapped on startup.
COPY intelligence_api/requirements.txt /app/intelligence_api/requirements.txt
RUN pip install -r /app/intelligence_api/requirements.txt

# Copy what the API actually serves.
COPY intelligence_api /app/intelligence_api
COPY detection_pipeline/out/events_full.jsonl /app/detection_pipeline/out/events_full.jsonl
# Annotated mp4s are intentionally NOT copied — VIDEO_BASE_URL points at
# GitHub raw so the Space stays small.

ENV PYTHONPATH=/app \
    INTEL_BOOTSTRAP_GLOB=/app/detection_pipeline/out/events_full.jsonl \
    INTEL_DB_URL=sqlite:////tmp/intel.db \
    INTEL_UPLOAD_DIR=/tmp/uploads \
    INTEL_ANNOTATED_DIR=/tmp/annotated \
    INTEL_CLIPS_DIR=/tmp/clips \
    PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn intelligence_api.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
