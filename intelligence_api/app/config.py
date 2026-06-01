from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_ROOT = Path(__file__).resolve().parents[1]

DATABASE_URL = os.environ.get(
    "INTEL_DB_URL",
    f"sqlite:///{API_ROOT / 'intel.db'}",
)

# Path to pre-existing JSONL produced by detection_pipeline. Used as a
# bootstrap source so the dashboard has data on first launch.
DEFAULT_EVENTS_GLOB = os.environ.get(
    "INTEL_BOOTSTRAP_GLOB",
    str(ROOT / "detection_pipeline" / "out" / "events_full.jsonl"),
)

# Folder containing the original CCTV clips for the dashboard player.
# Defaults to the GitHub-friendly H.264 transcodes under resources/clips/;
# falls back to the original heavy folder if that's all the user has.
_default_clips = ROOT / "resources" / "clips"
if not _default_clips.exists():
    _default_clips = ROOT / "resources" / "CCTV Footage-20260529T160731Z-3-00144614ea" / "CCTV Footage"
CLIPS_DIR = Path(os.environ.get("INTEL_CLIPS_DIR", str(_default_clips)))

# Folder containing pipeline-annotated clips (boxes, IDs, zones drawn on).
# Falls back to CLIPS_DIR when an annotated file is missing.
ANNOTATED_DIR = Path(os.environ.get(
    "INTEL_ANNOTATED_DIR",
    str(ROOT / "detection_pipeline" / "out" / "annotated"),
))
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Where uploaded clips are stored before processing.
UPLOAD_DIR = Path(os.environ.get("INTEL_UPLOAD_DIR", str(API_ROOT / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Detection pipeline config (used by upload endpoint).
PIPELINE_DIR = ROOT / "detection_pipeline"
PYTHON_EXE = os.environ.get("INTEL_PYTHON", r"C:\venvs\purplle_venv\Scripts\python.exe")

# Default clip-start timestamp used when ingesting raw JSONL whose ts already
# encodes wall-clock; dashboard syncs videos to this offset.
DEFAULT_CLIP_START_ISO = os.environ.get(
    "INTEL_DEFAULT_CLIP_START", "2026-04-10T14:39:50Z"
)

STORE_ID = os.environ.get("INTEL_STORE_ID", "ST1008")

# Camera → clip filename map. The dashboard plays these via /clips/{name}.
CAMERA_CLIPS = {
    "CAM_ENTRY_03":    "CAM 1.mp4",
    "CAM_MAKEUP_02":   "CAM 2.mp4",
    "CAM_SKINCARE_01": "CAM 3.mp4",
    "CAM_BACKROOM_04": "CAM 4.mp4",
    "CAM_BILLING_05":  "CAM 5.mp4",
}
