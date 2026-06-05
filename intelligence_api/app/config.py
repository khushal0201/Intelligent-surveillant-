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
# bootstrap source so the dashboard has data on first launch. Override with
# INTEL_BOOTSTRAP_GLOB (semicolon-separated list of paths or globs) if you
# want to load smoke / older runs.
DEFAULT_EVENTS_GLOB = os.environ.get(
    "INTEL_BOOTSTRAP_GLOB",
    ";".join([
        str(ROOT / "detection_pipeline" / "out" / "events_store1_full.jsonl"),
        str(ROOT / "detection_pipeline" / "out" / "events_store2_full.jsonl"),
    ]),
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

# Per-store config: camera→clip filename, layout path, footage dir, display name.
# Used by /dashboard/cameras?store_id=X and the upload endpoint.
STORES: dict[str, dict] = {
    "ST1008": {
        "name": "Brigade Bangalore",
        "city": "Bangalore",
        "layout": str(ROOT / "detection_pipeline" / "configs" / "store1_layout.json"),
        "footage_dir": str(ROOT / "updated_resources" / "Store 1-20260602T101818Z-3-001ec38db8" / "Store 1"),
        "cameras": {
            "CAM_ENTRY_03":    "CAM 3 - entry.mp4",
            "CAM_SKINCARE_01": "CAM 1 - zone.mp4",
            "CAM_MAKEUP_02":   "CAM 2 - zone.mp4",
            "CAM_BILLING_05":  "CAM 5 - billing.mp4",
        },
        "annotated_dir": str(ROOT / "detection_pipeline" / "out" / "annotated_store1_full"),
    },
    "ST2009": {
        "name": "Store 2",
        "city": "Unknown",
        "layout": str(ROOT / "detection_pipeline" / "configs" / "store2_layout.json"),
        "footage_dir": str(ROOT / "updated_resources" / "Store 2-20260602T101819Z-3-001099f208" / "Store 2"),
        "cameras": {
            "CAM_ENTRY_01":   "entry 1.mp4",
            "CAM_ENTRY_02":   "entry 2.mp4",
            "CAM_ZONE_03":    "zone.mp4",
            "CAM_BILLING_06": "billing_area.mp4",
        },
        "annotated_dir": str(ROOT / "detection_pipeline" / "out" / "annotated_store2_full"),
    },
}

# Camera → clip filename map. The dashboard plays these via /clips/{name}.
# Kept as flat dict for backwards-compat with old single-store endpoints.
CAMERA_CLIPS = {
    "CAM_SKINCARE_01": "CAM 1.mp4",
    "CAM_MAKEUP_02":   "CAM 2.mp4",
    "CAM_ENTRY_03":    "CAM 3.mp4",
    "CAM_BACKROOM_04": "CAM 4.mp4",
    "CAM_BILLING_05":  "CAM 5.mp4",
}

# Flat lookup: camera_id → store_id (for resolving uploads to the right store).
CAMERA_TO_STORE: dict[str, str] = {}
for _sid, _cfg in STORES.items():
    for _cam in _cfg["cameras"]:
        CAMERA_TO_STORE[_cam] = _sid
