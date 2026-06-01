# Detection Pipeline — Part A

Emits structured behavioural events from CCTV clips per the required schema.

## Stack
- **Detector + Tracker:** YOLOv8 (Ultralytics) with built-in **ByteTrack**
- **Re-ID:** CLIP (`ViT-B-32`, OpenCLIP) image embeddings + cosine gallery; falls back to HSV histogram if CLIP isn't installed
- **Staff classifier:** CLIP zero-shot (uniform/apron vs casual)
- **Zones:** Shapely polygons in normalised image coords; entry/exit via signed line crossing
- **POS correlation:** Brigade Bangalore CSV; `BILLING_QUEUE_ABANDON` if no sale within 120s after billing-zone exit
- **Schema:** JSON Schema (Draft-7) validated on every emit

## Layout
```
detection_pipeline/
├── configs/
│   ├── store_layout.json     # zones + entry line per camera (EDIT polygons!)
│   └── bytetrack.yaml
├── schema/event_schema.json
├── pipeline/
│   ├── detector.py   # YOLO + ByteTrack streamer
│   ├── reid.py       # CLIP gallery, REENTRY detection
│   ├── staff.py      # CLIP zero-shot is_staff
│   ├── zones.py      # polygons + entry-line side
│   ├── events.py     # per-camera state machine + emitter
│   └── pos.py        # BILLING_QUEUE_ABANDON correlation
├── run.py            # main runner
└── requirements.txt
```

## Install
```powershell
cd detection_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
First run downloads `yolov8n.pt` and the CLIP weights automatically.

## Calibrate zones (important)
`configs/store_layout.json` ships with **placeholder polygons**. Open one frame
from each clip, then edit each camera's `zones` (and the entry line for
`CAM_ENTRY_01`) using **normalised** coordinates `[0..1]`. Quick way:

```powershell
python - <<'PY'
import cv2
cap = cv2.VideoCapture(r"..\resources\CCTV Footage-20260529T160731Z-3-00144614ea\CCTV Footage\CAM 1.mp4")
ok, f = cap.read(); cv2.imwrite("cam1.png", f)
PY
```
Then sketch polygons on `cam1.png` and convert pixel coords to fractions of
width/height.

## Run
```powershell
python run.py `
  --footage-dir "..\resources\CCTV Footage-20260529T160731Z-3-00144614ea\CCTV Footage" `
  --pos-csv    "..\resources\Brigade_Bangalore_10_April_26 (1)bc6219c.csv" `
  --out        "out\events.jsonl" `
  --stride     2
```

Smoke test on a single camera, 500 frames:
```powershell
python run.py --footage-dir "..\resources\CCTV Footage-20260529T160731Z-3-00144614ea\CCTV Footage" `
              --cameras CAM_ENTRY_01 --max-frames 500 --out out\smoke.jsonl
```

## How the scoring criteria are addressed
| Criterion | Implementation |
|---|---|
| **Entry/exit accuracy** | Signed line-crossing on the *foot point* of each track; both directions emit distinct ENTRY/EXIT |
| **Staff exclusion** | `is_staff` set via CLIP zero-shot; events still emit (never silently dropped) |
| **Re-entry handling** | Visitor gallery (`pipeline/reid.py`) keeps embeddings *across* EXITs; a re-match yields `REENTRY` instead of a new `ENTRY` |
| **Group handling** | Each track gets its own `visitor_id`; 3 simultaneous tracks crossing → 3 ENTRY events |
| **Confidence calibration** | YOLO `conf` is passed through verbatim into the event; **no low-conf suppression** |
| **Schema compliance** | `jsonschema` Draft-7 validation on every emit; `event_id` is `uuid4`; `timestamp` derived from clip start + frame offset |

## Output
JSONL at `out/events.jsonl`, one event per line, validated against
`schema/event_schema.json`.

## Tuning knobs
- `--reid-threshold` (default `0.82`) — raise to reduce false REENTRY matches
- `--stride` — frame skip; `2` ≈ 12.5 fps from a 25 fps clip
- `--model yolov8s.pt` / `yolov8m.pt` for higher recall
- `configs/bytetrack.yaml` — adjust `track_buffer` for longer occlusions
