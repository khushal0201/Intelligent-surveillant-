"""One-time export of YOLOv8n weights to ONNX (run once, then never again).

After this finishes, the rest of the pipeline only uses onnxruntime;
ultralytics / torch are not imported by run.py.
"""
from pathlib import Path
import sys

OUT_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[export] target dir: {OUT_DIR}")
sys.stdout.flush()

from ultralytics import YOLO

# This downloads yolov8n.pt (~6 MB) into the working directory on first run.
m = YOLO("yolov8n.pt")
print("[export] loaded YOLOv8n; exporting to ONNX (imgsz=640, opset=12)...")
sys.stdout.flush()

p = m.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
print(f"[export] wrote: {p}")

# Move the produced .onnx and .pt into models/ for tidy storage
import shutil
src_onnx = Path(p)
dst_onnx = OUT_DIR / "yolov8n.onnx"
if src_onnx.resolve() != dst_onnx.resolve():
    shutil.move(str(src_onnx), str(dst_onnx))
    print(f"[export] moved ONNX -> {dst_onnx}")

src_pt = Path("yolov8n.pt")
if src_pt.exists():
    dst_pt = OUT_DIR / "yolov8n.pt"
    shutil.move(str(src_pt), str(dst_pt))
    print(f"[export] moved .pt -> {dst_pt}")

print("[export] done.")
