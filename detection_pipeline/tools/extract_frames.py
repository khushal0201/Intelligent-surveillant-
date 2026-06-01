"""Extract a mid-frame from each camera clip for visual inspection."""
import os
import cv2

footage = r"resources\CCTV Footage-20260529T160731Z-3-00144614ea\CCTV Footage"
out_dir = "detection_pipeline/out"
os.makedirs(out_dir, exist_ok=True)

for cam in ["CAM 1.mp4", "CAM 2.mp4", "CAM 3.mp4", "CAM 4.mp4", "CAM 5.mp4"]:
    path = os.path.join(footage, cam)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{cam}: frames={n} fps={fps:.1f} size={w}x{h}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    if ok:
        slug = cam.replace(".mp4", "").replace(" ", "_").lower()
        out = os.path.join(out_dir, f"{slug}_mid.jpg")
        cv2.imwrite(out, frame)
        print(f"  -> {out}")
    cap.release()
