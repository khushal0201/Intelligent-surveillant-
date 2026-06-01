"""Transcode annotated mp4v clips to browser-friendly H.264 in place.

Usage:
    python detection_pipeline/transcode_to_h264.py [path-or-dir ...]

Default: transcodes every *.mp4 under detection_pipeline/out/annotated/.
Files with a missing moov atom (still being written) are skipped.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


def _is_mp4_finalized(path: Path) -> bool:
    import cv2
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()
    return ok


def transcode(src: Path) -> bool:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if not _is_mp4_finalized(src):
        print(f"[skip] {src.name}: not finalized (no moov atom)")
        return False
    tmp = src.with_suffix(src.suffix + ".h264.mp4")
    print(f"[ffmpeg] {src.name} -> H.264 (yuv420p, faststart)")
    r = subprocess.run(
        [ff, "-y", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
         str(tmp)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[fail] {src.name}: {r.stderr[-400:]}")
        try: tmp.unlink()
        except OSError: pass
        return False
    os.replace(tmp, src)
    print(f"[ok]   {src.name} {src.stat().st_size // 1024} KB")
    return True


def main():
    args = sys.argv[1:] or ["detection_pipeline/out/annotated"]
    targets: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            targets.extend(sorted(p.glob("*.mp4")))
        elif p.is_file():
            targets.append(p)
    for t in targets:
        transcode(t)


if __name__ == "__main__":
    main()
