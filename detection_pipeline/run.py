"""Main entry: process all configured cameras and emit events to JSONL.

Usage (from detection_pipeline/):
    python run.py \
        --footage-dir "C:\\Users\\v-khmakhija\\OneDrive - Microsoft\\Documents\\DS\\Purplle\\resources\\CCTV Footage-20260529T160731Z-3-00144614ea\\CCTV Footage" \
        --pos-csv    "C:\\Users\\v-khmakhija\\OneDrive - Microsoft\\Documents\\DS\\Purplle\\resources\\Brigade_Bangalore_10_April_26 (1)bc6219c.csv" \
        --layout     "configs/store_layout.json" \
        --out        "out/events.jsonl"
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from jsonschema import Draft7Validator
from tqdm import tqdm

from pipeline.detector import PersonTracker
from pipeline.zones import ZoneSet
from pipeline.reid import ReIDGallery
from pipeline.staff import build_classifier
from pipeline.events import CameraEventEmitter
from pipeline.pos import POSCorrelator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--footage-dir", required=True)
    p.add_argument("--pos-csv", default=None)
    p.add_argument("--layout", default="configs/store_layout.json")
    p.add_argument("--schema", default="schema/event_schema.json")
    p.add_argument("--detector-preset", default="yolo_v8_m_pascalvoc",
                   help="keras-cv YOLOV8Detector preset (used when --detector-backend=kerascv).")
    p.add_argument("--detector-backend", default="kerascv",
                   choices=["kerascv", "onnx"],
                   help="Detector engine. 'onnx' = Ultralytics YOLOv8n via onnxruntime (~10x faster on CPU).")
    p.add_argument("--onnx-model-path", default="detection_pipeline/models/yolov8n.onnx",
                   help="Path to YOLOv8 ONNX model (used when --detector-backend=onnx).")
    p.add_argument("--detector-input-size", type=int, default=640,
                   help="Square input size for the detector. 640 = best accuracy; 416 ~2x faster; 320 ~4x faster.")
    p.add_argument("--staff-model", default=None,
                   help="Optional path to a trained Keras binary staff classifier (.keras).")
    p.add_argument("--staff-mode", default="prototype",
                   choices=["prototype", "openclip", "clip", "model", "heuristic"],
                   help="Staff classifier: prototype (MobileNet centroid), openclip (CLIP image+text, recommended), clip (Keras CLIP zero-shot), model (use --staff-model), heuristic (always False).")
    p.add_argument("--out", default="out/events.jsonl")
    p.add_argument("--stride", type=int, default=2,
                   help="Process every Nth frame (>=1).")
    p.add_argument("--cameras", nargs="*", default=None,
                   help="Optional subset of camera_ids to process.")
    p.add_argument("--reid-threshold", type=float, default=0.78)
    p.add_argument("--conf-thresh", type=float, default=0.40)
    p.add_argument("--staff-every-n", type=int, default=5,
                   help="Run the (expensive) CLIP staff classifier every Nth frame per visitor.")
    p.add_argument("--staff-ref-dir", default=None,
                   help="(prototype mode) folder of reference staff crops; each image becomes a staff anchor.")
    p.add_argument("--customer-ref-dir", default=None,
                   help="(prototype mode) folder of reference customer crops; each image becomes a customer anchor.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 = all. Else cap per clip for quick smoke runs.")
    p.add_argument("--show", action="store_true",
                   help="Open a live OpenCV preview window with detections, tracks and zones.")
    p.add_argument("--show-scale", type=float, default=0.6,
                   help="Scale factor for the preview window (0..1).")
    p.add_argument("--save-video-dir", default=None,
                   help="If set, write an annotated mp4 per camera into this directory "
                        "(<camera_id>.mp4) with detection boxes, IDs and zones drawn on each frame.")
    p.add_argument("--preview-jpg-path", default=None,
                   help="If set, the latest annotated frame is continuously written to this "
                        "JPG file (atomic replace). Used by the upload UI for live preview.")
    p.add_argument("--preview-every-n", type=int, default=4,
                   help="Write the preview JPG every N processed frames.")
    return p.parse_args()


def _draw_overlay(frame, fr, zoneset, camera_id, frame_count,
                  track_to_vid=None, vid_to_label=None,
                  vid_to_demo=None,
                  recent_events=None):
    """Draw zones, entry line, person boxes (with STAFF/CUST label) and a
    rolling list of the last few fired events on a frame copy."""
    img = frame.copy()
    h, w = img.shape[:2]

    # zones
    for name, poly in zoneset.polys.items():
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly.exterior.coords],
                       dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 180, 255))
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [pts], True, (0, 180, 255), 2)
        cx = int(pts[:, 0].mean()); cy = int(pts[:, 1].mean())
        cv2.putText(img, name, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 180, 255), 2, cv2.LINE_AA)

    # entry line
    if zoneset.entry_line is not None:
        el = zoneset.entry_line
        p1 = (int(el.p1[0] * w), int(el.p1[1] * h))
        p2 = (int(el.p2[0] * w), int(el.p2[1] * h))
        cv2.line(img, p1, p2, (0, 255, 0), 3)
        cv2.putText(img, "ENTRY LINE", (p1[0] + 6, p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # tracks
    track_to_vid = track_to_vid or {}
    vid_to_label = vid_to_label or {}
    vid_to_demo = vid_to_demo or {}
    for tb in fr.tracks:
        x1, y1, x2, y2 = map(int, (tb.x1, tb.y1, tb.x2, tb.y2))
        vid = track_to_vid.get(tb.track_id)
        is_staff = vid_to_label.get(vid, False)
        role_str = "STAFF" if is_staff else "CUST"
        color = (0, 0, 220) if is_staff else (255, 120, 0)  # red staff / blue cust
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        vid_short = (vid or f"id{tb.track_id}")[-6:]
        demo = vid_to_demo.get(vid) if vid else None
        demo_str = ""
        if demo:
            g = demo.get("gender") or ""
            a = demo.get("age_bucket") or ""
            if g or a:
                demo_str = f" {g}{('|' if g and a else '')}{a}"
        label = f"{role_str}{demo_str} {vid_short} {tb.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1),
                      color, -1)
        cv2.putText(img, label, (x1 + 3, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.circle(img, (int(tb.cx), int(tb.foot_y)), 4, (0, 0, 255), -1)

    return img


def load_layout(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_validator(schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    v = Draft7Validator(schema)
    def validate(evt):
        errs = sorted(v.iter_errors(evt), key=lambda e: e.path)
        if errs:
            raise ValueError("; ".join(e.message for e in errs))
    return validate


def main() -> None:
    args = parse_args()
    layout = load_layout(args.layout)
    store_id = layout["store_id"]
    sku_zone_map = layout.get("sku_zone_map", {})
    clip_starts = layout.get("clip_start_times_utc", {})
    cameras_cfg = layout["cameras"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # buffer all events; at end of run we apply per-visitor majority-vote
    # is_staff (from CLIP frame votes), then write the JSONL to disk.
    buffered: list[dict] = []
    # rolling per-camera ticker of recent events for the live preview
    from collections import deque
    recent_events: deque = deque(maxlen=8)
    cur_camera_id = {"id": None}

    def sink(evt):
        buffered.append(evt)
        if cur_camera_id["id"] == evt.get("camera_id"):
            tag = "S" if evt.get("is_staff") else "C"
            zone = evt.get("zone") or ""
            vid = (evt.get("visitor_id") or "")[-6:]
            recent_events.appendleft(
                f"{evt['event_type']:<22} {tag} {vid} {zone}")

    validator = load_validator(args.schema)

    print("[init] loading detector + tracker (Keras YOLOv8)...")
    tracker = PersonTracker(preset=args.detector_preset,
                            conf_thresh=args.conf_thresh,
                            input_size=args.detector_input_size,
                            backend=args.detector_backend,
                            onnx_model_path=args.onnx_model_path)
    print("[init] loading Re-ID gallery (MobileNetV3Large)...")
    gallery = ReIDGallery(match_threshold=args.reid_threshold)
    print(f"[init] loading staff classifier (mode={args.staff_mode})...")
    # Reuse the Re-ID embedder for prototype mode — avoids loading a second
    # MobileNet, and the embedding space is consistent across the pipeline.
    staff = build_classifier(args.staff_model, prefer=args.staff_mode,
                             embedder=gallery.embedder,
                             store_id=store_id)
    proto_mode = args.staff_mode in ("prototype", "openclip")
    if proto_mode and args.staff_ref_dir:
        staff.add_reference_crops(args.staff_ref_dir)
    if proto_mode and args.customer_ref_dir:
        staff.add_customer_reference_crops(args.customer_ref_dir)

    pos = POSCorrelator(args.pos_csv) if args.pos_csv else None

    footage_dir = Path(args.footage_dir)
    target_cams = set(args.cameras) if args.cameras else None

    # Process staff-only cameras (e.g. CAM 4 back-room) first so the prototype
    # classifier has anchor embeddings before classifying floor visitors.
    cam_items = sorted(
        cameras_cfg.items(),
        key=lambda kv: 0 if kv[1].get("role") == "staff_only" else 1,
    )

    for camera_id, cam in cam_items:
        if target_cams and camera_id not in target_cams:
            continue
        clip_name = cam["source_clip"]
        clip_path = footage_dir / clip_name
        if not clip_path.exists():
            print(f"[skip] {camera_id}: clip not found at {clip_path}")
            continue

        start_iso = clip_starts.get(clip_name, "2026-04-10T06:30:00Z")
        clip_start_utc = datetime.strptime(
            start_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

        zoneset = ZoneSet(zones=cam.get("zones", {}),
                          entry_line=cam.get("entry_line"))
        cam_role = cam.get("role", "floor")
        is_staff_only_cam = cam_role == "staff_only"
        staff_zone_names = set(cam.get("staff_zones", []))
        emitter = CameraEventEmitter(
            store_id=store_id, camera_id=camera_id,
            role=cam_role,
            sku_zone_map=sku_zone_map,
            clip_start_utc=clip_start_utc,
            sink=sink, schema_validator=validator,
        )

        # cache staff decision per (camera, track_id) and is_staff per visitor
        staff_cache: dict[int, tuple[bool, float]] = {}
        is_staff_by_visitor: dict[str, bool] = {}
        demo_by_visitor: dict[str, dict] = {}
        visitor_obs_count: dict[str, int] = {}
        track_to_vid: dict[int, str] = {}
        cur_camera_id["id"] = camera_id
        recent_events.clear()

        print(f"[run] {camera_id} <- {clip_path.name}")
        win_name = f"purplle | {camera_id}" if args.show else None
        if args.show:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # Probe source FPS so the annotated mp4 plays at real-time.
        writer = None
        writer_path = None
        # When saving video, stream every source frame so the writer outputs
        # at native fps; detection still runs only every `stride` frames.
        dense_stream = bool(args.save_video_dir or args.preview_jpg_path)
        if args.save_video_dir:
            Path(args.save_video_dir).mkdir(parents=True, exist_ok=True)
            writer_path = Path(args.save_video_dir) / f"{camera_id}.mp4"
            cap_probe = cv2.VideoCapture(str(clip_path))
            src_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
            cap_probe.release()
            out_fps = src_fps  # smooth real-time playback
        frame_count = 0
        user_quit = False
        for fr in tqdm(tracker.stream(clip_path, stride=args.stride,
                                      dense=dense_stream),
                       desc=camera_id):
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                break

            # In dense mode, only run staff/event logic on inference frames;
            # carry-over frames just get the previous boxes drawn for smooth
            # playback.
            run_logic = getattr(fr, "is_inference", True)

            # queue depth in BILLING
            queue_depth = None
            if run_logic and "BILLING" in zoneset.polys:
                qd = 0
                for tb in fr.tracks:
                    nx = tb.cx / fr.width
                    ny = tb.foot_y / fr.height
                    if zoneset.zone_at(nx, ny) == "BILLING":
                        qd += 1
                queue_depth = qd

            if not run_logic:
                # Carry-over frame: skip detection/staff/event work, just
                # render the cached overlay onto the new background.
                if args.save_video_dir or args.show or args.preview_jpg_path:
                    vis = _draw_overlay(fr.frame, fr, zoneset, camera_id, frame_count,
                                        track_to_vid=track_to_vid,
                                        vid_to_label=is_staff_by_visitor,
                                        vid_to_demo=demo_by_visitor,
                                        recent_events=list(recent_events))
                    if args.save_video_dir:
                        if writer is None:
                            h, w = vis.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(str(writer_path), fourcc, out_fps, (w, h))
                        writer.write(vis)
                    if args.preview_jpg_path and frame_count % max(1, args.preview_every_n) == 0:
                        try:
                            prev_path = Path(args.preview_jpg_path)
                            prev_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp = prev_path.with_suffix(".tmp.jpg")
                            cv2.imwrite(str(tmp), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            tmp.replace(prev_path)
                        except Exception:
                            pass
                    if args.show:
                        if args.show_scale and args.show_scale != 1.0:
                            vis = cv2.resize(vis, None, fx=args.show_scale, fy=args.show_scale)
                        cv2.imshow(win_name, vis)
                        cv2.waitKey(1)
                continue

            for tb in fr.tracks:
                crop = tb.crop(fr.frame)
                vid, is_reentry, _sim = gallery.assign_visitor(
                    camera_id, tb.track_id, crop, fr.timestamp_ms)
                track_to_vid[tb.track_id] = vid

                # Per-frame staff vote; final is_staff resolved at end-of-run.
                # On staff-only cameras (e.g. back-room) any visitor is staff
                # by definition — give an unambiguous vote.
                visitor_obs_count[vid] = visitor_obs_count.get(vid, 0) + 1
                # Track staff-zone dwell (geometry anchor); a visitor sitting
                # behind the cash counter / vanity counter for >=N samples
                # gets registered as a staff prototype anchor.
                nx = tb.cx / fr.width
                ny = tb.foot_y / fr.height
                cur_zone = zoneset.zone_at(nx, ny)
                if (proto_mode
                        and cur_zone in staff_zone_names):
                    staff.add_zone_anchor(crop)
                if proto_mode:
                    # Build prototype: feed every Nth crop with the camera role.
                    if visitor_obs_count[vid] % max(1, args.staff_every_n) == 1:
                        staff.observe_crop(vid, crop, cam_role)
                    # Best-effort live label using prototypes built so far.
                    if hasattr(staff, "live_vote"):
                        is_staff_flag = staff.live_vote(vid) or is_staff_only_cam
                    else:
                        is_staff_flag = True if is_staff_only_cam else False
                    if hasattr(staff, "live_demographics"):
                        d = staff.live_demographics(vid)
                        if d.get("gender") or d.get("age_bucket"):
                            demo_by_visitor[vid] = d
                elif is_staff_only_cam:
                    is_staff_flag = True
                    staff.observe(vid, True)
                elif visitor_obs_count[vid] % max(1, args.staff_every_n) == 1:
                    is_staff_flag, _p = staff.is_staff(crop)
                    staff.observe(vid, is_staff_flag)
                else:
                    # Reuse the visitor's last vote without re-running the model.
                    is_staff_flag = is_staff_by_visitor.get(vid, False)
                # Geometry override: anyone standing in a designated staff_zone
                # (e.g. behind the cash counter) is staff regardless of CLIP;
                # anyone standing in the customer-side BILLING queue strip is
                # a customer for *this* observation regardless of CLIP, so
                # BILLING_QUEUE_JOIN doesn't get suppressed by a stale STAFF vote.
                if cur_zone in staff_zone_names:
                    is_staff_flag = True
                elif cam_role == "billing" and cur_zone == "BILLING":
                    is_staff_flag = False
                is_staff_by_visitor[vid] = is_staff_flag

                # nx/ny were already computed above for the staff-zone test

                nx = tb.cx / fr.width
                ny = tb.foot_y / fr.height

                # ENTRY/EXIT line
                if zoneset.entry_line is not None:
                    side = zoneset.entry_line.side(nx, ny)
                    emitter.on_entry_line(
                        visitor_id=vid, ts_ms=fr.timestamp_ms,
                        side=side, is_staff=is_staff_flag,
                        confidence=tb.conf, is_reentry=is_reentry,
                        demographics=demo_by_visitor.get(vid))
                    # mark gallery exit so a subsequent visit becomes REENTRY
                    if emitter.visitors[vid].inside_store is False and side < 0:
                        gallery.mark_exit(vid)

                # Zones
                emitter.on_zone_observation(
                    visitor_id=vid, ts_ms=fr.timestamp_ms,
                    zone=cur_zone, is_staff=is_staff_flag,
                    confidence=tb.conf, queue_depth=queue_depth,
                    demographics=demo_by_visitor.get(vid))

            if args.show or args.save_video_dir or args.preview_jpg_path:
                vis = _draw_overlay(fr.frame, fr, zoneset, camera_id, frame_count,
                                    track_to_vid=track_to_vid,
                                    vid_to_label=is_staff_by_visitor,
                                    vid_to_demo=demo_by_visitor,
                                    recent_events=list(recent_events))
                if args.save_video_dir:
                    if writer is None:
                        h, w = vis.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(writer_path), fourcc, out_fps, (w, h))
                    writer.write(vis)
                if args.preview_jpg_path and frame_count % max(1, args.preview_every_n) == 0:
                    try:
                        prev_path = Path(args.preview_jpg_path)
                        prev_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = prev_path.with_suffix(".tmp.jpg")
                        cv2.imwrite(str(tmp), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        tmp.replace(prev_path)
                    except Exception:
                        pass
                if args.show:
                    if args.show_scale and args.show_scale != 1.0:
                        vis = cv2.resize(vis, None, fx=args.show_scale, fy=args.show_scale)
                    if frame_count == 1:
                        vh, vw = vis.shape[:2]
                        cv2.resizeWindow(win_name, vw, vh)
                    cv2.imshow(win_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):  # q or ESC
                        user_quit = True
                        break

        if args.show:
            cv2.destroyWindow(win_name)
        if writer is not None:
            writer.release()
            print(f"[save] {camera_id} annotated -> {writer_path}")
            # OpenCV's mp4v fourcc produces MPEG-4 Part 2, which browsers
            # don't support. Transcode to H.264 in place so the dashboard
            # can play the file. Falls back gracefully if ffmpeg is missing.
            try:
                from transcode_to_h264 import transcode  # type: ignore
                transcode(writer_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[save] {camera_id} h264 transcode skipped: {exc}")
        emitter.consume_pending_billing_exits(pos, is_staff_by_visitor)
        if user_quit:
            print("[run] user quit; stopping after current camera.")
            break

    # Resolve final is_staff per visitor via majority vote across all frames,
    # then write events in chronological order.
    if proto_mode and hasattr(staff, "finalize"):
        staff.finalize()
    final_is_staff = {vid: staff.vote(vid)
                      for vid in {e["visitor_id"] for e in buffered}}
    final_demo: dict[str, dict] = {}
    if hasattr(staff, "demographics"):
        for vid in {e["visitor_id"] for e in buffered}:
            final_demo[vid] = staff.demographics(vid)
    n_staff = sum(1 for v in final_is_staff.values() if v)
    print(f"[staff] {n_staff}/{len(final_is_staff)} visitors classified as staff")

    buffered.sort(key=lambda e: (e["timestamp"], e["camera_id"]))
    with out_path.open("w", encoding="utf-8") as out_f:
        for evt in buffered:
            evt["is_staff"] = bool(final_is_staff.get(evt["visitor_id"], evt["is_staff"]))
            d = final_demo.get(evt["visitor_id"]) if final_demo else None
            if d:
                meta = evt.setdefault("metadata", {})
                if d.get("gender"):
                    meta["gender"] = d["gender"]
                if d.get("age_bucket"):
                    meta["age_bucket"] = d["age_bucket"]
            out_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    print(f"[done] {len(buffered)} events written to {out_path}")


if __name__ == "__main__":
    main()
