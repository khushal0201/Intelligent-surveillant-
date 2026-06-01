"""Person detection (keras-cv YOLOv8) + lightweight IoU tracker.

No PyTorch. Streams frames from a video and yields TrackBox lists with stable
track_ids assigned by a simple IoU + Hungarian matcher.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal
import cv2
import numpy as np

import keras
import keras_cv
from scipy.optimize import linear_sum_assignment


# PASCAL VOC class index for "person" (used by keras-cv preset).
PERSON_CLASS_IDX = 14
# COCO class index for "person" (used by Ultralytics YOLOv8 ONNX export).
PERSON_CLASS_IDX_COCO = 0


@dataclass
class TrackBox:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = PERSON_CLASS_IDX

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def foot_y(self) -> float:
        return self.y2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(self.x1)); y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2)); y2 = min(h, int(self.y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_ms: int
    width: int
    height: int
    frame: np.ndarray
    tracks: list[TrackBox]
    # True if detection+tracking ran on this frame; False if `tracks` is
    # carried over from the previous inference frame (used by `dense=True`).
    is_inference: bool = True


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class _LiveTrack:
    track_id: int
    box: tuple[float, float, float, float]
    conf: float
    misses: int = 0
    last_seen_frame: int = 0


class IouTracker:
    """Online IoU + Hungarian-assignment tracker.

    - Existing tracks are matched to new detections by 1 - IoU cost.
    - A track survives `track_buffer` consecutive misses before deletion.
    - A new detection above `new_track_thresh` starts a fresh track_id.
    """

    def __init__(self, iou_thresh: float = 0.3,
                 new_track_thresh: float = 0.5,
                 track_buffer: int = 30):
        self.iou_thresh = iou_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self._tracks: list[_LiveTrack] = []
        self._next_id = 1

    def update(self, dets: list[tuple[tuple[float, float, float, float], float]],
               frame_idx: int) -> list[TrackBox]:
        if not self._tracks and not dets:
            return []
        if not self._tracks:
            for box, conf in dets:
                if conf >= self.new_track_thresh:
                    self._tracks.append(_LiveTrack(
                        track_id=self._next_id, box=box, conf=conf,
                        last_seen_frame=frame_idx))
                    self._next_id += 1
            return self._snapshot()

        if not dets:
            for t in self._tracks:
                t.misses += 1
            self._tracks = [t for t in self._tracks if t.misses <= self.track_buffer]
            return self._snapshot()

        n_t, n_d = len(self._tracks), len(dets)
        cost = np.ones((n_t, n_d), dtype=np.float32)
        for i, t in enumerate(self._tracks):
            for j, (b, _) in enumerate(dets):
                cost[i, j] = 1.0 - _iou(t.box, b)
        row, col = linear_sum_assignment(cost)
        assigned_t, assigned_d = set(), set()
        for i, j in zip(row, col):
            if 1.0 - cost[i, j] >= self.iou_thresh:
                t = self._tracks[i]
                t.box, t.conf = dets[j]
                t.misses = 0
                t.last_seen_frame = frame_idx
                assigned_t.add(i); assigned_d.add(j)

        for i, t in enumerate(self._tracks):
            if i not in assigned_t:
                t.misses += 1

        for j, (box, conf) in enumerate(dets):
            if j not in assigned_d and conf >= self.new_track_thresh:
                self._tracks.append(_LiveTrack(
                    track_id=self._next_id, box=box, conf=conf,
                    last_seen_frame=frame_idx))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.track_buffer]
        return self._snapshot()

    def _snapshot(self) -> list[TrackBox]:
        out = []
        for t in self._tracks:
            if t.misses == 0:
                out.append(TrackBox(
                    track_id=t.track_id,
                    x1=t.box[0], y1=t.box[1], x2=t.box[2], y2=t.box[3],
                    conf=t.conf,
                ))
        return out


class PersonTracker:
    """YOLOv8 detector (keras-cv or ONNX) + IouTracker, frame-streaming API."""

    def __init__(self,
                 preset: str = "yolo_v8_m_pascalvoc",
                 conf_thresh: float = 0.40,
                 iou_thresh: float = 0.5,
                 input_size: int = 640,
                 min_box_area_frac: float = 0.004,
                 min_aspect_ratio: float = 1.2,
                 max_box_area_frac: float = 0.6,
                 backend: Literal["kerascv", "onnx"] = "kerascv",
                 onnx_model_path: str | None = None):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.min_box_area_frac = min_box_area_frac
        self.min_aspect_ratio = min_aspect_ratio
        self.max_box_area_frac = max_box_area_frac
        self.backend = backend

        if backend == "kerascv":
            self.detector: keras.Model = keras_cv.models.YOLOV8Detector.from_preset(
                preset,
                bounding_box_format="xyxy",
                prediction_decoder=keras_cv.layers.NonMaxSuppression(
                    bounding_box_format="xyxy",
                    from_logits=True,
                    iou_threshold=iou_thresh,
                    confidence_threshold=conf_thresh,
                ),
            )
            self._person_idx = PERSON_CLASS_IDX
        elif backend == "onnx":
            import onnxruntime as ort
            if onnx_model_path is None:
                raise ValueError("onnx backend requires onnx_model_path")
            providers = ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            so.intra_op_num_threads = 0  # let ORT pick all cores
            self.ort_session = ort.InferenceSession(onnx_model_path,
                                                    sess_options=so,
                                                    providers=providers)
            self._ort_input = self.ort_session.get_inputs()[0].name
            self._person_idx = PERSON_CLASS_IDX_COCO
        else:
            raise ValueError(f"unknown backend: {backend}")

        self.tracker = IouTracker()

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame_bgr.shape[:2]
        s = self.input_size
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32)
        return x[None, ...], w / s, h / s

    def _detect_persons(self, frame_bgr: np.ndarray
                        ) -> list[tuple[tuple[float, float, float, float], float]]:
        if self.backend == "onnx":
            return self._detect_persons_onnx(frame_bgr)
        x, sx, sy = self._preprocess(frame_bgr)
        preds = self.detector.predict(x, verbose=0)
        boxes = preds["boxes"][0]
        confs = preds["confidence"][0]
        classes = preds["classes"][0]
        n = int(np.array(preds.get("num_detections", [boxes.shape[0]]))[0])
        H, W = frame_bgr.shape[:2]
        frame_area = float(W * H)
        out = []
        for i in range(n):
            c = int(classes[i])
            if c != self._person_idx:
                continue
            conf = float(confs[i])
            if conf < self.conf_thresh:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            X1, Y1, X2, Y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            bw = max(0.0, X2 - X1)
            bh = max(0.0, Y2 - Y1)
            if bw <= 1.0 or bh <= 1.0:
                continue
            area_frac = (bw * bh) / frame_area
            if area_frac < self.min_box_area_frac:
                continue                    # too small (poster face, distant glare)
            if area_frac > self.max_box_area_frac:
                continue                    # spans most of the frame; almost always wrong
            if (bh / bw) < self.min_aspect_ratio:
                continue                    # not tall enough to be a standing person
            out.append(((X1, Y1, X2, Y2), conf))
        return out

    # --- ONNX (Ultralytics YOLOv8) path ----------------------------------
    def _letterbox(self, img: np.ndarray
                   ) -> tuple[np.ndarray, float, float, float]:
        """Resize to input_size with aspect-ratio preserved + gray padding.
        Returns (canvas, scale, pad_x, pad_y) where map back is
        x_orig = (x_canvas - pad_x) / scale."""
        s = self.input_size
        h, w = img.shape[:2]
        r = min(s / h, s / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_x = (s - nw) // 2
        pad_y = (s - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, r, float(pad_x), float(pad_y)

    def _detect_persons_onnx(self, frame_bgr: np.ndarray
                              ) -> list[tuple[tuple[float, float, float, float], float]]:
        canvas, r, px, py = self._letterbox(frame_bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
        out = self.ort_session.run(None, {self._ort_input: x})[0]
        # YOLOv8 ONNX output: (1, 84, 8400) -> 4 box (cx,cy,w,h) + 80 class scores
        pred = out[0].T  # (8400, 84)
        cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        confs_all = cls_scores.max(axis=1)
        keep = (cls_ids == self._person_idx) & (confs_all >= self.conf_thresh)
        if not np.any(keep):
            return []
        boxes_xywh = pred[keep, :4]
        confs = confs_all[keep]
        # cx,cy,w,h (canvas pixels) -> x1,y1,x2,y2 in original frame
        cx = boxes_xywh[:, 0]; cy = boxes_xywh[:, 1]
        bw = boxes_xywh[:, 2]; bh = boxes_xywh[:, 3]
        x1 = (cx - bw / 2 - px) / r
        y1 = (cy - bh / 2 - py) / r
        x2 = (cx + bw / 2 - px) / r
        y2 = (cy + bh / 2 - py) / r
        # class-agnostic NMS over the person boxes
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        idxs = cv2.dnn.NMSBoxes(
            bboxes=[[float(b[0]), float(b[1]),
                     float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_xyxy],
            scores=confs.astype(np.float32).tolist(),
            score_threshold=float(self.conf_thresh),
            nms_threshold=float(self.iou_thresh),
        )
        if isinstance(idxs, np.ndarray):
            idxs = idxs.flatten().tolist()
        elif isinstance(idxs, tuple):
            idxs = list(idxs)
        H, W = frame_bgr.shape[:2]
        frame_area = float(W * H)
        out_list = []
        for i in idxs:
            X1, Y1, X2, Y2 = boxes_xyxy[i]
            X1 = max(0.0, float(X1)); Y1 = max(0.0, float(Y1))
            X2 = min(float(W), float(X2)); Y2 = min(float(H), float(Y2))
            bw_ = max(0.0, X2 - X1); bh_ = max(0.0, Y2 - Y1)
            if bw_ <= 1.0 or bh_ <= 1.0:
                continue
            area_frac = (bw_ * bh_) / frame_area
            if area_frac < self.min_box_area_frac or area_frac > self.max_box_area_frac:
                continue
            if (bh_ / bw_) < self.min_aspect_ratio:
                continue
            out_list.append(((X1, Y1, X2, Y2), float(confs[i])))
        return out_list

    def stream(self, video_path: str | Path,
               stride: int = 1,
               dense: bool = False) -> Iterator[FrameResult]:
        """Yield FrameResults from a video.

        - dense=False (default): legacy behaviour — yield only every Nth frame
          (one per inference). Cheapest, choppy when used to write videos.
        - dense=True: read every source frame, run detection+tracking only
          every Nth frame; in between, yield a FrameResult with the previous
          tracks (is_inference=False). Lets callers write smooth real-time
          video while only paying inference cost every Nth frame.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        idx = -1
        last_tracks: list[TrackBox] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                run_inf = (stride <= 1) or (idx % stride == 0)
                if not run_inf and not dense:
                    continue
                ts_ms = int(idx * 1000.0 / fps)
                if run_inf:
                    dets = self._detect_persons(frame)
                    last_tracks = self.tracker.update(dets, frame_idx=idx)
                yield FrameResult(
                    frame_idx=idx, timestamp_ms=ts_ms,
                    width=w, height=h, frame=frame, tracks=last_tracks,
                    is_inference=run_inf,
                )
        finally:
            cap.release()
