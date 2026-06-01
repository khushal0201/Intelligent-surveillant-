"""Re-ID using a Keras image backbone (MobileNetV3Large, ImageNet weights).

Extracts a global pooled feature from each person crop, L2-normalises it, and
maintains a cosine-similarity gallery. Returns (visitor_id, is_reentry, sim).
"""
from __future__ import annotations
from dataclasses import dataclass
import threading
import uuid
import numpy as np
import cv2

import keras
from keras.applications import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input


def _new_visitor_id() -> str:
    return "VIS_" + uuid.uuid4().hex[:6]


@dataclass
class GalleryEntry:
    visitor_id: str
    embedding: np.ndarray
    n: int = 1
    last_seen_ms: int = 0
    exited: bool = False


class _KerasEmbedder:
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.model: keras.Model = MobileNetV3Large(
            input_shape=(input_size, input_size, 3),
            include_top=False, weights="imagenet", pooling="avg",
        )
        self.dim = self.model.output_shape[-1]

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_LINEAR)
        x = preprocess_input(rgb.astype(np.float32))[None, ...]
        feat = self.model.predict(x, verbose=0)[0]
        n = float(np.linalg.norm(feat) + 1e-8)
        return (feat / n).astype(np.float32)


class _HistEmbedder:
    dim = 256

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        v = hist.flatten().astype(np.float32)
        n = float(np.linalg.norm(v) + 1e-8)
        return v / n


class ReIDGallery:
    def __init__(self, match_threshold: float = 0.78,
                 use_keras_backbone: bool = True):
        self.match_threshold = match_threshold
        self.entries: list[GalleryEntry] = []
        self._lock = threading.Lock()
        self._track_to_visitor: dict[tuple[str, int], str] = {}
        self._track_emb: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
        try:
            self.embedder = _KerasEmbedder() if use_keras_backbone else _HistEmbedder()
        except Exception as e:  # noqa: BLE001
            print(f"[reid] Keras backbone unavailable ({e}); using histogram fallback.")
            self.embedder = _HistEmbedder()

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.embedder(crop_bgr)

    def update_track(self, camera_id: str, track_id: int,
                     crop_bgr: np.ndarray) -> np.ndarray:
        key = (camera_id, track_id)
        emb = self.embed(crop_bgr)
        prev = self._track_emb.get(key)
        if prev is None:
            self._track_emb[key] = (emb, 1)
            return emb
        avg, n = prev
        new = (avg * n + emb) / (n + 1)
        new = new / (np.linalg.norm(new) + 1e-8)
        self._track_emb[key] = (new, n + 1)
        return new

    def assign_visitor(self, camera_id: str, track_id: int,
                       crop_bgr: np.ndarray, ts_ms: int
                       ) -> tuple[str, bool, float]:
        key = (camera_id, track_id)
        if key in self._track_to_visitor:
            return self._track_to_visitor[key], False, 1.0

        emb = self.update_track(camera_id, track_id, crop_bgr)
        with self._lock:
            best_idx, best_sim = -1, -1.0
            for i, e in enumerate(self.entries):
                sim = float(np.dot(emb, e.embedding))
                if sim > best_sim:
                    best_idx, best_sim = i, sim

            is_reentry = False
            if best_idx >= 0 and best_sim >= self.match_threshold:
                e = self.entries[best_idx]
                e.embedding = (e.embedding * e.n + emb) / (e.n + 1)
                e.embedding /= (np.linalg.norm(e.embedding) + 1e-8)
                e.n += 1
                is_reentry = e.exited
                e.exited = False
                e.last_seen_ms = ts_ms
                vid = e.visitor_id
            else:
                vid = _new_visitor_id()
                self.entries.append(GalleryEntry(
                    visitor_id=vid, embedding=emb, n=1, last_seen_ms=ts_ms,
                ))

        self._track_to_visitor[key] = vid
        return vid, is_reentry, best_sim

    def mark_exit(self, visitor_id: str) -> None:
        with self._lock:
            for e in self.entries:
                if e.visitor_id == visitor_id:
                    e.exited = True
                    return
