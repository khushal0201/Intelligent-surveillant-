"""Staff classification using Keras-Hub CLIP zero-shot (no torch, no labels).

Per frame, the person crop is encoded by CLIP and compared against staff vs
customer text prompts. The pipeline calls `vote(visitor_id)` at end-of-clip
to get a stable majority decision per visitor; intermediate frame decisions
are tallied via `observe()`.

If CLIP fails to load (offline / weights missing), falls back to a heuristic
that always returns False — events still emit, schema stays valid.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from .clip_keras import KerasCLIP


_STAFF_PROMPTS = [
    "a retail store employee in a plain black uniform t-shirt",
    "a beauty store sales associate wearing all black work clothes",
    "a store staff member behind the counter holding a phone or scanner",
    "a salesperson in a black uniform with a name badge and lanyard",
    "a beauty advisor in dark uniform attending a customer at a vanity counter",
]
_CUSTOMER_PROMPTS = [
    "a female customer shopping in a beauty store, carrying a handbag, wearing colourful casual clothes",
    "a shopper browsing skincare products, wearing a kurti or printed top",
    "a male customer in a casual shirt and jeans browsing cosmetics",
    "a shopper in patterned everyday street clothing holding a product",
    "a customer wearing a saree or salwar-kameez looking at beauty products",
]


class CLIPZeroShotStaff:
    """Zero-shot staff vs customer classifier using Keras-Hub CLIP."""

    def __init__(self, threshold: float = 0.50,
                 staff_prompts: list[str] | None = None,
                 customer_prompts: list[str] | None = None):
        self.threshold = threshold
        self.clip = KerasCLIP()
        staff_prompts = staff_prompts or _STAFF_PROMPTS
        customer_prompts = customer_prompts or _CUSTOMER_PROMPTS
        s = self.clip.encode_text(staff_prompts).mean(axis=0)
        c = self.clip.encode_text(customer_prompts).mean(axis=0)
        s = s / (np.linalg.norm(s) + 1e-8)
        c = c / (np.linalg.norm(c) + 1e-8)
        self._staff_vec = s.astype(np.float32)
        self._cust_vec = c.astype(np.float32)
        self._votes: dict[str, list[int]] = defaultdict(list)

    def is_staff(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return False, 0.0
        feat = self.clip.encode_image([crop_bgr])[0]
        s = float(np.dot(feat, self._staff_vec))
        c = float(np.dot(feat, self._cust_vec))
        tau = 100.0
        es, ec = np.exp(tau * s), np.exp(tau * c)
        p_staff = float(es / (es + ec))
        return p_staff >= self.threshold, p_staff

    def observe(self, visitor_id: str, is_staff_frame: bool) -> None:
        self._votes[visitor_id].append(1 if is_staff_frame else 0)

    def vote(self, visitor_id: str) -> bool:
        v = self._votes.get(visitor_id, [])
        if not v:
            return False
        return (sum(v) / len(v)) >= 0.5


class HeuristicStaffClassifier:
    """Fallback when CLIP can't load; everyone is treated as customer."""

    def is_staff(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        return False, 0.5

    def observe(self, visitor_id: str, is_staff_frame: bool) -> None:
        return

    def vote(self, visitor_id: str) -> bool:
        return False


class KerasStaffClassifier:
    """Optional: a trained Keras binary classifier (.keras file)."""

    def __init__(self, model_path: str | Path, threshold: float = 0.55,
                 input_size: int = 224):
        import keras
        self.model = keras.models.load_model(str(model_path))
        self.threshold = threshold
        self.input_size = input_size
        self._votes: dict[str, list[int]] = defaultdict(list)

    def is_staff(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return False, 0.0
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        x = (rgb.astype(np.float32) / 255.0)[None, ...]
        p = float(np.array(self.model.predict(x, verbose=0)).squeeze())
        return p >= self.threshold, p

    def observe(self, visitor_id: str, is_staff_frame: bool) -> None:
        self._votes[visitor_id].append(1 if is_staff_frame else 0)

    def vote(self, visitor_id: str) -> bool:
        v = self._votes.get(visitor_id, [])
        return bool(v) and (sum(v) / len(v)) >= 0.5


def build_classifier(model_path: Optional[str | Path] = None,
                     prefer: str = "clip",
                     embedder=None):
    """Factory.

    prefer:
      - "prototype" -> MobileNet prototype (auto staff prototype from CAM 4)
      - "clip"      -> CLIP zero-shot (uniform-aware, no labels)
      - "model"     -> load `model_path` as a trained Keras classifier
      - "heuristic" -> dummy fallback
    """
    if prefer == "prototype":
        if embedder is None:
            raise ValueError("prototype mode requires a Keras image embedder")
        return MobileNetPrototypeStaff(embedder)
    if prefer == "openclip":
        return OpenCLIPPrototypeStaff()
    if prefer == "model" and model_path:
        return KerasStaffClassifier(model_path)
    if prefer == "heuristic":
        return HeuristicStaffClassifier()
    try:
        return CLIPZeroShotStaff()
    except Exception as e:  # noqa: BLE001
        print(f"[staff] CLIP unavailable ({e}); falling back to heuristic.")
        return HeuristicStaffClassifier()


class MobileNetPrototypeStaff:
    """Staff vs customer using cosine to auto-built MobileNetV3 prototypes.

    Strategy
    --------
    During streaming, every person crop is reduced to a single L2-normalised
    MobileNetV3 vector (the same model the Re-ID gallery is using, so no extra
    backbone is loaded).  Each visitor accumulates a running mean embedding,
    plus the set of camera roles they were ever observed on
    ('staff_only', 'entry', 'floor', 'billing').

    At end of run we build prototypes:
      * staff prototype  = mean embedding of visitors ever seen on a
        staff_only camera (e.g. CAM_BACKROOM_04).  These are guaranteed staff.
      * customer prototype = mean embedding of visitors who were seen on the
        entry camera but NEVER on a staff_only camera.  These are customers
        crossing the entry line.

    For every visitor we then pick whichever prototype is more cosine-similar
    to the visitor's mean embedding.  Visitors ever on a staff_only camera
    are forced to staff regardless.

    If we don't have at least one example of each class we fall back gracefully
    (everyone unknown -> customer; if only staff seen -> only the CAM4 visitors
    are flagged staff).
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self._embs: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._roles: dict[str, set[str]] = defaultdict(set)
        # additional anchor embeddings collected from reference crops
        # and from "staff zone" geometry observations
        self._ref_anchors: list[np.ndarray] = []
        self._zone_anchors: list[np.ndarray] = []
        self._cust_ref_anchors: list[np.ndarray] = []
        self._final: dict[str, bool] = {}

    # streaming-time API ---------------------------------------------------
    def is_staff(self, crop_bgr):
        # No per-frame decision; finalisation happens after the whole run.
        return False, 0.5

    def observe(self, visitor_id, is_staff_frame):
        # Compatibility no-op; kept so existing run.py call sites don't break.
        return

    def observe_crop(self, visitor_id: str, crop_bgr: np.ndarray, role: str):
        if crop_bgr is None or crop_bgr.size == 0:
            return
        emb = self.embedder(crop_bgr)
        if emb is None:
            return
        c = self._counts.get(visitor_id, 0)
        prev = self._embs.get(visitor_id)
        new = emb if prev is None else (prev * c + emb) / (c + 1)
        n = float(np.linalg.norm(new) + 1e-8)
        self._embs[visitor_id] = (new / n).astype(np.float32)
        self._counts[visitor_id] = c + 1
        self._roles[visitor_id].add(role)

    def add_reference_crops(self, folder: Path | str) -> int:
        """Load every image in `folder` and add as a staff anchor."""
        return self._load_anchors(folder, self._ref_anchors, "staff")

    def add_customer_reference_crops(self, folder: Path | str) -> int:
        """Load every image in `folder` and add as a customer anchor."""
        return self._load_anchors(folder, self._cust_ref_anchors, "customer")

    def _load_anchors(self, folder, bucket, label):
        folder = Path(folder)
        n = 0
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for p in folder.glob(ext):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                emb = self.embedder(img)
                if emb is not None:
                    bucket.append(emb.astype(np.float32))
                    n += 1
        print(f"[staff/proto] loaded {n} reference {label} crops from {folder}")
        return n

    def add_zone_anchor(self, crop_bgr: np.ndarray) -> None:
        """Register a crop as a staff anchor (caller decided via geometry)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return
        emb = self.embedder(crop_bgr)
        if emb is not None:
            self._zone_anchors.append(emb.astype(np.float32))

    def _live_protos(self):
        """Build (staff_proto, cust_proto) from whatever anchors exist *now*.
        Used by `live_vote` to give an in-progress label during streaming."""
        def _mean(vecs):
            if not vecs:
                return None
            m = np.mean(np.stack(vecs, axis=0), axis=0)
            n = float(np.linalg.norm(m) + 1e-8)
            return (m / n).astype(np.float32)

        staff_ids = [v for v, r in self._roles.items() if "staff_only" in r]
        cust_ids = [v for v, r in self._roles.items()
                    if "staff_only" not in r and "entry" in r]
        staff_vecs = ([self._embs[v] for v in staff_ids]
                      + list(self._ref_anchors)
                      + list(self._zone_anchors))
        cust_vecs = ([self._embs[v] for v in cust_ids]
                     + list(self._cust_ref_anchors))
        return _mean(staff_vecs), _mean(cust_vecs)

    def live_vote(self, visitor_id: str) -> bool:
        """Best-effort staff/customer guess using prototypes built so far."""
        if visitor_id in self._final:
            return self._final[visitor_id]
        if "staff_only" in self._roles.get(visitor_id, set()):
            return True
        emb = self._embs.get(visitor_id)
        if emb is None:
            return False
        staff_p, cust_p = self._live_protos()
        if staff_p is None:
            return False
        s = float(np.dot(emb, staff_p))
        if cust_p is None:
            return s > 0.65
        c = float(np.dot(emb, cust_p))
        return s > c

    # finalisation ---------------------------------------------------------
    def finalize(self):
        staff_ids = [v for v, r in self._roles.items() if "staff_only" in r]
        cust_ids = [v for v, r in self._roles.items()
                    if "staff_only" not in r and "entry" in r]

        def proto(vecs):
            if not vecs:
                return None
            m = np.mean(np.stack(vecs, axis=0), axis=0)
            n = float(np.linalg.norm(m) + 1e-8)
            return (m / n).astype(np.float32)

        staff_vecs = ([self._embs[v] for v in staff_ids]
                      + list(self._ref_anchors)
                      + list(self._zone_anchors))
        cust_vecs = ([self._embs[v] for v in cust_ids]
                     + list(self._cust_ref_anchors))

        staff_proto = proto(staff_vecs)
        cust_proto = proto(cust_vecs)
        print(f"[staff/proto] staff_anchors=visitors:{len(staff_ids)}+refs:{len(self._ref_anchors)}+zones:{len(self._zone_anchors)} "
              f"cust_anchors=visitors:{len(cust_ids)}+refs:{len(self._cust_ref_anchors)}")

        for vid, emb in self._embs.items():
            if "staff_only" in self._roles[vid]:
                self._final[vid] = True
                print(f"[staff/proto]   {vid}: STAFF (staff_only camera)")
                continue
            if staff_proto is None:
                self._final[vid] = False
                continue
            s = float(np.dot(emb, staff_proto))
            if cust_proto is None:
                # No customer pole: use a moderate similarity threshold against
                # the staff prototype alone.
                decision = s > 0.65
                self._final[vid] = decision
                print(f"[staff/proto]   {vid}: s={s:.3f} -> "
                      f"{'STAFF' if decision else 'cust'} (no cust pole)")
                continue
            c = float(np.dot(emb, cust_proto))
            decision = s > c
            self._final[vid] = decision
            print(f"[staff/proto]   {vid}: s={s:.3f} c={c:.3f} -> "
                  f"{'STAFF' if decision else 'cust'}")

    def vote(self, visitor_id: str) -> bool:
        return self._final.get(visitor_id, False)


# ----------------------------------------------------------------------------
# OpenCLIP-based prototype classifier (best accuracy on uniform-vs-shopper).
# Uses ViT-B-32 (laion2b) to embed crops AND text prompts for both poles.
# ----------------------------------------------------------------------------

_OCLIP_STAFF_PROMPTS = [
    "a retail store employee in a fully black uniform: solid black shirt, solid black straight-fit trousers, and black ankle boots, no backpack and no handbag",
    "a beauty store sales associate wearing an all-black uniform, plain black shirt tucked into plain black pants, black boots on feet, name badge and lanyard, definitely not carrying a backpack or handbag",
    "a female salesperson in a plain black shirt and slim black trousers with black boots, not a saree, not a kurti, not harem pants, no bag of any kind, standing behind the counter",
    "a male salesperson in a plain black shirt and plain black formal trousers wearing black boots behind the cosmetics counter, no backpack and no shoulder bag",
    "a store staff member in solid black workwear holding a phone or barcode scanner at the billing counter, hands free, no bag on shoulders",
    "a female employee viewed from behind, wearing a plain black shirt and plain black trousers, hair tied back, no backpack straps over her shoulders and no handbag, working in the aisle"
]
_OCLIP_CUSTOMER_PROMPTS = [
    "a customer in a beauty store with a backpack on the shoulders, the two black backpack straps clearly visible running over both shoulders down the chest, browsing products in the aisle",
    "a shopper carrying a handbag or shoulder bag, even if dressed in a black shirt or black top, inspecting cosmetics on a display",
    "a customer with a backpack on the back, both shoulder straps visible from front or side, wearing any kind of clothing including black, looking at makeup testers from the customer side of the counter",
    "a person carrying a tote bag or handbag, possibly in dark clothing, standing in front of a shelf picking a product",
    "a shopper with a visible bag — backpack straps over the shoulders, or a handbag in hand, or a tote on the arm — browsing skincare or makeup, wearing casual sneakers or sandals, not store-uniform boots",
    "a female shopper viewed from behind in an all-black outfit (black top and black pants) with a backpack on her back, the backpack body visible between her shoulder blades and the straps clearly visible over both shoulders",
]


class OpenCLIPPrototypeStaff:
    """Staff vs customer using OpenCLIP image+text embeddings.

    Score for a visitor is a blended cosine:
        score_staff = a * cos(img, text_staff) + (1-a) * cos(img, ref_staff_mean)
        score_cust  = a * cos(img, text_cust)  + (1-a) * cos(img, ref_cust_mean)
        STAFF iff score_staff > score_cust + margin.

    Mirrors the API of MobileNetPrototypeStaff so run.py needs no changes
    in the per-frame loop.
    """

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 text_weight: float = 0.70,
                 margin: float = 0.0,
                 staff_prompts: list[str] | None = None,
                 customer_prompts: list[str] | None = None):
        import torch
        import open_clip
        self._torch = torch
        self._device = "cpu"
        print(f"[staff/openclip] loading {model_name} / {pretrained} on CPU...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_weight = float(text_weight)
        self.margin = float(margin)
        sp = staff_prompts or _OCLIP_STAFF_PROMPTS
        cp = customer_prompts or _OCLIP_CUSTOMER_PROMPTS
        self._text_staff = self._encode_text(sp)
        self._text_cust = self._encode_text(cp)
        # Per-visitor running mean image embedding.
        self._embs: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._roles: dict[str, set[str]] = defaultdict(set)
        self._ref_anchors: list[np.ndarray] = []
        self._zone_anchors: list[np.ndarray] = []
        self._cust_ref_anchors: list[np.ndarray] = []
        self._final: dict[str, bool] = {}
        print(f"[staff/openclip] ready. staff_prompts={len(sp)} "
              f"cust_prompts={len(cp)} text_weight={self.text_weight}")

    # ---- encoding helpers ----
    def _encode_text(self, prompts: list[str]) -> np.ndarray:
        toks = self.tokenizer(prompts).to(self._device)
        with self._torch.no_grad():
            t = self.model.encode_text(toks)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        m = t.mean(dim=0)
        m = m / (m.norm() + 1e-8)
        return m.cpu().numpy().astype(np.float32)

    def _encode_image(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            f = self.model.encode_image(x)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
        return f.squeeze(0).cpu().numpy().astype(np.float32)

    # ---- API matching MobileNetPrototypeStaff ----
    def is_staff(self, crop_bgr):
        return False, 0.5

    def observe(self, visitor_id, is_staff_frame):
        return

    def observe_crop(self, visitor_id: str, crop_bgr: np.ndarray, role: str):
        emb = self._encode_image(crop_bgr)
        if emb is None:
            return
        c = self._counts.get(visitor_id, 0)
        prev = self._embs.get(visitor_id)
        new = emb if prev is None else (prev * c + emb) / (c + 1)
        n = float(np.linalg.norm(new) + 1e-8)
        self._embs[visitor_id] = (new / n).astype(np.float32)
        self._counts[visitor_id] = c + 1
        self._roles[visitor_id].add(role)

    def add_reference_crops(self, folder) -> int:
        return self._load_anchors(folder, self._ref_anchors, "staff")

    def add_customer_reference_crops(self, folder) -> int:
        return self._load_anchors(folder, self._cust_ref_anchors, "customer")

    def _load_anchors(self, folder, bucket, label):
        folder = Path(folder)
        n = 0
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for p in folder.glob(ext):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                emb = self._encode_image(img)
                if emb is not None:
                    bucket.append(emb)
                    n += 1
        print(f"[staff/openclip] loaded {n} reference {label} crops from {folder}")
        return n

    def add_zone_anchor(self, crop_bgr: np.ndarray) -> None:
        emb = self._encode_image(crop_bgr)
        if emb is not None:
            self._zone_anchors.append(emb)

    def _ref_mean(self, vecs):
        if not vecs:
            return None
        m = np.mean(np.stack(vecs, axis=0), axis=0)
        n = float(np.linalg.norm(m) + 1e-8)
        return (m / n).astype(np.float32)

    def _scores(self, emb: np.ndarray, *, live: bool = False) -> tuple[float, float]:
        a = self.text_weight
        s_text = float(np.dot(emb, self._text_staff))
        c_text = float(np.dot(emb, self._text_cust))
        if live:
            # Live overlay: score only against fixed reference anchors so other
            # (still-noisy) visitors don't pollute the staff/cust pools.
            staff_refs = list(self._ref_anchors) + list(self._zone_anchors)
            cust_refs = list(self._cust_ref_anchors)
        else:
            # Final pass: include co-visitors' running means as weak anchors.
            staff_refs = (list(self._ref_anchors) + list(self._zone_anchors)
                          + [self._embs[v] for v, r in self._roles.items()
                             if "staff_only" in r])
            cust_refs = (list(self._cust_ref_anchors)
                         + [self._embs[v] for v, r in self._roles.items()
                            if "staff_only" not in r and "entry" in r])
        sm = self._ref_mean(staff_refs)
        cm = self._ref_mean(cust_refs)
        s_ref = float(np.dot(emb, sm)) if sm is not None else s_text
        c_ref = float(np.dot(emb, cm)) if cm is not None else c_text
        s = a * s_text + (1.0 - a) * s_ref
        c = a * c_text + (1.0 - a) * c_ref
        return s, c

    # Live-overlay tuning: don't decide until we have a few samples, and
    # require a small margin to flip — keeps overlay labels from oscillating.
    LIVE_MIN_OBS = 3
    LIVE_MARGIN = 0.02

    def live_vote(self, visitor_id: str) -> bool:
        if visitor_id in self._final:
            return self._final[visitor_id]
        if "staff_only" in self._roles.get(visitor_id, set()):
            return True
        emb = self._embs.get(visitor_id)
        if emb is None:
            return False
        if self._counts.get(visitor_id, 0) < self.LIVE_MIN_OBS:
            # Not enough evidence yet — default to customer (safer for a
            # retail floor) and let the running mean settle.
            return False
        s, c = self._scores(emb, live=True)
        prev = getattr(self, "_live_prev", {}).get(visitor_id, False)
        if prev:
            decision = s > c - self.LIVE_MARGIN  # sticky STAFF
        else:
            decision = s > c + self.LIVE_MARGIN  # need clear margin to flip
        if not hasattr(self, "_live_prev"):
            self._live_prev = {}
        self._live_prev[visitor_id] = decision
        return decision

    def finalize(self):
        print(f"[staff/openclip] finalising decisions for {len(self._embs)} visitors "
              f"(staff_refs={len(self._ref_anchors)} cust_refs={len(self._cust_ref_anchors)} "
              f"zone_anchors={len(self._zone_anchors)})")
        for vid, emb in self._embs.items():
            if "staff_only" in self._roles[vid]:
                self._final[vid] = True
                print(f"[staff/openclip]   {vid}: STAFF (staff_only camera)")
                continue
            s, c = self._scores(emb)
            decision = s > c + self.margin
            self._final[vid] = decision
            print(f"[staff/openclip]   {vid}: s={s:.3f} c={c:.3f} -> "
                  f"{'STAFF' if decision else 'cust'}")

    def vote(self, visitor_id: str) -> bool:
        return self._final.get(visitor_id, False)
