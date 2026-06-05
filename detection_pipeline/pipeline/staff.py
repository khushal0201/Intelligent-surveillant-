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
                     embedder=None,
                     store_id: Optional[str] = None):
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
        return OpenCLIPPrototypeStaff(store_id=store_id)
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
    "a beauty store sales associate in a fitted plain black t-shirt or polo and plain black trousers, hair tied back, no bag, leaning over a central display table arranging or restocking cosmetic products from the worker side",
    "a female beauty advisor in a solid black uniform shirt tucked into solid black pants, no backpack and no handbag visible, demonstrating a product to a shopper across a vanity counter",
    "a male retail employee in a plain black shirt and plain black formal trousers, hands free, holding a phone or barcode scanner, attending to the cosmetics counter",
    "a store staff member viewed from behind in a plain solid-black uniform top and plain solid-black bottoms, hair tied in a low bun or ponytail, no backpack straps over the shoulders, restocking a shelf",
    "a beauty store employee with a name badge or lanyard partially visible on a plain black uniform, standing inside the counter island, both hands free, organising tester products on the display",
    "a female sales associate in an all-black two-piece uniform (plain black top, plain black pants, black flat shoes), no bag of any kind on body, head tilted down toward products on the central table, working",
    "a retail staff member in a plain black uniform shirt with the store logo on the chest, plain black trousers, posture upright and attentive, hands engaged with merchandise rather than holding a personal phone for browsing",
    "a person viewed from behind wearing a plain solid-black short-sleeve or full-sleeve uniform top and plain solid-black trousers, no backpack on the back, no handbag, no shoulder bag, walking through the aisle of a beauty store as a worker",
    "the back of a retail employee in a fully black uniform — plain black shirt and plain black pants, both hands empty or holding store merchandise, NO backpack body and NO straps over the shoulders, NO handbag — moving between aisles",
    "a staff member in monochrome black workwear standing or walking in a beauty store aisle, completely empty back (no backpack body visible between the shoulder blades), arms relaxed at the sides, just doing rounds on the floor",
    "a female employee in a fitted plain black short-sleeve or three-quarter-sleeve uniform top (NOT a long flowing kurti, NOT a long anarkali, NOT an A-line tunic that flares below the hips) and slim black trousers, viewed from the side or rear, walking with empty hands and an empty back — no bag at all — between display fixtures",
    "a retail worker in a uniform of plain black top + plain black bottom, no patterns, no prints, no logos other than a small store logo, hair neat, no personal belongings on body, visibly part of the store team",
    "a female sales associate in a fitted plain black uniform top — straight hem, fitted body, NOT a loose flowing kurti and NOT a long anarkali — paired with fitted plain black trousers, viewed from the side standing between two display fixtures inside the staff side of the counter island",
    "a female beauty advisor in solid black uniform standing in the narrow worker walkway between two display gondolas, holding a product or phone in one hand, the other hand free, no bag on body, hair tied back, posture upright and professional",
    "a female retail employee in a plain black workwear set, body silhouette is fitted and crisp (not loose ethnic drape), positioned on the inside / staff-side of a cosmetics counter island with shelves on both sides, attending to merchandise",
    "a side-profile of a female store staff member in a plain black uniform top and plain black trousers, slim fit, neat appearance, no scarf or dupatta, no flowing fabric, holding a single product as part of her work",
    "a retail employee whose top AND bottom are BOTH solid black with no contrast — if the trousers are khaki, tan, beige, brown, blue or denim then this is NOT a staff member, it is a shopper",
    "a store staff member with strict monochrome black workwear: black top must match black bottom, both pieces equally dark, no two-tone outfit, no coloured trousers — colour mismatch between shirt and pants means this person is a customer not staff",
    "a person photographed from directly behind in solid black workwear, the camera clearly shows BOTH shoulders fully bare with no straps and the entire upper-back surface fully bare with no backpack body — only when the back is provably empty does this rear-view qualify as a staff member; if any strap or any bag body is visible, this rule does NOT apply",
    "a male or female retail staff member shown from the rear walking through the cosmetics aisle, wearing a fitted plain black short-sleeve top and plain black trousers, the back surface between the shoulder blades is completely flat and bare (no backpack of any size, no strap of any colour) — empty back is the decisive cue that this is staff",
    "a back view of a retail worker in monochrome black uniform passing in front of a display fixture, both arms relaxed, no bag in hand, no strap on either shoulder, no object slung across the body — empty hands plus empty back equals STAFF",
    "a person in solid black clothing facing away from the camera with both shoulders visible and clearly bare (no straps), torso visible and clearly bare (no backpack body), nothing hanging from arms or hips — uniformed staff member walking the floor",
    "a young woman in a plain solid-black short-sleeve or full-sleeve top tucked into plain solid-black trousers, no patterns, no prints, no warm tint, hair tied back in a low bun or sleek ponytail, no bag of any kind, leaning over a cosmetics display arranging products with both hands — a typical store staff member at work",
    "a side or three-quarter view of a young woman in plain matching black workwear leaning forward over a counter or low display unit to arrange or replenish merchandise, hands engaged with store products only, no personal bag visible on body, hair neat and tied back — the working posture itself is the cue",
    "a female employee in fitted plain black uniform standing close to a vanity counter with both hands on the counter top or on a product display, head tilted down, focused on merchandise rather than browsing for personal purchase, hair tied back — typical staff demeanour",
]
# Store 2 uniform: pink / peach / salmon top with dark trousers. ONLY appended
# to the staff prompt set when running on Store 2 footage — if loaded for Store
# 1 (where staff wear all-black) these prompts would falsely pull pink-clothed
# customers toward STAFF.
_OCLIP_STAFF_PROMPTS_STORE2 = [
    "a beauty store sales associate in a fitted plain pink, peach, salmon or coral short-sleeve uniform t-shirt with dark trousers, hair tied back in a low bun or ponytail, no bag on body, attending to merchandise from inside the counter island",
    "a female beauty advisor in a solid pink or peach uniform polo or t-shirt tucked into dark slim trousers, name badge or store logo on the chest, demonstrating a product to a customer at a vanity counter",
    "a retail employee wearing a uniform pink or salmon-coloured top with dark pants, hair pulled back, no backpack and no handbag, restocking cosmetics on a shelf as part of her job",
    "a store staff member in a peach, pink or coral uniform shirt and dark trousers, both hands engaged with merchandise or a barcode scanner, posture professional and attentive",
    "a female sales associate in a plain pink or peach short-sleeve top — uniform top, NOT a fashion top — with dark formal trousers, hair tied neatly back, standing on the worker side of a billing or vanity counter",
    "a back or side view of a retail worker in a pink, peach or salmon uniform t-shirt and dark trousers, hair in a tight bun or low ponytail, completely empty back (no backpack body, no straps over shoulders), arms relaxed at the sides",
    "a beauty advisor at a billing counter in a pink or peach uniform top with a name badge or store logo, dark trousers, both hands engaged operating the POS system or arranging products — clearly working, not shopping",
    "a female staff member in a uniform pink or peach top with closed-collar or polo cut, dark trousers, neat groomed appearance, no personal bag or backpack on body, walking through the makeup aisle as a worker",
]
_OCLIP_CUSTOMER_PROMPTS = [
    "a shopper in colourful or printed casual clothing — kurti, printed top, jeans, floral dress, saree or salwar-kameez — browsing cosmetics in a beauty store, NOT in an all-black uniform",
    "a customer carrying a backpack with both straps visible over the shoulders, wearing non-uniform clothing such as a printed top or jeans, inspecting a product from the customer side of the counter",
    "a female shopper holding a handbag or shoulder bag on the arm, wearing casual everyday clothes, picking up a tester from a display",
    "a customer with a tote bag hanging from one shoulder, wearing sneakers or sandals (not store-uniform black boots), browsing skincare shelves",
    "a male shopper in a casual coloured t-shirt or button-down shirt and jeans, standing in front of a shelf reading a product label, clearly not wearing a uniform",
    "a customer in any non-uniform outfit holding their personal phone up to take a photo of a product, browsing rather than working",
    "a shopper in everyday street clothes (denim, prints, bright colours, cultural wear) inspecting a single product they are about to buy, posture leaned toward shelf as a buyer not a worker",
    "a female customer wearing a long maroon, dark-brown, deep-purple or wine-coloured kurti or tunic over loose dark trousers or palazzo pants, with eyeglasses, gently holding a small product in her hand while browsing — this is NOT a black uniform",
    "a female shopper in a dark non-black ethnic outfit (kurti, kurta, salwar-kameez, anarkali) — colour is wine, maroon, brown, navy or charcoal but visibly not jet-black uniform — wearing glasses, inspecting cosmetics, NOT a staff member",
    "a female customer viewed from the side or behind with a small backpack on her back, wearing a long dark-coloured kurti or cardigan over loose pants, hair down or in a loose ponytail, leaning toward a shelf to pick up a product — the backpack and the loose ethnic silhouette mark her as a shopper",
    "a customer in a dark kurti or long top over black leggings or palazzo pants — silhouette is loose and flowing rather than fitted uniform — carrying a backpack or handbag, glasses on face, browsing a cosmetics shelf",
    "a female shopper in any dark outfit who is also wearing a backpack with green/teal/blue/grey straps and body visible, the backpack itself proves she is a customer regardless of clothing colour",
    "a female customer with a small backpack on her back — either both straps visible over both shoulders, OR one strap visible from the side, OR the body of the backpack peeking out behind one arm — wearing any clothing including a long dark kurti or tunic; the presence of a backpack is decisive proof this is NOT a staff member",
    "a female customer wearing eyeglasses and a long flowing dark kurti or knee-length tunic top over slim dark leggings or palazzo pants, holding a single product close to her face to inspect — the long flowing silhouette and the inspection posture together mark her as a shopper, NOT a uniformed staff member",
    "a side or rear view of a female shopper carrying a backpack — the backpack body or strap is visible somewhere on her torso — wearing any colour of top including all-black, browsing the cosmetics aisle; backpack overrides outfit colour, this is a CUSTOMER",
    "a male customer in a plain or printed grey, white, beige, blue, olive, brown or pastel t-shirt or polo (NOT solid black), paired with khaki, beige, tan, brown, blue or olive trousers or jeans, reaching toward a product on a shelf — clothing is clearly NOT a black uniform",
    "a male shopper in a casual coloured short-sleeve t-shirt and earth-tone or denim trousers, no logo, no name badge, browsing the cosmetics aisle from the customer side of a fixture",
    "a male customer wearing a grey, heather-grey, off-white, navy, olive or earth-tone top with khaki, tan, beige or stone-coloured pants — definitely not all-black workwear — picking up a tester product",
    "a male shopper in casual everyday clothing where TOP and BOTTOM are different colours (e.g. grey shirt + tan trousers, blue shirt + black jeans, white tee + denim) — NOT a monochrome black uniform — examining a product",
    "a man browsing a beauty store in a coloured t-shirt and contrasting trousers, posture is curious and exploratory rather than purposeful work routine, clearly a shopper not a staff member",
    "a male shopper in a plain GREY, heather-grey, light-grey or charcoal-grey short-sleeve t-shirt — the top is visibly NOT black, it is grey — paired with dark trousers or jeans, leaning toward a shelf with one hand reaching for a product; grey top alone disqualifies him from being staff regardless of trouser colour",
    "a male customer in a fitted plain grey or heather-grey crew-neck t-shirt and dark denim or black jeans, viewed from the front or three-quarter, inspecting a cosmetics product close to face — this is a typical urban male shopper, NOT a uniformed retail employee; staff uniforms are solid BLACK on top, never grey",
    "a male shopper in a solid grey t-shirt standing in front of a tall display gondola, both hands engaged with a single product he is examining, no name badge, no lanyard, no store logo on chest — a customer browsing, his grey top makes the staff label impossible",
    "a male customer in a light-coloured (grey, white, beige, off-white, pastel, sky-blue, olive) t-shirt with any colour of trousers — the moment the top is NOT solid black, this person is a SHOPPER not staff, even if they have empty hands and an empty back",
    "a female shopper in a dark wine, maroon, burgundy, oxblood, deep-red, dark-purple or aubergine short-sleeve t-shirt or top — the colour is reddish/purplish dark, NOT the jet-black of a staff uniform — carrying a backpack visible on her shoulder or back, browsing makeup shelves; this is a CUSTOMER",
    "a female customer wearing a deep-maroon or wine-red plain t-shirt with a backpack slung on the shoulder, side-profile view, the backpack body is partially visible behind her arm and the strap goes diagonally across the chest or shoulder — backpack + non-black coloured top = SHOPPER, never staff",
    "a young woman in a dark reddish or burgundy short-sleeve top with eyeglasses and a small everyday backpack (brand may say 'Safari', 'Wildcraft', 'Skybags' or similar) on one shoulder, walking the makeup aisle picking up products — this is a customer; staff never carry personal backpacks while on duty",
    "a side or three-quarter view of a female shopper whose top is clearly TINTED (dark red, wine, maroon, burgundy, navy, dark green, charcoal-with-warm-tone) rather than truly jet-black, AND who has any backpack body or shoulder-strap visible anywhere on the torso — together these two signals are unambiguous proof of CUSTOMER, never a staff member",
    "a person whose t-shirt has any visible warm-tone tint (red, maroon, burgundy, brown, purple) when viewed in store lighting — even if it looks dark — is NOT wearing the official solid-black retail uniform, and is therefore a CUSTOMER",
    "HARD RULE: any person with a backpack visible — whether the backpack body is on the upper back, between the shoulder blades, peeking from behind one arm, or hanging from one shoulder; whether the strap is dark, green, teal, blue, grey, red or any colour — is a CUSTOMER. Staff on duty NEVER wear backpacks. This rule overrides every other consideration including all-black clothing, fitted silhouette, empty hands, or rear-view pose",
    "a rear or side view of a shopper wearing an all-black outfit (black top, black trousers, OR black dress) WITH a small everyday backpack visible on the upper back — the backpack body sits between the shoulder blades and one or both straps cross the shoulders — backpack on back instantly disqualifies staff classification, this is a CUSTOMER",
    "a back view of a person in dark clothing where a green, teal, blue, grey or any-coloured backpack strap is visible crossing the shoulder OR the backpack body is visible against the dark torso — the presence of the backpack is decisive, this person is a SHOPPER not a staff member",
    "HARD RULE: any person carrying a HANDBAG — a structured leather or fabric handbag, a tote bag, a sling bag, a clutch, a crossbody, a beige/brown/tan/black/coloured shoulder bag held in the hand or on the forearm or hanging from the shoulder — is a CUSTOMER. Staff on duty NEVER carry personal handbags. This rule overrides all-black clothing, fitted silhouette, dress vs trousers, and pose",
    "a female shopper in a black dress or all-black outfit holding a tan, beige, brown, cream, white, pink or coloured handbag in one hand or on the forearm; the visible handbag instantly disqualifies the staff label, this is a CUSTOMER",
    "a woman in dark clothing with a structured handbag (rectangular or trapezoid shape) held by its top-handle in one hand near the hip, or hanging from the inner elbow — typical fashion handbag carry posture, NEVER a staff posture; this is a SHOPPER",
    "a female customer in a black dress, black tunic or all-black outfit, on a phone call or texting, with a handbag clearly visible in the other hand or hanging from the shoulder — phone-in-hand plus handbag-in-hand is the unambiguous signature of a SHOPPER, not a uniformed staff member",
    "a female shopper wearing a black DRESS — a single-piece short or knee-length dress, not a two-piece uniform — with bare legs visible below the hem and sandals or heels on the feet, holding a phone or product, browsing the store; staff uniform is always a two-piece (black top + black trousers), so any one-piece black dress means CUSTOMER",
    "a woman in a fitted black bodycon dress, black mini-dress, black sheath dress, black wrap-dress or black skirt-dress with bare legs, ankles or knees visible — no trousers covering the legs — this is a fashion shopper, NEVER a uniformed retail worker; staff always wear full-length black trousers",
    "a female customer in a one-piece black outfit (dress, jumpsuit, romper or playsuit) with skin visible on the legs, walking through the store with a handbag or phone — the bare-legs cue alone is decisive proof that this is NOT staff, because staff trousers fully cover the legs",
    "a person wearing a black skirt, black mini-skirt or black short dress with visible bare legs and open-toe sandals, heels or strappy footwear — fashion footwear, NOT uniform black flats — this is a SHOPPER, the staff dress code requires plain black trousers and closed flat shoes",
    "any person whose lower body shows skin between the hem of their black garment and their footwear (i.e. bare legs) is a CUSTOMER, not a staff member; the staff uniform is a black top tucked into ankle-length black trousers, no skin visible on the legs",
    "HARD RULE: any person whose top is BEIGE, CREAM, OFF-WHITE, NUDE, IVORY, KHAKI, SAND, TAUPE, LIGHT-BROWN, CAMEL, OATMEAL, ECRU, BUTTER-YELLOW or any pale neutral colour is a CUSTOMER. The staff uniform top is solid jet-black ONLY — any light or warm-neutral top instantly disqualifies the staff label, regardless of pose, hands, bag visibility or location in the store",
    "a female shopper in a light beige, cream, off-white, ivory or oatmeal short-sleeve top or kurti — the top is visibly pale and warm-toned, NOT black — standing on the customer side of a billing or vanity counter, leaning forward to look at products on the counter top; pale top alone marks her as a CUSTOMER",
    "a customer wearing a cream, beige, nude or sand-coloured blouse, shirt, kurti or t-shirt paired with dark trousers, denim or palazzo pants — top is unmistakably light-coloured against the dark store fixtures — browsing cosmetics; this colour combination is a SHOPPER outfit, never a uniform",
    "two customers in pale beige or cream tops standing shoulder-to-shoulder on the buyer side of a counter, leaning over the merchandise displayed on the counter top, inspecting products with both hands — the matching pale tops and the inspection posture confirm both are SHOPPERS, not staff",
    "a male or female customer in a light-beige, oatmeal, cream or off-white plain t-shirt or polo, no logo, no name badge, picking up a tester from a counter — pale neutral top is incompatible with the strict all-black staff uniform, so this is a CUSTOMER",
    "a person whose top has any warm pale tint (cream, beige, nude, peach, butter, oat, sand) when seen in store lighting — even under shadow — is NOT in staff uniform; staff tops are pure black with zero warm tint; pale warm top therefore proves CUSTOMER",
    "a female shopper in a sleeveless or short-sleeve cream/beige kurti or tunic over dark leggings, hair loose or in a ponytail, leaning over a glass billing counter to look at products — kurti silhouette plus pale colour is a clear SHOPPER signature, never a staff uniform",
    "HARD RULE: any person whose footwear is OPEN-TOE — sandals, flip-flops, slippers, slides, kolhapuris, mojaris, strappy sandals, gladiators, flat thong sandals, heeled sandals, espadrilles, or any shoe where the toes or heel are visible — is a CUSTOMER. Staff on duty wear closed flat black shoes only. Visible bare toes or open-back footwear is a decisive customer cue, regardless of clothing colour.",
    "a female shopper in any all-black or dark outfit (top, leggings, dress) wearing flat tan, beige, brown or black open-toe sandals or flip-flops with visible toes — staff dress code requires closed black flats, so open-toe footwear means CUSTOMER",
    "a male shopper wearing chappals, slippers, flip-flops or open sandals with bare feet visible, paired with any t-shirt and jeans or trousers — staff are required to wear closed shoes, so open footwear is unambiguous proof of CUSTOMER",
    "HARD RULE: a man wearing a grey, heather-grey, white, beige, blue, olive or any non-black t-shirt with light-coloured jeans, khakis or trousers, holding a personal phone in one hand at face or chest height while looking at it, with open-toe slippers or sandals on his feet, is a CUSTOMER. The combination of non-black top + light pants + open footwear + browsing-on-phone is unmistakable shopper behaviour, never staff.",
    "a male customer in a plain grey, white or pastel t-shirt and denim jeans or beige chinos, casually holding his own phone up at face level to read or photograph product info, wearing slippers or sandals — typical Indian male shopper at a beauty store, definitely NOT staff",
    "a young man in a casual grey or pastel-coloured short-sleeve t-shirt with light-coloured trousers and slippers, scrolling on his phone with one hand while standing near a display, no name badge, no store logo, no black uniform — this is a customer killing time while shopping",
    "HARD RULE: any person holding a personal smartphone in one hand UP at face or chin height — looking down at the screen, scrolling with thumb, reading or texting, the phone close to the face — is a CUSTOMER browsing. Staff hold phones LOW at hip or counter level for POS or scanning, not raised up at face level. Phone-at-face is the unambiguous shopper signature, regardless of clothing colour or gender.",
    "a young man in a dark grey, charcoal, navy, deep-blue or dark t-shirt (could appear black-ish under store lighting but is NOT the official jet-black staff uniform top), holding his personal phone raised up close to his face with the screen facing him, browsing on the phone — this is a CUSTOMER, not staff",
    "a male customer with a phone held vertically up near his chin or mouth, head tilted slightly forward to read the screen, body posture relaxed and stationary in front of a display fixture, casual t-shirt of any colour including dark grey or near-black — typical Indian male shopper checking product reviews on his phone, never staff",
    "a young woman or man holding their personal phone up at face level with both eyes on the screen, standing still in the aisle of a beauty store with no merchandise in either hand, no name badge, no store logo, no apron — phone-at-face plus empty merchandise hand is decisive proof of a CUSTOMER",
    "a young woman in a plain black or dark short-sleeve t-shirt with a small backpack visible on one shoulder, the backpack body hanging behind one arm and the strap diagonally crossing the chest or shoulder, hair in a low ponytail, walking past a makeup wall — the backpack on the shoulder is the decisive cue, this is a CUSTOMER not staff",
    "a female shopper in dark or all-black clothing with a backpack visible on her body — body of the backpack peeks out behind her arm or sits between her shoulder blades, strap visible across the shoulder — viewed from the side or three-quarter angle browsing cosmetics; backpack on body = CUSTOMER, no exception",
    "a young woman in a fitted dark t-shirt with a black, grey, brown or coloured backpack slung over one shoulder (single-strap carry), one hand at her side and the other near a shelf, side-profile view in a beauty store aisle — single-shoulder backpack carry is a typical shopper posture, never staff",
    "a male customer in a light-grey, white, off-white or near-white t-shirt (the top is visibly pale, NOT dark and definitely NOT jet-black) paired with denim jeans or light trousers, holding his personal phone up close to his face to read the screen, slippers on his feet — pale top + casual jeans + phone-at-face = a typical male SHOPPER, not staff",
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
                 text_weight: float = 0.80,
                 margin: float = 0.0,
                 staff_prompts: list[str] | None = None,
                 customer_prompts: list[str] | None = None,
                 store_id: Optional[str] = None):
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
        if staff_prompts is None:
            sp = list(_OCLIP_STAFF_PROMPTS)
            # Pink/peach uniform only applies at Store 2. Including these on
            # Store 1 footage (all-black uniform) would mislabel pink-wearing
            # customers as staff.
            sid = (store_id or "").lower().replace("_", "").replace(" ", "")
            is_store2 = (
                sid.startswith("st2")           # ST2009 etc.
                or "store2" in sid              # store2 / store_2 / Store 2
                or sid == "2"
            )
            if is_store2:
                sp.extend(_OCLIP_STAFF_PROMPTS_STORE2)
                print(f"[staff/openclip] store_id={store_id!r} -> added pink uniform prompts")
            else:
                print(f"[staff/openclip] store_id={store_id!r} -> black uniform only")
        else:
            sp = staff_prompts
        cp = customer_prompts or _OCLIP_CUSTOMER_PROMPTS
        self._text_staff = self._encode_text(sp)
        self._text_cust = self._encode_text(cp)
        # Per-prompt L2-normalised embeddings — used for top-K matching so a
        # single strongly-matching prompt (e.g. "maroon top + backpack") can
        # outscore the generic mean direction.
        self._text_staff_all = self._encode_text_all(sp)
        self._text_cust_all = self._encode_text_all(cp)
        # Per-visitor running mean image embedding.
        self._embs: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._roles: dict[str, set[str]] = defaultdict(set)
        self._ref_anchors: list[np.ndarray] = []
        self._zone_anchors: list[np.ndarray] = []
        self._cust_ref_anchors: list[np.ndarray] = []
        self._final: dict[str, bool] = {}
        # ------- gender / age zero-shot heads (reuse the same CLIP model) -------
        # Each "class" = mean-pooled embedding of N descriptive prompts.
        self._gender_labels = ["F", "M"]
        # Mean-pooled per class — more robust at low CCTV resolution where no
        # single prompt is a clear winner. Female list is heavy on "long hair
        # tied back / slim feminine frame / black dress" cues; male list is
        # narrowed to require explicit masculine evidence (facial hair / broad
        # shoulders) so casual rear-view women don't drift to M.
        self._gender_text = np.stack([
            self._encode_text([
                "a woman shopping in a beauty store, clearly feminine appearance",
                "a female customer browsing cosmetics, female facial features",
                "a young woman with long dark hair tied in a low ponytail down her back",
                "a young woman with long hair pulled into a tight bun at the back of the head, slim feminine frame",
                "a woman with narrow shoulders, narrow waist and feminine hips, slim female silhouette",
                "a young woman in a black dress or black tunic with bare arms or bare legs visible, feminine body shape",
                "a woman wearing a knee-length dress, kurti, blouse or other feminine clothing",
                "a young female shopper in a casual t-shirt and slim trousers with a long ponytail or bun visible at the back of her head",
                "a young woman with a small backpack, long hair, slim feminine build — the backpack does NOT make her male",
                "a side or rear view of a woman — long hair tied back, slim shoulders, narrow waist, feminine posture",
                "a young Indian woman in her twenties or thirties with long dark hair, smooth feminine face, slim shoulders",
                "a female shopper whose long hair, ponytail or bun is clearly visible at the back of the head",
                "a woman with no facial hair, no Adam's apple, smooth jaw, narrow shoulders — unmistakably female",
                "a woman wearing eyeglasses, hair tied back, holding a product, slim feminine build",
                "a female shopper in a fitted black dress with bare arms, hair tied back, holding a small product — clearly a woman",
                "a young woman whose silhouette has narrow shoulders relative to hips and a small upper body, female body type",
            ]),
            self._encode_text([
                "a man shopping in a beauty store, clearly masculine appearance, visible facial hair or stubble",
                "a male customer with a moustache, beard or stubble, very short cropped hair",
                "a young man with very short hair and a square masculine jawline, visible Adam's apple",
                "a man with broad shoulders much wider than his hips, flat chest, muscular or stocky masculine build",
                "an adult male with a beard or moustache and short hair, no ponytail and no long hair at all",
                "a male shopper whose silhouette is wide-shouldered and narrow-hipped — typical masculine V-shape — with very short hair",
            ]),
        ], axis=0).astype(np.float32)
        self._age_labels = ["child", "teen", "20s", "30s", "40s", "50s", "60+"]
        # Per-bucket list of per-prompt L2-normalised embeddings (variable length).
        # We use top-K matching per bucket so a strongly-matching prompt wins
        # instead of a class mean that gets dragged toward older buckets when
        # CCTV resolution is too low to read fine wrinkles.
        age_prompt_lists = [
            [  # child — require strong height/proportion cues vs adults so adult men never fall here
                "a small child under 10 years old, head only reaching the waist or hip of nearby adult shoppers, child-size body proportions with an oversized head relative to a tiny torso",
                "a tiny preschool-aged child holding a parent's hand, less than half the height of the adult, baby-face with chubby cheeks and no adult features at all",
                "a primary-school-aged little kid (5 to 9 years old), visibly much shorter than every adult in the frame, juvenile body proportions, no adult shoulders or jawline",
            ],
            [  # teen
                "a teenager between 13 and 19 years old, slim build, school-aged",
                "a high-school student shopping for cosmetics, youthful round face",
                "a young adolescent in a retail store, late teens, smooth childlike face",
                "a teenager with a school backpack and a thin youthful frame, aged 14 to 17",
            ],
            [  # 20s  — primary shopper bucket; many generic prompts so it wins on low-res faces
                "a young adult in their twenties shopping in a beauty store, fresh youthful face",
                "a person aged about 20 to 29 years old, fully-grown adult, smooth skin, no visible wrinkles",
                "a college-aged or fresh-graduate shopper in a beauty store, early-to-mid twenties",
                "a young woman in her early twenties with long dark hair, slim adult build, smooth youthful skin",
                "a young man in his twenties with short hair, smooth jawline, no grey hair, fully-grown adult",
                "a fully-grown young adult shopper aged 22 to 28, body proportions of a young adult, face is smooth",
                "a young Indian adult in their twenties browsing cosmetics, dark hair, smooth skin",
                "a typical young woman customer in her twenties wearing casual clothes in a beauty store",
                "a person who looks like a young working professional in their twenties",
            ],
            [  # 30s — also common; prompts emphasise mature adult, NOT explicitly old
                "an adult in their thirties shopping in a beauty store, mature adult face",
                "a person aged about 30 to 39 years old, fully adult features, slightly fuller face than a twenty-something",
                "a working-age adult in a beauty store, early-to-mid thirties, settled adult appearance",
                "a thirty-something professional shopper, mature adult face, no grey hair yet",
                "an Indian adult in their thirties browsing cosmetics, mature adult face",
            ],
            [  # 40s — require explicit middle-age cues
                "a clearly middle-aged adult in their forties, visible forehead and eye wrinkles",
                "a person aged about 40 to 49 years old, some grey hair starting to show, middle-aged face",
                "a middle-aged shopper in a beauty store with clear visible wrinkles around the eyes",
                "a forty-something parent with mature middle-aged appearance and crow's feet",
            ],
            [  # 50s
                "a clearly middle-aged-to-older adult in their fifties, prominent facial wrinkles and partial grey hair",
                "a person aged about 50 to 59 years old, salt-and-pepper hair, deeply lined mature face",
                "a fifty-something shopper with deeper facial lines and noticeably greying hair",
            ],
            [  # 60+ — require strong elderly cues so it does NOT fire on indistinct faces
                "a clearly elderly person aged 60 or older, mostly grey or white hair, deep wrinkles",
                "a senior citizen in a beauty store, grey or white hair, visibly sagging or aged skin",
                "an obviously older adult in their sixties or seventies, clearly elderly appearance, white hair",
                "a grandparent-aged shopper with mostly white or silver hair and a deeply lined face",
            ],
        ]
        self._age_prompts_per_bucket = [
            self._encode_text_all(prompts) for prompts in age_prompt_lists
        ]
        # Per-visitor running-mean softmax probabilities.
        self._gender_probs: dict[str, np.ndarray] = {}
        self._age_probs: dict[str, np.ndarray] = {}
        print(f"[staff/openclip] ready. staff_prompts={len(sp)} "
              f"cust_prompts={len(cp)} text_weight={self.text_weight} "
              f"+ gender({len(self._gender_labels)}) age({len(self._age_labels)})")

    # ---- encoding helpers ----
    def _encode_text(self, prompts: list[str]) -> np.ndarray:
        toks = self.tokenizer(prompts).to(self._device)
        with self._torch.no_grad():
            t = self.model.encode_text(toks)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        m = t.mean(dim=0)
        m = m / (m.norm() + 1e-8)
        return m.cpu().numpy().astype(np.float32)

    def _encode_text_all(self, prompts: list[str]) -> np.ndarray:
        """Return per-prompt L2-normalised embeddings, shape (N, D)."""
        toks = self.tokenizer(prompts).to(self._device)
        with self._torch.no_grad():
            t = self.model.encode_text(toks)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        return t.cpu().numpy().astype(np.float32)

    @staticmethod
    def _topk_text_score(emb: np.ndarray, prompt_embs: np.ndarray,
                         k: int = 3) -> float:
        """Mean cosine over the top-K best-matching prompts."""
        sims = prompt_embs @ emb
        if sims.size <= k:
            return float(sims.mean())
        idx = np.argpartition(sims, -k)[-k:]
        return float(sims[idx].mean())

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
        # ---- demographics: accumulate softmax probs over gender / age heads ----
        g_probs = self._softmax_against(emb, self._gender_text)
        a_probs = self._softmax_topk_perbucket(emb, self._age_prompts_per_bucket)
        prev_g = self._gender_probs.get(visitor_id)
        prev_a = self._age_probs.get(visitor_id)
        # `c` is pre-increment count of (image) observations; mirror that here.
        self._gender_probs[visitor_id] = (
            g_probs if prev_g is None else (prev_g * c + g_probs) / (c + 1)
        ).astype(np.float32)
        self._age_probs[visitor_id] = (
            a_probs if prev_a is None else (prev_a * c + a_probs) / (c + 1)
        ).astype(np.float32)

    @staticmethod
    def _softmax_against(emb: np.ndarray, class_texts: np.ndarray,
                         tau: float = 100.0) -> np.ndarray:
        sims = class_texts @ emb  # (K,) cosine since both are L2-normalised
        z = tau * sims
        z = z - z.max()
        e = np.exp(z)
        return (e / (e.sum() + 1e-8)).astype(np.float32)

    @staticmethod
    def _softmax_topk_perbucket(emb: np.ndarray,
                                buckets: list[np.ndarray],
                                k: int = 2,
                                tau: float = 100.0) -> np.ndarray:
        """Per-bucket top-K mean cosine, then softmax. Used for age so a
        single strongly-matching prompt (e.g. "young adult, smooth skin") can
        win over the bucket mean."""
        scores = np.empty(len(buckets), dtype=np.float32)
        for i, prompt_embs in enumerate(buckets):
            sims = prompt_embs @ emb
            if sims.size <= k:
                scores[i] = float(sims.mean())
            else:
                idx = np.argpartition(sims, -k)[-k:]
                scores[i] = float(sims[idx].mean())
        z = tau * scores
        z = z - z.max()
        e = np.exp(z)
        return (e / (e.sum() + 1e-8)).astype(np.float32)

    # --- demographic accessors (used by run.py for overlay + event metadata) ---
    DEMO_MIN_OBS = 2

    def live_demographics(self, visitor_id: str) -> dict:
        return self._demographics_for(visitor_id, min_obs=self.DEMO_MIN_OBS)

    def demographics(self, visitor_id: str) -> dict:
        # Final pass: lower bar so visitors with few crops still get a label.
        return self._demographics_for(visitor_id, min_obs=1)

    def _demographics_for(self, visitor_id: str, *, min_obs: int) -> dict:
        out = {"gender": None, "gender_conf": 0.0,
               "age_bucket": None, "age_conf": 0.0}
        if self._counts.get(visitor_id, 0) < min_obs:
            return out
        g = self._gender_probs.get(visitor_id)
        a = self._age_probs.get(visitor_id)
        if g is not None:
            i = int(np.argmax(g))
            out["gender"] = self._gender_labels[i]
            out["gender_conf"] = float(g[i])
        if a is not None:
            i = int(np.argmax(a))
            out["age_bucket"] = self._age_labels[i]
            out["age_conf"] = float(a[i])
        return out

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
        # Asymmetric top-K. Staff prompts are narrow and homogeneous (all
        # describe the same black uniform), so 3-of-N average is stable.
        # Customer prompts are diverse (kurtis, dresses, backpacks, grey
        # t-shirts, etc.) — for any specific person only 1-2 prompts truly
        # match. Averaging 3 dilutes the winner with weak matches, so use
        # k=1 (single best) for customer to let the most-specific prompt win.
        s_text = self._topk_text_score(emb, self._text_staff_all, k=3)
        c_text = self._topk_text_score(emb, self._text_cust_all, k=1)
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
    # Margin is asymmetric: easy to flip INTO staff, harder to flip out of it,
    # because the failure mode in this dataset is staff-as-customer.
    LIVE_MIN_OBS = 3
    LIVE_FLIP_TO_STAFF = 0.0     # ref anchors are now strong; flip to STAFF as soon as it wins
    LIVE_FLIP_TO_CUST = 0.0      # symmetric — flip back as soon as customer wins

    def live_vote(self, visitor_id: str) -> bool:
        if visitor_id in self._final:
            return self._final[visitor_id]
        if "staff_only" in self._roles.get(visitor_id, set()):
            return True
        emb = self._embs.get(visitor_id)
        if emb is None:
            return False
        if self._counts.get(visitor_id, 0) < self.LIVE_MIN_OBS:
            return False
        s, c = self._scores(emb, live=True)
        prev = getattr(self, "_live_prev", {}).get(visitor_id, False)
        if prev:
            decision = s > c - self.LIVE_FLIP_TO_CUST  # sticky STAFF
        else:
            decision = s > c + self.LIVE_FLIP_TO_STAFF
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
