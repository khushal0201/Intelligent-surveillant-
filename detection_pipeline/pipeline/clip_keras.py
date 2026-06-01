"""Keras CLIP wrapper (no PyTorch).

Uses `keras-hub`'s CLIPBackbone (`clip_vit_base_patch32`). Provides:
  - encode_image(list[bgr_crop]) -> (N, D) L2-normalised embeddings
  - encode_text(list[str])       -> (M, D) L2-normalised embeddings

Designed to be the single CLIP loader shared by reid.py and staff.py.
Falls back gracefully: callers should catch ImportError / RuntimeError.
"""
from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
import cv2

# OpenAI CLIP image normalization
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_INPUT = 224


class KerasCLIP:
    """Loads CLIP ViT-B/32 from keras-hub. Backbone + tokenizer."""

    PRESET = "clip_vit_base_patch32"

    def __init__(self, preset: str | None = None, max_text_len: int = 77):
        try:
            import keras_hub  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "keras-hub is required for CLIP. `pip install keras-hub`."
            ) from e
        import keras_hub
        self.preset = preset or self.PRESET
        self.max_text_len = max_text_len
        self.backbone = keras_hub.models.CLIPBackbone.from_preset(self.preset)
        self.tokenizer = keras_hub.tokenizers.CLIPTokenizer.from_preset(
            self.preset
        )
        # API surface differs between keras-hub versions; probe attribute names
        self.vision_encoder = (
            getattr(self.backbone, "vision_encoder", None)
            or getattr(self.backbone, "image_encoder", None)
        )
        self.text_encoder = getattr(self.backbone, "text_encoder", None)
        if self.vision_encoder is None or self.text_encoder is None:
            raise RuntimeError(
                "Unsupported keras-hub CLIPBackbone API: missing "
                "vision_encoder/text_encoder attributes."
            )

    # --------------------------------------------------------------- image
    @staticmethod
    def _preprocess_image(crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            crop_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (_INPUT, _INPUT), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - _CLIP_MEAN) / _CLIP_STD
        return x

    def encode_image(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        batch = np.stack([self._preprocess_image(c) for c in crops_bgr], axis=0)
        feats = self.vision_encoder(batch, training=False)
        feats = np.asarray(feats, dtype=np.float32)
        norm = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8
        return feats / norm

    # ---------------------------------------------------------------- text
    def _tokenize(self, prompts: Sequence[str]):
        import keras
        ids = self.tokenizer(list(prompts))
        # ids is a tf.RaggedTensor or list -> pad/truncate to max_text_len
        ids = keras.ops.convert_to_tensor(ids.to_tensor() if hasattr(ids, "to_tensor") else ids)
        ids = keras.ops.cast(ids, "int32")
        # pad/truncate
        L = self.max_text_len
        cur = ids.shape[-1] or 0
        if cur < L:
            pad_amt = L - cur
            pad = keras.ops.zeros((ids.shape[0], pad_amt), dtype="int32")
            ids = keras.ops.concatenate([ids, pad], axis=1)
        elif cur > L:
            ids = ids[:, :L]
        padding_mask = keras.ops.cast(ids != 0, "int32")
        return {"token_ids": ids, "padding_mask": padding_mask}

    def encode_text(self, prompts: Sequence[str]) -> np.ndarray:
        inputs = self._tokenize(prompts)
        feats = self.text_encoder(inputs, training=False)
        feats = np.asarray(feats, dtype=np.float32)
        norm = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8
        return feats / norm
