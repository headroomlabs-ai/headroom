"""ONNX-based image technique router — no PyTorch dependency.

Uses ONNX INT8 models for both query classification (technique-router)
and image analysis (SigLIP), matching the accuracy of the PyTorch
versions at 15x smaller size and no GPU requirement.

Models auto-downloaded from HuggingFace on first use:
- chopratejas/technique-router-onnx (~32 MB)
- chopratejas/siglip-image-encoder-onnx (~95 MB)
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from headroom.image.trained_router import ImageSignals, RouteDecision, Technique
from headroom.onnx_runtime import create_cpu_session_options, hf_hub_download_local_first

logger = logging.getLogger(__name__)

_TECHNIQUE_ROUTER_REPO = "chopratejas/technique-router-onnx"
_SIGLIP_ENCODER_REPO = "chopratejas/siglip-image-encoder-onnx"


# ImageSignals, RouteDecision, Technique imported from trained_router


class OnnxTechniqueRouter:
    """ONNX-based technique router — no PyTorch dependency.

    Uses:
    1. MiniLM ONNX INT8 classifier for query intent (~32 MB, ~5ms)
    2. SigLIP ONNX INT8 image encoder for image analysis (~95 MB, ~30ms)
    3. Pre-computed text embeddings for image property scoring (~25 KB)

    Total: ~127 MB, runs on CPU with onnxruntime only.
    """

    def __init__(self, use_siglip: bool = True):
        self.use_siglip = use_siglip
        self._classifier_session: Any = None
        self._tokenizer: Any = None
        self._id2label: dict[int, str] = {}
        self._siglip_session: Any = None
        self._text_embeddings: dict[str, np.ndarray] = {}
        self._siglip_processor: Any = None

    def _load_classifier(self) -> None:
        """Lazy-load the technique router ONNX model."""
        if self._classifier_session is not None:
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        logger.info("Loading technique-router ONNX INT8...")

        model_path = hf_hub_download_local_first(_TECHNIQUE_ROUTER_REPO, "model_quantized.onnx")
        self._classifier_session = ort.InferenceSession(
            model_path,
            create_cpu_session_options(ort),
            providers=["CPUExecutionProvider"],
        )

        tokenizer_path = hf_hub_download_local_first(_TECHNIQUE_ROUTER_REPO, "tokenizer.json")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=64)
        self._tokenizer.enable_padding(length=64)

        # Load label mapping
        import json

        config_path = hf_hub_download_local_first(_TECHNIQUE_ROUTER_REPO, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        self._id2label = {int(k): v for k, v in config.get("id2label", {}).items()}

        logger.info(
            f"Technique router loaded: {len(self._id2label)} classes, "
            f"ONNX INT8 ({Path(model_path).stat().st_size // 1024 // 1024} MB)"
        )

    def _load_siglip(self) -> None:
        """Lazy-load the SigLIP ONNX image encoder."""
        if self._siglip_session is not None:
            return

        import onnxruntime as ort

        logger.info("Loading SigLIP ONNX INT8 image encoder...")

        model_path = hf_hub_download_local_first(_SIGLIP_ENCODER_REPO, "image_encoder_int8.onnx")
        self._siglip_session = ort.InferenceSession(
            model_path,
            create_cpu_session_options(ort),
            providers=["CPUExecutionProvider"],
        )

        embeddings_path = hf_hub_download_local_first(_SIGLIP_ENCODER_REPO, "text_embeddings.npz")
        loaded = np.load(embeddings_path)
        self._text_embeddings = {k: loaded[k] for k in loaded.files}

        logger.info(
            f"SigLIP image encoder loaded: ONNX INT8 "
            f"({Path(model_path).stat().st_size // 1024 // 1024} MB)"
        )

    def preload(self) -> None:
        """Eagerly build the ONNX sessions so request-time use is warm.

        Idempotent: ``_load_classifier`` / ``_load_siglip`` no-op when their
        session already exists. Called by the proxy warmup at startup so the
        shared (registry-cached) router never cold-builds its InferenceSession
        objects on the first image request — the source of the multi-second
        per-image latency.
        """
        self._load_classifier()
        if self.use_siglip:
            self._load_siglip()

    def classify_query(self, query: str) -> tuple[Technique, float]:
        """Classify query intent using ONNX technique router."""
        self._load_classifier()

        encoded = self._tokenizer.encode(query)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        logits = self._classifier_session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0][0]

        probs = np.exp(logits) / np.exp(logits).sum()
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

        technique_name = self._id2label.get(pred_id, "preserve")
        technique = Technique(technique_name)

        return technique, confidence

    def analyze_image(self, image_data: bytes) -> ImageSignals | None:
        """Analyze image properties using SigLIP ONNX encoder."""
        if not self.use_siglip:
            return None

        self._load_siglip()

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            img = img.resize((224, 224), Image.Resampling.LANCZOS)

            # Convert to numpy: [1, 3, 224, 224], normalized to [-1, 1]
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            pixel_values = arr[np.newaxis, ...]  # Add batch dim

            embeds = self._siglip_session.run(None, {"pixel_values": pixel_values})[0]
            embeds = embeds / np.linalg.norm(embeds, axis=-1, keepdims=True)

            def sigmoid(x: float) -> float:
                return 1 / (1 + math.exp(-x * 5))

            scores = {}
            for signal_name, text_emb in self._text_embeddings.items():
                sim = (embeds @ text_emb.T).squeeze()
                scores[signal_name] = sigmoid(float(sim.max()))

            return ImageSignals(
                has_text=scores.get("has_text", 0.5),
                is_document=scores.get("is_document", 0.5),
                is_complex=scores.get("is_complex", 0.5),
                has_small_details=scores.get("has_small_details", 0.5),
            )
        except Exception as e:
            logger.warning(f"SigLIP image analysis failed: {e}")
            return None

    def classify(self, image_data: bytes, query: str) -> RouteDecision:
        """Combined query + image classification."""
        technique, query_confidence = self.classify_query(query)
        image_signals = self.analyze_image(image_data)

        confidence = query_confidence
        reason = f"Query → '{technique.value}' ({query_confidence:.0%})"

        # Apply image-based adjustments
        if image_signals:
            if technique == Technique.TRANSCODE:
                if image_signals.has_text < 0.4 and image_signals.is_document < 0.4:
                    confidence *= 0.8
                    reason += " (low text in image)"
            elif technique == Technique.FULL_LOW:
                if image_signals.has_small_details > 0.7:
                    reason += " (note: fine details detected)"
            elif technique == Technique.PRESERVE:
                if image_signals.has_small_details > 0.5 or image_signals.is_complex > 0.5:
                    confidence = min(1.0, confidence * 1.1)
                    reason += " (confirmed: complex/detailed)"

        return RouteDecision(
            technique=technique,
            confidence=confidence,
            reason=reason,
            image_signals=image_signals,
        )
