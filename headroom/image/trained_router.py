"""Trained Technique Router using fine-tuned MiniLM + SigLIP.

Uses a TRAINED classifier for query intent:
1. MiniLM classifier: Fine-tuned on 1157 examples (93.7% accuracy)
2. SigLIP: Analyzes image properties
3. Combined decision based on both signals

The MiniLM model is hosted on HuggingFace: headroom-ai/technique-router
"""

from __future__ import annotations

import gc
import io
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import torch
    from PIL import Image
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    _IMAGE_ML_AVAILABLE = True
except ImportError:
    _IMAGE_ML_AVAILABLE = False

from headroom.models.config import ML_MODEL_DEFAULTS


def _extract_tensor(output: torch.Tensor | BaseModelOutputWithPooling) -> torch.Tensor:
    """Extract tensor from model output.

    Some transformers versions return BaseModelOutputWithPooling instead of
    raw tensors from get_image_features() / get_text_features(). This helper
    handles both cases.

    Args:
        output: Either a tensor or BaseModelOutputWithPooling object.

    Returns:
        The extracted tensor.
    """
    if isinstance(output, BaseModelOutputWithPooling):
        # Use pooler_output if available, otherwise last_hidden_state[:, 0]
        if output.pooler_output is not None:
            return output.pooler_output
        if output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0]
        # Fallback: shouldn't happen, but return empty tensor
        raise ValueError(
            "BaseModelOutputWithPooling has neither pooler_output nor last_hidden_state"
        )
    return output


class Technique(Enum):
    """Image optimization techniques."""

    TRANSCODE = "transcode"  # Convert to text description (99% savings)
    CROP = "crop"  # Extract relevant region (50-90% savings)
    PRESERVE = "preserve"  # Keep full quality (0% savings)
    FULL_LOW = "full_low"  # Full image, lower quality (87% savings)


@dataclass
class ImageSignals:
    """Signals extracted from image analysis."""

    has_text: float
    is_document: float
    is_complex: float
    has_small_details: float


@dataclass
class RouteDecision:
    """Result of routing decision."""

    technique: Technique
    confidence: float
    reason: str
    image_signals: ImageSignals | None = None
    query_prediction: str | None = None
    query_confidence: float | None = None


class TrainedRouter:
    """Router using trained MiniLM classifier + SigLIP image analysis.

    This router uses:
    1. A fine-tuned MiniLM classifier for query intent (93.7% accuracy)
    2. SigLIP for image property analysis
    3. Combined decision logic

    The MiniLM model can be loaded from:
    - Local path (for development)
    - HuggingFace Hub: headroom-ai/technique-router (for production)
    """

    # Model identifiers (from centralized config)
    @property
    def default_hf_model(self) -> str:
        return ML_MODEL_DEFAULTS.technique_router

    @property
    def siglip_model(self) -> str:
        return ML_MODEL_DEFAULTS.siglip

    # Image analysis prompts for SigLIP
    IMAGE_DESCRIPTIONS = {
        "has_text": [
            "an image with visible text, words, or writing",
            "a sign, label, or document with readable text",
        ],
        "is_document": [
            "a document, form, receipt, or page with text",
            "a scanned paper or screenshot of text",
        ],
        "is_complex": [
            "a complex scene with many objects and details",
            "a cluttered or busy image with lots of elements",
        ],
        "has_small_details": [
            "an image with fine details, small text, or intricate patterns",
            "a close-up showing texture, small objects, or fine features",
        ],
    }

    def __init__(
        self,
        model_path: str | None = None,
        use_siglip: bool = True,
        device: str | None = None,
    ):
        """Initialize the router.

        Args:
            model_path: Path to trained model (local or HF hub).
                        If None, uses default HF model.
            use_siglip: Whether to use SigLIP for image analysis.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        self.model_path = model_path
        self.use_siglip = use_siglip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded models
        self._classifier: Any = None
        self._tokenizer: Any = None
        self._siglip_model: Any = None
        self._siglip_processor: Any = None
        self._text_embeddings: Any = None
        self._classifier_key: str | None = None
        self._siglip_key: str | None = None

    def is_available(self) -> bool:
        """Check if required models can be loaded."""
        try:
            self._load_models()
            return True
        except Exception:
            return False

    def _load_models(self) -> None:
        """Lazy load the classifier and optionally SigLIP."""
        if self._classifier is None:
            # Determine model path
            if self.model_path:
                model_id = self.model_path
            else:
                # Check for local model first (development)
                local_path = (
                    Path(__file__).parent.parent.parent
                    / "models"
                    / "technique-router-mini"
                    / "final"
                )
                if local_path.exists():
                    model_id = str(local_path)
                else:
                    model_id = self.default_hf_model

            # Use centralized registry for shared model instances
            from headroom.models.ml_models import MLModelRegistry

            self._classifier, self._tokenizer = MLModelRegistry.get_technique_router(
                model_path=model_id,
                device=self.device,
            )
            self._classifier_key = f"technique_router:{model_id}"

        if self.use_siglip and self._siglip_model is None:
            # Use centralized registry for shared model instances
            from headroom.models.ml_models import MLModelRegistry

            self._siglip_model, self._siglip_processor = MLModelRegistry.get_siglip(
                model_name=self.siglip_model,
                device=self.device,
            )
            self._siglip_key = f"siglip:{self.siglip_model}"

            # Pre-compute text embeddings for image analysis
            self._compute_text_embeddings()

    def release_models(self, unload_registry: bool = True) -> None:
        """Release router-held model references and optional shared cache entries."""
        classifier_key = self._classifier_key
        siglip_key = self._siglip_key

        self._text_embeddings = None
        self._siglip_processor = None
        self._siglip_model = None
        self._tokenizer = None
        self._classifier = None
        self._classifier_key = None
        self._siglip_key = None

        if unload_registry:
            from headroom.models.ml_models import MLModelRegistry

            keys = [key for key in (classifier_key, siglip_key) if key]
            MLModelRegistry.unload_many(keys)
        else:
            gc.collect()

    def close(self, unload_registry: bool = True) -> None:
        """Alias for release_models() while preserving subclass dispatch."""
        self.release_models(unload_registry=unload_registry)

    def _compute_text_embeddings(self) -> None:
        """Pre-compute SigLIP text embeddings for image analysis."""
        assert self._siglip_processor is not None
        assert self._siglip_model is not None

        self._text_embeddings = {}

        with torch.no_grad():
            for signal_name, descriptions in self.IMAGE_DESCRIPTIONS.items():
                embeddings = []
                for desc in descriptions:
                    inputs = self._siglip_processor(
                        text=[desc],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    text_output = self._siglip_model.get_text_features(**inputs)
                    text_embeds = _extract_tensor(text_output)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    embeddings.append(text_embeds)

                self._text_embeddings[signal_name] = torch.cat(embeddings, dim=0)

    def _classify_query(self, query: str) -> tuple[Technique, float]:
        """Classify query intent using trained model.

        Returns:
            Tuple of (predicted_technique, confidence)
        """
        self._load_models()
        assert self._tokenizer is not None
        assert self._classifier is not None

        inputs = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1).item())
            confidence = probs[0][pred_id].item()

        # Map ID to technique
        id2label = self._classifier.config.id2label
        technique_name = id2label[pred_id]
        technique = Technique(technique_name)

        return technique, confidence

    def _get_image_embedding(self, image_data: bytes) -> torch.Tensor:
        """Get SigLIP embedding for image."""
        assert self._siglip_processor is not None
        assert self._siglip_model is not None

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        inputs = self._siglip_processor(
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_output = self._siglip_model.get_image_features(**inputs)
            image_embeds: torch.Tensor = _extract_tensor(image_output)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds

    def _analyze_image(self, image_embedding: torch.Tensor) -> ImageSignals:
        """Analyze image properties using SigLIP."""
        assert self._text_embeddings is not None

        scores: dict[str, float] = {}

        def sigmoid(x: float) -> float:
            import math

            return 1 / (1 + math.exp(-x * 5))

        with torch.no_grad():
            for signal_name, text_embeds in self._text_embeddings.items():
                # Compute similarity with each description
                similarities = (image_embedding @ text_embeds.T).squeeze(0)
                # Take max similarity across descriptions
                max_sim = similarities.max().item()
                scores[signal_name] = max_sim

        return ImageSignals(
            has_text=sigmoid(scores["has_text"]),
            is_document=sigmoid(scores["is_document"]),
            is_complex=sigmoid(scores["is_complex"]),
            has_small_details=sigmoid(scores["has_small_details"]),
        )

    def classify(self, image_data: bytes, query: str) -> RouteDecision:
        """Classify query + image to determine optimal technique.

        Args:
            image_data: Raw image bytes
            query: User's query about the image

        Returns:
            RouteDecision with technique, confidence, and reasoning
        """
        self._load_models()

        # Step 1: Classify query with trained model
        technique, query_confidence = self._classify_query(query)

        # Step 2: Analyze image with SigLIP (if enabled)
        image_signals = None
        if self.use_siglip:
            image_embedding = self._get_image_embedding(image_data)
            image_signals = self._analyze_image(image_embedding)

        # Step 3: Combine signals for final decision
        final_technique = technique
        confidence = query_confidence
        reason = f"Query classified as '{technique.value}' with {query_confidence:.0%} confidence"

        # Apply image-based adjustments
        if image_signals:
            # If query says TRANSCODE but image has no text, might want to reconsider
            if technique == Technique.TRANSCODE:
                if image_signals.has_text < 0.4 and image_signals.is_document < 0.4:
                    # Low text signal - reduce confidence but keep technique
                    # (user explicitly asked for text, they may know better)
                    confidence *= 0.8
                    reason += " (note: low text detected in image)"

            # If query says FULL_LOW but image has small details, might need PRESERVE
            elif technique == Technique.FULL_LOW:
                if image_signals.has_small_details > 0.7:
                    # Image has fine details - suggest they might need PRESERVE
                    reason += " (note: image has fine details, consider PRESERVE)"

            # If query says PRESERVE, boost confidence if image confirms
            elif technique == Technique.PRESERVE:
                if image_signals.has_small_details > 0.5 or image_signals.is_complex > 0.5:
                    confidence = min(1.0, confidence * 1.1)
                    reason += " (confirmed: image has fine details)"

        return RouteDecision(
            technique=final_technique,
            confidence=confidence,
            reason=reason,
            image_signals=image_signals,
            query_prediction=technique.value,
            query_confidence=query_confidence,
        )


def get_trained_router(model_path: str | None = None) -> TrainedRouter:
    """Get a trained router instance.

    Args:
        model_path: Optional path to model (local or HF hub).
                   If None, uses local model if available, else HF hub.
    """
    return TrainedRouter(model_path=model_path)
