"""Central configuration for all ML models used in Headroom.

This is the SINGLE SOURCE OF TRUTH for model defaults. Change values here
to switch model variants across the entire codebase.

Usage:
    from headroom.models.config import ML_MODEL_DEFAULTS

    # Get default model name
    model = ML_MODEL_DEFAULTS.sentence_transformer

    # Or use environment variables to override at runtime:
    # HEADROOM_SENTENCE_TRANSFORMER=intfloat/e5-small-v2
    # HEADROOM_SIGLIP=google/siglip-base-patch16-224
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MLModelConfig:
    """Central configuration for all ML model defaults.

    All model names are defined here. Components throughout Headroom
    import these defaults, so changing a value here changes it everywhere.

    Environment variables can override any default:
    - HEADROOM_SENTENCE_TRANSFORMER
    - HEADROOM_SIGLIP
    - HEADROOM_SPACY
    - HEADROOM_TECHNIQUE_ROUTER

    Attributes:
        sentence_transformer: Model for text embeddings (semantic similarity, memory).
            Default: all-MiniLM-L6-v2 (22M params, 384 dim, ~90MB)
            Alternative: intfloat/e5-small-v2 (33M params, better accuracy)

        sentence_transformer_dim: Embedding dimension for the sentence transformer.
            Must match the model's output dimension.

        siglip: Model for image embeddings and analysis.
            Default: google/siglip-base-patch16-224 (~400MB)
            Alternative: google/siglip-so400m-patch14-384 (larger, more accurate)

        spacy: Model for named entity recognition.
            Default: en_core_web_sm (~40MB)
            Alternative: en_core_web_md (~120MB, more accurate)

        technique_router: Model for image optimization routing.
            Default: chopratejas/technique-router (~100MB)
    """

    # Text Embeddings (SentenceTransformer)
    sentence_transformer: str = field(
        default_factory=lambda: os.environ.get("HEADROOM_SENTENCE_TRANSFORMER", "all-MiniLM-L6-v2")
    )
    sentence_transformer_dim: int = 384

    # Image Embeddings (SIGLIP)
    siglip: str = field(
        default_factory=lambda: os.environ.get("HEADROOM_SIGLIP", "google/siglip-base-patch16-224")
    )

    # Named Entity Recognition (spaCy)
    spacy: str = field(default_factory=lambda: os.environ.get("HEADROOM_SPACY", "en_core_web_sm"))

    # Image Technique Router
    technique_router: str = field(
        default_factory=lambda: os.environ.get(
            "HEADROOM_TECHNIQUE_ROUTER", "chopratejas/technique-router"
        )
    )

    # Memory estimates in MB (for monitoring)
    _memory_estimates: dict[str, int] = field(
        default_factory=lambda: {
            # Sentence Transformers
            "all-MiniLM-L6-v2": 90,
            "all-mpnet-base-v2": 420,
            "intfloat/e5-small-v2": 130,
            "intfloat/e5-base-v2": 440,
            # SIGLIP
            "google/siglip-base-patch16-224": 400,
            "google/siglip-so400m-patch14-384": 900,
            "google/siglip-large-patch16-384": 1200,
            # spaCy
            "en_core_web_sm": 40,
            "en_core_web_md": 120,
            "en_core_web_lg": 560,
            # Technique Router
            "chopratejas/technique-router": 100,
        }
    )

    def get_memory_estimate(self, model_name: str) -> int:
        """Get estimated memory usage for a model in MB.

        Args:
            model_name: The model identifier.

        Returns:
            Estimated memory in MB, or 100 if unknown.
        """
        return self._memory_estimates.get(model_name, 100)

    def total_memory_estimate(self) -> int:
        """Get total estimated memory if all configured models are loaded.

        Returns:
            Total estimated memory in MB.
        """
        return (
            self.get_memory_estimate(self.sentence_transformer)
            + self.get_memory_estimate(self.siglip)
            + self.get_memory_estimate(self.spacy)
            + self.get_memory_estimate(self.technique_router)
        )


# Singleton instance - import this to get defaults
ML_MODEL_DEFAULTS = MLModelConfig()


# Convenience accessors for common use cases
def get_default_embedding_model() -> str:
    """Get the default sentence transformer model name."""
    return ML_MODEL_DEFAULTS.sentence_transformer


def get_default_embedding_dim() -> int:
    """Get the default embedding dimension."""
    return ML_MODEL_DEFAULTS.sentence_transformer_dim


def get_default_spacy_model() -> str:
    """Get the default spaCy model name."""
    return ML_MODEL_DEFAULTS.spacy


def get_default_siglip_model() -> str:
    """Get the default SIGLIP model name."""
    return ML_MODEL_DEFAULTS.siglip
