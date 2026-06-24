"""Centralized registry for ML model instances.

Provides shared access to ML models (sentence transformers, SIGLIP, spaCy, etc.)
to avoid loading the same model multiple times across different components.

This is different from registry.py which stores LLM metadata. This module
manages actual loaded model instances that consume memory.

Model defaults are configured in headroom/models/config.py - change them there
to switch model variants across the entire codebase.

Usage:
    from headroom.models.ml_models import MLModelRegistry

    # Get shared sentence transformer (loads on first access, uses config default)
    model = MLModelRegistry.get_sentence_transformer()
    embeddings = model.encode(["hello", "world"])

    # Get SIGLIP for image embeddings
    siglip_model, processor = MLModelRegistry.get_siglip()

    # Check what's loaded
    print(MLModelRegistry.loaded_models())
    print(f"Memory: {MLModelRegistry.estimated_memory_mb():.1f} MB")
"""

from __future__ import annotations

import contextlib
import gc
import logging
from threading import RLock
from typing import TYPE_CHECKING, Any

from .config import ML_MODEL_DEFAULTS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MLModelRegistry:
    """Singleton registry for shared ML model instances.

    Provides lazy-loaded, shared access to ML models across all components.
    This prevents the same model from being loaded multiple times.

    Thread-safe for concurrent access.
    """

    _instance: MLModelRegistry | None = None
    _lock = RLock()

    def __new__(cls) -> MLModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize the registry."""
        self._models: dict[str, Any] = {}
        self._model_lock = RLock()

    @classmethod
    def get(cls) -> MLModelRegistry:
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._models.clear()
            cls._instance = None

    @classmethod
    def _release_runtime_memory(cls) -> None:
        """Best-effort cleanup after unloading heavyweight models."""
        gc.collect()
        try:
            import torch
        except ImportError:
            return

        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()

    @classmethod
    def unload(cls, key: str) -> bool:
        """Unload one cached model entry."""
        return bool(cls.unload_many([key]))

    @classmethod
    def unload_many(cls, keys: list[str]) -> list[str]:
        """Unload several cached model entries with one runtime cleanup pass."""
        instance = cls.get()
        removed_keys: list[str] = []

        with instance._model_lock:
            for key in keys:
                if key not in instance._models:
                    continue
                value = instance._models.pop(key)
                del value
                removed_keys.append(key)

        if removed_keys:
            cls._release_runtime_memory()
        return removed_keys

    @classmethod
    def unload_prefix(cls, prefix: str) -> list[str]:
        """Unload every cached model entry matching a prefix."""
        instance = cls.get()
        removed_keys: list[str] = []

        with instance._model_lock:
            for key in list(instance._models):
                if key.startswith(prefix):
                    value = instance._models.pop(key)
                    del value
                    removed_keys.append(key)

        if removed_keys:
            cls._release_runtime_memory()
        return removed_keys

    # =========================================================================
    # Sentence Transformers
    # =========================================================================

    @classmethod
    def get_sentence_transformer(
        cls,
        model_name: str | None = None,
        device: str | None = None,
    ) -> Any:
        """Get a shared SentenceTransformer instance.

        Args:
            model_name: Model name. If None, uses ML_MODEL_DEFAULTS.sentence_transformer.
            device: Device to use (cuda, mps, cpu). Auto-detected if None.

        Returns:
            SentenceTransformer model instance.
        """
        if model_name is None:
            model_name = ML_MODEL_DEFAULTS.sentence_transformer

        instance = cls.get()
        key = f"sentence_transformer:{model_name}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading SentenceTransformer: {model_name}")
                from sentence_transformers import SentenceTransformer

                if device is None:
                    device = cls._detect_device()

                model = SentenceTransformer(model_name, device=device)
                instance._models[key] = model
                logger.info(f"Loaded SentenceTransformer: {model_name} on {device}")

            return instance._models[key]

    # =========================================================================
    # SIGLIP (Image Embeddings)
    # =========================================================================

    @classmethod
    def get_siglip(
        cls,
        model_name: str | None = None,
        device: str | None = None,
    ) -> tuple[Any, Any]:
        """Get shared SIGLIP model and processor.

        Args:
            model_name: Model name. If None, uses ML_MODEL_DEFAULTS.siglip.
            device: Device to use. Auto-detected if None.

        Returns:
            Tuple of (model, processor).
        """
        if model_name is None:
            model_name = ML_MODEL_DEFAULTS.siglip

        instance = cls.get()
        key = f"siglip:{model_name}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading SIGLIP: {model_name}")
                from transformers import AutoModel, AutoProcessor

                if device is None:
                    device = cls._detect_device()

                model = AutoModel.from_pretrained(model_name)
                processor = AutoProcessor.from_pretrained(model_name)

                # Move to device and set eval mode
                if device != "cpu":
                    import torch

                    model = model.to(torch.device(device))
                model.eval()

                instance._models[key] = (model, processor)
                logger.info(f"Loaded SIGLIP: {model_name} on {device}")

            result: tuple[Any, Any] = instance._models[key]
            return result

    # =========================================================================
    # spaCy
    # =========================================================================

    @classmethod
    def get_spacy(cls, model_name: str | None = None) -> Any:
        """Get a shared spaCy model.

        Args:
            model_name: Model name. If None, uses ML_MODEL_DEFAULTS.spacy.

        Returns:
            spaCy Language model.
        """
        if model_name is None:
            model_name = ML_MODEL_DEFAULTS.spacy

        instance = cls.get()
        key = f"spacy:{model_name}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading spaCy: {model_name}")
                import spacy

                model = spacy.load(model_name)
                instance._models[key] = model
                logger.info(f"Loaded spaCy: {model_name}")

            return instance._models[key]

    # =========================================================================
    # Technique Router (Sequence Classification)
    # =========================================================================

    @classmethod
    def get_technique_router(
        cls,
        model_path: str | None = None,
        device: str | None = None,
    ) -> tuple[Any, Any]:
        """Get shared technique router model and tokenizer.

        Args:
            model_path: Path to model (default: chopratejas/technique-router).
            device: Device to use. Auto-detected if None.

        Returns:
            Tuple of (model, tokenizer).
        """
        from pathlib import Path

        instance = cls.get()

        # Default to HuggingFace model, check for local first
        if model_path is None:
            local_path = Path("headroom/models/technique-router-mini/final/")
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = ML_MODEL_DEFAULTS.technique_router

        key = f"technique_router:{model_path}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading technique router: {model_path}")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                if device is None:
                    device = cls._detect_device()

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)

                # Move to device and set eval mode
                if device != "cpu":
                    import torch

                    model = model.to(torch.device(device))
                model.eval()

                instance._models[key] = (model, tokenizer)
                logger.info(f"Loaded technique router: {model_path} on {device}")

            result: tuple[Any, Any] = instance._models[key]
            return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @classmethod
    def _detect_device(cls) -> str:
        """Auto-detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def loaded_models(cls) -> list[str]:
        """Get list of currently loaded model keys."""
        instance = cls.get()
        with instance._model_lock:
            return list(instance._models.keys())

    @classmethod
    def is_loaded(cls, key: str) -> bool:
        """Check if a model is loaded."""
        instance = cls.get()
        with instance._model_lock:
            return key in instance._models

    @classmethod
    def estimated_memory_mb(cls) -> float:
        """Estimate total memory used by loaded models."""
        instance = cls.get()
        total = 0.0
        with instance._model_lock:
            for key in instance._models:
                # Extract model name from key (format: "type:model_name")
                model_name = key.split(":", 1)[1] if ":" in key else key
                total += ML_MODEL_DEFAULTS.get_memory_estimate(model_name)
        return total

    @classmethod
    def get_memory_stats(cls) -> dict[str, Any]:
        """Get memory statistics for all loaded models."""
        instance = cls.get()
        loaded_models: list[dict[str, Any]] = []
        total_estimated_mb: float = 0.0

        with instance._model_lock:
            for key in instance._models:
                # Extract model name from key (format: "type:model_name")
                model_name = key.split(":", 1)[1] if ":" in key else key
                size_mb = ML_MODEL_DEFAULTS.get_memory_estimate(model_name)
                loaded_models.append({"key": key, "size_mb": size_mb})
                total_estimated_mb += size_mb

        return {
            "loaded_models": loaded_models,
            "total_estimated_mb": total_estimated_mb,
        }


# Convenience functions for direct access
def get_sentence_transformer(
    model_name: str | None = None,
    device: str | None = None,
) -> Any:
    """Get a shared SentenceTransformer instance."""
    return MLModelRegistry.get_sentence_transformer(model_name, device)


def get_siglip(
    model_name: str | None = None,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Get shared SIGLIP model and processor."""
    return MLModelRegistry.get_siglip(model_name, device)


def get_spacy(model_name: str | None = None) -> Any:
    """Get a shared spaCy model."""
    return MLModelRegistry.get_spacy(model_name)
