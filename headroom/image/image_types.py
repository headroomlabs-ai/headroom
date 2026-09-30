"""Lightweight image-routing types shared across the image stack.

Kept dependency-free (pure enum + dataclasses, no torch / transformers / onnx)
so importing the image compressor or the ONNX router does not eagerly import
the heavy ML stack via ``trained_router``. On Python 3.13+ that eager import
crashed with ``AttributeError: module 'torch' has no attribute 'compiler'``
because ``transformers`` touched ``torch.compiler`` before torch finished
initializing inside the proxy process (#2513).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
