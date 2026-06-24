"""Universal compression with ML-based content detection.

This module provides intelligent, automatic compression that:
1. Detects content type using ML (Magika)
2. Preserves structure (keys, signatures, templates)
3. Compresses content with Kompress
4. Enables retrieval via CCR

Quick Start:
    # One-liner for simple use
    from headroom.compression import compress
    result = compress(content)

    # Or with configuration
    from headroom.compression import UniversalCompressor, UniversalCompressorConfig

    config = UniversalCompressorConfig(compression_ratio_target=0.5)
    compressor = UniversalCompressor(config=config)
    result = compressor.compress(content)
"""

from headroom.compression.detector import ContentType, MagikaDetector
from headroom.compression.masks import StructureMask
from headroom.compression.universal import (
    CompressionResult,
    UniversalCompressor,
    UniversalCompressorConfig,
    compress,
)

__all__ = [
    # Simple API
    "compress",
    # Full API
    "UniversalCompressor",
    "UniversalCompressorConfig",
    "CompressionResult",
    # Advanced
    "MagikaDetector",
    "ContentType",
    "StructureMask",
]
