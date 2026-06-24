"""Universal compressor with ML-based detection and structure preservation.

This is the main entry point for compression. It:
1. Detects content type using Magika (ML)
2. Extracts structure using appropriate handler
3. Compresses non-structural content with Kompress
4. Optionally stores original in CCR for retrieval

Usage:
    compressor = UniversalCompressor()
    result = compressor.compress(content)

    # Result contains:
    # - compressed: The compressed content
    # - compression_ratio: original_tokens / compressed_tokens
    # - content_type: Detected content type
    # - preservation_ratio: Fraction of content preserved as structure
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from headroom.compression.detector import (
    ContentType,
    DetectionResult,
    FallbackDetector,
    get_detector,
)
from headroom.compression.handlers.base import (
    NoOpHandler,
    StructureHandler,
)
from headroom.compression.handlers.code_handler import CodeStructureHandler
from headroom.compression.handlers.json_handler import JSONStructureHandler
from headroom.compression.masks import (
    StructureMask,
    compute_entropy_mask,
    mask_to_spans,
)

logger = logging.getLogger(__name__)


@dataclass
class UniversalCompressorConfig:
    """Configuration for UniversalCompressor.

    Attributes:
        use_magika: Use ML-based detection (requires magika package).
        use_kompress: Use Kompress for content compression.
        use_entropy_preservation: Preserve high-entropy tokens (UUIDs, etc.).
        entropy_threshold: Threshold for entropy-based preservation.
        min_content_length: Minimum content length to compress.
        compression_ratio_target: Target compression ratio (0.0-1.0).
        ccr_enabled: Store originals in CCR for retrieval.
    """

    use_magika: bool = True
    use_kompress: bool = True
    use_entropy_preservation: bool = True
    entropy_threshold: float = 0.85
    min_content_length: int = 100
    compression_ratio_target: float = 0.3  # Target 70% reduction
    ccr_enabled: bool = True


@dataclass
class CompressionResult:
    """Result from compression.

    Attributes:
        compressed: The compressed content.
        original: The original content (for reference).
        compression_ratio: compressed_length / original_length.
        tokens_before: Estimated token count before compression.
        tokens_after: Estimated token count after compression.
        content_type: Detected content type.
        detection_confidence: Confidence of content type detection.
        handler_used: Name of structure handler used.
        preservation_ratio: Fraction of content marked as structural.
        ccr_key: CCR storage key (if CCR enabled).
        metadata: Additional metadata.
    """

    compressed: str
    original: str
    compression_ratio: float
    tokens_before: int
    tokens_after: int
    content_type: ContentType
    detection_confidence: float
    handler_used: str
    preservation_ratio: float
    ccr_key: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.tokens_before == 0:
            return 0.0
        return (self.tokens_saved / self.tokens_before) * 100


class UniversalCompressor:
    """Universal compressor with ML detection and structure preservation.

    This compressor automatically:
    1. Detects content type (JSON, code, logs, text) using ML
    2. Extracts structure (keys, signatures, templates)
    3. Preserves structure while compressing content
    4. Stores original for CCR retrieval

    Example:
        >>> compressor = UniversalCompressor()
        >>> result = compressor.compress('{"users": [{"id": 1, "name": "Alice"}]}')
        >>> print(result.content_type)  # ContentType.JSON
        >>> print(result.compressed)     # Structure preserved, values compressed
    """

    def __init__(
        self,
        config: UniversalCompressorConfig | None = None,
        handlers: dict[ContentType, StructureHandler] | None = None,
        compress_fn: Callable[[str], str] | None = None,
    ):
        """Initialize the compressor.

        Args:
            config: Compression configuration.
            handlers: Custom handlers for content types.
            compress_fn: Custom compression function. If None, uses
                Kompress when available, else simple truncation.
        """
        self.config = config or UniversalCompressorConfig()

        # Initialize detector
        if self.config.use_magika:
            self._detector = get_detector(prefer_magika=True)
        else:
            self._detector = FallbackDetector()

        # Initialize handlers
        self._handlers: dict[ContentType, StructureHandler] = handlers or {
            ContentType.JSON: JSONStructureHandler(),
            ContentType.CODE: CodeStructureHandler(),
        }
        self._noop_handler = NoOpHandler()

        # Initialize compression function
        self._compress_fn = compress_fn or self._get_default_compress_fn()

        # CCR store (lazy initialized)
        self._ccr_store: Any | None = None

    def _get_default_compress_fn(self) -> Callable[[str], str]:
        """Get default compression function.

        Returns Kompress wrapper if available, else simple truncation.
        """
        if self.config.use_kompress:
            try:
                return self._kompress_compress
            except ImportError:
                logger.info("Kompress not available, using simple compression")

        return self._simple_compress

    def _kompress_compress(self, text: str) -> str:
        """Compress using Kompress.

        Args:
            text: Text to compress.

        Returns:
            Compressed text.
        """
        try:
            from headroom.transforms.kompress_compressor import KompressCompressor

            compressor = KompressCompressor()
            result = compressor.compress(text)
            return result.compressed
        except ImportError:
            return self._simple_compress(text)
        except Exception as e:
            logger.warning("Kompress compression failed: %s", e)
            return self._simple_compress(text)

    def _simple_compress(self, text: str) -> str:
        """Simple compression fallback (truncation with indicator).

        Args:
            text: Text to compress.

        Returns:
            Truncated text with indicator.
        """
        target_len = int(len(text) * self.config.compression_ratio_target)
        if len(text) <= target_len:
            return text

        # Keep first and last portions
        keep_start = target_len * 2 // 3
        keep_end = target_len // 3

        return text[:keep_start] + "\n...[compressed]...\n" + text[-keep_end:]

    def compress(
        self,
        content: str,
        content_type: ContentType | None = None,
        **kwargs: Any,
    ) -> CompressionResult:
        """Compress content with structure preservation.

        Args:
            content: Content to compress.
            content_type: Override content type detection.
            **kwargs: Handler-specific options.

        Returns:
            CompressionResult with compressed content and metadata.
        """
        # Handle empty/short content
        if not content or len(content) < self.config.min_content_length:
            return CompressionResult(
                compressed=content,
                original=content,
                compression_ratio=1.0,
                tokens_before=self._estimate_tokens(content),
                tokens_after=self._estimate_tokens(content),
                content_type=ContentType.UNKNOWN,
                detection_confidence=0.0,
                handler_used="none",
                preservation_ratio=1.0,
                metadata={"skipped": "content too short"},
            )

        # Detect content type
        if content_type is None:
            detection = self._detector.detect(content)
        else:
            detection = DetectionResult(
                content_type=content_type,
                confidence=1.0,
                raw_label="override",
            )

        # Get handler for content type
        handler = self._handlers.get(detection.content_type, self._noop_handler)

        # Tokenize content (character-level for masks)
        tokens = list(content)

        # Get structure mask from handler
        handler_result = handler.get_mask(content, tokens, **kwargs)
        structure_mask = handler_result.mask

        # Optionally add entropy-based preservation
        if self.config.use_entropy_preservation:
            entropy_mask = compute_entropy_mask(
                tokens,
                threshold=self.config.entropy_threshold,
            )
            # Union: preserve if either mask says preserve
            structure_mask = structure_mask.union(entropy_mask)

        # Apply compression to non-structural parts
        compressed = self._compress_with_mask(content, structure_mask)

        # Estimate tokens
        tokens_before = self._estimate_tokens(content)
        tokens_after = self._estimate_tokens(compressed)

        # Store in CCR if enabled
        ccr_key = None
        if self.config.ccr_enabled:
            ccr_key = self._store_in_ccr(content, compressed)

        return CompressionResult(
            compressed=compressed,
            original=content,
            compression_ratio=len(compressed) / len(content) if content else 1.0,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            content_type=detection.content_type,
            detection_confidence=detection.confidence,
            handler_used=handler_result.handler_name,
            preservation_ratio=structure_mask.preservation_ratio,
            ccr_key=ccr_key,
            metadata={
                "detection": {
                    "raw_label": detection.raw_label,
                    "language": detection.language,
                },
                "handler": handler_result.metadata,
            },
        )

    def _compress_with_mask(self, content: str, mask: StructureMask) -> str:
        """Apply compression respecting structure mask.

        Args:
            content: Original content.
            mask: Structure mask.

        Returns:
            Compressed content with structure preserved.
        """
        spans = mask_to_spans(mask)
        result_parts: list[str] = []

        for span in spans:
            span_content = content[span.start : span.end]

            if span.is_structural:
                # Preserve structural content
                result_parts.append(span_content)
            else:
                # Compress non-structural content
                if len(span_content) > 50:  # Only compress if substantial
                    compressed = self._compress_fn(span_content)
                    result_parts.append(compressed)
                else:
                    result_parts.append(span_content)

        return "".join(result_parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count.

        Uses simple heuristic: ~4 characters per token.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # Simple estimation: ~4 chars per token on average
        return len(text) // 4

    def _store_in_ccr(self, original: str, compressed: str) -> str | None:
        """Store original in CCR for retrieval.

        Args:
            original: Original content.
            compressed: Compressed content.

        Returns:
            CCR key if stored, None otherwise.
        """
        try:
            if self._ccr_store is None:
                from headroom.cache.compression_store import CompressionStore

                self._ccr_store = CompressionStore()

            key = self._ccr_store.store(
                original,
                compressed,
                original_tokens=self._estimate_tokens(original),
                compressed_tokens=self._estimate_tokens(compressed),
            )
            return str(key) if key else None
        except ImportError:
            logger.debug("CCR store not available")
            return None
        except Exception as e:
            logger.warning("Failed to store in CCR: %s", e)
            return None

    def compress_batch(
        self,
        contents: list[str],
        **kwargs: Any,
    ) -> list[CompressionResult]:
        """Compress multiple contents.

        More efficient than calling compress() in a loop for
        ML detection.

        Args:
            contents: List of contents to compress.
            **kwargs: Handler-specific options.

        Returns:
            List of CompressionResults.
        """
        if not contents:
            return []

        # Batch detection
        if hasattr(self._detector, "detect_batch"):
            detections = self._detector.detect_batch(contents)
        else:
            detections = [self._detector.detect(c) for c in contents]

        # Compress each with detected type
        results = []
        for content, detection in zip(contents, detections):
            result = self.compress(
                content,
                content_type=detection.content_type,
                **kwargs,
            )
            results.append(result)

        return results

    def get_handler(self, content_type: ContentType) -> StructureHandler:
        """Get handler for content type.

        Args:
            content_type: Content type.

        Returns:
            Handler for the content type.
        """
        return self._handlers.get(content_type, self._noop_handler)

    def register_handler(
        self,
        content_type: ContentType,
        handler: StructureHandler,
    ) -> None:
        """Register a custom handler for a content type.

        Args:
            content_type: Content type to handle.
            handler: Handler instance.
        """
        self._handlers[content_type] = handler


def compress(content: str, **kwargs: Any) -> CompressionResult:
    """Convenience function for one-off compression.

    Args:
        content: Content to compress.
        **kwargs: Passed to UniversalCompressor.compress().

    Returns:
        CompressionResult.

    Example:
        >>> from headroom.compression import compress
        >>> result = compress('{"users": [{"id": 1}, {"id": 2}]}')
        >>> print(result.compressed)
    """
    compressor = UniversalCompressor()
    return compressor.compress(content, **kwargs)
