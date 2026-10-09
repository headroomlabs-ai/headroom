"""Base transform interface for Headroom SDK."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def persist_rust_ccr_entry(original: str, compressed: str, cache_key: str, *, source: str) -> None:
    """Store a Rust-emitted CCR entry in the production ``CompressionStore``.

    The Rust search/diff/log compressors emit a retrieval marker but keep the
    original only in their in-memory test store. This writes it through to the
    long-lived Python store so the marker resolves. Failures are logged at
    warning level with ``source`` (the compressor name): a store hiccup must not
    break the response, just degrade retrieval.
    """
    try:
        from ..cache.compression_store import get_compression_store
    except ImportError as e:
        logger.warning(
            "CCR store import failed (%s); cache_key %s won't persist: %s", source, cache_key, e
        )
        return
    try:
        store: Any = get_compression_store()
        # The Rust-emitted marker embeds MD5(original)[:24], but store() has
        # defaulted to SHA-256(original)[:24] since PR #395. Pass the marker's
        # key explicitly so retrieving the marker hash finds the entry (#816).
        store.store(original, compressed, explicit_hash=cache_key)
    except Exception as e:
        logger.warning(
            "CCR store write failed (%s); cache_key %s remains in-marker only: %s",
            source,
            cache_key,
            e,
        )


def split_frozen(
    messages: list[dict[str, Any]],
    frozen_message_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into frozen (cached prefix) and mutable portions.

    Args:
        messages: All messages.
        frozen_message_count: Number of leading messages to freeze.

    Returns:
        (frozen, mutable) — frozen messages must not be modified.
    """
    if frozen_message_count <= 0 or frozen_message_count >= len(messages):
        return [], messages
    return messages[:frozen_message_count], messages[frozen_message_count:]


class Transform(ABC):
    """Abstract base class for message transforms."""

    name: str = "base"

    @abstractmethod
    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Apply the transform to messages.

        Args:
            messages: List of message dicts to transform.
            tokenizer: Tokenizer for token counting.
            **kwargs: Additional transform-specific arguments.
                frozen_message_count: Number of leading messages in the
                    provider's prefix cache. Transforms should skip these
                    to avoid invalidating the cache.

        Returns:
            TransformResult with transformed messages and metadata.
        """
        pass

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """
        Check if this transform should be applied.

        Default implementation always returns True.
        Override in subclasses for conditional application.

        Args:
            messages: List of message dicts.
            tokenizer: Tokenizer for token counting.
            **kwargs: Additional arguments.

        Returns:
            True if transform should be applied.
        """
        return True
