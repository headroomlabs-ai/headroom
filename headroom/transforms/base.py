"""Base transform interface for Headroom SDK."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer


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
