"""Base class and protocol for structure handlers.

Structure handlers extract structural information from content and create
masks identifying what should be preserved during compression.

The handler protocol is simple:
1. get_mask(content) -> StructureMask
2. can_handle(content) -> bool (optional)

Handlers are content-type specific but domain-agnostic. A JSONStructureHandler
preserves JSON keys whether it's user data, search results, or config files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from headroom.compression.masks import StructureMask


@dataclass
class HandlerResult:
    """Result from a structure handler.

    Contains the mask plus metadata about what was detected.
    """

    mask: StructureMask
    handler_name: str
    confidence: float = 1.0  # How confident the handler is in its detection
    metadata: dict = field(default_factory=dict)

    @property
    def preservation_ratio(self) -> float:
        """Fraction of content marked for preservation."""
        return self.mask.preservation_ratio


@runtime_checkable
class StructureHandler(Protocol):
    """Protocol for structure handlers.

    Any class implementing get_mask() can be used as a handler.
    """

    @property
    def name(self) -> str:
        """Handler name for logging and metadata."""
        ...

    def get_mask(
        self,
        content: str,
        tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> HandlerResult:
        """Extract structure mask from content.

        Args:
            content: The content to analyze.
            tokens: Pre-tokenized content (optional). If not provided,
                handler should tokenize internally.
            **kwargs: Handler-specific options.

        Returns:
            HandlerResult with mask and metadata.
        """
        ...

    def can_handle(self, content: str) -> bool:
        """Check if this handler can process the content.

        Default implementation returns True. Override for handlers
        that need to verify content format before processing.

        Args:
            content: The content to check.

        Returns:
            True if handler can process this content.
        """
        ...


class BaseStructureHandler(ABC):
    """Base implementation for structure handlers.

    Provides common functionality and enforces the handler interface.
    Subclasses must implement _extract_mask().
    """

    def __init__(self, name: str | None = None):
        """Initialize the handler.

        Args:
            name: Optional handler name. Defaults to class name.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Handler name."""
        return self._name

    def get_mask(
        self,
        content: str,
        tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> HandlerResult:
        """Extract structure mask from content.

        This is the main entry point. It handles common logic like
        empty content and delegates to _extract_mask() for the
        content-specific logic.

        Args:
            content: The content to analyze.
            tokens: Pre-tokenized content (optional).
            **kwargs: Handler-specific options.

        Returns:
            HandlerResult with mask and metadata.
        """
        # Handle empty content
        if not content or not content.strip():
            tokens = tokens or []
            return HandlerResult(
                mask=StructureMask.empty(tokens),
                handler_name=self.name,
                confidence=0.0,
                metadata={"empty": True},
            )

        # Tokenize if not provided
        if tokens is None:
            tokens = self._tokenize(content)

        # Delegate to subclass
        return self._extract_mask(content, tokens, **kwargs)

    def can_handle(self, content: str) -> bool:
        """Check if this handler can process the content.

        Default implementation returns True. Override for handlers
        that need to verify content format.

        Args:
            content: The content to check.

        Returns:
            True if handler can process this content.
        """
        return True

    @abstractmethod
    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Extract structure mask from content.

        Subclasses implement this to provide content-specific logic.

        Args:
            content: The content to analyze (non-empty, stripped).
            tokens: Tokenized content.
            **kwargs: Handler-specific options.

        Returns:
            HandlerResult with mask and metadata.
        """
        ...

    def _tokenize(self, content: str) -> list[str]:
        """Default tokenization - character-level.

        Subclasses may override for more sophisticated tokenization.
        For mask purposes, character-level is often sufficient and
        aligns well with token-level compression.

        Args:
            content: Content to tokenize.

        Returns:
            List of tokens (characters by default).
        """
        # Simple character-level tokenization
        # This aligns well with structure detection (we mark ranges)
        return list(content)


class NoOpHandler(BaseStructureHandler):
    """Handler that marks everything as compressible.

    Used as a fallback when no structure is detected.
    """

    def __init__(self) -> None:
        """Initialize the no-op handler."""
        super().__init__(name="noop")

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Return mask with everything compressible."""
        return HandlerResult(
            mask=StructureMask.empty(tokens),
            handler_name=self.name,
            confidence=1.0,
            metadata={"reason": "no structure detected"},
        )
