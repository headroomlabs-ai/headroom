"""Base provider protocol for Headroom SDK.

Providers are responsible for:
- Token counting (model-specific)
- Model context limits
- Cost estimation (optional)

This module defines the protocols that all providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count_text(self, text: str) -> int:
        """Count tokens in a text string."""
        ...

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message dict."""
        ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        ...


class Provider(ABC):
    """
    Abstract base class for LLM providers.

    Providers encapsulate all model-specific behavior:
    - Token counting
    - Context window limits
    - Cost estimation

    Implementations must be explicit - no silent fallbacks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    @abstractmethod
    def get_token_counter(self, model: str) -> TokenCounter:
        """
        Get a token counter for a specific model.

        Args:
            model: The model name.

        Returns:
            TokenCounter instance for the model.

        Raises:
            ValueError: If model is not supported by this provider.
        """
        ...

    @abstractmethod
    def get_context_limit(self, model: str) -> int:
        """
        Get the context window limit for a model.

        Args:
            model: The model name.

        Returns:
            Maximum context tokens for the model.

        Raises:
            ValueError: If model is not recognized.
        """
        ...

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """
        Check if this provider supports a given model.

        Args:
            model: The model name.

        Returns:
            True if the model is supported.
        """
        ...

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """
        Estimate API cost in USD.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            cached_tokens: Number of cached input tokens.

        Returns:
            Estimated cost in USD, or None if cost estimation not available.
        """
        return None

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        """
        Get recommended output buffer for a model.

        Some models (like reasoning models) produce longer outputs.

        Args:
            model: The model name.
            default: Default buffer if no model-specific recommendation.

        Returns:
            Recommended output token buffer.
        """
        return default
