"""Base backend interface for Headroom.

Backends translate between the canonical Anthropic Messages API format
(used by the proxy) and provider-specific APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendResponse:
    """Standardized response from a backend."""

    # Response body (Anthropic Messages API format)
    body: dict[str, Any]

    # HTTP status code
    status_code: int = 200

    # Response headers to forward
    headers: dict[str, str] = field(default_factory=dict)

    # Error message if any
    error: str | None = None


@dataclass
class StreamEvent:
    """A single event from a streaming response."""

    # Event type (message_start, content_block_delta, etc.)
    event_type: str

    # Event data (Anthropic SSE format)
    data: dict[str, Any]

    # Raw SSE line to forward (if available)
    raw_sse: str | None = None


class Backend(ABC):
    """Abstract base class for LLM API backends.

    Backends are responsible for:
    - Translating requests from Anthropic format to provider format
    - Making API calls to the provider
    - Translating responses back to Anthropic format
    - Handling streaming
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'anthropic', 'bedrock')."""
        ...

    @abstractmethod
    async def send_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> BackendResponse:
        """Send a non-streaming message request.

        Args:
            body: Request body in Anthropic Messages API format.
            headers: Request headers (may include API keys, etc.).

        Returns:
            BackendResponse with body in Anthropic Messages API format.
        """
        ...

    @abstractmethod
    def stream_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[StreamEvent]:
        """Stream a message request.

        Args:
            body: Request body in Anthropic Messages API format.
            headers: Request headers.

        Yields:
            StreamEvent objects in Anthropic SSE format.
        """
        ...

    @abstractmethod
    def map_model_id(self, anthropic_model: str) -> str:
        """Map Anthropic model ID to provider model ID.

        Args:
            anthropic_model: Model ID in Anthropic format (e.g., 'claude-3-opus-20240229').

        Returns:
            Model ID in provider format.
        """
        ...

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Check if this backend supports a model.

        Args:
            model: Model ID (can be Anthropic or provider format).

        Returns:
            True if model is supported.
        """
        ...

    async def send_openai_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> BackendResponse:
        """Send an OpenAI-format message request.

        Unlike send_message(), this takes OpenAI-format input and returns
        OpenAI-format output (no Anthropic conversion). Optional - only
        implemented by backends that support OpenAI-compatible APIs.

        Args:
            body: Request body in OpenAI chat completion format.
            headers: Request headers.

        Returns:
            BackendResponse with body in OpenAI chat completion format.

        Raises:
            NotImplementedError: If backend doesn't support OpenAI format.
        """
        raise NotImplementedError(f"{self.name} backend does not support OpenAI format")

    async def stream_openai_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[str]:
        """Stream an OpenAI-format chat completion.

        Yields SSE-formatted strings: 'data: {...}\\n\\n' for each chunk,
        ending with 'data: [DONE]\\n\\n'.

        Args:
            body: Request body in OpenAI chat completion format (stream: true).
            headers: Request headers.

        Yields:
            SSE-formatted strings ready to send to client.

        Raises:
            NotImplementedError: If backend doesn't support OpenAI streaming.
        """
        raise NotImplementedError(f"{self.name} backend does not support OpenAI streaming")
        # Make this an async generator (yield never reached but needed for type)
        yield ""  # type: ignore[misc]  # pragma: no cover

    async def close(self) -> None:  # noqa: B027
        """Clean up resources (e.g., close HTTP clients)."""
        pass
