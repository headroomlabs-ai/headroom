"""Strands SDK model wrapper for Headroom optimization.

This module provides HeadroomStrandsModel, which wraps any Strands model
to apply Headroom context optimization before API calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

# Strands imports - these are optional dependencies
try:
    from strands.models import Model
    from strands.types.content import Message, Messages, SystemContentBlock
    from strands.types.streaming import StreamEvent
    from strands.types.tools import ToolChoice, ToolSpec

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    Model = object  # type: ignore[misc,assignment]
    Message = dict  # type: ignore[misc,assignment]
    Messages = list  # type: ignore[misc,assignment]
    StreamEvent = dict  # type: ignore[misc,assignment]
    ToolChoice = dict  # type: ignore[misc,assignment]
    ToolSpec = dict  # type: ignore[misc,assignment]
    SystemContentBlock = dict  # type: ignore[misc,assignment]

T = TypeVar("T")

from headroom import HeadroomConfig  # noqa: E402
from headroom.providers import OpenAIProvider  # noqa: E402
from headroom.transforms import TransformPipeline  # noqa: E402

from .providers import get_headroom_provider, get_model_name_from_strands  # noqa: E402

logger = logging.getLogger(__name__)


def _check_strands_available() -> None:
    """Raise ImportError if Strands SDK is not installed."""
    if not STRANDS_AVAILABLE:
        raise ImportError(
            "Strands SDK is required for this integration. Install with: pip install strands-agents"
        )


def strands_available() -> bool:
    """Check if Strands SDK is installed."""
    return STRANDS_AVAILABLE


@dataclass
class OptimizationMetrics:
    """Metrics from a single optimization pass."""

    request_id: str
    timestamp: datetime
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    transforms_applied: list[str]
    model: str


class HeadroomStrandsModel(Model):  # type: ignore[misc]
    """Strands model wrapper that applies Headroom optimizations.

    Wraps any Strands Model and automatically optimizes the context
    before each API call. Works with any Strands-compatible model provider.

    Example:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from headroom.integrations.strands import HeadroomStrandsModel

        # Basic usage
        model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
        optimized = HeadroomStrandsModel(wrapped_model=model)

        # Use with agent
        agent = Agent(model=optimized)
        response = agent("Hello!")

        # Access metrics
        print(f"Saved {optimized.total_tokens_saved} tokens")

        # With custom config
        from headroom import HeadroomConfig
        config = HeadroomConfig()
        optimized = HeadroomStrandsModel(wrapped_model=model, config=config)

    Attributes:
        wrapped_model: The underlying Strands model
        total_tokens_saved: Running total of tokens saved
        metrics_history: List of OptimizationMetrics from recent calls
    """

    def __init__(
        self,
        wrapped_model: Any,
        config: HeadroomConfig | None = None,
        auto_detect_provider: bool = True,
    ) -> None:
        """Initialize HeadroomStrandsModel.

        Args:
            wrapped_model: The Strands model to wrap (e.g., BedrockModel, OpenAIModel)
            config: Optional HeadroomConfig for optimization settings
            auto_detect_provider: Whether to auto-detect the Headroom provider
                based on the wrapped model type. Default True.
        """
        _check_strands_available()

        if wrapped_model is None:
            raise ValueError("wrapped_model cannot be None")

        self.wrapped_model = wrapped_model
        self.headroom_config = config or HeadroomConfig()
        self.auto_detect_provider = auto_detect_provider

        # Internal state
        self._metrics_history: list[OptimizationMetrics] = []
        self._total_tokens_saved: int = 0
        self._pipeline: TransformPipeline | None = None
        self._headroom_provider: Any = None
        self._lock = threading.Lock()

    @property
    def config(self) -> Any:
        """Forward config access to wrapped model (required by Strands Agent)."""
        return self.wrapped_model.config

    @property
    def pipeline(self) -> TransformPipeline:
        """Lazily initialize TransformPipeline (thread-safe)."""
        if self._pipeline is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._pipeline is None:
                    if self.auto_detect_provider:
                        self._headroom_provider = get_headroom_provider(self.wrapped_model)
                        logger.debug(
                            f"Auto-detected provider: {self._headroom_provider.__class__.__name__}"
                        )
                    else:
                        self._headroom_provider = OpenAIProvider()
                    self._pipeline = TransformPipeline(
                        config=self.headroom_config,
                        provider=self._headroom_provider,
                    )
        return self._pipeline

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all calls."""
        return self._total_tokens_saved

    @property
    def metrics_history(self) -> list[OptimizationMetrics]:
        """History of optimization metrics."""
        return self._metrics_history.copy()

    def _convert_messages_to_openai(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert Strands messages to OpenAI format for Headroom.

        Strands uses dict-based messages similar to OpenAI format:
        - {"role": "user", "content": "..."}
        - {"role": "assistant", "content": "...", "tool_calls": [...]}
        - {"role": "tool", "content": "...", "tool_call_id": "..."}

        Args:
            messages: List of Strands messages (typically dicts or Message objects)

        Returns:
            List of messages in OpenAI dict format
        """
        result = []
        for msg in messages:
            # Handle dict format (most common in Strands)
            if isinstance(msg, dict):
                entry: dict[str, Any] = {
                    "role": msg.get("role", "user"),
                }

                # Handle content
                content = msg.get("content")
                if content is None:
                    entry["content"] = ""
                elif isinstance(content, list):
                    # Content blocks - preserve structure
                    entry["content"] = content
                else:
                    entry["content"] = content

                # Handle tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    entry["tool_calls"] = msg["tool_calls"]

                # Handle tool call ID for tool responses
                if "tool_call_id" in msg and msg["tool_call_id"]:
                    entry["tool_call_id"] = msg["tool_call_id"]

                # Handle name field (for tool messages)
                if "name" in msg and msg["name"]:
                    entry["name"] = msg["name"]

                result.append(entry)

            # Handle Strands Message objects (if they have role/content attrs)
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                entry = {
                    "role": msg.role,
                }

                content = msg.content
                if content is None:
                    entry["content"] = ""
                elif isinstance(content, list):
                    entry["content"] = content
                else:
                    entry["content"] = content

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id
                if hasattr(msg, "name") and msg.name:
                    entry["name"] = msg.name

                result.append(entry)

            else:
                # Fallback: convert to string
                content = str(msg) if msg is not None else ""
                result.append({"role": "user", "content": content})

        return result

    def _convert_messages_from_openai(
        self, messages: list[dict[str, Any]], original_messages: list[Any]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI format messages back to Strands format.

        Since Strands uses dict-based messages similar to OpenAI,
        this is largely a passthrough, but ensures proper structure.

        Args:
            messages: The optimized messages in OpenAI dict format
            original_messages: The original Strands messages (for reference)

        Returns:
            List of messages in Strands dict format
        """
        result = []
        for msg in messages:
            entry: dict[str, Any] = {
                "role": msg.get("role", "user"),
            }

            # Handle content
            content = msg.get("content")
            if content is not None:
                entry["content"] = content

            # Preserve tool-related fields
            if "tool_calls" in msg and msg["tool_calls"]:
                entry["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg and msg["tool_call_id"]:
                entry["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg and msg["name"]:
                entry["name"] = msg["name"]

            result.append(entry)

        return result

    def _optimize_messages(
        self, messages: list[Any]
    ) -> tuple[list[dict[str, Any]], OptimizationMetrics]:
        """Apply Headroom optimization to messages.

        Thread-safe with fallback on pipeline errors.

        Args:
            messages: List of Strands messages to optimize

        Returns:
            Tuple of (optimized_messages, metrics)
        """
        request_id = str(uuid4())

        # Convert to OpenAI format
        openai_messages = self._convert_messages_to_openai(messages)

        # Handle empty messages gracefully
        if not openai_messages:
            metrics = OptimizationMetrics(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                tokens_before=0,
                tokens_after=0,
                tokens_saved=0,
                savings_percent=0,
                transforms_applied=[],
                model=get_model_name_from_strands(self.wrapped_model),
            )
            return [], metrics

        # Get model name from wrapped model
        model = get_model_name_from_strands(self.wrapped_model)

        # Ensure pipeline is initialized
        _ = self.pipeline

        # Get model context limit
        model_limit = (
            self._headroom_provider.get_context_limit(model) if self._headroom_provider else 128000
        )

        try:
            # Apply Headroom transforms via pipeline
            result = self.pipeline.apply(
                messages=openai_messages,
                model=model,
                model_limit=model_limit,
            )
            optimized = result.messages
            tokens_before = result.tokens_before
            tokens_after = result.tokens_after
            transforms_applied = result.transforms_applied
        except (
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            KeyError,
            IndexError,
            ImportError,
            OSError,
        ) as e:
            # Fallback to original messages on pipeline error
            logger.warning(
                f"Headroom optimization failed, using original messages: {type(e).__name__}: {e}"
            )
            optimized = openai_messages
            # Estimate token count (rough approximation: ~4 chars/token)
            tokens_before = sum(len(str(m.get("content", ""))) // 4 for m in openai_messages)
            tokens_after = tokens_before
            transforms_applied = ["fallback:error"]

        # Create metrics
        tokens_saved = max(0, tokens_before - tokens_after)
        metrics = OptimizationMetrics(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            savings_percent=(tokens_saved / tokens_before * 100 if tokens_before > 0 else 0),
            transforms_applied=transforms_applied,
            model=model,
        )

        # Track metrics (thread-safe)
        with self._lock:
            self._metrics_history.append(metrics)
            self._total_tokens_saved += metrics.tokens_saved

            # Keep only last 100 metrics
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]

        # Convert back to Strands format
        optimized_messages = self._convert_messages_from_openai(optimized, messages)

        return optimized_messages, metrics

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream response with Headroom optimization.

        This is the main method required by Strands Model interface.
        Optimizes messages before delegating to the wrapped model's stream method.

        Args:
            messages: List of messages to send to the model
            tool_specs: Optional list of tool specifications
            system_prompt: Optional system prompt string
            tool_choice: Optional tool choice configuration
            system_prompt_content: Optional list of system content blocks
            invocation_state: Optional invocation state dictionary
            **kwargs: Additional arguments passed to the wrapped model

        Yields:
            Streaming events from the wrapped model
        """
        # Run optimization in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        optimized_messages, metrics = await loop.run_in_executor(
            None, self._optimize_messages, messages
        )

        logger.info(
            f"Headroom optimized (stream): {metrics.tokens_before} -> "
            f"{metrics.tokens_after} tokens ({metrics.savings_percent:.1f}% saved)"
        )

        # Delegate to wrapped model's stream method with all parameters
        async for event in self.wrapped_model.stream(
            optimized_messages,
            tool_specs=tool_specs,
            system_prompt=system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            invocation_state=invocation_state,
            **kwargs,
        ):
            yield event

    def get_config(self) -> Any:
        """Get the configuration of the wrapped model.

        Returns:
            The model configuration from the wrapped model.
        """
        return self.wrapped_model.get_config()

    def update_config(self, **model_config: Any) -> None:
        """Update the configuration of the wrapped model.

        Args:
            **model_config: Configuration options to update on the wrapped model.
        """
        self.wrapped_model.update_config(**model_config)

    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Generate structured output with Headroom optimization.

        Optimizes the prompt messages before delegating to the wrapped model's
        structured_output method.

        Args:
            output_model: The type/schema for the structured output
            prompt: List of prompt messages
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to the wrapped model

        Yields:
            Structured output events from the wrapped model
        """
        # Run optimization in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        optimized_prompt, metrics = await loop.run_in_executor(
            None, self._optimize_messages, prompt
        )

        logger.info(
            f"Headroom optimized (structured_output): {metrics.tokens_before} -> "
            f"{metrics.tokens_after} tokens ({metrics.savings_percent:.1f}% saved)"
        )

        # Delegate to wrapped model
        async for event in self.wrapped_model.structured_output(
            output_model, optimized_prompt, system_prompt=system_prompt, **kwargs
        ):
            yield event

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings."""
        if not self._metrics_history:
            return {
                "total_requests": 0,
                "total_tokens_saved": 0,
                "average_savings_percent": 0,
                "total_tokens_before": 0,
                "total_tokens_after": 0,
            }

        return {
            "total_requests": len(self._metrics_history),
            "total_tokens_saved": self._total_tokens_saved,
            "average_savings_percent": sum(m.savings_percent for m in self._metrics_history)
            / len(self._metrics_history),
            "total_tokens_before": sum(m.tokens_before for m in self._metrics_history),
            "total_tokens_after": sum(m.tokens_after for m in self._metrics_history),
        }

    def reset(self) -> None:
        """Reset all tracked metrics (thread-safe).

        Clears the metrics history and resets the total tokens saved counter.
        Useful for starting fresh measurements or between test runs.
        """
        with self._lock:
            self._metrics_history = []
            self._total_tokens_saved = 0

    # =========================================================================
    # Forward attribute access to wrapped model for compatibility
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        # Avoid infinite recursion for our own attributes
        if name in (
            "wrapped_model",
            "config",
            "auto_detect_provider",
            "_metrics_history",
            "_total_tokens_saved",
            "_pipeline",
            "_headroom_provider",
            "_lock",
            "pipeline",
            "total_tokens_saved",
            "metrics_history",
        ):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self.wrapped_model, name)


def optimize_messages(
    messages: list[Any],
    config: HeadroomConfig | None = None,
    model: str = "gpt-4o",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Standalone function to optimize Strands messages.

    Use this for manual optimization when you need fine-grained control.

    Args:
        messages: List of Strands messages (dicts)
        config: HeadroomConfig for optimization settings
        model: Model name for token estimation

    Returns:
        Tuple of (optimized_messages, metrics_dict)

    Example:
        from headroom.integrations.strands import optimize_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        optimized, metrics = optimize_messages(messages)
        print(f"Saved {metrics['tokens_saved']} tokens")
    """
    _check_strands_available()

    config = config or HeadroomConfig()
    provider = OpenAIProvider()
    pipeline = TransformPipeline(config=config, provider=provider)

    # Convert to OpenAI format (Strands uses similar format)
    openai_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            entry: dict[str, Any] = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            if "tool_calls" in msg and msg["tool_calls"]:
                entry["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg and msg["tool_call_id"]:
                entry["tool_call_id"] = msg["tool_call_id"]
            openai_messages.append(entry)
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            entry = {"role": msg.role, "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            openai_messages.append(entry)
        else:
            openai_messages.append({"role": "user", "content": str(msg)})

    # Get model context limit
    model_limit = provider.get_context_limit(model)

    # Apply transforms
    result = pipeline.apply(
        messages=openai_messages,
        model=model,
        model_limit=model_limit,
    )

    metrics = {
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_before - result.tokens_after,
        "savings_percent": (
            (result.tokens_before - result.tokens_after) / result.tokens_before * 100
            if result.tokens_before > 0
            else 0
        ),
        "transforms_applied": result.transforms_applied,
    }

    return result.messages, metrics
