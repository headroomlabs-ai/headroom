"""Agno model wrapper for Headroom optimization.

This module provides HeadroomAgnoModel, which wraps any Agno model
to apply Headroom context optimization before API calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

# Agno imports - these are optional dependencies
try:
    from agno.models.base import Model
    from agno.models.message import Message
    from agno.models.response import ModelResponse

    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    Model = object  # type: ignore[misc,assignment]
    Message = dict  # type: ignore[misc,assignment]
    ModelResponse = dict  # type: ignore[misc,assignment]

from headroom import HeadroomConfig, HeadroomMode
from headroom.providers import OpenAIProvider
from headroom.transforms import TransformPipeline

from .providers import get_headroom_provider, get_model_name_from_agno

logger = logging.getLogger(__name__)


def _check_agno_available() -> None:
    """Raise ImportError if Agno is not installed."""
    if not AGNO_AVAILABLE:
        raise ImportError("Agno is required for this integration. Install with: pip install agno")


def agno_available() -> bool:
    """Check if Agno is installed."""
    return AGNO_AVAILABLE


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


@dataclass
class HeadroomAgnoModel(Model):  # type: ignore[misc]
    """Agno model wrapper that applies Headroom optimizations.

    Extends agno.models.base.Model to be fully compatible with Agno Agent.
    Wraps any Agno Model and automatically optimizes the context
    before each API call. Works with OpenAIChat, Claude, Gemini, and
    other Agno model types.

    Important - Reasoning Modes:
        Claude's extended thinking and Agno's reasoning flow are INCOMPATIBLE.
        Choose ONE approach:

        1. Claude Extended Thinking (native):
           - Set thinking={"type": "enabled", "budget_tokens": N} on Claude model
           - Do NOT use reasoning=True on Agent
           - Claude handles reasoning internally with structured thinking blocks

        2. Agno Reasoning Flow (framework):
           - Do NOT set thinking config on the model
           - Use reasoning=True on Agent
           - Use underlying_model as reasoning_model for proper detection
           - Agno handles chain-of-thought with text-based <thinking> tags

    Example:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from headroom.integrations.agno import HeadroomAgnoModel

        # Basic usage
        model = OpenAIChat(id="gpt-4o")
        optimized = HeadroomAgnoModel(wrapped_model=model)

        # Use with agent
        agent = Agent(model=optimized)
        response = agent.run("Hello!")

        # Access metrics
        print(f"Saved {optimized.total_tokens_saved} tokens")

        # With custom config
        from headroom import HeadroomConfig, HeadroomMode
        config = HeadroomConfig(default_mode=HeadroomMode.OPTIMIZE)
        optimized = HeadroomAgnoModel(wrapped_model=model, headroom_config=config)

        # Agno reasoning with HeadroomAgnoModel
        model = OpenAIChat(id="gpt-4o")
        wrapped = HeadroomAgnoModel(wrapped_model=model)
        agent = Agent(
            model=wrapped,
            reasoning=True,
            reasoning_model=wrapped.underlying_model,  # Use underlying for detection
        )

    Attributes:
        wrapped_model: The underlying Agno model
        underlying_model: Same as wrapped_model, for framework introspection
        total_tokens_saved: Running total of tokens saved
        metrics_history: List of OptimizationMetrics from recent calls
    """

    # Required by Model base class - we'll derive from wrapped model
    id: str = field(default="headroom-wrapper")
    name: str | None = field(default=None)
    provider: str | None = field(default=None)

    # HeadroomAgnoModel specific fields
    wrapped_model: Any = field(default=None)
    headroom_config: HeadroomConfig | None = field(default=None)
    headroom_mode: HeadroomMode | None = field(default=None)
    auto_detect_provider: bool = field(default=True)

    # Internal state (not part of dataclass comparison)
    _metrics_history: list[OptimizationMetrics] = field(
        default_factory=list, repr=False, compare=False
    )
    _total_tokens_saved: int = field(default=0, repr=False, compare=False)
    _pipeline: TransformPipeline | None = field(default=None, repr=False, compare=False)
    _headroom_provider: Any = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _initialized: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize HeadroomAgnoModel after dataclass construction."""
        _check_agno_available()

        if self.wrapped_model is None:
            raise ValueError("wrapped_model cannot be None")

        # Set id from wrapped model
        if hasattr(self.wrapped_model, "id"):
            self.id = f"headroom:{self.wrapped_model.id}"

        # Set name and provider from wrapped model for compatibility
        if self.name is None and hasattr(self.wrapped_model, "name"):
            self.name = self.wrapped_model.name
        if self.provider is None and hasattr(self.wrapped_model, "provider"):
            self.provider = self.wrapped_model.provider

        # Forward capability attributes from wrapped model
        # These are critical for framework introspection (e.g., Agno reasoning detection)
        self._forward_capability_attributes()

        # Initialize config
        if self.headroom_config is None:
            self.headroom_config = HeadroomConfig()

        # Handle deprecated mode parameter
        if self.headroom_mode is not None:
            warnings.warn(
                "The 'headroom_mode' parameter is deprecated. Use HeadroomConfig(default_mode=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._initialized = True

        # Call parent __post_init__ if it exists
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def _forward_capability_attributes(self) -> None:
        """Forward capability attributes from wrapped model.

        This ensures that framework introspection (like Agno's reasoning detection)
        can access capabilities like 'thinking', 'reasoning_effort', etc.
        through the wrapper.
        """
        # Reasoning-related attributes
        capability_attrs = [
            "thinking",  # Claude extended thinking
            "reasoning_effort",  # OpenAI reasoning effort
            "supports_native_structured_outputs",  # Structured output support
            "supports_json_schema_outputs",  # JSON schema support
        ]

        for attr in capability_attrs:
            if hasattr(self.wrapped_model, attr):
                value = getattr(self.wrapped_model, attr)
                # Use object.__setattr__ to bypass any dataclass restrictions
                object.__setattr__(self, attr, value)

    def has_extended_thinking_enabled(self) -> bool:
        """Check if the wrapped model has extended thinking enabled.

        Extended thinking is a Claude-specific feature that uses structured
        content blocks. It is INCOMPATIBLE with Agno's reasoning flow.

        Returns:
            True if extended thinking is configured on the wrapped model.
        """
        thinking = getattr(self.wrapped_model, "thinking", None)
        if thinking is None:
            return False
        if isinstance(thinking, dict):
            return thinking.get("type") == "enabled"
        return bool(thinking)

    # Forward attribute access to wrapped model for compatibility
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        # Avoid infinite recursion during initialization
        if name.startswith("_") or not self.__dict__.get("_initialized", False):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name in (
            "wrapped_model",
            "headroom_config",
            "headroom_mode",
            "auto_detect_provider",
            "pipeline",
            "total_tokens_saved",
            "metrics_history",
            "id",
            "name",
            "provider",
            "underlying_model",
        ):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self.wrapped_model, name)

    # =========================================================================
    # Capability property: underlying_model for type introspection
    # =========================================================================

    @property
    def underlying_model(self) -> Any:
        """Return the underlying model for type introspection.

        Frameworks like Agno that need to check model capabilities
        (e.g., native reasoning detection via __class__.__name__) can use
        this to access the actual model class.

        Example:
            wrapped = HeadroomAgnoModel(wrapped_model=Claude(...))
            actual_class = wrapped.underlying_model.__class__.__name__  # "Claude"
        """
        return self.wrapped_model

    # =========================================================================
    # Pipeline and metrics
    # =========================================================================

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
        """Convert Agno messages to OpenAI format for Headroom.

        Preserves extended thinking content blocks (thinking, redacted_thinking)
        which must be passed through unchanged for Claude's extended thinking API.
        """
        result = []
        for msg in messages:
            # Handle Agno Message objects
            if hasattr(msg, "role") and hasattr(msg, "content"):
                entry: dict[str, Any] = {
                    "role": msg.role,
                }

                # Handle content - can be string or list of content blocks
                content = msg.content
                if content is None:
                    entry["content"] = ""
                elif isinstance(content, list):
                    # Content blocks (e.g., extended thinking)
                    # Preserve the structure - don't stringify
                    entry["content"] = content
                else:
                    entry["content"] = content

                # Handle tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                # Handle tool call ID for tool responses
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id

                # Preserve reasoning content for extended thinking
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    entry["reasoning_content"] = msg.reasoning_content
                if hasattr(msg, "redacted_reasoning_content") and msg.redacted_reasoning_content:
                    entry["redacted_reasoning_content"] = msg.redacted_reasoning_content

                # Preserve provider_data which may contain thinking signatures
                if hasattr(msg, "provider_data") and msg.provider_data:
                    entry["provider_data"] = msg.provider_data

                result.append(entry)
            # Handle dict format
            elif isinstance(msg, dict):
                result.append(msg.copy())
            else:
                # Try to extract content
                content = str(msg) if msg is not None else ""
                result.append({"role": "user", "content": content})
        return result

    def _ensure_message_objects(self, messages: list[Any]) -> list[Any]:
        """Ensure all messages are Agno Message objects (not dicts).

        Agno's base Model methods call _log_messages() which requires
        Message objects with a .log() method.

        Preserves extended thinking fields (reasoning_content, provider_data, etc.)
        which are critical for Claude's extended thinking API.

        Args:
            messages: List of messages (may be dicts or Message objects)

        Returns:
            List of Agno Message objects
        """
        from agno.models.message import Message as AgnoMessage

        result = []
        for msg in messages:
            if isinstance(msg, dict):
                # Convert dict to Agno Message
                try:
                    result.append(AgnoMessage.from_dict(msg))
                except Exception:
                    # If from_dict fails, create a Message with all relevant fields
                    result.append(
                        AgnoMessage(
                            role=msg.get("role", "user"),
                            content=msg.get("content"),
                            tool_calls=msg.get("tool_calls"),
                            tool_call_id=msg.get("tool_call_id"),
                            # Extended thinking fields
                            reasoning_content=msg.get("reasoning_content"),
                            redacted_reasoning_content=msg.get("redacted_reasoning_content"),
                            provider_data=msg.get("provider_data"),
                        )
                    )
            else:
                # Already a Message object, keep as-is
                result.append(msg)
        return result

    def _convert_messages_from_openai(
        self, messages: list[dict[str, Any]], original_messages: list[Any]
    ) -> list[Any]:
        """Convert OpenAI format messages back to Agno Message objects.

        The Agno base model's response() method expects Message objects,
        not dicts, because it calls .log() on them internally.

        Args:
            messages: The optimized messages in OpenAI dict format
            original_messages: The original Agno Message objects (for reference)

        Returns:
            List of Agno Message objects
        """
        # Reuse the ensure method which handles the conversion
        return self._ensure_message_objects(messages)

    def _has_thinking_blocks(self, messages: list[dict[str, Any]]) -> bool:
        """Check if any message contains extended thinking content blocks.

        Extended thinking blocks (thinking, redacted_thinking) must not be
        modified and require special handling by Claude's API.

        Args:
            messages: List of messages in dict format

        Returns:
            True if any message contains thinking blocks
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type in ("thinking", "redacted_thinking"):
                            return True
            # Also check for reasoning_content which indicates thinking was used
            if msg.get("reasoning_content") or msg.get("redacted_reasoning_content"):
                return True
        return False

    def _optimize_messages(self, messages: list[Any]) -> tuple[list[Any], OptimizationMetrics]:
        """Apply Headroom optimization to messages.

        Thread-safe with fallback on pipeline errors.
        Skips optimization for messages with extended thinking blocks to preserve
        Claude's thinking content structure.
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
                model=get_model_name_from_agno(self.wrapped_model),
            )
            return openai_messages, metrics

        # Get model name from wrapped model
        model = get_model_name_from_agno(self.wrapped_model)

        # Skip optimization for messages with extended thinking blocks
        # Thinking blocks must be passed through unchanged for Claude's API
        if self._has_thinking_blocks(openai_messages):
            logger.info("Skipping Headroom optimization: messages contain extended thinking blocks")
            # Estimate token count (rough approximation)
            tokens_estimate = sum(len(str(m.get("content", ""))) // 4 for m in openai_messages)
            metrics = OptimizationMetrics(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                tokens_before=tokens_estimate,
                tokens_after=tokens_estimate,
                tokens_saved=0,
                savings_percent=0,
                transforms_applied=["skipped:extended_thinking"],
                model=model,
            )
            # Convert back to Agno Message objects
            result_messages = self._convert_messages_from_openai(openai_messages, messages)
            return result_messages, metrics

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
            # Log at warning level (degraded behavior, not critical failure)
            logger.warning(
                f"Headroom optimization failed, using original messages: {type(e).__name__}: {e}"
            )
            optimized = openai_messages
            # Estimate token count for unoptimized messages (rough approximation)
            # Note: This uses ~4 chars/token which is approximate for English text
            tokens_before = sum(len(str(m.get("content", ""))) // 4 for m in openai_messages)
            tokens_after = tokens_before  # No optimization occurred
            transforms_applied = ["fallback:error"]

        # Create metrics
        tokens_saved = max(0, tokens_before - tokens_after)  # Never negative
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

        # Convert back to Agno Message objects (required for base model's .log() calls)
        optimized_messages = self._convert_messages_from_openai(optimized, messages)

        return optimized_messages, metrics

    def response(self, messages: list[Any], **kwargs: Any) -> Any:  # type: ignore[override]
        """Generate response with Headroom optimization.

        This method lets the inherited Model.response() handle the tool loop,
        which will call self.invoke() for each API call. Our invoke() override
        applies Headroom optimization before delegating to wrapped_model.invoke().

        This ensures tool outputs are compressed on subsequent API calls.
        """
        # Ensure messages are Message objects (Agno's _log_messages requires .log() method)
        messages = self._ensure_message_objects(messages)
        # Let the tool loop in Model.response() call invoke(),
        # which will optimize messages for EACH API call (including tool results)
        return super().response(messages, **kwargs)

    def response_stream(self, messages: list[Any], **kwargs: Any) -> Iterator[Any]:  # type: ignore[override]
        """Stream response with Headroom optimization.

        Like response(), delegates to inherited Model.response_stream() which
        calls self.invoke_stream() for each API call.
        """
        # Ensure messages are Message objects (Agno's _log_messages requires .log() method)
        messages = self._ensure_message_objects(messages)
        # Let the inherited streaming method handle the tool loop
        yield from super().response_stream(messages, **kwargs)

    async def aresponse(self, messages: list[Any], **kwargs: Any) -> Any:  # type: ignore[override]
        """Async generate response with Headroom optimization.

        Delegates to inherited Model.aresponse() which calls self.ainvoke()
        for each API call, ensuring tool outputs are optimized.
        """
        # Ensure messages are Message objects (Agno's _log_messages requires .log() method)
        messages = self._ensure_message_objects(messages)
        # Let the inherited async method handle the tool loop
        return await super().aresponse(messages, **kwargs)

    async def aresponse_stream(self, messages: list[Any], **kwargs: Any) -> AsyncIterator[Any]:  # type: ignore[override]
        """Async stream response with Headroom optimization.

        Delegates to inherited Model.aresponse_stream() which calls self.ainvoke_stream()
        for each API call, ensuring tool outputs are optimized.
        """
        # Ensure messages are Message objects (Agno's _log_messages requires .log() method)
        messages = self._ensure_message_objects(messages)
        # Let the inherited async streaming method handle the tool loop
        async for chunk in super().aresponse_stream(messages, **kwargs):
            yield chunk

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings."""
        if not self._metrics_history:
            return {
                "total_requests": 0,
                "total_tokens_saved": 0,
                "average_savings_percent": 0,
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
    # Abstract method implementations required by agno.models.base.Model
    # These delegate to the wrapped model after applying Headroom optimization
    # =========================================================================

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke the wrapped model with optimized messages.

        This is required by agno.models.base.Model abstract interface.
        """
        # Optimize messages before invoking
        optimized_messages, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized (invoke): {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        # Delegate to wrapped model
        return self.wrapped_model.invoke(optimized_messages, **kwargs)

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Async invoke the wrapped model with optimized messages.

        This is required by agno.models.base.Model abstract interface.
        """
        # Run optimization in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        optimized_messages, metrics = await loop.run_in_executor(
            None, self._optimize_messages, messages
        )

        logger.info(
            f"Headroom optimized (ainvoke): {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        # Delegate to wrapped model
        if hasattr(self.wrapped_model, "ainvoke"):
            return await self.wrapped_model.ainvoke(optimized_messages, **kwargs)
        else:
            # Fallback to sync in executor
            return await loop.run_in_executor(
                None, lambda: self.wrapped_model.invoke(optimized_messages, **kwargs)
            )

    def invoke_stream(self, messages: list[Any], **kwargs: Any) -> Iterator[Any]:
        """Stream invoke the wrapped model with optimized messages.

        This is required by agno.models.base.Model abstract interface.
        """
        # Optimize messages before streaming
        optimized_messages, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized (invoke_stream): {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        # Delegate to wrapped model
        yield from self.wrapped_model.invoke_stream(optimized_messages, **kwargs)

    async def ainvoke_stream(self, messages: list[Any], **kwargs: Any) -> AsyncIterator[Any]:
        """Async stream invoke the wrapped model with optimized messages.

        This is required by agno.models.base.Model abstract interface.
        """
        # Run optimization in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        optimized_messages, metrics = await loop.run_in_executor(
            None, self._optimize_messages, messages
        )

        logger.info(
            f"Headroom optimized (ainvoke_stream): {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        # Delegate to wrapped model
        if hasattr(self.wrapped_model, "ainvoke_stream"):
            async for chunk in self.wrapped_model.ainvoke_stream(optimized_messages, **kwargs):
                yield chunk
        else:
            # Fallback: wrap sync streaming
            def _sync_stream() -> list[Any]:
                return list(self.wrapped_model.invoke_stream(optimized_messages, **kwargs))

            chunks = await loop.run_in_executor(None, _sync_stream)
            for chunk in chunks:
                yield chunk

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> Any:
        """Parse provider response - delegates to wrapped model.

        This is required by agno.models.base.Model abstract interface.
        """
        return self.wrapped_model._parse_provider_response(response, **kwargs)

    def _parse_provider_response_delta(self, response: Any) -> Any:
        """Parse streaming response delta - delegates to wrapped model.

        This is required by agno.models.base.Model abstract interface.
        """
        return self.wrapped_model._parse_provider_response_delta(response)


def optimize_messages(
    messages: list[Any],
    config: HeadroomConfig | None = None,
    mode: HeadroomMode = HeadroomMode.OPTIMIZE,
    model: str = "gpt-4o",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Standalone function to optimize Agno messages.

    Use this for manual optimization when you need fine-grained control.

    Args:
        messages: List of Agno Message objects or dicts
        config: HeadroomConfig for optimization settings
        mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
        model: Model name for token estimation

    Returns:
        Tuple of (optimized_messages, metrics_dict)

    Example:
        from headroom.integrations.agno import optimize_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        optimized, metrics = optimize_messages(messages)
        print(f"Saved {metrics['tokens_saved']} tokens")
    """
    _check_agno_available()

    config = config or HeadroomConfig()
    provider = OpenAIProvider()
    pipeline = TransformPipeline(config=config, provider=provider)

    # Convert to OpenAI format
    openai_messages = []
    for msg in messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            openai_messages.append(entry)
        elif isinstance(msg, dict):
            openai_messages.append(msg.copy())
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
