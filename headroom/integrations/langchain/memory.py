"""Memory integration for LangChain with automatic compression.

This module provides HeadroomChatMessageHistory, a wrapper for any LangChain
chat message history that automatically compresses conversation history
when it exceeds a token threshold.

Example:
    from langchain.memory import ConversationBufferMemory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from headroom.integrations import HeadroomChatMessageHistory

    # Wrap any chat message history
    base_history = ChatMessageHistory()
    compressed_history = HeadroomChatMessageHistory(base_history)

    # Use with ConversationBufferMemory (zero code changes to chain)
    memory = ConversationBufferMemory(chat_memory=compressed_history)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from headroom.providers.base import Provider

# LangChain imports - these are optional dependencies
try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatMessageHistory = object  # type: ignore[misc,assignment]

from headroom import HeadroomConfig
from headroom.providers import OpenAIProvider
from headroom.transforms import TransformPipeline

logger = logging.getLogger(__name__)


def _check_langchain_available() -> None:
    """Raise ImportError if LangChain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install with: pip install headroom[langchain] "
            "or: pip install langchain-core"
        )


class HeadroomChatMessageHistory(BaseChatMessageHistory):
    """Wraps any LangChain chat message history with automatic compression.

    When conversation history exceeds the token threshold, automatically
    applies live-zone block compression (per-block content compression on
    the live zone, never dropping messages — that's what the live-zone
    refactor in PR-B1+ replaces the old RollingWindow strategy with).

    This works with ANY memory type because it wraps at the storage layer:
    - ConversationBufferMemory
    - ConversationSummaryMemory
    - ConversationBufferWindowMemory
    - Redis, PostgreSQL, or any custom history

    Example:
        from langchain.memory import ConversationBufferMemory
        from langchain_community.chat_message_histories import ChatMessageHistory
        from headroom.integrations import HeadroomChatMessageHistory

        # Wrap base history
        base = ChatMessageHistory()
        compressed = HeadroomChatMessageHistory(
            base,
            compress_threshold_tokens=4000,
            keep_recent_turns=5,
        )

        # Use with any memory class
        memory = ConversationBufferMemory(chat_memory=compressed)

        # Messages are compressed automatically when accessed
        chain = ConversationChain(llm=llm, memory=memory)
        chain.invoke({"input": "Hello!"})

    Attributes:
        base_history: The underlying chat message history
        compress_threshold_tokens: Token count that triggers compression
        keep_recent_turns: Minimum recent turns to always preserve
        model: Model name for token counting (default: "gpt-4o")
    """

    def __init__(
        self,
        base_history: BaseChatMessageHistory,
        compress_threshold_tokens: int = 4000,
        keep_recent_turns: int = 5,
        model: str = "gpt-4o",
        provider: Provider | None = None,
    ):
        """Initialize HeadroomChatMessageHistory.

        Args:
            base_history: Any LangChain BaseChatMessageHistory to wrap
            compress_threshold_tokens: Apply compression when history exceeds
                this many tokens. Default 4000.
            keep_recent_turns: Minimum number of recent user/assistant turns
                to always preserve during compression. Default 5.
            model: Model name for token counting. Default "gpt-4o".
            provider: Headroom provider for token counting. Auto-uses
                OpenAIProvider if not specified.
        """
        _check_langchain_available()

        self._base = base_history
        self._threshold = compress_threshold_tokens
        self._keep_recent_turns = keep_recent_turns
        self._model = model
        self._provider: Provider = provider or OpenAIProvider()

        # Track compression stats
        self._compression_count = 0
        self._total_tokens_saved = 0

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Get messages, applying compression if over threshold.

        Returns:
            List of messages, potentially compressed to fit within threshold.
        """
        raw_messages = self._base.messages

        if not raw_messages:
            return []

        # Count tokens
        token_count = self._count_tokens(raw_messages)

        if token_count <= self._threshold:
            return list(raw_messages)

        # Apply compression
        compressed = self._apply_compression(raw_messages)
        tokens_after = self._count_tokens(compressed)

        self._compression_count += 1
        self._total_tokens_saved += token_count - tokens_after

        logger.info(
            f"HeadroomChatMessageHistory compressed: {token_count} -> {tokens_after} tokens "
            f"({len(raw_messages)} -> {len(compressed)} messages)"
        )

        return compressed

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the underlying history.

        Args:
            message: The message to add.
        """
        self._base.add_message(message)

    def add_user_message(self, message: HumanMessage | str) -> None:
        """Add a user message to the history.

        Args:
            message: The user message (string or HumanMessage).
        """
        self._base.add_user_message(message)

    def add_ai_message(self, message: AIMessage | str) -> None:
        """Add an AI message to the history.

        Args:
            message: The AI message (string or AIMessage).
        """
        self._base.add_ai_message(message)

    def clear(self) -> None:
        """Clear all messages from history."""
        self._base.clear()

    def _count_tokens(self, messages: list[BaseMessage]) -> int:
        """Count tokens in messages using provider's tokenizer.

        Args:
            messages: List of messages to count.

        Returns:
            Total token count.
        """
        token_counter = self._provider.get_token_counter(self._model)
        total = 0
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += token_counter.count_text(content)
        return total

    def _apply_compression(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Apply live-zone-only compression to messages.

        After PR-B1, message-dropping is no longer a strategy — only
        per-block content compression. Result may still exceed the
        threshold; callers should treat the threshold as advisory.

        Args:
            messages: Messages to compress.

        Returns:
            Messages with their live-zone content compressed where
            applicable.
        """
        # Convert to OpenAI format for Headroom transforms
        openai_messages = self._convert_to_openai(messages)

        # Use TransformPipeline which handles tokenizer setup
        config = HeadroomConfig()
        pipeline = TransformPipeline(config=config, provider=self._provider)

        # Apply compression via pipeline
        result = pipeline.apply(
            messages=openai_messages,
            model=self._model,
            model_limit=self._threshold,
        )

        # Convert back to LangChain format
        return self._convert_from_openai(result.messages)

    def _convert_to_openai(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages to OpenAI format.

        Args:
            messages: LangChain messages.

        Returns:
            OpenAI format messages.
        """
        result = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                entry: dict[str, Any] = {"role": "assistant", "content": content}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(msg, "tool_call_id", ""),
                        "content": content,
                    }
                )
            else:
                # Generic fallback
                result.append(
                    {
                        "role": getattr(msg, "type", "user"),
                        "content": content,
                    }
                )
        return result

    def _convert_from_openai(self, messages: list[dict[str, Any]]) -> list[BaseMessage]:
        """Convert OpenAI format back to LangChain messages.

        Args:
            messages: OpenAI format messages.

        Returns:
            LangChain messages.
        """
        result: list[BaseMessage] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                result.append(AIMessage(content=content, tool_calls=tool_calls))
            elif role == "tool":
                result.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )
        return result

    def get_compression_stats(self) -> dict[str, Any]:
        """Get statistics about compression operations.

        Returns:
            Dictionary with compression_count, total_tokens_saved.
        """
        return {
            "compression_count": self._compression_count,
            "total_tokens_saved": self._total_tokens_saved,
            "threshold_tokens": self._threshold,
            "keep_recent_turns": self._keep_recent_turns,
        }
