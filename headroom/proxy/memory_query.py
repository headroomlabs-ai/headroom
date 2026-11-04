"""``MemoryQuery``: multi-source, full-fidelity retrieval query.

Raw user messages and recent tool/assistant turns are retained for retrieval
strategies that need their full content. Time-limited proxy lookups can render
a separate bounded embedding input without mutating those sources.

This value type captures the query at full fidelity from three
sources:

  * ``user_text`` — latest user message, untruncated
  * ``recent_tool_outputs`` — last N tool results
  * ``recent_assistant_turns`` — last K assistant turns for intent

The default embedding representation preserves the full sources. A caller can
request a character budget when embedding latency must fit a request deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory_query_policy import (
    extract_memory_query_sources,
    render_embedding_input,
)


@dataclass(frozen=True)
class MemoryQuery:
    """Frozen multi-source query for memory retrieval.

    Raw fields preserve full input fidelity. Budgeted embedding rendering
    produces a derived query without changing those fields; backends that
    chunk or mean-pool full inputs can still use the original sources.

    Tuples for the recent-* fields so the dataclass stays hashable
    (frozen + value-equal).
    """

    user_text: str
    recent_tool_outputs: tuple[str, ...]
    recent_assistant_turns: tuple[str, ...]
    conversation_id: str | None

    def to_embedding_input(self, *, max_chars: int | None = None) -> str:
        """Render delimited embedding input, optionally within a character budget.

        By default, context is chronological and user text comes last.
        With a character budget, user text comes first and recent context is
        selected newest-first, then rendered chronologically so a tokenizer
        can retain its newest tail. Proxy lookups separately reserve space
        for both sources inside the active local model's token window.
        """
        return render_embedding_input(
            user_text=self.user_text,
            recent_tool_outputs=self.recent_tool_outputs,
            recent_assistant_turns=self.recent_assistant_turns,
            max_chars=max_chars,
        )

    @classmethod
    def from_messages(
        cls,
        messages: list[dict[str, Any]] | None,
        *,
        lookback_assistant: int = 2,
        lookback_tools: int = 3,
        conversation_id: str | None = None,
    ) -> MemoryQuery:
        """Construct a MemoryQuery from a chat-style messages list.

        Walks the message list once. Extracts:
          * Latest ``role: user`` message → ``user_text``
          * Up to ``lookback_assistant`` most recent assistant turns →
            ``recent_assistant_turns`` (chronological order)
          * Up to ``lookback_tools`` most recent tool outputs →
            ``recent_tool_outputs`` (chronological order)

        Handles both OpenAI shape (``role: tool``) and Anthropic shape
        (``tool_result`` content block inside a ``role: user`` message).
        """
        user_text, recent_tool_outputs, recent_assistant_turns = extract_memory_query_sources(
            messages,
            lookback_assistant=lookback_assistant,
            lookback_tools=lookback_tools,
        )
        return cls(
            user_text=user_text,
            recent_tool_outputs=recent_tool_outputs,
            recent_assistant_turns=recent_assistant_turns,
            conversation_id=conversation_id,
        )
