"""``MemoryQuery``: multi-source, full-fidelity retrieval query.

Pre-this-PR, the retrieval query was "latest user message, truncated
to 500 chars" (memory_handler.py:807). The truncation was a real bug
— none of Letta / Mem0 / Cognee / Supermemory truncate the embedding
input. Tool outputs are often the strongest retrieval signal in
coding sessions, and they were ignored entirely.

This value type captures the query at full fidelity from three
sources:

  * ``user_text`` — latest user message, untruncated
  * ``recent_tool_outputs`` — last N tool results
  * ``recent_assistant_turns`` — last K assistant turns for intent

The embedding model handles its own context window (MiniLM 512 tok;
BGE-small 8K tok). Long inputs that exceed the model window become
the model's problem to mean-pool or chunk — they don't get
truncated upstream.
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

    All fields preserve full input fidelity — no truncation, no
    summarization. The caller assembles the sources; this type only
    holds them. The retrieval backend decides how to embed
    (mean-pool, chunk, model-side truncation, etc.) but cannot lose
    information before it sees the data.

    Tuples for the recent-* fields so the dataclass stays hashable
    (frozen + value-equal).
    """

    user_text: str
    recent_tool_outputs: tuple[str, ...]
    recent_assistant_turns: tuple[str, ...]
    conversation_id: str | None

    def to_embedding_input(self) -> str:
        """Concatenate sources into a delimited embedding input.

        Order: prior assistant turns (oldest first) → tool outputs
        (oldest first) → latest user text. User text last because the
        embedder's positional weighting often emphasizes the tail of
        the input.
        """
        return render_embedding_input(
            user_text=self.user_text,
            recent_tool_outputs=self.recent_tool_outputs,
            recent_assistant_turns=self.recent_assistant_turns,
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
