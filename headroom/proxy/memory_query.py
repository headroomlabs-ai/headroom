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

# Section delimiters surfaced in the embedding input so the embedder
# sees structured context rather than a wall of run-on text. Kept short
# so they don't dominate the embedding signal.
_USER_DELIM = "### USER ###\n"
_ASSISTANT_DELIM = "\n### PRIOR_ASSISTANT ###\n"
_TOOL_DELIM = "\n### TOOL_OUTPUT ###\n"


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
        parts: list[str] = []
        for asst in self.recent_assistant_turns:
            if asst:
                parts.append(_ASSISTANT_DELIM + asst)
        for tool_out in self.recent_tool_outputs:
            if tool_out:
                parts.append(_TOOL_DELIM + tool_out)
        if self.user_text:
            parts.append(_USER_DELIM + self.user_text)
        return "".join(parts)

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
        if not messages:
            return cls(
                user_text="",
                recent_tool_outputs=(),
                recent_assistant_turns=(),
                conversation_id=conversation_id,
            )

        latest_user = ""
        assistant_turns: list[str] = []
        tool_outputs: list[str] = []

        # Walk messages backward so we naturally find the LATEST entries
        # first; preserve chronological order in the output by reversing
        # the collected lists at the end.
        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                # Distinguish "real user text" from "Anthropic tool_result
                # masquerading as a user message". Anthropic uses
                # role=user with content=[{type: tool_result, ...}].
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_text = block.get("content", "")
                            if isinstance(tool_text, list):
                                # Nested content blocks; flatten text fields.
                                tool_text = "\n".join(
                                    b.get("text", "") for b in tool_text if isinstance(b, dict)
                                )
                            if tool_text and len(tool_outputs) < lookback_tools:
                                tool_outputs.append(str(tool_text))
                    # Anthropic tool_result is NOT a real user turn —
                    # don't use it as the user_text source. Continue
                    # walking back for the actual user message.
                elif isinstance(content, str):
                    if not latest_user:
                        latest_user = content

            elif role == "assistant":
                if isinstance(content, str) and content:
                    if len(assistant_turns) < lookback_assistant:
                        assistant_turns.append(content)
                elif isinstance(content, list):
                    text_parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    joined = "\n".join(p for p in text_parts if p)
                    if joined and len(assistant_turns) < lookback_assistant:
                        assistant_turns.append(joined)

            elif role == "tool":
                # OpenAI shape: tool messages carry the result.
                if isinstance(content, str) and content and len(tool_outputs) < lookback_tools:
                    tool_outputs.append(content)

        # Reverse to restore chronological order (we walked backward).
        return cls(
            user_text=latest_user,
            recent_tool_outputs=tuple(reversed(tool_outputs)),
            recent_assistant_turns=tuple(reversed(assistant_turns)),
            conversation_id=conversation_id,
        )
