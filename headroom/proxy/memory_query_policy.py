"""Pure policy helpers for building memory retrieval queries."""

from __future__ import annotations

from typing import Any

# Section delimiters surfaced in the embedding input so the embedder
# sees structured context rather than a wall of run-on text. Kept short
# so they don't dominate the embedding signal.
USER_DELIM = "### USER ###\n"
ASSISTANT_DELIM = "\n### PRIOR_ASSISTANT ###\n"
TOOL_DELIM = "\n### TOOL_OUTPUT ###\n"


def render_embedding_input(
    *,
    user_text: str,
    recent_tool_outputs: tuple[str, ...],
    recent_assistant_turns: tuple[str, ...],
    max_chars: int | None = None,
) -> str:
    """Render full sources, or a user-first query within an embedding budget."""
    parts: list[str] = []
    if max_chars is None:
        for asst in recent_assistant_turns:
            if asst:
                parts.append(ASSISTANT_DELIM + asst)
        for tool_out in recent_tool_outputs:
            if tool_out:
                parts.append(TOOL_DELIM + tool_out)
        if user_text:
            parts.append(USER_DELIM + user_text)
        return "".join(parts)

    return "".join(
        bounded_embedding_parts(
            user_text=user_text,
            recent_tool_outputs=recent_tool_outputs,
            recent_assistant_turns=recent_assistant_turns,
            max_chars=max_chars,
        )
    )


def bounded_embedding_parts(
    *,
    user_text: str,
    recent_tool_outputs: tuple[str, ...],
    recent_assistant_turns: tuple[str, ...],
    max_chars: int = 4096,
) -> tuple[str, str]:
    """Keep question and context separate so a tokenizer can budget both."""
    remaining = max(0, max_chars)
    user_chars = min(2048, max(0, remaining - len(USER_DELIM)))
    question = USER_DELIM + user_text[-user_chars:] if user_text and user_chars else ""
    remaining -= len(question) + bool(question)
    parts: list[str] = []
    for delimiter, sources in (
        (TOOL_DELIM, recent_tool_outputs),
        (ASSISTANT_DELIM, recent_assistant_turns),
    ):
        for source in reversed(sources):
            if not source:
                continue
            source_chars = min(512, remaining - len(delimiter))
            if source_chars <= 0:
                break
            part = delimiter + source[-source_chars:]
            parts.append(part)
            remaining -= len(part)
    return question, "".join(reversed(parts))


def extract_memory_query_sources(
    messages: list[dict[str, Any]] | None,
    *,
    lookback_assistant: int = 2,
    lookback_tools: int = 3,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Extract user, tool, and assistant sources from chat-style messages.

    Returns ``(user_text, recent_tool_outputs, recent_assistant_turns)``.
    """
    if not messages:
        return "", (), ()

    latest_user = ""
    assistant_turns: list[str] = []
    tool_outputs: list[str] = []

    for msg in reversed(messages):
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, list):
                _append_anthropic_tool_results(
                    content,
                    tool_outputs=tool_outputs,
                    lookback_tools=lookback_tools,
                )
                if not latest_user:
                    # Anthropic user turns carry the actual prompt as text blocks
                    # ({"type":"text","text":...}), not a plain string. Capture it
                    # so the memory retrieval query keys on the user's question and
                    # not just any tool_result blocks in the same turn. Skip Claude
                    # Code's <system-reminder> harness blocks: they are appended to
                    # the same turn and, concatenated into the embedding input,
                    # dilute the user's question below the similarity floor so
                    # nothing is retrieved (#2195). Filtering them keeps the query
                    # on the substantive question.
                    user_text = "\n".join(
                        text
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                        for text in (str(b.get("text", "")).strip(),)
                        if text and not text.startswith("<system-reminder")
                    ).strip()
                    if user_text:
                        latest_user = user_text
            elif isinstance(content, str) and not latest_user:
                latest_user = content

        elif role == "assistant":
            assistant_text = _assistant_text(content)
            if assistant_text and len(assistant_turns) < lookback_assistant:
                assistant_turns.append(assistant_text)

        elif role == "tool":
            if isinstance(content, str) and content and len(tool_outputs) < lookback_tools:
                tool_outputs.append(content)

    return (
        latest_user,
        tuple(reversed(tool_outputs)),
        tuple(reversed(assistant_turns)),
    )


def _append_anthropic_tool_results(
    content: list[Any],
    *,
    tool_outputs: list[str],
    lookback_tools: int,
) -> None:
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        tool_text = block.get("content", "")
        if isinstance(tool_text, list):
            tool_text = "\n".join(b.get("text", "") for b in tool_text if isinstance(b, dict))
        if tool_text and len(tool_outputs) < lookback_tools:
            tool_outputs.append(str(tool_text))


def _assistant_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in text_parts if p)
    return ""
