"""Message parsing utilities for Headroom SDK."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from .config import Block, WasteSignals

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


# Patterns for detecting waste signals
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_COMMENT_PATTERN = re.compile(r"<!--[\s\S]*?-->")
BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{50,}={0,2}")
WHITESPACE_PATTERN = re.compile(r"[ \t]{4,}|\n{3,}")
JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]{500,}\}")

# Tool results below this size legitimately repeat ("ok", empty diffs,
# exit codes) and are not evidence of a re-read.
REREAD_MIN_TOKENS = 50

# Repeats this close (in message positions) to the previous serve are
# polling, not re-reads. Consecutive tool turns sit 2 apart (the
# assistant tool_use message lies between results); 3 also absorbs a
# thinking/user nudge in the loop. Larger gaps mean the agent moved on
# and then came back — the over-compression signal we want.
REREAD_ADJACENT_GAP = 3

# Patterns for RAG detection (best effort)
RAG_MARKERS = [
    r"\[Document\s*\d+\]",
    r"\[Source:\s*",
    r"<context>",
    r"<document>",
    r"Retrieved from:",
    r"From the knowledge base:",
]
RAG_PATTERN = re.compile("|".join(RAG_MARKERS), re.IGNORECASE)


def compute_hash(text: str) -> str:
    """Compute hash of text, truncated to 16 chars."""
    return hashlib.md5(text.encode()).hexdigest()[:16]  # nosec B324


def _extract_tool_result_text(payload: dict[str, Any]) -> str:
    """Extract text from a tool result payload.

    Handles the Anthropic ``tool_result`` block (``payload["content"]``
    is a plain string or a list of ``{"type": "text", ...}`` blocks) and
    the Strands/Bedrock ``toolResult`` payload (content items keyed as
    ``{"text": ...}`` or ``{"json": ...}`` without a ``type`` field).
    Non-text inner blocks (e.g. images) are skipped.
    """
    inner = payload.get("content")
    if inner is None:
        return ""
    if isinstance(inner, str):
        return inner
    if isinstance(inner, list):
        pieces = []
        for item in inner:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    pieces.append(item.get("text", ""))
                elif "type" not in item and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
                elif "type" not in item and "json" in item:
                    pieces.append(json.dumps(item["json"], default=str))
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(pieces)
    if isinstance(inner, dict):
        return json.dumps(inner, default=str)
    return str(inner)


def detect_waste_signals(text: str, tokenizer: Tokenizer) -> WasteSignals:
    """
    Detect waste signals in text.

    Args:
        text: The text to analyze.
        tokenizer: Tokenizer for counting tokens.

    Returns:
        WasteSignals with detected waste.
    """
    signals = WasteSignals()

    if not text:
        return signals

    # HTML tags and comments
    html_matches = HTML_TAG_PATTERN.findall(text) + HTML_COMMENT_PATTERN.findall(text)
    if html_matches:
        html_text = "".join(html_matches)
        signals.html_noise_tokens = tokenizer.count_text(html_text)

    # Base64 blobs
    base64_matches = BASE64_PATTERN.findall(text)
    if base64_matches:
        base64_text = "".join(base64_matches)
        signals.base64_tokens = tokenizer.count_text(base64_text)

    # Excessive whitespace
    ws_matches = WHITESPACE_PATTERN.findall(text)
    if ws_matches:
        # Count tokens that could be saved by normalizing whitespace to single spaces
        ws_text = "".join(ws_matches)
        normalized_text = " ".join(ws_matches)
        signals.whitespace_tokens = max(
            0, tokenizer.count_text(ws_text) - tokenizer.count_text(normalized_text)
        )

    # Large JSON blocks
    json_matches = JSON_BLOCK_PATTERN.findall(text)
    if json_matches:
        for match in json_matches:
            tokens = tokenizer.count_text(match)
            if tokens > 500:
                signals.json_bloat_tokens += tokens

    return signals


def is_rag_content(text: str) -> bool:
    """Check if text appears to be RAG-injected content."""
    return RAG_PATTERN.search(text) is not None


def parse_message_to_blocks(
    message: dict[str, Any],
    index: int,
    tokenizer: Tokenizer,
) -> list[Block]:
    """
    Parse a single message into Block objects.

    Args:
        message: The message dict to parse.
        index: Position in the message list.
        tokenizer: Tokenizer for token counting.

    Returns:
        List of Block objects (usually 1, but tool_calls may produce multiple).
    """
    blocks: list[Block] = []
    role = message.get("role", "unknown")

    # Handle content
    content = message.get("content")
    if content:
        tool_result_parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Multi-modal - extract text parts
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "tool_result":
                    # Anthropic Messages format nests tool output one level
                    # deeper; collect for dedicated tool_result blocks below.
                    tool_result_parts.append(part)
                elif isinstance(part, dict) and "toolResult" in part:
                    # Strands/Bedrock converse format; same treatment.
                    tool_result_parts.append(part)
                elif isinstance(part, str):
                    text_parts.append(part)
            text = "\n".join(text_parts)
        else:
            text = str(content)

        # Determine block kind
        if role == "system":
            kind = "system"
        elif role == "user":
            # Check if this looks like RAG content
            kind = "rag" if is_rag_content(text) else "user"
        elif role == "assistant":
            kind = "assistant"
        elif role == "tool":
            kind = "tool_result"
        else:
            kind = "unknown"

        # Build flags
        flags: dict[str, Any] = {}
        if role == "tool":
            flags["tool_call_id"] = message.get("tool_call_id")

        # Detect waste
        waste = detect_waste_signals(text, tokenizer)
        if waste.total() > 0:
            flags["waste_signals"] = waste.to_dict()

        tr_blocks: list[Block] = []
        for part in tool_result_parts:
            payload = part["toolResult"] if "toolResult" in part else part
            if not isinstance(payload, dict):
                continue
            tr_text = _extract_tool_result_text(payload)
            if not tr_text:
                continue

            tr_id = payload.get("toolUseId") if "toolResult" in part else part.get("tool_use_id")
            tr_flags: dict[str, Any] = {"tool_call_id": tr_id}
            tr_waste = detect_waste_signals(tr_text, tokenizer)
            if tr_waste.total() > 0:
                tr_flags["waste_signals"] = tr_waste.to_dict()

            tr_blocks.append(
                Block(
                    kind="tool_result",
                    text=tr_text,
                    tokens_est=tokenizer.count_text(tr_text) + 4,  # Add message overhead
                    content_hash=compute_hash(tr_text),
                    source_index=index,
                    flags=tr_flags,
                )
            )

        # Tool-result-only messages are fully represented by their dedicated
        # blocks; skip the empty container block in that case.
        if text or not tr_blocks:
            blocks.append(
                Block(
                    kind=kind,  # type: ignore[arg-type]
                    text=text,
                    tokens_est=tokenizer.count_text(text) + 4,  # Add message overhead
                    content_hash=compute_hash(text),
                    source_index=index,
                    flags=flags,
                )
            )
        blocks.extend(tr_blocks)

    # Handle tool calls (assistant messages with tool_calls)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            tc_text = f"{func.get('name', 'unknown')}({func.get('arguments', '')})"

            blocks.append(
                Block(
                    kind="tool_call",
                    text=tc_text,
                    tokens_est=tokenizer.count_text(tc_text) + 10,
                    content_hash=compute_hash(tc_text),
                    source_index=index,
                    flags={
                        "tool_call_id": tc.get("id"),
                        "function_name": func.get("name"),
                    },
                )
            )

    # If no content or tool_calls, create a minimal block
    if not blocks:
        blocks.append(
            Block(
                kind="unknown",
                text="",
                tokens_est=4,
                content_hash=compute_hash(""),
                source_index=index,
                flags={},
            )
        )

    return blocks


def parse_messages(
    messages: list[dict[str, Any]],
    tokenizer: Tokenizer,
) -> tuple[list[Block], dict[str, int], WasteSignals]:
    """
    Parse all messages into blocks with analysis.

    Args:
        messages: List of message dicts.
        tokenizer: Tokenizer instance for token counting.

    Returns:
        Tuple of (blocks, block_breakdown, total_waste_signals)
    """
    all_blocks: list[Block] = []
    total_waste = WasteSignals()

    for i, msg in enumerate(messages):
        blocks = parse_message_to_blocks(msg, i, tokenizer)
        all_blocks.extend(blocks)

        # Accumulate waste signals
        for block in blocks:
            if "waste_signals" in block.flags:
                ws = block.flags["waste_signals"]
                total_waste.json_bloat_tokens += ws.get("json_bloat", 0)
                total_waste.html_noise_tokens += ws.get("html_noise", 0)
                total_waste.base64_tokens += ws.get("base64", 0)
                total_waste.whitespace_tokens += ws.get("whitespace", 0)
                total_waste.dynamic_date_tokens += ws.get("dynamic_date", 0)
                total_waste.repetition_tokens += ws.get("repetition", 0)

    # Cross-message re-read detection: identical tool_result content served
    # at more than one position means the agent re-fetched something already
    # in context — an over-compression signal (#853). The first serve is
    # free; every repeat is counted as waste.
    reread_groups: dict[str, list[Block]] = {}
    for block in all_blocks:
        if block.kind == "tool_result" and block.tokens_est >= REREAD_MIN_TOKENS:
            reread_groups.setdefault(block.content_hash, []).append(block)
    for group in reread_groups.values():
        # The message that first served the content is the original; only
        # copies appearing in *later* messages are re-reads. Duplicates
        # within the original message are excluded, and so are polling
        # repeats: agents that poll (repeated `git status`, CI checks)
        # legitimately produce byte-identical results a couple of messages
        # apart. A repeat only counts when it lands more than
        # REREAD_ADJACENT_GAP messages after the previous serve; nearer
        # repeats advance the baseline without counting, so a long polling
        # chain never accumulates waste.
        prev_index = group[0].source_index
        for block in group:
            if block.source_index == prev_index:
                continue
            is_polling = block.source_index - prev_index <= REREAD_ADJACENT_GAP
            prev_index = block.source_index
            if not is_polling:
                total_waste.reread_tokens += block.tokens_est

    # Compute block breakdown
    breakdown: dict[str, int] = {}
    for block in all_blocks:
        kind = block.kind
        breakdown[kind] = breakdown.get(kind, 0) + block.tokens_est

    return all_blocks, breakdown, total_waste


def find_tool_units(messages: list[dict[str, Any]]) -> list[tuple[int, list[int]]]:
    """
    Find tool call units (assistant with tool_calls + corresponding tool responses).

    A tool unit is atomic - if the assistant message is dropped, all its
    tool responses must also be dropped.

    Supports both OpenAI and Anthropic formats:
    - OpenAI: assistant.tool_calls[] + tool messages with tool_call_id
    - Anthropic: assistant.content[type=tool_use] + user.content[type=tool_result]

    Args:
        messages: List of message dicts.

    Returns:
        List of (assistant_index, [tool_response_indices]) tuples.
    """
    units: list[tuple[int, list[int]]] = []

    # Build map of tool_call_id -> message index for tool responses
    tool_response_map: dict[str, int] = {}
    for i, msg in enumerate(messages):
        # OpenAI format: role="tool" with tool_call_id
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                tool_response_map[tc_id] = i

        # Anthropic format: role="user" with content blocks containing tool_result
        # Also handles Strands SDK format: {"toolResult": {"toolUseId": "..."}}
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            tc_id = block.get("tool_use_id")
                            if tc_id:
                                tool_response_map[tc_id] = i
                        elif "toolResult" in block:
                            # Strands SDK format
                            tc_id = block["toolResult"].get("toolUseId")
                            if tc_id:
                                tool_response_map[tc_id] = i

    # Find assistant messages with tool calls
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        response_indices: list[int] = []

        # OpenAI format: tool_calls array
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tc_id = tc.get("id")
                if tc_id and tc_id in tool_response_map:
                    response_indices.append(tool_response_map[tc_id])

        # Anthropic format: content blocks with type=tool_use
        # Also handles Strands SDK format: {"toolUse": {"toolUseId": "..."}}
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use":
                        tc_id = block.get("id")
                        if tc_id and tc_id in tool_response_map:
                            response_indices.append(tool_response_map[tc_id])
                    elif "toolUse" in block:
                        # Strands SDK format
                        tc_id = block["toolUse"].get("toolUseId")
                        if tc_id and tc_id in tool_response_map:
                            response_indices.append(tool_response_map[tc_id])

        if response_indices:
            # Use set to deduplicate in case same message has both formats
            units.append((i, sorted(set(response_indices))))

    return units


def get_message_content_text(message: dict[str, Any]) -> str:
    """Extract text content from a message."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)
