"""Tests that Kompress never applies lossy ML compression to tool role messages.

Regression for https://github.com/headroomlabs-ai/headroom/issues/1307:
tool_result content is ground truth (grep/ls/cat/find output). Kompress
produces plausible-but-wrong reconstructions — fabricated file:line pairs,
missing files, etc. — that agents act on as fact.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from headroom.transforms.kompress_compressor import KompressCompressor
from headroom.config import TransformResult


GREP_OUTPUT = (
    "headroom/transforms/kompress_compressor.py:1375:            if role in (\"tool\", \"assistant\"):\n"
    "headroom/transforms/smart_crusher.py:1010:            if msg.get(\"role\") == \"tool\":\n"
    "headroom/transforms/content_router.py:2653:            if role == \"tool\":\n"
    "headroom/proxy/handlers/anthropic.py:44:                elif block.get(\"type\") == \"tool_result\":\n"
    "headroom/cache/prefix_tracker.py:88:            if message.get(\"role\") == \"tool\":\n"
    "headroom/proxy/helpers.py:102:        if msg.get(\"role\") == \"tool\":\n"
)

LS_OUTPUT = (
    "total 48\n"
    "drwxr-xr-x  12 peter staff  384 Jun 24 11:00 .\n"
    "drwxr-xr-x  33 peter staff 1056 Jun 24 10:00 ..\n"
    "drwxr-xr-x   9 peter staff  288 Jun 24 11:00 .git\n"
    "-rw-r--r--   1 peter staff 4963 Jun 24 11:00 build.mjs\n"
    "-rw-r--r--   1 peter staff 1042 Jun 24 11:00 assets/logo.svg\n"
    "drwxr-xr-x   8 peter staff  256 Jun 24 11:00 posts\n"
    "-rw-r--r--   1 peter staff 2048 Jun 24 11:00 _headers\n"
    "-rw-r--r--   1 peter staff 1024 Jun 24 11:00 package.json\n"
)

LONG_ASSISTANT = " ".join([f"word{i}" for i in range(200)])  # >10 words, compressible


def _make_compressor_with_mock() -> tuple[KompressCompressor, MagicMock]:
    """Return a KompressCompressor whose .compress() is mocked to track calls."""
    compressor = KompressCompressor.__new__(KompressCompressor)
    mock_compress = MagicMock(return_value=MagicMock(
        compressed="[KOMPRESS OUTPUT — WOULD BE LOSSY]",
        compression_ratio=0.3,
    ))
    compressor.compress = mock_compress
    return compressor, mock_compress


class _StubTokenizer:
    """Minimal tokenizer stub — word-count approximation, no model needed."""
    def count_text(self, text: str) -> int:
        return len(str(text).split())
    def count_messages(self, messages: list) -> int:
        return sum(self.count_text(str(m.get("content", ""))) for m in messages)

def _tokenizer() -> _StubTokenizer:
    return _StubTokenizer()


def test_tool_role_content_preserved_exactly() -> None:
    """tool role messages must pass through Kompress unchanged — bit-for-bit."""
    compressor, mock_compress = _make_compressor_with_mock()
    tokenizer = _tokenizer()

    messages = [{"role": "tool", "content": GREP_OUTPUT}]
    result = compressor.apply(messages, tokenizer)

    # Kompress must never have been called on a tool message
    mock_compress.assert_not_called()

    # Content must be exactly the original — no reconstruction
    assert result.messages[0]["content"] == GREP_OUTPUT


def test_tool_role_ls_output_preserved() -> None:
    """ls output (role=tool) must not be touched by Kompress."""
    compressor, mock_compress = _make_compressor_with_mock()
    tokenizer = _tokenizer()

    messages = [{"role": "tool", "content": LS_OUTPUT}]
    result = compressor.apply(messages, tokenizer)

    mock_compress.assert_not_called()
    assert result.messages[0]["content"] == LS_OUTPUT


def test_assistant_role_can_be_kompressed() -> None:
    """assistant messages are still eligible for Kompress compression."""
    compressor, mock_compress = _make_compressor_with_mock()
    tokenizer = _tokenizer()

    messages = [{"role": "assistant", "content": LONG_ASSISTANT}]
    compressor.apply(messages, tokenizer)

    # Kompress should have been called for assistant
    mock_compress.assert_called_once()


def test_mixed_messages_tool_untouched_assistant_eligible() -> None:
    """In a mixed conversation, tool messages are never Kompress'd."""
    compressor, mock_compress = _make_compressor_with_mock()
    tokenizer = _tokenizer()

    messages = [
        {"role": "user", "content": "what files are in the repo?"},
        {"role": "assistant", "content": LONG_ASSISTANT},
        {"role": "tool", "content": LS_OUTPUT, "tool_call_id": "call_abc123"},
        {"role": "assistant", "content": LONG_ASSISTANT},
        {"role": "tool", "content": GREP_OUTPUT, "tool_call_id": "call_def456"},
    ]
    result = compressor.apply(messages, tokenizer)

    # Kompress called at most twice (for the two assistant messages)
    assert mock_compress.call_count <= 2

    # Tool messages preserved exactly
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == LS_OUTPUT
    assert tool_messages[1]["content"] == GREP_OUTPUT


def test_anthropic_tool_result_list_content_already_protected() -> None:
    """Anthropic-format tool_result blocks (content as list) pass through unchanged.

    These hit the `isinstance(content, str)` guard before the role check,
    so they are already safe.  This test documents and guards that behavior.
    """
    compressor, mock_compress = _make_compressor_with_mock()
    tokenizer = _tokenizer()

    anthropic_tool_result = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_abc",
                "content": GREP_OUTPUT,
            }
        ],
    }
    messages = [anthropic_tool_result]
    result = compressor.apply(messages, tokenizer)

    mock_compress.assert_not_called()
    assert result.messages[0] == anthropic_tool_result


def test_no_transform_marker_emitted_for_tool_role() -> None:
    """transforms_applied must not contain a kompress:tool entry."""
    compressor, _ = _make_compressor_with_mock()
    tokenizer = _tokenizer()

    messages = [{"role": "tool", "content": GREP_OUTPUT}]
    result = compressor.apply(messages, tokenizer)

    assert not any("kompress:tool" in t for t in result.transforms_applied)
