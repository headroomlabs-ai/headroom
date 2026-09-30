"""Tool-description compaction must run on chat-completions, not just Anthropic/Responses.

``HEADROOM_TOOL_DESC_MAX_CHARS`` was wired into the Anthropic handler and the
Responses (Codex) handler but never into chat-completions, so the env var was a
silent no-op for every chat client — opencode, Cline, Aider, Roo, anything routed
through LiteLLM. Tool descriptions live on the ``tools`` array, which the message
pipeline never inspects, so no other pass was covering them.

These tests pin the shape handling and the opt-in gate rather than driving the
whole handler: the handler block is a thin adapter over
``compact_tool_descriptions``, and the thing that actually broke was that nobody
called it with the chat-shaped payload.
"""

from __future__ import annotations

import pytest

import headroom.proxy.tool_schema_compaction as tsc
from headroom.proxy.tool_schema_compaction import compact_tool_descriptions, tool_desc_max_chars

_LONG_DESC = "Reads a file from disk and returns the full text content as a string, with numbers."


@pytest.fixture(autouse=True)
def _reset_desc_cache():
    """The max-chars lookup is process-cached; clear it around each test."""
    tsc._TOOL_DESC_MAX_CHARS = None
    yield
    tsc._TOOL_DESC_MAX_CHARS = None


def _chat_tools() -> list[dict]:
    """chat-completions shape: name/description nested under "function"."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": _LONG_DESC,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path to read"}},
                },
            },
        }
    ]


def _responses_tools() -> list[dict]:
    """Responses shape: name/description flat on the tool."""
    return [
        {
            "type": "function",
            "name": "read",
            "description": _LONG_DESC,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
            },
        }
    ]


def test_compacts_the_nested_chat_completions_tool_shape(monkeypatch):
    """The shape the chat handler passes — the one that was never being compacted."""
    monkeypatch.setenv("HEADROOM_TOOL_DESC_MAX_CHARS", "30")

    payload, modified, before, after = compact_tool_descriptions(
        {"tools": _chat_tools()}, tool_desc_max_chars()
    )

    assert modified is True
    assert after < before
    desc = payload["tools"][0]["function"]["description"]
    assert len(desc) <= len(_LONG_DESC)
    assert desc != _LONG_DESC


def test_both_wire_shapes_are_handled(monkeypatch):
    """One helper serves both handlers, so chat needed wiring — not a new codec."""
    monkeypatch.setenv("HEADROOM_TOOL_DESC_MAX_CHARS", "30")
    max_chars = tool_desc_max_chars()

    _, chat_modified, chat_before, chat_after = compact_tool_descriptions(
        {"tools": _chat_tools()}, max_chars
    )
    _, resp_modified, resp_before, resp_after = compact_tool_descriptions(
        {"tools": _responses_tools()}, max_chars
    )

    assert chat_modified is resp_modified is True
    assert chat_after < chat_before
    assert resp_after < resp_before


def test_disabled_by_default_leaves_tools_untouched():
    """Opt-in only: an unset env var must not perturb the tools prefix or its cache."""
    tools = _chat_tools()
    assert tool_desc_max_chars() == 0

    payload, modified, before, after = compact_tool_descriptions(
        {"tools": tools}, tool_desc_max_chars()
    )

    assert modified is False
    assert payload["tools"] == tools
    assert (before, after) == (0, 0)


def test_chat_handler_calls_the_desc_pass(monkeypatch):
    """Guard the wiring itself: the handler source must invoke the L2 pass.

    ponytail: source-level check, not a live handler drive — spinning the full
    chat-completions path needs an upstream, and the regression here was a missing
    CALL, which is exactly what this catches.
    """
    import inspect

    from headroom.proxy.handlers import openai as openai_handler

    source = inspect.getsource(openai_handler)
    assert "openai:chat:tool_desc_compaction" in source
    # The Anthropic and Responses handlers already had their own labels; make sure
    # the chat one is distinct so `headroom perf --by-transform` can attribute it.
    assert "openai:responses:tool_desc_compaction" in source
