"""Tests for cache_control breakpoint diagnostics and log-privacy switches.

Covers the three pieces added for the uncached-tail investigation:
- ``count_cache_breakpoints`` / ``log_cache_breakpoints`` (proxy helpers)
- the ``HEADROOM_LOG_PAYLOAD_PREVIEW`` kill switch (compression store)
- the injection guard that keeps proactive expansion out of breakpointed blocks
"""

from __future__ import annotations

import logging

from headroom.cache.compression_store import _payload_for_retrieval_log
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import count_cache_breakpoints, log_cache_breakpoints

_CC = {"cache_control": {"type": "ephemeral"}}


def _claude_code_style_request() -> tuple[list[dict], list[dict], list[dict]]:
    """System/messages/tools shaped like a real Claude Code request."""
    system = [
        {"type": "text", "text": "You are Claude Code."},
        {"type": "text", "text": "project instructions", **_CC},
    ]
    tools = [
        {"name": "Bash", "input_schema": {}},
        {"name": "Read", "input_schema": {}, **_CC},
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi", **_CC}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ack"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "big output"}],
                    **_CC,
                }
            ],
        },
    ]
    return system, messages, tools


def test_count_cache_breakpoints_counts_all_sections() -> None:
    system, messages, tools = _claude_code_style_request()
    stats = count_cache_breakpoints(system, messages, tools)
    assert stats["system"] == 1
    assert stats["tools"] == 1
    assert stats["messages"] == 2
    assert stats["total"] == 4
    assert stats["message_count"] == 3
    assert stats["last_marker_tail"] == 0  # last message carries a marker


def test_count_cache_breakpoints_counts_nested_tool_result_markers() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "out", **_CC}],
                }
            ],
        }
    ]
    stats = count_cache_breakpoints("plain system string", messages, None)
    assert stats["system"] == 0
    assert stats["tools"] == 0
    assert stats["messages"] == 1
    assert stats["last_marker_tail"] == 0


def test_count_cache_breakpoints_tail_tracks_last_marker() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a", **_CC}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "c"}]},
    ]
    stats = count_cache_breakpoints(None, messages, None)
    assert stats["last_marker_tail"] == 2
    assert count_cache_breakpoints(None, [], None)["last_marker_tail"] == -1


def test_log_cache_breakpoints_warns_on_dropped_marker(caplog) -> None:
    system, messages, tools = _claude_code_style_request()
    inbound = count_cache_breakpoints(system, messages, tools)
    # Transform "lost" the final breakpoint: strip it from the last message.
    stripped = [dict(m) for m in messages]
    stripped[2] = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "compressed"}],
    }
    outbound = count_cache_breakpoints(system, stripped, tools)
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        log_cache_breakpoints(request_id="r1", inbound=inbound, outbound=outbound)
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "dropped=true" in record.getMessage()
    assert "tail_grew=true" in record.getMessage()


def test_log_cache_breakpoints_info_when_preserved(caplog) -> None:
    system, messages, tools = _claude_code_style_request()
    stats = count_cache_breakpoints(system, messages, tools)
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        log_cache_breakpoints(request_id="r1", inbound=stats, outbound=stats)
    [record] = caplog.records
    assert record.levelno == logging.INFO
    assert "dropped=false" in record.getMessage()


def test_payload_preview_disabled_omits_content(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", "0")
    payload = "secret file contents: api_key=sk-abcdefghijklmnop"
    event = _payload_for_retrieval_log(payload)
    assert event["payload_preview"] == ""
    assert event["payload_preview_chars"] == 0
    assert event["payload_chars"] == len(payload)
    assert event["payload_truncated"] is True


def test_payload_preview_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    event = _payload_for_retrieval_log("hello world")
    assert event["payload_preview"] == "hello world"


def test_append_context_skips_breakpointed_text_block() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "breakpointed", **_CC},
                {"type": "text", "text": "free"},
            ],
        }
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages, "CTX", frozen_message_count=0
    )
    blocks = result[0]["content"]
    assert blocks[0]["text"] == "breakpointed"  # untouched
    assert blocks[1]["text"].endswith("CTX")


def test_append_context_no_eligible_block_returns_unchanged() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "breakpointed", **_CC}],
        }
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages, "CTX", frozen_message_count=0
    )
    assert result == messages


def test_count_cache_breakpoints_tolerates_malformed_shapes() -> None:
    messages = [
        "not-a-dict",
        {"role": "user", "content": ["scalar-block", {"type": "text", "text": "x", **_CC}]},
        {"role": "user", "content": "plain string"},
    ]
    stats = count_cache_breakpoints("system-as-string", messages, "tools-as-string")
    assert stats["system"] == 0
    assert stats["tools"] == 0
    assert stats["messages"] == 1
    assert stats["message_count"] == 3
    assert stats["last_marker_tail"] == 1

    empty = count_cache_breakpoints(None, None, None)
    assert empty["total"] == 0
    assert empty["message_count"] == 0
