"""Handler-level regression tests for Anthropic compaction transforms_applied reporting.

Verifies that when tool-schema compaction (L1), tool-description compaction (L2),
or system-prompt compaction (L3) modifies an Anthropic request, the corresponding
label is appended to ``transforms_applied`` so that ``/stats`` and the
transformation accounting remain accurate.

These tests exercise the handler wiring directly (not just the helper functions)
to catch the specific bug where Anthropic omitted the append calls that the
OpenAI handler already had.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_anthropic_payload_with_tools() -> dict:
    """Minimal Anthropic-style payload with tools that will be compacted."""
    return {
        "model": "claude-sonnet-4-6",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "read_file",
                "description": "Read the contents of a file from disk.  "
                "Returns the full text content as a string.",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "read_file_schema",
                    "examples": [{"path": "/tmp/test.txt"}],
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            }
        ],
        "max_tokens": 1024,
    }


def _make_anthropic_payload_with_long_system() -> dict:
    """Payload with a long system prompt that qualifies for L3 compaction."""
    long_text = "x" * 5000  # well above default min_chars
    return {
        "model": "claude-sonnet-4-6",
        "system": [
            {"type": "text", "text": long_text, "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "max_tokens": 1024,
    }


# ---------------------------------------------------------------------------
# L1: Tool schema compaction
# ---------------------------------------------------------------------------


class TestAnthropicToolSchemaCompactionTransforms:
    """When L1 compaction modifies tools, ``anthropic:tool_schema_compaction``
    must appear in ``transforms_applied``."""

    def test_l1_appends_transform_label(self) -> None:
        from headroom.proxy.tool_schema_compaction import compact_tools

        payload = _make_anthropic_payload_with_tools()
        body, modified, before, after = compact_tools(payload)

        assert modified is True
        # The handler code does:
        #   if _tools_modified:
        #       transforms_applied.append("anthropic:tool_schema_compaction")
        # We verify the condition that triggers the append is met.
        assert before > after

    def test_l1_skips_label_when_no_compaction(self) -> None:
        from headroom.proxy.tool_schema_compaction import compact_tools

        payload = _make_anthropic_payload_with_tools()
        # Already compact — remove annotation keys AND normalise description
        # so compact_tools has nothing to change.
        schema = payload["tools"][0]["input_schema"]
        schema.pop("$schema", None)
        schema.pop("title", None)
        schema.pop("examples", None)
        # Normalise description whitespace to match compaction output.
        payload["tools"][0]["description"] = " ".join(payload["tools"][0]["description"].split())

        body, modified, before, after = compact_tools(payload)
        # When nothing can be compacted, the handler should NOT append the label.
        # We verify the condition: _tools_modified must be False.
        assert modified is False


# ---------------------------------------------------------------------------
# L2: Tool description compaction
# ---------------------------------------------------------------------------


class TestAnthropicToolDescCompactionTransforms:
    """When L2 compaction truncates descriptions,
    ``anthropic:tool_desc_compaction`` must appear in ``transforms_applied``."""

    def test_l2_appends_transform_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import headroom.proxy.tool_schema_compaction as _mod
        from headroom.proxy.tool_schema_compaction import (
            compact_tool_descriptions,
            tool_desc_max_chars,
        )

        # Opt-in with a very short max so truncation triggers. Reset the
        # per-process cache first: an earlier test in the shard may have read
        # the (unset) env and pinned max_chars to 0, which would swallow our
        # setenv below.
        monkeypatch.setenv("HEADROOM_TOOL_DESC_MAX_CHARS", "20")
        _mod._TOOL_DESC_MAX_CHARS = None

        payload = _make_anthropic_payload_with_tools()
        max_chars = tool_desc_max_chars()
        assert max_chars == 20

        body, modified, before, after = compact_tool_descriptions(payload, max_chars)
        assert modified is True
        assert before > after
        # Don't leak the cached 20 into later tests in this shard.
        _mod._TOOL_DESC_MAX_CHARS = None

    def test_l2_skips_label_when_disabled(self) -> None:
        import headroom.proxy.tool_schema_compaction as _mod
        from headroom.proxy.tool_schema_compaction import tool_desc_max_chars

        # Reset the per-process cache so the env var is re-read.
        _mod._TOOL_DESC_MAX_CHARS = None
        with patch.dict(os.environ, {}, clear=True):
            max_chars = tool_desc_max_chars()
        # Restore cache state for subsequent tests.
        _mod._TOOL_DESC_MAX_CHARS = None
        # When max_chars == 0, the handler skips the entire L2 block,
        # so no append happens.
        assert max_chars == 0


# ---------------------------------------------------------------------------
# L3: System prompt compaction
# ---------------------------------------------------------------------------


class TestAnthropicSystemCompactionTransforms:
    """When L3 compaction compresses system blocks,
    ``anthropic:system_prompt_compaction`` must appear in ``transforms_applied``."""

    def test_l3_appends_transform_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from headroom.proxy.system_compaction import (
            compact_system_prompt,
        )

        monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")

        payload = _make_anthropic_payload_with_long_system()

        # Mock the router so we don't need a real one.
        class _MockCompressResult:
            def __init__(self, compressed: str):
                self.compressed = compressed

        class _MockRouter:
            def compress(self, text: str, **kwargs):
                return _MockCompressResult(text[:100])

        body, modified, before, after = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="claude-sonnet-4-6",
            request_id="test-req",
        )

        assert modified is True
        assert before > after

    def test_l3_skips_label_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from headroom.proxy.system_compaction import system_compact_enabled

        monkeypatch.delenv("HEADROOM_SYSTEM_COMPACT", raising=False)
        assert system_compact_enabled() is False
        # When disabled, the handler skips L3 entirely, so no append.


# ---------------------------------------------------------------------------
# Handler-level end-to-end regression (issue: Anthropic omitted the append)
#
# The tests above exercise the helper return values in isolation. These below
# drive the *handler wiring* end-to-end: a real ``_handle_anthropic_request``
# runs against a tool-bearing payload, the live ``compact_tools`` mutates it,
# and the L1 label must surface on the ``x-headroom-transforms`` response
# header. This is the gap the maintainer flagged -- the bug lived in the
# handler's append call, not in the helpers.
# ---------------------------------------------------------------------------

import httpx  # noqa: E402

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_proxy_client() -> TestClient:
    config = ProxyConfig(
        optimize=True,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    return TestClient(app)


def _ok_response(msg_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )


class TestAnthropicHandlerReportsL1Transform:
    """End-to-end: when L1 tool-schema compaction mutates the request, the
    handler must append ``anthropic:tool_schema_compaction`` so it reaches the
    ``x-headroom-transforms`` response header -- not just the helper's
    ``modified`` flag."""

    def test_l1_label_reaches_response_header(self) -> None:
        from types import SimpleNamespace

        with _make_proxy_client() as client:
            proxy = client.app.state.proxy

            def _fake_apply(**kwargs):
                # Return the minimum result shape the handler reads; let the
                # handler's own compaction pass (which runs after apply) do the
                # real mutation we are testing.
                return SimpleNamespace(
                    messages=kwargs["messages"],
                    transforms_applied=[],
                    timing={},
                    tokens_before=10,
                    tokens_after=10,
                    waste_signals=None,
                )

            proxy.anthropic_pipeline.apply = _fake_apply

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                return _ok_response("msg_l1_e2e")

            proxy._retry_request = _fake_retry

            response = client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a file from disk.  Returns text content.",
                            "input_schema": {
                                "$schema": "https://json-schema.org/draft/2020-12/schema",
                                "title": "read_file_schema",
                                "examples": [{"path": "/tmp/test.txt"}],
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                },
                                "required": ["path"],
                            },
                        }
                    ],
                },
            )

        assert response.status_code == 200, response.text
        transforms_header = response.headers.get("x-headroom-transforms", "")
        assert "anthropic:tool_schema_compaction" in transforms_header, (
            f"expected L1 label in x-headroom-transforms, got: {transforms_header!r}"
        )
