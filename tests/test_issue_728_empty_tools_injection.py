"""Issue #728: proxy must not inject ``tools: []`` when the client omitted the tools field.

vLLM-based providers (Venice.ai, etc.) reject requests containing an empty ``tools``
array.  The bug: ``apply_session_sticky_ccr_tool`` / ``apply_session_sticky_memory_tools``
always return a list (empty when no tools exist and none were injected), and the
old handler guard ``if tools is not None`` evaluated True for ``[]``, causing
``body["tools"] = []`` to be sent upstream unconditionally.

Fix: the guard was changed to ``if tools or _original_tools is not None`` in both
the OpenAI and Anthropic handlers so that an empty result list only reaches the
outgoing body when the original request already carried a ``tools`` field.
"""

from __future__ import annotations

import pytest

from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import (
    _reset_session_ccr_tracker_for_test,
    apply_session_sticky_ccr_tool,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    _reset_session_ccr_tracker_for_test()
    yield
    _reset_session_ccr_tracker_for_test()


# ---------------------------------------------------------------------------
# Guard-condition logic (the actual fix)
# ---------------------------------------------------------------------------


def _should_set_body_tools(tools: list | None, original_tools: list | None) -> bool:
    """Mirror the fixed handler condition: ``if tools or _original_tools is not None``."""
    return bool(tools or original_tools is not None)


def _should_apply_presend_tools(presend_tools: list | None, original_tools: list | None) -> bool:
    """Mirror the fixed PRE_SEND write-back condition in the OpenAI handler."""
    return bool(presend_tools or original_tools is not None)


def _sort_tools(tools: list | None) -> list | None:
    return AnthropicHandlerMixin._sort_tools_deterministically(tools)


def _should_set_body_tools_after_sort(tools: list | None, original_tools: list | None) -> bool:
    """Mirror fixed logic when candidate tools may already be sorted."""
    if not _should_set_body_tools(tools, original_tools):
        return False
    sorted_tools = _sort_tools(tools)
    if sorted_tools != tools:
        tools = sorted_tools
    return tools != original_tools


def _legacy_should_set_body_tools_after_sort(
    tools: list | None, original_tools: list | None
) -> bool:
    """Older comparator that only wrote when sorting reordered."""
    if not _should_set_body_tools(tools, original_tools):
        return False
    return _sort_tools(tools) != tools


class TestHandlerGuardCondition:
    """Verify the guard condition that decides whether to write body['tools']."""

    def test_no_tools_no_injection_does_not_inject(self):
        """Client sent no tools and nothing was injected → body must stay tools-free."""
        original_tools = None  # client did not send tools
        tools_after_helpers = []  # helpers return [] when existing_tools=None and no inject

        assert not _should_set_body_tools(tools_after_helpers, original_tools), (
            "Empty tools from helpers + no original tools must NOT write body['tools']"
        )

    def test_client_sent_empty_tools_is_preserved(self):
        """Client explicitly sent ``tools: []`` → preserve that field (their choice)."""
        original_tools = []  # client explicitly sent an empty array
        tools_after_helpers = []  # nothing injected

        assert _should_set_body_tools(tools_after_helpers, original_tools), (
            "Client's explicit tools:[] should be preserved in body"
        )

    def test_ccr_injection_sets_body_tools(self):
        """When CCR injects a tool into an originally tool-free request → set body."""
        original_tools = None
        from headroom.ccr.tool_injection import create_ccr_tool_definition

        tools_after_helpers = [create_ccr_tool_definition("openai")]

        assert _should_set_body_tools(tools_after_helpers, original_tools), (
            "Injected CCR tool must reach body['tools']"
        )

    def test_client_tools_always_set(self):
        """Client provided real tools → always write body['tools']."""
        original_tools = [{"type": "function", "function": {"name": "my_tool"}}]
        tools_after_helpers = original_tools[:]

        assert _should_set_body_tools(tools_after_helpers, original_tools)

    def test_sorted_replacement_reaches_body(self):
        """A sorted replacement that differs from payload still needs to be written."""
        original_tools = [{"name": "zeta"}, {"name": "alpha"}]
        tools_after_helpers = [{"name": "alpha"}, {"name": "zeta"}]

        assert not _legacy_should_set_body_tools_after_sort(tools_after_helpers, original_tools)
        assert _should_set_body_tools_after_sort(tools_after_helpers, original_tools)

    def test_presend_empty_list_stays_omitted_when_client_omitted_tools(self):
        """PRE_SEND must not re-introduce ``tools: []`` for a tools-free request."""
        assert not _should_apply_presend_tools([], None)

    def test_presend_preserves_explicit_client_empty_tools(self):
        """PRE_SEND must still preserve an explicit client ``tools: []`` field."""
        assert _should_apply_presend_tools([], [])

    def test_presend_can_clear_previously_present_tools(self):
        """PRE_SEND may deliberately replace a real tool list with ``[]``."""
        original_tools = [{"type": "function", "function": {"name": "my_tool"}}]
        assert _should_apply_presend_tools([], original_tools)


# ---------------------------------------------------------------------------
# apply_session_sticky_ccr_tool behaviour with no existing tools
# ---------------------------------------------------------------------------


class TestCCRHelperNoToolsNoCompression:
    """Verify what the helper returns when there are no tools and no CCR happened."""

    def test_returns_empty_list_and_false_when_no_session_ccr(self):
        """No session CCR history + no compression this turn → ([], False)."""
        tools_out, was_injected = apply_session_sticky_ccr_tool(
            provider="openai",
            session_id="fresh-session-728",
            request_id="req-1",
            existing_tools=None,
            has_compressed_content_this_turn=False,
        )
        assert was_injected is False
        # Helper still returns [] — the guard in the handler is what prevents injection.
        assert tools_out == []

    def test_returns_tool_list_when_compression_occurred(self):
        """First turn with CCR → helper returns the CCR tool definition."""
        tools_out, was_injected = apply_session_sticky_ccr_tool(
            provider="openai",
            session_id="ccr-session-728",
            request_id="req-1",
            existing_tools=None,
            has_compressed_content_this_turn=True,
        )
        assert was_injected is True
        tool_names = [t.get("function", {}).get("name") or t.get("name") for t in tools_out]
        assert CCR_TOOL_NAME in tool_names

    def test_no_double_injection_when_client_pre_registered_ccr_tool(self):
        """If the client already included the CCR tool, the helper must not duplicate it."""
        from headroom.ccr.tool_injection import create_ccr_tool_definition

        existing = [create_ccr_tool_definition("openai")]
        tools_out, was_injected = apply_session_sticky_ccr_tool(
            provider="openai",
            session_id="pre-reg-session-728",
            request_id="req-1",
            existing_tools=existing,
            has_compressed_content_this_turn=True,
        )
        assert was_injected is False
        ccr_count = sum(
            1
            for t in tools_out
            if (t.get("function", {}).get("name") or t.get("name")) == CCR_TOOL_NAME
        )
        assert ccr_count == 1, "CCR tool should appear exactly once"
