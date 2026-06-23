"""Regression test for #1006: the proxy must not emit unredeemable CCR markers.

When frozen_message_count > 0, the old code deferred headroom_retrieve tool
injection unconditionally — even if compression just emitted NEW <<ccr:hash>>
markers the agent has no tool to redeem.

The fix: if new markers were emitted this turn, override the deferral and inject
the tool (one cache miss is acceptable; silent data loss is not).

This test verifies that when compression produces markers in a frozen-prefix
session that has never done CCR before, the headroom_retrieve tool appears in
the outbound tools array.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from headroom.ccr.tool_injection import CCR_TOOL_NAME, CCRToolInjector
from headroom.proxy.helpers import apply_session_sticky_ccr_tool


class TestCCRInjectionNotDeferredWhenMarkersEmitted:
    """apply_session_sticky_ccr_tool path: injector.has_compressed_content
    must force injection even when frozen_message_count > 0."""

    def test_inject_when_new_markers_and_frozen_prefix(self):
        """If the injector found new markers this turn, apply_session_sticky_ccr_tool
        must inject the tool even though we would normally defer.

        This mimics the condition at the fix site in anthropic.py:

            _must_inject_for_new_markers = (
                not inject_tool          # deferred due to frozen prefix
                and injector.has_compressed_content  # but new markers exist
            )
            if inject_tool or _must_inject_for_new_markers:
                tools, _ = apply_session_sticky_ccr_tool(...)
        """
        # Simulate the injector scanning messages that contain a fresh marker
        # (i.e. compression ran and emitted a CCR hash this turn).
        injector = CCRToolInjector(provider="anthropic")
        messages_with_marker = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_bash_x",
                        "content": "[50 items compressed to 5. Retrieve more: hash=abc123def456abc123def456]",
                    }
                ],
            }
        ]
        injector.scan_for_markers(messages_with_marker)
        assert injector.has_compressed_content, "test setup: injector should detect marker"

        # Simulate frozen_message_count > 0 → inject_tool = False  (old deferral)
        inject_tool = False  # deferred
        _must_inject_for_new_markers = not inject_tool and injector.has_compressed_content

        assert _must_inject_for_new_markers, (
            "coupling condition must be True when injection deferred but markers emitted"
        )

        # The tool should be injected via apply_session_sticky_ccr_tool
        with patch("headroom.proxy.helpers.get_session_ccr_tracker") as mock_tracker_fn:
            mock_tracker = MagicMock()
            mock_tracker.has_done_ccr.return_value = False  # first CCR ever
            mock_tracker.get_golden_tool_bytes.return_value = None
            mock_tracker_fn.return_value = mock_tracker

            tools: list[dict] = []
            tools_out, was_injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="session-frozen-test",
                request_id="req-test-1",
                existing_tools=tools,
                has_compressed_content_this_turn=injector.has_compressed_content,
            )

        # headroom_retrieve must now be present
        tool_names = [t.get("name") for t in tools_out]
        assert CCR_TOOL_NAME in tool_names, (
            f"headroom_retrieve not injected when markers emitted and prefix frozen (#1006). "
            f"tools={tool_names}"
        )

    def test_no_injection_when_no_markers_and_frozen_prefix(self):
        """If frozen prefix AND no new markers AND session hasn't done CCR,
        injection should still be skipped (no spurious tool list changes)."""
        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers([{"role": "user", "content": "hello"}])
        assert not injector.has_compressed_content, "test setup: no markers expected"

        inject_tool = False  # deferred due to frozen prefix
        _must_inject_for_new_markers = not inject_tool and injector.has_compressed_content

        assert not _must_inject_for_new_markers, (
            "no injection should be forced when no new markers were emitted"
        )

        with patch("headroom.proxy.helpers.get_session_ccr_tracker") as mock_tracker_fn:
            mock_tracker = MagicMock()
            mock_tracker.has_done_ccr.return_value = False
            mock_tracker.get_golden_tool_bytes.return_value = None
            mock_tracker_fn.return_value = mock_tracker

            tools: list[dict] = []
            tools_out, was_injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="session-frozen-no-markers",
                request_id="req-test-2",
                existing_tools=tools,
                has_compressed_content_this_turn=False,
            )

        tool_names = [t.get("name") for t in tools_out]
        assert CCR_TOOL_NAME not in tool_names, (
            "headroom_retrieve should NOT be injected when no markers and frozen prefix"
        )
