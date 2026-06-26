"""Regression test for #1006: the proxy must not emit unredeemable CCR markers.

When frozen_message_count > 0, the old code deferred headroom_retrieve tool
injection unconditionally — even if compression just emitted NEW <<ccr:hash>>
markers the agent has no tool to redeem.

The fix: if new markers were emitted this turn, override the deferral and inject
the tool (one cache miss is acceptable; silent data loss is not). That decision
lives in ``should_inject_ccr_tool``, which the Anthropic handler calls; this test
pins the decision at that function so removing the override would fail here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from headroom.ccr.tool_injection import CCR_TOOL_NAME, CCRToolInjector
from headroom.proxy.helpers import (
    apply_session_sticky_ccr_tool,
    should_inject_ccr_tool,
)


class TestShouldInjectCCRTool:
    """The decision the handler used to inline. This is where #1006 lived."""

    def test_overrides_deferral_when_markers_emitted(self):
        """Frozen prefix would normally defer, but fresh markers force injection."""
        should_inject, is_override = should_inject_ccr_tool(
            configured_inject_tool=True,
            frozen_message_count=3,
            has_compressed_content=True,
        )
        assert should_inject, "must inject to keep markers redeemable (#1006)"
        assert is_override, "this is the deferral override path"

    def test_defers_when_no_markers(self):
        """Frozen prefix with no new markers stays deferred — no spurious tool."""
        should_inject, is_override = should_inject_ccr_tool(
            configured_inject_tool=True,
            frozen_message_count=3,
            has_compressed_content=False,
        )
        assert not should_inject
        assert not is_override

    def test_injects_normally_without_frozen_prefix(self):
        """No frozen prefix → inject as configured, not via the override path."""
        should_inject, is_override = should_inject_ccr_tool(
            configured_inject_tool=True,
            frozen_message_count=0,
            has_compressed_content=False,
        )
        assert should_inject
        assert not is_override


class TestCCRInjectionEndToEnd:
    """The decision feeds apply_session_sticky_ccr_tool; assert the tool lands."""

    def test_marker_in_frozen_prefix_yields_injected_tool(self):
        # Injector detects a fresh marker, i.e. compression ran this turn.
        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers(
            [
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
        )
        assert injector.has_compressed_content, "test setup: injector should detect marker"

        # Drive the real decision the handler makes under a frozen prefix.
        should_inject, _ = should_inject_ccr_tool(
            configured_inject_tool=True,
            frozen_message_count=3,
            has_compressed_content=injector.has_compressed_content,
        )
        assert should_inject

        with patch("headroom.proxy.helpers.get_session_ccr_tracker") as mock_tracker_fn:
            mock_tracker = MagicMock()
            mock_tracker.has_done_ccr.return_value = False  # first CCR ever
            mock_tracker.get_golden_tool_bytes.return_value = None
            mock_tracker_fn.return_value = mock_tracker

            tools_out, _was_injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="session-frozen-test",
                request_id="req-test-1",
                existing_tools=[],
                has_compressed_content_this_turn=injector.has_compressed_content,
            )

        tool_names = [t.get("name") for t in tools_out]
        assert CCR_TOOL_NAME in tool_names, (
            f"headroom_retrieve not injected when markers emitted and prefix frozen (#1006). "
            f"tools={tool_names}"
        )

    def test_no_marker_in_frozen_prefix_skips_tool(self):
        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers([{"role": "user", "content": "hello"}])
        assert not injector.has_compressed_content, "test setup: no markers expected"

        should_inject, _ = should_inject_ccr_tool(
            configured_inject_tool=True,
            frozen_message_count=3,
            has_compressed_content=injector.has_compressed_content,
        )
        assert not should_inject, "no markers → no forced injection"

        with patch("headroom.proxy.helpers.get_session_ccr_tracker") as mock_tracker_fn:
            mock_tracker = MagicMock()
            mock_tracker.has_done_ccr.return_value = False
            mock_tracker.get_golden_tool_bytes.return_value = None
            mock_tracker_fn.return_value = mock_tracker

            tools_out, _was_injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="session-frozen-no-markers",
                request_id="req-test-2",
                existing_tools=[],
                has_compressed_content_this_turn=False,
            )

        tool_names = [t.get("name") for t in tools_out]
        assert CCR_TOOL_NAME not in tool_names, (
            "headroom_retrieve should NOT be injected when no markers and frozen prefix"
        )
