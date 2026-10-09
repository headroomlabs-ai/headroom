"""Regression test for #1006: the proxy must not emit unredeemable CCR markers.

If compression emits a fresh ``<<ccr:hash>>`` marker, the forwarded request must
also carry ``headroom_retrieve`` — a marker the agent has no tool to redeem is
silent data loss.

This used to be enforced by an override *inside* a ``frozen_message_count``
deferral gate. That gate is gone: deferring on the freeze counter dropped a tool
that was already inside the provider-cached prefix, and ``tools`` is the head of
Anthropic's cache key, so every toggle invalidated the whole prefix.
``apply_session_sticky_ccr_tool`` now decides alone, from what the session has
actually forwarded. #1006 is therefore pinned here, at that helper, and the
turn-over-turn cache property is pinned in
``tests/test_proxy_anthropic_cache_stability.py``.

These cases deliberately drive a real ``CCRToolInjector`` marker scan rather than
passing a hand-set boolean, so the marker -> flag -> tool chain stays covered end
to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from headroom.ccr.tool_injection import CCR_TOOL_NAME, CCRToolInjector
from headroom.proxy.helpers import apply_session_sticky_ccr_tool


class TestMarkersImplyRedeemableTool:
    """A marker emitted this turn must arrive with the tool that redeems it."""

    def test_fresh_marker_yields_injected_tool(self):
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
            f"headroom_retrieve not injected when markers were emitted (#1006). tools={tool_names}"
        )

    def test_no_marker_on_session_that_never_compressed_skips_tool(self):
        """The property that makes dropping the freeze gate safe.

        A session with no markers and no CCR history still gets no tool, so
        removing the gate cannot start injecting into non-CCR conversations.
        """
        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers([{"role": "user", "content": "hello"}])
        assert not injector.has_compressed_content, "test setup: no markers expected"

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
                has_compressed_content_this_turn=injector.has_compressed_content,
            )

        tool_names = [t.get("name") for t in tools_out]
        assert CCR_TOOL_NAME not in tool_names, (
            "headroom_retrieve should NOT be injected for a session that never compressed"
        )
