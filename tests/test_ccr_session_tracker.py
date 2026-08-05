from __future__ import annotations

import pytest

from headroom.proxy.ccr_session_tracker import SessionCcrTracker


def test_tracker_reports_unknown_session_as_not_done() -> None:
    tracker = SessionCcrTracker(max_sessions=10)

    assert tracker.has_done_ccr("anthropic", "s-1") is False
    assert tracker.get_golden_tool_bytes("anthropic", "s-1") is None


def test_tracker_records_monotonic_done_state_and_golden_bytes() -> None:
    tracker = SessionCcrTracker(max_sessions=10)

    tracker.record_ccr_done("anthropic", "s-1", b"first")
    tracker.record_ccr_done("anthropic", "s-1", b"second")

    assert tracker.has_done_ccr("anthropic", "s-1") is True
    assert tracker.get_golden_tool_bytes("anthropic", "s-1") == b"first"


def test_tracker_keeps_provider_namespaces_independent() -> None:
    tracker = SessionCcrTracker(max_sessions=10)

    tracker.record_ccr_done("anthropic", "shared", b"anthropic")
    tracker.record_ccr_done("openai", "shared", b"openai")

    assert tracker.get_golden_tool_bytes("anthropic", "shared") == b"anthropic"
    assert tracker.get_golden_tool_bytes("openai", "shared") == b"openai"


def test_tracker_evicts_least_recently_used_session() -> None:
    tracker = SessionCcrTracker(max_sessions=2)
    tracker.record_ccr_done("anthropic", "s-1", b"a")
    tracker.record_ccr_done("anthropic", "s-2", b"b")

    assert tracker.has_done_ccr("anthropic", "s-1") is True
    tracker.record_ccr_done("anthropic", "s-3", b"c")

    assert tracker.active_sessions == 2
    assert tracker.has_done_ccr("anthropic", "s-1") is True
    assert tracker.has_done_ccr("anthropic", "s-2") is False
    assert tracker.has_done_ccr("anthropic", "s-3") is True


def test_tracker_reset_clears_state() -> None:
    tracker = SessionCcrTracker(max_sessions=10)
    tracker.record_ccr_done("anthropic", "s-1", b"bytes")

    tracker.reset()

    assert tracker.active_sessions == 0
    assert tracker.has_done_ccr("anthropic", "s-1") is False


def test_tracker_validates_inputs() -> None:
    with pytest.raises(ValueError, match="max_sessions"):
        SessionCcrTracker(max_sessions=0)

    tracker = SessionCcrTracker(max_sessions=10)
    with pytest.raises(ValueError, match="provider"):
        tracker.has_done_ccr("", "s-1")
    with pytest.raises(ValueError, match="session_id"):
        tracker.get_golden_tool_bytes("anthropic", "")
    with pytest.raises(ValueError, match="golden_tool_bytes"):
        tracker.record_ccr_done("anthropic", "s-1", b"")
