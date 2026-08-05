from __future__ import annotations

import pytest

from headroom.proxy.tool_injection_tracker import SessionToolTracker


def test_tracker_reports_unknown_session_as_not_injected() -> None:
    tracker = SessionToolTracker(max_sessions=10)

    assert tracker.should_inject("anthropic", "s-1") is False


def test_tracker_records_and_returns_golden_bytes_in_order() -> None:
    tracker = SessionToolTracker(max_sessions=10)

    tracker.record_injection("anthropic", "s-1", "memory_save", b"save")
    tracker.record_injection("anthropic", "s-1", "memory_search", b"search")

    assert tracker.should_inject("anthropic", "s-1") is True
    assert tracker.get_golden_definitions("anthropic", "s-1") == [
        ("memory_save", b"save"),
        ("memory_search", b"search"),
    ]


def test_tracker_first_write_wins_per_tool_name() -> None:
    tracker = SessionToolTracker(max_sessions=10)

    tracker.record_injection("anthropic", "s-1", "memory_save", b"original")
    tracker.record_injection("anthropic", "s-1", "memory_save", b"drift")

    assert tracker.get_golden_definitions("anthropic", "s-1") == [("memory_save", b"original")]


def test_tracker_keeps_provider_namespaces_independent() -> None:
    tracker = SessionToolTracker(max_sessions=10)

    tracker.record_injection("anthropic", "shared", "memory_save", b"anthropic")
    tracker.record_injection("openai", "shared", "memory_save", b"openai")

    assert tracker.get_golden_definitions("anthropic", "shared") == [("memory_save", b"anthropic")]
    assert tracker.get_golden_definitions("openai", "shared") == [("memory_save", b"openai")]


def test_tracker_evicts_least_recently_used_session() -> None:
    tracker = SessionToolTracker(max_sessions=2)
    tracker.record_injection("anthropic", "s-1", "memory_save", b"a")
    tracker.record_injection("anthropic", "s-2", "memory_save", b"b")

    assert tracker.should_inject("anthropic", "s-1") is True
    tracker.record_injection("anthropic", "s-3", "memory_save", b"c")

    assert tracker.active_sessions == 2
    assert tracker.should_inject("anthropic", "s-1") is True
    assert tracker.should_inject("anthropic", "s-2") is False
    assert tracker.should_inject("anthropic", "s-3") is True


def test_tracker_validates_constructor_and_record_inputs() -> None:
    with pytest.raises(ValueError, match="max_sessions"):
        SessionToolTracker(max_sessions=0)

    tracker = SessionToolTracker(max_sessions=10)
    with pytest.raises(ValueError, match="provider"):
        tracker.should_inject("", "s-1")
    with pytest.raises(ValueError, match="session_id"):
        tracker.get_golden_definitions("anthropic", "")
    with pytest.raises(ValueError, match="tool_name"):
        tracker.record_injection("anthropic", "s-1", "", b"bytes")
    with pytest.raises(ValueError, match="tool_definition_bytes"):
        tracker.record_injection("anthropic", "s-1", "memory_save", b"")
