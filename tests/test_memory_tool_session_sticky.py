"""Session-sticky memory tool injection tests for PR-A7 (closes P0-6).

The cache-killer pattern this guards against (guide §6.3 #2):

  * Mid-session toggle: memory enabled in turn N injects `memory_save` /
    `memory_search` tool definitions into `body["tools"]`. Turn N+1
    disables memory; tool list shrinks; prefix bytes hash differently;
    prefix-cache misses; full prompt re-runs at provider cost.

  * Tool definition drift across deploys: the same logical tool is
    injected but the bytes differ (key insertion order, schema bump,
    description tweak). Even with the tool list intact, prefix bytes
    change.

The fix:

  * `SessionToolTracker`: bounded LRU keyed by (provider, session_id)
    storing GOLDEN tool-definition bytes from the first injection.
    Subsequent turns of that session always replay those bytes — even
    when memory is disabled mid-session (sticky-on per §6.3 #2).

  * `apply_session_sticky_memory_tools`: single coordination point
    used at every memory injection site (Anthropic custom + native,
    OpenAI Chat-Completions + Responses + WS).

Operator opt-in `HEADROOM_TOOL_INJECTION_STICKY=disabled` short-
circuits the tracker (per-turn decision flows through verbatim — the
broken behavior). That mode is loud and explicit per realignment build
constraint #4 — NOT a silent fallback. It exists for diagnostic shadow
tracing and emergency rollback only.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from headroom.proxy.helpers import (
    SessionToolTracker,
    _reset_session_tool_tracker_for_test,
    apply_session_sticky_memory_tools,
    get_session_tool_tracker,
    get_tool_injection_sticky_mode,
    get_tool_tracker_max_sessions,
    serialize_tool_definition_canonical,
)
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "memory_tool_definitions"


# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset env + tracker singleton between tests."""
    monkeypatch.delenv("HEADROOM_TOOL_INJECTION_STICKY", raising=False)
    monkeypatch.delenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", raising=False)
    _reset_session_tool_tracker_for_test()
    yield
    _reset_session_tool_tracker_for_test()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_memory_defs() -> list[dict[str, Any]]:
    h = MemoryHandler(MemoryConfig(enabled=True, inject_tools=True))
    return h.compute_memory_tool_definitions("anthropic")


def _openai_memory_defs() -> list[dict[str, Any]]:
    h = MemoryHandler(MemoryConfig(enabled=True, inject_tools=True))
    return h.compute_memory_tool_definitions("openai")


def _names_in(tools: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for t in tools:
        n = t.get("name") or (t.get("function") or {}).get("name") or t.get("type")
        if n:
            out.add(n)
    return out


# ---------------------------------------------------------------------------
# `SessionToolTracker` direct unit tests
# ---------------------------------------------------------------------------


def test_should_inject_false_for_unknown_session() -> None:
    tracker = SessionToolTracker(max_sessions=10)
    assert tracker.should_inject("anthropic", "s-1") is False


def test_record_then_should_inject_true() -> None:
    tracker = SessionToolTracker(max_sessions=10)
    tracker.record_injection(
        provider="anthropic",
        session_id="s-1",
        tool_name="memory_save",
        tool_definition_bytes=b'{"name":"memory_save"}',
    )
    assert tracker.should_inject("anthropic", "s-1") is True


def test_get_golden_definitions_returns_recorded_bytes() -> None:
    tracker = SessionToolTracker(max_sessions=10)
    tracker.record_injection(
        provider="anthropic",
        session_id="s-1",
        tool_name="memory_save",
        tool_definition_bytes=b'{"name":"memory_save","x":1}',
    )
    tracker.record_injection(
        provider="anthropic",
        session_id="s-1",
        tool_name="memory_search",
        tool_definition_bytes=b'{"name":"memory_search","x":2}',
    )
    golden = tracker.get_golden_definitions("anthropic", "s-1")
    assert golden is not None
    assert [name for name, _ in golden] == ["memory_save", "memory_search"]
    assert golden[0][1] == b'{"name":"memory_save","x":1}'
    assert golden[1][1] == b'{"name":"memory_search","x":2}'


def test_record_first_write_wins_on_duplicate_name() -> None:
    """Re-recording the same tool name is a no-op (prevents drift mid-session)."""
    tracker = SessionToolTracker(max_sessions=10)
    tracker.record_injection(
        provider="anthropic",
        session_id="s-1",
        tool_name="memory_save",
        tool_definition_bytes=b"original",
    )
    tracker.record_injection(
        provider="anthropic",
        session_id="s-1",
        tool_name="memory_save",
        tool_definition_bytes=b"new-bytes",
    )
    golden = tracker.get_golden_definitions("anthropic", "s-1") or []
    assert golden == [("memory_save", b"original")]


def test_provider_isolation_anthropic_vs_openai_same_session_id() -> None:
    """Same session_id under two providers keeps independent state."""
    tracker = SessionToolTracker(max_sessions=10)
    tracker.record_injection(
        provider="anthropic",
        session_id="shared",
        tool_name="memory_save",
        tool_definition_bytes=b"anthropic-bytes",
    )
    tracker.record_injection(
        provider="openai",
        session_id="shared",
        tool_name="memory_save",
        tool_definition_bytes=b"openai-bytes",
    )
    a_golden = tracker.get_golden_definitions("anthropic", "shared") or []
    o_golden = tracker.get_golden_definitions("openai", "shared") or []
    assert a_golden == [("memory_save", b"anthropic-bytes")]
    assert o_golden == [("memory_save", b"openai-bytes")]


def test_lru_eviction_at_max_sessions() -> None:
    """Bounded LRU pops oldest session when overflowing."""
    tracker = SessionToolTracker(max_sessions=2)
    tracker.record_injection(
        provider="anthropic",
        session_id="s-1",
        tool_name="memory_save",
        tool_definition_bytes=b"a",
    )
    tracker.record_injection(
        provider="anthropic",
        session_id="s-2",
        tool_name="memory_save",
        tool_definition_bytes=b"b",
    )
    assert tracker.active_sessions == 2

    # Touch s-1 so s-2 becomes the LRU.
    assert tracker.should_inject("anthropic", "s-1") is True

    # Add s-3: pops s-2.
    tracker.record_injection(
        provider="anthropic",
        session_id="s-3",
        tool_name="memory_save",
        tool_definition_bytes=b"c",
    )
    assert tracker.active_sessions == 2
    assert tracker.should_inject("anthropic", "s-2") is False
    assert tracker.should_inject("anthropic", "s-1") is True
    assert tracker.should_inject("anthropic", "s-3") is True


def test_max_sessions_invalid_raises() -> None:
    with pytest.raises(ValueError):
        SessionToolTracker(max_sessions=0)
    with pytest.raises(ValueError):
        SessionToolTracker(max_sessions=-1)


def test_blank_provider_or_session_raises() -> None:
    tracker = SessionToolTracker(max_sessions=10)
    with pytest.raises(ValueError):
        tracker.should_inject("", "s")
    with pytest.raises(ValueError):
        tracker.should_inject("anthropic", "")
    with pytest.raises(ValueError):
        tracker.record_injection(
            provider="",
            session_id="s",
            tool_name="x",
            tool_definition_bytes=b"y",
        )
    with pytest.raises(ValueError):
        tracker.record_injection(
            provider="anthropic",
            session_id="s",
            tool_name="",
            tool_definition_bytes=b"y",
        )
    with pytest.raises(ValueError):
        tracker.record_injection(
            provider="anthropic",
            session_id="s",
            tool_name="x",
            tool_definition_bytes=b"",
        )


def test_thread_safe_concurrent_access() -> None:
    """N threads on same session: no exceptions, all pinned bytes survive."""
    tracker = SessionToolTracker(max_sessions=10)
    n_threads = 16
    iterations = 50
    errors: list[BaseException] = []

    def worker(thread_idx: int) -> None:
        try:
            for i in range(iterations):
                tracker.record_injection(
                    provider="anthropic",
                    session_id="shared",
                    tool_name=f"t{thread_idx}-i{i}",
                    tool_definition_bytes=f"bytes-{thread_idx}-{i}".encode(),
                )
                # Concurrent reads.
                tracker.should_inject("anthropic", "shared")
                tracker.get_golden_definitions("anthropic", "shared")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    golden = tracker.get_golden_definitions("anthropic", "shared") or []
    names = {name for name, _ in golden}
    expected = {f"t{idx}-i{i}" for idx in range(n_threads) for i in range(iterations)}
    assert expected.issubset(names)


def test_singleton_returns_same_instance() -> None:
    a = get_session_tool_tracker()
    b = get_session_tool_tracker()
    assert a is b


def test_singleton_reset_replaces_instance() -> None:
    a = get_session_tool_tracker()
    _reset_session_tool_tracker_for_test()
    b = get_session_tool_tracker()
    assert a is not b


# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------


def test_max_sessions_env_var_default() -> None:
    assert get_tool_tracker_max_sessions() == 1000


def test_max_sessions_env_var_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", "42")
    assert get_tool_tracker_max_sessions() == 42


def test_max_sessions_env_var_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", "0")
    with pytest.raises(ValueError):
        get_tool_tracker_max_sessions()
    monkeypatch.setenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", "-3")
    with pytest.raises(ValueError):
        get_tool_tracker_max_sessions()
    monkeypatch.setenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", "not-int")
    with pytest.raises(ValueError):
        get_tool_tracker_max_sessions()


def test_sticky_mode_default_enabled() -> None:
    assert get_tool_injection_sticky_mode() == "enabled"


def test_sticky_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_INJECTION_STICKY", "disabled")
    assert get_tool_injection_sticky_mode() == "disabled"


def test_sticky_mode_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_INJECTION_STICKY", "yolo")
    with pytest.raises(ValueError, match="HEADROOM_TOOL_INJECTION_STICKY"):
        get_tool_injection_sticky_mode()


# ---------------------------------------------------------------------------
# `apply_session_sticky_memory_tools` integration
# ---------------------------------------------------------------------------


def test_injection_in_turn_1_repeats_in_turn_2_same_session_anthropic() -> None:
    """Core sticky-on guarantee for Anthropic provider."""
    defs = _anthropic_memory_defs()
    assert len(defs) >= 2

    # Turn 1: memory enabled — first-time injection.
    tools1, was1 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="s-1",
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was1 is True
    names1 = _names_in(tools1)
    assert "memory_save" in names1
    assert "memory_search" in names1

    # Turn 2: memory STILL enabled — bytes match turn 1.
    tools2, was2 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="s-1",
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was2 is True
    # Same set of memory tools.
    assert _names_in(tools2) == names1


def test_injection_in_turn_1_repeats_in_turn_2_same_session_openai() -> None:
    defs = _openai_memory_defs()
    assert len(defs) >= 2

    tools1, was1 = apply_session_sticky_memory_tools(
        provider="openai",
        session_id="o-1",
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was1 is True
    names1 = _names_in(tools1)
    assert "memory_save" in names1
    assert "memory_search" in names1

    tools2, was2 = apply_session_sticky_memory_tools(
        provider="openai",
        session_id="o-1",
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was2 is True
    assert _names_in(tools2) == names1


def test_byte_equal_tool_definition_across_turns() -> None:
    """The injected tool list serialization is BYTE-equal turn 1 vs turn 2.

    Pin the bytes via the golden snapshot fixture.
    """
    defs = _anthropic_memory_defs()

    tools1, _ = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="bytestable-1",
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    tools2, _ = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="bytestable-1",
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )

    # Byte-equality (the cache-stable invariant).
    bytes1 = b"".join(serialize_tool_definition_canonical(t) for t in tools1)
    bytes2 = b"".join(serialize_tool_definition_canonical(t) for t in tools2)
    assert bytes1 == bytes2

    # Match the golden fixture (computed via the pinned helper).
    fixture = json.loads((FIXTURES_DIR / "anthropic.json").read_text())
    fixture_bytes = b"".join(serialize_tool_definition_canonical(t) for t in fixture["tools"])
    assert bytes1 == fixture_bytes


def test_byte_equal_tool_definition_across_turns_openai() -> None:
    defs = _openai_memory_defs()
    tools1, _ = apply_session_sticky_memory_tools(
        provider="openai",
        session_id="bytestable-2",
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    tools2, _ = apply_session_sticky_memory_tools(
        provider="openai",
        session_id="bytestable-2",
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    bytes1 = b"".join(serialize_tool_definition_canonical(t) for t in tools1)
    bytes2 = b"".join(serialize_tool_definition_canonical(t) for t in tools2)
    assert bytes1 == bytes2

    fixture = json.loads((FIXTURES_DIR / "openai.json").read_text())
    fixture_bytes = b"".join(serialize_tool_definition_canonical(t) for t in fixture["tools"])
    assert bytes1 == fixture_bytes


def test_memory_disabled_after_inject_still_injects() -> None:
    """Turn 1 injects; turn 2 has memory disabled; turn 2 still injects golden bytes."""
    defs = _anthropic_memory_defs()

    # Turn 1: memory enabled.
    tools1, was1 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="s-cancel-1",
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was1 is True
    names1 = _names_in(tools1)

    # Turn 2: memory DISABLED for this turn (e.g. inject_tools flag flipped).
    # `inject_this_turn=False` AND `memory_tools_to_inject=[]` mimic the
    # caller's behavior under disabled-memory: nothing fresh to inject.
    tools2, was2 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="s-cancel-1",
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=[],
        inject_this_turn=False,
    )
    # Sticky-on: the golden bytes are still injected even though caller
    # passed nothing this turn.
    assert was2 is True
    assert _names_in(tools2) == names1

    # Bytes match.
    bytes1 = b"".join(serialize_tool_definition_canonical(t) for t in tools1)
    bytes2 = b"".join(serialize_tool_definition_canonical(t) for t in tools2)
    assert bytes1 == bytes2


def test_different_sessions_independent() -> None:
    """Session A injects; session B doesn't; verify isolation."""
    defs = _anthropic_memory_defs()

    # Session A: inject.
    _, was_a = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="A",
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was_a is True

    # Session B: NO inject this turn, no prior history.
    tools_b, was_b = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="B",
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=[],
        inject_this_turn=False,
    )
    assert was_b is False
    assert _names_in(tools_b) == set()

    # Session A still has its golden state.
    tools_a2, was_a2 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="A",
        request_id="r-3",
        existing_tools=[],
        memory_tools_to_inject=[],
        inject_this_turn=False,
    )
    assert was_a2 is True
    assert "memory_save" in _names_in(tools_a2)


def test_disabled_mode_passes_through_per_turn_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`HEADROOM_TOOL_INJECTION_STICKY=disabled` reverts to per-turn behavior.

    This is the broken behavior — explicit operator opt-in only. Turn 1
    injects; turn 2 with `inject_this_turn=False` does NOT replay (the
    sticky guarantee is bypassed).
    """
    monkeypatch.setenv("HEADROOM_TOOL_INJECTION_STICKY", "disabled")
    defs = _anthropic_memory_defs()

    tools1, was1 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="s-disabled",
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was1 is True
    assert "memory_save" in _names_in(tools1)

    # Turn 2: caller says don't inject. Disabled mode → tracker bypassed.
    tools2, was2 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="s-disabled",
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=[],
        inject_this_turn=False,
    )
    assert was2 is False
    assert _names_in(tools2) == set()


def test_existing_tool_with_memory_name_not_double_injected() -> None:
    """If client already has a tool by the same name, do not re-append it."""
    defs = _anthropic_memory_defs()
    client_tools: list[dict[str, Any]] = [
        {"name": "memory_save", "description": "client's own", "input_schema": {}}
    ]

    tools, _ = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id="s-dedup",
        request_id="r-1",
        existing_tools=client_tools,
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    # Exactly one tool named "memory_save".
    save_count = sum(1 for t in tools if t.get("name") == "memory_save")
    assert save_count == 1


def test_no_session_id_falls_back_to_per_turn(caplog: pytest.LogCaptureFixture) -> None:
    """`session_id=None` (e.g. WS pre-session) bypasses the tracker."""
    defs = _anthropic_memory_defs()
    tools1, was1 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id=None,
        request_id="r-1",
        existing_tools=[],
        memory_tools_to_inject=defs,
        inject_this_turn=True,
    )
    assert was1 is True
    assert "memory_save" in _names_in(tools1)

    # Without session_id we can't replay across turns.
    tools2, was2 = apply_session_sticky_memory_tools(
        provider="anthropic",
        session_id=None,
        request_id="r-2",
        existing_tools=[],
        memory_tools_to_inject=[],
        inject_this_turn=False,
    )
    assert was2 is False


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unsupported provider"):
        apply_session_sticky_memory_tools(
            provider="gemini",  # type: ignore[arg-type]
            session_id="s",
            request_id="r",
            existing_tools=[],
            memory_tools_to_inject=[],
            inject_this_turn=True,
        )


# ---------------------------------------------------------------------------
# Golden fixture pinning
# ---------------------------------------------------------------------------


def test_anthropic_fixture_matches_helper_output() -> None:
    """Fixture file pins the canonical bytes — regenerate if this fails."""
    fixture = json.loads((FIXTURES_DIR / "anthropic.json").read_text())
    assert fixture["provider"] == "anthropic"

    helper_defs = _anthropic_memory_defs()
    helper_bytes = b"".join(serialize_tool_definition_canonical(t) for t in helper_defs)
    fixture_bytes = b"".join(serialize_tool_definition_canonical(t) for t in fixture["tools"])
    assert helper_bytes == fixture_bytes, (
        "Anthropic memory tool definitions drifted from golden fixture. "
        "If intentional, regenerate "
        "tests/fixtures/memory_tool_definitions/anthropic.json. "
        f"Helper SHA-256: {hashlib.sha256(helper_bytes).hexdigest()} "
        f"Fixture SHA-256: {hashlib.sha256(fixture_bytes).hexdigest()}"
    )


def test_openai_fixture_matches_helper_output() -> None:
    fixture = json.loads((FIXTURES_DIR / "openai.json").read_text())
    assert fixture["provider"] == "openai"

    helper_defs = _openai_memory_defs()
    helper_bytes = b"".join(serialize_tool_definition_canonical(t) for t in helper_defs)
    fixture_bytes = b"".join(serialize_tool_definition_canonical(t) for t in fixture["tools"])
    assert helper_bytes == fixture_bytes, (
        "OpenAI memory tool definitions drifted from golden fixture. "
        "If intentional, regenerate "
        "tests/fixtures/memory_tool_definitions/openai.json. "
        f"Helper SHA-256: {hashlib.sha256(helper_bytes).hexdigest()} "
        f"Fixture SHA-256: {hashlib.sha256(fixture_bytes).hexdigest()}"
    )
