"""Session-sticky `anthropic-beta` tests for PR-A6 (P5-50, preps P0-6).

Two cache-killer patterns the merge + tracker must defeat:

  1. Mid-session mutation: when memory is enabled the proxy historically
     did an ad-hoc concat of `context-management-2025-06-27` onto the
     client value (anthropic.py:1244-1248). The order varied with the
     client value, breaking byte-stable headers across turns.

  2. Token drop-out across turns: clients (Claude Code, Codex CLI) MAY
     drop a beta token between turn N and turn N+1 even when the proxy
     mutated turn N to add it. The cache hot zone is positional, so the
     next turn's prefix bytes hash differently and the prefix-cache
     read misses.

The fix:

  * `merge_anthropic_beta`: deterministic, pure, order-preserving.
  * `SessionBetaTracker`: bounded LRU keyed by (provider, session_id),
    unioning client tokens with previously-seen tokens.

Operator opt-in `HEADROOM_BETA_HEADER_STICKY=disabled` short-circuits
the tracker (returns the client value verbatim). That mode is loud and
explicit per realignment build constraint #4 — NOT a silent fallback.
"""

from __future__ import annotations

import threading

import pytest

from headroom.proxy.helpers import (
    SessionBetaTracker,
    _reset_session_beta_tracker_for_test,
    get_beta_header_sticky_mode,
    get_beta_tracker_max_sessions,
    get_session_beta_tracker,
    merge_anthropic_beta,
)

# ---------------------------------------------------------------------------
# Pure helper: `merge_anthropic_beta`
# ---------------------------------------------------------------------------


def test_merge_helper_pure_function() -> None:
    """Same inputs → same output, no global state."""
    result1 = merge_anthropic_beta("a,b", ["c"])
    result2 = merge_anthropic_beta("a,b", ["c"])
    assert result1 == result2 == "a,b,c"


def test_merge_helper_empty_inputs_returns_empty_string() -> None:
    assert merge_anthropic_beta(None, []) == ""
    assert merge_anthropic_beta("", []) == ""
    assert merge_anthropic_beta("   ", []) == ""


def test_merge_helper_only_client() -> None:
    assert merge_anthropic_beta("a,b", []) == "a,b"


def test_merge_helper_only_headroom() -> None:
    assert merge_anthropic_beta(None, ["a", "b"]) == "a,b"


def test_merge_helper_preserves_client_order_and_appends_headroom() -> None:
    # Client tokens FIRST in their original order, headroom AFTER in passed order.
    assert (
        merge_anthropic_beta("client-1,client-2", ["headroom-a", "headroom-b"])
        == "client-1,client-2,headroom-a,headroom-b"
    )


def test_dedupe_case_insensitive_preserves_first_casing() -> None:
    # Client casing wins over headroom casing.
    assert (
        merge_anthropic_beta("Context-Management-2025-06-27", ["context-management-2025-06-27"])
        == "Context-Management-2025-06-27"
    )
    # Within client list: first occurrence's casing wins.
    assert merge_anthropic_beta("Foo,foo", []) == "Foo"


def test_test_memory_injection_appends_deterministic_order() -> None:
    """Memory injection appends `context-management-2025-06-27` AFTER client tokens."""
    merged = merge_anthropic_beta(
        "interleaved-thinking-2025-05-14",
        ["context-management-2025-06-27"],
    )
    assert merged == "interleaved-thinking-2025-05-14,context-management-2025-06-27"


def test_merge_helper_skips_empty_tokens() -> None:
    # Whitespace-only entries are dropped.
    assert merge_anthropic_beta("a, ,b", []) == "a,b"
    assert merge_anthropic_beta("a", ["", "b", "  "]) == "a,b"


def test_merge_helper_no_double_inject_when_already_present() -> None:
    # Headroom token already in client value → not re-appended.
    assert (
        merge_anthropic_beta("context-management-2025-06-27", ["context-management-2025-06-27"])
        == "context-management-2025-06-27"
    )


# ---------------------------------------------------------------------------
# `SessionBetaTracker`
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-wide tracker singleton + env flags between tests."""
    monkeypatch.delenv("HEADROOM_BETA_HEADER_STICKY", raising=False)
    monkeypatch.delenv("HEADROOM_BETA_TRACKER_MAX_SESSIONS", raising=False)
    _reset_session_beta_tracker_for_test()
    yield
    _reset_session_beta_tracker_for_test()


def test_beta_seen_turn_1_present_in_turn_2_even_if_client_drops() -> None:
    """The core sticky-on guarantee — token observed in turn 1 stays in turn 2."""
    tracker = SessionBetaTracker(max_sessions=10)

    # Turn 1: client sends two tokens.
    out1 = tracker.record_and_get_sticky_betas(
        provider="anthropic",
        session_id="s-1",
        client_value="prompt-caching-2024-07-31,interleaved-thinking-2025-05-14",
    )
    assert out1 == "prompt-caching-2024-07-31,interleaved-thinking-2025-05-14"

    # Turn 2: client drops `interleaved-thinking-2025-05-14` entirely.
    out2 = tracker.record_and_get_sticky_betas(
        provider="anthropic",
        session_id="s-1",
        client_value="prompt-caching-2024-07-31",
    )
    # Sticky-on: token survives.
    assert out2 == "prompt-caching-2024-07-31,interleaved-thinking-2025-05-14"


def test_client_value_preserved_when_no_injection() -> None:
    """First-turn client value flows through unchanged when no headroom adds."""
    tracker = SessionBetaTracker(max_sessions=10)
    out = tracker.record_and_get_sticky_betas(
        provider="anthropic",
        session_id="s-1",
        client_value="alpha,beta",
    )
    assert out == "alpha,beta"


def test_empty_client_value_returns_empty_string() -> None:
    tracker = SessionBetaTracker(max_sessions=10)
    assert (
        tracker.record_and_get_sticky_betas(
            provider="anthropic", session_id="s-1", client_value=None
        )
        == ""
    )
    assert (
        tracker.record_and_get_sticky_betas(provider="anthropic", session_id="s-1", client_value="")
        == ""
    )


def test_provider_namespaces_are_independent() -> None:
    """Same session_id under two providers keeps independent token sets."""
    tracker = SessionBetaTracker(max_sessions=10)
    tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="shared", client_value="a-only"
    )
    tracker.record_and_get_sticky_betas(
        provider="openai", session_id="shared", client_value="o-only"
    )
    a_out = tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="shared", client_value=None
    )
    o_out = tracker.record_and_get_sticky_betas(
        provider="openai", session_id="shared", client_value=None
    )
    assert a_out == "a-only"
    assert o_out == "o-only"


def test_first_seen_casing_preserved_across_turns() -> None:
    tracker = SessionBetaTracker(max_sessions=10)
    tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="s-1", client_value="Context-Management-2025-06-27"
    )
    out = tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="s-1", client_value="context-management-2025-06-27"
    )
    # Original casing wins.
    assert out == "Context-Management-2025-06-27"


def test_lru_eviction_at_max_sessions() -> None:
    """Bounded LRU pops the oldest session when overflowing."""
    tracker = SessionBetaTracker(max_sessions=2)
    tracker.record_and_get_sticky_betas(provider="anthropic", session_id="s-1", client_value="a")
    tracker.record_and_get_sticky_betas(provider="anthropic", session_id="s-2", client_value="b")
    assert tracker.active_sessions == 2

    # Touch s-1 so s-2 becomes the LRU.
    tracker.record_and_get_sticky_betas(provider="anthropic", session_id="s-1", client_value=None)

    # Add s-3: pops s-2 (oldest by recent access).
    tracker.record_and_get_sticky_betas(provider="anthropic", session_id="s-3", client_value="c")
    assert tracker.active_sessions == 2

    # s-2 was evicted: a fresh request shows no carry-over.
    out = tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="s-2", client_value=None
    )
    assert out == ""


def test_max_sessions_invalid_raises() -> None:
    with pytest.raises(ValueError):
        SessionBetaTracker(max_sessions=0)
    with pytest.raises(ValueError):
        SessionBetaTracker(max_sessions=-1)


def test_disabled_mode_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HEADROOM_BETA_HEADER_STICKY=disabled` returns client value verbatim."""
    monkeypatch.setenv("HEADROOM_BETA_HEADER_STICKY", "disabled")
    tracker = SessionBetaTracker(max_sessions=10)

    # Turn 1: record a token.
    out1 = tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="s-1", client_value="alpha"
    )
    # Disabled mode: client value flows through verbatim, no tracker update.
    assert out1 == "alpha"

    # Turn 2: client drops the token.
    out2 = tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="s-1", client_value=None
    )
    # Disabled mode: empty client value → empty result; NOT sticky.
    assert out2 == ""


def test_disabled_mode_invalid_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown values raise loudly — no silent fallback."""
    monkeypatch.setenv("HEADROOM_BETA_HEADER_STICKY", "yolo")
    with pytest.raises(ValueError, match="HEADROOM_BETA_HEADER_STICKY"):
        get_beta_header_sticky_mode()


def test_max_sessions_env_var_default() -> None:
    assert get_beta_tracker_max_sessions() == 1000


def test_max_sessions_env_var_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_BETA_TRACKER_MAX_SESSIONS", "42")
    assert get_beta_tracker_max_sessions() == 42


def test_max_sessions_env_var_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_BETA_TRACKER_MAX_SESSIONS", "0")
    with pytest.raises(ValueError):
        get_beta_tracker_max_sessions()
    monkeypatch.setenv("HEADROOM_BETA_TRACKER_MAX_SESSIONS", "-3")
    with pytest.raises(ValueError):
        get_beta_tracker_max_sessions()
    monkeypatch.setenv("HEADROOM_BETA_TRACKER_MAX_SESSIONS", "not-int")
    with pytest.raises(ValueError):
        get_beta_tracker_max_sessions()


def test_thread_safe_concurrent_access() -> None:
    """Spawn N threads on same session_id; assert no exceptions and final state correct."""
    tracker = SessionBetaTracker(max_sessions=10)
    n_threads = 16
    iterations = 50
    errors: list[BaseException] = []

    def worker(thread_idx: int) -> None:
        try:
            for i in range(iterations):
                tracker.record_and_get_sticky_betas(
                    provider="anthropic",
                    session_id="shared",
                    client_value=f"t{thread_idx}-i{i}",
                )
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Final state contains every thread's contributed token.
    final = tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="shared", client_value=None
    )
    final_tokens = set(final.split(","))
    expected = {f"t{idx}-i{i}" for idx in range(n_threads) for i in range(iterations)}
    assert expected.issubset(final_tokens)


def test_blank_provider_or_session_raises() -> None:
    tracker = SessionBetaTracker(max_sessions=10)
    with pytest.raises(ValueError):
        tracker.record_and_get_sticky_betas(provider="", session_id="s", client_value="a")
    with pytest.raises(ValueError):
        tracker.record_and_get_sticky_betas(provider="anthropic", session_id="", client_value="a")


def test_singleton_get_session_beta_tracker_returns_same_instance() -> None:
    a = get_session_beta_tracker()
    b = get_session_beta_tracker()
    assert a is b


def test_singleton_reset_replaces_instance() -> None:
    a = get_session_beta_tracker()
    _reset_session_beta_tracker_for_test()
    b = get_session_beta_tracker()
    assert a is not b


def test_memory_injection_appends_deterministic_order() -> None:
    """End-to-end: client value + memory beta token → deterministic merged value.

    Mirrors the ad-hoc concat that the handler used to do but via the
    new merge helper. Order is client first, headroom token after.
    """
    client = "interleaved-thinking-2025-05-14"
    merged = merge_anthropic_beta(client, ["context-management-2025-06-27"])
    assert merged == "interleaved-thinking-2025-05-14,context-management-2025-06-27"

    # When client already had the token, no double-inject.
    client2 = "context-management-2025-06-27,interleaved-thinking-2025-05-14"
    merged2 = merge_anthropic_beta(client2, ["context-management-2025-06-27"])
    assert merged2 == client2
