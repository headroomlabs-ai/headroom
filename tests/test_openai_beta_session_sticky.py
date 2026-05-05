"""Session-sticky `OpenAI-Beta` tests for PR-A6 (P5-50, preps P0-6).

OpenAI's `OpenAI-Beta` follows the same comma-separated convention as
Anthropic's `anthropic-beta`. The proxy auto-injects
`responses_websockets=2026-02-06` on the WS path when absent
(handlers/openai.py:~1711). PR-A6 routes that injection through
`merge_openai_beta` so the client's tokens are preserved and the
auto-injected token is appended deterministically.

The same `SessionBetaTracker` (provider-aware, keyed by
``(provider, session_id)``) backs both providers — one tracker, two
namespaces.
"""

from __future__ import annotations

import threading

import pytest

from headroom.proxy.helpers import (
    SessionBetaTracker,
    _reset_session_beta_tracker_for_test,
    merge_openai_beta,
)

# ---------------------------------------------------------------------------
# Pure helper: `merge_openai_beta`
# ---------------------------------------------------------------------------


def test_merge_helper_pure_function() -> None:
    """Same inputs → same output, no global state."""
    a = merge_openai_beta("a,b", ["c"])
    b = merge_openai_beta("a,b", ["c"])
    assert a == b == "a,b,c"


def test_merge_helper_empty_inputs_returns_empty_string() -> None:
    assert merge_openai_beta(None, []) == ""
    assert merge_openai_beta("", []) == ""


def test_merge_helper_only_client() -> None:
    assert merge_openai_beta("alpha=1,beta=2", []) == "alpha=1,beta=2"


def test_merge_helper_only_headroom() -> None:
    assert merge_openai_beta(None, ["responses_websockets=2026-02-06"]) == (
        "responses_websockets=2026-02-06"
    )


def test_merge_helper_preserves_client_order_appends_headroom() -> None:
    out = merge_openai_beta(
        "alpha=1,beta=2",
        ["responses_websockets=2026-02-06"],
    )
    assert out == "alpha=1,beta=2,responses_websockets=2026-02-06"


def test_merge_helper_no_double_inject_when_already_present() -> None:
    out = merge_openai_beta(
        "responses_websockets=2026-02-06,alpha=1",
        ["responses_websockets=2026-02-06"],
    )
    assert out == "responses_websockets=2026-02-06,alpha=1"


def test_dedupe_case_insensitive_preserves_first_casing() -> None:
    # OpenAI tokens are usually `kebab=value` so casing rarely differs in
    # practice, but the helper's contract is provider-agnostic.
    assert (
        merge_openai_beta("Responses_Websockets=2026-02-06", ["responses_websockets=2026-02-06"])
        == "Responses_Websockets=2026-02-06"
    )


def test_merge_helper_skips_empty_tokens() -> None:
    assert merge_openai_beta("a, ,b", []) == "a,b"
    assert merge_openai_beta("a", ["", "b", "  "]) == "a,b"


def test_test_memory_injection_appends_deterministic_order() -> None:
    """Auto-injected `responses_websockets` appends AFTER client tokens."""
    out = merge_openai_beta("client-token", ["responses_websockets=2026-02-06"])
    assert out == "client-token,responses_websockets=2026-02-06"


# ---------------------------------------------------------------------------
# `SessionBetaTracker` — provider="openai"
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_BETA_HEADER_STICKY", raising=False)
    monkeypatch.delenv("HEADROOM_BETA_TRACKER_MAX_SESSIONS", raising=False)
    _reset_session_beta_tracker_for_test()
    yield
    _reset_session_beta_tracker_for_test()


def test_beta_seen_turn_1_present_in_turn_2_even_if_client_drops() -> None:
    tracker = SessionBetaTracker(max_sessions=10)

    out1 = tracker.record_and_get_sticky_betas(
        provider="openai",
        session_id="s-1",
        client_value="responses_websockets=2026-02-06,extra-beta=1",
    )
    assert out1 == "responses_websockets=2026-02-06,extra-beta=1"

    out2 = tracker.record_and_get_sticky_betas(
        provider="openai",
        session_id="s-1",
        client_value="responses_websockets=2026-02-06",
    )
    assert out2 == "responses_websockets=2026-02-06,extra-beta=1"


def test_client_value_preserved_when_no_injection() -> None:
    tracker = SessionBetaTracker(max_sessions=10)
    out = tracker.record_and_get_sticky_betas(
        provider="openai", session_id="s-1", client_value="alpha,beta"
    )
    assert out == "alpha,beta"


def test_lru_eviction_at_max_sessions() -> None:
    tracker = SessionBetaTracker(max_sessions=2)
    tracker.record_and_get_sticky_betas(provider="openai", session_id="s-1", client_value="a")
    tracker.record_and_get_sticky_betas(provider="openai", session_id="s-2", client_value="b")
    tracker.record_and_get_sticky_betas(provider="openai", session_id="s-1", client_value=None)
    tracker.record_and_get_sticky_betas(provider="openai", session_id="s-3", client_value="c")
    # s-2 evicted.
    out = tracker.record_and_get_sticky_betas(
        provider="openai", session_id="s-2", client_value=None
    )
    assert out == ""


def test_disabled_mode_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_BETA_HEADER_STICKY", "disabled")
    tracker = SessionBetaTracker(max_sessions=10)

    out1 = tracker.record_and_get_sticky_betas(
        provider="openai", session_id="s-1", client_value="alpha"
    )
    assert out1 == "alpha"
    out2 = tracker.record_and_get_sticky_betas(
        provider="openai", session_id="s-1", client_value=None
    )
    assert out2 == ""


def test_thread_safe_concurrent_access() -> None:
    tracker = SessionBetaTracker(max_sessions=10)
    n_threads = 16
    iterations = 50
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            for i in range(iterations):
                tracker.record_and_get_sticky_betas(
                    provider="openai",
                    session_id="shared",
                    client_value=f"o-t{idx}-i{i}",
                )
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    final = tracker.record_and_get_sticky_betas(
        provider="openai", session_id="shared", client_value=None
    )
    final_tokens = set(final.split(","))
    expected = {f"o-t{idx}-i{i}" for idx in range(n_threads) for i in range(iterations)}
    assert expected.issubset(final_tokens)


def test_provider_namespaces_are_independent() -> None:
    tracker = SessionBetaTracker(max_sessions=10)
    tracker.record_and_get_sticky_betas(
        provider="openai", session_id="shared", client_value="o-token"
    )
    tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="shared", client_value="a-token"
    )
    out_openai = tracker.record_and_get_sticky_betas(
        provider="openai", session_id="shared", client_value=None
    )
    out_anth = tracker.record_and_get_sticky_betas(
        provider="anthropic", session_id="shared", client_value=None
    )
    assert out_openai == "o-token"
    assert out_anth == "a-token"


def test_ws_required_token_appended_deterministically() -> None:
    """Mirrors the WS handler logic — record client value, then merge required."""
    tracker = SessionBetaTracker(max_sessions=10)
    sticky = tracker.record_and_get_sticky_betas(
        provider="openai",
        session_id="ws-1",
        client_value="custom-beta=1",
    )
    merged = merge_openai_beta(sticky, ["responses_websockets=2026-02-06"])
    assert merged == "custom-beta=1,responses_websockets=2026-02-06"


def test_ws_required_token_no_double_when_client_already_has_it() -> None:
    tracker = SessionBetaTracker(max_sessions=10)
    sticky = tracker.record_and_get_sticky_betas(
        provider="openai",
        session_id="ws-2",
        client_value="responses_websockets=2026-02-06",
    )
    merged = merge_openai_beta(sticky, ["responses_websockets=2026-02-06"])
    assert merged == "responses_websockets=2026-02-06"
