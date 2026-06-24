"""Tests for PR-G2 — RTK ``tokens_saved`` data-plane wiring.

Phase G of the Headroom realignment retires the dead ``tokens_saved_rtk``
field by sourcing it from RTK's own stats endpoint (``rtk gain --format
json`` via :func:`headroom.proxy.helpers._get_rtk_stats`) and writing the
per-call delta into ``HeadroomContribution.tokens_saved_rtk``.

PR-G2 remediation (C1): the tracker reads the SESSION-incremental
``session.tokens_saved`` field of the helper payload, NOT the raw
``lifetime_tokens_saved`` counter. The helper de-baselines per proxy
session at startup, so the first poll after process startup correctly
reads 0 instead of the entire pre-Headroom RTK history.

These tests pin the wiring:

1. The delta is computed correctly across two consecutive
   :meth:`update_contribution` calls (monotonic session counter advances).
2. ``tokens_saved_rtk`` is exactly zero when ``_get_rtk_stats()`` returns
   ``None`` (RTK not installed / not selected).
3. ``_last_rtk_tokens_saved`` advances monotonically; deltas are not
   replayed across calls when the session counter does not move.
4. First poll reads 0 when the helper reports a fresh session baseline
   (the C1 regression fix — previously this poll emitted the entire RTK
   lifetime as a phantom delta).

Realignment build constraints honored:

- No silent fallback: a transient ``_get_rtk_stats()`` exception is
  structured-logged and yields ``tokens_saved_rtk = 0`` (test 4).
- Configurable: ``HEADROOM_RTK_WIRING=disabled`` opts the polling out and
  produces a clean zero, exercised by ``test_disabled_env_returns_zero``.
- Structured logs: each failure path emits a ``event=…`` line; the
  ``caplog`` assertions below pin the log payload so the "no silent
  fallback" constraint is verified.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

import headroom.subscription.tracker as tracker_module
from headroom.subscription.tracker import SubscriptionTracker


def _build_tracker(monkeypatch: pytest.MonkeyPatch) -> SubscriptionTracker:
    """Construct a tracker with persistence + multi-worker lock disabled.

    Tests use ``_build_tracker`` to keep persistence side effects out of
    unit tests and to force the RTK poll lock to "owner" so polling runs.
    """

    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    monkeypatch.setattr(SubscriptionTracker, "_try_acquire_rtk_poll_lock", lambda self: True)
    return SubscriptionTracker(enabled=True)


def _session_payload(tokens_saved: int, *, lifetime: int | None = None) -> dict[str, Any]:
    """Build a stats payload mimicking ``_get_context_tool_stats``.

    The tracker reads ``session.tokens_saved``. We always include the
    lifetime field so we can verify the tracker no longer reads it.
    """

    if lifetime is None:
        lifetime = tokens_saved + 50_000  # arbitrary pre-Headroom history
    return {
        "tokens_saved": tokens_saved,  # session-incremental (canonical)
        "lifetime_tokens_saved": lifetime,
        "session": {"tokens_saved": tokens_saved},
        "lifetime": {"tokens_saved": lifetime},
    }


def _stub_rtk_stats(
    monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, Any] | None]
) -> list[int]:
    """Stub ``_get_rtk_stats`` to return ``payloads`` in order.

    Returns a counter list (mutated by the stub) so callers can assert the
    number of polls.
    """

    call_count: list[int] = [0]

    def fake_get_rtk_stats() -> dict[str, Any] | None:
        idx = call_count[0]
        call_count[0] += 1
        if idx >= len(payloads):
            return payloads[-1]
        return payloads[idx]

    monkeypatch.setattr(
        "headroom.proxy.helpers._get_rtk_stats",
        fake_get_rtk_stats,
    )
    return call_count


# ---------------------------------------------------------------------------
# Test 1 — delta computed correctly across two consecutive polls
# ---------------------------------------------------------------------------


def test_tokens_saved_rtk_populated_from_session_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call seeds the baseline at the session counter, not lifetime.

    PR-G2 remediation (C1): previously the tracker read
    ``lifetime_tokens_saved`` and emitted the entire pre-Headroom RTK
    history as a phantom delta on the first poll. After the C1 fix the
    tracker reads ``session.tokens_saved`` which the helper has already
    de-baselined per proxy session.
    """

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            _session_payload(tokens_saved=100, lifetime=50_100),
            _session_payload(tokens_saved=175, lifetime=50_175),
        ],
    )

    # First call — session counter is 100 (50 000 lifetime history was
    # rebaselined by the helper at proxy startup, so we DON'T see it).
    tracker.update_contribution()
    contribution_after_first = tracker._state.contribution.tokens_saved_rtk
    assert contribution_after_first == 100
    assert tracker._last_rtk_tokens_saved == 100

    # Second call — delta is 175 - 100 = 75; cumulative contribution = 175.
    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 175
    assert tracker._last_rtk_tokens_saved == 175


def test_first_poll_zero_when_session_baseline_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1 fix verification: a freshly-baselined session yields zero on first poll.

    The helper's session baseline is captured at proxy startup. A brand
    new proxy with no RTK invocations since startup reports
    ``session.tokens_saved == 0`` even though ``lifetime_tokens_saved``
    may be enormous (months of accumulated RTK history). The tracker must
    NOT emit the lifetime as a phantom delta.
    """

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            # Pre-Headroom lifetime = 50 000 tokens. Helper rebaselines at
            # startup so session = 0.
            _session_payload(tokens_saved=0, lifetime=50_000),
        ],
    )

    tracker.update_contribution()

    assert tracker._state.contribution.tokens_saved_rtk == 0, (
        "first poll must NOT emit pre-Headroom RTK history as a phantom delta"
    )
    assert tracker._last_rtk_tokens_saved == 0


def test_delta_computed_correctly_across_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three consecutive polls — each adds only the new RTK delta."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            _session_payload(tokens_saved=0),  # baseline at zero
            _session_payload(tokens_saved=50),
            _session_payload(tokens_saved=250),
        ],
    )

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 50
    assert tracker._last_rtk_tokens_saved == 50

    tracker.update_contribution()
    # 50 + (250 - 50) = 250 cumulative; delta on the third call was 200.
    assert tracker._state.contribution.tokens_saved_rtk == 250
    assert tracker._last_rtk_tokens_saved == 250


# ---------------------------------------------------------------------------
# Test 2 — ``tokens_saved_rtk = 0`` when stats endpoint returns None
# ---------------------------------------------------------------------------


def test_rtk_stats_none_yields_zero_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    """No RTK selected / installed — contribution stays at zero, no throw."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(monkeypatch, [None, None])

    tracker.update_contribution()
    tracker.update_contribution()

    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0


# ---------------------------------------------------------------------------
# Test 3 — monotonic advancement; no replay on flat poll
# ---------------------------------------------------------------------------


def test_last_rtk_advances_monotonically(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two polls returning the same session total contribute exactly once."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            _session_payload(tokens_saved=42),
            _session_payload(tokens_saved=42),  # no movement
            _session_payload(tokens_saved=42),  # still no movement
        ],
    )

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 42
    assert tracker._last_rtk_tokens_saved == 42

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 42  # unchanged
    assert tracker._last_rtk_tokens_saved == 42

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 42
    assert tracker._last_rtk_tokens_saved == 42


def test_counter_regression_rebaselines_without_negative_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper rebaselines the session counter — re-baseline, do not subtract."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            _session_payload(tokens_saved=500),
            _session_payload(tokens_saved=100),  # regression!
            _session_payload(tokens_saved=150),
        ],
    )

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 500
    assert tracker._last_rtk_tokens_saved == 500

    tracker.update_contribution()
    # Regression: contribution stays at 500 (no negative subtraction).
    assert tracker._state.contribution.tokens_saved_rtk == 500
    # Baseline now points at the new (smaller) session total so subsequent
    # polls can compute a meaningful delta.
    assert tracker._last_rtk_tokens_saved == 100

    tracker.update_contribution()
    # 150 - 100 = 50 new delta; contribution = 500 + 50 = 550.
    assert tracker._state.contribution.tokens_saved_rtk == 550
    assert tracker._last_rtk_tokens_saved == 150


# ---------------------------------------------------------------------------
# Test 4 — transient exception in the stats endpoint
# ---------------------------------------------------------------------------


def test_rtk_stats_exception_zero_delta_no_throw_with_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raised ``_get_rtk_stats()`` is caught, structured-logged, yields 0.

    PR-G2 remediation (H3): pins the loud-log requirement so the
    "no silent fallback" constraint is verified.
    """

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)

    def boom() -> dict[str, Any] | None:
        raise RuntimeError("transient subprocess failure")

    monkeypatch.setattr("headroom.proxy.helpers._get_rtk_stats", boom)

    caplog.set_level(logging.WARNING, logger="headroom.subscription.tracker")

    # Must not raise.
    tracker.update_contribution()

    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0
    assert any(
        "event=subscription_rtk_stats_fetch_failed" in rec.getMessage() for rec in caplog.records
    ), "expected structured log on RTK stats fetch failure"


# ---------------------------------------------------------------------------
# Test 5 — explicit env-var opt-out
# ---------------------------------------------------------------------------


def test_disabled_env_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HEADROOM_RTK_WIRING=disabled`` skips the poll entirely."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.setenv(tracker_module._RTK_WIRING_ENV, "disabled")

    polls = _stub_rtk_stats(
        monkeypatch,
        [_session_payload(tokens_saved=999)],
    )

    tracker.update_contribution()

    # Stats endpoint never called when wiring is disabled.
    assert polls[0] == 0
    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0


# ---------------------------------------------------------------------------
# Test 6 — explicit override from caller (back-compat for callers that
# already know the RTK delta out-of-band).
# ---------------------------------------------------------------------------


def test_explicit_rtk_override_skips_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller-supplied ``tokens_saved_rtk`` short-circuits the poll."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)

    polls = _stub_rtk_stats(monkeypatch, [_session_payload(tokens_saved=999)])

    tracker.update_contribution(tokens_saved_rtk=17)

    # Stats endpoint not consulted when the caller passes an explicit value.
    assert polls[0] == 0
    assert tracker._state.contribution.tokens_saved_rtk == 17
    assert tracker._last_rtk_tokens_saved == 0


# ---------------------------------------------------------------------------
# Test 7 — cli_filtering decoupled from rtk
# ---------------------------------------------------------------------------


def test_cli_filtering_no_longer_mirrors_rtk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-PR-G2 bug: ``cli_filtering`` and ``rtk`` were always equal.

    After PR-G2 they are independent counters fed by separate sources.
    """

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(monkeypatch, [_session_payload(tokens_saved=25)])

    tracker.update_contribution(tokens_saved_cli_filtering=8)

    assert tracker._state.contribution.tokens_saved_cli_filtering == 8
    # rtk comes from the polled delta, not from cli_filtering.
    assert tracker._state.contribution.tokens_saved_rtk == 25


# ---------------------------------------------------------------------------
# Test 8 (H1) — invalid HEADROOM_RTK_WIRING fails loudly at startup
# ---------------------------------------------------------------------------


def test_garbage_wiring_env_raises_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """PR-G2 remediation (H1, M4): bad env value crashes startup loudly.

    Previously the typo would be silently swallowed at every
    ``update_contribution`` call. Now :func:`configure_subscription_tracker`
    validates eagerly and raises ``ValueError``.
    """

    monkeypatch.setenv(tracker_module._RTK_WIRING_ENV, "garbage")
    # Reset the singleton so configure() actually runs the validator.
    monkeypatch.setattr(tracker_module, "_tracker_instance", None)

    with pytest.raises(ValueError, match="HEADROOM_RTK_WIRING"):
        tracker_module.configure_subscription_tracker(enabled=True)


def test_garbage_wiring_env_logs_loudly_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If env is flipped to garbage AFTER startup, runtime path emits ERROR.

    This is the defence-in-depth tier — startup-validation is the primary
    barrier (test above) but a env-var rotation could still flip the value
    mid-run.
    """

    tracker = _build_tracker(monkeypatch)
    # Set garbage AFTER tracker construction so the constructor doesn't see it.
    monkeypatch.setenv(tracker_module._RTK_WIRING_ENV, "garbage")

    caplog.set_level(logging.ERROR, logger="headroom.subscription.tracker")

    tracker.update_contribution()

    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert any(
        rec.levelno >= logging.ERROR and "event=subscription_rtk_invalid_env" in rec.getMessage()
        for rec in caplog.records
    ), "expected ERROR-level structured log on invalid HEADROOM_RTK_WIRING"


# ---------------------------------------------------------------------------
# Test 9 (C2) — restart-seeding behavior: no phantom delta on second process
# ---------------------------------------------------------------------------


def test_restart_does_not_emit_phantom_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """PR-G2 remediation (C2): post-restart first poll must not phantom.

    Scenario:
    1. Tracker A runs, accumulates ``c.tokens_saved_rtk = 100``, persists.
    2. Process restarts (tracker B loads from disk).
    3. First poll on tracker B: helper returns ``session.tokens_saved = 5``
       (small new value since startup). Delta = 5 - 0 = 5. Cumulative =
       100 + 5 = 105. NOT 100 + 50 000 (lifetime).

    The C1 fix (read session, not lifetime) inherently dissolves this
    because the helper rebaselines session counters at every proxy
    startup. This test verifies that property.
    """

    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    monkeypatch.setattr(SubscriptionTracker, "_try_acquire_rtk_poll_lock", lambda self: True)

    persist_path = tmp_path / "state.json"

    # Phase 1 — tracker A runs and persists state with non-zero counters.
    _stub_rtk_stats(
        monkeypatch,
        [_session_payload(tokens_saved=100, lifetime=50_100)],
    )
    tracker_a = SubscriptionTracker(persist_path=persist_path, enabled=True)
    tracker_a.update_contribution()
    assert tracker_a._state.contribution.tokens_saved_rtk == 100
    tracker_a._persist_state()

    # Phase 2 — simulate process restart. New tracker loads state from
    # disk. Helper rebaselines (session counter starts fresh at 5 — only
    # one RTK invocation since restart).
    _stub_rtk_stats(
        monkeypatch,
        [_session_payload(tokens_saved=5, lifetime=50_105)],
    )
    tracker_b = SubscriptionTracker(persist_path=persist_path, enabled=True)
    # Loaded from disk.
    assert tracker_b._state.contribution.tokens_saved_rtk == 100
    # Tracker B's _last_rtk_tokens_saved starts at 0 (correct — the
    # session baseline was just re-pinned in the helper).
    assert tracker_b._last_rtk_tokens_saved == 0

    tracker_b.update_contribution()
    # 100 (loaded) + 5 (new session delta) = 105. NOT 50 100 + anything.
    assert tracker_b._state.contribution.tokens_saved_rtk == 105


# ---------------------------------------------------------------------------
# Test 10 (M2 + M3) — legacy state file migration
# ---------------------------------------------------------------------------


def test_legacy_state_migrates_rtk_from_cli_filtering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PR-G2 remediation (M2): pre-G2 state has no ``rtk_raw`` key.

    Pre-G2 the ``rtk`` field silently mirrored ``cli_filtering`` (the
    exact bug PR-G2 retires). When loading a legacy file we treat the
    aliased ``rtk`` value as the authoritative rtk_raw so historical
    accumulation isn't silently zeroed. A migration log line is emitted.
    """

    import json

    persist_path = tmp_path / "legacy.json"
    persist_path.write_text(
        json.dumps(
            {
                "contribution": {
                    "tokens_submitted": 50,
                    "tokens_saved": {
                        "proxy_compression": 10,
                        "cli_filtering": 42,
                        "rtk": 42,  # pre-G2 alias
                        "cache_reads": 3,
                        # NO rtk_raw / cli_filtering_raw keys (legacy)
                    },
                    "savings_usd": {"compression": 0.0, "cache": 0.0},
                },
                "poll_count": 7,
            }
        )
    )

    caplog.set_level(logging.INFO, logger="headroom.subscription.tracker")

    tracker = SubscriptionTracker(persist_path=persist_path, enabled=True)

    # Legacy ``rtk == cli_filtering`` got carried forward into rtk_raw.
    assert tracker._state.contribution.tokens_saved_rtk == 42
    assert tracker._state.contribution.tokens_saved_cli_filtering == 42
    # Migration log emitted.
    assert any(
        "event=subscription_state_legacy_load" in rec.getMessage() for rec in caplog.records
    ), "expected legacy migration structured log"


# ---------------------------------------------------------------------------
# Test 11 (H2) — helper logs structured warning on subprocess failure
# ---------------------------------------------------------------------------


def test_rtk_subprocess_failure_logs_structured_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-G2 remediation (H2): synthetic-zero path must log loudly.

    Without this, a broken RTK and a healthy "0 tokens saved" RTK are
    indistinguishable at the tracker layer.

    Implementation note: earlier attempts used pytest's ``caplog`` fixture
    (both scoped to ``logger="headroom.proxy"`` and root-level capture).
    Both passed locally but failed in CI — likely a logger-propagation /
    handler-config difference in the CI test harness. The robust approach
    is to mock ``_helpers.logger.warning`` directly: when the production
    code calls ``logger.warning(...)`` the mock intercepts regardless of
    propagation, formatters, or handler order.
    """

    from unittest.mock import MagicMock

    import headroom.rtk as _rtk
    from headroom.proxy import helpers as _helpers

    # Point get_rtk_path at a definitely-nonexistent absolute path so the
    # real ``subprocess.run`` raises FileNotFoundError → except branch
    # fires the structured warning.
    monkeypatch.setattr(_rtk, "get_rtk_path", lambda: "/nonexistent/headroom-test-rtk")

    mock_warning = MagicMock()
    monkeypatch.setattr(_helpers.logger, "warning", mock_warning)

    payload = _helpers._read_rtk_lifetime_stats()
    assert payload is not None
    assert payload["scope"] == "global"
    assert payload["tokens_saved"] == 0

    # Concatenate all warning call args so the failure message shows what
    # the helper actually emitted (debug aid for CI flakes).
    all_warning_calls = " ".join(str(call) for call in mock_warning.call_args_list)
    assert "event=rtk_stats_subprocess_failed" in all_warning_calls, (
        f"expected structured warning; actual logger.warning calls: {mock_warning.call_args_list}"
    )


# ---------------------------------------------------------------------------
# Test 12 (C3) — multi-worker poll deduplication via file lock
# ---------------------------------------------------------------------------


def test_multi_worker_only_one_polls(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """PR-G2 remediation (C3): two trackers sharing a state path elect one owner.

    The owner polls; the non-owner returns 0 from ``_poll_rtk_delta``.
    Without this gate each worker would add the same RTK delta to its
    own ``c.tokens_saved_rtk``, inflating dashboard savings by N× workers.
    """

    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)

    # Both trackers share a state directory so they share the lock file.
    persist_path = tmp_path / "state.json"
    lock_path = tmp_path / ".rtk_poll_lock"
    monkeypatch.setenv(tracker_module._RTK_POLL_LOCK_ENV, str(lock_path))

    _stub_rtk_stats(
        monkeypatch,
        [_session_payload(tokens_saved=100)],
    )

    # Worker A — first to attempt acquisition wins.
    tracker_a = SubscriptionTracker(persist_path=persist_path, enabled=True)
    # Worker B — same lock path; flock will fail.
    tracker_b = SubscriptionTracker(persist_path=persist_path, enabled=True)

    tracker_a.update_contribution()
    tracker_b.update_contribution()

    # Owner polled and got 100; non-owner returned 0.
    a_rtk = tracker_a._state.contribution.tokens_saved_rtk
    b_rtk = tracker_b._state.contribution.tokens_saved_rtk
    # One worker saw the full 100; the other saw 0. (Order is OS-dependent
    # but exactly one owns the lock.)
    assert {a_rtk, b_rtk} == {0, 100}, (
        f"expected exactly one worker to poll; got a={a_rtk}, b={b_rtk}"
    )

    # Cleanup so subsequent tests don't see a stale lock.
    tracker_a._release_rtk_poll_lock()
    tracker_b._release_rtk_poll_lock()
