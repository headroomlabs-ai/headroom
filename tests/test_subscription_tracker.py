from __future__ import annotations

import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import headroom.subscription.tracker as tracker_module
from headroom.subscription.models import (
    HeadroomContribution,
    RateLimitWindow,
    SubscriptionSnapshot,
    WindowDiscrepancy,
    WindowTokens,
    _utc_now,
)
from headroom.subscription.tracker import SubscriptionTracker


def _make_snapshot(
    *,
    token_prefix: str = "token123",
    reset_offset_hours: int = 5,
    resets_at: datetime | None = None,
) -> SubscriptionSnapshot:
    return SubscriptionSnapshot(
        five_hour=RateLimitWindow(
            used=10,
            limit=100,
            utilization_pct=10.0,
            resets_at=(
                resets_at
                if resets_at is not None
                else _utc_now() + timedelta(hours=reset_offset_hours)
            ),
        ),
        seven_day=RateLimitWindow(used=20, limit=200, utilization_pct=10.0),
        token_prefix=token_prefix,
    )


def test_tracker_notify_active_update_and_basic_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    # PR-G2: keep the unit test deterministic — do not let
    # ``update_contribution`` call out to ``rtk gain`` via the proxy helper.
    monkeypatch.setattr(SubscriptionTracker, "_poll_rtk_delta", lambda self: 0)
    tracker = SubscriptionTracker(enabled=False)

    assert tracker.is_available() is False
    assert tracker.latest_snapshot is None
    assert tracker.is_active() is False
    assert isinstance(tracker.get_stats(), dict)

    tracker.notify_active("")
    tracker.notify_active("Basic token")
    tracker.notify_active("Bearer sk-ant-api-key")
    assert tracker._current_token is None

    tracker.notify_active("Bearer oauth-token-123")
    assert tracker._current_token == "oauth-token-123"
    assert tracker._full_tokens["oauth-to"] == 1
    assert tracker.is_active() is True

    tracker.update_contribution(
        tokens_submitted=10,
        tokens_saved_compression=5,
        tokens_saved_cli_filtering=-1,
        tokens_saved_cache_reads=3,
        compression_savings_usd=1.25,
        cache_savings_usd=-2.0,
    )
    contribution = tracker._state.contribution
    assert contribution.tokens_submitted == 10
    assert contribution.tokens_saved_compression == 5
    assert contribution.tokens_saved_cli_filtering == 0
    assert contribution.tokens_saved_rtk == 0
    assert contribution.tokens_saved_cache_reads == 3
    assert contribution.to_dict()["tokens_saved"]["compression"] == 5
    assert contribution.to_dict()["tokens_saved"]["proxy_compression"] == 5
    assert contribution.to_dict()["tokens_saved"]["cli_filtering"] == 0
    assert contribution.to_dict()["tokens_saved"]["rtk"] == 0
    assert contribution.compression_savings_usd == 1.25
    assert contribution.cache_savings_usd == 0.0


@pytest.mark.asyncio
async def test_tracker_start_stop_and_rollover_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker(persist_path=tmp_path / "state.json")

    async def fake_poll_loop() -> None:
        assert tracker._stop_event is not None
        await tracker._stop_event.wait()

    tracker._poll_loop = fake_poll_loop  # type: ignore[method-assign]
    await tracker.start()
    first_task = tracker._poll_task
    assert first_task is not None

    await tracker.start()
    assert tracker._poll_task is first_task

    await tracker.stop()
    assert tracker._stop_event is not None and tracker._stop_event.is_set()
    assert tracker._persist_path.exists()

    tracker._state.history = [
        _make_snapshot(reset_offset_hours=5),
        _make_snapshot(reset_offset_hours=6),
    ]
    tracker._state.contribution = HeadroomContribution(tokens_submitted=99)
    tracker._maybe_reset_contribution(tracker._state.history[-1])
    assert tracker._state.contribution.tokens_submitted == 0


def test_second_level_reset_jitter_does_not_reset_contribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: the usage API reports ``resets_at`` with second-level jitter
    within a single window (observed flapping between ``01:59:59Z`` and
    ``02:00:00Z`` on consecutive polls). That must NOT be treated as a rollover,
    or the contribution counters get zeroed every poll and the dashboard sticks
    at ~0% savings.
    """
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker(persist_path=tmp_path / "state.json")

    base = _utc_now() + timedelta(hours=3)
    tracker._state.history = [
        _make_snapshot(resets_at=base),
        _make_snapshot(resets_at=base + timedelta(seconds=1)),
    ]
    tracker._state.contribution = HeadroomContribution(tokens_submitted=99)
    tracker._maybe_reset_contribution(tracker._state.history[-1])
    assert tracker._state.contribution.tokens_submitted == 99


def test_genuine_five_hour_rollover_resets_contribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real rollover advances ``resets_at`` by ~5 hours and still resets."""
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker(persist_path=tmp_path / "state.json")

    base = _utc_now()
    tracker._state.history = [
        _make_snapshot(resets_at=base),
        _make_snapshot(resets_at=base + timedelta(hours=5)),
    ]
    tracker._state.contribution = HeadroomContribution(tokens_submitted=99)
    tracker._maybe_reset_contribution(tracker._state.history[-1])
    assert tracker._state.contribution.tokens_submitted == 0


@pytest.mark.asyncio
async def test_maybe_poll_handles_inactive_and_none_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker()

    monkeypatch.setattr("headroom.subscription.client.read_cached_oauth_token", lambda: None)
    await tracker._maybe_poll()
    assert tracker._state.poll_count == 0

    monkeypatch.setattr(
        "headroom.subscription.client.read_cached_oauth_token", lambda: "cached-token"
    )

    async def fetch_none(token: str | None):
        return None

    tracker._client = SimpleNamespace(fetch=fetch_none)
    await tracker._maybe_poll()
    assert tracker._state.last_error == "fetch returned None"
    assert tracker._state.poll_errors == 1


@pytest.mark.asyncio
async def test_maybe_poll_success_updates_state_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker()
    tracker.notify_active("Bearer live-oauth-token")

    snapshot = _make_snapshot()
    discrepancies = [WindowDiscrepancy(kind="cache_miss", description="miss", severity="warning")]
    metrics_calls: list[dict] = []

    async def fetch_snapshot(token: str | None):
        assert token == "live-oauth-token"
        return snapshot

    tracker._client = SimpleNamespace(fetch=fetch_snapshot)
    monkeypatch.setattr(
        tracker_module, "_compute_window_tokens_for_snapshot", lambda snap: WindowTokens(input=7)
    )
    monkeypatch.setattr(tracker_module, "_detect_discrepancies", lambda snap, tokens: discrepancies)
    monkeypatch.setattr(
        tracker, "_persist_state", lambda: metrics_calls.append({"persisted": True})
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.observability.metrics",
        SimpleNamespace(
            get_otel_metrics=lambda: SimpleNamespace(
                record_subscription_window=lambda state: metrics_calls.append(state)
            )
        ),
    )

    await tracker._maybe_poll()
    assert tracker.latest_snapshot is snapshot
    assert tracker._state.window_tokens.input == 7
    assert tracker._state.discrepancies[-1].kind == "cache_miss"
    assert tracker._state.last_error is None
    assert tracker._state.poll_count == 1
    assert metrics_calls[0] == {"persisted": True}
    assert isinstance(metrics_calls[1], dict)


@pytest.mark.asyncio
async def test_maybe_poll_runs_transcript_scan_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the transcript scan must run off the event-loop thread, or a
    multi-second ~/.claude/projects scan wedges the proxy every poll interval."""
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker()
    tracker.notify_active("Bearer live-oauth-token")

    snapshot = _make_snapshot()

    async def fetch_snapshot(token: str | None) -> SubscriptionSnapshot:
        return snapshot

    tracker._client = SimpleNamespace(fetch=fetch_snapshot)

    loop_thread_id = threading.get_ident()
    seen: dict[str, int] = {}

    def recording_compute(snap: SubscriptionSnapshot) -> WindowTokens:
        seen["thread_id"] = threading.get_ident()
        return WindowTokens(input=7)

    monkeypatch.setattr(tracker_module, "_compute_window_tokens_for_snapshot", recording_compute)
    monkeypatch.setattr(tracker_module, "_detect_discrepancies", lambda snap, tokens: [])
    monkeypatch.setattr(tracker, "_persist_state", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "headroom.observability.metrics",
        SimpleNamespace(
            get_otel_metrics=lambda: SimpleNamespace(record_subscription_window=lambda state: None)
        ),
    )

    await tracker._maybe_poll()

    # The blocking scan ran on a worker thread, not the event-loop thread.
    assert seen["thread_id"] != loop_thread_id


def test_persist_and_load_state_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # PR-G2: ``update_contribution`` polls RTK by default; pin the helper to
    # 0 so the round-trip is deterministic.
    monkeypatch.setattr(SubscriptionTracker, "_poll_rtk_delta", lambda self: 0)

    persist_path = tmp_path / "tracker-state.json"
    tracker = SubscriptionTracker(persist_path=persist_path)
    tracker.update_contribution(
        tokens_submitted=11,
        tokens_saved_compression=2,
        tokens_saved_cli_filtering=3,
        tokens_saved_cache_reads=4,
        compression_savings_usd=1.5,
        cache_savings_usd=2.5,
    )
    # PR-G2: also write a raw RTK delta directly to assert the persisted
    # ``rtk_raw`` field round-trips independently of cli_filtering.
    tracker.update_contribution(tokens_saved_rtk=9)
    tracker._state.poll_count = 7
    tracker._persist_state()

    loader = SubscriptionTracker(persist_path=persist_path)
    assert loader._state.contribution.tokens_submitted == 11
    assert loader._state.contribution.tokens_saved_compression == 2
    # PR-G2: the raw counters now round-trip independently of the legacy
    # dashboard alias.
    assert loader._state.contribution.tokens_saved_cli_filtering == 3
    assert loader._state.contribution.tokens_saved_rtk == 9
    assert loader._state.contribution.tokens_saved_cache_reads == 4
    # ``compression`` is ``proxy_compression + cli_filtering_saved()`` =
    # ``2 + max(3, 9)`` = 11 after PR-G2 (was 5 when rtk mirrored
    # cli_filtering).
    assert loader._state.contribution.to_dict()["tokens_saved"]["compression"] == 11
    assert loader._state.contribution.to_dict()["tokens_saved"]["proxy_compression"] == 2
    # Dashboard ``cli_filtering`` / ``rtk`` keys remain ``max(cli, rtk)``
    # for legacy display — 9 wins. Raw counters expose the un-aliased
    # values for the tracker's own round-trip.
    assert loader._state.contribution.to_dict()["tokens_saved"]["cli_filtering"] == 9
    assert loader._state.contribution.to_dict()["tokens_saved"]["rtk"] == 9
    assert loader._state.contribution.to_dict()["tokens_saved"]["cli_filtering_raw"] == 3
    assert loader._state.contribution.to_dict()["tokens_saved"]["rtk_raw"] == 9
    assert loader._state.contribution.compression_savings_usd == 1.5
    assert loader._state.contribution.cache_savings_usd == 2.5
    assert loader._state.poll_count == 7

    persist_path.write_text("{invalid", encoding="utf-8")
    broken = SubscriptionTracker(persist_path=persist_path)
    assert broken._state.poll_count == 0

    missing = SubscriptionTracker(persist_path=tmp_path / "missing.json")
    assert missing._state.poll_count == 0
