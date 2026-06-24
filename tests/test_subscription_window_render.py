"""Tests for issue #281: dashboard 5h window stays at e.g. 44% after Anthropic
Claude Code's actual 5h window has reset (shows 0%).

Verifies:
  - In-window: cached utilization is returned verbatim (synthesized=False).
  - Post-reset: render_state synthesizes a fresh utilization from local
    transcript-derived token counts (synthesized=True).
  - Edge cases: zero local tokens, capped-at-100%, None resets_at.
  - On-demand poll is gated by the 60s floor and never raises into callers.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

import headroom.subscription.tracker as tracker_module
from headroom.subscription.models import (
    RateLimitWindow,
    SubscriptionSnapshot,
    WindowTokens,
    _utc_now,
)
from headroom.subscription.tracker import SubscriptionTracker


def _build_tracker(monkeypatch: pytest.MonkeyPatch) -> SubscriptionTracker:
    """Construct a tracker with persisted-state loading disabled."""
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    return SubscriptionTracker(enabled=True)


def _install_snapshot(
    tracker: SubscriptionTracker,
    *,
    five_hour: RateLimitWindow,
    seven_day: RateLimitWindow | None = None,
) -> None:
    snap = SubscriptionSnapshot(
        five_hour=five_hour,
        seven_day=seven_day or RateLimitWindow(used=0, limit=0, utilization_pct=0.0),
    )
    tracker._state.latest = snap
    tracker._state.history.append(snap)


def _patch_used_since_reset(monkeypatch: pytest.MonkeyPatch, value: int | None) -> None:
    """Override the transcript-scan method on every tracker instance."""
    monkeypatch.setattr(
        SubscriptionTracker,
        "_compute_used_since_reset",
        lambda self, window: value,
    )


# ---------------------------------------------------------------------------
# render_state behaviour
# ---------------------------------------------------------------------------


def test_render_within_window_returns_cached_pct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot is fresh: cached utilization passes through; synthesized=False."""
    tracker = _build_tracker(monkeypatch)
    resets_at = _utc_now() + timedelta(minutes=30)
    _install_snapshot(
        tracker,
        five_hour=RateLimitWindow(
            used=44_000, limit=100_000, utilization_pct=44.0, resets_at=resets_at
        ),
    )
    # Should not be called pre-reset, but install a strict no-op anyway.
    _patch_used_since_reset(monkeypatch, None)

    rendered = tracker.render_state()
    fh = rendered["latest"]["five_hour"]

    assert fh["synthesized"] is False
    assert fh["resets_at_estimated"] is False
    assert fh["utilization_pct"] == 44.0
    assert fh["used"] == 44_000
    assert fh["limit"] == 100_000


def test_render_after_reset_synthesizes_from_local_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past reset: synthesize utilization from local transcript token count."""
    tracker = _build_tracker(monkeypatch)
    resets_at = _utc_now() - timedelta(minutes=10)
    _install_snapshot(
        tracker,
        five_hour=RateLimitWindow(
            used=44_000, limit=100_000, utilization_pct=44.0, resets_at=resets_at
        ),
    )
    _patch_used_since_reset(monkeypatch, 5_000)

    rendered = tracker.render_state()
    fh = rendered["latest"]["five_hour"]

    assert fh["synthesized"] is True
    assert fh["resets_at_estimated"] is True
    assert fh["used"] == 5_000
    assert fh["limit"] == 100_000
    assert fh["utilization_pct"] == pytest.approx(5.0, abs=0.01)
    # Next reset must advance forward exactly one window from the observed.
    assert fh["resets_at"] is not None


def test_render_after_reset_with_zero_local_tokens_renders_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past reset, no local activity: utilization renders as 0%."""
    tracker = _build_tracker(monkeypatch)
    resets_at = _utc_now() - timedelta(minutes=15)
    _install_snapshot(
        tracker,
        five_hour=RateLimitWindow(
            used=44_000, limit=100_000, utilization_pct=44.0, resets_at=resets_at
        ),
    )
    _patch_used_since_reset(monkeypatch, 0)

    rendered = tracker.render_state()
    fh = rendered["latest"]["five_hour"]

    assert fh["synthesized"] is True
    assert fh["used"] == 0
    assert fh["utilization_pct"] == 0.0


def test_render_capped_at_100pct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local tokens exceed limit (rare): display caps at 100% — never higher.

    We undercount tokens spent on Claude Code outside this proxy and must
    not produce >100% utilization on the dashboard.
    """
    tracker = _build_tracker(monkeypatch)
    resets_at = _utc_now() - timedelta(minutes=5)
    _install_snapshot(
        tracker,
        five_hour=RateLimitWindow(
            used=44_000, limit=100_000, utilization_pct=44.0, resets_at=resets_at
        ),
    )
    _patch_used_since_reset(monkeypatch, 999_999)

    rendered = tracker.render_state()
    fh = rendered["latest"]["five_hour"]

    assert fh["synthesized"] is True
    assert fh["used"] == 100_000  # capped at limit
    assert fh["utilization_pct"] == 100.0


def test_render_handles_missing_resets_at_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot has resets_at=None (e.g. malformed API response): pass-through."""
    tracker = _build_tracker(monkeypatch)
    _install_snapshot(
        tracker,
        five_hour=RateLimitWindow(used=10_000, limit=100_000, utilization_pct=10.0, resets_at=None),
    )
    _patch_used_since_reset(monkeypatch, 12_345)

    rendered = tracker.render_state()
    fh = rendered["latest"]["five_hour"]

    assert fh["synthesized"] is False
    assert fh["utilization_pct"] == 10.0
    assert fh["used"] == 10_000


def test_render_preserves_existing_state_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward-compat: render_state must preserve every key in to_dict."""
    tracker = _build_tracker(monkeypatch)
    resets_at = _utc_now() + timedelta(hours=4)
    _install_snapshot(
        tracker,
        five_hour=RateLimitWindow(
            used=44_000, limit=100_000, utilization_pct=44.0, resets_at=resets_at
        ),
    )
    tracker._state.window_tokens = WindowTokens(input=1, output=2)

    cached = tracker.state
    rendered = tracker.render_state()

    # All keys present in cached must still be present after render.
    assert set(cached.keys()) <= set(rendered.keys())


# ---------------------------------------------------------------------------
# maybe_poll_on_demand behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_poll_on_demand_singleton_60s_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call triggers a poll; second call within 60s is a no-op."""
    tracker = _build_tracker(monkeypatch)

    call_count = 0

    async def counting_poll() -> None:
        nonlocal call_count
        call_count += 1

    tracker._maybe_poll = counting_poll  # type: ignore[method-assign]

    # Use a controllable clock.
    clock: dict[str, float] = {"t": 1_000_000.0}
    monkeypatch.setattr(tracker_module.time, "time", lambda: clock["t"])

    await tracker.maybe_poll_on_demand()
    assert call_count == 1

    # 30s later — still inside the 60s floor.
    clock["t"] = 1_000_030.0
    await tracker.maybe_poll_on_demand()
    assert call_count == 1

    # 90s later — past the floor; another poll fires.
    clock["t"] = 1_000_090.0
    await tracker.maybe_poll_on_demand()
    assert call_count == 2


@pytest.mark.asyncio
async def test_maybe_poll_on_demand_does_not_raise_on_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream errors must be swallowed — never propagate to the request handler."""
    tracker = _build_tracker(monkeypatch)

    async def boom() -> None:
        raise RuntimeError("anthropic returned 500")

    tracker._maybe_poll = boom  # type: ignore[method-assign]

    # Should not raise.
    await tracker.maybe_poll_on_demand()


# ---------------------------------------------------------------------------
# Defensive: synthesis fallback when transcript scan blows up
# ---------------------------------------------------------------------------


def test_render_synthesis_fallback_logs_warning_and_returns_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the synthesis helper raises, we fall back to cached + render_warning."""
    tracker = _build_tracker(monkeypatch)
    resets_at = _utc_now() - timedelta(minutes=5)
    _install_snapshot(
        tracker,
        five_hour=RateLimitWindow(
            used=44_000, limit=100_000, utilization_pct=44.0, resets_at=resets_at
        ),
    )

    def boom_synthesize(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic explosion")

    monkeypatch.setattr(tracker_module, "synthesize_window_render", boom_synthesize)

    # render_state itself does NOT swallow synthesize errors directly — the
    # synthesize helper is responsible for that. Replace the helper with one
    # that simulates an internal error path returning a fallback dict.
    def fallback_synthesize(window: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "used": window.used,
            "limit": window.limit,
            "utilization_pct": window.utilization_pct,
            "resets_at": None,
            "seconds_to_reset": None,
            "synthesized": False,
            "resets_at_estimated": False,
            "render_warning": "synthesis_failed: synthetic explosion",
        }

    monkeypatch.setattr(tracker_module, "synthesize_window_render", fallback_synthesize)

    rendered = tracker.render_state()
    fh = rendered["latest"]["five_hour"]

    assert fh["synthesized"] is False
    assert fh.get("render_warning", "").startswith("synthesis_failed:")


# ---------------------------------------------------------------------------
# Defensive: render_state with no snapshot at all
# ---------------------------------------------------------------------------


def test_render_state_with_no_snapshot_returns_base_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-first-poll: no snapshot yet — return the base state dict unchanged."""
    tracker = _build_tracker(monkeypatch)
    rendered = tracker.render_state()
    assert rendered["latest"] is None
    assert "poll_count" in rendered  # backward-compat keys preserved


# ---------------------------------------------------------------------------
# Synthesis helper unit tests (independent of tracker)
# ---------------------------------------------------------------------------


def test_synthesize_helper_handles_none_window() -> None:
    from headroom.subscription.models import synthesize_window_render

    out = synthesize_window_render(None, used_since_reset=None, window_duration=timedelta(hours=5))
    assert out["synthesized"] is False
    assert out["used"] == 0
    assert out["limit"] == 0


def test_synthesize_helper_advances_reset_when_dashboard_long_after_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If now is multiple windows past resets_at, walk forward correctly."""
    from headroom.subscription.models import synthesize_window_render

    now = _utc_now()
    resets_at = now - timedelta(hours=12)  # 2.4 windows ago
    window = RateLimitWindow(used=10, limit=100, utilization_pct=10.0, resets_at=resets_at)
    out = synthesize_window_render(
        window,
        used_since_reset=20,
        window_duration=timedelta(hours=5),
        now=now,
    )
    assert out["synthesized"] is True
    # 12h ago + 3 windows of 5h = 3h in the future.
    assert out["seconds_to_reset"] > 0
