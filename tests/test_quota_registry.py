"""Tests for the pluggable QuotaTracker / QuotaTrackerRegistry abstraction."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from headroom.subscription.base import (
    QuotaTracker,
    QuotaTrackerRegistry,
    get_quota_registry,
    reset_quota_registry,
)

# ---------------------------------------------------------------------------
# Minimal concrete trackers for test use
# ---------------------------------------------------------------------------


class _AlwaysOnTracker(QuotaTracker):
    key = "always_on"
    label = "AlwaysOn"

    def __init__(self, stats: dict | None = None) -> None:
        self._stats = stats or {"value": 1}
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def get_stats(self) -> dict[str, Any] | None:
        return self._stats


class _UnavailableTracker(QuotaTracker):
    key = "unavailable"
    label = "Unavailable"

    def is_available(self) -> bool:
        return False

    def get_stats(self) -> dict[str, Any] | None:
        return {"should_not_appear": True}


class _NoDataTracker(QuotaTracker):
    key = "no_data"
    label = "NoData"

    def get_stats(self) -> dict[str, Any] | None:
        return None


class _PassiveTracker(QuotaTracker):
    """Uses inherited no-op start/stop."""

    key = "passive"
    label = "Passive"

    def get_stats(self) -> dict[str, Any] | None:
        return {"passive": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_registry():
    """Reset the global registry before every test."""
    reset_quota_registry()
    yield
    reset_quota_registry()


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_register_single_tracker():
    registry = QuotaTrackerRegistry()
    tracker = _AlwaysOnTracker()
    registry.register(tracker)
    assert registry.get("always_on") is tracker


def test_register_multiple_trackers():
    registry = QuotaTrackerRegistry()
    t1 = _AlwaysOnTracker()
    t2 = _PassiveTracker()
    registry.register(t1)
    registry.register(t2)
    assert len(registry.trackers) == 2


def test_duplicate_key_raises():
    registry = QuotaTrackerRegistry()
    registry.register(_AlwaysOnTracker())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_AlwaysOnTracker())


def test_get_unknown_key_returns_none():
    registry = QuotaTrackerRegistry()
    assert registry.get("nonexistent") is None


def test_trackers_property_is_snapshot():
    registry = QuotaTrackerRegistry()
    t = _AlwaysOnTracker()
    registry.register(t)
    snapshot = registry.trackers
    # Mutations to the snapshot don't affect the registry
    snapshot.clear()
    assert len(registry.trackers) == 1


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


def test_start_all_calls_available_trackers():
    registry = QuotaTrackerRegistry()
    t = _AlwaysOnTracker()
    registry.register(t)
    asyncio.run(registry.start_all())
    assert t.started is True


def test_start_all_skips_unavailable_trackers():
    registry = QuotaTrackerRegistry()
    unavailable = _UnavailableTracker()
    registry.register(unavailable)

    # Patch start to detect if it was called
    called = []

    async def _start() -> None:
        called.append(True)

    unavailable.start = _start  # type: ignore[method-assign]
    asyncio.run(registry.start_all())
    assert called == []


def test_stop_all_calls_stop_on_all_registered():
    registry = QuotaTrackerRegistry()
    t = _AlwaysOnTracker()
    unavailable = _UnavailableTracker()
    registry.register(t)
    registry.register(unavailable)
    asyncio.run(registry.stop_all())
    assert t.stopped is True


def test_stop_all_continues_on_exception():
    registry = QuotaTrackerRegistry()

    class _BrokenTracker(QuotaTracker):
        key = "broken"
        label = "Broken"

        async def stop(self) -> None:
            raise RuntimeError("stop failed")

        def get_stats(self) -> dict | None:
            return None

    registry.register(_BrokenTracker())
    # Should not raise
    asyncio.run(registry.stop_all())


def test_passive_tracker_start_stop_are_noops():
    registry = QuotaTrackerRegistry()
    t = _PassiveTracker()
    registry.register(t)
    asyncio.run(registry.start_all())
    asyncio.run(registry.stop_all())
    # No assertions needed — we verify no exception is raised


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


def test_get_all_stats_includes_available_with_data():
    registry = QuotaTrackerRegistry()
    registry.register(_AlwaysOnTracker({"x": 1}))
    stats = registry.get_all_stats()
    assert "always_on" in stats
    assert stats["always_on"] == {"x": 1}


def test_get_all_stats_excludes_unavailable():
    registry = QuotaTrackerRegistry()
    registry.register(_UnavailableTracker())
    stats = registry.get_all_stats()
    assert "unavailable" not in stats


def test_get_all_stats_excludes_none_data():
    registry = QuotaTrackerRegistry()
    registry.register(_NoDataTracker())
    stats = registry.get_all_stats()
    assert "no_data" not in stats


def test_get_all_stats_mixed():
    registry = QuotaTrackerRegistry()
    registry.register(_AlwaysOnTracker({"ok": True}))
    registry.register(_UnavailableTracker())
    registry.register(_NoDataTracker())
    stats = registry.get_all_stats()
    assert set(stats.keys()) == {"always_on"}


def test_get_stats_single_key():
    registry = QuotaTrackerRegistry()
    registry.register(_AlwaysOnTracker({"z": 99}))
    assert registry.get_stats("always_on") == {"z": 99}


def test_get_stats_missing_key_returns_none():
    registry = QuotaTrackerRegistry()
    assert registry.get_stats("missing") is None


# ---------------------------------------------------------------------------
# Global singleton tests
# ---------------------------------------------------------------------------


def test_get_quota_registry_returns_same_instance():
    r1 = get_quota_registry()
    r2 = get_quota_registry()
    assert r1 is r2


def test_reset_quota_registry_gives_fresh_instance():
    r1 = get_quota_registry()
    r1.register(_AlwaysOnTracker())
    reset_quota_registry()
    r2 = get_quota_registry()
    assert r1 is not r2
    assert len(r2.trackers) == 0


def test_reset_quota_registry_clears_registrations():
    registry = get_quota_registry()
    registry.register(_AlwaysOnTracker())
    reset_quota_registry()
    assert len(get_quota_registry().trackers) == 0
