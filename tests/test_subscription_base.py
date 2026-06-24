from __future__ import annotations

import pytest

import headroom.subscription.base as base_module
from headroom.subscription.base import (
    QuotaTracker,
    QuotaTrackerRegistry,
    get_quota_registry,
    reset_quota_registry,
)


class DummyTracker(QuotaTracker):
    def __init__(
        self,
        key: str,
        *,
        label: str = "Dummy",
        available: bool = True,
        stats: dict | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self._key = key
        self._label = label
        self._available = available
        self._stats = stats
        self._stop_error = stop_error
        self.started = 0
        self.stopped = 0

    @property
    def key(self) -> str:
        return self._key

    @property
    def label(self) -> str:
        return self._label

    def is_available(self) -> bool:
        return self._available

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1
        if self._stop_error:
            raise self._stop_error

    def get_stats(self) -> dict | None:
        return self._stats


class PassiveTracker(QuotaTracker):
    @property
    def key(self) -> str:
        return "passive"

    @property
    def label(self) -> str:
        return "Passive"

    def get_stats(self) -> dict | None:
        return {"passive": True}


def test_register_get_trackers_and_duplicate_keys() -> None:
    registry = QuotaTrackerRegistry()
    tracker = DummyTracker("alpha", stats={"ok": True})
    registry.register(tracker)

    assert registry.get("alpha") is tracker
    assert registry.get("missing") is None
    assert registry.trackers == [tracker]
    assert registry.get_stats("alpha") == {"ok": True}
    assert registry.get_stats("missing") is None

    snapshot = registry.trackers
    snapshot.clear()
    assert registry.trackers == [tracker]

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DummyTracker("alpha"))


@pytest.mark.asyncio
async def test_start_all_stop_all_and_stats_filtering() -> None:
    registry = QuotaTrackerRegistry()
    enabled = DummyTracker("enabled", label="Enabled", stats={"value": 1})
    disabled = DummyTracker("disabled", label="Disabled", available=False, stats={"skip": True})
    empty = DummyTracker("empty", label="Empty", stats=None)
    broken = DummyTracker(
        "broken", label="Broken", stats={"value": 2}, stop_error=RuntimeError("boom")
    )

    for tracker in (enabled, disabled, empty, broken):
        registry.register(tracker)

    await registry.start_all()
    assert enabled.started == 1
    assert disabled.started == 0
    assert empty.started == 1
    assert broken.started == 1

    assert registry.get_all_stats() == {
        "enabled": {"value": 1},
        "broken": {"value": 2},
    }

    await registry.stop_all()
    assert enabled.stopped == 1
    assert disabled.stopped == 1
    assert empty.stopped == 1
    assert broken.stopped == 1


def test_quota_registry_singleton_reset() -> None:
    reset_quota_registry()
    base_module._registry = None
    first = get_quota_registry()
    second = get_quota_registry()
    assert first is second

    first.register(DummyTracker("singleton"))
    assert get_quota_registry().get("singleton") is not None

    reset_quota_registry()
    refreshed = get_quota_registry()
    assert refreshed is not first
    assert refreshed.trackers == []


@pytest.mark.asyncio
async def test_quota_tracker_default_methods() -> None:
    tracker = PassiveTracker()
    assert tracker.is_available() is True
    await tracker.start()
    await tracker.stop()
    assert tracker.get_stats() == {"passive": True}
