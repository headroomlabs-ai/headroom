"""Base abstractions for pluggable AI-tool quota / rate-limit trackers.

Every provider tracker (Anthropic, Codex, Copilot, …) inherits from
:class:`QuotaTracker` and is registered with the process-global
:class:`QuotaTrackerRegistry`.  ``server.py`` only interacts with the
registry — adding a new provider requires *zero* changes to the server.

Quick-start for a new provider::

    from headroom.subscription.base import QuotaTracker, get_quota_registry

    class GeminiQuotaTracker(QuotaTracker):
        key   = "gemini_quota"
        label = "Google Gemini"

        def is_available(self) -> bool:
            return bool(os.environ.get("GOOGLE_API_KEY"))

        async def start(self) -> None: ...   # launch background poll
        async def stop(self)  -> None: ...   # cancel poll task

        def get_stats(self) -> dict | None:
            return ...  # serialisable dict or None if no data yet

    get_quota_registry().register(GeminiQuotaTracker())
"""

from __future__ import annotations

import abc
import logging
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class QuotaTracker(abc.ABC):
    """Abstract base for a single AI-tool quota / rate-limit tracker.

    Subclasses must define :attr:`key`, :attr:`label`, and
    :meth:`get_stats`.  All other methods have sensible defaults.
    """

    # ------------------------------------------------------------------ #
    # Class-level identity — subclasses should override as class attributes
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """Stats key used in ``/stats`` and the dashboard.

        Must be unique across all registered trackers.
        Examples: ``"subscription_window"``, ``"codex_rate_limits"``.
        """

    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable name for log messages.

        Example: ``"Anthropic Claude Code"``.
        """

    # ------------------------------------------------------------------ #
    # Availability gate
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Return ``True`` if this tracker should be activated.

        Override to gate on environment variables, config flags, etc.
        The registry calls this before :meth:`start` and skips trackers
        that return ``False``.  Default: always available.
        """
        return True

    # ------------------------------------------------------------------ #
    # Lifecycle — default no-ops (suitable for passive/header-based trackers)
    # ------------------------------------------------------------------ #

    async def start(self) -> None:  # noqa: B027
        """Start background polling.  No-op for passive trackers."""

    async def stop(self) -> None:  # noqa: B027
        """Stop background polling.  No-op for passive trackers."""

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_stats(self) -> dict[str, Any] | None:
        """Return the current snapshot as a serialisable dict, or ``None``.

        ``None`` means "no data yet" and causes the key to be omitted from
        ``/stats`` rather than appearing as ``null``.
        """


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class QuotaTrackerRegistry:
    """Process-global registry of all :class:`QuotaTracker` instances.

    Typical usage::

        registry = get_quota_registry()
        registry.register(SubscriptionTracker(...))
        registry.register(get_codex_rate_limit_state())
        registry.register(get_copilot_quota_tracker())

        # server startup
        await registry.start_all()

        # /stats assembly
        stats.update(registry.get_all_stats())

        # server shutdown
        await registry.stop_all()
    """

    def __init__(self) -> None:
        self._trackers: list[QuotaTracker] = []
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, tracker: QuotaTracker) -> None:
        """Register a tracker.  Duplicate keys are rejected."""
        with self._lock:
            existing_keys = {t.key for t in self._trackers}
            if tracker.key in existing_keys:
                raise ValueError(
                    f"A tracker with key '{tracker.key}' is already registered. "
                    "Each tracker must have a unique key."
                )
            self._trackers.append(tracker)

    def get(self, key: str) -> QuotaTracker | None:
        """Return the registered tracker for *key*, or ``None``."""
        with self._lock:
            for t in self._trackers:
                if t.key == key:
                    return t
        return None

    @property
    def trackers(self) -> list[QuotaTracker]:
        """Read-only snapshot of the registered tracker list."""
        with self._lock:
            return list(self._trackers)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start_all(self) -> None:
        """Start every available tracker and log its status."""
        for tracker in self.trackers:
            if tracker.is_available():
                await tracker.start()
                logger.info("%s quota tracking: ENABLED", tracker.label)
            else:
                logger.info("%s quota tracking: DISABLED (not available)", tracker.label)

    async def stop_all(self) -> None:
        """Stop all registered trackers (regardless of availability)."""
        for tracker in self.trackers:
            try:
                await tracker.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping %s tracker: %s", tracker.label, exc)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def get_all_stats(self) -> dict[str, dict[str, Any] | None]:
        """Return ``{key: stats_dict}`` for every available tracker.

        Trackers that are unavailable or return ``None`` are excluded.
        """
        result: dict[str, dict[str, Any] | None] = {}
        for tracker in self.trackers:
            if not tracker.is_available():
                continue
            stats = tracker.get_stats()
            if stats is not None:
                result[tracker.key] = stats
        return result

    def get_stats(self, key: str) -> dict[str, Any] | None:
        """Return stats for a single tracker by key, or ``None``."""
        tracker = self.get(key)
        return tracker.get_stats() if tracker is not None else None


# --------------------------------------------------------------------------- #
# Process-global singleton
# --------------------------------------------------------------------------- #

_registry: QuotaTrackerRegistry | None = None
_registry_lock = Lock()


def get_quota_registry() -> QuotaTrackerRegistry:
    """Return the process-global :class:`QuotaTrackerRegistry` singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = QuotaTrackerRegistry()
    return _registry


def reset_quota_registry() -> None:
    """Replace the global registry with a fresh empty instance.

    Intended for use in tests only.
    """
    global _registry
    with _registry_lock:
        _registry = QuotaTrackerRegistry()
