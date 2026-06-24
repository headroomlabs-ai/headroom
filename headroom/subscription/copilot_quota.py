"""GitHub Copilot monthly quota tracking via the copilot_internal/user API.

GitHub Copilot exposes per-category monthly quotas (chat, completions,
premium_interactions) at ``GET https://api.github.com/copilot_internal/user``
authenticated with a GitHub Bearer token.

Token discovery order (first non-empty wins):
    1. GITHUB_COPILOT_GITHUB_TOKEN
    2. GITHUB_TOKEN
    3. COPILOT_GITHUB_TOKEN
    4. GITHUB_COPILOT_API_TOKEN

The poll is triggered every ``poll_interval_s`` seconds (default 60) as long as
a GitHub token is available.  No proxy traffic interception is needed — tokens
come from the environment at headroom start-up.

API response schema (relevant fields from ``/copilot_internal/user``):
    login                  str        GitHub username
    copilot_plan           str        "free" | "individual" | "business" | "enterprise"
    access_type_sku        str        plan SKU string
    quota_reset_date_utc   str        ISO-8601 date when monthly quota resets
    quota_snapshots:
        chat / completions / premium_interactions:
            entitlement       int  total monthly allocation
            remaining         int  remaining uses this month
            quota_remaining   int  (alias for remaining)
            percent_remaining float  0-100
            overage_count     int  uses beyond entitlement
            overage_permitted bool whether overage is allowed
            unlimited         bool whether this category is unlimited
            timestamp_utc     str  when the snapshot was recorded
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from headroom.subscription.base import QuotaTracker

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_TOKEN_ENV_VARS = [
    "GITHUB_COPILOT_GITHUB_TOKEN",
    "GITHUB_TOKEN",
    "COPILOT_GITHUB_TOKEN",
    "GITHUB_COPILOT_API_TOKEN",
]

# Categories surfaced by the quota_snapshots endpoint
QUOTA_CATEGORIES = ("chat", "completions", "premium_interactions")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CopilotQuotaCategory:
    """Quota data for a single Copilot usage category."""

    name: str
    entitlement: int | None = None  # total monthly allocation
    remaining: int | None = None  # remaining uses
    percent_remaining: float | None = None  # 0-100
    overage_count: int = 0  # uses beyond entitlement
    overage_permitted: bool = False
    unlimited: bool = False
    timestamp_utc: str | None = None

    @property
    def used(self) -> int | None:
        if self.entitlement is not None and self.remaining is not None:
            return max(0, self.entitlement - self.remaining)
        return None

    @property
    def used_percent(self) -> float | None:
        if self.unlimited:
            return 0.0
        if self.percent_remaining is not None:
            return max(0.0, 100.0 - self.percent_remaining)
        if self.entitlement and self.entitlement > 0 and self.remaining is not None:
            return 100.0 * (self.entitlement - self.remaining) / self.entitlement
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entitlement": self.entitlement,
            "remaining": self.remaining,
            "used": self.used,
            "percent_remaining": self.percent_remaining,
            "used_percent": self.used_percent,
            "overage_count": self.overage_count,
            "overage_permitted": self.overage_permitted,
            "unlimited": self.unlimited,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class CopilotQuotaSnapshot:
    """Full quota snapshot from one /copilot_internal/user response."""

    login: str | None = None
    copilot_plan: str | None = None
    access_type_sku: str | None = None
    quota_reset_date_utc: str | None = None
    categories: dict[str, CopilotQuotaCategory] = field(default_factory=dict)
    fetched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "login": self.login,
            "copilot_plan": self.copilot_plan,
            "access_type_sku": self.access_type_sku,
            "quota_reset_date_utc": self.quota_reset_date_utc,
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "fetched_at": self.fetched_at,
        }


@dataclass
class CopilotQuotaState:
    """Thread-safe singleton state for Copilot quota tracking."""

    latest: CopilotQuotaSnapshot | None = None
    last_error: str | None = None
    last_updated: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest": self.latest.to_dict() if self.latest else None,
            "last_error": self.last_error,
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_copilot_quota(data: dict[str, Any]) -> CopilotQuotaSnapshot:
    """Parse a /copilot_internal/user response into a ``CopilotQuotaSnapshot``."""
    snapshot = CopilotQuotaSnapshot(
        login=data.get("login"),
        copilot_plan=data.get("copilot_plan"),
        access_type_sku=data.get("access_type_sku"),
        quota_reset_date_utc=data.get("quota_reset_date_utc") or data.get("quota_reset_date"),
    )

    raw_qs = data.get("quota_snapshots") or {}
    for cat_name in QUOTA_CATEGORIES:
        raw = raw_qs.get(cat_name)
        if not raw:
            continue
        remaining = raw.get("remaining") or raw.get("quota_remaining")
        cat = CopilotQuotaCategory(
            name=cat_name,
            entitlement=raw.get("entitlement"),
            remaining=remaining,
            percent_remaining=raw.get("percent_remaining"),
            overage_count=raw.get("overage_count") or 0,
            overage_permitted=bool(raw.get("overage_permitted")),
            unlimited=bool(raw.get("unlimited")),
            timestamp_utc=raw.get("timestamp_utc"),
        )
        snapshot.categories[cat_name] = cat

    return snapshot


# ---------------------------------------------------------------------------
# Token discovery
# ---------------------------------------------------------------------------


def discover_github_token() -> str | None:
    """Return the first GitHub token found from known environment variables."""
    for var in _TOKEN_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return None


# ---------------------------------------------------------------------------
# Background tracker singleton
# ---------------------------------------------------------------------------


class _CopilotQuotaTracker(QuotaTracker):
    """Singleton background poller for GitHub Copilot quota.

    Implements :class:`~headroom.subscription.base.QuotaTracker` so it can be
    registered with the :class:`~headroom.subscription.base.QuotaTrackerRegistry`.
    Availability is gated on a GitHub token being present in the environment.
    """

    # QuotaTracker identity
    key = "copilot_quota"
    label = "GitHub Copilot"

    def __init__(self, poll_interval_s: float = 60.0) -> None:
        self._poll_interval_s = poll_interval_s
        self._state = CopilotQuotaState()
        self._lock = Lock()
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # QuotaTracker interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Returns ``True`` when a GitHub token is available in the environment."""
        return discover_github_token() is not None

    def get_stats(self) -> dict[str, Any] | None:
        """Return the latest quota state dict, or ``None`` if no data yet."""
        data = self.state
        if not data.get("latest"):
            return None
        return data

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the polling loop."""
        if self._stop_event:
            self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Mirror SubscriptionTracker.stop(): on timeout or outer
                # cancellation, cancel the underlying poll task. Without
                # this, a wedged poll task would leak past ``stop()``.
                self._task.cancel()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._maybe_poll()
            except Exception as exc:
                logger.warning("Copilot quota poll error: %s", exc)
            try:
                # NOTE: do NOT wrap in asyncio.shield() — shield prevents the
                # inner Event.wait() from being cancelled when wait_for times
                # out, leaking one Task per poll interval. See the matching
                # note in headroom/subscription/tracker.py:_poll_loop.
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
                break  # stop event was set
            except asyncio.TimeoutError:
                pass  # normal: poll interval elapsed

    async def _maybe_poll(self) -> None:
        token = discover_github_token()
        if not token:
            return

        try:
            import aiohttp
        except ImportError:
            logger.debug("aiohttp not available; skipping Copilot quota poll")
            return

        url = f"{_GITHUB_API_BASE}/copilot_internal/user"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 401:
                        with self._lock:
                            self._state.last_error = "unauthorized — check GITHUB_TOKEN"
                        return
                    if resp.status == 404:
                        # API-key-only account or endpoint not available
                        with self._lock:
                            self._state.last_error = "endpoint not found (non-Copilot account?)"
                        return
                    if not resp.ok:
                        with self._lock:
                            self._state.last_error = f"HTTP {resp.status}"
                        return

                    data = await resp.json()
        except Exception as exc:
            with self._lock:
                self._state.last_error = str(exc)
            logger.debug("Copilot quota fetch failed: %s", exc)
            return

        try:
            snapshot = parse_copilot_quota(data)
        except Exception as exc:
            with self._lock:
                self._state.last_error = f"parse error: {exc}"
            return

        with self._lock:
            self._state.latest = snapshot
            self._state.last_error = None
            self._state.last_updated = time.time()

        logger.debug(
            "Copilot quota polled: plan=%s categories=%s",
            snapshot.copilot_plan,
            list(snapshot.categories.keys()),
        )


_singleton_lock = Lock()
_singleton: _CopilotQuotaTracker | None = None


def get_copilot_quota_tracker() -> _CopilotQuotaTracker:
    """Return the global ``_CopilotQuotaTracker`` singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = _CopilotQuotaTracker()
    return _singleton
