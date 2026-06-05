"""Background subscription window tracker for Anthropic OAuth accounts.

Polls GET https://api.anthropic.com/api/oauth/usage on a configurable interval
while there has been at least one active OAuth session within the last minute.
Falls back to a stored token from ~/.claude/.credentials.json when no live
request has come through the proxy recently.

Architecture:
- Single asyncio.Task polling loop (started in start(), stopped via asyncio.Event)
- Thread-safe state updates via threading.Lock (consistent with headroom patterns)
- Atomic JSON persistence via tempfile + os.replace()
- Module-level singleton via get_subscription_tracker() / configure_subscription_tracker()

Also reads Claude transcript JSONL files (via session_tracking module) to provide
token breakdowns per window that enable:
  - Headroom efficiency metrics (tokens saved = raw - what proxy sent)
  - Surge pricing detection (API utilization vs expected from weighted tokens)
  - Cache miss detection (low cache_reads despite high input tokens)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from headroom import paths as _paths
from headroom.subscription.base import QuotaTracker
from headroom.subscription.client import SubscriptionClient
from headroom.subscription.models import (
    HeadroomContribution,
    RateLimitWindow,
    SubscriptionSnapshot,
    SubscriptionState,
    WindowDiscrepancy,
    WindowTokens,
    _utc_now,
    synthesize_window_render,
)

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_S = 300
_DEFAULT_ACTIVE_WINDOW_S = 60
_PERSIST_FILE_ENV = _paths.HEADROOM_SUBSCRIPTION_STATE_PATH_ENV
_DEFAULT_PERSIST_DIR = ".headroom"
_DEFAULT_PERSIST_FILE = "subscription_state.json"

# PR-G2 (Realignment) — RTK savings wiring.
#
# Operators can disable the RTK polling from inside ``update_contribution``
# without uninstalling the binary or unsetting ``HEADROOM_CONTEXT_TOOL``.
# Used for diagnostics and for environments where RTK is intentionally
# excluded from headroom accounting (e.g. shadow tests).
#
# Loud / configurable / no silent fallback: unknown values raise loudly via
# the parser below — they do not silently default to ``enabled``.
_RTK_WIRING_ENV = "HEADROOM_RTK_WIRING"
_RTK_WIRING_DEFAULT = "enabled"
_RTK_WIRING_ALLOWED = ("enabled", "disabled")

# PR-G2 remediation (C3) — multi-worker poll ownership.
#
# Each uvicorn worker independently runs ``configure_subscription_tracker``,
# so each worker would poll RTK and add the same delta to its own
# ``c.tokens_saved_rtk``. The persisted state is shared by atomic
# os.replace, but the in-memory counters diverge per worker — and any
# dashboard hitting a non-owner worker would see drifted values.
#
# The owner-election strategy mirrors the beacon's file-lock pattern in
# ``headroom/proxy/server.py``: a non-blocking ``fcntl.flock`` on
# ``HEADROOM_RTK_POLL_LOCK`` (default under the workspace dir). Only the
# lock holder polls; non-owners return 0 from ``_poll_rtk_delta`` and
# delegate to whatever the owner writes to the shared state file.
#
# Loud / no silent fallback: when ``fcntl`` is unavailable (Windows), every
# worker polls — but the explicit ``WindowsNoLockMode`` log line surfaces
# the choice so operators see it in startup logs.
_RTK_POLL_LOCK_ENV = "HEADROOM_RTK_POLL_LOCK"


def _rtk_wiring_mode() -> str:
    """Return ``enabled`` or ``disabled``. Raises on unknown values.

    Read at call-time so operators can flip the env var without a restart.
    """
    raw = os.environ.get(_RTK_WIRING_ENV, "").strip().lower()
    if not raw:
        return _RTK_WIRING_DEFAULT
    if raw in _RTK_WIRING_ALLOWED:
        return raw
    raise ValueError(f"Invalid {_RTK_WIRING_ENV}={raw!r}; expected one of {_RTK_WIRING_ALLOWED}")


def _validate_rtk_env_at_startup() -> None:
    """Validate RTK env vars eagerly at proxy startup.

    Raises ``ValueError`` loudly if ``HEADROOM_RTK_WIRING`` is set to an
    invalid value. PR-G2 remediation (H1): previously a typo at startup
    would silently default to enabled but get swallowed at every
    ``update_contribution`` call — fail loudly here instead.
    """
    _rtk_wiring_mode()


# Singleton on-demand poll floor (seconds): the dashboard may request a fresh
# poll if the cached snapshot is stale, but we cap how often we will actually
# hit Anthropic to avoid 429s / OAuth-token flagging. Bounded across users.
_DEFAULT_ON_DEMAND_POLL_FLOOR_S = 60.0

# Hard timeout for the on-demand poll so a slow upstream never blocks the
# dashboard request handler.
_ON_DEMAND_POLL_TIMEOUT_S = 2.0

# Rolling-window lengths. Single-value constants because Anthropic's API
# defines exactly one 5-hour and one 7-day window.
_FIVE_HOUR_WINDOW = timedelta(hours=5)
_SEVEN_DAY_WINDOW = timedelta(days=7)

# Surge pricing threshold: if actual utilization is >N% higher than expected,
# flag it as a potential surge pricing event.
_SURGE_THRESHOLD_PCT = 15.0

# Cache miss threshold: if cache_reads < N% of total input when we expect
# heavy caching (>50k input tokens in window), flag it.
_CACHE_MISS_RATIO_THRESHOLD = 0.10


def _get_persist_path() -> Path:
    return _paths.subscription_state_path()


class SubscriptionTracker(QuotaTracker):
    """Background tracker for Anthropic Claude Code subscription windows.

    Implements :class:`~headroom.subscription.base.QuotaTracker` so it can
    be registered with :func:`~headroom.subscription.base.get_quota_registry`
    alongside the Codex and Copilot trackers.

    Args:
        poll_interval_s: Seconds between polls while active (1–3600, default 300).
        active_window_s: Seconds since last notify_active call that keeps
            polling alive (default 60 = 1 minute).
        enabled: Set to ``False`` to disable tracking (mirrors
            ``ProxyConfig.subscription_tracking_enabled``).
        persist_path: Where to persist state across restarts.
        client: Injected client (for testing); defaults to SubscriptionClient().
    """

    # QuotaTracker identity
    key = "subscription_window"
    label = "Anthropic Claude Code"

    def __init__(
        self,
        poll_interval_s: int = _DEFAULT_POLL_INTERVAL_S,
        active_window_s: float = _DEFAULT_ACTIVE_WINDOW_S,
        enabled: bool = True,
        persist_path: Path | None = None,
        client: SubscriptionClient | None = None,
    ) -> None:
        self._enabled = enabled
        self._poll_interval_s = max(1, min(poll_interval_s, 3600))
        self._active_window_s = max(5.0, active_window_s)
        self._persist_path = persist_path or _get_persist_path()
        self._client = client or SubscriptionClient()

        self._lock = threading.Lock()
        self._state = SubscriptionState()
        self._current_token: str | None = None
        self._full_tokens: dict[str, int] = {}  # token_prefix -> count of requests

        # PR-G2 (Realignment) — most recent session-incremental ``tokens_saved``
        # observed in ``_get_rtk_stats()['session']['tokens_saved']``. This
        # field is de-baselined by the helper (see
        # :func:`~headroom.proxy.helpers._get_context_tool_stats`) so it
        # already represents savings accumulated since the proxy session
        # baseline was pinned at startup.
        #
        # PR-G2 remediation (C1): previously we read
        # ``lifetime_tokens_saved`` (the raw monotonic counter from
        # ``rtk gain --project``), which on the very first poll emitted the
        # entire pre-Headroom RTK history as one fake delta. Switching to
        # the session-incremental field dissolves both the first-poll
        # phantom and the post-restart phantom (the helper rebaselines
        # session counters at every proxy startup, so a fresh process sees
        # ``session.tokens_saved == 0`` until new RTK invocations land).
        #
        # Monotonic non-decreasing within a proxy session: only advances on
        # positive delta and on explicit counter-reset detection (session
        # value drops below the last seen value).
        self._last_rtk_tokens_saved: int = 0

        # PR-G2 remediation (C3) — owner-election state for multi-worker
        # poll deduplication. None means we haven't tried to elect yet; True
        # means this worker holds the lock and polls; False means another
        # worker owns the lock and we skip polling.
        self._rtk_poll_owner: bool | None = None
        self._rtk_poll_lock_fd: Any = None

        self._stop_event: asyncio.Event | None = None
        self._poll_task: asyncio.Task[None] | None = None

        # Unix-ts of the last on-demand poll triggered from the dashboard.
        # Floor-gated by ``_DEFAULT_ON_DEMAND_POLL_FLOOR_S``.
        self._last_on_demand_poll: float = 0.0
        self._on_demand_poll_floor_s: float = _DEFAULT_ON_DEMAND_POLL_FLOOR_S

        self._load_persisted_state()

    # ------------------------------------------------------------------
    # QuotaTracker interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Returns ``True`` when subscription tracking is enabled in config."""
        return self._enabled

    def get_stats(self) -> dict[str, Any] | None:
        """Return current tracker state dict for ``/stats``."""
        return self.state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._poll_task and not self._poll_task.done():
            return
        self._stop_event = asyncio.Event()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="subscription-tracker")
        logger.info("Subscription tracker started (poll_interval=%ds)", self._poll_interval_s)

    async def stop(self) -> None:
        """Stop the background polling loop and persist current state."""
        if self._stop_event:
            self._stop_event.set()
        if self._poll_task:
            try:
                await asyncio.wait_for(self._poll_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._poll_task.cancel()
        self._persist_state()
        # PR-G2 remediation (C3): release the poll lock so a subsequent
        # process / worker restart can re-elect the owner.
        self._release_rtk_poll_lock()
        logger.info("Subscription tracker stopped")

    # ------------------------------------------------------------------
    # Proxy integration hooks
    # ------------------------------------------------------------------

    def notify_active(self, token: str) -> None:
        """Called by the proxy handler when an OAuth request comes through.

        Stores the token for polling and marks the tracker as recently active.
        Only processes Bearer tokens that look like OAuth (not API keys).
        """
        if not token or not token.startswith("Bearer "):
            return
        raw = token[len("Bearer ") :]
        # Skip raw API keys (not OAuth tokens)
        if raw.startswith("sk-ant-api"):
            return
        with self._lock:
            self._current_token = raw
            self._state.last_active_at = _utc_now()
            prefix = raw[:8]
            self._full_tokens[prefix] = self._full_tokens.get(prefix, 0) + 1

    def update_contribution(
        self,
        *,
        tokens_submitted: int = 0,
        tokens_saved_compression: int = 0,
        tokens_saved_cli_filtering: int | None = None,
        tokens_saved_rtk: int | None = None,
        tokens_saved_cache_reads: int = 0,
        compression_savings_usd: float = 0.0,
        cache_savings_usd: float = 0.0,
    ) -> None:
        """Update headroom contribution counters for the current session window.

        Called after each proxy request completes with the actual token deltas.

        PR-G2 (Realignment) — ``tokens_saved_rtk`` is now sourced from RTK's
        own stats endpoint (``rtk gain --format json`` via
        :func:`headroom.proxy.helpers._get_rtk_stats`) when the caller does
        not pass an explicit value. The tracker computes the delta against
        the last per-session ``tokens_saved`` it observed and feeds only the
        delta into the contribution counter.

        PR-G2 remediation (C1): the RTK source is the SESSION-incremental
        ``session.tokens_saved`` field of the helper payload, NOT the raw
        ``lifetime_tokens_saved`` counter. The helper de-baselines per
        proxy session, so the first poll after process startup correctly
        reads 0 instead of the entire pre-Headroom RTK history.

        Args:
            tokens_saved_cli_filtering: Explicit per-call CLI-filtering
                contribution. PR-G2 remediation (M5): ``None`` means "caller
                omitted, default 0" (mirrors the ``tokens_saved_rtk is
                None`` semantic). An explicit ``0`` is honored verbatim.
            tokens_saved_rtk: Explicit override for tokens saved by RTK on
                this call. If ``None`` (the default), the tracker polls RTK
                stats itself and writes the per-call delta. If passed
                (including ``0``), the override is used verbatim.
        """
        # Polled outside the lock so the subprocess call can't deadlock the
        # event loop or contend with concurrent ``notify_active`` callers.
        if tokens_saved_rtk is None:
            tokens_saved_rtk = self._poll_rtk_delta()

        with self._lock:
            c = self._state.contribution
            # PR-G2 remediation (M5): explicit None-guard so callers can
            # pass ``0`` and have it honored without colliding with the
            # default "caller omitted" semantic.
            if tokens_saved_cli_filtering is None:
                cli_filtering = 0
            else:
                cli_filtering = tokens_saved_cli_filtering
            c.tokens_submitted += max(tokens_submitted, 0)
            c.tokens_saved_compression += max(tokens_saved_compression, 0)
            c.tokens_saved_cli_filtering += max(cli_filtering, 0)
            c.tokens_saved_rtk += max(tokens_saved_rtk, 0)
            c.tokens_saved_cache_reads += max(tokens_saved_cache_reads, 0)
            c.compression_savings_usd += max(compression_savings_usd, 0.0)
            c.cache_savings_usd += max(cache_savings_usd, 0.0)

    def _poll_rtk_delta(self) -> int:
        """Return the delta of the session-scoped RTK ``tokens_saved`` since last poll.

        Implementation of PR-G2 data-plane wiring. Calls
        :func:`headroom.proxy.helpers._get_rtk_stats` and reads the
        session-incremental ``session.tokens_saved`` field (de-baselined per
        proxy session by the helper), then diffs against
        ``self._last_rtk_tokens_saved``.

        Returns ``0`` (never negative) when:
        - ``HEADROOM_RTK_WIRING=disabled`` — operator opt-out.
        - ``_get_rtk_stats()`` returns ``None`` — RTK not selected / not
          installed; explicit zero is the right answer.
        - ``_get_rtk_stats()`` raises — transient error, logged loudly.
        - The session counter regressed (RTK reset / new project) — that
          path also re-baselines ``_last_rtk_tokens_saved`` to the new
          (smaller) value so subsequent polls return correct deltas.
        - PR-G2 remediation (C3): another worker holds the RTK poll lock —
          this worker skips polling so we don't double-count.

        Otherwise advances ``self._last_rtk_tokens_saved`` to the new
        session total and returns the positive delta.

        PR-G2 remediation (H1): ``HEADROOM_RTK_WIRING`` is now validated at
        startup via :func:`_validate_rtk_env_at_startup`; an invalid value
        raises at proxy boot rather than being silently swallowed here.
        We still catch + structured-log per-call as a defence-in-depth so
        that an env var flipped to garbage after startup is at least loud
        in the logs.
        """
        try:
            wiring_mode = _rtk_wiring_mode()
        except ValueError as exc:
            # PR-G2 remediation (H1): elevate to ERROR — this is config
            # corruption, not a transient runtime hiccup. The bad value
            # should have been caught at startup but a rotation could flip
            # it mid-run; either way the operator must see this.
            logger.error(
                "event=subscription_rtk_invalid_env error=%s",
                exc,
            )
            return 0
        if wiring_mode == "disabled":
            return 0

        # PR-G2 remediation (C3): only the lock-holder worker polls. We try
        # once per tracker instance and cache the verdict; the lock is
        # released on tracker stop.
        if not self._try_acquire_rtk_poll_lock():
            return 0

        try:
            # Local import keeps the tracker module decoupled from the proxy
            # helper at import time (helpers.py imports many heavy deps).
            from headroom.proxy.helpers import _get_rtk_stats
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "event=subscription_rtk_helper_import_failed error=%s",
                exc,
            )
            return 0

        try:
            stats = _get_rtk_stats()
        except Exception as exc:
            logger.warning(
                "event=subscription_rtk_stats_fetch_failed error=%s",
                exc,
            )
            return 0

        if stats is None:
            logger.info(
                "event=subscription_rtk_stats_unavailable wiring=%s",
                wiring_mode,
            )
            return 0

        # PR-G2 remediation (C1): read the SESSION-incremental field, not
        # the raw lifetime counter. The helper de-baselines per proxy
        # session at startup, so ``session.tokens_saved`` already excludes
        # the pre-Headroom RTK history. Falls back to the top-level
        # ``tokens_saved`` (which is also session-scoped in the canonical
        # payload; see ``_get_context_tool_stats``) and finally to 0 for
        # synthetic-zero payloads.
        session_payload = stats.get("session")
        if isinstance(session_payload, dict) and "tokens_saved" in session_payload:
            current_total_raw = session_payload.get("tokens_saved", 0)
        else:
            current_total_raw = stats.get("tokens_saved", 0)
        try:
            current_total = int(current_total_raw or 0)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "event=subscription_rtk_stats_coerce_failed value=%r error=%s",
                current_total_raw,
                exc,
            )
            return 0

        with self._lock:
            last = self._last_rtk_tokens_saved
            if current_total < last:
                # Counter regressed: helper rebaselined (session reset) or
                # RTK rebuilt its DB. Re-baseline silently — losing one
                # delta is preferable to reporting a giant negative number.
                logger.info(
                    "event=subscription_rtk_counter_regressed previous=%d current=%d",
                    last,
                    current_total,
                )
                self._last_rtk_tokens_saved = current_total
                return 0
            delta = current_total - last
            if delta > 0:
                self._last_rtk_tokens_saved = current_total
            return delta

    # ------------------------------------------------------------------
    # Multi-worker poll ownership (PR-G2 remediation C3)
    # ------------------------------------------------------------------

    def _try_acquire_rtk_poll_lock(self) -> bool:
        """Try to acquire the RTK poll file lock (non-blocking).

        Returns ``True`` if this worker owns the lock and should poll RTK.
        Caches the verdict so we don't pay the syscall on every call. The
        lock is released in :meth:`_release_rtk_poll_lock`, called from
        :meth:`stop`.

        Mirrors the beacon's ``_try_acquire_beacon_lock`` pattern in
        ``headroom/proxy/server.py`` (fcntl.flock, LOCK_EX | LOCK_NB).
        """
        if self._rtk_poll_owner is not None:
            return self._rtk_poll_owner

        try:
            import fcntl as _fcntl
        except ImportError:
            # Platform without fcntl (Windows). Every worker polls; log
            # loudly so the operator knows the multi-worker invariant is
            # weaker on this platform.
            logger.warning("event=subscription_rtk_poll_lock_unavailable platform=no-fcntl")
            self._rtk_poll_owner = True
            return True
        fcntl: Any = _fcntl

        lock_path = self._rtk_poll_lock_path()
        fd = None
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = open(lock_path, "w")  # noqa: SIM115
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd.write(str(os.getpid()))
            fd.flush()
            self._rtk_poll_lock_fd = fd
            self._rtk_poll_owner = True
            logger.info(
                "event=subscription_rtk_poll_lock_acquired pid=%d path=%s",
                os.getpid(),
                lock_path,
            )
            return True
        except OSError:
            if fd is not None:
                fd.close()
            self._rtk_poll_owner = False
            logger.info(
                "event=subscription_rtk_poll_lock_skipped pid=%d path=%s",
                os.getpid(),
                lock_path,
            )
            return False

    def _release_rtk_poll_lock(self) -> None:
        """Release the RTK poll file lock; safe to call repeatedly."""
        fd = self._rtk_poll_lock_fd
        if fd is None:
            return
        try:
            import fcntl as _fcntl

            fcntl: Any = _fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            fd.close()
        except Exception:
            pass
        self._rtk_poll_lock_fd = None
        try:
            self._rtk_poll_lock_path().unlink(missing_ok=True)
        except Exception:
            pass

    def _rtk_poll_lock_path(self) -> Path:
        """Return the path to the RTK poll lock file."""
        override = os.environ.get(_RTK_POLL_LOCK_ENV, "").strip()
        if override:
            return Path(override)
        return self._persist_path.parent / ".rtk_poll_lock"

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """Return current tracker state as a serialisable dict."""
        with self._lock:
            return self._state.to_dict()

    @property
    def latest_snapshot(self) -> SubscriptionSnapshot | None:
        with self._lock:
            return self._state.latest

    def is_active(self) -> bool:
        with self._lock:
            return self._state.is_active(active_window_s=self._active_window_s)

    # ------------------------------------------------------------------
    # Display-time rendering (issue #281)
    # ------------------------------------------------------------------

    def render_state(self) -> dict[str, Any]:
        """Return the dashboard-facing state dict, synthesizing post-reset.

        Background poll cadence is capped at 5 minutes by Anthropic-tolerance
        constraints; if the user's 5-hour window rolls over between two polls
        the cached ``utilization_pct`` is stale and the dashboard would render
        the OLD window's percentage even though Claude Code itself shows 0%
        (issue #281). To avoid lying to the user without hammering the API,
        we synthesize the post-reset windows here using locally-tracked
        transcript token counts.

        The returned dict preserves every key in :meth:`SubscriptionState
        .to_dict` for backward compatibility; per-window dicts inside
        ``latest`` gain ``synthesized: bool`` and ``resets_at_estimated:
        bool`` keys plus an optional ``render_warning`` string when synthesis
        falls back.
        """
        with self._lock:
            base = self._state.to_dict()
            snapshot = self._state.latest

        if snapshot is None or base.get("latest") is None:
            return base

        latest_dict = base["latest"]
        latest_dict["five_hour"] = self._render_window(
            snapshot.five_hour,
            window_duration=_FIVE_HOUR_WINDOW,
            window_name="five_hour",
        )
        latest_dict["seven_day"] = self._render_window(
            snapshot.seven_day,
            window_duration=_SEVEN_DAY_WINDOW,
            window_name="seven_day",
        )
        if snapshot.seven_day_opus is not None:
            latest_dict["seven_day_opus"] = self._render_window(
                snapshot.seven_day_opus,
                window_duration=_SEVEN_DAY_WINDOW,
                window_name="seven_day_opus",
            )
        if snapshot.seven_day_sonnet is not None:
            latest_dict["seven_day_sonnet"] = self._render_window(
                snapshot.seven_day_sonnet,
                window_duration=_SEVEN_DAY_WINDOW,
                window_name="seven_day_sonnet",
            )
        return base

    def _render_window(
        self,
        window: RateLimitWindow | None,
        *,
        window_duration: timedelta,
        window_name: str,
    ) -> dict[str, Any]:
        used_since_reset = self._compute_used_since_reset(window)
        return synthesize_window_render(
            window,
            used_since_reset=used_since_reset,
            window_duration=window_duration,
            window_name=window_name,
        )

    def _compute_used_since_reset(self, window: RateLimitWindow | None) -> int | None:
        """Read transcripts to count tokens spent strictly after ``window.resets_at``.

        Returns ``None`` if the window has no observed reset time or if the
        transcript scan fails (caller treats ``None`` as 0 for arithmetic
        but propagates a render_warning when synthesis is otherwise
        attempted).
        """
        if window is None or window.resets_at is None:
            return None
        now = _utc_now()
        if now < window.resets_at:
            return None
        try:
            from headroom.subscription import session_tracking

            tokens = session_tracking.compute_window_tokens(
                window.resets_at.timestamp(), now.timestamp()
            )
            return int(tokens.weighted_token_equivalent or tokens.total_raw())
        except Exception as exc:
            logger.warning("event=subscription_render_used_since_reset_failed error=%s", exc)
            return None

    async def maybe_poll_on_demand(self) -> None:
        """Trigger at most one bounded poll per ``_DEFAULT_ON_DEMAND_POLL_FLOOR_S``.

        Called from the ``/subscription-window`` endpoint so a dashboard
        refresh can pull a fresh snapshot when the cache is stale, while
        keeping fleet-wide poll rate within Anthropic tolerance. All
        exceptions are swallowed and logged — never propagated.
        """
        now_ts = time.time()
        with self._lock:
            elapsed = now_ts - self._last_on_demand_poll
            if elapsed < self._on_demand_poll_floor_s:
                logger.info(
                    "event=subscription_on_demand_poll_skipped_floor elapsed_s=%.1f floor_s=%.1f",
                    elapsed,
                    self._on_demand_poll_floor_s,
                )
                return
            self._last_on_demand_poll = now_ts

        logger.info(
            "event=subscription_on_demand_poll_triggered floor_s=%.1f",
            self._on_demand_poll_floor_s,
        )
        try:
            await asyncio.wait_for(self._maybe_poll(), timeout=_ON_DEMAND_POLL_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "event=subscription_on_demand_poll_timeout timeout_s=%.1f",
                _ON_DEMAND_POLL_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning("event=subscription_on_demand_poll_failed error=%s", exc)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._maybe_poll()
            except Exception as exc:
                logger.warning("Subscription tracker poll error: %s", exc)
            try:
                # NOTE: do NOT wrap in asyncio.shield() — shield prevents the
                # inner Event.wait() from being cancelled when wait_for times
                # out, leaking one Task per poll interval. Over hours the
                # accumulated idle waiters bog down the event loop scheduler
                # (observed as the "aged proxy degradation" in 2026-04-17).
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
                break  # stop event was set
            except asyncio.TimeoutError:
                pass  # normal: poll interval elapsed

    async def _maybe_poll(self) -> None:
        with self._lock:
            is_active = self._state.is_active(active_window_s=self._active_window_s)
            token = self._current_token

        if not is_active:
            # Try background poll using credentials file token
            from headroom.subscription.client import read_cached_oauth_token

            bg_token = read_cached_oauth_token()
            if not bg_token:
                return
            token = token or bg_token

        snapshot = await self._client.fetch(token)
        if snapshot is None:
            with self._lock:
                self._state.mark_error("fetch returned None")
            return

        # Read transcript-based window tokens
        window_tokens = _compute_window_tokens_for_snapshot(snapshot)

        # Detect anomalies
        discrepancies = _detect_discrepancies(snapshot, window_tokens)

        with self._lock:
            self._state.add_snapshot(snapshot)
            self._state.window_tokens = window_tokens
            for d in discrepancies:
                self._state.add_discrepancy(d)
            self._state.last_error = None
            # Reset contribution when 5h window rolls over
            self._maybe_reset_contribution(snapshot)

        self._persist_state()
        logger.debug(
            "Subscription poll: 5h=%.1f%% 7d=%.1f%%",
            snapshot.five_hour.utilization_pct,
            snapshot.seven_day.utilization_pct,
        )

        # Update OTEL metrics if configured
        try:
            from headroom.observability.metrics import get_otel_metrics

            get_otel_metrics().record_subscription_window(self._state.to_dict())
        except Exception:
            pass

    def _maybe_reset_contribution(self, snapshot: SubscriptionSnapshot) -> None:
        """Reset contribution counters when the 5h window rolls over."""
        prev = self._state.history[-2] if len(self._state.history) >= 2 else None
        if prev is None:
            return
        prev_resets_at = prev.five_hour.resets_at
        curr_resets_at = snapshot.five_hour.resets_at
        if (
            prev_resets_at is not None
            and curr_resets_at is not None
            and curr_resets_at != prev_resets_at
        ):
            logger.info("5h window rolled over; resetting headroom contribution counters")
            self._state.contribution = HeadroomContribution()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = self._state.to_persist_dict()
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self._persist_path.parent,
                delete=False,
                suffix=".tmp",
                encoding="utf-8",
            ) as fh:
                json.dump(data, fh, indent=2)
                tmp_path = fh.name
            os.replace(tmp_path, self._persist_path)
        except Exception as exc:
            logger.debug("Failed to persist subscription state: %s", exc)

    def _load_persisted_state(self) -> None:
        try:
            with open(self._persist_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Restore only the contribution counters and poll counts for now;
            # snapshot data is re-fetched on first active poll.
            contrib = raw.get("contribution", {})
            c = self._state.contribution
            c.tokens_submitted = int(contrib.get("tokens_submitted", 0))
            saved = contrib.get("tokens_saved", {})
            # Newer state writes dashboard-facing ``compression`` as
            # proxy-compression + CLI filtering. Prefer the raw proxy field when
            # present so loading does not double-count CLI filtering.
            c.tokens_saved_compression = int(
                saved.get("proxy_compression", saved.get("compression", 0))
            )
            # PR-G2 (Realignment) — prefer the raw counters when present
            # (new format). For backward compatibility with state written
            # before PR-G2 we fall back to the dashboard-aliased keys.
            cli_filtering = int(
                saved.get(
                    "cli_filtering_raw",
                    saved.get("cli_filtering", saved.get("rtk", 0)),
                )
            )
            c.tokens_saved_cli_filtering = cli_filtering
            # PR-G2 remediation (M2): legacy state files written before this
            # PR have no ``rtk_raw`` key — but pre-G2 the ``rtk`` field
            # silently mirrored ``cli_filtering``. Treat the legacy ``rtk``
            # field as authoritative for the rtk_raw counter to avoid
            # zeroing historical accumulation. New format writes ``rtk_raw``
            # explicitly; legacy writes had only ``rtk`` (aliased) and we
            # honor it on read.
            is_legacy_state = "rtk_raw" not in saved
            if is_legacy_state:
                # Pre-G2 semantic: ``rtk`` == ``cli_filtering``. Carry the
                # accumulated value forward instead of silently zeroing it.
                legacy_rtk_value = int(saved.get("rtk", cli_filtering))
                c.tokens_saved_rtk = legacy_rtk_value
                logger.info(
                    "event=subscription_state_legacy_load "
                    "migrated_rtk_raw_from_cli_filtering=%d path=%s",
                    legacy_rtk_value,
                    self._persist_path,
                )
            else:
                c.tokens_saved_rtk = int(saved.get("rtk_raw", 0))
            c.tokens_saved_cache_reads = int(saved.get("cache_reads", 0))
            savings_usd = contrib.get("savings_usd", {})
            c.compression_savings_usd = float(savings_usd.get("compression", 0.0))
            c.cache_savings_usd = float(savings_usd.get("cache", 0.0))
            self._state.poll_count = int(raw.get("poll_count", 0))
            logger.debug("Loaded persisted subscription state from %s", self._persist_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.debug("Could not load persisted subscription state: %s", exc)


# ---------------------------------------------------------------------------
# Transcript-based window token computation
# ---------------------------------------------------------------------------


def _compute_window_tokens_for_snapshot(snapshot: SubscriptionSnapshot) -> WindowTokens:
    """Read Claude transcript files and sum tokens for the current 5h window."""
    try:
        from headroom.subscription import session_tracking

        resets_at = snapshot.five_hour.resets_at
        if resets_at is None:
            return WindowTokens()
        window_duration_s = 5 * 3600  # 5-hour window
        start_ts = resets_at.timestamp() - window_duration_s
        end_ts = resets_at.timestamp()
        return session_tracking.compute_window_tokens(start_ts, end_ts)
    except Exception as exc:
        logger.debug("Could not compute window tokens from transcripts: %s", exc)
        return WindowTokens()


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def _detect_discrepancies(
    snapshot: SubscriptionSnapshot,
    window_tokens: WindowTokens,
) -> list[WindowDiscrepancy]:
    """Detect surge pricing or cache miss anomalies in the snapshot."""
    discrepancies: list[WindowDiscrepancy] = []

    if snapshot.five_hour.limit > 0 and window_tokens.weighted_token_equivalent > 0:
        expected_pct = window_tokens.weighted_token_equivalent / snapshot.five_hour.limit * 100.0
        actual_pct = snapshot.five_hour.utilization_pct
        delta = actual_pct - expected_pct

        if delta > _SURGE_THRESHOLD_PCT:
            discrepancies.append(
                WindowDiscrepancy(
                    kind="surge_pricing",
                    description=(
                        f"API 5h utilization ({actual_pct:.1f}%) is "
                        f"{delta:.1f}% higher than transcript-implied "
                        f"({expected_pct:.1f}%); possible surge weighting."
                    ),
                    severity="warning" if delta < 30 else "alert",
                    expected_utilization_pct=round(expected_pct, 2),
                    actual_utilization_pct=round(actual_pct, 2),
                    delta_pct=round(delta, 2),
                )
            )

    total_input = window_tokens.input
    total_cache_reads = window_tokens.cache_reads
    if total_input > 50_000 and total_cache_reads < total_input * _CACHE_MISS_RATIO_THRESHOLD:
        cache_ratio = total_cache_reads / total_input if total_input else 0
        discrepancies.append(
            WindowDiscrepancy(
                kind="cache_miss",
                description=(
                    f"Cache-read ratio is {cache_ratio:.1%} (threshold "
                    f"{_CACHE_MISS_RATIO_THRESHOLD:.0%}); system may not be "
                    "using prefix cache effectively."
                ),
                severity="warning",
                expected_utilization_pct=None,
                actual_utilization_pct=None,
                delta_pct=None,
            )
        )

    return discrepancies


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker_lock = threading.Lock()
_tracker_instance: SubscriptionTracker | None = None


def get_subscription_tracker() -> SubscriptionTracker | None:
    """Return the global singleton tracker, or None if not configured."""
    return _tracker_instance


def configure_subscription_tracker(
    poll_interval_s: int = _DEFAULT_POLL_INTERVAL_S,
    active_window_s: float = _DEFAULT_ACTIVE_WINDOW_S,
    enabled: bool = True,
    persist_path: Path | None = None,
    client: SubscriptionClient | None = None,
) -> SubscriptionTracker:
    """Create (or return existing) global tracker singleton.

    PR-G2 remediation (H1): validates RTK-related env vars eagerly here so
    a typo (``HEADROOM_RTK_WIRING=enabld``) crashes the proxy at startup
    instead of being silently swallowed at every ``update_contribution``
    call.
    """
    _validate_rtk_env_at_startup()
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = SubscriptionTracker(
                poll_interval_s=poll_interval_s,
                active_window_s=active_window_s,
                enabled=enabled,
                persist_path=persist_path,
                client=client,
            )
    return _tracker_instance


async def shutdown_subscription_tracker() -> None:
    """Stop and clean up the global tracker."""
    global _tracker_instance
    with _tracker_lock:
        tracker = _tracker_instance
        _tracker_instance = None
    if tracker:
        await tracker.stop()
