"""Tracking of OpenAI Codex rate-limit window data.

Historically Codex embedded rate-limit data in API *response headers*
(``x-codex-primary-used-percent`` etc.) and headroom captured those headers
from proxied responses (:meth:`CodexRateLimitState.update_from_headers`).
Current Codex (codex_exec / TUI on the ChatGPT WebSocket transport) no longer
emits those headers on the ``/responses`` handshake or stream -- the window is
served from a dedicated endpoint instead:

    GET https://chatgpt.com/backend-api/wham/usage   (ChatGPT OAuth/session)

So this module also exposes :func:`maybe_schedule_usage_poll`, a throttled
fire-and-forget GET against that endpoint using the client's own bearer token
and ``ChatGPT-Account-Id``.  The header-capture path is kept intact: if OpenAI
ever returns ``x-codex-*`` again it still works, and API-key requests (no
account id) simply never trigger a poll.

Header schema (parsed by codex-rs ``rate_limits.rs``):
    x-codex-primary-used-percent      float 0-100
    x-codex-primary-window-minutes    int   window size
    x-codex-primary-reset-at          int   Unix timestamp (seconds)
    x-codex-secondary-used-percent    float 0-100   (optional)
    x-codex-secondary-window-minutes  int            (optional)
    x-codex-secondary-reset-at        int            (optional)
    x-codex-credits-has-credits       bool
    x-codex-credits-unlimited         bool
    x-codex-credits-balance           str   e.g. "$5.00"
    x-codex-promo-message             str   server announcement
    x-codex-limit-name                str   e.g. "gpt-5.2-codex-sonic"

``GET /wham/usage`` JSON schema (mapped by :func:`parse_codex_usage_payload`):
    plan_type                                   str
    rate_limit.primary_window.used_percent      float 0-100
    rate_limit.primary_window.limit_window_seconds  int
    rate_limit.primary_window.reset_at          int   Unix timestamp (seconds)
    rate_limit.secondary_window.*               same shape (optional)
    credits.has_credits / unlimited / balance   bool / bool / str
    rate_limit_reached_type                     str | null
    promo                                       obj | str | null
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock

import httpx

from headroom.subscription.base import QuotaTracker

logger = logging.getLogger(__name__)

# Dedicated Codex usage endpoint for ChatGPT OAuth/session auth. Overridable
# for tests / self-hosted gateways via env.
CODEX_USAGE_URL = (
    os.environ.get("HEADROOM_CODEX_USAGE_URL", "https://chatgpt.com/backend-api/wham/usage").strip()
    or "https://chatgpt.com/backend-api/wham/usage"
)

# Minimum seconds between live usage polls. Codex turns can arrive in bursts;
# one GET per minute is plenty to keep the gauge fresh without hammering.
USAGE_POLL_MIN_INTERVAL_S = 60.0

# Bound the usage GET so a slow upstream never wedges the fire-and-forget task.
_USAGE_POLL_TIMEOUT_S = 10.0


@dataclass
class CodexRateLimitWindow:
    """Usage data for a single rolling rate-limit window."""

    used_percent: float
    window_minutes: int | None = None
    resets_at: int | None = None  # Unix timestamp (seconds)

    @property
    def window_label(self) -> str:
        if self.window_minutes is None:
            return "unknown"
        if self.window_minutes < 60:
            return f"{self.window_minutes}m"
        hours = self.window_minutes // 60
        mins = self.window_minutes % 60
        return f"{hours}h{mins:02d}m" if mins else f"{hours}h"

    @property
    def seconds_until_reset(self) -> int | None:
        if self.resets_at is None:
            return None
        return max(0, self.resets_at - int(time.time()))

    def to_dict(self) -> dict:
        return {
            "used_percent": self.used_percent,
            "window_minutes": self.window_minutes,
            "window_label": self.window_label,
            "resets_at": self.resets_at,
            "seconds_until_reset": self.seconds_until_reset,
        }


@dataclass
class CodexCreditsSnapshot:
    """OpenAI credits balance for Codex."""

    has_credits: bool
    unlimited: bool
    balance: str | None = None

    def to_dict(self) -> dict:
        return {
            "has_credits": self.has_credits,
            "unlimited": self.unlimited,
            "balance": self.balance,
        }


@dataclass
class CodexRateLimitSnapshot:
    """Full rate-limit snapshot parsed from a single Codex API response."""

    limit_id: str = "codex"
    limit_name: str | None = None
    primary: CodexRateLimitWindow | None = None
    secondary: CodexRateLimitWindow | None = None
    credits: CodexCreditsSnapshot | None = None
    promo_message: str | None = None
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "limit_id": self.limit_id,
            "limit_name": self.limit_name,
            "primary": self.primary.to_dict() if self.primary else None,
            "secondary": self.secondary.to_dict() if self.secondary else None,
            "credits": self.credits.to_dict() if self.credits else None,
            "promo_message": self.promo_message,
            "captured_at": self.captured_at,
        }


# ---------------------------------------------------------------------------
# Header parsing helpers
# ---------------------------------------------------------------------------


def _parse_float(headers: dict[str, str], name: str) -> float | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        v = float(raw)
        return v if v == v else None  # NaN guard
    except (ValueError, TypeError):
        return None


def _parse_int(headers: dict[str, str], name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _parse_bool(headers: dict[str, str], name: str) -> bool | None:
    raw = headers.get(name)
    if raw is None:
        return None
    if raw.lower() in ("true", "1"):
        return True
    if raw.lower() in ("false", "0"):
        return False
    return None


def _parse_window(headers: dict[str, str], prefix: str, which: str) -> CodexRateLimitWindow | None:
    used_pct = _parse_float(headers, f"{prefix}-{which}-used-percent")
    if used_pct is None:
        return None
    return CodexRateLimitWindow(
        used_percent=used_pct,
        window_minutes=_parse_int(headers, f"{prefix}-{which}-window-minutes"),
        resets_at=_parse_int(headers, f"{prefix}-{which}-reset-at"),
    )


def _parse_credits(headers: dict[str, str]) -> CodexCreditsSnapshot | None:
    has_credits = _parse_bool(headers, "x-codex-credits-has-credits")
    if has_credits is None:
        return None
    unlimited = _parse_bool(headers, "x-codex-credits-unlimited") or False
    raw_balance = headers.get("x-codex-credits-balance", "").strip()
    return CodexCreditsSnapshot(
        has_credits=has_credits,
        unlimited=unlimited,
        balance=raw_balance or None,
    )


def parse_codex_rate_limits(headers: dict[str, str]) -> CodexRateLimitSnapshot | None:
    """Parse a :class:`CodexRateLimitSnapshot` from a dict of HTTP response headers.

    Returns ``None`` when no Codex rate-limit headers are present (e.g. the
    response came from a non-Codex OpenAI endpoint or a cached reply).
    """
    prefix = "x-codex"
    primary = _parse_window(headers, prefix, "primary")
    secondary = _parse_window(headers, prefix, "secondary")
    credits = _parse_credits(headers)
    raw_promo = headers.get("x-codex-promo-message", "").strip()
    promo = raw_promo or None
    raw_limit_name = headers.get("x-codex-limit-name", "").strip()
    limit_name = raw_limit_name or None

    if primary is None and secondary is None and credits is None and promo is None:
        return None  # Not a Codex response with rate-limit headers

    return CodexRateLimitSnapshot(
        limit_id="codex",
        limit_name=limit_name,
        primary=primary,
        secondary=secondary,
        credits=credits,
        promo_message=promo,
    )


# ---------------------------------------------------------------------------
# Usage-endpoint (GET /wham/usage) JSON parsing
# ---------------------------------------------------------------------------


def _window_from_usage_json(win: object) -> CodexRateLimitWindow | None:
    """Map one ``rate_limit.{primary,secondary}_window`` object to a window."""
    if not isinstance(win, dict):
        return None
    used = win.get("used_percent")
    try:
        used_f = float(used)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    if used_f != used_f:  # NaN guard
        return None

    window_minutes: int | None = None
    secs = win.get("limit_window_seconds")
    if isinstance(secs, (int, float)) and secs > 0:
        # Round up, matching codex-rs window_minutes_from_seconds.
        window_minutes = (int(secs) + 59) // 60

    resets_at = win.get("reset_at")
    resets_at = int(resets_at) if isinstance(resets_at, (int, float)) else None

    return CodexRateLimitWindow(
        used_percent=used_f,
        window_minutes=window_minutes,
        resets_at=resets_at,
    )


def parse_codex_usage_payload(payload: object) -> CodexRateLimitSnapshot | None:
    """Parse a snapshot from a ``GET /wham/usage`` JSON body.

    Returns ``None`` when the body carries no usable rate-limit data.
    """
    if not isinstance(payload, dict):
        return None

    rate_limit = payload.get("rate_limit")
    rate_limit = rate_limit if isinstance(rate_limit, dict) else {}
    primary = _window_from_usage_json(rate_limit.get("primary_window"))
    secondary = _window_from_usage_json(rate_limit.get("secondary_window"))

    credits: CodexCreditsSnapshot | None = None
    cred = payload.get("credits")
    if isinstance(cred, dict) and cred.get("has_credits") is not None:
        has = bool(cred.get("has_credits"))
        raw_balance = cred.get("balance")
        credits = CodexCreditsSnapshot(
            has_credits=has,
            unlimited=bool(cred.get("unlimited")),
            # Only surface a balance when the account actually has credits;
            # a "0" balance on a no-credits plan is noise to the gauge.
            balance=(str(raw_balance) if has and raw_balance not in (None, "") else None),
        )

    promo = payload.get("promo")
    if isinstance(promo, dict):
        promo_message = promo.get("message")
    elif isinstance(promo, str):
        promo_message = promo
    else:
        promo_message = None
    promo_message = (promo_message or "").strip() or None

    raw_limit_name = payload.get("rate_limit_reached_type")
    limit_name = (
        raw_limit_name.strip()
        if isinstance(raw_limit_name, str) and raw_limit_name.strip()
        else None
    )

    if primary is None and secondary is None and credits is None and promo_message is None:
        return None

    return CodexRateLimitSnapshot(
        limit_id="codex",
        limit_name=limit_name,
        primary=primary,
        secondary=secondary,
        credits=credits,
        promo_message=promo_message,
    )


# ---------------------------------------------------------------------------
# Singleton state store
# ---------------------------------------------------------------------------


class CodexRateLimitState(QuotaTracker):
    """Thread-safe store for the latest Codex rate-limit snapshot.

    Implements :class:`~headroom.subscription.base.QuotaTracker` so it can
    be registered with the :class:`~headroom.subscription.base.QuotaTrackerRegistry`.
    This tracker is *passive* — it is updated by the OpenAI proxy handler
    each time a response containing ``x-codex-*`` headers passes through
    headroom, so :meth:`start` and :meth:`stop` are no-ops.
    """

    # QuotaTracker identity
    key = "codex_rate_limits"
    label = "OpenAI Codex"

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: CodexRateLimitSnapshot | None = None
        self._last_poll_monotonic: float = 0.0
        self._poll_inflight: bool = False

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update state from a response header dict (no-op if no Codex headers)."""
        snapshot = parse_codex_rate_limits(headers)
        if snapshot is None:
            return
        with self._lock:
            self._latest = snapshot

    def update_from_usage_payload(self, payload: object) -> bool:
        """Update state from a ``GET /wham/usage`` JSON body.

        Returns ``True`` when a snapshot was stored.
        """
        snapshot = parse_codex_usage_payload(payload)
        if snapshot is None:
            return False
        with self._lock:
            self._latest = snapshot
        return True

    def _try_begin_poll(self, min_interval_s: float) -> bool:
        """Atomically claim a usage-poll slot.

        Returns ``False`` when a poll is already in flight or one ran within
        ``min_interval_s``. On ``True`` the caller MUST call :meth:`_end_poll`.
        """
        now = time.monotonic()
        with self._lock:
            if self._poll_inflight:
                return False
            if (now - self._last_poll_monotonic) < min_interval_s:
                return False
            self._poll_inflight = True
            self._last_poll_monotonic = now
            return True

    def _end_poll(self) -> None:
        with self._lock:
            self._poll_inflight = False

    @property
    def latest(self) -> CodexRateLimitSnapshot | None:
        with self._lock:
            return self._latest

    def get_stats(self) -> dict | None:
        snap = self.latest
        return snap.to_dict() if snap is not None else None


_state: CodexRateLimitState | None = None
_state_lock = Lock()


def get_codex_rate_limit_state() -> CodexRateLimitState:
    """Return the process-global :class:`CodexRateLimitState` singleton."""
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = CodexRateLimitState()
    return _state


# ---------------------------------------------------------------------------
# Live usage poll (GET /wham/usage)
# ---------------------------------------------------------------------------


def _build_usage_headers(request_headers: dict[str, str]) -> dict[str, str] | None:
    """Build outbound /wham/usage headers from a client's request headers.

    Returns ``None`` unless the request carries a bearer token *and* a
    ``ChatGPT-Account-Id`` -- the latter scopes the poll to ChatGPT OAuth
    sessions (Codex), so API-key and non-Codex OAuth traffic never triggers it.
    """
    lower = {str(k).lower(): v for k, v in request_headers.items()}
    auth = str(lower.get("authorization", ""))
    if not auth.startswith("Bearer ") or not auth[len("Bearer ") :].strip():
        return None
    account_id = lower.get("chatgpt-account-id")
    if not account_id:
        return None

    headers = {
        "Authorization": auth,
        "ChatGPT-Account-Id": str(account_id),
        "Accept": "application/json",
    }
    # Mirror the client's own UA/originator so the request looks like Codex.
    for src, dst in (("user-agent", "User-Agent"), ("originator", "originator")):
        val = lower.get(src)
        if val:
            headers[dst] = str(val)
    return headers


async def _fetch_and_store_usage(url: str, headers: dict[str, str]) -> None:
    state = get_codex_rate_limit_state()
    try:
        async with httpx.AsyncClient(timeout=_USAGE_POLL_TIMEOUT_S) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            if state.update_from_usage_payload(resp.json()):
                logger.debug("codex usage poll: refreshed rate-limit window")
            else:
                logger.debug("codex usage poll: 200 but no usable rate-limit data")
        else:
            logger.debug("codex usage poll: HTTP %s", resp.status_code)
    except Exception as exc:  # pragma: no cover - network/JSON defensive
        logger.debug("codex usage poll failed: %s", exc)
    finally:
        state._end_poll()


def maybe_schedule_usage_poll(
    request_headers: dict[str, str],
    *,
    url: str = CODEX_USAGE_URL,
    min_interval_s: float = USAGE_POLL_MIN_INTERVAL_S,
) -> bool:
    """Fire-and-forget a throttled ``GET /wham/usage`` to refresh the window.

    Safe to call on every Codex request: scoped to ChatGPT-session traffic via
    :func:`_build_usage_headers` and internally throttled to at most one live
    poll per ``min_interval_s``. Returns ``True`` when a poll was scheduled.
    """
    headers = _build_usage_headers(request_headers)
    if headers is None:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    state = get_codex_rate_limit_state()
    if not state._try_begin_poll(min_interval_s):
        return False
    loop.create_task(_fetch_and_store_usage(url, headers))
    return True
