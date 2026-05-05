"""Data models for Anthropic subscription window tracking.

Mirrors the Anthropic OAuth usage API response exactly, including:
  - five_hour / seven_day rolling windows (utilization + reset times)
  - seven_day_opus / seven_day_sonnet per-model 7-day windows
  - extra_usage overage block (credits stored in cents by Anthropic)
  - Headroom contribution: tokens conserved by compression, rtk, cache
  - Window discrepancy detection (surge pricing, cache-miss anomalies)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Rate-limit window (five_hour / seven_day / seven_day_opus / seven_day_sonnet)
# ---------------------------------------------------------------------------


@dataclass
class RateLimitWindow:
    """A single rolling rate-limit window returned by the Anthropic usage API.

    ``used`` and ``limit`` are in Anthropic's internal token-equivalent units
    (not raw tokens; Anthropic weights tokens differently per model family).
    ``utilization_pct`` is the authoritative 0–100 % figure from the API.
    """

    used: int = 0
    limit: int = 0
    utilization_pct: float = 0.0
    resets_at: datetime | None = None

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> RateLimitWindow:
        return cls(
            used=int(data.get("used") or 0),
            limit=int(data.get("limit") or 0),
            utilization_pct=float(data.get("utilization") or 0.0),
            resets_at=_parse_timestamp(data.get("resets_at")),
        )

    def seconds_to_reset(self, *, now: datetime | None = None) -> float | None:
        if self.resets_at is None:
            return None
        return max((self.resets_at - (now or _utc_now())).total_seconds(), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "limit": self.limit,
            "utilization_pct": round(self.utilization_pct, 2),
            "resets_at": _to_utc_iso(self.resets_at) if self.resets_at else None,
            "seconds_to_reset": self.seconds_to_reset(),
        }


# ---------------------------------------------------------------------------
# Display-time synthesis
# ---------------------------------------------------------------------------


def synthesize_window_render(
    window: RateLimitWindow | None,
    *,
    used_since_reset: int | None,
    now: datetime | None = None,
    window_duration: timedelta,
    window_name: str = "window",
) -> dict[str, Any]:
    """Render a rate-limit window for the dashboard, synthesizing post-reset.

    If ``now`` is past ``window.resets_at``, the cached snapshot is stale: we
    return a synthesized dict whose ``used`` is the local transcript-derived
    token count since the reset (capped at ``limit`` so we never display
    >100%; we undercount tokens spent on Claude Code outside this proxy and
    must never report >100%), and whose ``resets_at`` advances by
    ``window_duration`` (marked ``estimated``). Otherwise the cached values
    are returned verbatim with ``synthesized=False``.

    On any unexpected data shape the function logs a warning and falls back
    to the cached values with ``synthesized=False`` and a ``render_warning``
    string — never raises into the caller.

    Args:
        window: The cached ``RateLimitWindow`` from the most recent poll.
        used_since_reset: Locally-counted token usage strictly after
            ``window.resets_at`` (None if we couldn't compute it).
        now: Override the wall clock for testability. Defaults to UTC now.
        window_duration: Length of the rolling window (e.g. 5h or 7d).
        window_name: Label used in structured logs.

    Returns:
        A dict matching :meth:`RateLimitWindow.to_dict` plus the keys
        ``synthesized: bool``, ``resets_at_estimated: bool`` and optional
        ``render_warning: str``.
    """
    if window is None:
        return {
            "used": 0,
            "limit": 0,
            "utilization_pct": 0.0,
            "resets_at": None,
            "seconds_to_reset": None,
            "synthesized": False,
            "resets_at_estimated": False,
        }

    cached = window.to_dict()
    cached["synthesized"] = False
    cached["resets_at_estimated"] = False

    if window.resets_at is None:
        return cached

    current_now = now if now is not None else _utc_now()

    try:
        if current_now < window.resets_at:
            logger.debug(
                "event=subscription_window_render_cached "
                "window=%s used=%d limit=%d utilization_pct=%.2f",
                window_name,
                window.used,
                window.limit,
                window.utilization_pct,
            )
            return cached

        # Past the reset boundary — synthesize.
        limit = max(int(window.limit), 0)
        used_local = int(used_since_reset) if used_since_reset is not None else 0
        if used_local < 0:
            used_local = 0
        # Cap at limit: we undercount tokens spent on Claude Code outside this
        # proxy; never report >100%.
        capped_used = min(used_local, limit) if limit > 0 else used_local
        utilization_pct = (capped_used / limit * 100.0) if limit > 0 else 0.0

        # Walk forward by whole window_durations from the observed reset to
        # land strictly after `current_now` — handles the rare case where the
        # dashboard is loaded long after the reset (e.g. machine was asleep).
        next_reset = window.resets_at
        while next_reset <= current_now:
            next_reset = next_reset + window_duration

        seconds_to_reset = max((next_reset - current_now).total_seconds(), 0.0)

        logger.debug(
            "event=subscription_window_synthesized "
            "window=%s used=%d limit=%d utilization_pct=%.2f "
            "resets_at_estimated=%s",
            window_name,
            capped_used,
            limit,
            utilization_pct,
            _to_utc_iso(next_reset),
        )

        return {
            "used": capped_used,
            "limit": limit,
            "utilization_pct": round(utilization_pct, 2),
            "resets_at": _to_utc_iso(next_reset),
            "seconds_to_reset": seconds_to_reset,
            "synthesized": True,
            "resets_at_estimated": True,
        }
    except Exception as exc:
        # Hard guarantee: never crash the dashboard. Loud warning so the
        # operator can fix the underlying data shape.
        logger.warning(
            "event=subscription_render_synthesis_failed window=%s error=%s",
            window_name,
            exc,
        )
        cached["render_warning"] = f"synthesis_failed: {exc}"
        return cached


# ---------------------------------------------------------------------------
# Extra-usage / overage block
# ---------------------------------------------------------------------------


@dataclass
class ExtraUsage:
    """Overage / extra-usage block from the Anthropic usage API.

    ``monthly_limit_cents`` and ``used_credits_cents`` are in US cents as
    returned by the API (divide by 100 for USD).
    """

    is_enabled: bool = False
    monthly_limit_cents: int | None = None
    used_credits_cents: int | None = None
    utilization_pct: float | None = None

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> ExtraUsage:
        return cls(
            is_enabled=bool(data.get("is_enabled", False)),
            monthly_limit_cents=_safe_int(data.get("monthly_limit")),
            used_credits_cents=_safe_int(data.get("used_credits")),
            utilization_pct=_safe_float(data.get("utilization")),
        )

    @property
    def monthly_limit_usd(self) -> float | None:
        if self.monthly_limit_cents is None:
            return None
        return self.monthly_limit_cents / 100.0

    @property
    def used_credits_usd(self) -> float | None:
        if self.used_credits_cents is None:
            return None
        return self.used_credits_cents / 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_enabled": self.is_enabled,
            "monthly_limit_usd": round(self.monthly_limit_usd, 2)
            if self.monthly_limit_usd is not None
            else None,
            "used_credits_usd": round(self.used_credits_usd, 4)
            if self.used_credits_usd is not None
            else None,
            "utilization_pct": round(self.utilization_pct, 2)
            if self.utilization_pct is not None
            else None,
        }


# ---------------------------------------------------------------------------
# Full snapshot from one API poll
# ---------------------------------------------------------------------------


@dataclass
class SubscriptionSnapshot:
    """One complete poll of GET /api/oauth/usage."""

    five_hour: RateLimitWindow = field(default_factory=RateLimitWindow)
    seven_day: RateLimitWindow = field(default_factory=RateLimitWindow)
    seven_day_opus: RateLimitWindow | None = None
    seven_day_sonnet: RateLimitWindow | None = None
    extra_usage: ExtraUsage = field(default_factory=ExtraUsage)
    polled_at: datetime = field(default_factory=_utc_now)
    token_prefix: str = ""
    """First 8 chars of the OAuth token (for multi-account detection)."""

    @classmethod
    def from_api_response(cls, data: dict[str, Any], *, token: str = "") -> SubscriptionSnapshot:
        snap = cls(token_prefix=token[:8] if token else "")
        if "five_hour" in data and data["five_hour"]:
            snap.five_hour = RateLimitWindow.from_api_dict(data["five_hour"])
        if "seven_day" in data and data["seven_day"]:
            snap.seven_day = RateLimitWindow.from_api_dict(data["seven_day"])
        if "seven_day_opus" in data and data["seven_day_opus"]:
            snap.seven_day_opus = RateLimitWindow.from_api_dict(data["seven_day_opus"])
        if "seven_day_sonnet" in data and data["seven_day_sonnet"]:
            snap.seven_day_sonnet = RateLimitWindow.from_api_dict(data["seven_day_sonnet"])
        if "extra_usage" in data and data["extra_usage"]:
            snap.extra_usage = ExtraUsage.from_api_dict(data["extra_usage"])
        return snap

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "five_hour": self.five_hour.to_dict(),
            "seven_day": self.seven_day.to_dict(),
            "extra_usage": self.extra_usage.to_dict(),
            "polled_at": _to_utc_iso(self.polled_at),
            "token_prefix": self.token_prefix,
        }
        if self.seven_day_opus:
            d["seven_day_opus"] = self.seven_day_opus.to_dict()
        if self.seven_day_sonnet:
            d["seven_day_sonnet"] = self.seven_day_sonnet.to_dict()
        return d


# ---------------------------------------------------------------------------
# Transcript-based window token breakdown
# ---------------------------------------------------------------------------


@dataclass
class WindowTokens:
    """Token breakdown from Claude transcript JSONL files for one time window."""

    input: int = 0
    output: int = 0
    cache_reads: int = 0
    cache_writes_5m: int = 0
    cache_writes_1h: int = 0
    cache_writes_total: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    weighted_token_equivalent: float = 0.0
    """Sonnet-normalised weighted total (opus×2, sonnet×1, haiku×0.5)."""

    def total_raw(self) -> int:
        return self.input + self.output + self.cache_reads + self.cache_writes_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_reads": self.cache_reads,
            "cache_writes_5m": self.cache_writes_5m,
            "cache_writes_1h": self.cache_writes_1h,
            "cache_writes_total": self.cache_writes_total,
            "total_raw": self.total_raw(),
            "weighted_token_equivalent": round(self.weighted_token_equivalent, 1),
            "by_model": self.by_model,
        }


# ---------------------------------------------------------------------------
# Headroom contribution estimate
# ---------------------------------------------------------------------------


@dataclass
class HeadroomContribution:
    """Tokens conserved within the current 5h window by Headroom's layers.

    These are cumulative counters reset when the 5h window rolls over.
    """

    tokens_submitted: int = 0
    """Raw input tokens actually forwarded to Anthropic by the proxy."""

    tokens_saved_compression: int = 0
    """Input tokens removed by proxy compression."""

    tokens_saved_rtk: int = 0
    """Tokens avoided by CLI filtering (rtk) before reaching context."""

    tokens_saved_cache_reads: int = 0
    """Input tokens served from Anthropic prefix-cache (discounted reads)."""

    compression_savings_usd: float = 0.0
    cache_savings_usd: float = 0.0

    def total_saved(self) -> int:
        return self.tokens_saved_compression + self.tokens_saved_rtk + self.tokens_saved_cache_reads

    def total_savings_usd(self) -> float:
        return self.compression_savings_usd + self.cache_savings_usd

    def raw_without_headroom(self) -> int:
        return self.tokens_submitted + self.tokens_saved_compression + self.tokens_saved_rtk

    def efficiency_pct(self) -> float:
        raw = self.raw_without_headroom()
        if raw == 0:
            return 0.0
        return round(self.total_saved() / raw * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_submitted": self.tokens_submitted,
            "tokens_saved": {
                "compression": self.tokens_saved_compression,
                "rtk": self.tokens_saved_rtk,
                "cache_reads": self.tokens_saved_cache_reads,
                "total": self.total_saved(),
            },
            "raw_without_headroom": self.raw_without_headroom(),
            "efficiency_pct": self.efficiency_pct(),
            "savings_usd": {
                "compression": round(self.compression_savings_usd, 4),
                "cache": round(self.cache_savings_usd, 4),
                "total": round(self.total_savings_usd(), 4),
            },
        }


# ---------------------------------------------------------------------------
# Anomaly / discrepancy record
# ---------------------------------------------------------------------------


@dataclass
class WindowDiscrepancy:
    """Detected anomaly between expected and API-reported utilization."""

    kind: str
    """'surge_pricing' | 'cache_miss' | 'none'"""

    description: str = ""
    severity: str = "info"
    """'info' | 'warning' | 'alert'"""

    expected_utilization_pct: float | None = None
    actual_utilization_pct: float | None = None
    delta_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "severity": self.severity,
            "expected_utilization_pct": self.expected_utilization_pct,
            "actual_utilization_pct": self.actual_utilization_pct,
            "delta_pct": self.delta_pct,
        }


# ---------------------------------------------------------------------------
# Full tracker state
# ---------------------------------------------------------------------------


@dataclass
class SubscriptionState:
    """Persistent state for the subscription tracker."""

    latest: SubscriptionSnapshot | None = None
    window_tokens: WindowTokens | None = None
    """Transcript-derived token breakdown for the current 5h window."""

    contribution: HeadroomContribution = field(default_factory=HeadroomContribution)
    discrepancies: list[WindowDiscrepancy] = field(default_factory=list)
    history: list[SubscriptionSnapshot] = field(default_factory=list)

    poll_count: int = 0
    poll_errors: int = 0
    last_error: str | None = None
    last_active_at: datetime | None = None

    _MAX_HISTORY: int = field(default=100, init=False, repr=False)
    _MAX_DISCREPANCIES: int = field(default=20, init=False, repr=False)

    def add_snapshot(self, snapshot: SubscriptionSnapshot) -> None:
        self.latest = snapshot
        self.history.append(snapshot)
        if len(self.history) > self._MAX_HISTORY:
            self.history = self.history[-self._MAX_HISTORY :]
        self.poll_count += 1

    def mark_error(self, msg: str) -> None:
        self.poll_errors += 1
        self.last_error = msg

    def add_discrepancy(self, d: WindowDiscrepancy) -> None:
        self.discrepancies.append(d)
        if len(self.discrepancies) > self._MAX_DISCREPANCIES:
            self.discrepancies = self.discrepancies[-self._MAX_DISCREPANCIES :]

    def is_active(self, *, active_window_s: float = 60.0) -> bool:
        if self.last_active_at is None:
            return False
        return (_utc_now() - self.last_active_at).total_seconds() <= active_window_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest": self.latest.to_dict() if self.latest else None,
            "window_tokens": self.window_tokens.to_dict() if self.window_tokens else None,
            "contribution": self.contribution.to_dict(),
            "discrepancies": [d.to_dict() for d in self.discrepancies[-5:]],
            "poll_count": self.poll_count,
            "poll_errors": self.poll_errors,
            "last_error": self.last_error,
            "last_active_at": _to_utc_iso(self.last_active_at) if self.last_active_at else None,
        }

    def to_persist_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        d["history"] = [s.to_dict() for s in self.history[-20:]]
        return d
