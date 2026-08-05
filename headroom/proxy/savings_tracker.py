"""Durable proxy savings and display-session tracking.

Persists cumulative proxy compression savings plus a canonical display session
window to a local JSON file so historical charts and dashboard session stats
survive proxy restarts and can be shared by multiple Headroom frontends.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import tempfile
import threading
from csv import DictWriter
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from headroom import paths as _paths
from headroom.proxy import project_name_policy
from headroom.proxy.persistent_metrics import PersistentMetricsState

PROJECT_NAME_MAX_LENGTH = project_name_policy.PROJECT_NAME_MAX_LENGTH
sanitize_project_name = project_name_policy.sanitize_project_name

logger = logging.getLogger(__name__)

HEADROOM_SAVINGS_PATH_ENV_VAR = _paths.HEADROOM_SAVINGS_PATH_ENV
DEFAULT_SAVINGS_DIR = ".headroom"
DEFAULT_SAVINGS_FILE = "proxy_savings.json"
SCHEMA_VERSION = 5
DEFAULT_MAX_HISTORY_POINTS = 5000
DEFAULT_MAX_PROJECTS = 50
DEFAULT_MAX_HISTORY_AGE_DAYS = 365
DEFAULT_MAX_RESPONSE_HISTORY_POINTS = 500
DEFAULT_DISPLAY_SESSION_INACTIVITY_MINUTES = 60
DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
# Blended output price used only when litellm cannot price the model.
DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000

LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None
litellm: Any | None = None


def _get_litellm_module() -> Any | None:
    """Import LiteLLM only when cost metadata is requested."""
    global litellm

    if not LITELLM_AVAILABLE:
        return None
    if litellm is not None:
        return litellm

    try:
        import litellm as imported_litellm
    except ImportError:
        return None

    litellm = imported_litellm
    return litellm


def get_default_savings_storage_path() -> str:
    """Return the configured savings storage path."""
    # Preserve legacy behavior: when HEADROOM_SAVINGS_PATH is set we return
    # the raw string exactly as supplied (no tilde expansion, no
    # path-separator normalization) to match prior behavior and existing tests.
    env_path = os.environ.get(HEADROOM_SAVINGS_PATH_ENV_VAR, "").strip()
    if env_path:
        return env_path
    return str(_paths.savings_path())


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


def _bucket_start(timestamp: datetime, bucket: str) -> datetime:
    if bucket == "hour":
        return timestamp.replace(minute=0, second=0, microsecond=0)
    if bucket == "day":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket == "week":
        day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start - timedelta(days=day_start.weekday())
    if bucket == "month":
        return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported savings history bucket: {bucket}")


def _coerce_int(value: Any, default: int = 0) -> int:
    # OverflowError: int(float("inf")) — json accepts bare Infinity, and a
    # corrupted state file must not crash proxy startup.
    try:
        return max(int(value), 0)
    except (TypeError, ValueError, OverflowError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    # NaN is absorbing under += — one poisoned value would brick an
    # accumulator forever, so reject non-finite values outright.
    try:
        coerced = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(coerced):
        return default
    return max(coerced, 0.0)


PROVIDER_UNKNOWN = "unknown"


def _normalize_provider(value: Any) -> str:
    """Normalize a provider label, falling back to a stable sentinel.

    History checkpoints persisted before per-provider attribution existed have
    no provider field, so they collapse into ``PROVIDER_UNKNOWN`` rather than
    silently dropping their savings from the per-provider breakdown.
    """
    if not isinstance(value, str):
        return PROVIDER_UNKNOWN
    cleaned = value.strip()
    return cleaned or PROVIDER_UNKNOWN


MODEL_UNKNOWN = "unknown"


def _normalize_model(value: Any) -> str:
    """Normalize a model label, falling back to a stable sentinel.

    History checkpoints persisted before per-model attribution existed have
    no model field, so they collapse into ``MODEL_UNKNOWN`` rather than
    silently dropping their savings from the per-model breakdown.
    """
    if not isinstance(value, str):
        return MODEL_UNKNOWN
    cleaned = value.strip()
    return cleaned or MODEL_UNKNOWN


def _resolve_litellm_model(model: str) -> str:
    """Resolve model name to one LiteLLM recognizes.

    Delegates to the shared alias-map-aware resolver in
    ``headroom.pricing.litellm_pricing`` so the persisted /stats-history funnel
    (PROXY $ SAVED tile + Historical Checkpoints) prices gateway aliases like
    "claude-opus" identically to the live /stats path. Uses the shared result
    only when it maps to a priced model_cost key; otherwise falls through to the
    bare-prefix logic below. Fail-soft: pricing never breaks bookkeeping.
    """
    litellm = _get_litellm_module()
    if litellm is None:
        return model

    try:
        from headroom.pricing.litellm_pricing import resolve_litellm_model

        resolved = resolve_litellm_model(model)
        info = litellm.model_cost.get(resolved)
        if info and info.get("input_cost_per_token") is not None:
            return resolved
    except Exception:
        pass

    try:
        litellm.cost_per_token(model=model, prompt_tokens=1, completion_tokens=0)
        return model
    except Exception:
        pass

    prefixes = {
        "claude-": "anthropic/",
        "gpt-": "openai/",
        "o1-": "openai/",
        "o3-": "openai/",
        "o4-": "openai/",
        "gemini-": "google/",
    }
    for pattern, prefix in prefixes.items():
        if model.startswith(pattern):
            candidate = f"{prefix}{model}"
            try:
                litellm.cost_per_token(
                    model=candidate,
                    prompt_tokens=1,
                    completion_tokens=0,
                )
                return candidate
            except Exception:
                break

    return model


def _estimate_compression_savings_usd(model: str, tokens_saved: int) -> float:
    """Estimate compression savings in USD from saved input tokens."""
    litellm = _get_litellm_module()
    if tokens_saved <= 0:
        return 0.0
    if litellm is None:
        return float(tokens_saved) * float(DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN)

    try:
        resolved = _resolve_litellm_model(model)
        info = litellm.model_cost.get(resolved, {})
        input_cost_per_token = info.get("input_cost_per_token")
        # Distinguish "price unknown" (missing key → fall back) from a model that
        # is legitimately free (input_cost_per_token == 0.0). `if not ...` treated
        # a real 0.0 as unavailable and billed the $3/M fallback — phantom savings
        # for a model that costs nothing.
        if input_cost_per_token is None:
            raise RuntimeError("input cost unavailable")
        return float(tokens_saved) * float(input_cost_per_token)
    except Exception:
        return float(tokens_saved) * float(DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN)


def _estimate_output_savings_usd(model: str, tokens_saved: int) -> float:
    """Estimate output-shaping savings in USD from saved *output* tokens.

    Mirrors ``_estimate_compression_savings_usd`` but prices at the model's
    output rate, since the shaper reduces generated (output) tokens, not input.
    """
    litellm = _get_litellm_module()
    if tokens_saved <= 0:
        return 0.0
    if litellm is None:
        return float(tokens_saved) * float(DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN)
    try:
        resolved = _resolve_litellm_model(model)
        info = litellm.model_cost.get(resolved, {})
        output_cost_per_token = info.get("output_cost_per_token")
        # Distinguish "price unknown" (missing key -> fall back to the estimate)
        # from a model that is legitimately free (output_cost_per_token == 0.0).
        # `if not ...` treated a real 0.0 as unavailable and billed the fallback
        # rate -> phantom output savings for a model that costs nothing. Mirrors
        # the fix already applied to `_estimate_compression_savings_usd`.
        if output_cost_per_token is None:
            raise RuntimeError("output cost unavailable")
        return float(tokens_saved) * float(output_cost_per_token)
    except Exception:
        return float(tokens_saved) * float(DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN)


def _estimate_cache_savings_usd(model: str, cache_read_tokens: int) -> float:
    """Estimate cache-read savings in USD — the discount delta vs list price.

    Cache reads bill at the provider's discounted rate, so the saving per token
    is ``input_cost_per_token - cache_read_input_token_cost``. Unknown models
    price as 0.0 (fail open); tokens still accumulate. An unavailable litellm
    falls back to ``DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN``, matching
    ``_estimate_input_cost_usd``/``_estimate_compression_savings_usd`` — otherwise
    cache_savings_usd silently reads as $0 forever on any install without
    litellm (e.g. Python 3.14, where headroom's own dependency spec excludes it).

    Deliberately diverges from ``proxy/cost.py``'s session-scoped provider
    multipliers (``_CACHE_ECONOMICS``): this lifetime figure follows the
    per-model litellm pricing the rest of this module already uses.
    """
    litellm = _get_litellm_module()
    if cache_read_tokens <= 0:
        return 0.0
    if litellm is None:
        return float(cache_read_tokens) * float(DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN)

    try:
        resolved = _resolve_litellm_model(model)
        info = litellm.model_cost.get(resolved, {})
        input_cost_per_token = info.get("input_cost_per_token")
        if not input_cost_per_token:
            return 0.0
        cache_read_cost = info.get("cache_read_input_token_cost", input_cost_per_token)
        discount = float(input_cost_per_token) - float(cache_read_cost)
        if discount <= 0:
            return 0.0
        return float(cache_read_tokens) * discount
    except Exception:
        return 0.0


def _estimate_input_cost_usd(
    model: str,
    input_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    uncached_input_tokens: int = 0,
) -> float:
    """Estimate input spend in USD for a request.

    Uses provider cache pricing when a complete cache breakdown is available and
    otherwise falls back to list-price input tokens.
    """
    total_input_tokens = _coerce_int(input_tokens)
    cache_read = _coerce_int(cache_read_tokens)
    cache_write = _coerce_int(cache_write_tokens)
    uncached = _coerce_int(uncached_input_tokens)

    # Prefer the breakdown when callers supply segmented token counts.
    # Never add `input_tokens` on top of the breakdown to avoid double-counting.
    use_breakdown = (cache_read + cache_write + uncached) > 0
    chargeable_tokens = (
        (cache_read + cache_write + uncached) if use_breakdown else total_input_tokens
    )
    if chargeable_tokens <= 0:
        return 0.0

    litellm = _get_litellm_module()
    # Keep exact provider pricing authoritative when available.
    # `litellm` can be present but lack an entry for the resolved model,
    # in which case we fall back to a blended rate instead of zeroing usage.
    if litellm is None:
        return float(chargeable_tokens) * float(DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN)

    try:
        resolved = _resolve_litellm_model(model)
        info = litellm.model_cost.get(resolved, {})
        input_cost_per_token = info.get("input_cost_per_token")
        # A missing key means the model is unknown → fall back to a blended rate.
        # A present 0.0 means the model is free and must cost $0, not the fallback.
        if input_cost_per_token is None:
            raise RuntimeError("input cost unavailable")

        if use_breakdown:
            cache_read_cost = info.get(
                "cache_read_input_token_cost",
                input_cost_per_token,
            )
            cache_write_cost = info.get(
                "cache_creation_input_token_cost",
                input_cost_per_token,
            )
            return (
                float(cache_read) * float(cache_read_cost)
                + float(cache_write) * float(cache_write_cost)
                + float(uncached) * float(input_cost_per_token)
            )

        return float(total_input_tokens) * float(input_cost_per_token)
    except Exception:
        return float(chargeable_tokens) * float(DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN)


def _normalize_history_entry(entry: Any) -> dict[str, Any] | None:
    """Normalize persisted history entries across schema shapes."""
    timestamp: datetime | None = None
    total_tokens_saved = 0
    compression_savings_usd = 0.0
    cache_read_tokens = 0
    cache_savings_usd = 0.0
    total_input_tokens = 0
    total_input_cost_usd = 0.0
    output_tokens_saved = 0
    output_savings_usd = 0.0
    provider = PROVIDER_UNKNOWN
    model = MODEL_UNKNOWN

    if isinstance(entry, dict):
        timestamp = _parse_timestamp(entry.get("timestamp"))
        total_tokens_saved = _coerce_int(entry.get("total_tokens_saved"))
        compression_savings_usd = _coerce_float(entry.get("compression_savings_usd"))
        # Older history points predate cache-savings tracking and omit these
        # keys entirely; default to 0/0.0 rather than raising or dropping the
        # entry, matching how every other field here handles legacy shapes.
        cache_read_tokens = _coerce_int(entry.get("cache_read_tokens"))
        cache_savings_usd = _coerce_float(entry.get("cache_savings_usd"))
        total_input_tokens = _coerce_int(entry.get("total_input_tokens"))
        total_input_cost_usd = _coerce_float(entry.get("total_input_cost_usd"))
        output_tokens_saved = _coerce_int(entry.get("output_tokens_saved"))
        output_savings_usd = _coerce_float(entry.get("output_savings_usd"))
        provider = _normalize_provider(entry.get("provider"))
        model = _normalize_model(entry.get("model"))
    elif isinstance(entry, list | tuple) and len(entry) >= 2:
        timestamp = _parse_timestamp(entry[0])
        total_tokens_saved = _coerce_int(entry[1])
        if len(entry) >= 3:
            compression_savings_usd = _coerce_float(entry[2])
        if len(entry) >= 4:
            total_input_tokens = _coerce_int(entry[3])
        if len(entry) >= 5:
            total_input_cost_usd = _coerce_float(entry[4])
    else:
        return None

    if timestamp is None:
        return None

    return {
        "timestamp": _to_utc_iso(timestamp),
        "provider": provider,
        "model": model,
        "total_tokens_saved": total_tokens_saved,
        "compression_savings_usd": round(compression_savings_usd, 6),
        "cache_read_tokens": cache_read_tokens,
        "cache_savings_usd": round(cache_savings_usd, 6),
        "total_input_tokens": total_input_tokens,
        "total_input_cost_usd": round(total_input_cost_usd, 6),
        "output_tokens_saved": output_tokens_saved,
        "output_savings_usd": round(output_savings_usd, 6),
    }


def _empty_display_session() -> dict[str, Any]:
    return {
        "requests": 0,
        "tokens_saved": 0,
        "compression_savings_usd": 0.0,
        "cache_read_tokens": 0,
        "cache_savings_usd": 0.0,
        "total_input_tokens": 0,
        "total_input_cost_usd": 0.0,
        "savings_percent": 0.0,
        "started_at": None,
        "last_activity_at": None,
    }


def _empty_by_model_entry() -> dict[str, Any]:
    return {
        "requests": 0,
        "tokens_saved": 0,
        "compression_savings_usd": 0.0,
        "total_input_tokens": 0,
        "total_input_cost_usd": 0.0,
    }


def _empty_project_entry() -> dict[str, Any]:
    return {
        "requests": 0,
        "tokens_saved": 0,
        "compression_savings_usd": 0.0,
        "total_input_tokens": 0,
        "total_input_cost_usd": 0.0,
        "last_activity_at": None,
    }


def _normalize_projects(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    projects: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        cleaned_name = sanitize_project_name(name)
        if cleaned_name is None or not isinstance(entry, dict):
            continue
        normalized = _empty_project_entry()
        normalized["requests"] = _coerce_int(entry.get("requests"))
        normalized["tokens_saved"] = _coerce_int(entry.get("tokens_saved"))
        normalized["compression_savings_usd"] = round(
            _coerce_float(entry.get("compression_savings_usd")), 6
        )
        normalized["total_input_tokens"] = _coerce_int(entry.get("total_input_tokens"))
        normalized["total_input_cost_usd"] = round(
            _coerce_float(entry.get("total_input_cost_usd")), 6
        )
        last_activity = _parse_timestamp(entry.get("last_activity_at"))
        normalized["last_activity_at"] = _to_utc_iso(last_activity) if last_activity else None
        projects[cleaned_name] = normalized
    if len(projects) > DEFAULT_MAX_PROJECTS:
        # Oversized persisted maps (hand-edited or future versions) would
        # otherwise shrink only one entry per recorded request.
        kept = sorted(
            projects.items(),
            key=lambda item: (item[1]["tokens_saved"], item[1]["last_activity_at"] or ""),
            reverse=True,
        )[:DEFAULT_MAX_PROJECTS]
        projects = dict(kept)
    return projects


def _normalize_by_model(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for model_name, entry in raw.items():
        model = _normalize_model(model_name)
        if not isinstance(entry, dict):
            continue
        normalized = _empty_by_model_entry()
        normalized["requests"] = _coerce_int(entry.get("requests"))
        normalized["tokens_saved"] = _coerce_int(entry.get("tokens_saved"))
        normalized["compression_savings_usd"] = round(
            _coerce_float(entry.get("compression_savings_usd")), 6
        )
        normalized["total_input_tokens"] = _coerce_int(entry.get("total_input_tokens"))
        normalized["total_input_cost_usd"] = round(
            _coerce_float(entry.get("total_input_cost_usd")), 6
        )
        result[model] = normalized
    return result


def _normalize_display_session(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return _empty_display_session()

    started_at = _parse_timestamp(entry.get("started_at"))
    last_activity_at = _parse_timestamp(entry.get("last_activity_at"))

    if started_at is None or last_activity_at is None or last_activity_at < started_at:
        return _empty_display_session()

    tokens_saved = _coerce_int(entry.get("tokens_saved"))
    total_input_tokens = _coerce_int(entry.get("total_input_tokens"))
    total_before = tokens_saved + total_input_tokens
    savings_percent = round(
        (tokens_saved / total_before * 100) if total_before > 0 else 0.0,
        2,
    )

    return {
        "requests": _coerce_int(entry.get("requests")),
        "tokens_saved": tokens_saved,
        "compression_savings_usd": round(
            _coerce_float(entry.get("compression_savings_usd")),
            6,
        ),
        "cache_read_tokens": _coerce_int(entry.get("cache_read_tokens")),
        "cache_savings_usd": round(
            _coerce_float(entry.get("cache_savings_usd")),
            6,
        ),
        "total_input_tokens": total_input_tokens,
        "total_input_cost_usd": round(
            _coerce_float(entry.get("total_input_cost_usd")),
            6,
        ),
        "savings_percent": savings_percent,
        "started_at": _to_utc_iso(started_at),
        "last_activity_at": _to_utc_iso(last_activity_at),
    }


class SavingsTracker:
    """Persist bounded proxy compression savings history."""

    def __init__(
        self,
        path: str | None = None,
        max_history_points: int = DEFAULT_MAX_HISTORY_POINTS,
        max_history_age_days: int = DEFAULT_MAX_HISTORY_AGE_DAYS,
        max_response_history_points: int = DEFAULT_MAX_RESPONSE_HISTORY_POINTS,
        display_session_inactivity_minutes: int = (DEFAULT_DISPLAY_SESSION_INACTIVITY_MINUTES),
        stateless: bool = False,
        save_flush_every: int = 1,
    ) -> None:
        # In stateless mode the tracker keeps live counters in memory but never
        # writes proxy_savings.json (honors HeadroomConfig.stateless, which
        # disables all filesystem writes for read-only / container deployments).
        self._stateless = stateless
        self._path = Path(path or get_default_savings_storage_path())
        self._max_history_points = max_history_points
        self._max_history_age_days = max_history_age_days
        self._max_response_history_points = max(
            _coerce_int(
                max_response_history_points,
                DEFAULT_MAX_RESPONSE_HISTORY_POINTS,
            ),
            1,
        )
        self._display_session_inactivity_minutes = max(
            _coerce_int(
                display_session_inactivity_minutes,
                DEFAULT_DISPLAY_SESSION_INACTIVITY_MINUTES,
            ),
            1,
        )
        # ponytail: per-record save throttle. Default 1 = persist every call
        # (the durable default that direct/CLI callers rely on). The async proxy
        # opts into a higher value so it doesn't json.dumps + fsync the whole
        # history on every request. Lossless because _save_locked always writes
        # the FULL state — a skipped save just means the next one is complete.
        self._save_flush_every = max(_coerce_int(save_flush_every, 1), 1)
        self._since_save = 0
        self._lock = threading.Lock()
        self._persistence_healthy = True
        self._persistence_error: str | None = None
        self._needs_schema_save = False
        self._state = self._load_state()
        self._persistent_metrics = PersistentMetricsState(self._state.pop("lifetime_metrics", None))

    @property
    def storage_path(self) -> str:
        return str(self._path)

    def record_compression_savings(
        self,
        *,
        model: str,
        tokens_saved: int,
        provider: str | None = None,
        total_input_tokens: int | None = None,
        total_input_cost_usd: float | None = None,
        timestamp: datetime | str | None = None,
    ) -> bool:
        """Persist a cumulative savings checkpoint when compression changed totals."""
        delta_tokens = _coerce_int(tokens_saved)
        if delta_tokens <= 0:
            return False

        timestamp_dt = (
            _parse_timestamp(timestamp)
            if isinstance(timestamp, str)
            else timestamp.astimezone(timezone.utc)
            if isinstance(timestamp, datetime)
            else _utc_now()
        )
        if timestamp_dt is None:
            timestamp_dt = _utc_now()

        delta_usd = _estimate_compression_savings_usd(model, delta_tokens)

        with self._lock:
            lifetime = self._state["lifetime"]
            lifetime["tokens_saved"] += delta_tokens
            lifetime["compression_savings_usd"] = round(
                lifetime["compression_savings_usd"] + delta_usd, 6
            )
            lifetime["total_input_tokens"] = max(
                lifetime["total_input_tokens"],
                _coerce_int(total_input_tokens, default=lifetime["total_input_tokens"]),
            )
            lifetime["total_input_cost_usd"] = round(
                max(
                    lifetime["total_input_cost_usd"],
                    _coerce_float(
                        total_input_cost_usd,
                        default=lifetime["total_input_cost_usd"],
                    ),
                ),
                6,
            )

            self._record_by_model_locked(
                model,
                tokens_saved_delta=delta_tokens,
                savings_usd_delta=delta_usd,
            )

            self._state["history"].append(
                {
                    "timestamp": _to_utc_iso(timestamp_dt),
                    "provider": _normalize_provider(provider),
                    "model": _normalize_model(model),
                    "total_tokens_saved": lifetime["tokens_saved"],
                    "compression_savings_usd": lifetime["compression_savings_usd"],
                    "total_input_tokens": lifetime["total_input_tokens"],
                    "total_input_cost_usd": lifetime["total_input_cost_usd"],
                }
            )
            self._trim_history_locked(reference_time=timestamp_dt)
            self._maybe_save_locked()
            return True

    def record_request(
        self,
        *,
        model: str,
        input_tokens: int,
        tokens_saved: int,
        output_tokens_saved: int = 0,
        provider: str | None = None,
        project: str | None = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        uncached_input_tokens: int = 0,
        total_input_tokens: int | None = None,
        total_input_cost_usd: float | None = None,
        timestamp: datetime | str | None = None,
    ) -> bool:
        """Persist a canonical display-session update for every request."""
        timestamp_dt = (
            _parse_timestamp(timestamp)
            if isinstance(timestamp, str)
            else timestamp.astimezone(timezone.utc)
            if isinstance(timestamp, datetime)
            else _utc_now()
        )
        if timestamp_dt is None:
            timestamp_dt = _utc_now()

        delta_tokens_saved = _coerce_int(tokens_saved)
        delta_input_tokens = _coerce_int(input_tokens)
        delta_savings_usd = _estimate_compression_savings_usd(model, delta_tokens_saved)
        delta_output_tokens_saved = max(_coerce_int(output_tokens_saved), 0)
        delta_output_savings_usd = _estimate_output_savings_usd(model, delta_output_tokens_saved)
        delta_cache_read_tokens = _coerce_int(cache_read_tokens)
        delta_cache_savings_usd = _estimate_cache_savings_usd(model, delta_cache_read_tokens)
        delta_input_cost_usd = _estimate_input_cost_usd(
            model,
            delta_input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            uncached_input_tokens=uncached_input_tokens,
        )

        with self._lock:
            lifetime = self._state["lifetime"]
            previous_total_input_tokens = lifetime["total_input_tokens"]
            previous_total_input_cost_usd = lifetime["total_input_cost_usd"]

            next_total_input_tokens = max(
                previous_total_input_tokens + delta_input_tokens,
                _coerce_int(
                    total_input_tokens,
                    default=previous_total_input_tokens + delta_input_tokens,
                ),
            )
            next_total_input_cost_usd = round(
                max(
                    previous_total_input_cost_usd + delta_input_cost_usd,
                    _coerce_float(
                        total_input_cost_usd,
                        default=previous_total_input_cost_usd + delta_input_cost_usd,
                    ),
                ),
                6,
            )
            session_input_tokens_delta = max(
                next_total_input_tokens - previous_total_input_tokens,
                0,
            )
            session_input_cost_delta = round(
                max(next_total_input_cost_usd - previous_total_input_cost_usd, 0.0),
                6,
            )

            lifetime["requests"] += 1
            lifetime["tokens_saved"] += delta_tokens_saved
            lifetime["compression_savings_usd"] = round(
                lifetime["compression_savings_usd"] + delta_savings_usd,
                6,
            )
            lifetime["cache_read_tokens"] += delta_cache_read_tokens
            lifetime["cache_savings_usd"] = round(
                lifetime["cache_savings_usd"] + delta_cache_savings_usd,
                6,
            )
            lifetime["total_input_tokens"] = next_total_input_tokens
            lifetime["total_input_cost_usd"] = next_total_input_cost_usd
            lifetime["output_tokens_saved"] = (
                lifetime.get("output_tokens_saved", 0) + delta_output_tokens_saved
            )
            lifetime["output_savings_usd"] = round(
                lifetime.get("output_savings_usd", 0.0) + delta_output_savings_usd,
                6,
            )

            session = self._state["display_session"]
            last_activity = _parse_timestamp(session.get("last_activity_at"))
            if last_activity is None or self._is_display_session_expired(
                last_activity,
                reference_time=timestamp_dt,
            ):
                session = _empty_display_session()
                session["started_at"] = _to_utc_iso(timestamp_dt)
                self._state["display_session"] = session

            session["requests"] += 1
            session["tokens_saved"] += delta_tokens_saved
            session["compression_savings_usd"] = round(
                session["compression_savings_usd"] + delta_savings_usd,
                6,
            )
            session["cache_read_tokens"] += delta_cache_read_tokens
            session["cache_savings_usd"] = round(
                session["cache_savings_usd"] + delta_cache_savings_usd,
                6,
            )
            session["total_input_tokens"] += session_input_tokens_delta
            session["total_input_cost_usd"] = round(
                session["total_input_cost_usd"] + session_input_cost_delta,
                6,
            )
            total_before = session["tokens_saved"] + session["total_input_tokens"]
            session["savings_percent"] = round(
                (session["tokens_saved"] / total_before * 100) if total_before > 0 else 0.0,
                2,
            )
            session["last_activity_at"] = _to_utc_iso(timestamp_dt)
            if session.get("started_at") is None:
                session["started_at"] = session["last_activity_at"]

            self._record_by_model_locked(
                model,
                requests_delta=1,
                tokens_saved_delta=delta_tokens_saved,
                savings_usd_delta=delta_savings_usd,
                input_tokens_delta=delta_input_tokens,
                input_cost_usd_delta=delta_input_cost_usd,
            )

            self._record_project_locked(
                project,
                timestamp_dt=timestamp_dt,
                requests_delta=1,
                tokens_saved_delta=delta_tokens_saved,
                savings_usd_delta=delta_savings_usd,
                input_tokens_delta=delta_input_tokens,
                input_cost_usd_delta=delta_input_cost_usd,
            )

            # In --mode cache, headroom's own compression (tokens_saved) is
            # near-always 0 by design — the frozen prefix is byte-replayed,
            # not lossy-compressed, to keep Bedrock's prompt cache warm. Gating
            # on tokens_saved alone silently dropped every history point on
            # those requests even though real cache-read savings occurred.
            # Append whenever any savings mechanism produced a saving.
            if (
                delta_tokens_saved > 0
                or delta_cache_read_tokens > 0
                or delta_output_tokens_saved > 0
            ):
                self._state["history"].append(
                    {
                        "timestamp": _to_utc_iso(timestamp_dt),
                        "provider": _normalize_provider(provider),
                        "model": _normalize_model(model),
                        "total_tokens_saved": lifetime["tokens_saved"],
                        "compression_savings_usd": lifetime["compression_savings_usd"],
                        "cache_read_tokens": lifetime["cache_read_tokens"],
                        "cache_savings_usd": lifetime["cache_savings_usd"],
                        "total_input_tokens": lifetime["total_input_tokens"],
                        "total_input_cost_usd": lifetime["total_input_cost_usd"],
                        "output_tokens_saved": lifetime.get("output_tokens_saved", 0),
                        "output_savings_usd": lifetime.get("output_savings_usd", 0.0),
                    }
                )
                self._trim_history_locked(reference_time=timestamp_dt)

            self._maybe_save_locked()
            return True

    def record_lifetime_request(self, *, persist: bool = True, **metrics: Any) -> None:
        """Record one completed request in the durable Lifetime aggregate."""

        model = _normalize_model(metrics.get("model"))
        input_tokens = _coerce_int(metrics.get("input_tokens"))
        cache_read_tokens = _coerce_int(metrics.get("cache_read_tokens"))
        cache_write_tokens = _coerce_int(metrics.get("cache_write_tokens"))
        uncached_input_tokens = _coerce_int(metrics.get("uncached_input_tokens"))
        metrics.setdefault(
            "input_usd",
            _estimate_input_cost_usd(
                model,
                input_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                uncached_input_tokens=uncached_input_tokens,
            ),
        )
        metrics.setdefault(
            "compression_savings_usd",
            _estimate_compression_savings_usd(model, _coerce_int(metrics.get("tokens_saved"))),
        )
        metrics.setdefault(
            "cache_savings_usd", _estimate_cache_savings_usd(model, cache_read_tokens)
        )
        with self._lock:
            self._persistent_metrics.record_request(**metrics)
            if persist:
                self._maybe_save_locked()

    def record_lifetime_stack(self, stack: str | None) -> None:
        """Mirror the existing inbound stack dimension without another save."""

        with self._lock:
            self._persistent_metrics.record_stack(stack)

    def record_lifetime_failed(
        self, *, provider: str | None = None, model: str | None = None
    ) -> None:
        """Record a failed proxy request without changing legacy history."""

        with self._lock:
            self._persistent_metrics.record_failed(provider=provider, model=model)
            self._maybe_save_locked()

    def record_lifetime_rate_limited(
        self, *, provider: str | None = None, model: str | None = None
    ) -> None:
        """Record a rate-limited proxy request without changing legacy history."""

        with self._lock:
            self._persistent_metrics.record_rate_limited(provider=provider, model=model)
            self._maybe_save_locked()

    def record_lifetime_cache_bust(self, *, tokens_lost: int) -> None:
        with self._lock:
            self._persistent_metrics.record_cache_bust(tokens_lost=tokens_lost)
            self._maybe_save_locked()

    def record_lifetime_cache_miss(self, *, provider: str | None, reason: str | None) -> None:
        with self._lock:
            self._persistent_metrics.record_cache_miss(provider=provider, reason=reason)
            self._maybe_save_locked()

    def _record_project_locked(
        self,
        project: str | None,
        *,
        timestamp_dt: datetime,
        requests_delta: int = 0,
        tokens_saved_delta: int = 0,
        savings_usd_delta: float = 0.0,
        input_tokens_delta: int = 0,
        input_cost_usd_delta: float = 0.0,
    ) -> None:
        """Accumulate per-project savings. Caller must hold ``self._lock``.

        Unattributed traffic (``project`` missing or unusable) is skipped so
        existing aggregate behavior is unchanged. The map is capped at
        ``DEFAULT_MAX_PROJECTS`` entries, evicting the smallest/oldest bucket.
        """
        name = sanitize_project_name(project)
        if name is None:
            return
        projects: dict[str, dict[str, Any]] = self._state.setdefault("projects", {})
        entry = projects.setdefault(name, _empty_project_entry())
        entry["requests"] += max(requests_delta, 0)
        entry["tokens_saved"] += max(tokens_saved_delta, 0)
        entry["compression_savings_usd"] = round(
            entry["compression_savings_usd"] + max(savings_usd_delta, 0.0), 6
        )
        entry["total_input_tokens"] += max(input_tokens_delta, 0)
        entry["total_input_cost_usd"] = round(
            entry["total_input_cost_usd"] + max(input_cost_usd_delta, 0.0), 6
        )
        entry["last_activity_at"] = _to_utc_iso(timestamp_dt)
        if len(projects) > DEFAULT_MAX_PROJECTS:
            evict = min(
                (key for key in projects if key != name),
                key=lambda key: (
                    projects[key]["tokens_saved"],
                    projects[key]["last_activity_at"] or "",
                ),
            )
            del projects[evict]

    def _record_by_model_locked(
        self,
        model: str,
        *,
        requests_delta: int = 0,
        tokens_saved_delta: int = 0,
        savings_usd_delta: float = 0.0,
        input_tokens_delta: int = 0,
        input_cost_usd_delta: float = 0.0,
    ) -> None:
        """Accumulate per-model savings. Caller must hold ``self._lock``.

        Lazy-inits ``by_model`` so existing state files without the key work
        without migration.
        """
        by_model: dict[str, dict[str, Any]] = self._state.setdefault("by_model", {})
        key = _normalize_model(model)
        entry = by_model.setdefault(key, _empty_by_model_entry())
        entry["requests"] += max(requests_delta, 0)
        entry["tokens_saved"] += max(tokens_saved_delta, 0)
        entry["compression_savings_usd"] = round(
            entry["compression_savings_usd"] + max(savings_usd_delta, 0.0), 6
        )
        entry["total_input_tokens"] += max(input_tokens_delta, 0)
        entry["total_input_cost_usd"] = round(
            entry["total_input_cost_usd"] + max(input_cost_usd_delta, 0.0), 6
        )

    def _projects_snapshot_locked(self) -> dict[str, dict[str, Any]]:
        """Per-project stats with a derived ``savings_percent``, sorted by savings."""
        projects = self._state.get("projects", {})
        ranked = sorted(
            projects.items(),
            key=lambda item: item[1]["tokens_saved"],
            reverse=True,
        )
        result: dict[str, dict[str, Any]] = {}
        for name, entry in ranked:
            view = dict(entry)
            total_before = entry["tokens_saved"] + entry["total_input_tokens"]
            view["savings_percent"] = round(
                (entry["tokens_saved"] / total_before * 100) if total_before > 0 else 0.0,
                2,
            )
            result[name] = view
        return result

    def _by_model_snapshot_locked(self) -> dict[str, dict[str, Any]]:
        """Per-model stats ranked by savings."""
        by_model = self._state.get("by_model", {})
        ranked = sorted(
            by_model.items(),
            key=lambda item: item[1]["tokens_saved"],
            reverse=True,
        )
        result: dict[str, dict[str, Any]] = {}
        for model, entry in ranked:
            view = dict(entry)
            total_before = entry["tokens_saved"] + entry["total_input_tokens"]
            view["savings_percent"] = round(
                (entry["tokens_saved"] / total_before * 100) if total_before > 0 else 0.0,
                2,
            )
            result[model] = view
        return result

    def lifetime_response(self) -> dict[str, Any]:
        """Return the durable aggregate used only by ``/stats-lifetime``."""

        with self._lock:
            response = self._persistent_metrics.snapshot(
                persistence={
                    "enabled": not self._stateless,
                    "healthy": self._persistence_healthy,
                    "error": (
                        "Lifetime metrics unavailable in stateless mode"
                        if self._stateless
                        else self._persistence_error
                    ),
                    "pending_records": 0 if self._stateless else self._since_save,
                }
            )
            response["projects"] = self._projects_snapshot_locked()
            return response

    def stats_preview(self, recent_points: int = 20) -> dict[str, Any]:
        """Return a compact preview for `/stats`."""
        snapshot = self.snapshot()
        return {
            "schema_version": snapshot["schema_version"],
            "storage_path": snapshot["storage_path"],
            "lifetime": snapshot["lifetime"],
            "display_session": snapshot["display_session"],
            "display_session_policy": snapshot["display_session_policy"],
            "history_points": len(snapshot["history"]),
            "recent_history": snapshot["history"][-recent_points:],
            "retention": snapshot["retention"],
            "projects": snapshot["projects"],
            "projects_limit": DEFAULT_MAX_PROJECTS,
            "by_model": snapshot["by_model"],
        }

    def history_response(self, history_mode: str = "compact") -> dict[str, Any]:
        """Return frontend-friendly historical data for `/stats-history`."""
        snapshot = self.snapshot()
        raw_history = snapshot["history"]
        series = {
            "hourly": self._build_rollup(raw_history, bucket="hour"),
            "daily": self._build_rollup(raw_history, bucket="day"),
            "weekly": self._build_rollup(raw_history, bucket="week"),
            "monthly": self._build_rollup(raw_history, bucket="month"),
        }
        history = self._history_for_response(raw_history, mode=history_mode)
        return {
            "schema_version": snapshot["schema_version"],
            "generated_at": _to_utc_iso(_utc_now()),
            "storage_path": snapshot["storage_path"],
            "lifetime": snapshot["lifetime"],
            "display_session": snapshot["display_session"],
            "display_session_policy": snapshot["display_session_policy"],
            "history": history,
            "series": series,
            "exports": {
                "default_format": "json",
                "available_formats": ["json", "csv"],
                "available_series": ["history", *series.keys()],
            },
            "retention": snapshot["retention"],
            "projects": snapshot["projects"],
            "by_model": snapshot["by_model"],
            "history_summary": {
                "mode": history_mode,
                "stored_points": len(raw_history),
                "returned_points": len(history),
                "compacted": len(history) < len(raw_history),
            },
        }

    def export_rows(self, series: str = "history") -> list[dict[str, Any]]:
        """Return export rows for history or a rollup series."""
        response = self.history_response()
        if series == "history":
            return [dict(item) for item in response["history"]]
        return [dict(item) for item in response["series"].get(series, [])]

    def export_csv(self, series: str = "history") -> str:
        """Export history or rollup series as CSV."""
        rows = self.export_rows(series=series)
        if series == "history":
            fieldnames = [
                "timestamp",
                "total_tokens_saved",
                "compression_savings_usd",
                "total_input_tokens",
                "total_input_cost_usd",
            ]
        else:
            fieldnames = [
                "timestamp",
                "tokens_saved",
                "compression_savings_usd_delta",
                "total_tokens_saved",
                "compression_savings_usd",
                "total_input_tokens_delta",
                "total_input_tokens",
                "total_input_cost_usd_delta",
                "total_input_cost_usd",
                "output_tokens_saved_delta",
                "output_savings_usd_delta",
            ]

        buffer = StringIO()
        writer = DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        return buffer.getvalue()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            history = [dict(item) for item in self._state["history"]]
            return {
                "schema_version": SCHEMA_VERSION,
                "storage_path": str(self._path),
                "lifetime": dict(self._state["lifetime"]),
                "display_session": self._display_session_snapshot_locked(),
                "display_session_policy": {
                    "rollover_inactivity_minutes": (self._display_session_inactivity_minutes),
                },
                "history": history,
                "retention": {
                    "max_history_points": self._max_history_points,
                    "max_history_age_days": self._max_history_age_days,
                    "max_response_history_points": self._max_response_history_points,
                },
                "projects": self._projects_snapshot_locked(),
                "by_model": self._by_model_snapshot_locked(),
            }

    def _default_state(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "lifetime": {
                "requests": 0,
                "tokens_saved": 0,
                "compression_savings_usd": 0.0,
                "cache_read_tokens": 0,
                "cache_savings_usd": 0.0,
                "total_input_tokens": 0,
                "total_input_cost_usd": 0.0,
            },
            "display_session": _empty_display_session(),
            "history": [],
            "projects": {},
            "by_model": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self._path.exists():
            return self._default_state()

        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load savings history from %s: %s", self._path, e)
            self._persistence_healthy = False
            self._persistence_error = str(e)
            self._preserve_corrupt_file()
            return self._default_state()

        if not isinstance(raw, dict):
            self._persistence_healthy = False
            self._persistence_error = "Savings state root must be a JSON object"
            self._preserve_corrupt_file()
            return self._default_state()
        if _coerce_int(raw.get("schema_version")) != SCHEMA_VERSION:
            self._needs_schema_save = True
        return self._sanitize_state(raw)

    def _preserve_corrupt_file(self) -> None:
        """Best-effort retention of unreadable state before starting fresh."""

        timestamp = _to_utc_iso(_utc_now()).replace(":", "-")
        corrupt_path = self._path.with_name(f"{self._path.name}.corrupt-{timestamp}")
        try:
            self._path.replace(corrupt_path)
        except OSError:
            logger.warning("Could not preserve corrupt savings history at %s", self._path)

    def _sanitize_state(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._default_state()

        history_raw = raw.get("history", [])
        normalized_history = []
        if isinstance(history_raw, list):
            for item in history_raw:
                normalized = _normalize_history_entry(item)
                if normalized is not None:
                    normalized_history.append(normalized)

        normalized_history.sort(key=lambda item: item["timestamp"])

        lifetime_raw = raw.get("lifetime", {})
        lifetime_requests = 0
        lifetime_tokens_saved = 0
        lifetime_savings_usd = 0.0
        lifetime_cache_read_tokens = 0
        lifetime_cache_savings_usd = 0.0
        lifetime_input_tokens = 0
        lifetime_input_cost_usd = 0.0
        if isinstance(lifetime_raw, dict):
            lifetime_requests = _coerce_int(lifetime_raw.get("requests"))
            lifetime_tokens_saved = _coerce_int(lifetime_raw.get("tokens_saved"))
            lifetime_savings_usd = _coerce_float(lifetime_raw.get("compression_savings_usd"))
            lifetime_cache_read_tokens = _coerce_int(lifetime_raw.get("cache_read_tokens"))
            lifetime_cache_savings_usd = _coerce_float(lifetime_raw.get("cache_savings_usd"))
            lifetime_input_tokens = _coerce_int(lifetime_raw.get("total_input_tokens"))
            lifetime_input_cost_usd = _coerce_float(lifetime_raw.get("total_input_cost_usd"))

        if normalized_history:
            last = normalized_history[-1]
            lifetime_tokens_saved = max(
                lifetime_tokens_saved,
                last["total_tokens_saved"],
            )
            lifetime_savings_usd = max(
                lifetime_savings_usd,
                _coerce_float(last["compression_savings_usd"]),
            )
            lifetime_input_tokens = max(
                lifetime_input_tokens,
                _coerce_int(last.get("total_input_tokens")),
            )
            lifetime_input_cost_usd = max(
                lifetime_input_cost_usd,
                _coerce_float(last.get("total_input_cost_usd")),
            )

        state = {
            "schema_version": SCHEMA_VERSION,
            "lifetime": {
                "requests": lifetime_requests,
                "tokens_saved": lifetime_tokens_saved,
                "compression_savings_usd": round(lifetime_savings_usd, 6),
                "cache_read_tokens": lifetime_cache_read_tokens,
                "cache_savings_usd": round(lifetime_cache_savings_usd, 6),
                "total_input_tokens": lifetime_input_tokens,
                "total_input_cost_usd": round(lifetime_input_cost_usd, 6),
            },
            "display_session": _normalize_display_session(raw.get("display_session")),
            "history": normalized_history,
            "projects": _normalize_projects(raw.get("projects")),
            "by_model": _normalize_by_model(raw.get("by_model")),
        }
        raw_lifetime_metrics = raw.get("lifetime_metrics")
        if isinstance(raw_lifetime_metrics, dict):
            state["lifetime_metrics"] = raw_lifetime_metrics
        else:
            state["lifetime_metrics"] = self._migrate_v4_lifetime_metrics(state)
            self._needs_schema_save = True

        if normalized_history:
            reference_time = _parse_timestamp(normalized_history[-1]["timestamp"]) or _utc_now()
            original_state = self._state if hasattr(self, "_state") else None
            self._state = state
            try:
                self._trim_history_locked(reference_time=reference_time)
                state = self._state
            finally:
                if original_state is not None:
                    self._state = original_state

        return state

    def _migrate_v4_lifetime_metrics(self, state: dict[str, Any]) -> dict[str, Any]:
        """Seed the v5 aggregate from the best information available in v4."""

        history = state["history"]
        display_session = state["display_session"]
        started_at = (
            history[0]["timestamp"]
            if history
            else display_session.get("started_at") or _to_utc_iso(_utc_now())
        )
        last_activity_at = (
            history[-1]["timestamp"] if history else display_session.get("last_activity_at")
        )
        legacy = state["lifetime"]
        return {
            "started_at": started_at,
            "last_activity_at": last_activity_at,
            "full_fidelity_started_at": _to_utc_iso(_utc_now()),
            "requests": {"total": legacy["requests"]},
            "tokens": {
                "input": legacy["total_input_tokens"],
                "attempted_input": legacy["total_input_tokens"] + legacy["tokens_saved"],
                "saved": legacy["tokens_saved"],
            },
            "prefix_cache": {"cache_read_tokens": legacy["cache_read_tokens"]},
            "cost": {
                "input_usd": legacy["total_input_cost_usd"],
                "compression_savings_usd": legacy["compression_savings_usd"],
                "cache_savings_usd": legacy["cache_savings_usd"],
            },
            "models": {
                "tracked": {},
                "other": {
                    "requests": legacy["requests"],
                    "input_tokens": legacy["total_input_tokens"],
                    "attempted_input_tokens": legacy["total_input_tokens"] + legacy["tokens_saved"],
                    "tokens_saved": legacy["tokens_saved"],
                    "last_activity_at": last_activity_at,
                },
            },
        }

    def _trim_history_locked(self, reference_time: datetime | None = None) -> None:
        history = self._state["history"]
        if not history:
            return

        if self._max_history_age_days > 0:
            cutoff = (reference_time or _utc_now()) - timedelta(days=self._max_history_age_days)
            filtered = [
                item
                for item in history
                if (_parse_timestamp(item["timestamp"]) or _utc_now()) >= cutoff
            ]
            if not filtered:
                filtered = [history[-1]]
            history = filtered

        if self._max_history_points > 0 and len(history) > self._max_history_points:
            history = history[-self._max_history_points :]

        self._state["history"] = history

    def _history_for_response(
        self,
        history: list[dict[str, Any]],
        *,
        mode: str,
    ) -> list[dict[str, Any]]:
        if mode == "none":
            return []
        if mode == "full":
            return [dict(item) for item in history]
        return self._compact_history(history)

    def _compact_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(history) <= self._max_response_history_points:
            return [dict(item) for item in history]

        # Keep the recent tail dense for charts while evenly sampling older
        # checkpoints so long-running installs don't return unbounded payloads.
        recent_points = min(
            max(self._max_response_history_points // 3, 50),
            self._max_response_history_points - 1,
        )
        recent = history[-recent_points:]
        older = history[:-recent_points]
        older_slots = self._max_response_history_points - len(recent)
        if older_slots <= 0 or not older:
            return [dict(item) for item in recent[-self._max_response_history_points :]]

        if older_slots == 1:
            sampled_older = [older[0]]
        else:
            sampled_older = [
                older[((len(older) - 1) * index) // (older_slots - 1)]
                for index in range(older_slots)
            ]

        compacted: list[dict[str, Any]] = []
        seen_timestamps: set[str] = set()
        for point in [*sampled_older, *recent]:
            timestamp = point.get("timestamp")
            if not isinstance(timestamp, str) or timestamp in seen_timestamps:
                continue
            seen_timestamps.add(timestamp)
            compacted.append(dict(point))

        return compacted

    def flush(self) -> None:
        """Persist any records held back by the save throttle.

        Call on graceful shutdown so a batched proxy doesn't drop the tail of
        recent requests. No-op when nothing is buffered.
        """
        with self._lock:
            if self._since_save > 0 or self._needs_schema_save:
                self._save_locked()

    def _maybe_save_locked(self) -> None:
        """Throttled persist: write only every ``_save_flush_every`` records.

        Caller must hold ``self._lock``. Lossless by design — see ``__init__``.
        """
        self._since_save += 1
        if self._since_save >= self._save_flush_every:
            self._save_locked()

    def _save_locked(self) -> None:
        if self._stateless:
            # Stateless mode: live counters stay in memory; nothing is persisted.
            self._since_save = 0
            self._needs_schema_save = False
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            saved_at = _to_utc_iso(_utc_now())
            lifetime_metrics = self._persistent_metrics.to_dict()
            lifetime_metrics["persistence"]["last_saved_at"] = saved_at
            payload = {
                "schema_version": SCHEMA_VERSION,
                "lifetime": self._state["lifetime"],
                "display_session": self._state["display_session"],
                "history": self._state["history"],
                "projects": self._state.get("projects", {}),
                "by_model": self._state.get("by_model", {}),
                "lifetime_metrics": lifetime_metrics,
            }
            json_data = json.dumps(payload, indent=2)

            fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=".proxy_savings_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json_data)
                    f.flush()
                    os.fsync(f.fileno())
                Path(tmp_path).replace(self._path)
            except Exception:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass
                raise

            # Persist the rename itself — the fsync above flushed the file's
            # bytes, but the directory entry the rename created isn't durable
            # until the parent directory is fsynced too (POSIX). Best-effort —
            # directory fsync is unsupported on Windows and some virtual
            # filesystems; the file and atomic rename are already durable, so a
            # failure here only forgoes the last-save crash guarantee, never
            # correctness. (FP4b)
            try:
                dir_fd = os.open(self._path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass

            # Reset only after a durable write. A failed save leaves the counter
            # untouched so the next record retries instead of waiting a full window.
            self._persistent_metrics.set_last_saved_at(saved_at)
            self._persistence_healthy = True
            self._persistence_error = None
            self._needs_schema_save = False
            self._since_save = 0
        except OSError as e:
            logger.warning("Failed to save savings history to %s: %s", self._path, e)
            self._persistence_healthy = False
            self._persistence_error = str(e)

    def _display_session_snapshot_locked(
        self,
        reference_time: datetime | None = None,
    ) -> dict[str, Any]:
        session = dict(self._state["display_session"])
        last_activity = _parse_timestamp(session.get("last_activity_at"))
        if last_activity is None or self._is_display_session_expired(
            last_activity,
            reference_time=reference_time,
        ):
            return _empty_display_session()

        total_before = _coerce_int(session.get("tokens_saved")) + _coerce_int(
            session.get("total_input_tokens")
        )
        session["savings_percent"] = round(
            (_coerce_int(session.get("tokens_saved")) / total_before * 100)
            if total_before > 0
            else 0.0,
            2,
        )
        session["compression_savings_usd"] = round(
            _coerce_float(session.get("compression_savings_usd")),
            6,
        )
        session["total_input_cost_usd"] = round(
            _coerce_float(session.get("total_input_cost_usd")),
            6,
        )
        return session

    def _is_display_session_expired(
        self,
        last_activity: datetime,
        *,
        reference_time: datetime | None = None,
    ) -> bool:
        return (reference_time or _utc_now()) - last_activity > timedelta(
            minutes=self._display_session_inactivity_minutes
        )

    def _build_rollup(
        self,
        history: list[dict[str, Any]],
        bucket: str,
    ) -> list[dict[str, Any]]:
        if not history:
            return []

        aggregated: dict[str, dict[str, Any]] = {}
        prev_total_tokens = 0
        prev_total_usd = 0.0
        prev_total_input_tokens = 0
        prev_total_input_cost_usd = 0.0
        prev_output_tokens = 0
        prev_output_usd = 0.0

        for point in history:
            timestamp = _parse_timestamp(point["timestamp"])
            if timestamp is None:
                continue

            bucket_start = _bucket_start(timestamp, bucket)

            bucket_key = _to_utc_iso(bucket_start)
            total_tokens_saved = _coerce_int(point.get("total_tokens_saved"))
            total_usd = _coerce_float(point.get("compression_savings_usd"))
            total_input_tokens = _coerce_int(point.get("total_input_tokens"))
            total_input_cost_usd = _coerce_float(point.get("total_input_cost_usd"))
            total_output_tokens = _coerce_int(point.get("output_tokens_saved"))
            total_output_usd = _coerce_float(point.get("output_savings_usd"))
            delta_tokens = max(total_tokens_saved - prev_total_tokens, 0)
            delta_usd = max(total_usd - prev_total_usd, 0.0)
            delta_input_tokens = max(total_input_tokens - prev_total_input_tokens, 0)
            delta_input_cost_usd = max(
                total_input_cost_usd - prev_total_input_cost_usd,
                0.0,
            )

            delta_output_tokens = max(total_output_tokens - prev_output_tokens, 0)
            delta_output_usd = max(total_output_usd - prev_output_usd, 0.0)

            prev_total_tokens = total_tokens_saved
            prev_total_usd = total_usd
            prev_total_input_tokens = total_input_tokens
            prev_total_input_cost_usd = total_input_cost_usd
            prev_output_tokens = total_output_tokens
            prev_output_usd = total_output_usd

            entry = aggregated.setdefault(
                bucket_key,
                {
                    "timestamp": bucket_key,
                    "tokens_saved": 0,
                    "compression_savings_usd_delta": 0.0,
                    "total_tokens_saved": total_tokens_saved,
                    "compression_savings_usd": total_usd,
                    "total_input_tokens_delta": 0,
                    "total_input_tokens": total_input_tokens,
                    "total_input_cost_usd_delta": 0.0,
                    "total_input_cost_usd": total_input_cost_usd,
                    "output_tokens_saved_delta": 0,
                    "output_savings_usd_delta": 0.0,
                    "by_provider": {},
                    "by_model": {},
                },
            )
            entry["tokens_saved"] += delta_tokens
            entry["compression_savings_usd_delta"] = round(
                entry["compression_savings_usd_delta"] + delta_usd,
                6,
            )
            entry["total_input_tokens_delta"] += delta_input_tokens
            entry["total_input_cost_usd_delta"] = round(
                entry["total_input_cost_usd_delta"] + delta_input_cost_usd,
                6,
            )
            entry["total_tokens_saved"] = total_tokens_saved
            entry["compression_savings_usd"] = round(total_usd, 6)
            entry["total_input_tokens"] = total_input_tokens
            entry["total_input_cost_usd"] = round(total_input_cost_usd, 6)
            entry["output_tokens_saved_delta"] += delta_output_tokens
            entry["output_savings_usd_delta"] = round(
                entry["output_savings_usd_delta"] + delta_output_usd,
                6,
            )

            # Attribute this checkpoint's delta to the provider that produced
            # it. Each checkpoint comes from a single request, so its delta is
            # wholly owned by one provider. Skip no-op checkpoints so providers
            # only appear in a bucket where they actually moved a counter.
            if delta_tokens or delta_usd or delta_input_tokens or delta_input_cost_usd:
                provider = _normalize_provider(point.get("provider"))
                prov = entry["by_provider"].setdefault(
                    provider,
                    {
                        "tokens_saved": 0,
                        "compression_savings_usd_delta": 0.0,
                        "total_input_tokens_delta": 0,
                        "total_input_cost_usd_delta": 0.0,
                    },
                )
                prov["tokens_saved"] += delta_tokens
                prov["compression_savings_usd_delta"] = round(
                    prov["compression_savings_usd_delta"] + delta_usd,
                    6,
                )
                prov["total_input_tokens_delta"] += delta_input_tokens
                prov["total_input_cost_usd_delta"] = round(
                    prov["total_input_cost_usd_delta"] + delta_input_cost_usd,
                    6,
                )

                model = _normalize_model(point.get("model"))
                mod = entry["by_model"].setdefault(
                    model,
                    {
                        "tokens_saved": 0,
                        "compression_savings_usd_delta": 0.0,
                        "total_input_tokens_delta": 0,
                        "total_input_cost_usd_delta": 0.0,
                    },
                )
                mod["tokens_saved"] += delta_tokens
                mod["compression_savings_usd_delta"] = round(
                    mod["compression_savings_usd_delta"] + delta_usd,
                    6,
                )
                mod["total_input_tokens_delta"] += delta_input_tokens
                mod["total_input_cost_usd_delta"] = round(
                    mod["total_input_cost_usd_delta"] + delta_input_cost_usd,
                    6,
                )

        return list(aggregated.values())
