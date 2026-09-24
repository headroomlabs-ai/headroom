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
import time
from collections.abc import Mapping
from csv import DictWriter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
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
# v6: ``compression_savings_usd`` changed BASIS (not shape). It now holds the
# cache-aware counterfactual — what the removed tokens would actually have been
# billed at, given the cache mix of the requests they came out of — instead of
# flat list price. ``compression_savings_list_usd`` carries the old list-priced
# figure alongside it as the upper bound.
#
# The basis flip is deliberately made in place rather than in a new field. The
# mix was never persisted, so pre-v6 dollars cannot be re-derived either way;
# a parallel field would have to be SEEDED from the same list-priced history and
# would blend exactly as much, only in a field nothing reads. Flipping in place
# means every existing surface (dashboard tiles, per-model table, per-project
# rows, the history chart) reports the right number with no change of its own.
# ``savings_basis_migrated_at`` records when the flip happened so a reader can
# tell which part of a lifetime total predates it.
SCHEMA_VERSION = 6
DEFAULT_MAX_HISTORY_POINTS = 5000
DEFAULT_MAX_PROJECTS = 50
DEFAULT_MAX_HISTORY_AGE_DAYS = 365
DEFAULT_MAX_RESPONSE_HISTORY_POINTS = 500
DEFAULT_DISPLAY_SESSION_INACTIVITY_MINUTES = 60
# Throttle for the negative-savings warning. The first one is immediate; after
# that a losing deployment gets one summary line per interval rather than one
# per request. Module-level indirection over the clock so tests can drive it.
NEGATIVE_SAVINGS_WARN_INTERVAL_S = 300.0
_monotonic = time.monotonic
DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
# Blended output price used only when litellm cannot price the model.
DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000

#: Basis label for a total that has not priced a single request yet. Not one of
#: ``counterfactual``'s BASIS_* values on purpose: those all describe how a real
#: figure was derived, and "no figure yet" is not a derivation. Treated as
#: absent by :func:`_blend_basis`, so the first priced request sets the label
#: outright instead of being dragged down to it forever.
BASIS_UNKNOWN = "unknown"

#: Basis stamped on state migrated from a schema older than v6. Those dollars
#: were accumulated at flat list price and the cache mix that produced them was
#: never persisted, so they cannot be re-derived — the total stays honestly
#: labelled as containing list-priced history for the life of the install.
BASIS_LIST = "list"

LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None
litellm: Any | None = None


def _blend_basis(existing: Any, incoming: Any) -> str:
    """Fold one request's pricing basis into a running total's label.

    A total is only as sound as its weakest contributing request, so this keeps
    the worst basis seen. ``BASIS_UNKNOWN`` means "nothing priced yet" and is
    replaced outright rather than treated as the weakest — otherwise a fresh
    install would report ``unknown`` forever after its first request.

    Ranking lives in ``counterfactual``; importing it here is deferred for the
    same startup reason as ``estimate_request_savings_usd``, and a missing
    incoming label leaves the existing one alone.
    """
    incoming_label = str(incoming or "").strip()
    existing_label = str(existing or "").strip() or BASIS_UNKNOWN
    if not incoming_label:
        return existing_label
    if existing_label == BASIS_UNKNOWN:
        return incoming_label
    if incoming_label == existing_label:
        return existing_label
    try:
        from headroom.pricing.counterfactual import weakest_basis

        return weakest_basis(existing_label, incoming_label)
    except Exception:  # pragma: no cover - defensive; labelling must not raise
        return existing_label


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


def _coerce_signed_float(value: Any, default: float = 0.0) -> float:
    """``_coerce_float`` without the zero floor, for quantities that can lose.

    The floor in ``_coerce_float`` is right for the things it mostly guards --
    token counts, input costs, cumulative totals -- none of which have a
    meaningful negative value. It is wrong for per-request compression savings,
    which genuinely go negative when a rewrite displaces tokens out of the
    cached prefix and they come back billed at the fresh-input rate. Flooring
    those reported a gain on a turn that lost money. Kept as a separate helper
    rather than a flag on ``_coerce_float`` so the callers that want a floor
    keep getting one by default.
    """

    try:
        coerced = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(coerced):
        return default
    return coerced


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


def _empty_cache_delta() -> dict[str, Any]:
    """Zeroed cache fields for a rollup bucket or one of its breakdowns.

    ``cache_read_cost_usd_delta`` is None when any contributing checkpoint's
    reads could not be priced (see ``_build_rollup``).
    """
    return {
        "cache_read_tokens_delta": 0,
        "cache_savings_usd_delta": 0.0,
        "cache_read_cost_usd_delta": 0.0,
    }


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


# `_resolve_litellm_model` is called on every savings-tracking update (i.e.
# every request), and `model` is client-controlled — it comes straight off
# the request body. For a model LiteLLM can't price (a custom / local /
# gateway name), the uncached fallback below calls `litellm.cost_per_token`
# purely to probe resolvability, which prints LiteLLM's noisy "Provider
# List: https://docs.litellm.ai/docs/providers" banner on every failed probe
# (#2851). Cache the resolution per model name so that probe runs at most
# once per distinct model — bounded, not a plain dict: a request-facing
# proxy must not let a caller grow an unbounded cache for free by sending a
# fresh model string on every request. `maxsize` caps memory; LRU eviction
# means a model that stops being sent eventually falls out and simply
# re-probes if it's ever sent again — never a correctness issue, only
# whether the probe (and its noisy failure banner) reruns.
_MODEL_RESOLUTION_CACHE_MAXSIZE = 256


@lru_cache(maxsize=_MODEL_RESOLUTION_CACHE_MAXSIZE)
def _resolve_litellm_model(model: str) -> str:
    """Resolve model name to one LiteLLM recognizes.

    Delegates to the shared alias-map-aware resolver in
    ``headroom.pricing.litellm_pricing`` so the persisted /stats-history funnel
    (PROXY $ SAVED tile + Historical Checkpoints) prices gateway aliases like
    "claude-opus" identically to the live /stats path. Uses the shared result
    only when it maps to a priced model_cost key; otherwise falls through to the
    bare-prefix logic below. Fail-soft: pricing never breaks bookkeeping.

    Bounded LRU cache, keyed by model name — see
    ``_MODEL_RESOLUTION_CACHE_MAXSIZE`` above. Tests that mock the LiteLLM
    module across calls with the same model name must call
    ``_resolve_litellm_model.cache_clear()`` between cases, or results from
    an earlier case leak in.
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
    litellm.

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


def estimate_request_savings_usd(
    model: str,
    *,
    compression_tokens_saved: int = 0,
    tool_schema_tokens_saved: int = 0,
    output_tokens_saved: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_5m_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    uncached_input_tokens: int = 0,
    cache_inferred: bool = False,
    local_input_tokens: int = 0,
    provider: str | None = None,
) -> dict[str, Any]:
    """Price one request's distinct savings layers for external telemetry.

    The layers stay separate because provider-cache benefit is not caused by
    compression, and extension attribution can be explanatory rather than
    additive.

    The two input-side layers are priced CACHE-AWARE, each against the region of
    the request it actually came out of (see
    :mod:`headroom.pricing.counterfactual`):

    * ``compression`` — live-zone content, which was never a cache read.
    * ``tool_schema`` — prefix content, which on a warm turn was *entirely* a
      cache read.

    Both also report a ``*_list`` companion: the same tokens at flat list price.
    That is the upper bound, the figure this function used to return for both
    layers, and what budget enforcement keeps consuming because it is monotonic
    in tokens and needs no provider cooperation. ``basis`` says how sound the
    cache-aware numbers are — see the BASIS_* constants.

    Passing no cache breakdown at all is fully supported and is what every
    non-reporting harness and the MCP tool path do: the effective figures then
    equal the list figures and ``basis`` reads ``no-mix``.
    """

    # Imported at CALL time, not module scope. `headroom.pricing.__init__`
    # eagerly imports litellm, which measures 4.1s — and savings_tracker is
    # imported during proxy startup, so a module-level import here would put
    # that on every boot. No request can be priced without litellm anyway, so
    # deferring to the first priced request costs nothing and keeps startup
    # unchanged. Repeat calls are a `sys.modules` dict hit. Same idiom as
    # `_resolve_litellm_model` below.
    from headroom.pricing.counterfactual import (
        CacheMix,
        PricedSavings,
        Region,
        price_savings,
        weakest_basis,
    )

    mix = CacheMix.from_usage(
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_write_5m_tokens=cache_write_5m_tokens,
        cache_write_1h_tokens=cache_write_1h_tokens,
        uncached_input_tokens=uncached_input_tokens,
        cache_inferred=cache_inferred,
    )
    long_context = mix.is_long_context(local_tokens=local_input_tokens)

    def _price(tokens: Any, region: Region) -> PricedSavings:
        return price_savings(
            max(_coerce_int(tokens), 0),
            model=model,
            mix=mix,
            region=region,
            long_context=long_context,
            provider=provider,
            # Matches the blended rate the flat estimators fall back to, so an
            # unpriceable model reports the same dollars it always has.
            fallback_rate_per_token=DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
        )

    compression = _price(compression_tokens_saved, Region.LIVE_ZONE)
    tool_schema = _price(tool_schema_tokens_saved, Region.PREFIX)

    return {
        "compression": compression.usd,
        "compression_list": compression.usd_list,
        "tool_schema": tool_schema.usd,
        "tool_schema_list": tool_schema.usd_list,
        "output_shaping": _estimate_output_savings_usd(
            model, max(_coerce_int(output_tokens_saved), 0)
        ),
        "provider_cache": _estimate_cache_savings_usd(
            model, max(_coerce_int(cache_read_tokens), 0)
        ),
        # An aggregate is only as sound as its weakest input, so a request whose
        # tool-schema layer priced from the catalog but whose compression layer
        # had no mix to work with reports the weaker of the two.
        "basis": weakest_basis(
            compression.basis if compression.tokens else None,
            tool_schema.basis if tool_schema.tokens else None,
        ),
    }


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
        "compression_savings_list_usd": 0.0,
        "savings_basis": BASIS_UNKNOWN,
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
        # Message-level compression only. Kept as its own bucket rather than
        # widened in place, so the persisted meaning of an existing field never
        # changes under a reader that predates this.
        "tokens_saved": 0,
        # Tool-schema deferral, attributed to the model that benefited. This was
        # dropped on the floor: `record_request` was handed it and passed only
        # the message figure down, so a tool-heavy model read "0 tokens saved"
        # while the headline counted its deferral. Deferral is routinely the
        # larger half -- 589,206 of 625,277 in the report that prompted this.
        "tool_tokens_saved": 0,
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
            _coerce_signed_float(entry.get("compression_savings_usd")), 6
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
        # Absent in state files written before this field existed -> 0.
        normalized["tool_tokens_saved"] = _coerce_int(entry.get("tool_tokens_saved"))
        normalized["compression_savings_usd"] = round(
            _coerce_signed_float(entry.get("compression_savings_usd")), 6
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

    # Presence of the list column is what distinguishes a v6+ session from one
    # written before cache-aware pricing existed. A pre-v6 session's single
    # dollar figure WAS the list figure, so it seeds both columns exactly, and
    # the session is labelled `list` because its mix was never persisted and
    # cannot be recovered. Written out rather than inlined into the dict below:
    # a money path should not hinge on the reader parsing a nested ternary.
    is_pre_v6 = "compression_savings_list_usd" not in entry
    savings_usd = _coerce_signed_float(entry.get("compression_savings_usd"))
    if is_pre_v6:
        savings_list_usd = savings_usd
        savings_basis = BASIS_LIST
    else:
        savings_list_usd = _coerce_signed_float(entry.get("compression_savings_list_usd"))
        savings_basis = str(entry.get("savings_basis") or BASIS_UNKNOWN)

    return {
        "requests": _coerce_int(entry.get("requests")),
        "tokens_saved": tokens_saved,
        "compression_savings_usd": round(savings_usd, 6),
        "compression_savings_list_usd": round(savings_list_usd, 6),
        "savings_basis": savings_basis,
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
        # Negative-savings warning state. A deployment that loses money loses it
        # on most turns, not one, so warning per turn would bury the signal in
        # its own repetition. Accumulate instead and emit a summary no more than
        # once per NEGATIVE_SAVINGS_WARN_INTERVAL_S — except the first, which is
        # immediate, because the operator should learn on turn one rather than a
        # minute in.
        self._negative_savings_count = 0
        self._negative_savings_usd = 0.0
        self._negative_savings_last_warn: float | None = None

    @property
    def storage_path(self) -> str:
        return str(self._path)

    def _note_negative_savings(
        self,
        *,
        model: str | None,
        delta_usd: float,
        compression: float,
        tool_schema: float,
    ) -> None:
        """Warn that a request's measured savings came out negative.

        Negative savings mean the rewrite cost more than it removed on that turn
        — normally because it displaced tokens out of the cached prefix. It is
        not an accounting artifact and it is not self-correcting, so it warrants
        WARNING rather than a metric an operator has to go looking for.
        """

        self._negative_savings_count += 1
        self._negative_savings_usd += delta_usd
        now = _monotonic()
        last = self._negative_savings_last_warn
        if last is not None and (now - last) < NEGATIVE_SAVINGS_WARN_INTERVAL_S:
            return
        self._negative_savings_last_warn = now
        logger.warning(
            "event=savings_negative model=%s request_usd=%.6f compression_usd=%.6f "
            "tool_schema_usd=%.6f negative_requests=%d negative_usd_total=%.6f "
            "hint=%s",
            model or "unknown",
            delta_usd,
            compression,
            tool_schema,
            self._negative_savings_count,
            self._negative_savings_usd,
            "compression is costing more than it saves on this traffic; "
            "check prompt-cache busts in /stats prefix_cache.compression_vs_cache",
        )

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
        # Tool-schema deferral for this request, DISJOINT from ``tokens_saved``
        # (which is the bare message figure). The caller has always had this
        # number and already folds it into the priced dollars below; it simply
        # had no parameter to arrive through, so per-model tokens stayed
        # message-only while per-model dollars did not.
        tool_search_saved: int = 0,
        output_tokens_saved: int = 0,
        provider: str | None = None,
        project: str | None = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        uncached_input_tokens: int = 0,
        total_input_tokens: int | None = None,
        total_input_cost_usd: float | None = None,
        # Mixed value types: per-layer dollars plus the string `basis` label
        # that says how soundly they were priced. See
        # ``estimate_request_savings_usd``, which produces it.
        estimated_savings_usd: Mapping[str, Any] | None = None,
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
        delta_tool_tokens_saved = max(_coerce_int(tool_search_saved), 0)
        delta_input_tokens = _coerce_int(input_tokens)
        delta_output_tokens_saved = max(_coerce_int(output_tokens_saved), 0)
        delta_cache_read_tokens = _coerce_int(cache_read_tokens)
        priced = estimated_savings_usd
        # ``estimate_request_savings_usd`` prices FOUR buckets, but this method
        # only ever read three — ``tool_schema`` was computed and dropped on the
        # floor. Tool-schema deferral is a quarter of the token headline on real
        # traffic (2.7M of 11.2M), and ``tokens_saved`` here is the bare
        # message-level figure (the caller folds deferral in separately for the
        # ledger, see prometheus_metrics.record_request), so the dollars were
        # simply missing rather than counted elsewhere. That is why "Cost saved"
        # read materially below the token-savings percent on the same traffic.
        # Add it to the compression bucket, matching how the PERF headline and
        # perf/analyzer fold deferral into one number.
        if priced is not None:
            # Two DISJOINT buckets: the caller passes bare message savings as
            # ``compression_tokens_saved`` and deferral separately as
            # ``tool_schema_tokens_saved`` (see prometheus_metrics.record_request),
            # so summing them is additive, not double counting. Written as a
            # statement rather than folded into the ternary below — a money path
            # should not depend on the reader knowing that ``a + b if c else d``
            # groups as ``(a + b) if c else d``.
            #
            # As of v6 these two arrive CACHE-AWARE and region-correct: the
            # compression bucket priced against the live zone (never a cache
            # read) and the tool-schema bucket against the prefix (on a warm
            # turn, entirely a cache read).
            #
            # These two are NOT floored at zero. A negative value here is a real
            # measurement, not an artifact: it says this turn's rewrite cost more
            # than it removed — the usual cause being that the rewrite moved
            # tokens out of the cached prefix (0.1x) and into the live zone,
            # where a cold Anthropic turn bills them at 1.25x list. Flooring that
            # at zero reported a gain on a turn that lost money, and the loss was
            # unrecoverable downstream because it was discarded here rather than
            # carried. The other two buckets below keep their floors: a provider
            # cache read is cheaper than a fresh read by construction, and output
            # shaping cannot lengthen the completion, so a negative there would
            # be an artifact rather than a loss.
            compression_usd = _coerce_signed_float(priced.get("compression"))
            tool_schema_usd = _coerce_signed_float(priced.get("tool_schema"))
            delta_savings_usd = compression_usd + tool_schema_usd
            delta_savings_list_usd = _coerce_signed_float(
                priced.get("compression_list")
            ) + _coerce_signed_float(priced.get("tool_schema_list"))
            delta_basis = priced.get("basis")
            if delta_savings_usd < 0:
                self._note_negative_savings(
                    model=model,
                    delta_usd=delta_savings_usd,
                    compression=compression_usd,
                    tool_schema=tool_schema_usd,
                )
        else:
            # No priced breakdown available: only message savings are known here,
            # so this path stays message-only exactly as before — and at list
            # price, which is what "no mix to price against" honestly means.
            delta_savings_usd = _estimate_compression_savings_usd(model, delta_tokens_saved)
            delta_savings_list_usd = delta_savings_usd
            delta_basis = BASIS_LIST if delta_tokens_saved > 0 else None
        delta_output_savings_usd = (
            max(_coerce_float(priced.get("output_shaping")), 0.0)
            if priced is not None
            else _estimate_output_savings_usd(model, delta_output_tokens_saved)
        )
        delta_cache_savings_usd = (
            max(_coerce_float(priced.get("provider_cache")), 0.0)
            if priced is not None
            else _estimate_cache_savings_usd(model, delta_cache_read_tokens)
        )
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
            lifetime["compression_savings_list_usd"] = round(
                _coerce_signed_float(lifetime.get("compression_savings_list_usd"))
                + delta_savings_list_usd,
                6,
            )
            lifetime["savings_basis"] = _blend_basis(lifetime.get("savings_basis"), delta_basis)
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
            session["compression_savings_list_usd"] = round(
                _coerce_signed_float(session.get("compression_savings_list_usd"))
                + delta_savings_list_usd,
                6,
            )
            session["savings_basis"] = _blend_basis(session.get("savings_basis"), delta_basis)
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
                tool_tokens_saved_delta=delta_tool_tokens_saved,
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
        # POPPED, not read: `metrics` is forwarded verbatim to
        # `PersistentMetricsState.record_request`, whose signature is an
        # explicit keyword list with no **kwargs. `cache_inferred` is a pricing
        # input (it tells the counterfactual to drop a write bucket the
        # provider never billed), not a metric that aggregate stores, so it
        # must not survive into that call.
        cache_inferred = bool(metrics.pop("cache_inferred", False))
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
        # Cache-aware, like every other savings figure as of v6. ``tokens_saved``
        # here is message compression, so it prices against the LIVE ZONE — the
        # region handlers actually compress, and the one region that can never
        # have been a cache read. The caller (prometheus_metrics.record_request)
        # already hands us the full breakdown, so no new plumbing is needed;
        # a caller that omits it degrades to list price and says so.
        if "compression_savings_usd" not in metrics:
            priced = estimate_request_savings_usd(
                model,
                compression_tokens_saved=_coerce_int(metrics.get("tokens_saved")),
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_write_5m_tokens=_coerce_int(metrics.get("cache_write_5m_tokens")),
                cache_write_1h_tokens=_coerce_int(metrics.get("cache_write_1h_tokens")),
                uncached_input_tokens=uncached_input_tokens,
                cache_inferred=cache_inferred,
                local_input_tokens=input_tokens,
                provider=metrics.get("provider"),
            )
            metrics["compression_savings_usd"] = priced["compression"]
            metrics["compression_savings_list_usd"] = priced["compression_list"]
            metrics["savings_basis"] = priced["basis"]
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
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        source: str = "headroom",
    ) -> None:
        """Record a rate-limited proxy request without changing legacy history.

        ``source`` distinguishes Headroom's own limiter from an upstream 429 —
        see ``PrometheusMetrics.record_rate_limited``.
        """

        with self._lock:
            self._persistent_metrics.record_rate_limited(
                provider=provider, model=model, source=source
            )
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
            entry["compression_savings_usd"] + savings_usd_delta, 6
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
        tool_tokens_saved_delta: int = 0,
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
        entry["tool_tokens_saved"] += max(tool_tokens_saved_delta, 0)
        entry["compression_savings_usd"] = round(
            entry["compression_savings_usd"] + savings_usd_delta, 6
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
            key=lambda item: item[1]["tokens_saved"] + item[1].get("tool_tokens_saved", 0),
            reverse=True,
        )
        result: dict[str, dict[str, Any]] = {}
        for model, entry in ranked:
            view = dict(entry)
            tool_saved = _coerce_int(entry.get("tool_tokens_saved"))
            # Deferred tool schemas never reached the model, so they were never in
            # ``total_input_tokens`` — the pre-Headroom denominator is the input we
            # sent plus what we withheld. Same construction the PERF headline and
            # perf/analyzer use, so a row and the total printed above it share one
            # definition of "saved".
            headline_saved = entry["tokens_saved"] + tool_saved
            total_before = headline_saved + entry["total_input_tokens"]
            # Named "headline" rather than "total" because this module already uses
            # ``total_tokens_saved`` for the cumulative lifetime figure on history
            # points; reusing it here would mean two different things in one file.
            view["headline_tokens_saved"] = headline_saved
            view["savings_percent"] = round(
                (headline_saved / total_before * 100) if total_before > 0 else 0.0,
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
                "compression_savings_list_usd": 0.0,
                "savings_basis": BASIS_UNKNOWN,
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
        lifetime_savings_list_usd = 0.0
        lifetime_basis = BASIS_UNKNOWN
        migrated_at = None
        is_v6_lifetime = False
        if isinstance(lifetime_raw, dict):
            lifetime_requests = _coerce_int(lifetime_raw.get("requests"))
            lifetime_tokens_saved = _coerce_int(lifetime_raw.get("tokens_saved"))
            lifetime_savings_usd = _coerce_float(lifetime_raw.get("compression_savings_usd"))
            lifetime_cache_read_tokens = _coerce_int(lifetime_raw.get("cache_read_tokens"))
            lifetime_cache_savings_usd = _coerce_float(lifetime_raw.get("cache_savings_usd"))
            lifetime_input_tokens = _coerce_int(lifetime_raw.get("total_input_tokens"))
            lifetime_input_cost_usd = _coerce_float(lifetime_raw.get("total_input_cost_usd"))
            migrated_at = lifetime_raw.get("savings_basis_migrated_at")
            is_v6_lifetime = "compression_savings_list_usd" in lifetime_raw
            if is_v6_lifetime:
                lifetime_savings_list_usd = _coerce_float(
                    lifetime_raw.get("compression_savings_list_usd")
                )
                lifetime_basis = str(lifetime_raw.get("savings_basis") or BASIS_UNKNOWN)

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

        # v6 migration seed. Runs HERE, after the history back-fill above, not
        # beside the lifetime read: a state whose `lifetime` block is missing or
        # empty still recovers its totals from the last history point, and
        # seeding before that ran left the list column at 0 while the effective
        # column carried the recovered dollars — a total that reads as a 100%
        # cache-aware saving on data that was entirely list-priced.
        #
        # Everything a pre-v6 install accumulated was flat list price, so it
        # seeds BOTH columns: for that history they genuinely are the same
        # number. The cache mix that would let us re-derive the honest value was
        # never persisted — which is the whole reason v6 exists — so the total
        # stays labelled `list` for the life of the install, and
        # `savings_basis_migrated_at` marks where the honest numbers begin.
        if not is_v6_lifetime and (lifetime_requests or lifetime_savings_usd):
            lifetime_savings_list_usd = lifetime_savings_usd
            lifetime_basis = BASIS_LIST
            if not migrated_at:
                migrated_at = _to_utc_iso(_utc_now())
                self._needs_schema_save = True

        state = {
            "schema_version": SCHEMA_VERSION,
            "lifetime": {
                "requests": lifetime_requests,
                "tokens_saved": lifetime_tokens_saved,
                "compression_savings_usd": round(lifetime_savings_usd, 6),
                "compression_savings_list_usd": round(lifetime_savings_list_usd, 6),
                "savings_basis": lifetime_basis,
                "savings_basis_migrated_at": migrated_at,
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
        prev_cache_read_tokens = 0
        prev_cache_savings_usd = 0.0
        # What the bucket's cache reads actually COST, priced per checkpoint
        # with the same function that put them into ``total_input_cost_usd``.
        # Consumers need it to take reads out of the input bill, and cannot
        # derive it from ``cache_savings_usd``: the read discount is not a
        # fixed multiple of the read cost (reads bill at 0.1x on most models,
        # 0.05x or 0.025x on others), so "discount / 9" misprices exactly the
        # models with the steepest cache discount.
        read_cost_per_token: dict[str, float] = {}

        def _read_cost(model: str, reads: int) -> float | None:
            if reads <= 0:
                return 0.0
            # Checkpoints written before per-model attribution carry no model,
            # so their reads cannot be priced the way the request was. Report
            # the bucket's read cost as unknown rather than guess.
            if model == MODEL_UNKNOWN:
                return None
            if model not in read_cost_per_token:
                read_cost_per_token[model] = (
                    _estimate_input_cost_usd(model, 1_000_000, cache_read_tokens=1_000_000)
                    / 1_000_000
                )
            return reads * read_cost_per_token[model]

        def _add_cache(
            target: dict[str, Any], reads: int, discount: float, cost: float | None
        ) -> None:
            target["cache_read_tokens_delta"] += reads
            target["cache_savings_usd_delta"] = round(
                target["cache_savings_usd_delta"] + discount, 6
            )
            if cost is None or target["cache_read_cost_usd_delta"] is None:
                target["cache_read_cost_usd_delta"] = None
            else:
                target["cache_read_cost_usd_delta"] = round(
                    target["cache_read_cost_usd_delta"] + cost, 6
                )

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

            total_cache_read_tokens = _coerce_int(point.get("cache_read_tokens"))
            total_cache_savings_usd = _coerce_float(point.get("cache_savings_usd"))
            delta_cache_read_tokens = max(total_cache_read_tokens - prev_cache_read_tokens, 0)
            delta_cache_savings_usd = max(total_cache_savings_usd - prev_cache_savings_usd, 0.0)
            prev_cache_read_tokens = total_cache_read_tokens
            prev_cache_savings_usd = total_cache_savings_usd
            model = _normalize_model(point.get("model"))
            delta_cache_read_cost_usd = _read_cost(model, delta_cache_read_tokens)

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
                    **_empty_cache_delta(),
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
            _add_cache(
                entry, delta_cache_read_tokens, delta_cache_savings_usd, delta_cache_read_cost_usd
            )

            # Attribute this checkpoint's delta to the provider that produced
            # it. Each checkpoint comes from a single request, so its delta is
            # wholly owned by one provider. Skip no-op checkpoints so providers
            # only appear in a bucket where they actually moved a counter.
            if (
                delta_tokens
                or delta_usd
                or delta_input_tokens
                or delta_input_cost_usd
                or delta_cache_read_tokens
            ):
                provider = _normalize_provider(point.get("provider"))
                prov = entry["by_provider"].setdefault(
                    provider,
                    {
                        "tokens_saved": 0,
                        "compression_savings_usd_delta": 0.0,
                        "total_input_tokens_delta": 0,
                        "total_input_cost_usd_delta": 0.0,
                        **_empty_cache_delta(),
                    },
                )
                _add_cache(
                    prov,
                    delta_cache_read_tokens,
                    delta_cache_savings_usd,
                    delta_cache_read_cost_usd,
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

                mod = entry["by_model"].setdefault(
                    model,
                    {
                        "tokens_saved": 0,
                        "compression_savings_usd_delta": 0.0,
                        "total_input_tokens_delta": 0,
                        "total_input_cost_usd_delta": 0.0,
                        **_empty_cache_delta(),
                    },
                )
                _add_cache(
                    mod, delta_cache_read_tokens, delta_cache_savings_usd, delta_cache_read_cost_usd
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
