"""Cost tracking and budget management for the Headroom proxy.

Contains the CostTracker class and cost-related helper functions
for prefix cache statistics, cost merging, and session summaries.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

from headroom.providers.cache_economics import cache_economics_for_provider
from headroom.providers.registry import model_matches_provider
from headroom.proxy.budget_basis_policy import (
    BUDGET_BASIS_BLOCK,
    BUDGET_BASIS_IGNORE,
    COST_BASIS_ESTIMATED,
    COST_BASIS_MEASURED,
    DEFAULT_POLICY,
    ENV_VAR,
    resolve_estimated_basis_policy,
)
from headroom.proxy.modes import PROXY_MODE_CACHE

if TYPE_CHECKING:
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None
litellm: Any | None = None


def _get_litellm_module() -> Any | None:
    """Import LiteLLM only when pricing data is actually requested."""
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


logger = logging.getLogger("headroom.proxy")

# Pricing-lookup warnings are emitted on the per-request cost path, so an
# unresolvable model (a custom / OpenAI-compatible name LiteLLM can't price,
# e.g. glm-5.2) floods proxy.log with an identical WARNING every single request
# (#2504). Track which models have already been warned so each fires once per
# process; the set is tiny and bounded by the number of distinct models seen.
_warned_pricing_models: set[str] = set()


def _warn_pricing_once(model: str, message: str) -> None:
    """Emit ``message`` at WARNING only the first time ``model`` fails pricing."""
    if model in _warned_pricing_models:
        return
    _warned_pricing_models.add(model)
    logger.warning(message)


# A route whose responses never carry a usage breakdown hits the estimated-basis
# fallback on *every* request, so the warning is deduped per model for the same
# reason pricing warnings are (#2504): one line per model per process, not one
# per request. The distinction stays permanently visible in /stats regardless.
_warned_estimated_basis_models: set[str] = set()


def _warn_estimated_basis_once(model: str) -> None:
    """Warn the first time ``model`` books a cost against Headroom's estimate."""
    if model in _warned_estimated_basis_models:
        return
    _warned_estimated_basis_models.add(model)
    logger.warning(
        "budget basis estimated: no usage breakdown from provider for %s — "
        "input cost booked from Headroom's own token count",
        model,
    )


class CostEntry(NamedTuple):
    """One booked cost, with the provenance of the input count behind it.

    ``basis`` is :data:`~headroom.proxy.budget_basis_policy.COST_BASIS_MEASURED`
    when the provider reported a usage breakdown, and ``COST_BASIS_ESTIMATED``
    when it didn't and Headroom's own ``tokens_sent`` stood in for the input
    count. Budget enforcement is a hard control, so the two must stay separable
    in the ledger rather than collapsing into an undifferentiated dollar figure.
    """

    timestamp: datetime
    cost_usd: float
    basis: str


#: Context size at which the major catalogs publish a second, higher price
#: tier (LiteLLM spells it ``*_above_200k_tokens``). A request's billed prompt
#: is compared against this to pick which rate applies.
_LONG_CONTEXT_THRESHOLD_TOKENS = 200_000


def _bucket_by_cache_mix(
    tokens: int,
    *,
    cache_read_tokens: int,
    cache_write_tokens: int,
    uncached_tokens: int,
) -> tuple[float, float, float]:
    """Split ``tokens`` into (read, write, list) shares by a request's cache mix.

    Tokens Headroom kept off the wire would have ridden the SAME region of the
    request as the mix passed in: cold-written at the provider's cache-write
    rate, read at its cache-read rate on warm turns, re-written when the TTL
    expired. The observed mix is the best available estimate of that rhythm (a
    TTL expiry shows up as a write-heavy mix on the real request, so the
    counterfactual re-write is captured per request rather than modelled).

    Callers choose WHICH mix to pass, because the two savings layers live in
    different regions of the request — see the call sites in ``record_tokens``.

    A request with no billed input breakdown falls back to list price for the
    whole amount.
    """
    billed_in = max(0, cache_read_tokens) + max(0, cache_write_tokens) + max(0, uncached_tokens)
    if tokens <= 0:
        return 0.0, 0.0, 0.0
    if billed_in <= 0:
        return 0.0, 0.0, float(tokens)
    read_part = tokens * max(0, cache_read_tokens) / billed_in
    write_part = tokens * max(0, cache_write_tokens) / billed_in
    return read_part, write_part, tokens - read_part - write_part


def _summarize_transforms(transforms: list[str]) -> str:
    """Collapse repeated transforms into counted summary.

    e.g. ['router:excluded:tool', 'router:excluded:tool', 'read_lifecycle:stale']
      → 'router:excluded:tool*2 read_lifecycle:stale'
    """
    if not transforms:
        return "none"
    counts: dict[str, int] = {}
    for t in transforms:
        counts[t] = counts.get(t, 0) + 1
    parts = [f"{k}*{v}" if v > 1 else k for k, v in counts.items()]
    return " ".join(parts)


def header_safe_transforms(transforms: list[str]) -> list[str]:
    """Strip enriched detail so each tag is safe in the comma-joined header.

    ``x-headroom-transforms`` is built as ``",".join(transforms_applied)``, so a
    tag must not itself contain a comma or the header can't be split back into
    tags. The enriched ``read_lifecycle:<state>:<path>`` and
    ``smart_crush:<n>:<names>`` tags carry comma-bearing detail (file paths may
    contain commas; tool-name lists are comma-separated), so collapse them back
    to their legacy counter shape for the header. Full detail stays in the
    structured ``transforms_applied`` list (dashboards, request logs, the
    desktop activity feed) — only the opaque header is normalized.
    """
    safe: list[str] = []
    for t in transforms:
        if t.startswith("smart_crush:"):
            parts = t.split(":")
            safe.append(f"smart_crush:{parts[1]}" if len(parts) >= 2 else t)
        elif t.startswith("read_lifecycle:"):
            parts = t.split(":")
            safe.append(f"read_lifecycle:{parts[1]}" if len(parts) >= 2 else t)
        else:
            safe.append(t)
    return safe


def build_prefix_cache_stats(
    metrics: PrometheusMetrics,
    cost_tracker: CostTracker | None,
) -> dict:
    """Build provider-aware prefix cache statistics for the dashboard."""
    by_provider: dict[str, dict[str, Any]] = {}
    totals: dict[str, Any] = {
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_write_5m_tokens": 0,
        "cache_write_1h_tokens": 0,
        "cache_write_5m_requests": 0,
        "cache_write_1h_requests": 0,
        "uncached_input_tokens": 0,
        "requests": 0,
        "hit_requests": 0,
        "bust_count": 0,
        "bust_write_tokens": 0,
        "savings_usd": 0.0,
        "write_premium_usd": 0.0,
    }

    for provider, pc in metrics.cache_by_provider.items():
        if pc["requests"] == 0:
            continue

        econ = cache_economics_for_provider(provider)
        read_mult = econ.read_multiplier
        write_mult = econ.write_multiplier

        # Get the base input price per token for the most-used model on this
        # provider. Pick the provider-matching, priced model with the highest
        # token volume — not the first one recorded. A Claude Code session sends
        # both Sonnet (main loop) and Haiku (titles/subagents); breaking on the
        # first-inserted model would price all cache savings at whichever happened
        # to be seen first (e.g. Haiku's $0.80/M vs Sonnet's $3/M), skewing the
        # dashboard's savings figure ~3.75x.
        input_price_per_token = None
        # Per-token cache rates for that same model, straight from LiteLLM's
        # catalog. Preferred over the _CACHE_ECONOMICS ratios below: those are
        # hardcoded per provider and can't express a model's or context tier's
        # actual published cache rates.
        cache_prices = None
        if cost_tracker:
            best_tokens = -1
            for model_name, tokens_sent in cost_tracker._tokens_sent_by_model.items():
                if model_matches_provider(provider, model_name) and tokens_sent > best_tokens:
                    price_per_1m = cost_tracker._get_list_price(model_name)
                    if price_per_1m:
                        input_price_per_token = price_per_1m / 1_000_000
                        cache_prices = cost_tracker._get_cache_prices(model_name)
                        best_tokens = tokens_sent

        # Catalog rates win; provider-owned defaults are the fallback for a model
        # that publishes no cache pricing.
        pricing_source = "provider_default"
        if cache_prices and input_price_per_token:
            _cr_price, _cw_price, _uncached_price = cache_prices
            if _uncached_price:
                read_mult = _cr_price / _uncached_price
                write_mult = _cw_price / _uncached_price
                pricing_source = "catalog"

        # Calculate savings:
        # Cache reads save (1.0 - read_mult) per token vs uncached input price.
        # Cache write premium stays visible as its own gross field, and net
        # savings subtract it so the dashboard reflects billed cache impact.
        read_tokens: int = pc["cache_read_tokens"]  # type: ignore[assignment]
        write_tokens: int = pc["cache_write_tokens"]  # type: ignore[assignment]
        write_5m_tokens: int = pc["cache_write_5m_tokens"]  # type: ignore[assignment]
        write_1h_tokens: int = pc["cache_write_1h_tokens"]  # type: ignore[assignment]
        write_5m_requests: int = pc["cache_write_5m_requests"]  # type: ignore[assignment]
        write_1h_requests: int = pc["cache_write_1h_requests"]  # type: ignore[assignment]
        savings_usd = 0.0
        write_premium_usd = 0.0

        if input_price_per_token:
            # Savings from reads: tokens * price * (1.0 - read_multiplier)
            savings_usd = read_tokens * input_price_per_token * (1.0 - read_mult)
            # Write premium is reported separately and subtracted from net savings.
            if write_mult > 1.0:
                write_premium_usd = write_tokens * input_price_per_token * (write_mult - 1.0)

        # Token-level hit rate: what % of total input tokens were served from cache?
        # This is more meaningful than request-level (binary "had any cache read").
        uncached_tokens: int = pc["uncached_input_tokens"]  # type: ignore[assignment]
        total_input = read_tokens + write_tokens + uncached_tokens
        hit_rate = round(read_tokens / total_input * 100, 1) if total_input > 0 else 0
        request_hit_rate = (
            round(pc["hit_requests"] / pc["requests"] * 100, 1) if pc["requests"] > 0 else 0
        )

        provider_stats: dict[str, Any] = {
            "cache_read_tokens": read_tokens,
            "cache_write_tokens": write_tokens,
            "cache_write_5m_tokens": write_5m_tokens,
            "cache_write_1h_tokens": write_1h_tokens,
            "cache_write_5m_requests": write_5m_requests,
            "cache_write_1h_requests": write_1h_requests,
            "uncached_input_tokens": uncached_tokens,
            "requests": pc["requests"],
            "hit_requests": pc["hit_requests"],
            "hit_rate": hit_rate,
            "request_hit_rate": request_hit_rate,
            "bust_count": pc["bust_count"],
            "bust_write_tokens": pc["bust_write_tokens"],
            "read_discount": f"{(1.0 - read_mult) * 100:.0f}%",
            "write_premium": f"{(write_mult - 1.0) * 100:.0f}%" if write_mult > 1.0 else "none",
            "savings_usd": round(savings_usd, 4),
            "write_premium_usd": round(write_premium_usd, 4),
            "net_savings_usd": round(savings_usd - write_premium_usd, 4),
            "cache_pricing_source": pricing_source,
            "label": econ.label,
            "observed_ttl_buckets": {
                "5m": {
                    "tokens": write_5m_tokens,
                    "requests": write_5m_requests,
                },
                "1h": {
                    "tokens": write_1h_tokens,
                    "requests": write_1h_requests,
                },
            },
        }
        total_observed_ttl_tokens = write_5m_tokens + write_1h_tokens
        if total_observed_ttl_tokens > 0:
            provider_stats["observed_ttl_mix"] = {
                "5m_pct": round(write_5m_tokens / total_observed_ttl_tokens * 100, 1),
                "1h_pct": round(write_1h_tokens / total_observed_ttl_tokens * 100, 1),
                "active_buckets": [
                    bucket
                    for bucket, tokens in (("5m", write_5m_tokens), ("1h", write_1h_tokens))
                    if tokens > 0
                ],
            }
        by_provider[provider] = provider_stats

        # Accumulate totals
        totals["cache_read_tokens"] += read_tokens
        totals["cache_write_tokens"] += write_tokens
        totals["cache_write_5m_tokens"] += write_5m_tokens
        totals["cache_write_1h_tokens"] += write_1h_tokens
        totals["cache_write_5m_requests"] += write_5m_requests
        totals["cache_write_1h_requests"] += write_1h_requests
        totals["uncached_input_tokens"] += uncached_tokens
        totals["requests"] += pc["requests"]
        totals["hit_requests"] += pc["hit_requests"]
        totals["bust_count"] += pc["bust_count"]
        totals["bust_write_tokens"] += pc["bust_write_tokens"]
        totals["savings_usd"] += savings_usd
        totals["write_premium_usd"] += write_premium_usd

    totals["net_savings_usd"] = round(totals["savings_usd"] - totals["write_premium_usd"], 4)
    totals["savings_usd"] = round(totals["savings_usd"], 4)
    totals["write_premium_usd"] = round(totals["write_premium_usd"], 4)
    # Token-level hit rate across all providers
    _total_input = (
        totals["cache_read_tokens"] + totals["cache_write_tokens"] + totals["uncached_input_tokens"]
    )
    totals["hit_rate"] = (
        round(totals["cache_read_tokens"] / _total_input * 100, 1) if _total_input > 0 else 0
    )
    totals["request_hit_rate"] = (
        round(totals["hit_requests"] / totals["requests"] * 100, 1) if totals["requests"] > 0 else 0
    )
    total_observed_ttl_tokens = totals["cache_write_5m_tokens"] + totals["cache_write_1h_tokens"]
    totals["observed_ttl_buckets"] = {
        "5m": {
            "tokens": totals["cache_write_5m_tokens"],
            "requests": totals["cache_write_5m_requests"],
        },
        "1h": {
            "tokens": totals["cache_write_1h_tokens"],
            "requests": totals["cache_write_1h_requests"],
        },
    }
    totals["observed_ttl_mix"] = {
        "5m_pct": round(totals["cache_write_5m_tokens"] / total_observed_ttl_tokens * 100, 1)
        if total_observed_ttl_tokens > 0
        else 0.0,
        "1h_pct": round(totals["cache_write_1h_tokens"] / total_observed_ttl_tokens * 100, 1)
        if total_observed_ttl_tokens > 0
        else 0.0,
        "active_buckets": [
            bucket
            for bucket, tokens in (
                ("5m", totals["cache_write_5m_tokens"]),
                ("1h", totals["cache_write_1h_tokens"]),
            )
            if tokens > 0
        ],
    }

    # Cache-miss attribution (#1313): why turns that expected a prompt-cache
    # hit missed instead. Per-provider reason buckets plus an aggregate total,
    # so the dashboard can show "of N expected-cache misses, X were TTL lapses
    # vs Y prefix changes" — the signal a user needs to decide 5m vs 1h TTL.
    _miss_by_provider: dict[str, dict[str, int]] = {}
    # Holds integer counts AND float percentages (ttl_expiry_pct etc.), so the
    # value type is float — ints coerce cleanly and the counts stay whole.
    _miss_totals: dict[str, float] = {
        "ttl_expiry": 0,
        "prefix_change": 0,
        "unknown": 0,
        "total": 0,
    }
    for _provider, _reasons in metrics.cache_miss_attribution_by_provider.items():
        provider_reasons = {reason: int(count) for reason, count in _reasons.items()}
        provider_total = sum(provider_reasons.values())
        if provider_total == 0:
            continue
        provider_reasons["total"] = provider_total
        _miss_by_provider[_provider] = provider_reasons
        for reason, count in provider_reasons.items():
            if reason == "total":
                continue
            _miss_totals[reason] = _miss_totals.get(reason, 0) + count
        _miss_totals["total"] += provider_total

    # Share of misses attributable to TTL lapse vs prefix change — the headline
    # the dashboard renders. Computed against attributed (non-unknown) misses
    # so an "unknown" bucket doesn't dilute the actionable split.
    _attributed = _miss_totals["ttl_expiry"] + _miss_totals["prefix_change"]
    _miss_totals["ttl_expiry_pct"] = (
        round(_miss_totals["ttl_expiry"] / _attributed * 100, 1) if _attributed > 0 else 0.0
    )
    _miss_totals["prefix_change_pct"] = (
        round(_miss_totals["prefix_change"] / _attributed * 100, 1) if _attributed > 0 else 0.0
    )

    return {
        "by_provider": by_provider,
        "totals": totals,
        "miss_attribution": {
            "totals": _miss_totals,
            "by_provider": _miss_by_provider,
        },
        "prefix_freeze": {
            "busts_avoided": metrics.prefix_freeze_busts_avoided,
            "tokens_preserved": metrics.prefix_freeze_tokens_preserved,
            "compression_foregone_tokens": metrics.prefix_freeze_compression_foregone,
            "net_benefit_tokens": (
                metrics.prefix_freeze_tokens_preserved - metrics.prefix_freeze_compression_foregone
            ),
        },
        "compression_vs_cache": {
            "tokens_saved_by_compression": metrics.tokens_saved_total,
            "tokens_lost_to_cache_bust": metrics.cache_bust_tokens_lost,
            "cache_bust_count": metrics.cache_bust_count,
            "net_tokens": metrics.tokens_saved_total - metrics.cache_bust_tokens_lost,
        },
        "attribution": (
            "Prefix caching is performed by the LLM provider (Anthropic, OpenAI). "
            "Headroom reports cache stats as observed from API responses. "
            "CacheAligner and prefix freeze improve cache hit rates by stabilizing "
            "the message prefix, but baseline caching happens without Headroom. "
            "Observed TTL bucket metrics reflect provider-reported cache write usage "
            "(for example Anthropic 5m vs 1h), not configured or remaining TTL."
        ),
    }


def merge_cost_stats(
    cost_stats: dict | None,
    cache_stats: dict,
) -> dict | None:
    """Merge compression and cache savings into cost stats.

    Each savings layer is reported separately with its own scope:
    - savings_usd: compression savings at model list price (monotonic)
    - cache_savings_usd: prefix cache discount from provider (separate)

    The dollar metric (savings_usd) remains ONLY proxy compression savings
    priced at the model's published input rate.
    Prefix cache savings stay separate because they are a provider discount,
    not token removal. This avoids the non-monotonic moving-average repricing
    bug (#83).
    """
    if cost_stats is None:
        return None

    cache_net = cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
    compression_savings = cost_stats.get("savings_usd", 0.0)

    return {
        **cost_stats,
        "savings_usd": round(compression_savings, 4),
        "compression_savings_usd": round(compression_savings, 4),
        "cache_savings_usd": round(cache_net, 4),
    }


def _aggregate_mcp_events() -> dict[str, int]:
    """Aggregate compression / retrieval events written by Headroom MCP
    server instances to the cross-process shared events file.

    The Headroom MCP server (``headroom mcp serve``) records every
    ``headroom_compress`` and ``headroom_retrieve`` invocation to a
    file-locked shared log (see :func:`headroom.ccr.mcp_server._append_shared_event`).
    This helper reads that log and aggregates within the rolling window
    so the proxy's ``/stats`` can surface MCP-side work alongside the
    proxy's own HTTP-path compression numbers.

    Returns zeros for every key if the MCP SDK isn't installed, the
    shared file doesn't exist yet, or any read error occurs — the
    intent is "if there's nothing to report, report zero" so this
    helper never blocks the summary.

    Keys: ``compressions`` (count of headroom_compress calls),
    ``tokens_removed`` (sum of input_tokens-output_tokens across
    compress events), ``retrievals`` (count of headroom_retrieve
    calls — the load-bearing over-compression signal).
    """
    zero = {"compressions": 0, "tokens_removed": 0, "retrievals": 0}
    try:
        from headroom.ccr.mcp_server import _read_shared_events
    except ImportError:
        return zero

    try:
        events = _read_shared_events()
    except Exception:  # noqa: BLE001 — never break /stats on a stats-read error
        return zero

    compressions = 0
    tokens_removed = 0
    retrievals = 0
    for evt in events:
        kind = evt.get("type")
        if kind == "compress":
            compressions += 1
            in_tok = int(evt.get("input_tokens", 0) or 0)
            out_tok = int(evt.get("output_tokens", 0) or 0)
            tokens_removed += max(0, in_tok - out_tok)
        elif kind == "retrieve":
            retrievals += 1
    return {
        "compressions": compressions,
        "tokens_removed": tokens_removed,
        "retrievals": retrievals,
    }


def build_session_summary(
    proxy: Any,
    metrics: Any,
    prefix_cache_stats: dict,
    total_tokens_before: int,
) -> dict[str, Any]:
    """Build a human-readable session summary from metrics and request logs.

    This is the headline view users see first in /stats — designed to answer
    "is Headroom working?" at a glance.
    """
    # Analyze per-request compression from the logger
    compressed_requests: list[dict] = []
    uncompressed_reasons: dict[str, int] = {
        "prefix_frozen": 0,
        "too_small": 0,
        "passthrough": 0,
        "no_compressible_content": 0,
        "unknown_token_accounting": 0,
    }

    def _entry_has_number(entry: Any, attr: str) -> bool:
        value = getattr(entry, attr, None)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def _entry_number(entry: Any, attr: str) -> int | float:
        value = getattr(entry, attr, 0)
        return value if _entry_has_number(entry, attr) else 0

    if proxy.logger:
        for entry in proxy.logger._logs:
            if entry.model and "count_tokens" in entry.model:
                uncompressed_reasons["passthrough"] += 1
                continue
            tokens_saved = _entry_number(entry, "tokens_saved")
            input_tokens_original = _entry_number(entry, "input_tokens_original")
            input_tokens_optimized = _entry_number(entry, "input_tokens_optimized")
            has_complete_token_accounting = all(
                _entry_has_number(entry, attr)
                for attr in (
                    "input_tokens_original",
                    "input_tokens_optimized",
                    "tokens_saved",
                    "savings_percent",
                )
            )
            if tokens_saved > 0 and has_complete_token_accounting:
                compressed_requests.append(
                    {
                        "savings_pct": round(_entry_number(entry, "savings_percent"), 1),
                        "tokens_saved": tokens_saved,
                        "original": input_tokens_original,
                        "optimized": input_tokens_optimized,
                    }
                )
            elif not has_complete_token_accounting:
                uncompressed_reasons["unknown_token_accounting"] += 1
            elif input_tokens_original > 0:
                # Categorize why it wasn't compressed
                transforms = entry.transforms_applied or []
                if not transforms:
                    # Pipeline returned unchanged — likely all frozen
                    uncompressed_reasons["prefix_frozen"] += 1
                elif all("excluded" in t or "protected" in t for t in transforms):
                    uncompressed_reasons["no_compressible_content"] += 1
                elif input_tokens_original < 500:
                    uncompressed_reasons["too_small"] += 1
                else:
                    uncompressed_reasons["prefix_frozen"] += 1

    # Compute compression stats for requests that DID compress
    avg_compression = 0.0
    best_compression = 0.0
    best_detail = ""
    if compressed_requests:
        # Size-WEIGHTED, not a mean of per-request percentages. An unweighted
        # mean lets one tiny, highly-compressible request (e.g. a repeated log
        # line that folds 6,070 -> 83 tokens, 98.6%) dominate the headline while
        # the large real requests it is averaged with barely moved, so the card
        # can read "18.6% saved" on traffic whose forwarded bytes fell ~3%.
        # Weighting by original size makes the number mean what an operator
        # reads it as: the share of total tokens actually removed.
        _orig_total = sum(r["original"] for r in compressed_requests)
        _saved_total = sum(r["tokens_saved"] for r in compressed_requests)
        avg_compression = round(100.0 * _saved_total / _orig_total, 1) if _orig_total else 0.0
        best = max(compressed_requests, key=lambda r: r["savings_pct"])
        best_compression = best["savings_pct"]
        best_detail = f"{best['original']:,} → {best['optimized']:,} tokens"

    # Cost summary — what Headroom itself saved, priced into one card.
    #
    # The headline counts only layers Headroom causes: message compression,
    # tool-schema deferral and extension-attributed dollars. The provider's
    # prefix-cache discount is reported BESIDE it, never inside it: that
    # discount is paid on cache reads whether or not Headroom is in the path
    # (CacheAligner and prefix freeze improve the hit rate, they don't create
    # the discount), and it is an order of magnitude larger than compression on
    # a long agent session, so folding it in made a card that couldn't be
    # reconciled with the Tokens Saved counter next to it.
    #
    # Rows the attribution ledger flags ``realized=False`` (e.g. a model-routing
    # extension's decision-time delta) are summed under a ``projected`` key so
    # a reader can still split measured from estimated. Budget enforcement
    # keeps reading the unwidened ``savings_usd`` from the tracker.
    cost_stats = proxy.cost_tracker.stats() if proxy.cost_tracker else {}
    cost_input = cost_stats.get("cost_with_headroom_usd", 0.0)
    cost_output = cost_stats.get("output_cost_usd", 0.0)
    cost_with = cost_stats.get("total_cost_usd", cost_input)
    # Cache-aware valuation of the compressed-away tokens: what they would
    # actually have been billed at, given the cache mix of the requests they
    # were removed from. The list-priced figure feeds budgets and is kept in
    # the breakdown for comparison.
    compression_savings_list = cost_stats.get("savings_usd", 0.0)
    compression_savings = cost_stats.get("cache_aware_savings_usd", compression_savings_list)
    tool_savings = cost_stats.get("tool_savings_usd", 0.0)
    cache_net = prefix_cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
    ext_realized_usd = 0.0
    ext_projected_usd = 0.0
    for row in getattr(metrics, "savings_by_source", {}).values():
        try:
            usd = float(row.get("usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        # tokens-only rows (tool_search) carry usd=0 here — the deferral is
        # priced from the tracker above, so nothing double-counts.
        if row.get("realized", True):
            ext_realized_usd += usd
        else:
            ext_projected_usd += usd
    measured_saved = compression_savings + tool_savings + ext_realized_usd
    total_saved_usd = round(measured_saved + ext_projected_usd, 2)
    # Baseline = what this session would have cost without Headroom, on the same
    # provider terms (cache discount included on both sides, since it applies
    # either way).
    cost_without = cost_with + measured_saved + ext_projected_usd
    # `> 0` is not a sufficient guard. ext_projected_usd is SIGNED -- routing
    # reports an upgrade as a negative saving -- so a session that routed up can
    # drive the baseline arbitrarily close to zero while real money was spent:
    # $1.59 spent + $0.63 measured - $2.21 projected left a $0.01 baseline, and
    # the card read "-39666.1%". A baseline that small is a cancellation
    # artifact, not a measurement, so report no percentage rather than one that
    # is wrong by three orders of magnitude.
    _baseline_is_meaningful = cost_without > max(0.01, 0.05 * cost_with)
    savings_pct_cost = (
        round(total_saved_usd / cost_without * 100, 1) if _baseline_is_meaningful else 0.0
    )

    # Primary models used
    models = dict(metrics.requests_by_model)
    primary_model = max(models, key=lambda k: models[k]) if models else "unknown"
    api_requests = sum(v for k, v in models.items() if "count_tokens" not in k)

    # Build the summary
    summary: dict[str, Any] = {
        "mode": proxy.config.mode,
        "api_requests": api_requests,
        "primary_model": primary_model,
        "compression": {
            "requests_compressed": len(compressed_requests),
            "avg_compression_pct": avg_compression,
            "best_compression_pct": best_compression,
            "best_detail": best_detail,
            "total_tokens_removed": metrics.tokens_saved_total,
            "total_tokens_before": total_tokens_before,
            # Tool-schema deferral / turn-hook tool shrink, tracked apart from
            # message compression, so consumers can see the full picture.
            "tool_schema_tokens_saved": getattr(metrics, "tool_search_saved_total", 0),
            "total_tokens_saved_all_layers": (
                metrics.tokens_saved_total + getattr(metrics, "tool_search_saved_total", 0)
            ),
        },
        "uncompressed_requests": {k: v for k, v in uncompressed_reasons.items() if v > 0},
        "cost": {
            "without_headroom_usd": round(cost_without, 2),
            "with_headroom_usd": round(cost_with, 2),
            "with_headroom_input_usd": round(cost_input, 2),
            "with_headroom_output_usd": round(cost_output, 2),
            "total_saved_usd": total_saved_usd,
            "measured_saved_usd": round(measured_saved, 2),
            "projected_saved_usd": round(ext_projected_usd, 2),
            "savings_pct": savings_pct_cost,
            # Provider-side, NOT part of total_saved_usd: the discount the
            # provider gives on cache reads, which Headroom helps land but does
            # not create. Already reflected in with_headroom_usd.
            "provider_cache_discount_usd": round(cache_net, 2),
            "breakdown": {
                "cache_savings_usd": round(cache_net, 2),
                "compression_savings_usd": round(compression_savings, 2),
                "compression_savings_list_usd": round(compression_savings_list, 2),
                "tool_search_savings_usd": round(tool_savings, 2),
                "extension_realized_usd": round(ext_realized_usd, 2),
                "extension_projected_usd": round(ext_projected_usd, 2),
            },
        },
    }

    # MCP-side compression: events written by `headroom mcp serve`
    # instances (one or more) to the shared stats log. Surfaces direct
    # tool invocations the proxy HTTP path never sees, plus the
    # `retrievals` counter — the load-bearing signal for over-compression
    # (if it grows linearly with turn count, our lossy compressors are
    # dropping info the model actually needs).
    summary["mcp"] = _aggregate_mcp_events()

    # Codex WS sessions compress per-unit on the long-lived /responses socket,
    # but turn-level records (which feed tokens_saved_total above) only land
    # when a response.completed frame carries usage. Surface the live per-unit
    # counters so a WS-only session doesn't read as "no activity" mid-turn.
    # Kept as a separate block rather than summed into the compression totals:
    # turns that DID record already contributed the same savings there, so
    # adding the unit sums on top would double-count.
    ws_units = getattr(metrics, "codex_ws_units_total", 0)
    if ws_units:
        summary["codex_ws"] = {
            "units_total": ws_units,
            "units_modified": getattr(metrics, "codex_ws_units_modified_total", 0),
            "tokens_saved": getattr(metrics, "codex_ws_unit_tokens_saved_sum", 0),
        }

    # Add tip if token mode would help
    if proxy.config.mode == PROXY_MODE_CACHE and uncompressed_reasons["prefix_frozen"] > 10:
        summary["tip"] = (
            "Most requests are prefix-frozen. Set HEADROOM_MODE=token "
            "to compress frozen messages and extend your session by ~25-35%."
        )

    return summary


class CostTracker:
    """Track costs and enforce budgets.

    Cost history is automatically pruned to prevent unbounded memory growth:
    - Entries older than 24 hours are removed
    - Maximum of 100,000 entries are kept

    Uses LiteLLM's community-maintained pricing database for accurate costs.
    See: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
    """

    MAX_COST_ENTRIES = 100_000
    # Used by _prune_old_costs(), called from record_tokens() on every request.
    # Must be >= the longest budget_period (monthly = up to 31 days), otherwise
    # get_period_cost() undercounts and check_budget() silently under-enforces.
    COST_RETENTION_HOURS = 744  # 31 days

    def __init__(
        self,
        budget_limit_usd: float | None = None,
        budget_period: str = "daily",
        estimated_basis_policy: str = DEFAULT_POLICY,
    ):
        self.budget_limit_usd = budget_limit_usd
        self.budget_period = budget_period
        # What estimated-basis spend does to enforcement. Normalized here so a
        # bad value degrades to the default instead of quietly disabling the
        # budget. See headroom.proxy.budget_basis_policy.
        self.estimated_basis_policy = resolve_estimated_basis_policy(estimated_basis_policy)

        # Cost tracking - using deque for efficient left-side removal
        self._costs: deque[CostEntry] = deque(maxlen=self.MAX_COST_ENTRIES)
        self._last_prune_time: datetime = datetime.now()

        # Token savings per model (exact, no dollar estimation)
        self._tokens_saved_by_model: dict[str, int] = {}
        # Cache-aware counterfactual buckets for compressed-away message tokens,
        # the same treatment the deferred tool schemas get below. Floats: shares
        # of a request's removed tokens, not whole tokens. Keyed by
        # ``(model, long_context)`` so a >200k-context turn is priced at the tier
        # the provider actually billed it at rather than the base rate.
        self._saved_write_by_tier: dict[tuple[str, bool], float] = {}
        self._saved_list_by_tier: dict[tuple[str, bool], float] = {}
        # Tool-schema deferral per model, DISJOINT from _tokens_saved_by_model
        # (deferred schemas are never in the message counts). Tracked separately
        # so the compression-only figure stays available; `stats()` reports both
        # the split and the sum.
        self._tool_saved_by_model: dict[str, int] = {}
        # Cache-aware counterfactual buckets for the deferred schemas. Had the
        # schemas been sent, they would have ridden the SAME prefix as the rest
        # of the request — cold-written at the provider's cache-write rate,
        # read at its cache-read rate on warm turns, re-written when the TTL
        # expired. The observed request's own cache mix is the best available
        # estimate of that rhythm (a TTL expiry shows up as a write-heavy mix
        # on the real request, so the counterfactual re-write is captured per
        # request, not modelled). Priced in stats() via _get_cache_prices —
        # LiteLLM's per-model catalog — so the rates stay provider-agnostic.
        # Floats: shares of a request's schema tokens, not whole tokens.
        self._tool_saved_read_by_model: dict[str, float] = {}
        self._tool_saved_write_by_model: dict[str, float] = {}
        self._tool_saved_list_by_model: dict[str, float] = {}
        self._tokens_sent_by_model: dict[str, int] = {}
        # Completion tokens keyed by ``(model, long_context)`` — same reason as
        # the savings buckets: the >200k completion rate is a different number.
        self._output_tokens_by_tier: dict[tuple[str, bool], int] = {}
        self._requests_by_model: dict[str, int] = {}

        # API-reported cache breakdown per model (for accurate cost calculation)
        self._api_cache_read_by_model: dict[str, int] = {}
        self._api_cache_write_by_model: dict[str, int] = {}
        self._api_cache_write_5m_by_model: dict[str, int] = {}
        self._api_cache_write_1h_by_model: dict[str, int] = {}
        self._api_uncached_by_model: dict[str, int] = {}

    def reset_runtime(self) -> None:
        """Reset in-memory cost/token counters for local test/debug use."""
        self._costs.clear()
        self._last_prune_time = datetime.now()
        self._tokens_saved_by_model.clear()
        self._saved_write_by_tier.clear()
        self._saved_list_by_tier.clear()
        self._tool_saved_by_model.clear()
        self._tool_saved_read_by_model.clear()
        self._tool_saved_write_by_model.clear()
        self._tool_saved_list_by_model.clear()
        self._tokens_sent_by_model.clear()
        self._output_tokens_by_tier.clear()
        self._requests_by_model.clear()
        self._api_cache_read_by_model.clear()
        self._api_cache_write_by_model.clear()
        self._api_cache_write_5m_by_model.clear()
        self._api_cache_write_1h_by_model.clear()
        self._api_uncached_by_model.clear()

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        """Estimate cost in USD using LiteLLM's pricing database.

        LiteLLM natively handles cache_read and cache_creation pricing
        for all providers (Anthropic, OpenAI, Google, etc.) in a single call.

        Args:
            model: Model name for pricing lookup
            input_tokens: Non-cached input tokens (excludes cache_read)
            output_tokens: Output tokens
            cache_read_tokens: Tokens served from cache (~10% of input rate)
            cache_write_tokens: Tokens written to cache (~125% of input rate)
        """
        litellm = _get_litellm_module()
        if litellm is None:
            _warn_pricing_once(
                f"__litellm_unavailable__:{model}",
                f"LiteLLM not available - cannot calculate costs for model {model}",
            )
            return None

        try:
            from headroom.pricing.litellm_pricing import resolve_litellm_model

            resolved_model = resolve_litellm_model(model)

            # litellm.cost_per_token handles all token types natively:
            # prompt_tokens at input rate, cache_read at ~10%, cache_creation at ~125%
            input_cost, output_cost = litellm.cost_per_token(
                model=resolved_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cache_read_input_tokens=cache_read_tokens,
                cache_creation_input_tokens=cache_write_tokens,
            )

            total_cost = input_cost + output_cost
            return float(total_cost) if total_cost > 0 else None

        except Exception as e:
            _warn_pricing_once(model, f"Failed to get pricing for model {model}: {e}")
            return None

    def _prune_old_costs(self):
        """Remove cost entries older than retention period.

        Called periodically (every 5 minutes) to prevent unbounded memory growth.
        The deque maxlen provides a hard cap, but time-based pruning keeps
        memory usage proportional to actual traffic patterns.
        """
        now = datetime.now()
        # Only prune every 5 minutes to avoid overhead
        if (now - self._last_prune_time).total_seconds() < 300:
            return

        self._last_prune_time = now
        cutoff = now - timedelta(hours=self.COST_RETENTION_HOURS)

        # Remove entries from the left (oldest) while they're older than cutoff
        while self._costs and self._costs[0].timestamp < cutoff:
            self._costs.popleft()

    def record_tokens(
        self,
        model: str,
        tokens_saved: int,
        tokens_sent: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
        uncached_tokens: int = 0,
        output_tokens: int = 0,
        cache_inferred: bool = False,
        tool_schema_saved: int = 0,
    ):
        """Record token counts per model and accumulate request cost for budget enforcement.

        Args:
            model: Model name.
            tokens_saved: Tokens removed by compression (Headroom's count).
            tokens_sent: Compressed message tokens sent (Headroom's count).
            cache_read_tokens: Cache read tokens from API response usage.
            cache_write_tokens: Cache write tokens from API response usage.
            uncached_tokens: Non-cached input tokens from API response usage.
            output_tokens: Output tokens from API response usage.
            cache_inferred: True when ``cache_write_tokens`` was DERIVED from the
                uncached portion rather than reported by the provider (OpenAI
                exposes no write counter). Such a value is the same tokens as
                ``uncached_tokens``, so it is excluded from the billed prompt
                total and from the write premium. Defaults False, which preserves
                behaviour for providers that report disjoint buckets.
            tool_schema_saved: Tokens withheld by tool-schema deferral for this
                request. Disjoint from ``tokens_saved`` — deferred schemas never
                enter the message token counts, so they moved neither
                ``tokens_saved`` nor ``tokens_sent`` and had nowhere to be
                attributed. The dashboard's per-model "Tokens Saved" column
                therefore showed compression only, while the headline above it
                counted both.
        """
        # Post-guard invariant (all providers): Headroom never forwards a request
        # larger than the original (handlers revert any inflation before sending),
        # so compression savings are >= 0 by construction. A negative here is an
        # intermediate/hook token-count artifact that never reached the model;
        # clamp it so `total_tokens_removed` reflects actually-forwarded bytes
        # instead of surfacing spurious negatives (verified clean on the wire).
        if tokens_saved < 0:
            logger.debug(
                "record_tokens: clamping negative tokens_saved=%d to 0 for %s (artifact; wire not inflated)",
                tokens_saved,
                model,
            )
            tokens_saved = 0
        self._tokens_saved_by_model[model] = (
            self._tokens_saved_by_model.get(model, 0) + tokens_saved
        )
        self._tool_saved_by_model[model] = self._tool_saved_by_model.get(model, 0) + max(
            0, tool_schema_saved
        )
        # A request the provider reported no cache data for (or whose write
        # counter was inferred, not billed) falls back to list price for its
        # full share.
        write_eff = 0 if cache_inferred else max(0, cache_write_tokens)
        billed_prompt = max(0, cache_read_tokens) + write_eff + max(0, uncached_tokens)
        long_context = max(billed_prompt, tokens_sent) > _LONG_CONTEXT_THRESHOLD_TOKENS
        if tokens_saved > 0:
            # Message compression works the LIVE ZONE only: handlers freeze the
            # cached prefix (system + prior turns) byte-for-byte for prefix-cache
            # safety and compress the newly appended delta. The removed tokens
            # therefore could never have been billed as cache reads — the read
            # bucket is the frozen prefix, which Headroom did not touch. Pricing
            # them by the WHOLE request's mix valued a warm turn's savings at
            # ~0.1x, an order of magnitude under what the provider would have
            # charged for that same content in the live zone. Split over the
            # live-zone mix (write + uncached) instead; tool-schema deferral
            # below keeps the full-request mix, because deferred schemas do sit
            # in the cached prefix.
            _read, c_write, c_list = _bucket_by_cache_mix(
                tokens_saved,
                cache_read_tokens=0,
                cache_write_tokens=write_eff,
                uncached_tokens=uncached_tokens,
            )
            wkey = (model, long_context)
            self._saved_write_by_tier[wkey] = self._saved_write_by_tier.get(wkey, 0.0) + c_write
            self._saved_list_by_tier[wkey] = self._saved_list_by_tier.get(wkey, 0.0) + c_list
        if tool_schema_saved > 0:
            # Split this request's deferred schema tokens by the request's own
            # observed cache mix: the schemas would have shared the prefix, so
            # they inherit its read/write/uncached proportions.
            read_part, write_part, list_part = _bucket_by_cache_mix(
                tool_schema_saved,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=write_eff,
                uncached_tokens=uncached_tokens,
            )
            self._tool_saved_read_by_model[model] = (
                self._tool_saved_read_by_model.get(model, 0.0) + read_part
            )
            self._tool_saved_write_by_model[model] = (
                self._tool_saved_write_by_model.get(model, 0.0) + write_part
            )
            self._tool_saved_list_by_model[model] = (
                self._tool_saved_list_by_model.get(model, 0.0) + list_part
            )
        self._tokens_sent_by_model[model] = self._tokens_sent_by_model.get(model, 0) + tokens_sent
        okey = (model, long_context)
        self._output_tokens_by_tier[okey] = self._output_tokens_by_tier.get(okey, 0) + max(
            0, output_tokens
        )
        self._requests_by_model[model] = self._requests_by_model.get(model, 0) + 1
        self._api_cache_read_by_model[model] = (
            self._api_cache_read_by_model.get(model, 0) + cache_read_tokens
        )
        self._api_cache_write_by_model[model] = (
            self._api_cache_write_by_model.get(model, 0) + cache_write_tokens
        )
        self._api_cache_write_5m_by_model[model] = (
            self._api_cache_write_5m_by_model.get(model, 0) + cache_write_5m_tokens
        )
        self._api_cache_write_1h_by_model[model] = (
            self._api_cache_write_1h_by_model.get(model, 0) + cache_write_1h_tokens
        )
        self._api_uncached_by_model[model] = (
            self._api_uncached_by_model.get(model, 0) + uncached_tokens
        )

        # Populate _costs so check_budget() has real data to enforce against.
        # When the call site had no API usage breakdown (all cache/uncached
        # fields are 0), fall back to tokens_sent so input cost isn't
        # silently dropped from the budget.
        #
        # That fallback is a guess, and check_budget() is a hard control, so the
        # record is stamped ``estimated`` and warned about once per model (#2713).
        # The fallback behaviour itself is unchanged — the estimate is now
        # labelled rather than indistinguishable from provider-reported usage.
        # ``litellm.cost_per_token`` wants the TOTAL prompt in ``prompt_tokens``:
        # measured, it charges
        #     (prompt - cache_read - cache_creation) * input_rate
        #   + cache_read * read_rate
        #   + cache_creation * write_rate
        # Passing only the uncached slice therefore drives the input term
        # NEGATIVE once anything was cached, and ``estimate_cost`` returns None on
        # a non-positive total — so no CostEntry was appended and ``check_budget()``
        # saw $0. Every cache-warm request, i.e. the normal case in an agent
        # session, was booking zero spend and the budget could never trip.
        # Measured before this fix, 100k prompt with 80k cached:
        #   gpt-5 $-0.065, gpt-4o-mini $-0.003, claude-sonnet-4-5 $-0.156.
        #
        # An INFERRED cache-write (OpenAI exposes no write counter, so the
        # uncached portion is used as a write proxy) is the SAME tokens as
        # ``uncached_tokens``. Adding it to the total would double-count the
        # prompt, and charging it at the write premium would invent a cost OpenAI
        # does not have — so it is excluded from both.
        effective_cache_write = write_eff
        basis = COST_BASIS_MEASURED
        input_tokens = uncached_tokens + cache_read_tokens + effective_cache_write
        if not (uncached_tokens or cache_read_tokens or cache_write_tokens):
            input_tokens = tokens_sent
            basis = COST_BASIS_ESTIMATED
            _warn_estimated_basis_once(model)
        cost = self.estimate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=effective_cache_write,
        )
        if cost is not None:
            self._costs.append(CostEntry(datetime.now(), cost, basis))
            self._prune_old_costs()

    def _period_cutoff(self) -> datetime:
        """Start of the current budget period."""
        now = datetime.now()

        if self.budget_period == "hourly":
            return now - timedelta(hours=1)
        if self.budget_period == "daily":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        # monthly
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def get_period_cost(self, basis: str | None = None) -> float:
        """Get cost for current budget period.

        With no argument this is the total spend booked in the period,
        regardless of how each record's input count was derived. Pass a basis
        (``"measured"`` / ``"estimated"``) to get just that slice.
        """
        cutoff = self._period_cutoff()
        return sum(
            entry.cost_usd
            for entry in self._costs
            if entry.timestamp >= cutoff and (basis is None or entry.basis == basis)
        )

    def period_cost_breakdown(self) -> dict[str, Any]:
        """Split the period's booked spend by how its input count was derived.

        ``estimated_usd`` is spend whose input token count came from Headroom's
        own estimate because the provider returned no usage breakdown. Keeping
        it separable is the point: a budget refusal driven by a guess should be
        distinguishable from one driven by provider-reported usage (#2713).
        """
        cutoff = self._period_cutoff()
        measured_usd = 0.0
        estimated_usd = 0.0
        records = 0
        estimated_records = 0
        for entry in self._costs:
            if entry.timestamp < cutoff:
                continue
            records += 1
            if entry.basis == COST_BASIS_ESTIMATED:
                estimated_usd += entry.cost_usd
                estimated_records += 1
            else:
                measured_usd += entry.cost_usd

        total_usd = measured_usd + estimated_usd
        return {
            "period": self.budget_period,
            "policy": self.estimated_basis_policy,
            "total_usd": total_usd,
            "measured_usd": measured_usd,
            "estimated_usd": estimated_usd,
            "estimated_pct": round(estimated_usd / total_usd * 100, 1) if total_usd > 0 else 0.0,
            "records": records,
            "estimated_records": estimated_records,
        }

    def check_budget(self) -> tuple[bool, float]:
        """Check if within budget. Returns (allowed, remaining).

        How estimated-basis spend participates is governed by
        ``estimated_basis_policy``: ``count`` (default) enforces against total
        spend exactly as before, ``ignore`` enforces against provider-measured
        spend only, and ``block`` refuses outright once the period holds any
        estimated spend rather than enforcing a hard limit against a guess.
        """
        if self.budget_limit_usd is None:
            return True, float("inf")

        breakdown = self.period_cost_breakdown()

        if self.estimated_basis_policy == BUDGET_BASIS_BLOCK and breakdown["estimated_usd"] > 0:
            return False, 0.0

        if self.estimated_basis_policy == BUDGET_BASIS_IGNORE:
            period_cost = breakdown["measured_usd"]
        else:
            period_cost = breakdown["total_usd"]

        remaining = self.budget_limit_usd - period_cost
        return remaining > 0, max(0, remaining)

    def budget_denial_detail(self) -> str:
        """Human-readable reason a request was refused on budget grounds.

        Built here rather than in the handler so the message can name what the
        ledger actually knows — specifically how much of the period's spend was
        booked from Headroom's own token estimate.
        """
        breakdown = self.period_cost_breakdown()
        estimated_usd = breakdown["estimated_usd"]

        if self.estimated_basis_policy == BUDGET_BASIS_BLOCK and estimated_usd > 0:
            return (
                f"Budget enforcement blocked for {self.budget_period} period: "
                f"${estimated_usd:.4f} of ${breakdown['total_usd']:.4f} was booked from "
                "Headroom's own token estimate because the provider returned no usage "
                f"breakdown, and {ENV_VAR}=block refuses to enforce a budget on an "
                "estimate. Set it to 'count' or 'ignore' to serve these requests."
            )

        detail = f"Budget exceeded for {self.budget_period} period"
        if estimated_usd > 0:
            detail += (
                f" (${estimated_usd:.4f} of ${breakdown['total_usd']:.4f} booked from "
                f"Headroom token estimates, not provider-reported usage)"
            )
        return detail

    def _get_list_price(self, model: str) -> float | None:
        """Get list input price per 1M tokens for a model."""
        litellm = _get_litellm_module()
        if litellm is None:
            return None
        try:
            from headroom.pricing.litellm_pricing import resolve_litellm_model

            resolved = resolve_litellm_model(model)
            info = litellm.model_cost.get(resolved, {})
            cost_per_token = info.get("input_cost_per_token")
            return cost_per_token * 1_000_000 if cost_per_token else None
        except Exception:
            return None

    def _get_output_price(self, model: str, *, long_context: bool = False) -> float | None:
        """Get the per-token completion price for a model, or None if unpriced.

        ``long_context`` selects the catalog's above-200k completion rate where
        the model publishes one (Anthropic charges 1.5x there).
        """
        litellm = _get_litellm_module()
        if litellm is None:
            return None
        try:
            from headroom.pricing.litellm_pricing import resolve_litellm_model

            resolved = resolve_litellm_model(model)
            info = litellm.model_cost.get(resolved, {})
            base = info.get("output_cost_per_token")
            if long_context:
                return info.get("output_cost_per_token_above_200k_tokens") or base or None
            return base or None
        except Exception:
            return None

    def _get_cache_prices(
        self, model: str, *, long_context: bool = False
    ) -> tuple[float, float, float] | None:
        """Get per-token prices for cache read, cache write, and uncached input.

        Returns (cache_read, cache_write, uncached) per-token costs, or None
        if pricing is unavailable. Uses LiteLLM's native cache pricing data.

        ``long_context`` picks the above-200k tier for each of the three rates,
        falling back per rate to the base one for a model that publishes no
        long-context price.
        """
        litellm = _get_litellm_module()
        if litellm is None:
            return None
        try:
            from headroom.pricing.litellm_pricing import resolve_litellm_model

            resolved = resolve_litellm_model(model)
            info = litellm.model_cost.get(resolved, {})
            uncached = info.get("input_cost_per_token")
            if not uncached:
                return None
            cache_read = info.get("cache_read_input_token_cost", uncached)
            cache_write = info.get("cache_creation_input_token_cost", uncached)
            if long_context:
                uncached = info.get("input_cost_per_token_above_200k_tokens") or uncached
                cache_read = info.get("cache_read_input_token_cost_above_200k_tokens") or cache_read
                cache_write = (
                    info.get("cache_creation_input_token_cost_above_200k_tokens") or cache_write
                )
            return (cache_read, cache_write, uncached)
        except Exception:
            return None

    def totals(self) -> tuple[int, float]:
        """Return just ``(total_input_tokens, total_input_cost_usd)``.

        The same two numbers ``stats()`` reports, computed without the rest of
        it. ``stats()`` is called once per request by the metrics path, which
        reads exactly these two fields and discards ``per_model``,
        ``savings_usd``, ``cost_with_headroom_usd`` and — the expensive one —
        ``budget_basis``, whose ``period_cost_breakdown()`` walks up to 31 days
        of retained cost records. MEASURED 2.8ms at 20k records and 13.6ms at
        100k, on the event loop and holding the metrics lock, so it degraded
        with proxy uptime rather than with load.

        This loop is over models, not records, so it is bounded by how many
        models a deployment talks to.
        """
        total_input_tokens = 0
        cost_with_headroom = 0.0
        for model in self._tokens_saved_by_model:
            sent = self._tokens_sent_by_model.get(model, 0)
            cr = self._api_cache_read_by_model.get(model, 0)
            cw = self._api_cache_write_by_model.get(model, 0)
            uncached = self._api_uncached_by_model.get(model, 0)
            total_input_tokens += sent

            prices = self._get_cache_prices(model)
            if prices:
                cr_price, cw_price, uncached_price = prices
                if cr + cw + uncached > 0:
                    cost_with_headroom += cr * cr_price + cw * cw_price + uncached * uncached_price
                else:
                    cost_with_headroom += sent * uncached_price
        return total_input_tokens, round(cost_with_headroom, 4)

    def stats(self) -> dict:
        """Get token statistics per model."""
        per_model = {}
        total_saved = 0
        total_compression_saved = 0
        total_tool_saved = 0
        # A model may have tool savings and no compression savings at all (every
        # turn deferral-only), so iterate the union — keying off
        # ``_tokens_saved_by_model`` alone would drop such a model from the table
        # entirely rather than merely under-report it.
        for model in sorted(set(self._tokens_saved_by_model) | set(self._tool_saved_by_model)):
            compression_saved = self._tokens_saved_by_model.get(model, 0)
            tool_saved = self._tool_saved_by_model.get(model, 0)
            # What the "Tokens Saved" column means: everything Headroom kept off
            # the wire for this model. The two components stay addressable beside
            # it so a caller can show the split.
            saved = compression_saved + tool_saved
            sent = self._tokens_sent_by_model.get(model, 0)
            reqs = self._requests_by_model.get(model, 0)
            total_saved += saved
            total_compression_saved += compression_saved
            total_tool_saved += tool_saved
            per_model[model] = {
                "requests": reqs,
                "tokens_saved": saved,
                "compression_tokens_saved": compression_saved,
                "tool_tokens_saved": tool_saved,
                "tokens_sent": sent,
                "cache_write_5m_tokens": self._api_cache_write_5m_by_model.get(model, 0),
                "cache_write_1h_tokens": self._api_cache_write_1h_by_model.get(model, 0),
                # Deferred schemas were never in ``sent``, so ``saved + sent`` is
                # still the pre-Headroom volume with the wider numerator.
                "reduction_pct": round(saved / (saved + sent) * 100, 1)
                if (saved + sent) > 0
                else 0,
            }

        # Compute actual input cost using API-reported cache breakdown and
        # LiteLLM's per-category pricing (cache reads discounted, writes at
        # premium, uncached at list). Falls back to list price when cache
        # data is unavailable.
        cost_with_headroom = 0.0
        total_billed_input_tokens = 0
        total_input_tokens = 0
        for model in self._tokens_saved_by_model:
            saved = self._tokens_saved_by_model[model]
            sent = self._tokens_sent_by_model.get(model, 0)
            cr = self._api_cache_read_by_model.get(model, 0)
            cw = self._api_cache_write_by_model.get(model, 0)
            uncached = self._api_uncached_by_model.get(model, 0)
            total_input_tokens += sent

            prices = self._get_cache_prices(model)
            if prices:
                cr_price, cw_price, uncached_price = prices
                if cr + cw + uncached > 0:
                    # Use API's real cache breakdown with LiteLLM pricing
                    model_cost = cr * cr_price + cw * cw_price + uncached * uncached_price
                    billed_tokens = cr + cw + uncached
                else:
                    # No cache data from API — fall back to list price
                    model_cost = sent * uncached_price
                    billed_tokens = sent
                cost_with_headroom += model_cost
                total_billed_input_tokens += billed_tokens

        # Compression savings: price saved tokens at the model's list input price.
        # This is simple, monotonic, and transparent — each saved token is valued
        # at the published $/token rate for its model. Not affected by cache mix.
        # Budget enforcement reads this figure, so it stays as-is; the
        # cache-aware valuation below is reported alongside it.
        savings_usd = 0.0
        for model in self._tokens_saved_by_model:
            saved = self._tokens_saved_by_model[model]
            if saved <= 0:
                continue
            prices = self._get_cache_prices(model)
            if prices:
                _cr_price, _cw_price, uncached_price = prices
                savings_usd += saved * uncached_price

        # Completion spend. The input-only figure above is what a budget and the
        # per-model table want; a card that says "$X spent" has to include the
        # tokens the model emitted, or the spend it shows isn't the bill.
        output_cost_usd = 0.0
        for (model, long_context), out_tokens in self._output_tokens_by_tier.items():
            price = self._get_output_price(model, long_context=long_context)
            if price:
                output_cost_usd += out_tokens * price

        # Compression savings priced at the rate the removed tokens would have
        # been billed at: live-zone content, so the provider's cache-write rate
        # for the share that would have been cached on this turn and list for
        # the rest, at the context tier the request was billed in. Flat list
        # pricing (``savings_usd``) ignores both. Reported separately so budget
        # enforcement keeps its monotonic list-priced basis; the dashboard's
        # cost card prefers this one.
        cache_aware_savings_usd = 0.0
        for key in set(self._saved_write_by_tier) | set(self._saved_list_by_tier):
            model, long_context = key
            prices = self._get_cache_prices(model, long_context=long_context)
            if not prices:
                continue
            _cr_price, cw_price, uncached_price = prices
            cache_aware_savings_usd += (
                self._saved_write_by_tier.get(key, 0.0) * cw_price
                + self._saved_list_by_tier.get(key, 0.0) * uncached_price
            )

        # Tool-schema deferral, priced CACHE-AWARE rather than flat at list:
        # had the schemas shipped they would have been part of the same prefix
        # as the rest of the request — cold-written once at the provider's
        # cache-write rate, read at its cache-read rate on warm turns, and
        # re-written on TTL expiry. Each request's schema tokens were bucketed
        # at record time by that request's observed read/write/uncached mix, so
        # expiry cycles are captured from real traffic instead of modelled.
        # Rates come from _get_cache_prices (LiteLLM per-model catalog:
        # cache_read_input_token_cost / cache_creation_input_token_cost, list
        # rate when a provider publishes no cache pricing) — the same
        # provider-agnostic source the real cost math uses, never a hardcoded
        # multiplier. Reported as its OWN key rather than widening
        # ``savings_usd``: that figure feeds budget enforcement, and the
        # session summary widens the cost card from this key instead.
        tool_savings_usd = 0.0
        tool_models = (
            set(self._tool_saved_read_by_model)
            | set(self._tool_saved_write_by_model)
            | set(self._tool_saved_list_by_model)
        )
        for model in tool_models:
            prices = self._get_cache_prices(model)
            if not prices:
                continue
            cr_price, cw_price, uncached_price = prices
            tool_savings_usd += (
                self._tool_saved_read_by_model.get(model, 0.0) * cr_price
                + self._tool_saved_write_by_model.get(model, 0.0) * cw_price
                + self._tool_saved_list_by_model.get(model, 0.0) * uncached_price
            )

        return {
            # Sum of the per-model rows above, so the payload reconciles with
            # itself. ``savings_usd`` below is deliberately NOT widened: tool
            # deferral is already priced by SavingsTracker, and this tracker's
            # dollars feed budget enforcement — counting it in both places would
            # double-book the saving against a budget.
            "total_tokens_saved": total_saved,
            "total_compression_tokens_saved": total_compression_saved,
            "total_tool_tokens_saved": total_tool_saved,
            "total_input_tokens": total_input_tokens,
            "total_input_cost_usd": round(cost_with_headroom, 4),
            "cache_write_5m_tokens": sum(self._api_cache_write_5m_by_model.values()),
            "cache_write_1h_tokens": sum(self._api_cache_write_1h_by_model.values()),
            "per_model": per_model,
            # Input-only, unchanged: budgets, the per-model table and the
            # persistent savings tracker all read it as input spend.
            "cost_with_headroom_usd": round(cost_with_headroom, 4),
            "output_cost_usd": round(output_cost_usd, 4),
            # What the session actually cost, input + output.
            "total_cost_usd": round(cost_with_headroom + output_cost_usd, 4),
            "savings_usd": round(savings_usd, 4),
            "cache_aware_savings_usd": round(cache_aware_savings_usd, 4),
            "tool_savings_usd": round(tool_savings_usd, 4),
            # Budget config passthrough — surfaces in /stats["cost"] so
            # `headroom doctor` can report whether a budget is set.
            "budget_limit_usd": self.budget_limit_usd,
            "budget_period": self.budget_period,
            "budget_estimated_basis": self.estimated_basis_policy,
            # Period spend split by input-count provenance, so estimate-derived
            # spend stays separable from provider-reported spend (#2713).
            "budget_basis": self.period_cost_breakdown(),
        }
