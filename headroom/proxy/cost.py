"""Cost tracking and budget management for the Headroom proxy.

Contains the CostTracker class and cost-related helper functions
for prefix cache statistics, cost merging, and session summaries.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import importlib.util
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

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

# Provider-specific cache discount multipliers (what fraction of input price)
# Used to calculate dollar savings from prefix caching
_CACHE_ECONOMICS = {
    "anthropic": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Explicit breakpoints, 5-min TTL",
    },
    "openai": {
        "read_multiplier": 0.5,
        "write_multiplier": 1.0,
        "label": "Automatic, no TTL control",
    },
    "gemini": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.0,
        "label": "Explicit cachedContent, configurable TTL",
    },
    "bedrock": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Same as Anthropic (Bedrock)",
    },
}


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

        econ = _CACHE_ECONOMICS.get(provider, _CACHE_ECONOMICS["anthropic"])
        read_mult: float = econ["read_multiplier"]  # type: ignore[assignment]
        write_mult: float = econ["write_multiplier"]  # type: ignore[assignment]

        # Get the base input price per token for the most-used model on this provider
        input_price_per_token = None
        if cost_tracker:
            for model_name in cost_tracker._tokens_sent_by_model:
                # Match model to provider
                _openai_prefixes = ("gpt", "o1", "o3", "o4")
                is_match = (
                    (provider == "anthropic" and "claude" in model_name)
                    or (provider == "openai" and any(p in model_name for p in _openai_prefixes))
                    or (provider == "gemini" and "gemini" in model_name)
                    or (provider == "bedrock" and "claude" in model_name)
                )
                if is_match:
                    price_per_1m = cost_tracker._get_list_price(model_name)
                    if price_per_1m:
                        input_price_per_token = price_per_1m / 1_000_000
                        break

        # Calculate savings:
        # Cache reads save (1.0 - read_mult) per token vs uncached input price.
        # Cache write premium is NOT deducted — it's baseline cost that the
        # client (e.g. Claude Code) pays regardless of Headroom. We track it
        # for observability but don't penalise our savings number.
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
            # Write premium (observability only — not subtracted from savings)
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
            "net_savings_usd": round(savings_usd, 4),
            "label": str(econ["label"]),
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

    totals["net_savings_usd"] = round(totals["savings_usd"], 4)
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
    cli_tokens_avoided: int = 0,
) -> dict | None:
    """Merge compression, cache, and CLI savings into cost stats.

    Each savings layer is reported separately with its own scope:
    - savings_usd: compression savings at model list price (monotonic)
    - cache_savings_usd: prefix cache discount from provider (separate)
    - cli_tokens_avoided: tokens filtered by the selected CLI context tool
      (token count only, no $ estimate)

    The dollar metric (savings_usd) remains ONLY proxy compression savings
    priced at the model's published input rate. CLI filtering is folded into
    the dashboard's compression token total, but it has no reliable
    model-specific dollar estimate because those tokens never reached the
    proxy request.
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
        "cli_tokens_avoided": cli_tokens_avoided,
        "cli_filtering_tokens_avoided": cli_tokens_avoided,
        "cli_tokens_included_in_compression": True,
        "cli_filtering_tokens_included_in_compression": True,
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
    cli_tokens_avoided: int,
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
    }

    if proxy.logger:
        for entry in proxy.logger._logs:
            if entry.model and "count_tokens" in entry.model:
                uncompressed_reasons["passthrough"] += 1
                continue
            if entry.tokens_saved > 0:
                compressed_requests.append(
                    {
                        "savings_pct": round(entry.savings_percent, 1),
                        "tokens_saved": entry.tokens_saved,
                        "original": entry.input_tokens_original,
                        "optimized": entry.input_tokens_optimized,
                    }
                )
            elif entry.input_tokens_original > 0:
                # Categorize why it wasn't compressed
                transforms = entry.transforms_applied or []
                if not transforms:
                    # Pipeline returned unchanged — likely all frozen
                    uncompressed_reasons["prefix_frozen"] += 1
                elif all("excluded" in t or "protected" in t for t in transforms):
                    uncompressed_reasons["no_compressible_content"] += 1
                elif entry.input_tokens_original < 500:
                    uncompressed_reasons["too_small"] += 1
                else:
                    uncompressed_reasons["prefix_frozen"] += 1

    # Compute compression stats for requests that DID compress
    avg_compression = 0.0
    best_compression = 0.0
    best_detail = ""
    if compressed_requests:
        avg_compression = round(
            sum(r["savings_pct"] for r in compressed_requests) / len(compressed_requests),
            1,
        )
        best = max(compressed_requests, key=lambda r: r["savings_pct"])
        best_compression = best["savings_pct"]
        best_detail = f"{best['original']:,} → {best['optimized']:,} tokens"

    # Cost summary — dollar savings are proxy-compression only at model list
    # price. CLI filtering tokens are counted in token savings but have no
    # model-specific price because they never reached the proxy request.
    cost_stats = proxy.cost_tracker.stats() if proxy.cost_tracker else {}
    cost_with = cost_stats.get("cost_with_headroom_usd", 0.0)
    compression_savings = cost_stats.get("savings_usd", 0.0)
    cache_net = prefix_cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
    total_saved_usd = round(compression_savings, 2)
    cost_without = cost_with + compression_savings
    savings_pct_cost = round(total_saved_usd / cost_without * 100, 1) if cost_without > 0 else 0.0

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
            "cli_filtering_tokens_avoided": cli_tokens_avoided,
            "total_tokens_saved_with_cli_filtering": (
                metrics.tokens_saved_total + cli_tokens_avoided
            ),
            "total_tokens_before_with_cli_filtering": total_tokens_before,
            "rtk_tokens_avoided": cli_tokens_avoided,
            "total_tokens_saved_with_rtk": metrics.tokens_saved_total + cli_tokens_avoided,
            "total_tokens_before_with_rtk": total_tokens_before,
        },
        "uncompressed_requests": {k: v for k, v in uncompressed_reasons.items() if v > 0},
        "cost": {
            "without_headroom_usd": round(cost_without, 2),
            "with_headroom_usd": round(cost_with, 2),
            "total_saved_usd": total_saved_usd,
            "savings_pct": savings_pct_cost,
            "breakdown": {
                "cache_savings_usd": round(cache_net, 2),
                "compression_savings_usd": round(compression_savings, 2),
                "cli_filtering_savings_usd": None,
                "cli_filtering_savings_note": (
                    "CLI filtering tokens are included in token savings only; "
                    "dollar savings use proxy compression tokens at model list price."
                ),
                "rtk_savings_usd": None,
                "rtk_savings_note": (
                    "CLI filtering tokens are included in token savings only; dollar savings "
                    "use proxy compression tokens at model list price."
                ),
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

    def __init__(self, budget_limit_usd: float | None = None, budget_period: str = "daily"):
        self.budget_limit_usd = budget_limit_usd
        self.budget_period = budget_period

        # Cost tracking - using deque for efficient left-side removal
        self._costs: deque[tuple[datetime, float]] = deque(maxlen=self.MAX_COST_ENTRIES)
        self._last_prune_time: datetime = datetime.now()

        # Token savings per model (exact, no dollar estimation)
        self._tokens_saved_by_model: dict[str, int] = {}
        self._tokens_sent_by_model: dict[str, int] = {}
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
        self._tokens_sent_by_model.clear()
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
            logger.warning("LiteLLM not available - cannot calculate costs")
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
            logger.warning(f"Failed to get pricing for model {model}: {e}")
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
        while self._costs and self._costs[0][0] < cutoff:
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
        """
        self._tokens_saved_by_model[model] = (
            self._tokens_saved_by_model.get(model, 0) + tokens_saved
        )
        self._tokens_sent_by_model[model] = self._tokens_sent_by_model.get(model, 0) + tokens_sent
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
        input_tokens = uncached_tokens
        if not (uncached_tokens or cache_read_tokens or cache_write_tokens):
            input_tokens = tokens_sent
        cost = self.estimate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        if cost is not None:
            self._costs.append((datetime.now(), cost))
            self._prune_old_costs()

    def get_period_cost(self) -> float:
        """Get cost for current budget period."""
        now = datetime.now()

        if self.budget_period == "hourly":
            cutoff = now - timedelta(hours=1)
        elif self.budget_period == "daily":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # monthly
            cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return sum(cost for ts, cost in self._costs if ts >= cutoff)

    def check_budget(self) -> tuple[bool, float]:
        """Check if within budget. Returns (allowed, remaining)."""
        if self.budget_limit_usd is None:
            return True, float("inf")

        period_cost = self.get_period_cost()
        remaining = self.budget_limit_usd - period_cost
        return remaining > 0, max(0, remaining)

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

    def _get_cache_prices(self, model: str) -> tuple[float, float, float] | None:
        """Get per-token prices for cache read, cache write, and uncached input.

        Returns (cache_read, cache_write, uncached) per-token costs, or None
        if pricing is unavailable. Uses LiteLLM's native cache pricing data.
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
            return (cache_read, cache_write, uncached)
        except Exception:
            return None

    def stats(self) -> dict:
        """Get token statistics per model."""
        per_model = {}
        total_saved = 0
        for model in sorted(self._tokens_saved_by_model.keys()):
            saved = self._tokens_saved_by_model[model]
            sent = self._tokens_sent_by_model.get(model, 0)
            reqs = self._requests_by_model.get(model, 0)
            total_saved += saved
            per_model[model] = {
                "requests": reqs,
                "tokens_saved": saved,
                "tokens_sent": sent,
                "cache_write_5m_tokens": self._api_cache_write_5m_by_model.get(model, 0),
                "cache_write_1h_tokens": self._api_cache_write_1h_by_model.get(model, 0),
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
        savings_usd = 0.0
        for model in self._tokens_saved_by_model:
            saved = self._tokens_saved_by_model[model]
            if saved <= 0:
                continue
            prices = self._get_cache_prices(model)
            if prices:
                _cr_price, _cw_price, uncached_price = prices
                savings_usd += saved * uncached_price

        return {
            "total_tokens_saved": total_saved,
            "total_input_tokens": total_input_tokens,
            "total_input_cost_usd": round(cost_with_headroom, 4),
            "cache_write_5m_tokens": sum(self._api_cache_write_5m_by_model.values()),
            "cache_write_1h_tokens": sum(self._api_cache_write_1h_by_model.values()),
            "per_model": per_model,
            "cost_with_headroom_usd": round(cost_with_headroom, 4),
            "savings_usd": round(savings_usd, 4),
            # Budget config passthrough — surfaces in /stats["cost"] so
            # `headroom doctor` can report whether a budget is set.
            "budget_limit_usd": self.budget_limit_usd,
            "budget_period": self.budget_period,
        }
