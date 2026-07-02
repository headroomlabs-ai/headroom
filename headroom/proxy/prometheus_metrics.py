"""Prometheus-compatible metrics for the Headroom proxy.

Tracks request counts, token usage, latency, overhead, TTFB,
per-transform timing, waste signals, prefix cache stats, and
cumulative savings history.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headroom.observability import HeadroomOtelMetrics
    from headroom.proxy.cost import CostTracker

from headroom import savings_ledger
from headroom.observability import get_otel_metrics
from headroom.proxy.savings_tracker import SavingsTracker

logger = logging.getLogger("headroom.proxy")


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: dict[str, str] | None = None) -> str:
    if not labels:
        return ""

    rendered = ",".join(
        f'{key}="{_escape_label_value(str(value))}"' for key, value in sorted(labels.items())
    )
    return f"{{{rendered}}}"


def _append_metric(
    lines: list[str],
    *,
    name: str,
    metric_type: str,
    help_text: str,
    value: int | float,
    labels: dict[str, str] | None = None,
) -> None:
    lines.extend(
        [
            f"# HELP {name} {help_text}",
            f"# TYPE {name} {metric_type}",
            f"{name}{_format_labels(labels)} {value}",
            "",
        ]
    )


class PrometheusMetrics:
    """Prometheus-compatible metrics."""

    def __init__(
        self,
        savings_tracker: SavingsTracker | None = None,
        cost_tracker: CostTracker | None = None,
        otel_metrics: HeadroomOtelMetrics | None = None,
        stateless: bool = False,
    ):
        # Stateless mode: keep live in-memory metrics but never write the
        # durable savings files (proxy_savings.json, savings_events.jsonl).
        self._stateless = stateless
        self.requests_total = 0
        self.requests_by_provider: dict[str, int] = defaultdict(int)
        self.requests_by_model: dict[str, int] = defaultdict(int)
        # Populated via X-Headroom-Stack header (TS SDK adapters, etc.)
        self.requests_by_stack: dict[str, int] = defaultdict(int)
        self.requests_cached = 0
        self.requests_rate_limited = 0
        self.requests_failed = 0
        self.inbound_requests_total = 0
        self.inbound_requests_completed = 0
        self.inbound_requests_active = 0
        self.inbound_requests_by_method: dict[str, int] = defaultdict(int)
        self.inbound_requests_by_path: dict[str, int] = defaultdict(int)
        self.inbound_responses_by_status: dict[str, int] = defaultdict(int)

        self.tokens_input_total = 0
        self.tokens_output_total = 0
        self.tokens_saved_total = 0
        # Sum of tokens we actually attempted to compress across the
        # session: extracted units that passed all gates + tool-schema
        # tokens we ran compaction against. Excludes prefix-frozen
        # content (instructions, user/system messages, prior turns).
        # This is the right denominator for an "active compression
        # ratio" — what fraction of the compressible-eligible tokens
        # did we actually save?
        self.attempted_input_tokens_total = 0

        # Per-strategy compression counters. Populated lazily as we see
        # each strategy tag — no hardcoded list of strategies; the keys
        # come from ContentRouter's `CompressionStrategy.value` and
        # SmartCrusher's literal `"smart_crusher"`. The forcing
        # function for catching strategy-level silent regressions:
        # if SmartCrusher events drop to zero in production, the
        # `headroom_compressions_total{strategy="smart_crusher"}`
        # counter shows it on day 1, not week 3.
        self.compressions_by_strategy: dict[str, int] = defaultdict(int)
        self.tokens_saved_by_strategy: dict[str, int] = defaultdict(int)

        # Codex WebSocket compression observability. These are intentionally
        # aggregate counters/sums, not per-unit storage, so /stats can answer
        # routing questions without growing with traffic volume.
        self.codex_ws_units_total = 0
        self.codex_ws_units_modified_total = 0
        self.codex_ws_units_to_kompress_total = 0
        self.codex_ws_units_kompress_attempted_total = 0
        self.codex_ws_units_by_strategy: dict[str, int] = defaultdict(int)
        self.codex_ws_units_by_category: dict[str, int] = defaultdict(int)
        self.codex_ws_units_by_content_type: dict[str, int] = defaultdict(int)
        self.codex_ws_units_by_text_shape: dict[str, int] = defaultdict(int)
        self.codex_ws_unit_elapsed_ms_sum = 0.0
        self.codex_ws_unit_elapsed_ms_max = 0.0
        self.codex_ws_unit_bytes_sum = 0
        self.codex_ws_unit_tokens_before_sum = 0
        self.codex_ws_unit_tokens_after_sum = 0
        self.codex_ws_unit_tokens_saved_sum = 0

        self.codex_ws_frames_attempted_total = 0
        self.codex_ws_frames_compressed_total = 0
        self.codex_ws_frames_failed_total = 0
        self.codex_ws_frames_to_kompress_total = 0
        self.codex_ws_frames_kompress_attempted_total = 0
        self.codex_ws_frame_elapsed_ms_sum = 0.0
        self.codex_ws_frame_elapsed_ms_max = 0.0
        self.codex_ws_frame_bytes_before_sum = 0
        self.codex_ws_frame_bytes_after_sum = 0
        self.codex_ws_frame_attempted_tokens_sum = 0
        self.codex_ws_frame_tokens_saved_sum = 0

        self.latency_sum_ms = 0.0
        self.latency_min_ms = float("inf")
        self.latency_max_ms = 0.0
        self.latency_count = 0

        # Headroom overhead (optimization time only, excludes LLM)
        self.overhead_sum_ms = 0.0
        self.overhead_min_ms = float("inf")
        self.overhead_max_ms = 0.0
        self.overhead_count = 0

        # Time to first byte (TTFB) from upstream — what the user actually feels
        self.ttfb_sum_ms = 0.0
        self.ttfb_min_ms = float("inf")
        self.ttfb_max_ms = 0.0
        self.ttfb_count = 0

        # Per-transform timing (name → cumulative ms, count)
        self.transform_timing_sum: dict[str, float] = defaultdict(float)
        self.transform_timing_count: dict[str, int] = defaultdict(int)
        self.transform_timing_max: dict[str, float] = defaultdict(float)

        # Per-stage timing (Unit 2). Keyed by ``(path, stage)`` tuples so
        # a single metric name can distinguish between, e.g.,
        # ``openai_responses_ws`` ``upstream_connect`` and
        # ``anthropic_messages`` ``upstream_connect``.
        self.stage_timing_sum: dict[tuple[str, str], float] = defaultdict(float)
        self.stage_timing_count: dict[tuple[str, str], int] = defaultdict(int)
        self.stage_timing_max: dict[tuple[str, str], float] = defaultdict(float)

        # WS session lifecycle (Unit 3). Gauges are live counters updated
        # by the Codex handler on register/deregister + attach_tasks/
        # detach. Histograms record completed-session durations bucketed
        # by termination cause so we can distinguish slow happy-path
        # sessions from long client-hold followed by client_disconnect.
        self.active_ws_sessions: int = 0
        self.active_relay_tasks: int = 0
        self.ws_session_duration_sum_ms: dict[str, float] = defaultdict(float)
        self.ws_session_duration_count: dict[str, int] = defaultdict(int)
        self.ws_session_duration_max_ms: dict[str, float] = defaultdict(float)

        # Aggregate waste signals
        self.waste_signals_total: dict[str, int] = defaultdict(int)

        # Cumulative ContentRouter protection counts. Each routing pass
        # categorises every message — `user_msg`, `system_msg`,
        # `recent_code`, `excluded_tool`, `analysis_ctx`, `small`,
        # `ratio_too_high`, `already_compressed`, `non_string`,
        # `content_blocks`. Surfacing these in `/stats` gives operators a
        # way to diagnose "why is my compression rate low?" — e.g. a high
        # `user_msg` count on OpenAI/Azure traffic explains why most
        # input was protected and never reached the compressor (#454).
        self.router_route_counts: dict[str, int] = defaultdict(int)

        # Provider-specific prefix cache tracking
        # Each provider has different cache economics:
        #   Anthropic: cache_read=0.1x, cache_write=1.25x, explicit breakpoints
        #   OpenAI: cache_read=0.5x, no write penalty, automatic
        #   Google: cache_read=~0.1x, explicit cachedContent API, storage cost
        #   Bedrock: no cache metrics
        self.cache_by_provider: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cache_write_5m_tokens": 0,
                "cache_write_1h_tokens": 0,
                "cache_write_5m_requests": 0,
                "cache_write_1h_requests": 0,
                "uncached_input_tokens": 0,
                "requests": 0,
                "hit_requests": 0,  # requests with cache_read > 0
                "bust_count": 0,
                "bust_write_tokens": 0,
            }
        )
        # Track per-model cache request count to distinguish cold starts from busts
        self._cache_requests_by_model: dict[str, int] = defaultdict(int)

        # Prefix freeze stats (cache-aware compression)
        self.prefix_freeze_busts_avoided: int = 0
        self.prefix_freeze_tokens_preserved: int = 0
        self.prefix_freeze_compression_foregone: int = 0

        # Cache bust tracking: how many tokens lost their cache discount due to compression
        self.cache_bust_tokens_lost: int = 0
        self.cache_bust_count: int = 0

        # Cache-miss attribution (#1313): when a turn expected a prompt-cache
        # hit but got none, why? Bucketed by reason so operators can tell a
        # TTL lapse (idle longer than the cache lifetime → consider a longer
        # TTL) from a prefix-content change (the cacheable prefix shifted).
        # Reasons are the MISS_* literals from prefix_tracker. Per provider so
        # the dashboard can scope; populated by `record_cache_miss_attribution`.
        self.cache_miss_attribution_by_provider: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Cumulative savings history (timestamp → cumulative tokens saved)
        self.savings_history: list[tuple[str, int]] = []
        self.savings_tracker = savings_tracker or SavingsTracker(stateless=stateless)
        self.cost_tracker = cost_tracker
        tracker_lifetime = self.savings_tracker.snapshot()["lifetime"]
        self._savings_tracker_input_tokens_offset = max(
            int(tracker_lifetime.get("total_input_tokens", 0) or 0),
            0,
        )
        self._savings_tracker_input_cost_usd_offset = max(
            float(tracker_lifetime.get("total_input_cost_usd", 0.0) or 0.0),
            0.0,
        )

        self._lock = asyncio.Lock()
        # Tiny synchronous critical section for stage-timing triple updates
        # (sum + count + max must move together for a consistent scrape).
        # threading.Lock is cheaper than asyncio.Lock and does NOT contend
        # with the async ``export()`` path — scrapes snapshot these dicts
        # under this lock in a microsecond block, then build the metrics
        # string without holding anything.
        self._stage_timing_lock = threading.Lock()
        self._otel_metrics = otel_metrics

    async def reset_runtime(self) -> None:
        """Reset in-memory request/compression counters for local test/debug use."""
        async with self._lock:
            self.requests_total = 0
            self.requests_by_provider.clear()
            self.requests_by_model.clear()
            self.requests_by_stack.clear()
            self.requests_cached = 0
            self.requests_rate_limited = 0
            self.requests_failed = 0
            self.inbound_requests_total = 0
            self.inbound_requests_completed = 0
            self.inbound_requests_active = 0
            self.inbound_requests_by_method.clear()
            self.inbound_requests_by_path.clear()
            self.inbound_responses_by_status.clear()

            self.tokens_input_total = 0
            self.tokens_output_total = 0
            self.tokens_saved_total = 0
            self.attempted_input_tokens_total = 0

            self.compressions_by_strategy.clear()
            self.tokens_saved_by_strategy.clear()

            self.codex_ws_units_total = 0
            self.codex_ws_units_modified_total = 0
            self.codex_ws_units_to_kompress_total = 0
            self.codex_ws_units_kompress_attempted_total = 0
            self.codex_ws_units_by_strategy.clear()
            self.codex_ws_units_by_category.clear()
            self.codex_ws_units_by_content_type.clear()
            self.codex_ws_units_by_text_shape.clear()
            self.codex_ws_unit_elapsed_ms_sum = 0.0
            self.codex_ws_unit_elapsed_ms_max = 0.0
            self.codex_ws_unit_bytes_sum = 0
            self.codex_ws_unit_tokens_before_sum = 0
            self.codex_ws_unit_tokens_after_sum = 0
            self.codex_ws_unit_tokens_saved_sum = 0

            self.codex_ws_frames_attempted_total = 0
            self.codex_ws_frames_compressed_total = 0
            self.codex_ws_frames_failed_total = 0
            self.codex_ws_frames_to_kompress_total = 0
            self.codex_ws_frames_kompress_attempted_total = 0
            self.codex_ws_frame_elapsed_ms_sum = 0.0
            self.codex_ws_frame_elapsed_ms_max = 0.0
            self.codex_ws_frame_bytes_before_sum = 0
            self.codex_ws_frame_bytes_after_sum = 0
            self.codex_ws_frame_attempted_tokens_sum = 0
            self.codex_ws_frame_tokens_saved_sum = 0

            self.latency_sum_ms = 0.0
            self.latency_min_ms = float("inf")
            self.latency_max_ms = 0.0
            self.latency_count = 0

            self.overhead_sum_ms = 0.0
            self.overhead_min_ms = float("inf")
            self.overhead_max_ms = 0.0
            self.overhead_count = 0

            self.ttfb_sum_ms = 0.0
            self.ttfb_min_ms = float("inf")
            self.ttfb_max_ms = 0.0
            self.ttfb_count = 0

            self.transform_timing_sum.clear()
            self.transform_timing_count.clear()
            self.transform_timing_max.clear()

            self.waste_signals_total.clear()
            self.cache_by_provider.clear()
            self._cache_requests_by_model.clear()

            self.prefix_freeze_busts_avoided = 0
            self.prefix_freeze_tokens_preserved = 0
            self.prefix_freeze_compression_foregone = 0
            self.cache_bust_tokens_lost = 0
            self.cache_bust_count = 0
            self.cache_miss_attribution_by_provider.clear()
            self.savings_history = []

        with self._stage_timing_lock:
            self.stage_timing_sum.clear()
            self.stage_timing_count.clear()
            self.stage_timing_max.clear()
            self.ws_session_duration_sum_ms.clear()
            self.ws_session_duration_count.clear()
            self.ws_session_duration_max_ms.clear()

    def _get_otel_metrics(self) -> HeadroomOtelMetrics:
        return self._otel_metrics or get_otel_metrics()

    def _current_savings_tracker_totals(self) -> tuple[int, float]:
        total_input_tokens = self._savings_tracker_input_tokens_offset + self.tokens_input_total
        total_input_cost_usd = self._savings_tracker_input_cost_usd_offset

        if self.cost_tracker is None:
            return total_input_tokens, total_input_cost_usd

        try:
            cost_stats = self.cost_tracker.stats()
        except Exception:
            logger.debug("Failed to read cost tracker totals for savings history", exc_info=True)
            return total_input_tokens, total_input_cost_usd

        tracked_input_tokens = cost_stats.get("total_input_tokens")
        tracked_input_cost_usd = cost_stats.get("total_input_cost_usd")

        if tracked_input_tokens is not None:
            try:
                total_input_tokens = self._savings_tracker_input_tokens_offset + max(
                    int(tracked_input_tokens),
                    0,
                )
            except (TypeError, ValueError):
                pass

        if tracked_input_cost_usd is not None:
            try:
                total_input_cost_usd = self._savings_tracker_input_cost_usd_offset + max(
                    float(tracked_input_cost_usd),
                    0.0,
                )
            except (TypeError, ValueError):
                pass

        return total_input_tokens, total_input_cost_usd

    def record_stack(self, stack: str | None) -> None:
        """Increment the per-stack request counter.

        ``stack`` is the ``X-Headroom-Stack`` header value (e.g.
        ``adapter_ts_openai``). Called once per inbound request from the
        proxy's stack middleware; a no-op when the header is absent, fails
        validation, or would exceed the cardinality cap.
        """

        from headroom.telemetry.context import MAX_DISTINCT_STACKS, normalize_stack

        slug = normalize_stack(stack)
        if not slug:
            return
        if (
            slug not in self.requests_by_stack
            and len(self.requests_by_stack) >= MAX_DISTINCT_STACKS
        ):
            return
        self.requests_by_stack[slug] += 1

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
    ) -> None:
        """Implements `headroom.transforms.observability.CompressionObserver`.

        Called once per real compression event by the configured
        transforms (ContentRouter at routing-decision granularity;
        SmartCrusher at message granularity in the legacy direct-
        pipeline path). Increments the per-strategy counters that
        get exported as labelled Prometheus metrics, so silent
        regressions in any single strategy become visible in the
        scrape.

        Synchronous + lock-free: `defaultdict(int)` writes are
        atomic under the GIL for these key types; the proxy serves
        many requests concurrently and the contention here would be
        a single dict write per routing decision.

        Tokens saved is `max(0, original - compressed)` — the
        observer never records "negative savings" even if a
        compressor goofs and emits more tokens than it received.
        """
        self.compressions_by_strategy[strategy] += 1
        saved = original_tokens - compressed_tokens
        if saved > 0:
            self.tokens_saved_by_strategy[strategy] += saved

    def record_router_route_counts(self, counts: dict[str, int]) -> None:
        """Accumulate ContentRouter routing-category counts for a single
        pass. The router emits a dict like ``{"user_msg": 12,
        "recent_code": 4, ...}`` summarising how it categorised each
        message in that request. Adding these into a long-running
        counter gives `/stats` a session-level breakdown so operators
        can see, e.g., that 80% of messages were protected as
        `user_msg` and only 5% reached the compressor (#454).
        """
        for category, count in counts.items():
            if count > 0:
                self.router_route_counts[category] += int(count)

    def record_codex_ws_unit(
        self,
        *,
        strategy: str,
        reason_category: str,
        elapsed_ms: float,
        text_bytes: int,
        tokens_before: int,
        tokens_after: int,
        tokens_saved: int,
        modified: bool,
        strategy_chain: list[str] | None = None,
        content_type: str = "unknown",
        text_shape: str = "unknown",
    ) -> None:
        """Record one Codex WS compression unit decision."""

        strategy = strategy or "unknown"
        reason_category = reason_category or "unknown"
        chain = strategy_chain or []

        self.codex_ws_units_total += 1
        self.codex_ws_units_by_strategy[strategy] += 1
        self.codex_ws_units_by_category[reason_category] += 1
        self.codex_ws_units_by_content_type[content_type or "unknown"] += 1
        self.codex_ws_units_by_text_shape[text_shape or "unknown"] += 1
        if modified:
            self.codex_ws_units_modified_total += 1
        if strategy == "kompress":
            self.codex_ws_units_to_kompress_total += 1
        if "kompress" in chain or strategy == "kompress":
            self.codex_ws_units_kompress_attempted_total += 1

        elapsed_ms = max(0.0, float(elapsed_ms))
        self.codex_ws_unit_elapsed_ms_sum += elapsed_ms
        self.codex_ws_unit_elapsed_ms_max = max(self.codex_ws_unit_elapsed_ms_max, elapsed_ms)
        self.codex_ws_unit_bytes_sum += max(0, int(text_bytes))
        self.codex_ws_unit_tokens_before_sum += max(0, int(tokens_before))
        self.codex_ws_unit_tokens_after_sum += max(0, int(tokens_after))
        self.codex_ws_unit_tokens_saved_sum += max(0, int(tokens_saved))

    def record_codex_ws_frame(
        self,
        *,
        elapsed_ms: float,
        bytes_before: int,
        bytes_after: int = 0,
        attempted_tokens: int = 0,
        tokens_saved: int = 0,
        modified: bool = False,
        failed: bool = False,
        strategy_chain: list[str] | None = None,
        final_strategies: list[str] | None = None,
    ) -> None:
        """Record one Codex WS response.create compression attempt."""

        chain = strategy_chain or []
        strategies = final_strategies or []

        self.codex_ws_frames_attempted_total += 1
        if modified:
            self.codex_ws_frames_compressed_total += 1
        if failed:
            self.codex_ws_frames_failed_total += 1
        if "kompress" in strategies:
            self.codex_ws_frames_to_kompress_total += 1
        if "kompress" in chain or "kompress" in strategies:
            self.codex_ws_frames_kompress_attempted_total += 1

        elapsed_ms = max(0.0, float(elapsed_ms))
        self.codex_ws_frame_elapsed_ms_sum += elapsed_ms
        self.codex_ws_frame_elapsed_ms_max = max(self.codex_ws_frame_elapsed_ms_max, elapsed_ms)
        self.codex_ws_frame_bytes_before_sum += max(0, int(bytes_before))
        self.codex_ws_frame_bytes_after_sum += max(0, int(bytes_after))
        self.codex_ws_frame_attempted_tokens_sum += max(0, int(attempted_tokens))
        self.codex_ws_frame_tokens_saved_sum += max(0, int(tokens_saved))

    def record_inbound_request(self, *, method: str, path: str) -> None:
        self.inbound_requests_total += 1
        self.inbound_requests_active += 1
        self.inbound_requests_by_method[method.upper()] += 1
        self.inbound_requests_by_path[path] += 1

    def record_inbound_response(self, *, status_code: int | str) -> None:
        self.inbound_requests_completed += 1
        self.inbound_requests_active = max(0, self.inbound_requests_active - 1)
        self.inbound_responses_by_status[str(status_code)] += 1

    def record_inbound_aborted(self, *, reason: str) -> None:
        self.inbound_requests_completed += 1
        self.inbound_requests_active = max(0, self.inbound_requests_active - 1)
        self.inbound_responses_by_status[f"aborted:{reason}"] += 1

    def inbound_snapshot(self) -> dict[str, object]:
        return {
            "total": self.inbound_requests_total,
            "completed": self.inbound_requests_completed,
            "active": self.inbound_requests_active,
            "by_method": dict(self.inbound_requests_by_method),
            "by_path": dict(self.inbound_requests_by_path),
            "by_status": dict(self.inbound_responses_by_status),
        }

    async def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tokens_saved: int,
        latency_ms: float,
        cached: bool = False,
        overhead_ms: float = 0,
        ttfb_ms: float = 0,
        pipeline_timing: dict[str, float] | None = None,
        waste_signals: dict[str, int] | None = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
        uncached_input_tokens: int = 0,
        attempted_input_tokens: int = 0,
        project: str | None = None,
        client: str | None = None,
    ):
        """Record metrics for a request."""
        async with self._lock:
            self.requests_total += 1
            self.requests_by_provider[provider] += 1
            self.requests_by_model[model] += 1

            if cached:
                self.requests_cached += 1

            self.tokens_input_total += input_tokens
            self.tokens_output_total += output_tokens
            self.tokens_saved_total += tokens_saved
            # See the attribute definition for why this is the right
            # denominator for the active-compression ratio.
            self.attempted_input_tokens_total += max(0, int(attempted_input_tokens))

            # Track provider-specific prefix cache metrics
            if cache_read_tokens > 0 or cache_write_tokens > 0:
                pc = self.cache_by_provider[provider]
                pc["cache_read_tokens"] += cache_read_tokens
                pc["cache_write_tokens"] += cache_write_tokens
                pc["cache_write_5m_tokens"] += cache_write_5m_tokens
                pc["cache_write_1h_tokens"] += cache_write_1h_tokens
                if cache_write_5m_tokens > 0:
                    pc["cache_write_5m_requests"] += 1
                if cache_write_1h_tokens > 0:
                    pc["cache_write_1h_requests"] += 1
                pc["uncached_input_tokens"] += uncached_input_tokens
                pc["requests"] += 1
                if cache_read_tokens > 0:
                    pc["hit_requests"] += 1
                # Model-aware bust detection: the first request for any model
                # is always a cold start (100% write, 0% read) — not a bust.
                # Only flag as bust when a previously-warm model suddenly has
                # high write ratio, indicating prefix invalidation.
                model_req_num = self._cache_requests_by_model[model]
                self._cache_requests_by_model[model] += 1
                if provider == "anthropic" and model_req_num > 0:
                    total_cached = cache_read_tokens + cache_write_tokens
                    if total_cached > 0 and cache_write_tokens > total_cached * 0.5:
                        pc["bust_count"] += 1
                        pc["bust_write_tokens"] += cache_write_tokens

            self.latency_sum_ms += latency_ms
            self.latency_min_ms = min(self.latency_min_ms, latency_ms)
            self.latency_max_ms = max(self.latency_max_ms, latency_ms)
            self.latency_count += 1

            # Track Headroom overhead separately
            if overhead_ms > 0:
                self.overhead_sum_ms += overhead_ms
                self.overhead_min_ms = min(self.overhead_min_ms, overhead_ms)
                self.overhead_max_ms = max(self.overhead_max_ms, overhead_ms)
                self.overhead_count += 1

            # Track TTFB (time to first byte from upstream)
            if ttfb_ms > 0:
                self.ttfb_sum_ms += ttfb_ms
                self.ttfb_min_ms = min(self.ttfb_min_ms, ttfb_ms)
                self.ttfb_max_ms = max(self.ttfb_max_ms, ttfb_ms)
                self.ttfb_count += 1

            # Track per-transform timing
            if pipeline_timing:
                for name, ms in pipeline_timing.items():
                    self.transform_timing_sum[name] += ms
                    self.transform_timing_count[name] += 1
                    self.transform_timing_max[name] = max(self.transform_timing_max[name], ms)

            # Track waste signals
            if waste_signals:
                for signal_name, token_count in waste_signals.items():
                    self.waste_signals_total[signal_name] += token_count

            # Track cumulative savings history (record every request)
            self.savings_history.append((datetime.now().isoformat(), self.tokens_saved_total))
            # Keep last 500 data points
            if len(self.savings_history) > 500:
                self.savings_history = self.savings_history[-500:]

            total_input_tokens, total_input_cost_usd = self._current_savings_tracker_totals()
            self.savings_tracker.record_request(
                model=model,
                input_tokens=input_tokens,
                tokens_saved=tokens_saved,
                provider=provider,
                project=project,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                uncached_input_tokens=uncached_input_tokens,
                total_input_tokens=total_input_tokens,
                total_input_cost_usd=total_input_cost_usd,
            )

            # Also append to the durable, multi-process savings ledger so
            # `headroom savings` reflects proxy traffic alongside MCP-tool usage.
            # The real upstream model means litellm prices it accurately. The
            # client is the harness classified from the User-Agent / X-Client
            # (claude-code, codex, cursor, ...); it falls back to "proxy" only
            # when the harness is unidentified.
            if tokens_saved > 0 and not self._stateless:
                savings_ledger.record_savings_event(
                    tokens_before=input_tokens,
                    tokens_after=max(input_tokens - tokens_saved, 0),
                    model=model,
                    client=client or "proxy",
                    source="proxy",
                )

        self._get_otel_metrics().record_proxy_request(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            latency_ms=latency_ms,
            cached=cached,
            overhead_ms=overhead_ms,
            ttfb_ms=ttfb_ms,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_5m_tokens=cache_write_5m_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
            uncached_input_tokens=uncached_input_tokens,
        )

    async def record_stage_timings(
        self,
        path: str,
        timings: dict[str, float],
    ) -> None:
        """Record per-stage timings as histogram-style observations.

        ``path`` identifies the code path that emitted the timings (e.g.
        ``openai_responses_ws`` or ``anthropic_messages``). ``timings``
        maps stage names to millisecond durations. Mirrors the
        ``transform_timing_*`` aggregation pattern so the ``/metrics``
        endpoint exposes sum/count/max series per ``(path, stage)``.

        Uses a tiny synchronous ``threading.Lock`` around the triple
        update (sum + count + max) rather than the async
        ``self._lock``: (1) the updates have no awaits, so there is no
        async contention benefit, and (2) the async lock is also held
        by ``export()`` during Prometheus scrapes — which does
        string-building while holding it. Under N concurrent request
        finalizations + an active scrape, callers would queue behind
        the scrape's string-building.
        """
        if not timings:
            return
        with self._stage_timing_lock:
            for stage, ms in timings.items():
                try:
                    ms_val = float(ms)
                except (TypeError, ValueError):
                    continue
                key = (path, stage)
                self.stage_timing_sum[key] += ms_val
                self.stage_timing_count[key] += 1
                if ms_val > self.stage_timing_max[key]:
                    self.stage_timing_max[key] = ms_val

    async def record_cache_bust(self, tokens_lost: int) -> None:
        """Record tokens that lost their cache discount due to compression."""
        async with self._lock:
            self.cache_bust_tokens_lost += tokens_lost
            self.cache_bust_count += 1
        self._get_otel_metrics().record_proxy_cache_bust(tokens_lost=tokens_lost)

    async def record_cache_miss_attribution(self, provider: str, reason: str) -> None:
        """Record why a turn that expected a prompt-cache hit missed instead.

        ``reason`` is one of the MISS_* literals produced by
        ``PrefixCacheTracker.classify_cache_miss`` (``ttl_expiry``,
        ``prefix_change``, ``unknown``). Cold starts and hits are not recorded
        — only actual misses against a previously-cached prefix reach here.
        Bucketed per provider so the dashboard can scope or aggregate.
        """
        async with self._lock:
            self.cache_miss_attribution_by_provider[provider][reason] += 1

    # ------------------------------------------------------------------
    # Unit 3: WS session lifecycle gauges / histogram
    # ------------------------------------------------------------------

    def inc_active_ws_sessions(self) -> None:
        """Increment the live WS session gauge (called on register)."""
        self.active_ws_sessions += 1

    def dec_active_ws_sessions(self) -> None:
        """Decrement the live WS session gauge (called on deregister)."""
        self.active_ws_sessions = max(0, self.active_ws_sessions - 1)

    def inc_active_relay_tasks(self, n: int = 1) -> None:
        """Increment the live relay-task gauge (attach_tasks)."""
        self.active_relay_tasks += n

    def dec_active_relay_tasks(self, n: int = 1) -> None:
        """Decrement the live relay-task gauge (deregister)."""
        self.active_relay_tasks = max(0, self.active_relay_tasks - n)

    def record_ws_session_duration(
        self,
        duration_ms: float,
        cause: str = "unknown",
    ) -> None:
        """Record a completed WS session's duration, bucketed by cause.

        Mirrors the ``stage_timing_*`` shape so ``/metrics`` exposes
        sum/count/max per termination cause. Uses synchronous dict
        updates (no ``_lock``) because Unit 3 callers run on the event
        loop — matching the gauges above.
        """
        try:
            ms_val = float(duration_ms)
        except (TypeError, ValueError):
            return
        self.ws_session_duration_sum_ms[cause] += ms_val
        self.ws_session_duration_count[cause] += 1
        if ms_val > self.ws_session_duration_max_ms[cause]:
            self.ws_session_duration_max_ms[cause] = ms_val

    async def record_rate_limited(self, *, provider: str | None = None, model: str | None = None):
        async with self._lock:
            self.requests_rate_limited += 1
        self._get_otel_metrics().record_proxy_rate_limited(provider=provider, model=model)

    async def record_failed(self, *, provider: str | None = None, model: str | None = None):
        async with self._lock:
            self.requests_failed += 1
        self._get_otel_metrics().record_proxy_failed(provider=provider, model=model)

    async def export(self) -> str:
        """Export metrics in Prometheus format."""
        # Snapshot stage-timing dicts under the tiny synchronous lock so
        # we don't race a concurrent ``record_stage_timings`` and observe
        # an inconsistent (sum, count, max) triple. Freeze into plain
        # dicts so the scrape's string-building below doesn't hold the
        # stage-timing lock during I/O-ish work.
        with self._stage_timing_lock:
            stage_timing_sum_snapshot = dict(self.stage_timing_sum)
            stage_timing_count_snapshot = dict(self.stage_timing_count)
            stage_timing_max_snapshot = dict(self.stage_timing_max)
        async with self._lock:
            lifetime_savings = self.savings_tracker.snapshot()["lifetime"]
            lines: list[str] = []
            _append_metric(
                lines,
                name="headroom_requests_total",
                metric_type="counter",
                help_text="Total number of requests",
                value=self.requests_total,
            )
            _append_metric(
                lines,
                name="headroom_requests_cached_total",
                metric_type="counter",
                help_text="Cached request count",
                value=self.requests_cached,
            )
            _append_metric(
                lines,
                name="headroom_requests_rate_limited_total",
                metric_type="counter",
                help_text="Rate limited requests",
                value=self.requests_rate_limited,
            )
            _append_metric(
                lines,
                name="headroom_requests_failed_total",
                metric_type="counter",
                help_text="Failed requests",
                value=self.requests_failed,
            )
            _append_metric(
                lines,
                name="headroom_inbound_requests_total",
                metric_type="counter",
                help_text="All inbound HTTP requests accepted by the proxy",
                value=self.inbound_requests_total,
            )
            _append_metric(
                lines,
                name="headroom_inbound_requests_completed_total",
                metric_type="counter",
                help_text="Inbound HTTP requests completed or aborted by the proxy",
                value=self.inbound_requests_completed,
            )
            _append_metric(
                lines,
                name="headroom_inbound_requests_active",
                metric_type="gauge",
                help_text="Inbound HTTP requests currently active in the proxy",
                value=self.inbound_requests_active,
            )
            _append_metric(
                lines,
                name="headroom_tokens_input_total",
                metric_type="counter",
                help_text="Total input tokens",
                value=self.tokens_input_total,
            )
            _append_metric(
                lines,
                name="headroom_tokens_output_total",
                metric_type="counter",
                help_text="Total output tokens",
                value=self.tokens_output_total,
            )
            _append_metric(
                lines,
                name="headroom_tokens_saved_total",
                metric_type="counter",
                help_text="Tokens saved by optimization",
                value=self.tokens_saved_total,
            )
            _append_metric(
                lines,
                name="headroom_persistent_savings_requests_total",
                metric_type="counter",
                help_text="Durable lifetime requests recorded by the proxy savings tracker",
                value=lifetime_savings["requests"],
            )
            _append_metric(
                lines,
                name="headroom_persistent_savings_tokens_saved_total",
                metric_type="counter",
                help_text="Durable lifetime input tokens saved by proxy compression",
                value=lifetime_savings["tokens_saved"],
            )
            _append_metric(
                lines,
                name="headroom_persistent_savings_input_tokens_total",
                metric_type="counter",
                help_text="Durable lifetime input tokens recorded by the proxy savings tracker",
                value=lifetime_savings["total_input_tokens"],
            )
            _append_metric(
                lines,
                name="headroom_persistent_savings_input_cost_usd_total",
                metric_type="counter",
                help_text="Durable lifetime input spend in USD estimated by the proxy savings tracker",
                value=lifetime_savings["total_input_cost_usd"],
            )
            _append_metric(
                lines,
                name="headroom_persistent_savings_compression_savings_usd_total",
                metric_type="counter",
                help_text=(
                    "Durable lifetime compression savings in USD estimated by the "
                    "proxy savings tracker"
                ),
                value=lifetime_savings["compression_savings_usd"],
            )
            # NOTE: per-strategy compression breakdown is tracked
            # internally on `self.compressions_by_strategy` and
            # `self.tokens_saved_by_strategy` (populated by
            # `record_compression`) but **deliberately not exported
            # here** as individual Prometheus series. The state is
            # still observable via /stats + tests + programmatic
            # introspection.
            _append_metric(
                lines,
                name="headroom_latency_ms_sum",
                metric_type="counter",
                help_text="Sum of request latencies in milliseconds",
                value=round(self.latency_sum_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_latency_ms_count",
                metric_type="counter",
                help_text="Count of observed request latencies",
                value=self.latency_count,
            )
            _append_metric(
                lines,
                name="headroom_latency_ms_min",
                metric_type="gauge",
                help_text="Minimum observed request latency in milliseconds",
                value=0 if self.latency_count == 0 else round(self.latency_min_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_latency_ms_max",
                metric_type="gauge",
                help_text="Maximum observed request latency in milliseconds",
                value=round(self.latency_max_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_sum",
                metric_type="counter",
                help_text="Sum of Headroom processing overhead in milliseconds",
                value=round(self.overhead_sum_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_count",
                metric_type="counter",
                help_text="Count of observed Headroom overhead samples",
                value=self.overhead_count,
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_min",
                metric_type="gauge",
                help_text="Minimum observed Headroom overhead in milliseconds",
                value=0 if self.overhead_count == 0 else round(self.overhead_min_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_max",
                metric_type="gauge",
                help_text="Maximum observed Headroom overhead in milliseconds",
                value=round(self.overhead_max_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_sum",
                metric_type="counter",
                help_text="Sum of time to first byte in milliseconds",
                value=round(self.ttfb_sum_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_count",
                metric_type="counter",
                help_text="Count of observed time-to-first-byte samples",
                value=self.ttfb_count,
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_min",
                metric_type="gauge",
                help_text="Minimum observed time to first byte in milliseconds",
                value=0 if self.ttfb_count == 0 else round(self.ttfb_min_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_max",
                metric_type="gauge",
                help_text="Maximum observed time to first byte in milliseconds",
                value=round(self.ttfb_max_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_cache_bust_total",
                metric_type="counter",
                help_text="Requests that lost provider cache efficiency because of compression",
                value=self.cache_bust_count,
            )
            _append_metric(
                lines,
                name="headroom_cache_bust_tokens_lost_total",
                metric_type="counter",
                help_text="Tokens that lost provider cache discount because of compression",
                value=self.cache_bust_tokens_lost,
            )

            if self.cache_miss_attribution_by_provider:
                lines.extend(
                    [
                        "# HELP headroom_cache_miss_attribution_total Cache misses on an "
                        "expected-cached prefix, bucketed by reason (ttl_expiry|prefix_change|unknown)",
                        "# TYPE headroom_cache_miss_attribution_total counter",
                    ]
                )
                for _provider, _reasons in self.cache_miss_attribution_by_provider.items():
                    for _reason, _count in _reasons.items():
                        lines.append(
                            f'headroom_cache_miss_attribution_total{{provider="{_provider}",'
                            f'reason="{_reason}"}} {_count}'
                        )
                lines.append("")

            lines.extend(
                [
                    "# HELP headroom_requests_by_provider Requests by provider",
                    "# TYPE headroom_requests_by_provider counter",
                ]
            )
            for provider, count in self.requests_by_provider.items():
                lines.append(f'headroom_requests_by_provider{{provider="{provider}"}} {count}')
            lines.append("")

            lines.extend(
                [
                    "# HELP headroom_requests_by_model Requests by model",
                    "# TYPE headroom_requests_by_model counter",
                ]
            )
            for model, count in self.requests_by_model.items():
                lines.append(f'headroom_requests_by_model{{model="{model}"}} {count}')
            lines.append("")

            if self.transform_timing_sum:
                lines.extend(
                    [
                        "# HELP headroom_transform_timing_ms_sum Sum of transform timing in milliseconds",
                        "# TYPE headroom_transform_timing_ms_sum counter",
                    ]
                )
                for name, total in self.transform_timing_sum.items():
                    lines.append(
                        f'headroom_transform_timing_ms_sum{{transform="{_escape_label_value(name)}"}} {round(total, 2)}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_transform_timing_ms_count Count of transform timing samples",
                        "# TYPE headroom_transform_timing_ms_count counter",
                    ]
                )
                for name, count in self.transform_timing_count.items():
                    lines.append(
                        f'headroom_transform_timing_ms_count{{transform="{_escape_label_value(name)}"}} {count}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_transform_timing_ms_max Maximum transform timing in milliseconds",
                        "# TYPE headroom_transform_timing_ms_max gauge",
                    ]
                )
                for name, max_value in self.transform_timing_max.items():
                    lines.append(
                        f'headroom_transform_timing_ms_max{{transform="{_escape_label_value(name)}"}} {round(max_value, 2)}'
                    )
                lines.append("")

            if stage_timing_sum_snapshot:
                lines.extend(
                    [
                        "# HELP headroom_stage_timing_ms_sum Sum of per-stage handler timings in milliseconds",
                        "# TYPE headroom_stage_timing_ms_sum counter",
                    ]
                )
                for (path_label, stage), total in stage_timing_sum_snapshot.items():
                    lines.append(
                        f'headroom_stage_timing_ms_sum{{path="{_escape_label_value(path_label)}",stage="{_escape_label_value(stage)}"}} {round(total, 2)}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_stage_timing_ms_count Count of per-stage handler timing samples",
                        "# TYPE headroom_stage_timing_ms_count counter",
                    ]
                )
                for (path_label, stage), count in stage_timing_count_snapshot.items():
                    lines.append(
                        f'headroom_stage_timing_ms_count{{path="{_escape_label_value(path_label)}",stage="{_escape_label_value(stage)}"}} {count}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_stage_timing_ms_max Maximum per-stage handler timing in milliseconds",
                        "# TYPE headroom_stage_timing_ms_max gauge",
                    ]
                )
                for (path_label, stage), max_value in stage_timing_max_snapshot.items():
                    lines.append(
                        f'headroom_stage_timing_ms_max{{path="{_escape_label_value(path_label)}",stage="{_escape_label_value(stage)}"}} {round(max_value, 2)}'
                    )
                lines.append("")

            # Unit 3: WS session lifecycle gauges + duration histogram.
            lines.extend(
                [
                    "# HELP headroom_active_ws_sessions Active Codex WebSocket sessions",
                    "# TYPE headroom_active_ws_sessions gauge",
                    f"headroom_active_ws_sessions {self.active_ws_sessions}",
                    "",
                    "# HELP headroom_active_relay_tasks Active Codex WS relay tasks",
                    "# TYPE headroom_active_relay_tasks gauge",
                    f"headroom_active_relay_tasks {self.active_relay_tasks}",
                    "",
                ]
            )
            if self.ws_session_duration_sum_ms:
                lines.extend(
                    [
                        "# HELP headroom_ws_session_duration_ms_sum Sum of Codex WS session durations",
                        "# TYPE headroom_ws_session_duration_ms_sum counter",
                    ]
                )
                for cause, total in self.ws_session_duration_sum_ms.items():
                    lines.append(
                        f'headroom_ws_session_duration_ms_sum{{cause="{_escape_label_value(cause)}"}} {round(total, 2)}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_ws_session_duration_ms_count Count of completed Codex WS sessions",
                        "# TYPE headroom_ws_session_duration_ms_count counter",
                    ]
                )
                for cause, count in self.ws_session_duration_count.items():
                    lines.append(
                        f'headroom_ws_session_duration_ms_count{{cause="{_escape_label_value(cause)}"}} {count}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_ws_session_duration_ms_max Maximum Codex WS session duration",
                        "# TYPE headroom_ws_session_duration_ms_max gauge",
                    ]
                )
                for cause, max_value in self.ws_session_duration_max_ms.items():
                    lines.append(
                        f'headroom_ws_session_duration_ms_max{{cause="{_escape_label_value(cause)}"}} {round(max_value, 2)}'
                    )
                lines.append("")

            if self.waste_signals_total:
                lines.extend(
                    [
                        "# HELP headroom_waste_signal_tokens_total Tokens attributed to detected waste signals",
                        "# TYPE headroom_waste_signal_tokens_total counter",
                    ]
                )
                for signal_name, token_count in self.waste_signals_total.items():
                    lines.append(
                        f'headroom_waste_signal_tokens_total{{signal="{_escape_label_value(signal_name)}"}} {token_count}'
                    )
                lines.append("")

            if self.cache_by_provider:
                lines.extend(
                    [
                        "# HELP headroom_cache_read_tokens_total Provider cache read tokens",
                        "# TYPE headroom_cache_read_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_read_tokens_total{{provider="{provider}"}} {stats["cache_read_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_cache_write_tokens_total Provider cache write tokens",
                        "# TYPE headroom_cache_write_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_write_tokens_total{{provider="{provider}"}} {stats["cache_write_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_cache_write_ttl_tokens_total Provider cache write tokens by observed TTL bucket",
                        "# TYPE headroom_cache_write_ttl_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_write_ttl_tokens_total{{provider="{provider}",ttl="5m"}} {stats["cache_write_5m_tokens"]}'
                    )
                    lines.append(
                        f'headroom_cache_write_ttl_tokens_total{{provider="{provider}",ttl="1h"}} {stats["cache_write_1h_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_cache_write_ttl_requests_total Provider cache write requests by observed TTL bucket",
                        "# TYPE headroom_cache_write_ttl_requests_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_write_ttl_requests_total{{provider="{provider}",ttl="5m"}} {stats["cache_write_5m_requests"]}'
                    )
                    lines.append(
                        f'headroom_cache_write_ttl_requests_total{{provider="{provider}",ttl="1h"}} {stats["cache_write_1h_requests"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_uncached_input_tokens_total Input tokens not served from provider cache",
                        "# TYPE headroom_uncached_input_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_uncached_input_tokens_total{{provider="{provider}"}} {stats["uncached_input_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_requests_total Requests with provider cache observations",
                        "# TYPE headroom_provider_cache_requests_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_requests_total{{provider="{provider}"}} {stats["requests"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_hit_requests_total Requests with provider cache reads",
                        "# TYPE headroom_provider_cache_hit_requests_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_hit_requests_total{{provider="{provider}"}} {stats["hit_requests"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_bust_total Provider-specific cache bust count",
                        "# TYPE headroom_provider_cache_bust_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_bust_total{{provider="{provider}"}} {stats["bust_count"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_bust_write_tokens_total Provider cache write tokens attributed to busts",
                        "# TYPE headroom_provider_cache_bust_write_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_bust_write_tokens_total{{provider="{provider}"}} {stats["bust_write_tokens"]}'
                    )
                lines.append("")

            # Phase G PR-G3 remediation (C3): image-redacted counter
            # lives Python-side because base64 redaction is purely a
            # Python-proxy concern (request_logger.py). The Rust
            # proxy previously held a dead counter for this; that's
            # been removed in favour of this Python export.
            #
            # The counter is read at scrape-time from the module-
            # level redaction tracker rather than mirrored into the
            # PrometheusMetrics instance, so we never lose a count
            # to ordering between RequestLogger setup and metrics
            # init.
            from headroom.proxy.request_logger import redactions_total

            _append_metric(
                lines,
                name="proxy_image_generation_call_log_redacted_total",
                metric_type="counter",
                help_text=(
                    "Count of base64-encoded image payloads redacted from request "
                    "logs by the Python proxy's request logger"
                ),
                value=redactions_total(),
            )

            # Phase G PR-G3 remediation (C4): RTK invocations counter
            # also lives Python-side. RTK is wrapped by the
            # `headroom wrap` CLI (headroom.cli.wrap); the proxy
            # observes invocation counts via a process-local tracker
            # the wrap tail bumps. The Rust proxy previously held a
            # dead counter for this; that's been removed.
            from headroom.cli.wrap_rtk_metrics import rtk_invocation_counts

            counts = rtk_invocation_counts()
            lines.extend(
                [
                    "# HELP wrap_rtk_invocations_total RTK invocations observed via the wrap CLI tail",
                    "# TYPE wrap_rtk_invocations_total counter",
                ]
            )
            if not counts:
                # Emit a zero-row under the sentinel tool name so
                # the family advertises HELP/TYPE on a fresh boot
                # and dashboards can probe it before any RTK
                # invocation has happened. Matches the Rust side's
                # H3 force-zero contract.
                lines.append('wrap_rtk_invocations_total{tool="__init__"} 0')
            else:
                for tool, count in counts.items():
                    safe_tool = _escape_label_value(str(tool))
                    lines.append(f'wrap_rtk_invocations_total{{tool="{safe_tool}"}} {count}')
            lines.append("")

            return "\n".join(lines)
