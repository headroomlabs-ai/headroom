# Observability — proxy metrics

The Headroom Rust proxy exposes Prometheus-format metrics on the
`/metrics` endpoint of every running proxy instance. The metric
catalogue below covers Phase D (Bedrock route instrumentation) and
Phase G PR-G3 (per-invocation RTK + proxy-wide observability).

All metric names + label keys are constants in
`crates/headroom-proxy/src/observability/metric_names.rs`, so any
rename catches one file in code review.

## Metric catalogue

### Bedrock route (Phase D PR-D3)

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `bedrock_invoke_count_total` | Counter | `model`, `region`, `auth_mode` | One increment per Bedrock `/invoke` or `/converse` request. |
| `bedrock_invoke_latency_seconds` | Histogram | `model`, `region` | Latency from proxy entry to upstream completion. Buckets target 50ms–60s. |
| `bedrock_eventstream_message_count_total` | Counter | `model`, `region`, `event_type` | One increment per parsed binary EventStream message. |

### Proxy-wide (Phase G PR-G3)

#### Cache + compression

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_cache_hit_rate_per_session` | Histogram | `provider` | Per-session cache hit rate. **Phase H canary gate.** |
| `proxy_compression_ratio_by_strategy` | Histogram | `strategy`, `content_type` | `compressed_tokens / original_tokens` per shrunk block. |
| `proxy_compression_rejected_by_token_check_total` | Counter | `strategy` | Compressor ran but failed the shrink check. |

#### Cache-safety alarm

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_passthrough_bytes_modified_total` | Counter | `path` | Bytes mutated on a passthrough path. **Must stay 0 outside the compression hot path** — any non-zero rate fires the cache-safety alarm. |

The alarm metric is wired in `crates/headroom-proxy/src/proxy.rs`:
when the dispatcher returns `Outcome::NoCompression` or
`Outcome::Passthrough`, the post-dispatcher byte length is compared
to the original buffered length and any delta increments the
counter (by the byte delta) under the request's path label. The
PR-E4 prompt_cache_key injector runs AFTER the alarm check, so its
intentional byte mutations do not trip the alarm.

#### Upstream rate limits

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_rate_limit_remaining_requests` | Gauge | `provider` | Last-seen remaining requests in the current window. |
| `proxy_rate_limit_remaining_tokens` | Gauge | `provider` | Last-seen remaining tokens in the current window. |
| `proxy_rate_limit_remaining_input_tokens` | Gauge | `provider` | Anthropic-only input-token bucket. |
| `proxy_rate_limit_remaining_output_tokens` | Gauge | `provider` | Anthropic-only output-token bucket. |

#### OpenAI Responses telemetry

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_service_tier_count_total` | Counter | `tier` | Service-tier distribution observed at the proxy. |
| `proxy_response_status_count_total` | Counter | `status` | Terminal status distribution (`completed`, `incomplete`, `failed`, `cancelled`, `in_progress`). |

#### Wrap CLI / RTK (Python-side)

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `wrap_rtk_invocations_total` | Counter | `tool` | RTK invocations observed via the wrap-CLI tail. Surfaced via the Python proxy's `/metrics` exporter; the wrap CLI bumps `headroom.cli.wrap_rtk_metrics.record_rtk_invocation(...)`. |

> **C4 remediation:** This counter is Python-side because RTK is
> wrapped by `headroom wrap` (Python CLI) and the wrap-side tail
> is the natural emit site. The Rust proxy previously held a dead
> counter for this metric; that has been removed.

#### Image log redaction (Python-side)

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_image_generation_call_log_redacted_total` | Counter | _none_ | Base64-encoded image payloads redacted from request logs. Driven from `headroom.proxy.request_logger.redactions_total()`. |

> **C3 remediation:** Image redaction is purely a Python-proxy
> operation (the request logger walks JSON and replaces over-
> threshold image payloads with placeholders). The counter lives
> Python-side so we have one source of truth instead of two. The
> Rust proxy previously held a dead counter for this metric; that
> has been removed.

## How to query

The proxy renders Prometheus text-format on `GET /metrics`:

```bash
curl -s http://127.0.0.1:8787/metrics
```

### Phase H canary gate

The canary script that decides "ship Rust, retire Python" uses
**all four** of these queries against `proxy_cache_hit_rate_per_session`
to confirm parity vs the Python baseline. A single percentile is
not enough — a regression that only shows up at the tail (a small
class of long sessions losing cache hits) would slip through a
median-only check.

```promql
# p50, p95, p99 of cache hit rate over the last 5 minutes, per provider.
histogram_quantile(0.50, sum by (provider, le) (rate(proxy_cache_hit_rate_per_session_bucket{provider!="__init__"}[5m])))
histogram_quantile(0.95, sum by (provider, le) (rate(proxy_cache_hit_rate_per_session_bucket{provider!="__init__"}[5m])))
histogram_quantile(0.99, sum by (provider, le) (rate(proxy_cache_hit_rate_per_session_bucket{provider!="__init__"}[5m])))

# Mean cache hit rate over the last 5 minutes, per provider. The
# `sum / count` form is the cleanest "average without a quantile"
# query and is what the Python baseline reports.
sum by (provider) (rate(proxy_cache_hit_rate_per_session_sum{provider!="__init__"}[5m]))
  /
sum by (provider) (rate(proxy_cache_hit_rate_per_session_count{provider!="__init__"}[5m]))
```

The canary fails if ANY of `p50`, `p95`, `p99`, or `mean` regresses
below the Python baseline for any provider over the canary window.

### Other common queries

```promql
# Cache-safety alarm. Should always be 0 (post-`__init__` row).
sum(rate(proxy_passthrough_bytes_modified_total{path!="__init__"}[5m]))

# Per-strategy compression value at p50 (post-H1 fix: each strategy
# reports its own before/after; pre-fix this was the same aggregate
# ratio repeated per strategy).
histogram_quantile(0.50, sum by (strategy, le) (rate(proxy_compression_ratio_by_strategy_bucket{strategy!="__init__"}[1h])))

# Per-strategy compression value at p95 and p99 (catch outlier
# strategies that fail to shrink at the tail).
histogram_quantile(0.95, sum by (strategy, le) (rate(proxy_compression_ratio_by_strategy_bucket{strategy!="__init__"}[1h])))
histogram_quantile(0.99, sum by (strategy, le) (rate(proxy_compression_ratio_by_strategy_bucket{strategy!="__init__"}[1h])))

# Strategies that ran but failed the token-check (compressor ran
# but its output was not strictly smaller, so the original was
# kept). High rate here means the compressor needs tuning.
sum by (strategy) (rate(proxy_compression_rejected_by_token_check_total{strategy!="__init__"}[1h]))

# Upstream rate-limit headroom (smaller = closer to throttle).
proxy_rate_limit_remaining_tokens{provider="anthropic"}

# RTK invocation rate (Python-side).
sum by (tool) (rate(wrap_rtk_invocations_total{tool!="__init__"}[5m]))

# Image-redaction rate (Python-side).
rate(proxy_image_generation_call_log_redacted_total[5m])
```

All queries above include a `{... != "__init__"}` filter so the
sentinel zero-rows the boot-touch contract emits do not skew the
result. See "Wiring → H3 force-zero" below.

## Wiring

Every metric registration is `OnceLock`-backed and lazy: the first
call to a `*_counter()` / `*_gauge()` / `*_histogram()` helper
registers the family with the shared registry. `handle_metrics`
force-touches every Phase G PR-G3 family before scraping.

### H3 force-zero

The `prometheus` crate v0.13 skips empty MetricVecs from `gather()`
entirely — neither HELP/TYPE lines nor rows appear until the
family has been incremented at least once with a label tuple.
Operators expect to see the catalogue from boot, so
`handle_metrics` increments each counter / gauge MetricVec by 0
under a sentinel `__init__` label tuple before the first scrape.
HELP/TYPE then surface from boot and dashboards/alarms see a
predictable scrape shape.

Counters with the `__init__` label increment by 0, so the
alarm-able "must stay 0" semantic of
`proxy_passthrough_bytes_modified_total` is preserved (the family
becomes visible, the rate stays 0). PromQL queries should filter
`{... != "__init__"}` so the sentinel rows are excluded from
aggregations (the catalogue above does this).

Histograms are NOT force-zeroed: a synthetic `observe(0.0)` would
contribute a real sample to the per-label distribution and pollute
percentile readings. The two histogram families
(`proxy_cache_hit_rate_per_session` and
`proxy_compression_ratio_by_strategy`) only surface in the scrape
after the first real session, by design.

### H4 prometheus crate version pin

The H3 contract above relies on the `prometheus` crate's v0.13
`gather()` semantics — empty MetricVec families are omitted from
the scrape. **This is implementation-defined behaviour.** If
`crates/headroom-proxy/Cargo.toml` ever bumps the `prometheus`
dependency, retest the alarm contract:

1. Start a fresh proxy.
2. `curl /metrics` and confirm every counter / gauge family has
   HELP/TYPE + an `__init__` row.
3. Confirm histograms (`*_cache_hit_rate_per_session`,
   `*_compression_ratio_by_strategy`) DO NOT appear (no
   `observe()` calls yet).
4. Drive one cache-hit session, scrape again, confirm histograms
   now appear.
5. Confirm `passthrough_bytes_modified_total` stays at 0 across
   passthrough requests.

The crate version is pinned exactly (`= "0.13.4"`, no caret) in
`Cargo.toml` precisely so a silent semver bump cannot break the
contract without a code-review trigger.

### C2 alarm wiring

`proxy_passthrough_bytes_modified_total` fires from `proxy.rs` when
a dispatcher arm that promised byte-equal passthrough
(`Outcome::NoCompression` or `Outcome::Passthrough`) produces a
final body of a different byte length. The check runs BEFORE the
PR-E4 prompt_cache_key injector so the injector's intentional byte
mutations do not trip the alarm.

### H1 per-strategy ratio wiring

`proxy_compression_ratio_by_strategy` samples one observation per
strategy using the strategy's OWN before/after token counts
(plumbed through `Outcome::Compressed.per_strategy_tokens` from
the manifest in `live_zone_anthropic` / `live_zone_openai` /
`live_zone_responses`). Pre-H1 the same aggregate ratio was
emitted per strategy when multiple strategies ran on one body,
making Phase H per-strategy dashboards read garbage.

### H2 aborted-stream gate

The `proxy_cache_hit_rate_per_session` histogram observes ONLY
when the SSE stream completed:

* Anthropic: `state.status == StreamStatus::MessageStop` after the
  channel closes.
* OpenAI Chat: `state.usage.is_some()` (the final usage chunk only
  arrives at stream completion).
* OpenAI Responses: `state.terminal_status().is_some()`.

A client disconnect mid-stream closes the channel without setting
the terminal flag — under H2 we log + skip rather than observe a
garbage half-stream sample.

## Cardinality discipline

Every label vocabulary is bounded by code, not customer input:

- `model` / `region`: read from path params + `Config::bedrock_region`.
- `auth_mode`: 3-variant enum (`payg`, `oauth`, `subscription`).
- `provider`: 3 values (`anthropic`, `openai_chat`, `openai_responses`).
- `strategy`: `&'static str` from the compressor's `BlockAction::Compressed`.
- `content_type`: `&'static str` from `headroom_core::transforms::ContentType`.
- `tier`: validated through
  `crate::observability::metric_names::service_tier::validate(raw: &str)`.
  Returns one of `{auto, default, flex, on_demand, priority, scale}`
  or the sentinel `"other"` for anything else. **The raw inbound
  value is never used as a label.** A malicious client posting
  `{"service_tier":"<random>"}` per request gets bucketed to
  `"other"` and a `tracing::warn!` is emitted so wire-format drift
  surfaces loudly in logs.
- `status`: 5-variant enum.
- `tool` (Python-side `wrap_rtk_invocations_total`): bounded by the
  set of tools the wrap CLI rewrites, captured by
  `headroom.cli.wrap_rtk_metrics`.

There is no code path where a malicious client can drive label
cardinality unbounded.

## See also

- `docs/rtk-architecture.md` — why RTK lives wrap-side, not proxy-side.
- `crates/headroom-proxy/src/observability/` — implementation.
- `REALIGNMENT/09-phase-G-rtk-observability.md` — spec.
- `REALIGNMENT/10-phase-H-python-retirement.md` — H1 acceptance gate.
