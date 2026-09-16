# Headroom Metrics — Dashboard Guide

What each metric shows, so you can build panels against it.

**Two endpoints.** Both are on the proxy (default `:8787`).

| Surface | How to get it | Use it for |
|---|---|---|
| **Prometheus** — `GET /metrics` | Always on, no config | Everything below. Start here. |
| **OpenTelemetry** — OTLP/HTTP | `HEADROOM_OTEL_METRICS_ENABLED=1` + `pip install "headroom-ai[proxy,otel]"` | Same data, dotted names, plus per-tenant labels |

Names differ between them: Prometheus uses `headroom_tokens_saved_total` (**milliseconds** for timings), OTel uses `headroom.proxy.tokens.saved` (**seconds**). Both are listed below.

---

## The savings panel — start here

**`headroom.proxy.tokens.saved`** is the headline number. It already combines compression + tool-schema deferral — no need to add anything to it.

| Metric | What it shows |
|---|---|
| **`headroom.proxy.tokens.saved`** *(OTel)* | **Total input tokens Headroom kept out of the request.** Compression + tool savings, combined. This is your hero number. |
| `headroom.proxy.savings.usd{source}` *(OTel)* | **Dollars saved**, split by layer: `compression`, `tool_schema`, `output_shaping`, `provider_cache`. Sum for the total. |
| `headroom_persistent_savings_tokens_saved_total` | Same tokens-saved number, but **survives proxy restarts**. Use for "lifetime saved" tiles. |
| `headroom_persistent_savings_compression_savings_usd_total` | **Lifetime dollars saved**, durable across restarts. |
| `headroom_tokens_input_total` | Input tokens actually sent upstream (post-compression). The denominator for a reduction %. |
| `headroom_tokens_output_total` | Output tokens returned by the provider. |

```promql
# Hero tile: tokens saved per second
rate(headroom_tokens_saved_total[5m])
  + sum(rate(headroom_savings_attributed_tokens_total{source="tool_search",realized="true"}[5m]))

# Context reduction %
100 * rate(headroom_tokens_saved_total[5m])
    / clamp_min(rate(headroom_tokens_input_total[5m]) + rate(headroom_tokens_saved_total[5m]), 1)

# Lifetime tiles (survive restart)
headroom_persistent_savings_tokens_saved_total
headroom_persistent_savings_compression_savings_usd_total
```

> **One catch on the Prometheus side.** `headroom_tokens_saved_total` is compression **only** — it leaves out tool-schema deferral. The OTel `headroom.proxy.tokens.saved` includes both. That's why the query above adds the `tool_search` term back in. On tool-heavy workloads the gap is large.

---

## Latency panel

All Prometheus timings are in **milliseconds**, exposed as `_sum` / `_count` / `_min` / `_max`. Build means with `rate(sum)/rate(count)`.

| Metric | What it shows |
|---|---|
| **`headroom_overhead_ms_*`** | **Latency Headroom itself adds.** Handler entry → end of compression. Excludes the LLM call. This is the "what does this cost us" number. |
| `headroom_latency_ms_*` | Total request duration, including the provider. |
| `headroom_ttfb_ms_*` | Time to first byte from upstream. Streaming requests only. |
| `headroom_stage_timing_ms_*{path,stage}` | Where time went inside the handler — `compression_first_stage`, `upstream_connect`, `memory_context`, etc. |
| `headroom_transform_timing_ms_*{transform}` | Time per compression transform. Use to find a slow transform. |

```promql
# Headroom's added overhead, mean ms
rate(headroom_overhead_ms_sum[5m]) / rate(headroom_overhead_ms_count[5m])

# End-to-end, mean ms
rate(headroom_latency_ms_sum[5m]) / rate(headroom_latency_ms_count[5m])

# Slowest stages
topk(5, rate(headroom_stage_timing_ms_sum[5m]) / rate(headroom_stage_timing_ms_count[5m]))
```

> **No percentiles are available.** There are no histogram buckets on `/metrics`, and the OTel histograms ship with default buckets that put every request into one bucket, so `histogram_quantile()` returns nonsense. **Means work fine.** For real p95/p99 today, use the `headroom perf` CLI.
>
> Also: divide each `_sum` by **its own** `_count`. Overhead and TTFB are only sampled when > 0, so their counts are smaller than the latency count.

---

## Cache panel

| Metric | What it shows |
|---|---|
| `headroom_provider_cache_hit_requests_total{provider}` | Requests that read from the provider's prompt cache. |
| `headroom_provider_cache_requests_total{provider}` | Requests with any cache activity. **The correct denominator for hit rate.** |
| `headroom_cache_read_tokens_total{provider}` | Tokens served from cache (the discounted ones). |
| `headroom_cache_write_tokens_total{provider}` | Tokens written into cache (these carry a premium). |
| `headroom_cache_write_ttl_tokens_total{provider,ttl}` | Cache writes split by TTL — `5m` vs `1h`. |
| `headroom_uncached_input_tokens_total{provider}` | Input tokens that missed cache entirely. |
| `headroom_cache_bust_total` | Requests where compression broke a cached prefix. **Should stay near zero.** |
| `headroom_cache_miss_attribution_total{provider,reason}` | Why a cached prefix missed — `ttl_expiry`, `prefix_change`, `unknown`. |

```promql
# Cache hit rate by provider
sum by (provider) (rate(headroom_provider_cache_hit_requests_total[5m]))
  / sum by (provider) (rate(headroom_provider_cache_requests_total[5m]))

# Compression breaking cache — alert if this rises
rate(headroom_cache_bust_total[5m])
```

> **Don't use `headroom_requests_cached_total` as a hit rate.** It mixes the provider's prompt cache with Headroom's own response cache into one boolean, so it measures neither.

---

## Traffic & health panel

| Metric | What it shows |
|---|---|
| `headroom_requests_total` | Requests handled. Unlabelled. |
| `headroom_requests_by_provider{provider}` | Traffic split by provider — `anthropic`, `openai`, `gemini`, `bedrock`… |
| `headroom_requests_by_model{model}` | Traffic split by model. Capped at 1024 distinct; overflow lands in `model="other"`. |
| `headroom_requests_failed_total` | Upstream 5xx errors. |
| `headroom_requests_rate_limited_total` | Requests **Headroom** rejected via its own rate limiter (not upstream 429s). |
| `headroom_compression_failed_total{reason}` | Compression failures — `timeout` or `error`. Fails open, so traffic keeps flowing but savings quietly stop. **Worth an alert.** |
| `headroom_compression_quarantine_total{event}` | Compression disabled after repeated timeouts — `activated`, `skipped`, `released`. |
| `headroom_inbound_requests_active` | In-flight requests, gauge. Counts all HTTP including `/metrics`. |
| `headroom_active_ws_sessions` | Live Codex WebSocket sessions, gauge. |

```promql
# Failure rate
rate(headroom_requests_failed_total[5m])
  / clamp_min(rate(headroom_requests_total[5m]) + rate(headroom_requests_failed_total[5m]), 1)

# Savings silently stopped
sum by (reason) (rate(headroom_compression_failed_total[5m]))

# Traffic mix
sum by (provider) (rate(headroom_requests_by_provider[5m]))
```

---

## Anthropic subscription panel

Only if you're on an Anthropic OAuth/subscription plan. OTel only, gauges, no labels.

| Metric | What it shows |
|---|---|
| `headroom.subscription.5h_utilization_pct` | How much of the 5-hour rate-limit window is used (0–100). |
| `headroom.subscription.7d_utilization_pct` | Same for the 7-day window. |
| `headroom.subscription.5h_seconds_to_reset` | Seconds until the 5-hour window resets. |
| `headroom.subscription.7d_seconds_to_reset` | Seconds until the 7-day window resets. |
| `headroom.subscription.overage_usd` | Extra-usage credits consumed, in dollars. |

---

## Attribution — where savings came from

| Metric | What it shows |
|---|---|
| `headroom_savings_attributed_tokens_total{source,realized}` | Tokens saved, broken out by named source. `source="tool_search"` is tool-schema deferral. |
| `headroom_savings_attributed_usd_total{source,realized}` | Dollars saved by source. **Gauge, can go negative** — don't `rate()` it. |
| `headroom_savings_attribution_events_total{source,realized}` | How often each source contributed. |
| `headroom_waste_signal_tokens_total{signal}` | Wasteful patterns *detected* in the input — `json_bloat`, `base64`, `repetition`, `reread`… This is diagnosis, **not savings**. |

These rows *explain* the headline total — they are never added to it.

---

## Compression internals

| Metric | What it shows |
|---|---|
| `headroom.compression.tokens.input` *(OTel)* | Tokens going into the compression pipeline. |
| `headroom.compression.tokens.output` *(OTel)* | Tokens coming out. |
| `headroom.compression.tokens.saved` *(OTel)* | The difference. Pipeline-level view of compression only. |
| `headroom.compression.runs` *(OTel)* | Pipeline executions. Note: **per pipeline run, not per request.** |
| `headroom.compression.pipeline.duration` *(OTel, seconds)* | How long the pipeline took. |
| `headroom.compression.transforms{transform}` *(OTel)* | Which transforms fired. **High cardinality — drop or aggregate at the collector.** |

---

## Five things that will break a dashboard

1. **Only savings counters survive a restart.** 55 of 60 Prometheus families reset to zero when the proxy restarts. Only `headroom_persistent_savings_*` is durable, and it needs `HEADROOM_WORKSPACE_DIR` on a persistent volume — otherwise it resets on every deploy.

2. **No percentiles anywhere.** Use means. See the latency section.

3. **`headroom_latency_ms` measures differently for streaming.** On streaming requests the timer starts *after* compression, so end-to-end is `latency + overhead`. On non-streaming it's just `latency`. Don't mix both in one panel.

4. **A 5xx erases its own savings.** Requests that fail upstream are dropped from every savings and token counter. During a provider incident, savings rates look artificially clean while throughput falls.

5. **`/metrics` needs auth if you set a proxy token.** With `HEADROOM_PROXY_TOKEN` set, any non-loopback scraper must send `Authorization: Bearer <token>`. Loopback is always exempt.

---

## Metrics the docs mention that don't exist

If panels came back empty, this is probably why. These names appear in the published docs but not in the code:

`headroom_compression_ratio` · `headroom_latency_seconds` (and `_bucket`) · `headroom_cache_hits_total` · `headroom_cache_misses_total` · `headroom_cost_usd_total` · the `mode="optimize"` label on `headroom_requests_total`

The shipped `examples/grafana/headroom-dashboard.json` also filters every panel on `pool` and `hook` labels that no metric emits — the dropdowns will be permanently empty. Its metric names are otherwise correct.

---

## Setup reference

```bash
# Prometheus — nothing to do, GET /metrics is always on

# OpenTelemetry
pip install "headroom-ai[proxy,otel]"
export HEADROOM_OTEL_METRICS_ENABLED=1
export HEADROOM_OTEL_METRICS_ENDPOINT=https://otel.corp.example/v1/metrics
export HEADROOM_OTEL_METRICS_HEADERS="authorization=Bearer XXX"
export HEADROOM_OTEL_RESOURCE_ATTRIBUTES="service.instance.id=$HOSTNAME"
```

| Variable | Default | Notes |
|---|---|---|
| `HEADROOM_OTEL_METRICS_ENABLED` | `0` | Master switch |
| `HEADROOM_OTEL_METRICS_EXPORTER` | `otlp_http` | Or `console`. No gRPC exporter exists. |
| `HEADROOM_OTEL_METRICS_ENDPOINT` | unset | Passed verbatim — `/v1/metrics` is **not** appended |
| `HEADROOM_OTEL_METRICS_HEADERS` | unset | `k=v,k2=v2` |
| `HEADROOM_OTEL_METRICS_EXPORT_INTERVAL_MS` | `10000` | |
| `HEADROOM_OTEL_SERVICE_NAME` | `headroom-proxy` | |
| `HEADROOM_OTEL_RESOURCE_ATTRIBUTES` | unset | **Set `service.instance.id` here** — Headroom doesn't, and replicas will collide |

Verify with `curl -s localhost:8787/stats | jq .otel`.

**Multi-tenant labels:** `register_otel_metric_attribute_provider()` adds request-scoped attributes (tenant, team, cost centre) to every OTel datapoint. Max 16 attributes, 256 chars each.

**Air-gapped deployments:** `HEADROOM_OFFLINE=1` disables all outbound traffic — the anonymous usage beacon (which is **on by default**), the update check, and model downloads.

---
