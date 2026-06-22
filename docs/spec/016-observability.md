# 016. Observability

**Status:** done

## Telemetry

### Metrics

Headroom exposes Prometheus metrics at `/metrics`.

**Key Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `headroom_requests_total` | Counter | Total requests |
| `headroom_tokens_original` | Counter | Original token count |
| `headroom_tokens_compressed` | Counter | Compressed token count |
| `headroom_savings_percent` | Histogram | Savings distribution |
| `headroom_cache_hits_total` | Counter | Cache hits |
| `headroom_cache_misses_total` | Counter | Cache misses |
| `headroom_compression_duration_seconds` | Histogram | Compression latency |
| `headroom_request_duration_seconds` | Histogram | Total request latency |

**Prometheus scrape config:**
```yaml
scrape_configs:
  - job_name: 'headroom'
    static_configs:
      - targets: ['localhost:8787']
    metrics_path: '/metrics'
```

---

### Tracing

OpenTelemetry tracing support.

**Configuration (Langfuse):**
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
HEADROOM_LANGFUSE_ENABLED=1
# Optional: override endpoint and service name
# LANGFUSE_BASE_URL=https://cloud.langfuse.com
# HEADROOM_LANGFUSE_SERVICE_NAME=headroom
```

**Spans:**
| Span | Description |
|------|-------------|
| `headroom.proxy.request` | Full request lifecycle |
| `headroom.compression` | Compression operation |
| `headroom.cache.lookup` | Cache check |
| `headroom.provider.call` | Provider API call |

---

### Logging

**Log Levels:**

| Level | Use Case |
|-------|----------|
| `DEBUG` | Detailed debugging |
| `INFO` | General operation |
| `WARNING` | Degraded operation |
| `ERROR` | Failures |

**Log Format (JSON):**
```json
{
  "timestamp": "2026-04-16T12:00:00Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "abc123",
  "savings": 0.45,
  "duration_ms": 120
}
```

**Configuration:**
```bash
# Logging level is controlled via the --log-level CLI flag (headroom proxy --log-level debug)
# or RUST_LOG env var for the Rust proxy. No HEADROOM_LOG_LEVEL env var exists.
```

Or in config:
```yaml
logging:
  level: INFO
  format: json
```

---

## Dashboard

**URL:** `http://localhost:8787/dashboard`

**Metrics Shown:**
- Total savings over time
- Requests per day
- Cache hit rate
- Top compressed endpoints
- Session overview
- Recent requests feed (latest requests with per-request token savings)
- Per-provider / per-model breakdown

**Requires:** the proxy process to be running. The dashboard is served by default at `/dashboard`.

### Stats are already unified across backends

The Rust proxy fronts every backend (Anthropic, OpenAI, Bedrock, Vertex) in a
**single process**, and the savings store
(`crates/headroom-proxy/src/observability/stats.rs`) attributes each request to
its provider and model. So `/stats` is already one unified view across every
backend in use — there is no separate "combine" or "federation" step, and
nothing to opt into. The dashboard at `/dashboard` is the canonical source of
truth for savings.

```json
{
  "requests": { "total": 184, "by_provider": { "anthropic": 120, "bedrock": 64 } },
  "tokens":   { "saved": 79000, "savings_percent": 41.2 },
  "cost":     { "compression_savings_usd": 2.81, "per_model": { "claude-haiku-4-5": { "tokens_saved": 51000 } } },
  "persistent_savings": {
    "lifetime":        { "requests": 184, "tokens_saved": 79000 },
    "display_session": { "requests": 12,  "tokens_saved": 9000 }
  }
}
```

**Properties:**

- **On by default.** `/stats` and `/dashboard` are always served; no flag.
- **Backend-agnostic.** Every provider the single process serves is attributed
  and aggregated together.
- **Durable.** Lifetime totals, the rolling display-session, history, and
  per-provider / per-model breakdowns persist across restarts
  (`--savings-path`), independent of the ephemeral Prometheus `/metrics`
  counters.
- **Single source of truth during migration.** Set `--upstream-stats-url` to a
  still-running Python proxy's `/stats` and the Rust `/stats` folds in the
  blocks not yet ported to Rust (provider quota / subscription / rate-limit
  panels). Fail-open; each block drops out as it is reimplemented in Rust.

### Surfaces

| Endpoint | Source | Resets on restart | Use |
|---|---|---|---|
| `/metrics` | `observability/prometheus.rs` | yes | Prometheus scrape → Grafana / alerting |
| `/stats` (JSON), `/dashboard` (HTML) | `observability/stats.rs` | no (persisted) | Dashboard savings view |

> **Access control:** like `/metrics`, the `/stats` and `/dashboard` surfaces are
> unauthenticated and expose usage/savings data (model names, token counts, USD).
> They carry no secrets, but operators should keep the proxy port on a trusted
> network or behind a firewall / authenticating reverse proxy rather than the
> public internet.

---

## Alerting

### Recommended Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighErrorRate | error_rate > 5% | warning |
| LowSavings | savings < 20% | warning |
| CacheDown | cache_hits < 10% for 1h | critical |
| ProxyDown | health check fails | critical |

**Alert rule example (Prometheus):**
```yaml
groups:
  - name: headroom
    rules:
      - alert: HighErrorRate
        expr: rate(headroom_errors_total[5m]) / rate(headroom_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in Headroom"
```

---

## Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/livez` | GET | Liveness check (process alive) |
| `/readyz` | GET | Readiness check (can serve traffic) |

**Health response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**Readiness response:**
```json
{
  "ready": true,
  "checks": {
    "database": true,
    "cache": true,
    "provider": true
  }
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial observability document |
