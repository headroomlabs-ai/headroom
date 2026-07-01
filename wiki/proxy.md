# Proxy Server Documentation

The Headroom proxy server is a production-ready HTTP server that applies context optimization to all requests passing through it.

> **New:** The proxy now supports the [TypeScript SDK](typescript-sdk.md) via the `POST /v1/compress` endpoint, enabling compression-as-a-service for any HTTP client without calling an LLM.

## Starting the Proxy

```bash
# Basic usage
headroom proxy

# Custom port
headroom proxy --port 8080

# With all options
headroom proxy \
  --host 0.0.0.0 \
  --port 8787 \
  --log-file /var/log/headroom.jsonl \
  --budget 100.0
```

### Common agent CLI entrypoints

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# GitHub Copilot CLI
headroom wrap copilot -- --model claude-sonnet-4-20250514

# OpenAI-compatible clients
OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

`headroom wrap copilot` uses Copilot CLI's BYOK provider settings under the hood. In `provider-type=auto`, it chooses Headroom's Anthropic route for the default proxy backend and the OpenAI-compatible `/v1` route for translated backends such as `anyllm` and LiteLLM.

Anonymous aggregate telemetry is **off by default** (opt-in). Opt in with `HEADROOM_TELEMETRY=on` or `headroom proxy --telemetry`. Downstream apps can set `HEADROOM_SDK=headroom-app` to override the anonymous telemetry `sdk` label; the default remains `proxy`.

Operational OTEL metrics are configured separately and are **off by default**. Install `headroom-ai[proxy,otel]` and set:

```bash
HEADROOM_OTEL_METRICS_ENABLED=1
HEADROOM_OTEL_METRICS_EXPORTER=otlp_http
HEADROOM_OTEL_METRICS_ENDPOINT=http://127.0.0.1:4318/v1/metrics
HEADROOM_OTEL_SERVICE_NAME=headroom-proxy
```

Use `HEADROOM_OTEL_METRICS_EXPORTER=console` for local smoke testing. `HEADROOM_TELEMETRY` controls the anonymous data-flywheel beacon only; it does not disable or enable OTEL export.

Langfuse can be enabled alongside this OTEL path for **trace ingestion**. Langfuse does **not** ingest OTEL metrics, so Headroom keeps metrics and Langfuse traces as complementary signals:

```bash
HEADROOM_LANGFUSE_ENABLED=1
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

When configured, Headroom emits OTLP traces for the shared compression pipeline to Langfuse while continuing to expose metrics through `/metrics` and OTEL metric exporters.

## Command Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8787` | Port to bind to |
| `--mode` | `token` | Run mode: `token` (maximize compression) or `cache` (freeze prior turns) |
| `--no-optimize` | `false` | Disable optimization (passthrough mode) |
| `--no-cache` | `false` | Disable semantic caching |
| `--no-rate-limit` | `false` | Disable rate limiting |
| `--log-file` | None | Path to JSONL log file |
| `--budget` | None | Daily budget limit in USD |
| `--code-aware` | true | Enable AST-based code compression (env: HEADROOM_CODE_AWARE_ENABLED) |
| `--no-code-aware` | false | Disable code-aware compression |
| `--anthropic-api-url` | `https://api.anthropic.com` | Custom Anthropic API URL endpoint |
| `--openai-api-url` | `https://api.openai.com` | Custom OpenAI API URL endpoint |

### Run Modes

Headroom proxy has two explicit run modes:

- `token` mode: prioritize token reduction. Prior history may be rewritten when that improves compression.
- `cache` mode: prioritize provider prefix cache stability. Prior turns are frozen; only the newest turn is mutable.

Set via CLI or env:

```bash
headroom proxy --mode token
HEADROOM_MODE=cache headroom proxy
```

When to pick each:

- `token`: best for maximizing immediate compression savings.
- `cache`: best for long conversations where preserving prior-turn bytes improves prefix-cache reuse.

Legacy values (`token_headroom`, `cost_savings`) are still accepted as aliases.

### Context Management Options

Context management in the proxy is handled automatically by the compression pipeline. CCR (Compress-Cache-Retrieve) ensures that when content is compressed or messages are dropped, the original data remains accessible for the LLM to retrieve on demand. See [CCR documentation](ccr.md) for details.

Key CCR-related proxy flags:

| Option | Description |
|--------|-------------|
| `--no-ccr-inject-tool` | Do not inject the `headroom_retrieve` tool into the LLM's available tools |
| `--no-ccr-marker` | Do not add retrieval markers to compressed output |
| `--no-ccr-proactive-expansion` | Disable proactive context expansion before the LLM asks |

### ML Compression — RETIRED `--llmlingua` flag

The `--llmlingua` / `--llmlingua-device` / `--llmlingua-rate` flags and
the `headroom-ai[llmlingua]` extra were retired and replaced by Kompress
(ModernBERT). For the current opt-in path, install `headroom-ai[ml]`
and see [transforms.md](transforms.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## API Endpoints

### Liveness

```bash
curl http://localhost:8787/livez
```

Response:
```json
{
  "service": "headroom-proxy",
  "status": "healthy",
  "alive": true,
  "version": "0.5.21",
  "timestamp": "2026-04-10T16:36:25Z",
  "uptime_seconds": 12.483
}
```

### Readiness

```bash
curl http://localhost:8787/readyz
```

Response:
```json
{
  "service": "headroom-proxy",
  "status": "healthy",
  "ready": true,
  "version": "0.5.21",
  "timestamp": "2026-04-10T16:36:25Z",
  "uptime_seconds": 12.483,
  "checks": {
    "startup": {"enabled": true, "ready": true, "status": "healthy"},
    "http_client": {"enabled": true, "ready": true, "status": "healthy"},
    "cache": {"enabled": true, "ready": true, "status": "healthy"},
    "rate_limiter": {"enabled": true, "ready": true, "status": "healthy"},
    "memory": {"enabled": false, "ready": true, "status": "disabled"}
  }
}
```

`/readyz` returns HTTP 503 when Headroom has not completed startup or a required enabled subsystem is unavailable. This is the endpoint used by the container health checks.

### Aggregate Health

```bash
curl http://localhost:8787/health
```

Response:
```json
{
  "status": "healthy",
  "ready": true,
  "version": "0.5.21",
  "config": {
    "backend": "anthropic",
    "optimize": true,
    "cache": true,
    "rate_limit": true
  },
  "checks": {
    "startup": {"enabled": true, "ready": true, "status": "healthy"},
    "http_client": {"enabled": true, "ready": true, "status": "healthy"}
  }
}
```

#### Multi-hop upstream probe

By default `/health` makes no upstream call, so a `200` only confirms Headroom
itself is up — not that the rest of a multi-hop chain (e.g. Headroom → corp
proxy → Bedrock) is reachable. Pass `?probe=true` to actively probe the
configured upstream:

```bash
curl "http://localhost:8787/health?probe=true"
```

The response gains an `upstreamProbe` block, and `status` becomes `degraded` if
the probe fails (the request still returns `200` — inspect the body):

```json
{
  "status": "healthy",
  "upstreamProbe": {"url": "http://localhost:6655", "status": 200, "latencyMs": 12, "error": null}
}
```

```json
{
  "status": "degraded",
  "upstreamProbe": {"url": "http://localhost:6655", "status": null, "latencyMs": null, "error": "ECONNREFUSED"}
}
```

The probe is a `HEAD` request with a 5-second timeout — it verifies TLS + TCP
reachability without triggering an inference call.

### Detailed Statistics

```bash
curl http://localhost:8787/stats
```

`/stats` remains the live/session-oriented endpoint and now also includes a
`persistent_savings` block with durable proxy compression lifetime totals plus a
small recent preview. The existing `savings_history` field is still present and
remains session-scoped for backward compatibility.

For providers that return cache-write TTL bucket usage, `/stats` also includes
observed TTL breakdowns under `prefix_cache`:

- `observed_ttl_buckets.5m.tokens`
- `observed_ttl_buckets.1h.tokens`
- `observed_ttl_mix`

These are provider-reported observations, not configured TTL and not remaining
expiration time.

### Historical Savings

```bash
curl http://localhost:8787/stats-history
```

`/stats-history` exposes durable proxy compression history for dashboards and
other Headroom frontends. It returns:

- lifetime proxy compression totals
- compact checkpoint history by default, with `history_mode=full` available for
  export/debug flows
- derived hourly, daily, weekly, and monthly rollups for charts
- a `history_summary` block describing stored versus returned checkpoint counts
- UTC timestamps throughout

By default the proxy stores this history at
`${HEADROOM_WORKSPACE_DIR}/proxy_savings.json` (i.e.
`~/.headroom/proxy_savings.json` when `HEADROOM_WORKSPACE_DIR` is unset).
Set `HEADROOM_SAVINGS_PATH` to override the location directly, or set
`HEADROOM_WORKSPACE_DIR` to relocate the full state root. See the
[Filesystem Contract](filesystem-contract.md).

`/dashboard` uses this endpoint directly for its historical view, including the
daily/weekly/monthly rollups and built-in JSON / CSV export buttons.

```bash
curl "http://localhost:8787/stats-history?format=csv&series=weekly"
curl "http://localhost:8787/stats-history?format=csv&series=monthly"
curl "http://localhost:8787/stats-history?history_mode=full"
```

### Prometheus Metrics

```bash
curl http://localhost:8787/metrics
```

`/metrics` remains the built-in Prometheus-formatted operational view. The proxy now also emits the same operational events through the OTEL facade when OTEL metrics are configured.

### LLM APIs

The proxy supports both Anthropic and OpenAI API formats:

```bash
# Anthropic format
POST /v1/messages

# OpenAI format
POST /v1/chat/completions
```

### `POST /v1/compress`

Compression-only endpoint. Compresses messages without calling any LLM. Used by the [TypeScript SDK](typescript-sdk.md) and any HTTP client that wants compression as a service.

**Request:**
```json
{
  "messages": [...],     // OpenAI chat format
  "model": "gpt-4o"     // model name (for token counting)
}
```

**Response:**
```json
{
  "messages": [...],            // compressed messages
  "tokens_before": 15000,
  "tokens_after": 3500,
  "tokens_saved": 11500,
  "compression_ratio": 0.23,
  "transforms_applied": ["router:smart_crusher:0.35"],
  "ccr_hashes": ["a1b2c3"]
}
```

**Headers:**
- `x-headroom-bypass: true` — skip compression, return messages as-is

**Error responses:** 400 (missing fields), 401 (bad API key), 503 (compression failed)

## Using with Claude Code

```bash
# Start proxy
headroom proxy --port 8787

# In another terminal
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

## Using with Cursor

1. Start the proxy: `headroom proxy`
2. In Cursor settings, set the base URL to `http://localhost:8787`

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8787/v1",
    api_key="your-api-key",  # Still needed for upstream
)
```

## Features

### ML Compression (Opt-In, Kompress)

> The earlier LLMLingua-2 integration documented in this section
> (`--llmlingua`, `--llmlingua-device`, `--llmlingua-rate`,
> `headroom-ai[llmlingua]`, `LLMLinguaCompressor`) was retired and
> replaced by **Kompress** (ModernBERT). Install with `pip install
> 'headroom-ai[ml]'`. See [transforms.md](transforms.md) and
> [ARCHITECTURE.md](ARCHITECTURE.md) for current configuration.

### Semantic Caching

The proxy caches responses for repeated queries:

- LRU eviction with configurable max entries
- TTL-based expiration
- Cache key based on message content hash

### Rate Limiting

Token bucket rate limiting protects against runaway costs:

- Configurable requests per minute
- Configurable tokens per minute
- Per-API-key tracking

### Cost Tracking

Track spending and enforce budgets:

- Real-time cost estimation
- Budget periods: hourly, daily, monthly
- Automatic request rejection when over budget

### Prometheus Metrics

Export metrics for monitoring:

```
headroom_requests_total
headroom_tokens_saved_total
headroom_cost_usd_total
headroom_latency_ms_sum
```

## Configuration via Environment

```bash
export HEADROOM_HOST=0.0.0.0
export HEADROOM_PORT=8787
export HEADROOM_BUDGET=100.0

# Route OpenAI passthrough requests to a custom endpoint
export OPENAI_TARGET_API_URL=https://custom.openai.endpoint.com

# Route Anthropic passthrough requests to a custom endpoint
export ANTHROPIC_TARGET_API_URL=https://litellm.company.internal

headroom proxy
```

## Running in Production

For production deployments:

```bash
# Use a process manager
pip install gunicorn

# Run with gunicorn
gunicorn headroom.proxy.server:app \
  --workers 4 \
  --bind 0.0.0.0:8787 \
  --worker-class uvicorn.workers.UvicornWorker
```

Or with Docker:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install "headroom-ai[proxy]" \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
EXPOSE 8787
CMD ["headroom", "proxy", "--host", "0.0.0.0"]
```

> **Note:** `build-essential` is required at install time because `headroom-ai` includes `hnswlib`, a C++ extension that must be compiled from source. It is removed after installation to keep the image slim.
