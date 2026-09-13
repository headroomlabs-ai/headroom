# Proxy Server Documentation

The Headroom proxy server is a production-ready HTTP server that applies context optimization to all requests passing through it.

> The proxy exposes compression-as-a-service via the `POST /v1/compress` endpoint — used by the [TypeScript SDK](typescript-sdk.md), LiteLLM's `headroom` guardrail, and gateway sidecars. It is loopback-only by default; see the endpoint section below.

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
| `--mode` | `cache` | Run mode: `token` (maximize compression) or `cache` (freeze prior turns) |
| `--no-optimize` | `false` | Disable optimization (passthrough mode) |
| `--no-cache` | `false` | Disable semantic caching |
| `--no-rate-limit` | `false` | Disable rate limiting |
| `--log-file` | None | Path to JSONL log file |
| `--budget` | None | Daily budget limit in USD |
| `--code-aware` / `--no-code-aware` | disabled | Enable or disable AST-based code compression. Requires `headroom-ai[code]` (env: HEADROOM_CODE_AWARE_ENABLED=1 to enable) |
| `--anthropic-api-url` | `https://api.anthropic.com` | Custom Anthropic API URL endpoint |
| `--openai-api-url` | `https://api.openai.com` | Custom OpenAI API URL endpoint |
| `--anthropic-extra-headers` | unset | JSON object of extra headers merged into (and overriding) forwarded Anthropic requests, e.g. `'{"Api-Key": "..."}'` |
| `--openai-extra-headers` | unset | JSON object of extra headers merged into (and overriding) forwarded OpenAI requests |

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
| `--no-ccr` | Disable CCR entirely — no retrieval markers in compressed output and no injected `headroom_retrieve` tool (lossy, no recovery path) |
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

Compression-only endpoint. Compresses messages without ever making a **completion request to an LLM provider** — no generation, no provider API key. Used by the [TypeScript SDK](typescript-sdk.md), LiteLLM's `headroom` guardrail, and gateway sidecars.

**It does run local ML models.** Compression is ML-backed: Kompress is a ModernBERT encoder scoring tokens for retention (classification, not generation) and Magika classifies content types, both in-process by default. If `HEADROOM_KOMPRESS_ENDPOINT` is set, Kompress inference is offloaded over HTTP to that model server — real egress from the sidecar. Only inference goes remote; the CCR store and markers stay proxy-local. `HEADROOM_DISABLE_KOMPRESS=1` gives structural compression only.

**Loopback-only by default.** Non-loopback callers get **404** (not 403 — the route stays invisible to scanners). Set `HEADROOM_COMPRESS_ALLOW_REMOTE=1` to allow remote callers.

**No format conversion.** `messages` may be OpenAI-shaped (`role: "tool"` + `tool_call_id`) or Anthropic-shaped (`tool_use` / `tool_result` content blocks); the same shape comes back. `model` selects the tokenizer and context limit — send the real name, including gateway-prefixed forms like `bedrock/anthropic.claude-3-5-sonnet`.

**`system` and `tools` are ignored.** Anthropic sends both out of band. This endpoint accepts them without complaint (200, no warning) and returns neither, so neither is compressed — keep carrying them yourself. That means the Anthropic system prompt is not compressed here, and tool-schema compaction / tool-search deferral are not reachable through this route; run Headroom as the proxy if you need those.

**Request:**
```json
{
  "messages": [...],          // either wire format
  "model": "gpt-4o",          // tokenizer + context limit
  "token_budget": 8000,       // optional: override the context limit
  "config": {                 // optional
    "mode": "lossy_inline",       // ccr | lossy_inline | lossless_then_lossy
    "frozen_message_count": 12,   // pin an already-cached prefix
    "compress_user_messages": false,
    "target_ratio": 0.5,
    "protect_recent": 2,
    "protect_analysis_context": true
  }
}
```

**Response:**
```json
{
  "messages": [...],            // compressed messages
  "tokens_before": 15000,
  "tokens_after": 3500,
  "tokens_saved": 11500,
  "compression_ratio": 0.23,    // tokens_after / tokens_before — LOWER is better
  "transforms_applied": ["router:smart_crusher:0.35"],
  "transforms_summary": {"router:smart_crusher:0.35": 1},
  "ccr_hashes": []              // non-empty only with mode="ccr"
}
```

**Headers:**
- `x-headroom-bypass: true` — skip compression, return messages as-is with zeroed metrics

**Error responses:** 400 (missing/invalid fields, bad `config.mode` or `config.frozen_message_count`), 401 (bad `HEADROOM_PROXY_TOKEN`), 404 (non-loopback without `HEADROOM_COMPRESS_ALLOW_REMOTE=1`), 503 (compression failed)

**Fail-open:** on timeout you get 200 with the original messages plus `compression_skipped: true` and `skip_reason: "compression_timeout"`.

**Multi-turn callers — don't lose the prefix cache.** This endpoint is stateless: unlike the proxy's own request path (which runs a CacheAligner and tracks provider cache hits across turns), it has no idea what the provider already cached.

The provider caches the bytes you *forwarded*, which compression already changed — so your originals and the cached prefix are no longer the same thing, and it is the forwarded version you must keep reproducing. Compression also varies with position: an older tool result can fall outside the recent-read protection window as the conversation grows and be compressed harder than last turn, so re-compression is not guaranteed to reproduce earlier output either. Two rules:

1. Pass `config.frozen_message_count` = the number of leading messages already cached upstream.
2. Send back the messages you **previously forwarded**, not the pristine originals. `frozen_message_count` returns leading messages exactly as passed in, so feeding it originals hands the provider different bytes than last turn and busts the cache anyway.

```python
forwarded = []


def next_turn(new_messages):
    r = requests.post(
        f"{proxy}/v1/compress",
        json={
            "messages": forwarded + new_messages,
            "model": "claude-sonnet-4-6",
            "config": {"frozen_message_count": len(forwarded)},
        },
    ).json()
    forwarded[:] = r["messages"]  # next turn's frozen prefix
    return forwarded
```

Note `protect_recent` is not a substitute — it guards the newest messages, while `frozen_message_count` guards the oldest, which is the cached end.

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

# Run with gunicorn — server.py has no module-level `app`; FastAPI is built
# by the create_app() factory, so gunicorn needs --factory
gunicorn headroom.proxy.server:create_app \
  --workers 4 \
  --bind 0.0.0.0:8787 \
  --worker-class uvicorn.workers.UvicornWorker \
  --factory
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
