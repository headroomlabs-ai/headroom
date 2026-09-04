# Environment Variables Reference

Full inventory of environment variables Headroom reads or writes, extracted
from source (`headroom/`) as of 2026-07-16. For the curated, user-facing
subset see `wiki/configuration.md` and `wiki/filesystem-contract.md` — this
file is the exhaustive superset, including internal/advanced knobs not
exposed in the Settings GUI. Descriptions for GUI-curated knobs are sourced
from `headroom/settings_store.py`; the rest are inferred from source context
(comments, `Knob`/env-constant definitions) and may be experimental or
undocumented elsewhere.

**Precedence** (all resources): explicit CLI/SDK argument >
`~/.headroom/settings.json` (GUI-managed subset) > shell-exported env var > code
default. A saved GUI setting therefore overrides a shell export. The one
exception is `manifest_managed` knobs (e.g. `HEADROOM_HOST`/`HEADROOM_PORT` on
supervised Docker/service installs): the install manifest owns their env var, so
there the exported value keeps precedence over `settings.json`.

---

## 1. Filesystem roots (two-root model)

See `wiki/filesystem-contract.md` for the full bucket table and precedence chain.

| Variable | Default | Purpose |
|---|---|---|
| `HEADROOM_CONFIG_DIR` | `~/.headroom/config` | Read-mostly config root (models.json, plugin settings). |
| `HEADROOM_WORKSPACE_DIR` | `~/.headroom` | Read-write state root (savings, logs, memory DB, telemetry, caches). |
| `HEADROOM_SAVINGS_PATH` | derived from workspace | Full path to the proxy savings JSON ledger. Always wins when set. |
| `HEADROOM_TOIN_PATH` | derived from workspace | Full path to the TOIN telemetry JSON file. |
| `HEADROOM_SUBSCRIPTION_STATE_PATH` | derived from workspace | Full path to the subscription tracker state file. |
| `HEADROOM_SETTINGS_PATH` | derived from workspace | Full path to `settings.json` (Settings GUI store). |
| `HEADROOM_MEMORY_DB_PATH` | `{cwd}/.headroom/memory.db` | Path to the legacy single-file memory SQLite DB. |
| `HEADROOM_MEMORY_PROJECT_ROOT` | cwd | Override the project root used for `--memory-storage=project`. |
| `HEADROOM_CCR_SQLITE_PATH` | derived | Path to the CCR (Compress-Cache-Retrieve) SQLite store. |
| `HEADROOM_MODEL_LIMITS` | — | Custom model context/pricing config: JSON string or file path. |
| `HEADROOM_WORKSPACE` | — | **Docker only**, host-side: dir to bind-mount as `/workspace`. Not the same as `HEADROOM_WORKSPACE_DIR`. |

---

## 2. Settings GUI knobs (curated, `~/.headroom/settings.json`)

Editable at `http://127.0.0.1:<port>/dashboard/settings` without hand-exporting
env vars. The GUI groups these onto feature pages in a left sidebar — General,
Output Shaping, Compression, CCR & Caching, Limits & Budget, Networking &
Security, Endpoints, Memory, Observability — each with a collapsed-by-default
Advanced section. Fields below are startup-captured (restart required to apply)
**except** the Output Shaping knobs (§3), which are GUI-editable and apply live
on Save. The Observability toggles (§10) are also GUI-editable now. Beyond the
tables in this section, the panel now also surfaces (all restart-required): the
Kompress engine backend, cross-turn dedup and tool-search toggles (§6); the CCR
storage backend, Redis URL and CCR TTL (§7); stateless/offline mode, strict-TLS,
and CORS/WebSocket origins (§5); the Vertex/Bedrock/Gemini/Cloud Code base URLs
(above); the Qdrant URL/host/port/API-key (§9); the proxy mode and the core
CLI toggles — optimization, semantic cache, and rate-limiting on/off, plus
tool-result interception (§2 Compression); and the embedding-server sidecar (§9).
`manifest_managed` fields are read-only on supervised Docker/service installs.

### Compression

| Variable | Type | Default | Description |
|---|---|---|---|
| `HEADROOM_MODE` | enum | `cache` | Proxy posture: `token` (compress; history may be rewritten for max savings) or `cache` (freeze prior turns for provider prefix-cache stability). |
| `HEADROOM_OPTIMIZE` | bool | `true` | Master optimization switch. `false` = passthrough (no compression); mirrors `--no-optimize`. |
| `HEADROOM_CACHE_ENABLED` | bool | `true` | Semantic response cache. `false` mirrors `--no-cache`. |
| `HEADROOM_RATE_LIMIT_ENABLED` | bool | `true` | Enforce RPM/TPM limits. `false` mirrors `--no-rate-limit`. |
| `HEADROOM_INTERCEPT_ENABLED` | bool | `false` | Enable ast-grep tool_result interceptors (Read outliner); mirrors `--intercept-tool-results`. Rollout-gated: a legacy alias for the `tool_result_interceptors` feature, so it also needs `HEADROOM_ROLLOUT_CHANNEL=canary` or higher. |
| `HEADROOM_SAVINGS_PROFILE` | enum | `coding` | Named compression posture: `agent-90`, `balanced`, `coding`, `general`. |
| `HEADROOM_TARGET_RATIO` | float 0-1 | adaptive | Kompress keep-ratio (lower = more aggressive). |
| `HEADROOM_DISABLE_KOMPRESS` | bool | `false` | Disable Kompress ML compression (structural compression stays on). |
| `HEADROOM_LOSSLESS` | bool | `false` | No-CCR lossless compaction; no retrieval marker emitted. |
| `HEADROOM_CODE_AWARE_ENABLED` | bool | `true` | AST-based code compression (requires `[code]` extra). |
| `HEADROOM_PROTECT_TOOL_RESULTS` | str (CSV) | — | Tool names whose results are never lossy-compressed. |
| `HEADROOM_NO_CCR` | bool | `false` | Disable CCR entirely (no markers, no injected retrieve tool). |
| `HEADROOM_NO_CCR_PROACTIVE_EXPANSION` | bool | `false` | Disable proactive expansion of previously compressed content. |
| `HEADROOM_COMPRESSION_MAX_WORKERS` | int | cpu_count | Bound the dedicated Kompress threadpool. |
| `HEADROOM_DISABLE_KOMPRESS_FALLBACK` | bool | `false` | With disable-kompress, route fallthrough to passthrough instead of Kompress fallback. |
| `HEADROOM_DISABLE_KOMPRESS_ANTHROPIC` | optional-bool | inherit | Disable/force-enable Kompress for the Anthropic pipeline only. |
| `HEADROOM_DISABLE_KOMPRESS_OPENAI` | optional-bool | inherit | Disable/force-enable Kompress for the OpenAI/Codex pipeline only. |

### Limits / Budget / Networking

| Variable | Type | Default | Description |
|---|---|---|---|
| `HEADROOM_RPM` | int | — | Max requests per minute. |
| `HEADROOM_TPM` | int | — | Max tokens per minute. |
| `HEADROOM_LIMIT_CONCURRENCY` | int | 1000 | Max concurrent connections before Uvicorn returns 503. |
| `HEADROOM_WORKERS` | int | 1 | Uvicorn worker processes. |
| `HEADROOM_BUDGET` | float | — | Budget limit per period; requests rejected 429 once reached. |
| `HEADROOM_BUDGET_PERIOD` | enum | `daily` | `hourly`, `daily`, or `monthly`. |
| `HEADROOM_HOST` | str | `127.0.0.1` | Bind host. Manifest-managed on docker/service installs. |
| `HEADROOM_PORT` | int | 8787 | Bind port. Manifest-managed on docker/service installs. |
| `HEADROOM_MAX_CONNECTIONS` | int | 500 | Max upstream HTTP connections in the shared httpx pool. |
| `HEADROOM_MAX_KEEPALIVE` | int | 100 | Max upstream keep-alive connections. |
| `HEADROOM_HTTP2` | bool | `true` | Use HTTP/2 for upstream provider connections. |
| `HEADROOM_HTTP_PROXY` | str | — | HTTP proxy URL for upstream provider requests only. |
| `HEADROOM_KEEPALIVE_EXPIRY` | float | 90.0 | Seconds an idle upstream keep-alive connection stays open. |

### Logging

| Variable | Type | Default | Description |
|---|---|---|---|
| `HEADROOM_LOG_MESSAGES` | bool | `false` | Log full request/response content. **Warning: may log sensitive data.** |
| `HEADROOM_LOG_FILE` | str | — | Path for the message log file. |

### Timeouts (retry/upstream)

| Variable | Type | Default | Description |
|---|---|---|---|
| `HEADROOM_RETRY_MAX_ATTEMPTS` | int | 3 | Max upstream retry attempts on connect/read/5xx failures. |
| `HEADROOM_RETRY_BASE_DELAY_MS` | int | 1000 | Initial upstream retry delay (ms). |
| `HEADROOM_RETRY_MAX_DELAY_MS` | int | 30000 | Max upstream retry delay (ms). |
| `HEADROOM_REQUEST_TIMEOUT` | int | 300 | Overall upstream request timeout (s). |
| `HEADROOM_CONNECT_TIMEOUT_SECONDS` | int | 10 | Upstream connection timeout (s). |
| `HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS` | int | 600 | Buffered Anthropic read timeout for non-streaming batch paths. |
| `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY` | int | max(2,min(8,cpu)) | Cap concurrent Anthropic pre-upstream work. |
| `HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS` | float | 15.0 | Fail-fast timeout waiting on the pre-upstream semaphore. |
| `HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS` | float | 2.0 | Fail-open timeout for memory-context lookup during pre-upstream. |

### CCR (experimental read-maturation)

| Variable | Type | Default | Description |
|---|---|---|---|
| `HEADROOM_READ_MATURATION` | bool | `false` | **Experimental**: hold fresh Reads out of compression until the file quiesces. |
| `HEADROOM_READ_MATURATION_QUIESCE_TURNS` | int | 5 | Turns a file must stay quiet before a held Read matures. |
| `HEADROOM_READ_MATURATION_MAX_HOLD_TURNS` | int | 25 | Force-mature a held Read after this many turns regardless. |
| `HEADROOM_READ_MATURATION_MIN_SIZE_BYTES` | int | 2048 | Only hold/mature Read outputs at least this many bytes. |

### Extensions / Backend / Memory

| Variable | Type | Default | Description |
|---|---|---|---|
| `HEADROOM_PROXY_EXTENSIONS` | CSV list | — | Opt-in proxy extension entry-point names (`*` = all discovered). |
| `HEADROOM_NO_SUBSCRIPTION_TRACKING` | bool | `false` | Disable the Anthropic Claude subscription usage poller. |
| `HEADROOM_SUBSCRIPTION_POLL_INTERVAL` | int | 300 | Seconds between subscription usage polls. |
| `HEADROOM_BACKEND` | str | `anthropic` | Upstream backend: `anthropic`, `bedrock`, `openrouter`, `anyllm`, `litellm-<provider>`. |
| `HEADROOM_ANYLLM_PROVIDER` | str | `openai` | Provider for the any-llm backend (openai, mistral, groq, ollama, ...). |
| `HEADROOM_REGION` | str | `us-west-2` | Cloud region for Bedrock/Vertex/etc backends. |
| `HEADROOM_NO_MEMORY_TOOLS` | bool | `false` | Disable auto-injection of `memory_save`/`memory_search` tools. |
| `HEADROOM_NO_MEMORY_CONTEXT` | bool | `false` | Disable auto-injection of relevant memories into the system prompt. |
| `HEADROOM_MEMORY_TOP_K` | int | 10 | Number of semantically-relevant memories retrieved. |
| `HEADROOM_MIN_EVIDENCE` | int | 5 | Times a pattern must be observed before persisting to memory. |

### Endpoints (custom upstream base URLs)

| Variable | Type | Description |
|---|---|---|
| `ANTHROPIC_TARGET_API_URL` | str | Custom Anthropic API base URL (Azure Foundry, corporate gateway). Overrides `https://api.anthropic.com`. |
| `OPENAI_TARGET_API_URL` | str | Custom OpenAI API base URL. Overrides `https://api.openai.com`. |
| `ANTHROPIC_TARGET_API_HEADERS` | secret JSON | Extra headers merged into (and overriding) forwarded Anthropic requests. |
| `OPENAI_TARGET_API_HEADERS` | secret JSON | Extra headers merged into (and overriding) forwarded OpenAI requests. |
| `VERTEX_TARGET_API_URL` | str | Custom Vertex AI regional API URL for publisher endpoints. |
| `BEDROCK_TARGET_API_URL` | str | Custom AWS Bedrock API URL. |
| `GEMINI_TARGET_API_URL` | str | Custom Gemini API base URL for passthrough endpoints. |
| `CLOUDCODE_TARGET_API_URL` | str | Custom Cloud Code Assist API base URL for compatibility endpoints. |

---

## 3. Live / hot-reload knobs (output shaping)

Read on every request via `headroom/proxy/runtime_env.py` — `headroom wrap`
pushes new values via `POST /admin/runtime-env` so a reused proxy picks them
up without a restart. These are also GUI-editable on the **Output Shaping**
page, where Save persists them to `settings.json` and applies them live (no
restart) through the same override store.

| Variable | Type | Description |
|---|---|---|
| `HEADROOM_OUTPUT_SHAPER` | bool | Master switch for output-token shaping. |
| `HEADROOM_VERBOSITY_LEVEL` | int 0-4 | Verbosity steering level (unset = learned/default). |
| `HEADROOM_EFFORT_ROUTER` | bool | Lower effort on mechanical tool-result continuations. |
| `HEADROOM_MECHANICAL_EFFORT` | str | Effort value used on mechanical continuations. |
| `HEADROOM_VERBOSITY_AUTOTUNE` | bool | Use the AIMD verbosity controller state. |
| `HEADROOM_OUTPUT_HOLDOUT` | float | Fraction of conversations held out for A/B measurement. |
| `HEADROOM_INTERCEPT_READ_MIN_CHARS` | int | Min tool-output chars before the ast-grep read rewrite. |

---

## 4. Session cache-hit stability

| Variable | Default | Description |
|---|---|---|
| `HEADROOM_BETA_HEADER_STICKY` | `enabled` | Per-session union of `anthropic-beta`/`OpenAI-Beta` tokens so a dropped token doesn't bust prefix cache. Set `disabled` to forward verbatim. |
| `HEADROOM_BETA_TRACKER_MAX_SESSIONS` | 1000 | LRU capacity of the in-memory session beta tracker. |
| `HEADROOM_TOOL_INJECTION_STICKY` | enabled | Sticky re-injection of memory tools across a session. Set `disabled` to bypass. |
| `HEADROOM_TOOL_TRACKER_MAX_SESSIONS` | — | LRU capacity of the session tool-injection tracker. |
| `HEADROOM_STRIP_INTERNAL_HEADERS` | enabled | Strip internal `x-headroom-*` headers before forwarding upstream. Set `disabled` to opt out. |

---

## 5. Networking / security

| Variable | Description |
|---|---|
| `HEADROOM_CORS_ORIGINS` | Allowed CORS origins (comma-separated). |
| `HEADROOM_WS_ORIGINS` | Allowed WebSocket origins; falls back to `HEADROOM_CORS_ORIGINS`. |
| `HEADROOM_TLS_STRICT` | Set `0` to relax OpenSSL 3.2+ `VERIFY_X509_STRICT` — needed behind corporate TLS-inspection proxies. |
| `HEADROOM_STATELESS` | `true`/`1`/`yes`/`on`: no filesystem writes (savings, logs). |
| `HEADROOM_OFFLINE` | Air-gap / no-egress master switch — disables update checks, model downloads, etc. |
| `HEADROOM_REQUIRE_RUST_CORE` | Gate whether the Rust core extension is mandatory at startup (`false` opts out). |
| `HEADROOM_PROXY_TOKEN` | Bearer token required to authenticate to the proxy (dashboard/admin routes). |
| `HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS` | CIDR allow-list for trusting `X-Forwarded-*` headers from a fronting gateway. |
| `HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS` | CIDR allow-list for dashboard-origin client trust. |
| `HEADROOM_PROXY_BODY_TOO_LARGE_STATUS` | HTTP status code returned for oversized request bodies (operator-configurable). |
| `HEADROOM_SSE_BUFFER_MAX_BYTES` | Max buffered bytes per SSE event (guards against pathological huge events). |
| `HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT` | Enforce pay-as-you-go compression-policy defaults when auth mode requires it. |
| `HEADROOM_WS_FAIL_OPEN_ON_COMPRESSION_FAILURE` | Let WS Responses traffic through unmodified on a compression failure instead of erroring. |
| `HEADROOM_WS_COMPRESSION_FAIL_THRESHOLD_BYTES` | Oversize threshold that triggers the WS compression fail-open path. |

---

## 6. Compression internals (Kompress engine tuning)

Selecting the ML backend: see `wiki/configuration.md` → *Kompress backend
selection* for the `HEADROOM_KOMPRESS_BACKEND` value table (`auto`, `onnx`,
`onnx_coreml`, `pytorch`, `pytorch_mps`, and shorthand aliases).

| Variable | Description |
|---|---|
| `HEADROOM_KOMPRESS_BACKEND` | Engine selection: onnx/pytorch/mps/auto (see table above). |
| `HEADROOM_KOMPRESS_ENDPOINT` | Remote Kompress inference endpoint URL (server-mode compression). |
| `HEADROOM_KOMPRESS_ENDPOINT_TOKEN` | Auth token for the remote Kompress endpoint. |
| `HEADROOM_KOMPRESS_MAX_TOKENS` | Max tokens routed through Kompress per call (default 50000). |
| `HEADROOM_KOMPRESS_BATCH_SIZE` | Inference batch size. |
| `HEADROOM_KOMPRESS_MUST_KEEP` | Content patterns Kompress must never drop. |
| `HEADROOM_KOMPRESS_ACQUIRE_TIMEOUT_SECONDS` | Timeout acquiring a Kompress worker slot. |
| `HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS` | Per-call Kompress execution timeout. |
| `HEADROOM_KOMPRESS_MAX_CONCURRENT` | Max concurrent Kompress inference calls. |
| `HEADROOM_KOMPRESS_TIME_BUDGET_SECONDS` | Overall time budget for a Kompress pass. |
| `HEADROOM_KOMPRESS_CANARY_SECONDS` | Canary/health-check interval for the Kompress backend. |
| `HEADROOM_KOMPRESS_ONNX_FILENAME` | Override the ONNX model filename loaded. |
| `HEADROOM_KOMPRESS_ONNX_INTER_THREADS` | ONNX Runtime inter-op thread count. |
| `HEADROOM_KOMPRESS_ONNX_INTRA_THREADS` | ONNX Runtime intra-op thread count. |
| `HEADROOM_KOMPRESS_COREML_CACHE_DIR` | Cache directory for the CoreML execution provider. |
| `HEADROOM_ONNX_CPU_ARENA` | Toggle ONNX Runtime CPU memory arena. |
| `HEADROOM_FORCE_KOMPRESS` / `HEADROOM_FORCE_KOMPRESS_ALL` | Force Kompress even below normal size/heuristic thresholds. |
| `HEADROOM_ACCURACY_GUARD` | Accuracy-guard mode/threshold gating aggressive compression. |
| `HEADROOM_LOSSLESS_ONLY` / `HEADROOM_LOSSLESS_THEN_LOSSY` | Compaction ordering: lossless-only, or lossless first then fall through to lossy. |
| `HEADROOM_LOSSY_MIN_EXTRA_SAVINGS` | Minimum extra savings required to justify a lossy step over lossless-only. |
| `HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO` | Experimental keep-ratio override specifically for Read tool output. |
| `HEADROOM_FREEZE_BLOCK_DECISION` | Debug/test override to freeze a specific compression block decision. |
| `HEADROOM_MIN_CHARS_FOR_BLOCK` | Minimum characters a content block must have to be eligible for compression. |
| `HEADROOM_SYSTEM_COMPACT` / `HEADROOM_SYSTEM_COMPACT_MIN_CHARS` | Enable/threshold for system-prompt compaction. |
| `HEADROOM_TOOL_DESC_MAX_CHARS` / `HEADROOM_TOOL_DESC_STRIP_SEMANTIC` | Truncate/strip verbose tool schema descriptions before forwarding. |
| `HEADROOM_TEXT_CRUSHER` | Toggle/select the plain-text crusher transform. |
| `HEADROOM_DETECT_BACKEND` | Content-type detection backend selection (structural router). |
| `HEADROOM_COMPACTION_FORMAT` | Output format for SmartCrusher JSON compaction (default `csv-schema`). |
| `HEADROOM_COMPRESSION_DEADLINE_MS` | Hard deadline (ms) for a single compression pass before bail-out. |
| `HEADROOM_COMPRESSION_TIMEOUT_SECONDS` | Overall compression pipeline timeout. |
| `HEADROOM_COMPRESS_WORKERS` | Worker count for the content-router compression pool. |
| `HEADROOM_BACKGROUND_COMPRESSION` / `HEADROOM_BACKGROUND_COMPRESSION_MIN_TOKENS` | Enable/threshold for async background (post-response) compression. |
| `HEADROOM_DEDUPE` | Enable cross-turn duplicate-content dedup. |
| `HEADROOM_COLD_RECOMPACT` | Recompact the whole prefix when the provider prompt cache has lapsed (confirmed-cold turns only, so nothing warm is busted). |
| `HEADROOM_NET_COST_POLICY` | `1` unlocks the frozen floor: messages already inside the provider's cached prefix may be rewritten when the token saving beats the cache-rewrite penalty. Without it, savings stop growing once the prefix goes warm. |
| `HEADROOM_TOOL_SEARCH` | Enable tool-search injection (replace large tool lists with a search tool). |
| `HEADROOM_PIPELINE_BREAKER_THRESHOLD` / `HEADROOM_PIPELINE_BREAKER_COOLDOWN_S` | Circuit-breaker: consecutive transform failures before tripping, and cooldown before retry. |
| `HEADROOM_PREFER_CODE_AWARE_FOR_CODE` | Prefer AST code-aware compression over Kompress for detected code content. |
| `HEADROOM_MIN_TOKENS` / `HEADROOM_MAX_ITEMS` / `HEADROOM_SMART_CRUSHER_COMPACTION` | SmartCrusher tuning: min tokens to compress, max items kept, compaction mode (mirrors SDK `SmartCrusherConfig`). |
| `HEADROOM_PROTECT_READS` / `HEADROOM_PROTECT_RECENT` / `HEADROOM_PROTECT_ANALYSIS_CONTEXT` | Protect Read tool output / N most-recent turns / active-analysis context from compression. |
| `HEADROOM_COMPRESS_USER_MESSAGES` / `HEADROOM_COMPRESS_SYSTEM_MESSAGES` | Whether user / system messages are eligible for compression (savings-profile flags). |

---

## 7. CCR (Compress-Cache-Retrieve) backend

| Variable | Description |
|---|---|
| `HEADROOM_CCR_BACKEND` | Storage backend for CCR: sqlite (default) or redis. |
| `HEADROOM_CCR_SQLITE_PATH` | SQLite file path when backend is sqlite. |
| `HEADROOM_REDIS_URL` | Redis connection URL when backend is redis. |
| `HEADROOM_CCR_TENANT_PREFIX` | Key-namespace prefix for multi-tenant CCR storage. |
| `HEADROOM_CCR_TTL_SECONDS` | TTL for CCR-retrievable compressed content. |

---

## 8. Model routing / model config

| Variable | Description |
|---|---|
| `HEADROOM_MODEL_LIMITS` | Custom model context-limit/pricing overrides (JSON string or file path). See `wiki/configuration.md`. |
| `HEADROOM_MODEL_ROUTER_ENABLED` | Enable the model router (route requests to different models by policy). |
| `HEADROOM_MODEL_ROUTES` | JSON array of routing rules consumed by the model router. |
| `HEADROOM_TECHNIQUE_ROUTER` | HF model id for the technique-router ML model (default `chopratejas/technique-router`). |
| `HEADROOM_SENTENCE_TRANSFORMER` | HF model id for the embedding model (default `all-MiniLM-L6-v2`). |
| `HEADROOM_SIGLIP` | HF model id for the vision model (default `google/siglip-base-patch16-224`). |
| `HEADROOM_SPACY` | spaCy model name (default `en_core_web_sm`). |
| `HEADROOM_HF_PIN` | Pin a specific HuggingFace model revision/commit. |
| `HEADROOM_BEDROCK_MODEL_MAP` | JSON mapping of model names → Bedrock model IDs. |
| `HEADROOM_OPENAI_TOOL_SEARCH_MODELS` | Comma-separated OpenAI model names eligible for tool-search injection. |

---

## 9. Memory system (embeddings + Qdrant)

| Variable | Description |
|---|---|
| `HEADROOM_EMBEDDER_RUNTIME` | Set `pytorch_mps` to run the memory embedder on Apple GPU (requires `[pytorch-mps]` extra, MPS-availability-gated). |
| `HEADROOM_EMBEDDING_SERVER_SOCKET` | Unix socket path for the out-of-process embedding server. |
| `HEADROOM_EMBEDDING_SERVER` | Run a shared out-of-process embedding server sidecar across workers (saves ~600 MB RSS). GUI-editable (§2 Memory). |
| `HEADROOM_EMBED_CONCURRENCY` | Max concurrent embedding calls. |
| `HEADROOM_EMBED_NUM_THREADS` | Thread count for the embedding backend. |
| `HEADROOM_QDRANT_URL` | Full Qdrant URL (e.g. hosted Qdrant Cloud). |
| `HEADROOM_QDRANT_HOST` | Qdrant hostname (default `localhost`). |
| `HEADROOM_QDRANT_PORT` | Qdrant HTTP port (default `6333`). |
| `HEADROOM_QDRANT_API_KEY` | API key for hosted Qdrant. |
| `HEADROOM_QDRANT_HTTPS` | Force HTTPS on/off for the Qdrant connection. |
| `HEADROOM_QDRANT_PREFER_GRPC` | Use gRPC instead of HTTP for Qdrant. |
| `HEADROOM_QDRANT_GRPC_PORT` | Qdrant gRPC port (default `6334`). |

---

## 10. Observability / telemetry

The curated subset below — `HEADROOM_OTEL_METRICS_ENABLED`,
`HEADROOM_OTEL_METRICS_ENDPOINT`, `HEADROOM_OTEL_SERVICE_NAME`,
`HEADROOM_LANGFUSE_ENABLED`, `HEADROOM_TELEMETRY`, `HEADROOM_PERIODIC_TOIN_STATS`
— is GUI-editable on the **Observability** page (restart required). Langfuse/OTel
keys and header maps stay env-only and are never exposed in the GUI.

### OpenTelemetry metrics

| Variable | Default | Description |
|---|---|---|
| `HEADROOM_OTEL_METRICS_ENABLED` | — | Enable OTel metrics export. |
| `HEADROOM_OTEL_METRICS_ENDPOINT` | — | OTLP collector endpoint. |
| `HEADROOM_OTEL_METRICS_EXPORTER` | `otlp_http` | Exporter type. |
| `HEADROOM_OTEL_METRICS_EXPORT_INTERVAL_MS` | — | Export interval (ms). |
| `HEADROOM_OTEL_METRICS_HEADERS` | — | Extra headers sent with OTLP exports. |
| `HEADROOM_OTEL_RESOURCE_ATTRIBUTES` | — | Extra OTel resource attributes. |
| `HEADROOM_OTEL_SERVICE_NAME` | derived | Service name reported to OTel. |

### Langfuse tracing

| Variable | Description |
|---|---|
| `HEADROOM_LANGFUSE_ENABLED` | Enable Langfuse trace export. |
| `HEADROOM_LANGFUSE_RESOURCE_ATTRIBUTES` | Extra resource attributes attached to Langfuse traces. |
| `HEADROOM_LANGFUSE_SERVICE_NAME` | Service name reported to Langfuse. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse project credentials (third-party, standard Langfuse SDK vars). |
| `LANGFUSE_BASE_URL` / `LANGFUSE_OTEL_HOST` | Langfuse server endpoint (self-hosted or cloud). |
| `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2` | LangSmith/LangChain tracing passthrough (third-party). |

### TOIN (token-in telemetry) and usage tracking

| Variable | Description |
|---|---|
| `HEADROOM_TOIN_BACKEND` | Storage backend for TOIN telemetry. |
| `HEADROOM_TOIN_PATH` | Full path to the TOIN JSON file (see §1). |
| `HEADROOM_TOIN_TENANT_PREFIX` | Key-namespace prefix for multi-tenant TOIN storage. |
| `HEADROOM_TOIN_URL` | Remote TOIN aggregation endpoint. |
| `HEADROOM_PERIODIC_TOIN_STATS` | Enable periodic TOIN stats logging (default on; set `0` to disable). |
| `HEADROOM_NET_COST_CACHE_TTL_SECONDS` | Cache TTL for the net-cost (savings-vs-overhead) estimator. |
| `HEADROOM_NET_COST_EXPECTED_READS` | Expected-reads parameter feeding the net-cost model. |
| `HEADROOM_NET_COST_P_ALIVE` | Probability-of-reuse parameter for the net-cost model. |

### Anonymous usage beacon

| Variable | Description |
|---|---|
| `HEADROOM_TELEMETRY` | Opt in/out of the anonymous usage telemetry beacon. |
| `HEADROOM_TELEMETRY_DISABLED` | Hard-disable the telemetry beacon. |
| `HEADROOM_TELEMETRY_WARN` | Control the one-time telemetry notice (default `on`). |

---

## 11. Deployment / install detection

| Variable | Description |
|---|---|
| `HEADROOM_DEPLOYMENT_PRESET` | Named deployment preset applied at install/startup. |
| `HEADROOM_DEPLOYMENT_PROFILE` | Active persistent-install deployment profile name. |
| `HEADROOM_DEPLOYMENT_RUNTIME` | Detected runtime: foreground, service, docker, task. |
| `HEADROOM_DEPLOYMENT_SCOPE` | Install scope (user vs system). |
| `HEADROOM_DEPLOYMENT_SUPERVISOR` | Detected process supervisor (launchd/systemd/Task Scheduler). |
| `HEADROOM_IN_DOCKER` | Set when running inside the official Docker image. |
| `HEADROOM_DOCKER_GPUS` | GPU passthrough config for the Docker install. |
| `HEADROOM_UPDATE_CHECK` | Set `off` to disable the update-check ping. |
| `HEADROOM_LICENSE_KEY` | License key unlocking paid features. |
| `HEADROOM_MARKETPLACE_SOURCE` | Override marketplace source used by `headroom init`. |
| `HEADROOM_BINARIES_CACHE` | Cache directory for downloaded native binaries. |
| `HEADROOM_BINARIES_MIRROR` | Alternate mirror URL for binary downloads. |
| `HEADROOM_BINARIES_OFFLINE` | Skip binary downloads entirely (air-gapped installs). |

---

## 12. Timeouts (startup / cold-start / ML init)

| Variable | Description |
|---|---|
| `HEADROOM_COLD_START_FAST_PASS_TIMEOUT_SECONDS` | Timeout for the fast structural-only pass before the ML stack warms up. |
| `HEADROOM_EAGER_PRELOAD_TIMEOUT_SECONDS` | Timeout for eager ML model preloading at startup. |
| `HEADROOM_CONTEXT_TOOL_STATS_TTL_SECONDS` | TTL for cached context-tool usage stats. |
| `HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS` | Timeout loading the tiktoken tokenizer. |
| `HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS` | Timeout loading a HuggingFace tokenizer. |
| `HEADROOM_MAGIKA_INIT_TIMEOUT_SECS` | Timeout initializing the Magika content-type detector. |
| `HEADROOM_DETECT_TIMEOUT_SECS` | Timeout for content-type/backend auto-detection. |
| `HEADROOM_WRAP_PROXY_TIMEOUT` | Timeout `headroom wrap` waits for the proxy to become ready. |
| `HEADROOM_LEARN_CLI_TIMEOUT_SECS` / `HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS` | Overall / idle timeout for `headroom learn` CLI runs. |

---

## 13. `headroom wrap` / CLI provider integration

| Variable | Description |
|---|---|
| `HEADROOM_MODE` | Proxy posture: `token` (prioritize compression) or `cache` (prioritize prefix-cache stability); aliases like `cost_savings`/`token_mode` are normalized. Read at startup, so also editable in the Settings GUI (§2 Compression). |
| `HEADROOM_PROXY_URL` | URL of the proxy a wrapped CLI/agent should target. |
| `HEADROOM_AGENT_TYPE` | Detected/declared wrapped-agent type (claude, codex, copilot, ...), used for telemetry. |
| `HEADROOM_STACK` | Declared tech stack context for telemetry. |
| `HEADROOM_USER_ID` | Stable user identifier for telemetry/savings attribution. |
| `HEADROOM_EXCLUDE_TOOLS` | Comma-separated tool names excluded from compression/injection. |
| `HEADROOM_TOOL_PROFILES` | Per-tool compression profile overrides (JSON). |
| `HEADROOM_CONTEXT_TOOL` | Selected context tool integration, e.g. `lean-ctx`. |
| `HEADROOM_COPILOT_AUTH_FILE` | Path to the GitHub Copilot auth token file. |
| `HEADROOM_CODEX_WIRE_DEBUG` / `HEADROOM_CODEX_WIRE_DEBUG_DIR` | Dump raw Codex wire traffic for debugging, and target directory. |
| `HEADROOM_CODEX_USAGE_URL` | Codex/ChatGPT usage endpoint override (default `https://chatgpt.com/backend-api/wham/usage`). |
| `HEADROOM_CODEX_UPSTREAM_BASE_URL` | Override the Codex upstream base URL. |
| `HEADROOM_OPENCODE_PLUGIN_PATH` | Path to the OpenCode transport plugin `headroom wrap opencode` installs. |
| `HEADROOM_PROXY_CONFIG_JSON` | Inline JSON proxy config passed from `wrap` to the launched proxy. |
| `HEADROOM_CC_SWITCH_RECONCILE` / `HEADROOM_CC_SWITCH_ROUTE_OFFICIAL` | `cc-switch` integration: reconcile config on start / route official traffic. |
| `HEADROOM_SKIP_UPSTREAM_CHECK` | Skip the upstream-reachability probe before wrapping. |
| `HEADROOM_LEAN_CTX_TARGET` | Target directory for the `lean-ctx` MCP installer. |
| `HEADROOM_RTK_TARGET` | Target directory for the `rtk` (tokensave/graph) binary installer. |
| `HEADROOM_RTK_GAIN_SCOPE` / `HEADROOM_RTK_POLL_LOCK` / `HEADROOM_RTK_WIRING` | RTK loop-learning integration knobs (scope, poll lock, wiring mode). |
| `HEADROOM_LEARN_CLI` | Marks that the current process is `headroom learn` (used to adjust logging/paths). |
| `HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED` | Allow an unverified/unsigned tokensave binary (bypasses signature check). |
| `HEADROOM_TOKENSAVE_VERSION` | Pin a specific tokensave graph-tool version. |
| `HEADROOM_MCP_CLIENT` / `HEADROOM_MCP_MODEL` | Declared MCP client name / model for the CCR MCP server. |
| `HEADROOM_MCP_READ` | Toggle the CCR MCP server's retrieve-by-read mode (default `off`). |

---

## 14. Misc / debug

| Variable | Description |
|---|---|
| `HEADROOM_DEBUG_DUMP` | Dump intermediate compression state for debugging. |
| `HEADROOM_PROBE_RECORD_DIR` | Directory to record probe/replay fixtures. |
| `HEADROOM_VERSION` / `HEADROOM_BUILD_VERSION` | Version strings baked in at Docker build time (not user-facing). |

---

## 15. Third-party credentials & endpoints Headroom reads

These are not `HEADROOM_*` but are read directly by providers/backends/wrap
integrations.

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI provider / backend. |
| `OPENROUTER_API_KEY` | OpenRouter backend. |
| `GOOGLE_API_KEY` | Google/Gemini subscription detection. |
| `AWS_PROFILE` / `AWS_SESSION_TOKEN` | Bedrock backend (standard AWS SDK credential resolution). |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code OAuth passthrough (`headroom wrap claude`). |
| `CLAUDE_CODE_USE_VERTEX` / `ANTHROPIC_VERTEX_PROJECT_ID` / `ANTHROPIC_VERTEX_BASE_URL` | Claude Code → Vertex AI routing detection/override. |
| `CLAUDE_CODE_USE_FOUNDRY` / `ANTHROPIC_FOUNDRY_RESOURCE` / `ANTHROPIC_FOUNDRY_BASE_URL` | Claude Code → Azure AI Foundry routing detection/override. |
| `CLAUDE_CODE_USE_BEDROCK` | Claude Code → Bedrock routing detection. |
| `NEO4J_AUTH` | Neo4j credentials for the tokensave/graph docker-compose stack (see `.env.example`). |
| `GITHUB_COPILOT_API_TOKEN`, `GITHUB_COPILOT_API_URL`, `GITHUB_COPILOT_ENTERPRISE_DOMAIN`, `GITHUB_COPILOT_ENTERPRISE_URL`, `GITHUB_COPILOT_HOST`, `GITHUB_COPILOT_SECRET_TOOL`, `GITHUB_COPILOT_TOKEN_EXCHANGE_URL`, `GITHUB_COPILOT_TOKEN_FILE`, `GITHUB_COPILOT_USER_AGENT`, `GITHUB_COPILOT_USER_INFO_URL`, `GITHUB_COPILOT_USE_TOKEN_EXCHANGE` | GitHub Copilot backend integration (`headroom wrap copilot`). |
| `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `OPENCODE_HOME`, `OPENCODE_CONFIG`, `GROK_HOME`, `COPILOT_HOME`, `PI_CODING_AGENT_DIR` | Third-party CLI agent config directories `headroom wrap` detects/reads (not Headroom's own settings). |

---

## 16. TypeScript SDK

| Variable | Default | Description |
|---|---|---|
| `HEADROOM_BASE_URL` | `http://localhost:8787` | Base URL of the Headroom proxy. |
| `HEADROOM_API_KEY` | — | Optional API key for authenticated Headroom endpoints (also read by the Python ASGI/litellm callback integrations). |

---

## Notes

- Boolean env vars generally accept `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off` (case-insensitive), per-reader — check the specific module if precision matters.
- Vars listed as "experimental" or "internal" in source comments may change or be removed without the deprecation notice given to documented ones.
- For the values table of `HEADROOM_KOMPRESS_BACKEND` and the full filesystem precedence chain, see `wiki/configuration.md` and `wiki/filesystem-contract.md` rather than duplicating them here.
