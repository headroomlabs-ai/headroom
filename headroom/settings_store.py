"""File-backed store for a curated subset of Headroom's runtime knobs (mostly ``HEADROOM_*``,

The dashboard settings GUI persists these knobs to ``settings.json`` in the
workspace dir and this module applies them to ``os.environ`` at CLI startup —
the stored value wins over a shell export (``os.environ[...] = value``), so the
GUI is authoritative. Precedence: ``CLI arg > settings.json > export > code
default``. The exception is ``manifest_managed`` knobs, whose env is owned by the
install manifest: there we ``setdefault`` so the export/manifest keeps priority.

Deliberately dependency-light (stdlib + ``headroom.paths`` only, no FastAPI or
proxy imports) so the early CLI apply hook — which must run
before Click parses ``envvar=`` options — stays cheap and import-safe.

``load()`` is fail-open: a corrupt or unreadable ``settings.json`` yields
``{}`` (defaults) rather than raising, so it can never crash-loop the proxy on
startup. ``save()`` writes atomically (temp file + ``os.replace``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from headroom import paths

logger = logging.getLogger(__name__)

_MASK = "••••••"  # ●●●●●● for masked secret values


@dataclass(frozen=True)
class SettingField:
    """One curated, GUI-editable env knob.

    ``env`` is the ``HEADROOM_*`` variable the knob maps to; ``key`` is the
    JSON/API key. ``type`` drives coercion, validation and the UI control.
    ``manifest_managed`` marks a knob baked into the install manifest on
    supervised (docker/service) deploys — ``settings.json`` cannot change it
    there, so the UI renders it read-only.
    """

    env: str
    key: str
    label: str
    group: str
    type: str  # "bool" | "int" | "float" | "str" | "enum" | "optional-bool" | "csv-list"
    default: Any = None
    choices: tuple[str, ...] = ()
    help: str = ""
    secret: bool = False
    manifest_managed: bool = False
    minimum: float | None = None
    maximum: float | None = None
    tier: str = "advanced"  # "basic" | "advanced" — Normal vs Advanced within a page

    @property
    def page(self) -> str:
        """Sidebar nav category, derived from ``group`` via ``_GROUP_TO_PAGE``."""
        return _GROUP_TO_PAGE.get(self.group, "General")

    @property
    def live(self) -> bool:
        """True when the knob applies via ``runtime_env`` with no restart."""
        return self.page == "Output Shaping"


# Sidebar nav order for the settings GUI. Each ``group`` maps to exactly one
# page; several groups share a page (e.g. Limits + Budget). This tuple is the
# single source of truth for page order — the UI renders pages in this sequence.
PAGES: tuple[str, ...] = (
    "General",
    "Output Shaping",
    "Compression",
    "CCR & Caching",
    "Limits & Budget",
    "Networking & Security",
    "Endpoints",
    "Memory",
    "Observability",
)

# Each field's ``group`` (its in-page sub-section header) maps to one nav page.
_GROUP_TO_PAGE: dict[str, str] = {
    "Backend": "General",
    "Extensions": "General",
    "Output Shaping": "Output Shaping",
    "Compression": "Compression",
    "CCR": "CCR & Caching",
    "Limits": "Limits & Budget",
    "Budget": "Limits & Budget",
    "Networking": "Networking & Security",
    "Timeouts": "Networking & Security",
    "Endpoints": "Endpoints",
    "Memory": "Memory",
    "Logging": "Observability",
    "Observability": "Observability",
}
# Curated registry. Env formats verified against each knob's Click option in
# headroom/cli/proxy.py (bools serialize to "1"/"0", which Click's BOOL type and
# the body-resolved HEADROOM_CODE_AWARE_ENABLED reader both accept).
SETTINGS: tuple[SettingField, ...] = (
    # --- Compression ---
    SettingField(
        "HEADROOM_MODE",
        "mode",
        "Proxy mode",
        "Compression",
        "enum",
        default="cache",
        choices=("token", "cache"),
        help="Proxy posture: token prioritizes compression (history may be rewritten for max "
        "savings); cache prioritizes provider prefix-cache stability (prior turns frozen).",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_SAVINGS_PROFILE",
        "savings_profile",
        "Savings profile",
        "Compression",
        "enum",
        default="coding",
        choices=("agent-90", "balanced", "coding", "general"),
        help="Named compression posture applied at startup.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_TARGET_RATIO",
        "target_ratio",
        "Target keep-ratio",
        "Compression",
        "float",
        default=None,
        minimum=0.0,
        maximum=1.0,
        help="Kompress keep-ratio 0-1 (lower = more aggressive). Unset = adaptive.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_DISABLE_KOMPRESS",
        "disable_kompress",
        "Disable Kompress",
        "Compression",
        "bool",
        default=False,
        help="Disable Kompress ML compression (structural compression stays on).",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_LOSSLESS",
        "lossless",
        "Lossless mode",
        "Compression",
        "bool",
        default=False,
        help="No-CCR lossless compaction; no retrieval marker emitted.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_CODE_AWARE_ENABLED",
        "code_aware_enabled",
        "Code-aware compression",
        "Compression",
        "bool",
        default=True,
        help="AST-based code compression (requires the [code] extra).",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_PROTECT_TOOL_RESULTS",
        "protect_tool_results",
        "Protect tool results",
        "Compression",
        "str",
        default=None,
        help="Comma-separated tool names whose results are never lossy-compressed.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_NO_CCR",
        "no_ccr",
        "Disable CCR",
        "CCR",
        "bool",
        default=False,
        help="Disable CCR entirely (no markers, no injected retrieve tool).",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_CCR_INLINE_RESOLVE",
        "ccr_inline_resolve",
        "Resolve CCR markers inline",
        "Compression",
        "bool",
        default=False,
        help=(
            "Resolve <<ccr:...>> markers on the response path instead of "
            "relying on headroom_retrieve tool calls. For callers with no "
            "tool-call round-trip (e.g. a LiteLLM guardrail/proxy hop). "
            "Non-streaming responses only."
        ),
        tier="advanced",
    ),
    # --- Limits ---
    SettingField(
        "HEADROOM_RPM",
        "rpm",
        "Requests / min",
        "Limits",
        "int",
        default=None,
        minimum=1,
        help="Max requests per minute.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_TPM",
        "tpm",
        "Tokens / min",
        "Limits",
        "int",
        default=None,
        minimum=1,
        help="Max tokens per minute.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_LIMIT_CONCURRENCY",
        "limit_concurrency",
        "Concurrency limit",
        "Limits",
        "int",
        default=1000,
        minimum=1,
        help="Max concurrent connections before Uvicorn returns 503.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_WORKERS",
        "workers",
        "Worker processes",
        "Limits",
        "int",
        default=1,
        minimum=1,
        help="Uvicorn worker processes.",
        tier="advanced",
    ),
    # --- Budget ---
    SettingField(
        "HEADROOM_BUDGET",
        "budget",
        "Budget (USD)",
        "Budget",
        "float",
        default=None,
        minimum=0.0,
        help="Budget limit per period; requests are rejected with 429 once reached.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_BUDGET_PERIOD",
        "budget_period",
        "Budget period",
        "Budget",
        "enum",
        default="daily",
        choices=("hourly", "daily", "monthly"),
        help="Period the budget applies to.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_BUDGET_ESTIMATED_BASIS",
        "budget_estimated_basis",
        "Estimated-basis spend",
        "Budget",
        "enum",
        default="count",
        choices=("count", "ignore", "block"),
        help=(
            "What spend booked from Headroom's own token estimate does to the budget "
            "when a provider response carries no input-token breakdown. count: it "
            "consumes the budget. ignore: only provider-reported spend does. block: "
            "refuse requests rather than enforce a hard limit on an estimate."
        ),
        tier="advanced",
    ),
    # --- Networking (baked into the install manifest on supervised deploys) ---
    SettingField(
        "HEADROOM_HOST",
        "host",
        "Host",
        "Networking",
        "str",
        default="127.0.0.1",
        manifest_managed=True,
        help="Bind host. Managed by the install manifest on docker/service installs.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_PORT",
        "port",
        "Port",
        "Networking",
        "int",
        default=8787,
        minimum=1,
        maximum=65535,
        manifest_managed=True,
        help="Bind port. Managed by the install manifest on docker/service installs.",
        tier="advanced",
    ),
    # --- Logging ---
    SettingField(
        "HEADROOM_LOG_MESSAGES",
        "log_messages",
        "Log message content",
        "Logging",
        "bool",
        default=False,
        help="Log full request/response content. WARNING: may log sensitive data.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_LOG_FILE",
        "log_file",
        "Log file path",
        "Logging",
        "str",
        default=None,
        help="Path for the message log file.",
        tier="basic",
    ),
    # --- Observability (metrics / tracing / telemetry; restart-required) -----
    SettingField(
        "HEADROOM_OTEL_METRICS_ENABLED",
        "otel_metrics_enabled",
        "OpenTelemetry metrics",
        "Observability",
        "bool",
        default=False,
        help="Export OpenTelemetry metrics (requires the [otel] extra).",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_OTEL_METRICS_ENDPOINT",
        "otel_metrics_endpoint",
        "OTel metrics endpoint",
        "Observability",
        "str",
        default=None,
        help="OTLP metrics endpoint URL (e.g. http://localhost:4318/v1/metrics).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_OTEL_SERVICE_NAME",
        "otel_service_name",
        "OTel service name",
        "Observability",
        "str",
        default=None,
        help="service.name resource attribute for exported telemetry.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_LANGFUSE_ENABLED",
        "langfuse_enabled",
        "Langfuse tracing",
        "Observability",
        "bool",
        default=False,
        help="Send LLM traces to Langfuse (LANGFUSE_* keys stay env-only).",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_TELEMETRY",
        "telemetry",
        "Anonymous usage telemetry",
        "Observability",
        "optional-bool",
        default=None,
        help="Opt in/out of the anonymous usage beacon. Unset = off (opt-in).",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_PERIODIC_TOIN_STATS",
        "periodic_toin_stats",
        "Periodic TOIN stats",
        "Observability",
        "bool",
        default=True,
        help="Log periodic tokens-out/-in efficiency stats.",
        tier="advanced",
    ),
    # --- Networking (upstream connection pool tuning) ---
    SettingField(
        "HEADROOM_MAX_CONNECTIONS",
        "max_connections",
        "Max upstream connections",
        "Networking",
        "int",
        default=500,
        minimum=1,
        help="Maximum upstream HTTP connections in the shared httpx pool.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_MAX_KEEPALIVE",
        "max_keepalive_connections",
        "Max keep-alive connections",
        "Networking",
        "int",
        default=100,
        minimum=0,
        help="Maximum upstream keep-alive connections in the shared httpx pool.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_HTTP2",
        "http2",
        "HTTP/2 upstream",
        "Networking",
        "bool",
        default=True,
        help="Use HTTP/2 for upstream provider connections.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_HTTP_PROXY",
        "http_proxy",
        "Outbound HTTP proxy",
        "Networking",
        "str",
        default=None,
        help="HTTP proxy URL for upstream provider requests only (HTTPS uses CONNECT).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_KEEPALIVE_EXPIRY",
        "keepalive_expiry",
        "Keep-alive expiry (s)",
        "Networking",
        "float",
        default=90.0,
        minimum=0.0,
        help="Seconds an idle upstream keep-alive connection is kept open.",
        tier="advanced",
    ),
    # --- Compression (additional internals) ---
    SettingField(
        "HEADROOM_NO_CCR_PROACTIVE_EXPANSION",
        "no_ccr_proactive_expansion",
        "Disable CCR proactive expansion",
        "CCR",
        "bool",
        default=False,
        help="Disable proactive expansion of previously compressed content.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_COMPRESSION_MAX_WORKERS",
        "compression_max_workers",
        "Compression worker pool size",
        "Compression",
        "int",
        default=None,
        help="Bound the dedicated compression threadpool (CPU-bound Kompress work). Unset = cpu_count.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_DISABLE_KOMPRESS_FALLBACK",
        "disable_kompress_fallback",
        "Disable Kompress fallback",
        "Compression",
        "bool",
        default=False,
        help="With disable-kompress, route fall-through content to passthrough instead of the Kompress fallback.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_DISABLE_KOMPRESS_ANTHROPIC",
        "disable_kompress_anthropic",
        "Disable Kompress (Anthropic)",
        "Compression",
        "optional-bool",
        default=None,
        help="Disable (false) or force-enable (true) Kompress for the Anthropic pipeline only. Unset = inherit.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_DISABLE_KOMPRESS_OPENAI",
        "disable_kompress_openai",
        "Disable Kompress (OpenAI)",
        "Compression",
        "optional-bool",
        default=None,
        help="Disable (false) or force-enable (true) Kompress for the OpenAI/Codex pipeline only. Unset = inherit.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_COMPRESSORS",
        "compressors",
        "Enabled compressors",
        "Compression",
        "csv-list",
        default=None,
        help="Comma-separated opt-in built-in compressor names ('*' enables all built-ins).",
        tier="advanced",
    ),
    # --- CCR (experimental read-maturation) ---
    SettingField(
        "HEADROOM_READ_MATURATION",
        "read_maturation",
        "Read maturation",
        "CCR",
        "bool",
        default=False,
        help="EXPERIMENTAL: hold fresh Reads out of compression until the file quiesces.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_READ_MATURATION_QUIESCE_TURNS",
        "read_maturation_quiesce_turns",
        "Maturation quiesce turns",
        "CCR",
        "int",
        default=5,
        minimum=1,
        help="Turns a file must stay quiet before a held Read is matured.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_READ_MATURATION_MAX_HOLD_TURNS",
        "read_maturation_max_hold_turns",
        "Maturation max hold turns",
        "CCR",
        "int",
        default=25,
        minimum=1,
        help="Force-mature a held Read after this many turns even if the file stays active.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_READ_MATURATION_MIN_SIZE_BYTES",
        "read_maturation_min_size_bytes",
        "Maturation min size (bytes)",
        "CCR",
        "int",
        default=2048,
        minimum=0,
        help="Only hold/mature Read outputs at least this many bytes.",
        tier="advanced",
    ),
    # --- Extensions ---
    SettingField(
        "HEADROOM_PROXY_EXTENSIONS",
        "proxy_extensions",
        "Enabled proxy extensions",
        "Extensions",
        "csv-list",
        default=None,
        help="Comma-separated opt-in proxy extension entry-point names ('*' enables all discovered).",
        tier="advanced",
    ),
    # --- Backend ---
    SettingField(
        "HEADROOM_NO_SUBSCRIPTION_TRACKING",
        "no_subscription_tracking",
        "Disable subscription tracking",
        "Backend",
        "bool",
        default=False,
        help="Disable the Anthropic Claude subscription usage poller (GET /api/oauth/usage).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_SUBSCRIPTION_POLL_INTERVAL",
        "subscription_poll_interval",
        "Subscription poll interval (s)",
        "Backend",
        "int",
        default=None,
        minimum=1,
        maximum=3600,
        help="Seconds between Anthropic subscription usage polls. Default: 300.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_BACKEND",
        "backend",
        "Upstream backend",
        "Backend",
        "str",
        default="anthropic",
        help="API backend: anthropic, bedrock, openrouter, anyllm, or litellm-<provider>.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_ANYLLM_PROVIDER",
        "anyllm_provider",
        "any-llm provider",
        "Backend",
        "str",
        default="openai",
        help="Provider for the any-llm backend: openai, mistral, groq, ollama, etc.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_REGION",
        "region",
        "Cloud region",
        "Backend",
        "str",
        default="us-west-2",
        help="Cloud region for Bedrock/Vertex/etc backends.",
        tier="advanced",
    ),
    # --- Timeouts ---
    SettingField(
        "HEADROOM_RETRY_MAX_ATTEMPTS",
        "retry_max_attempts",
        "Upstream retry attempts",
        "Timeouts",
        "int",
        default=None,
        minimum=1,
        maximum=10,
        help="Maximum upstream retry attempts on connect/read/5xx failures. Default: 3.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_RETRY_BASE_DELAY_MS",
        "retry_base_delay_ms",
        "Retry base delay (ms)",
        "Timeouts",
        "int",
        default=1000,
        minimum=0,
        help="Initial upstream retry delay in milliseconds. Default: 1000.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_RETRY_MAX_DELAY_MS",
        "retry_max_delay_ms",
        "Retry max delay (ms)",
        "Timeouts",
        "int",
        default=30000,
        minimum=0,
        help="Maximum upstream retry delay in milliseconds. Default: 30000.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_REQUEST_TIMEOUT",
        "request_timeout",
        "Request timeout (s)",
        "Timeouts",
        "int",
        default=None,
        help="Overall upstream request timeout in seconds. Default: 300.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_CONNECT_TIMEOUT_SECONDS",
        "connect_timeout_seconds",
        "Connect timeout (s)",
        "Timeouts",
        "int",
        default=None,
        minimum=1,
        maximum=300,
        help="Upstream connection timeout in seconds. Default: 10.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_WRITE_TIMEOUT_SECONDS",
        "write_timeout_seconds",
        "Write timeout (s)",
        "Timeouts",
        "int",
        default=None,
        minimum=1,
        help=(
            "Seconds the upstream send may take before it is abandoned. Default: 150. "
            "On HTTP/1.1 this bounds the whole request body, so raise it if you push "
            "large bodies over a slow link. Lower it to fail over a dead pooled "
            "connection faster; the connect timeout only guards a fresh connect."
        ),
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS",
        "anthropic_buffered_request_timeout_seconds",
        "Anthropic buffered timeout (s)",
        "Timeouts",
        "int",
        default=None,
        minimum=1,
        help="Buffered Anthropic read timeout for non-streaming message batch paths. Default: 600.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY",
        "anthropic_pre_upstream_concurrency",
        "Pre-upstream concurrency gate",
        "Timeouts",
        "int",
        default=None,
        help="Cap concurrent Anthropic pre-upstream work. Default: max(2, min(8, cpu_count)).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS",
        "anthropic_pre_upstream_acquire_timeout_seconds",
        "Pre-upstream acquire timeout (s)",
        "Timeouts",
        "float",
        default=None,
        help="Fail-fast timeout waiting on the pre-upstream semaphore. Default: 15.0.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS",
        "anthropic_pre_upstream_memory_context_timeout_seconds",
        "Pre-upstream memory-context timeout (s)",
        "Timeouts",
        "float",
        default=None,
        help="Fail-open timeout for memory-context lookup while holding a pre-upstream slot. Default: 2.0.",
        tier="advanced",
    ),
    # --- Memory ---
    SettingField(
        "HEADROOM_MEMORY_DB_PATH",
        "memory_db_path",
        "Memory DB path",
        "Memory",
        "str",
        default="",
        help="Path to the legacy single-file memory SQLite DB. Default: {cwd}/.headroom/memory.db.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_MEMORY_PROJECT_ROOT",
        "memory_project_root",
        "Memory project root",
        "Memory",
        "str",
        default="",
        help="Override the project root used for --memory-storage=project.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_NO_MEMORY_TOOLS",
        "no_memory_tools",
        "Disable memory tools",
        "Memory",
        "bool",
        default=False,
        help="Disable automatic injection of memory_save/memory_search tools.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_NO_MEMORY_CONTEXT",
        "no_memory_context",
        "Disable memory context injection",
        "Memory",
        "bool",
        default=False,
        help="Disable automatic injection of relevant memories into the system prompt.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_MEMORY_TOP_K",
        "memory_top_k",
        "Memory retrieval top-K",
        "Memory",
        "int",
        default=10,
        minimum=1,
        maximum=100,
        help="Number of semantically-relevant memories to retrieve.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_MIN_EVIDENCE",
        "min_evidence",
        "Minimum evidence count",
        "Memory",
        "int",
        default=None,
        minimum=1,
        help="Minimum times a pattern must be observed before it is persisted to memory. Default: 5.",
        tier="advanced",
    ),
    # --- Endpoints (custom Anthropic/OpenAI upstream) ---
    SettingField(
        "ANTHROPIC_TARGET_API_URL",
        "anthropic_base_url",
        "Anthropic base URL",
        "Endpoints",
        "str",
        default=None,
        help="Custom Anthropic API base URL (e.g. Azure Foundry, corporate gateway). Overrides https://api.anthropic.com.",
        tier="advanced",
    ),
    SettingField(
        "OPENAI_TARGET_API_URL",
        "openai_base_url",
        "OpenAI base URL",
        "Endpoints",
        "str",
        default=None,
        help="Custom OpenAI API base URL (e.g. corporate gateway). Overrides https://api.openai.com.",
        tier="advanced",
    ),
    SettingField(
        "ANTHROPIC_TARGET_API_HEADERS",
        "anthropic_extra_headers",
        "Anthropic extra headers",
        "Endpoints",
        "header-map",
        default=None,
        secret=True,
        help='JSON object of extra headers merged into (and overriding) forwarded Anthropic requests, e.g. {"Api-Key": "..."}.',
        tier="advanced",
    ),
    SettingField(
        "OPENAI_TARGET_API_HEADERS",
        "openai_extra_headers",
        "OpenAI extra headers",
        "Endpoints",
        "header-map",
        default=None,
        secret=True,
        help="JSON object of extra headers merged into (and overriding) forwarded OpenAI requests.",
        tier="advanced",
    ),
    # --- Additional curated knobs -------------------------------------------
    # Each reuses an existing ``group`` (so it renders on that group's page);
    # env formats verified against each knob's own reader. All restart-required
    # (non-live). See docs/environment-variables.md for the full inventory.
    # Compression page
    SettingField(
        "HEADROOM_KOMPRESS_BACKEND",
        "kompress_backend",
        "Kompress engine backend",
        "Compression",
        "enum",
        default="auto",
        choices=("auto", "onnx", "onnx_cpu", "onnx_coreml", "pytorch", "pytorch_mps"),
        help="Kompress ML backend. auto tries ONNX CPU first, then PyTorch.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_DEDUPE",
        "dedupe",
        "Cross-turn dedup",
        "Compression",
        "bool",
        default=False,
        help="Deduplicate content repeated across turns (the original stays in context).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_TOOL_SEARCH",
        "tool_search",
        "Tool-search injection",
        "Compression",
        "bool",
        default=False,
        help="Replace a large tool list with a single search tool to cut tool-schema tokens.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_COLD_RECOMPACT",
        "cold_recompact",
        "Cold-prefix recompaction",
        "Compression",
        "bool",
        default=False,
        help="Recompact the whole history when the provider prompt cache has lapsed. Fires only "
        "on a confirmed-cold turn, so there is no warm cache to bust.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_NET_COST_POLICY",
        "net_cost_policy",
        "Net-cost frozen-floor unlock",
        "Compression",
        "bool",
        default=False,
        help="Allow compression to rewrite messages already inside the provider's cached prefix "
        "when the token saving beats the cache-rewrite penalty. Without it, savings stop growing "
        "once the prefix goes warm.",
        tier="advanced",
    ),
    # CCR & Caching page
    SettingField(
        "HEADROOM_CCR_BACKEND",
        "ccr_backend",
        "CCR storage backend",
        "CCR",
        "enum",
        default="sqlite",
        choices=("sqlite", "redis", "memory"),
        help="Where CCR-retrievable compressed content lives. redis enables cross-worker sharing.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_REDIS_URL",
        "redis_url",
        "Redis URL",
        "CCR",
        "str",
        default=None,
        help="Redis connection URL, used when the CCR backend is redis (e.g. redis://localhost:6379/0).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_CCR_TTL_SECONDS",
        "ccr_ttl_seconds",
        "CCR TTL (s)",
        "CCR",
        "int",
        default=None,
        minimum=0,
        help="Seconds compressed content stays retrievable. Default: 1800 (30 min).",
        tier="advanced",
    ),
    # Networking & Security page
    SettingField(
        "HEADROOM_STATELESS",
        "stateless",
        "Stateless mode",
        "Networking",
        "bool",
        default=False,
        help="Disable all filesystem writes (savings, logs) — run purely in-memory.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_OFFLINE",
        "offline",
        "Offline mode",
        "Networking",
        "bool",
        default=False,
        help="Air-gap / no-egress master switch — disables update checks, model downloads, etc.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_TLS_STRICT",
        "tls_strict",
        "Strict TLS verification",
        "Networking",
        "bool",
        default=True,
        help="OpenSSL 3.2+ strict X.509 verification. Turn off behind a corporate TLS-inspection proxy.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_CORS_ORIGINS",
        "cors_origins",
        "Allowed CORS origins",
        "Networking",
        "csv-list",
        default=None,
        help="Comma-separated allowed CORS origins for dashboard/API access.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_WS_ORIGINS",
        "ws_origins",
        "Allowed WebSocket origins",
        "Networking",
        "csv-list",
        default=None,
        help="Comma-separated allowed WebSocket origins. Falls back to the CORS origins when unset.",
        tier="advanced",
    ),
    # Endpoints page (additional upstream base URLs)
    SettingField(
        "VERTEX_TARGET_API_URL",
        "vertex_base_url",
        "Vertex AI base URL",
        "Endpoints",
        "str",
        default=None,
        help="Custom Vertex AI regional API URL for publisher endpoints.",
        tier="advanced",
    ),
    SettingField(
        "BEDROCK_TARGET_API_URL",
        "bedrock_base_url",
        "Bedrock base URL",
        "Endpoints",
        "str",
        default=None,
        help="Custom AWS Bedrock InvokeModel API URL (gateway/LocalStack; not re-signed for real AWS SigV4).",
        tier="advanced",
    ),
    SettingField(
        "GEMINI_TARGET_API_URL",
        "gemini_base_url",
        "Gemini base URL",
        "Endpoints",
        "str",
        default=None,
        help="Custom Gemini API URL for passthrough endpoints.",
        tier="advanced",
    ),
    SettingField(
        "CLOUDCODE_TARGET_API_URL",
        "cloudcode_base_url",
        "Cloud Code base URL",
        "Endpoints",
        "str",
        default=None,
        help="Custom Cloud Code Assist API URL for compatibility endpoints.",
        tier="advanced",
    ),
    # Memory page (Qdrant vector store for the qdrant-neo4j backend)
    SettingField(
        "HEADROOM_QDRANT_URL",
        "qdrant_url",
        "Qdrant URL",
        "Memory",
        "str",
        default=None,
        help="Full Qdrant URL (e.g. https://xyz.cloud.qdrant.io:6333). Overrides host/port.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_QDRANT_HOST",
        "qdrant_host",
        "Qdrant host",
        "Memory",
        "str",
        default=None,
        help="Qdrant hostname for the qdrant-neo4j backend. Default: localhost.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_QDRANT_PORT",
        "qdrant_port",
        "Qdrant port",
        "Memory",
        "int",
        default=None,
        minimum=1,
        maximum=65535,
        help="Qdrant HTTP port for the qdrant-neo4j backend. Default: 6333.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_QDRANT_API_KEY",
        "qdrant_api_key",
        "Qdrant API key",
        "Memory",
        "str",
        default=None,
        secret=True,
        help="API key for hosted Qdrant (e.g. Qdrant Cloud).",
        tier="advanced",
    ),
    # --- CLI-arg toggles (env-backed; restart-required) ---------------------
    # These mirror `headroom proxy` flags that resolve from an env var, so the
    # settings.json -> setdefault path reaches them.
    # Compression page
    SettingField(
        "HEADROOM_OPTIMIZE",
        "optimize",
        "Optimization enabled",
        "Compression",
        "bool",
        default=True,
        help="Master switch. Off = passthrough mode (no compression/optimization); mirrors --no-optimize.",
        tier="basic",
    ),
    # Compression page (tool_result interception is a startup-read transform, so
    # it must NOT sit on the Output Shaping page, whose fields are auto-live).
    SettingField(
        "HEADROOM_INTERCEPT_ENABLED",
        "intercept_enabled",
        "Tool-result interception",
        "Compression",
        "bool",
        default=False,
        help=(
            "Enable ast-grep tool_result interceptors (Read outliner, etc.); mirrors "
            "--intercept-tool-results. Rollout-gated: also needs "
            "HEADROOM_ROLLOUT_CHANNEL=canary (or higher), else it stays off."
        ),
        tier="advanced",
    ),
    # CCR & Caching page
    SettingField(
        "HEADROOM_CACHE_ENABLED",
        "cache_enabled",
        "Semantic cache enabled",
        "CCR",
        "bool",
        default=True,
        help="Semantic response cache. Off mirrors --no-cache.",
        tier="basic",
    ),
    # Limits & Budget page
    SettingField(
        "HEADROOM_RATE_LIMIT_ENABLED",
        "rate_limit_enabled",
        "Rate limiting enabled",
        "Limits",
        "bool",
        default=True,
        help="Enforce RPM/TPM limits. Off mirrors --no-rate-limit.",
        tier="basic",
    ),
    # Memory page (out-of-process embedding server sidecar)
    SettingField(
        "HEADROOM_EMBEDDING_SERVER",
        "embedding_server",
        "Embedding server sidecar",
        "Memory",
        "bool",
        default=False,
        help="Run a shared out-of-process embedder across workers (saves ~600 MB RSS); mirrors --embedding-server.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_EMBEDDING_SERVER_SOCKET",
        "embedding_server_socket",
        "Embedding server socket",
        "Memory",
        "str",
        default=None,
        help="Unix socket path for the embedding sidecar. Default: /tmp/headroom-embed-<port>.sock.",
        tier="advanced",
    ),
    # --- Output Shaping (live: applied via runtime_env, no restart) ----------
    # These mirror headroom/proxy/runtime_env.py RUNTIME_ENV_KNOBS. A save
    # persists to settings.json AND hot-reloads through set_overrides(), so it
    # takes effect on the next request. Unset means "adaptive default", so the
    # boolean knobs are optional-bool (unset is distinct from an explicit off).
    SettingField(
        "HEADROOM_OUTPUT_SHAPER",
        "output_shaper",
        "Output shaping",
        "Output Shaping",
        "optional-bool",
        default=None,
        help="Master switch for output-token shaping. Unset = adaptive default.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_VERBOSITY_LEVEL",
        "verbosity_level",
        "Verbosity level",
        "Output Shaping",
        "int",
        default=None,
        minimum=0,
        maximum=4,
        help="Verbosity steering level 0-4. Unset = learned/default.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_EFFORT_ROUTER",
        "effort_router",
        "Effort router",
        "Output Shaping",
        "optional-bool",
        default=None,
        help="Lower effort on mechanical tool-result continuations. Unset = adaptive.",
        tier="basic",
    ),
    SettingField(
        "HEADROOM_MECHANICAL_EFFORT",
        "mechanical_effort",
        "Mechanical effort",
        "Output Shaping",
        "str",
        default=None,
        help="Effort value used on mechanical continuations (e.g. low, medium, high).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_VERBOSITY_AUTOTUNE",
        "verbosity_autotune",
        "Verbosity autotune",
        "Output Shaping",
        "optional-bool",
        default=None,
        help="Use the AIMD verbosity controller state. Unset = adaptive.",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_OUTPUT_HOLDOUT",
        "output_holdout",
        "Output holdout fraction",
        "Output Shaping",
        "float",
        default=None,
        minimum=0.0,
        maximum=1.0,
        help="Fraction of conversations held out for A/B measurement (0-1).",
        tier="advanced",
    ),
    SettingField(
        "HEADROOM_INTERCEPT_READ_MIN_CHARS",
        "intercept_read_min_chars",
        "Read-rewrite min chars",
        "Output Shaping",
        "int",
        default=None,
        minimum=0,
        help="Minimum tool-output chars before the ast-grep read rewrite.",
        tier="advanced",
    ),
)

_BY_KEY: dict[str, SettingField] = {f.key: f for f in SETTINGS}
_BY_ENV: dict[str, SettingField] = {f.env: f for f in SETTINGS}


class SettingsValidationError(Exception):
    """Raised when a settings payload has unknown keys or invalid values.

    Carries structured detail so the API layer can map unknown keys to 400 and
    per-field type/range errors to 422.
    """

    def __init__(self, unknown_keys: list[str], field_errors: dict[str, str]) -> None:
        self.unknown_keys = unknown_keys
        self.field_errors = field_errors
        super().__init__(
            f"settings validation failed: unknown={unknown_keys} errors={field_errors}"
        )


def _normalize_values(values: dict[str, Any]) -> dict[str, Any]:
    """Rewrite known env aliases to their JSON/API keys."""
    normalized: dict[str, Any] = {}
    source_keys: dict[str, str] = {}
    conflicts: dict[str, str] = {}
    for incoming_key, value in values.items():
        field = _BY_ENV.get(incoming_key)
        key = field.key if field is not None else incoming_key
        if key in normalized:
            if normalized[key] != value:
                conflicts[key] = (
                    f"conflicting values supplied for {source_keys[key]!r} and {incoming_key!r}"
                )
            continue
        normalized[key] = value
        source_keys[key] = incoming_key
    if conflicts:
        raise SettingsValidationError([], conflicts)
    return normalized


def _coerce(field: SettingField, value: Any) -> Any:
    """Coerce a raw JSON/env value to the field's Python type.

    Returns ``None`` for null and empty values (empty coerces to ``None`` for
    every type except a plain ``bool``, which becomes ``False``). Raises
    ``ValueError`` on bad input so callers can surface a per-field message.
    """
    if value is None:
        return None
    if field.type in ("bool", "optional-bool"):
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if field.type == "optional-bool" and token == "":
            return None
        if token in ("1", "true", "yes", "on"):
            return True
        if token in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if field.type in ("int", "float"):
        if isinstance(value, bool):  # bool is an int subclass — reject explicitly
            raise ValueError(f"expected a number, got {value!r}")
        number: int | float
        if field.type == "int":
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"expected an integer, got {value!r}")
            number = int(value)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"expected a finite number, got {value!r}")
        if field.minimum is not None and number < field.minimum:
            raise ValueError(f"must be >= {field.minimum}")
        if field.maximum is not None and number > field.maximum:
            raise ValueError(f"must be <= {field.maximum}")
        return number
    if field.type == "enum":
        token = str(value)
        if token not in field.choices:
            raise ValueError(f"{token!r} not one of {list(field.choices)}")
        return token
    if field.type == "csv-list":
        tokens = value if isinstance(value, list | tuple) else str(value).split(",")
        tokens = [str(token).strip() for token in tokens]
        tokens = [token for token in tokens if token]
        return ",".join(tokens) if tokens else None
    if field.type == "header-map":
        if isinstance(value, dict):
            parsed = value
        else:
            try:
                parsed = json.loads(str(value))
            except (ValueError, TypeError) as exc:
                raise ValueError("expected a JSON object of header name/value strings") from exc
        if not isinstance(parsed, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
        ):
            raise ValueError("expected a JSON object of header name/value strings")
        return json.dumps(parsed, sort_keys=True) if parsed else None
    # str
    token = str(value)
    return token if token != "" else None


def _serialize(field: SettingField, value: Any) -> str:
    """Serialize a coerced value to the exact string its env var expects."""
    if field.type in ("bool", "optional-bool"):
        return "1" if value else "0"
    return str(value)


def validate(values: dict[str, Any]) -> dict[str, Any]:
    """Validate/coerce ``values`` against the registry.

    Raises :class:`SettingsValidationError` when any key is unknown or any value
    fails coercion. Returns the coerced dict (``None`` values dropped) on success.
    """
    values = _normalize_values(values)
    unknown = [key for key in values if key not in _BY_KEY]
    field_errors: dict[str, str] = {}
    coerced: dict[str, Any] = {}
    for key, value in values.items():
        field = _BY_KEY.get(key)
        if field is None:
            continue
        try:
            result = _coerce(field, value)
        except (ValueError, TypeError) as exc:
            field_errors[key] = str(exc)
            continue
        if result is not None:
            coerced[key] = result
    if unknown or field_errors:
        raise SettingsValidationError(unknown, field_errors)
    return coerced


def load() -> dict[str, Any]:
    """Return validated stored values. Fail-open: ``{}`` if missing or corrupt."""
    path = paths.settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("settings_store: cannot read %s: %s", path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        logger.warning("settings_store: ignoring corrupt settings.json: %s", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("settings_store: settings.json is not a JSON object; ignoring")
        return {}
    out: dict[str, Any] = {}
    for key, value in data.items():
        field = _BY_KEY.get(key)
        if field is None:
            continue  # drop unknown keys
        try:
            result = _coerce(field, value)
        except (ValueError, TypeError) as exc:
            logger.warning("settings_store: dropping invalid %s: %s", key, exc)
            continue
        if result is not None:
            out[key] = result
    return out


def _atomic_write_text(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``).

    A crash mid-write leaves either the previous file or the complete new one on
    disk — never a truncated settings.json that would fail to parse and (via the
    startup apply hook) crash-loop a supervised proxy. ``load()`` is also
    fail-open as a second line of defence.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save(values: dict[str, Any]) -> None:
    """Validate ``values`` and merge them into the existing stored settings.

    A merge, not a wholesale replace -- callers submit only the fields that
    changed, and anything already on disk for other keys is preserved so a
    first save doesn't permanently pin every field's current default.

    Three submission shapes per key, beyond the default "absent -> unchanged":
    explicit ``None`` (JSON ``null``) clears the key from stored settings;
    a secret field resent as the mask sentinel (``_MASK``) is retained as-is
    and never coerced/overwritten, since the GUI always resends a masked
    secret's display value verbatim when the user hasn't touched it; anything
    else is validated/coerced and stored.
    """
    values = _normalize_values(values)
    clear_keys = {key for key, value in values.items() if value is None and key in _BY_KEY}
    retained_keys = {
        key
        for key, value in values.items()
        if key in _BY_KEY and _BY_KEY[key].secret and value == _MASK
    }
    to_validate = {
        key: value
        for key, value in values.items()
        if key not in clear_keys and key not in retained_keys
    }
    validated = validate(to_validate)
    merged = {**load(), **validated}
    for key in clear_keys:
        merged.pop(key, None)
    payload = json.dumps(merged, indent=2, sort_keys=True) + "\n"
    paths.ensure_workspace_dir()
    _atomic_write_text(paths.settings_path(), payload)


def apply_to_environ(values: dict[str, Any]) -> None:
    """Apply each stored value into ``os.environ``.

    settings.json is the highest-priority source below an explicit CLI arg, so a
    stored value *overrides* a shell-exported env var. ``manifest_managed`` knobs
    are the exception: the install manifest (docker/service) owns their env, so
    there we ``setdefault`` and let the exported value win.

    Live knobs are skipped: the proxy seeds them into the ``runtime_env``
    override store at startup instead.
    """
    for key, value in values.items():
        field = _BY_KEY.get(key)
        if field is None or value is None or field.live:
            continue
        if field.manifest_managed:
            os.environ.setdefault(field.env, _serialize(field, value))
        else:
            os.environ[field.env] = _serialize(field, value)


def effective_values(stored: dict[str, Any] | None = None) -> dict[str, Any]:
    """The value actually active now for each knob.

    settings.json is the highest-priority source, so for a normal knob the file
    wins over the environment: default ← environ ← file. ``manifest_managed``
    knobs are the exception — the install manifest (env) owns them, so there the
    environment wins: default ← file ← environ.
    """
    if stored is None:
        stored = load()
    result: dict[str, Any] = {}
    for field in SETTINGS:
        env_val: Any = None
        env_raw = os.environ.get(field.env)
        if env_raw is not None and env_raw != "":
            try:
                env_val = _coerce(field, env_raw)
            except (ValueError, TypeError):
                env_val = None  # unparseable env: ignore it
        has_file = field.key in stored
        if field.manifest_managed:
            # Install manifest (env) wins over the stored file.
            value = stored[field.key] if has_file else field.default
            if env_val is not None:
                value = env_val
        else:
            # settings.json wins over the environment.
            value = field.default
            if env_val is not None:
                value = env_val
            if has_file:
                value = stored[field.key]
        result[field.key] = value
    return result


def _mask(field: SettingField, value: Any) -> Any:
    if field.secret and value not in (None, ""):
        return _MASK
    return value


def stored_values(mask_secrets: bool = True) -> dict[str, Any]:
    """Stored file values (for ``GET /settings``); secret values masked."""
    values = load()
    if not mask_secrets:
        return values
    return {key: _mask(_BY_KEY[key], value) for key, value in values.items()}


def to_schema() -> dict[str, Any]:
    """Registry + grouped fields + effective values for the UI. Secrets masked.

    Each field carries ``page`` (sidebar nav category) and ``group`` (in-page
    sub-section header); ``pages`` lists the populated pages in nav order.
    ``live`` knobs apply via ``runtime_env`` with no restart, so only non-live
    keys land in ``needs_restart_keys`` for the UI's "restart to apply" banner.
    """
    stored = load()
    effective = effective_values(stored)
    fields: list[dict[str, Any]] = []
    for field in SETTINGS:
        fields.append(
            {
                "key": field.key,
                "env": field.env,
                "label": field.label,
                "group": field.group,
                "page": field.page,
                "type": field.type,
                "choices": list(field.choices),
                "default": field.default,
                "help": field.help,
                "secret": field.secret,
                "manifest_managed": field.manifest_managed,
                "minimum": field.minimum,
                "maximum": field.maximum,
                "tier": field.tier,
                "live": field.live,
                # settings.json wins over the environment, so an env var only
                # *overrides* (locks) a knob when the install manifest owns it.
                # ``env_present`` is purely informational for the UI.
                "env_override": bool(field.manifest_managed and os.environ.get(field.env)),
                "env_present": bool(os.environ.get(field.env)),
                "value": _mask(field, effective.get(field.key)),
                "stored": _mask(field, stored.get(field.key)),
            }
        )
    groups: list[str] = []
    for field in SETTINGS:
        if field.group not in groups:
            groups.append(field.group)
    populated = {field.page for field in SETTINGS}
    pages = [page for page in PAGES if page in populated]
    return {
        "pages": pages,
        "groups": groups,
        "fields": fields,
        "values": {f["key"]: f["value"] for f in fields},
        "needs_restart_keys": [field.key for field in SETTINGS if not field.live],
    }


def live_keys(keys: list[str]) -> list[str]:
    """Subset of ``keys`` whose registry field applies live (no restart)."""
    return [key for key in keys if (f := _BY_KEY.get(key)) is not None and f.live]


def coerce_env_value(key: str, raw: str | None) -> Any:
    """Coerce a raw env string to registry ``key``'s typed value (for display).

    Returns ``None`` when ``raw`` is unset/empty, or the key is unknown or the
    value unparseable. Lets the ``/settings/schema`` handler reflect a live
    ``runtime_env`` override as the field's typed value without importing the
    proxy into this dependency-light module.
    """
    if raw is None or raw == "":
        return None
    field = _BY_KEY.get(key)
    if field is None:
        return None
    try:
        return _coerce(field, raw)
    except (ValueError, TypeError):
        return None


def runtime_overrides(keys: list[str], values: dict[str, Any]) -> dict[str, str]:
    """Map live registry ``keys`` to ``{env: serialized}`` for a hot-reload push.

    ``values`` is a coerced stored dict (e.g. from :func:`load`). Non-live or
    unknown keys, and keys absent from ``values``, are skipped — so the result
    is exactly what ``runtime_env.set_overrides`` should apply after a save.
    """
    out: dict[str, str] = {}
    for key in keys:
        field = _BY_KEY.get(key)
        if field is None or not field.live or key not in values:
            continue
        out[field.env] = _serialize(field, values[key])
    return out


def env_for(key: str) -> str | None:
    """Return the ``HEADROOM_*`` env var name for a registry ``key`` (or None)."""
    field = _BY_KEY.get(key)
    return field.env if field is not None else None
