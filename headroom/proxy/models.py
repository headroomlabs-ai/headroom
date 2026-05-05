"""Data models for the Headroom proxy.

Contains configuration and data classes used across the proxy modules.
Extracted from server.py to keep the codebase maintainable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from headroom.memory import qdrant_env
from headroom.providers.registry import ProviderApiOverrides

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RequestLog:
    """Complete log of a single request."""

    request_id: str
    timestamp: str
    provider: str
    model: str

    # Tokens
    input_tokens_original: int
    input_tokens_optimized: int
    output_tokens: int | None
    tokens_saved: int
    savings_percent: float

    # Performance
    optimization_latency_ms: float
    total_latency_ms: float | None

    # Metadata
    tags: dict[str, str]
    cache_hit: bool
    transforms_applied: list[str]

    # Waste signals detected in original messages
    waste_signals: dict[str, int] | None = None

    # Request/Response (optional, for debugging)
    request_messages: list[dict] | None = None
    response_content: str | None = None
    error: str | None = None

    # Groups every agent-loop API call from one user prompt into a single turn.
    # See ``headroom.proxy.helpers.compute_turn_id`` for the derivation. None
    # when no user-text message is present in the request.
    turn_id: str | None = None

    # NOTE (Unit 2 follow-up): stage timings and session_id were briefly
    # added here but are now emitted exclusively through
    # ``emit_stage_timings_log`` (structured log line) and Prometheus.
    # They were never populated on ``RequestLog`` instances, so the
    # fields were removed to avoid confusing readers who expect
    # them to be set. If a JSONL consumer needs them, have the consumer
    # merge ``stage_timings`` log lines by ``request_id``.


@dataclass
class CacheEntry:
    """Cached response entry."""

    response_body: bytes
    response_headers: dict[str, str]
    created_at: datetime
    ttl_seconds: int
    hit_count: int = 0
    tokens_saved_per_hit: int = 0


@dataclass
class RateLimitState:
    """Token bucket rate limiter state."""

    tokens: float
    last_update: float


@dataclass
class ProxyConfig:
    """Proxy configuration."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8787
    anthropic_api_url: str | None = None  # Custom Anthropic API URL override
    openai_api_url: str | None = None  # Custom OpenAI API URL override
    gemini_api_url: str | None = None  # Custom Gemini API URL override
    cloudcode_api_url: str | None = None  # Custom Cloud Code Assist API URL override

    # Backend: "anthropic" (direct API), "litellm-*" (via LiteLLM), or "anyllm" (via any-llm)
    backend: str = "anthropic"
    bedrock_region: str = "us-west-2"
    bedrock_profile: str | None = None
    anyllm_provider: str = "openai"

    # Optimization mode: "token" (rewrite for max compression) or
    # "cache" (freeze prior turns for prefix-cache stability).
    mode: str = "token"

    # Optimization
    optimize: bool = True
    image_optimize: bool = True
    min_tokens_to_crush: int = 500
    max_items_after_crush: int = 50
    keep_last_turns: int = 4

    # CCR Tool Injection
    ccr_inject_tool: bool = True
    ccr_inject_system_instructions: bool = False

    # CCR Response Handling
    ccr_handle_responses: bool = True
    ccr_max_retrieval_rounds: int = 3

    # CCR Context Tracking
    ccr_context_tracking: bool = True
    ccr_proactive_expansion: bool = True
    ccr_max_proactive_expansions: int = 2

    # Code-aware compression (disabled by default — use code graph tools instead)
    code_aware_enabled: bool = False

    # Code graph live watcher (triggers incremental reindex on file changes)
    code_graph_watcher: bool = False

    # Per-tool compression profiles
    tool_profiles: dict[str, Any] | None = None

    # Read lifecycle management
    read_lifecycle: bool = True

    # Smart content routing
    smart_routing: bool = True

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 1000

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100000

    # Retry
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay_ms: int = 1000
    retry_max_delay_ms: int = 30000

    # Prefix freeze
    prefix_freeze_enabled: bool = True
    prefix_freeze_session_ttl: int = 600

    # Cost tracking
    cost_tracking_enabled: bool = True
    budget_limit_usd: float | None = None
    budget_period: Literal["hourly", "daily", "monthly"] = "daily"

    # Logging
    log_requests: bool = True
    log_file: str | None = None
    log_full_messages: bool = False

    # Third-party proxy extensions (opt-in only). List of entry-point names
    # to enable from the `headroom.proxy_extension` group, or `["*"]` for
    # wildcard. Empty/None means no extensions run, even if installed.
    # CLI: --proxy-extension <name1,name2>; env: HEADROOM_PROXY_EXTENSIONS.
    proxy_extensions: list[str] | None = None

    # Fallback
    fallback_enabled: bool = False
    fallback_provider: str | None = None

    # Timeouts
    request_timeout_seconds: int = 300
    connect_timeout_seconds: int = 10

    # Connection pool
    max_connections: int = 500
    max_keepalive_connections: int = 100
    http2: bool = True

    # Memory System
    memory_enabled: bool = False
    memory_backend: Literal["local", "qdrant-neo4j"] = "local"
    memory_db_path: str = ""  # Empty = auto: {cwd}/.headroom/memory.db
    memory_inject_tools: bool = True
    traffic_learning_enabled: bool = False
    traffic_learning_agent_type: str = "unknown"  # Which agent is being wrapped
    # Minimum evidence count before a learned pattern is persisted to memory.
    # Higher values reduce one-shot noise at the cost of slower learning.
    traffic_learning_min_evidence: int = 5
    memory_use_native_tool: bool = False
    memory_inject_context: bool = True
    memory_top_k: int = 10
    memory_min_similarity: float = 0.3
    # PR-B6: Memory injection mode. ``"auto_tail"`` (default) auto-appends
    # retrieved memory to the latest user message tail (live zone).
    # ``"tool"`` disables auto-injection — the model must call
    # ``memory_search`` to retrieve. See REALIGNMENT/04-phase-B-live-zone.md
    # PR-B6.
    memory_mode: Literal["auto_tail", "tool"] = "auto_tail"
    # Qdrant connection (defaults resolve from HEADROOM_QDRANT_* env vars)
    memory_qdrant_url: str | None = field(default_factory=qdrant_env.qdrant_env_url)
    memory_qdrant_host: str = field(default_factory=qdrant_env.qdrant_env_host)
    memory_qdrant_port: int = field(default_factory=qdrant_env.qdrant_env_port)
    memory_qdrant_api_key: str | None = field(default_factory=qdrant_env.qdrant_env_api_key)
    memory_neo4j_uri: str = "neo4j://localhost:7687"
    memory_neo4j_user: str = "neo4j"
    memory_neo4j_password: str = "password"
    memory_bridge_enabled: bool = False
    memory_bridge_md_paths: list[str] = field(default_factory=list)
    memory_bridge_md_format: str = "auto"
    memory_bridge_auto_import: bool = False
    memory_bridge_export_path: str = ""

    # License / Usage Reporting
    license_key: str | None = None
    license_cloud_url: str = "https://app.headroomlabs.ai"
    license_report_interval: int = 300

    # Compression Hooks
    hooks: Any = None
    pipeline_extensions: list[Any] = field(default_factory=list)
    discover_pipeline_extensions: bool = True

    # Subscription Window Tracking (Anthropic OAuth accounts)
    subscription_tracking_enabled: bool = True
    subscription_poll_interval_s: int = 300
    subscription_active_window_s: int = 60

    # Stateless mode — disable all filesystem writes for read-only / container deployments
    stateless: bool = False

    # Unit 4: Bounded pre-upstream concurrency for Anthropic replay storms.
    #
    # Caps the number of simultaneous requests allowed to run the
    # pre-upstream phase of ``handle_anthropic_messages`` (request JSON
    # read → deep-copy → first compression stage → memory-context lookup
    # → first upstream connect). Prevents cold-start replay storms from
    # monopolising the event loop / thread pool and starving ``/livez``,
    # ``/readyz``, and new Codex WS opens. Compression stays on.
    #
    # ``None`` (default) -> auto-compute ``max(2, min(8, os.cpu_count() or 4))``.
    # ``0`` or negative  -> disables the semaphore (unbounded); useful for
    # the Unit 6 counter-factual and for deliberately reproducing the
    # original starvation. Any positive integer is honored verbatim.
    #
    # CLI: ``--anthropic-pre-upstream-concurrency``.
    # Env: ``HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY``.
    # Precedence: CLI > env > auto-compute.
    anthropic_pre_upstream_concurrency: int | None = None
    # Upper bound for waiting on the Anthropic pre-upstream semaphore
    # before failing fast with a 503 + Retry-After. Keeps the queue bounded
    # when all pre-upstream slots are occupied by slow/hung work.
    anthropic_pre_upstream_acquire_timeout_seconds: float = 15.0
    # Fail-open timeout for Anthropic memory-context lookup while the request
    # is still holding a pre-upstream slot. Compression already has its own
    # COMPRESSION_TIMEOUT_SECONDS guard; this bounds the memory leg too.
    anthropic_pre_upstream_memory_context_timeout_seconds: float = 2.0

    # Bound the dedicated compression threadpool. CPU-bound Rust work runs
    # here; the pool is separate from asyncio's default executor so other
    # ``asyncio.to_thread`` callers (file IO, etc.) are not contended by
    # compression bursts. ``None`` resolves to ``min(32, (cpu_count or 1) * 4)``,
    # matching asyncio's default executor sizing today. Lower the cap to
    # tighten resource use on multi-tenant hosts; raise it to handle larger
    # bursts. CLI: ``--compression-max-workers``. Env:
    # ``HEADROOM_COMPRESSION_MAX_WORKERS``.
    #
    # Background: ``asyncio.wait_for`` cancellation does NOT propagate into
    # the threadpool worker that's running Rust code — once the worker has
    # picked up the task, ``concurrent.futures.Future.cancel()`` returns
    # ``False`` and the thread runs to completion. A bounded pool lets us
    # observe the worst case (max queue depth, "leaked" threads that
    # finished post-deadline) and fail fast under contention rather than
    # piling unboundedly on the default executor. See
    # ``HeadroomProxy._run_compression_in_executor``.
    compression_max_workers: int | None = None

    @property
    def provider_api_overrides(self) -> ProviderApiOverrides:
        """Return provider API URL overrides as a dedicated provider config object."""
        return ProviderApiOverrides(
            anthropic=self.anthropic_api_url,
            openai=self.openai_api_url,
            gemini=self.gemini_api_url,
            cloudcode=self.cloudcode_api_url,
        )
