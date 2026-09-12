"""Data models for the Headroom proxy.

Contains configuration and data classes used across the proxy modules.
Extracted from server.py to keep the codebase maintainable.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from typing import Any, Literal

from headroom.memory import qdrant_env
from headroom.providers.registry import ProviderApiOverrides
from headroom.proxy.buffered_ccr_response import DEFAULT_BUFFERED_CCR_GRACE_SECONDS
from headroom.proxy.model_router import ModelRouterConfig
from headroom.rollout import RolloutSnapshot, resolve_rollout

logger = logging.getLogger(__name__)


def _qdrant_env_port_or_default() -> int:
    """Resolve ``HEADROOM_QDRANT_PORT``, falling back to the default on a bad value.

    ``qdrant_env.qdrant_env_port`` raises on an invalid port (intended for
    explicit qdrant setup). As a ``ProxyConfig`` field ``default_factory`` it
    runs on EVERY ``ProxyConfig()`` construction, regardless of whether
    memory/qdrant is enabled (both off by default), so a stray or typo'd
    ``HEADROOM_QDRANT_PORT`` would crash proxy startup for an unrelated,
    off-by-default subsystem. Fail soft here so config construction never raises.
    """
    try:
        return qdrant_env.qdrant_env_port()
    except ValueError:
        logger.warning(
            "Ignoring invalid HEADROOM_QDRANT_PORT; using default %d. "
            "Set a valid 1-65535 port to override.",
            qdrant_env.DEFAULT_QDRANT_PORT,
        )
        return qdrant_env.DEFAULT_QDRANT_PORT


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
    tags: dict[str, Any]
    cache_hit: bool
    transforms_applied: list[str]

    # Per-request attribution. Headline totals remain authoritative, so these
    # explanatory rows are never added a second time.
    savings_breakdown: list[dict[str, Any]] = field(default_factory=list)

    # Provider-side cache economics (Anthropic prompt caching, #2438).
    # ``cache_hit`` alone is ambiguous: a call billed cache-*creation* (write)
    # cannot be told apart from a real cache-*read* hit. These raw deltas —
    # already carried on RequestOutcome from the upstream response usage —
    # let the JSONL telemetry reflect true economics (uncached input +
    # cache_creation), not just the proxy's boolean.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    uncached_input_tokens: int = 0

    # Waste signals detected in original messages
    waste_signals: dict[str, int] | None = None

    # Request/Response (optional, for debugging)
    request_messages: list[dict] | None = None
    # Messages after compression, as actually sent upstream. Paired with
    # `request_messages` (the pre-compression snapshot) so consumers can diff
    # the two sides of the compression. Governed by the same
    # `log_full_messages` gate as `request_messages`.
    compressed_messages: list[dict] | None = None
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
    # Resolved at this configuration boundary and then injected unchanged.
    rollout: RolloutSnapshot | None = None
    anthropic_api_url: str | None = None  # Custom Anthropic API URL override
    openai_api_url: str | None = None  # Custom OpenAI API URL override
    # Display label for the OpenAI-compatible upstream (dashboard/stats only).
    # Overrides hostname detection from ``openai_api_url``; the internal
    # provider stays ``openai`` so pricing/format keys are unaffected.
    provider_name: str | None = None
    gemini_api_url: str | None = None  # Custom Gemini API URL override
    cloudcode_api_url: str | None = None  # Custom Cloud Code Assist API URL override
    vertex_api_url: str | None = None  # Custom Vertex AI regional API URL override
    # Extra headers merged into (and overriding) forwarded Anthropic/OpenAI requests.
    # JSON-object config knobs; see settings_store's anthropic_extra_headers/
    # openai_extra_headers and providers.registry.resolve_extra_headers.
    anthropic_extra_headers: dict[str, str] | None = None
    openai_extra_headers: dict[str, str] | None = None

    # Backend: "anthropic" (direct API), "litellm-*" (via LiteLLM), or "anyllm" (via any-llm)
    backend: str = "anthropic"
    bedrock_region: str = "us-west-2"
    bedrock_profile: str | None = None
    # Custom upstream for the Bedrock InvokeModel passthrough routes
    # (`/model/{id}/invoke[-with-response-stream]`). When set, those routes are
    # registered and compress the request body before forwarding here. Point it
    # at a re-signing gateway (LiteLLM, LocalStack, a corporate Bedrock
    # proxy) — NOT raw AWS, since rewriting the body invalidates the caller's
    # SigV4 signature. Leave unset (default) to keep `--backend bedrock`'s
    # direct-to-AWS, re-signing behavior unchanged.
    bedrock_api_url: str | None = None
    anyllm_provider: str = "openai"

    # Optimization mode: "token" (rewrite for max compression) or
    # "cache" (freeze prior turns for prefix-cache stability).
    mode: str = "token"

    # Optimization
    optimize: bool = True
    image_optimize: bool = True
    min_tokens_to_crush: int = 500
    max_items_after_crush: int = 50
    smart_crusher_with_compaction: bool | None = None
    keep_last_turns: int = 4

    # Cost-aware model routing (issue #1706). Opt-in and disabled by default;
    # when configured, an ordered rule set can rewrite the outgoing model based
    # on request size / tool presence. None keeps behavior unchanged.
    model_router: ModelRouterConfig | None = None

    # CCR Tool Injection
    ccr_inject_tool: bool = True
    ccr_inject_system_instructions: bool = False
    # Proxy-level mirror of ContentRouterConfig.ccr_inject_marker, so retrieval
    # markers can be toggled from the CLI (--no-ccr, which also drops the retrieve
    # tool). Threaded into the router in server.py; default preserves current behavior.
    ccr_inject_marker: bool = True
    # Explicit opt-in fallback for callers with no path to redeem a marker via
    # the headroom_retrieve tool (e.g. a LiteLLM guardrail/proxy hop with no
    # tool-call turn — issue #2509). Off by default: guessing "this caller
    # can't use tools" is fragile, so operators must opt in with
    # --ccr-inline-resolve / HEADROOM_CCR_INLINE_RESOLVE. Applies to
    # non-streaming responses only; buffered-CCR streaming is untouched.
    ccr_resolve_markers_inline: bool = False

    # CCR Response Handling
    ccr_handle_responses: bool = True
    ccr_max_retrieval_rounds: int = 3

    # CCR Context Tracking
    ccr_context_tracking: bool = True
    ccr_proactive_expansion: bool = True
    ccr_max_proactive_expansions: int = 2
    ccr_max_turn_distance: int = 10

    # Code-aware compression (disabled by default — use code graph tools instead)
    code_aware_enabled: bool = False

    # Disable Kompress ML compression while keeping structural compressors
    # such as SmartCrusher, log/search/diff, and schema compaction enabled.
    # CLI: --disable-kompress; env: HEADROOM_DISABLE_KOMPRESS=1.
    disable_kompress: bool = False

    # With disable_kompress, route fall-through content to PASSTHROUGH instead
    # of the default KOMPRESS fallback strategy. Restores the legacy
    # --disable-kompress behaviour for callers that relied on it. No effect
    # unless disable_kompress is also set.
    # CLI: --disable-kompress-fallback; env: HEADROOM_DISABLE_KOMPRESS_FALLBACK=1.
    disable_kompress_fallback: bool = False

    # Per-provider overrides for `disable_kompress`. None inherits the global
    # value above; True/False force-disable/enable Kompress for that provider's
    # pipeline only (other compressors and all routing/exclusion are unaffected).
    # Lets e.g. Anthropic run without lossy text compression while OpenAI/Codex
    # keeps it. CLI: --disable-kompress-anthropic / --enable-kompress-anthropic
    # (and -openai); env: HEADROOM_DISABLE_KOMPRESS_ANTHROPIC / _OPENAI
    # (1 = disable, 0 = enable).
    disable_kompress_anthropic: bool | None = None
    disable_kompress_openai: bool | None = None

    # Force ALL compressible content through Kompress (kompress-v2-base),
    # bypassing per-type compressor selection (SmartCrusher/CodeAware/log/
    # diff/html/tabular/search). Tool ground truth stays protected: excluded
    # tools (Read/Glob/Grep/...) and reversibility-gated tool output are never
    # touched. Off by default; opt-in for systems that want one uniform
    # compressor at the cost of per-type structural fidelity.
    # CLI: --force-kompress-all; env: HEADROOM_FORCE_KOMPRESS_ALL=1.
    force_kompress_all: bool = False

    lossless: bool = False  # CLI: --lossless; env: HEADROOM_LOSSLESS=1. No-CCR mode: compress without any retrieval marker.

    # Compress requests that fall through to the catch-all passthrough handler
    # (custom proxy paths that don't match a built-in API route, e.g.
    # `/api/codex-proxy/<key>/v1/responses` fronted by another proxy). Off by
    # default because passthrough targets are unknown upstreams; opt-in for
    # wrapper-proxy architectures that need coding-agent traffic compressed.
    # Currently applies to OpenAI Responses-shaped bodies (paths ending in
    # `/responses`). CLI: --compress-passthrough; env:
    # HEADROOM_COMPRESS_PASSTHROUGH=1.
    compress_passthrough: bool = False

    # Code graph live watcher (triggers incremental reindex on file changes)
    code_graph_watcher: bool = False

    # Per-tool compression profiles
    tool_profiles: dict[str, Any] | None = None

    # Opt in to compressing `user` role messages. Off by default because user
    # content is typically the subject of the request and is part of the
    # prefix-cache zone. Enable for OpenAI/Azure chat workloads where the bulk
    # of input lives in user messages (pasted code/text, RAG context) and the
    # router would otherwise have nothing eligible to compress.
    # CLI: --compress-user-messages; env: HEADROOM_COMPRESS_USER_MESSAGES=1.
    compress_user_messages: bool = False
    # Named savings policy shared across Claude/Codex/Cursor proxy handlers.
    # CLI/env: HEADROOM_SAVINGS_PROFILE=agent-90.
    savings_profile: str | None = None
    target_ratio: float | None = None
    compress_system_messages: bool | None = None
    protect_recent: int | None = None
    protect_analysis_context: bool | None = None
    accuracy_guard: str | None = None

    # Extra tool names whose outputs are never compressed, merged with the
    # built-in DEFAULT_EXCLUDE_TOOLS. None means built-in defaults only.
    # CLI: --exclude-tools <name1,name2>; env: HEADROOM_EXCLUDE_TOOLS=<name1,name2>
    exclude_tools: set[str] | None = None

    # Tool names whose results must never be lossy-compressed (e.g. Bash, WebFetch).
    # Merged into exclude_tools before ContentRouter processes the conversation.
    # CLI: --protect-tool-results <name1,name2>; env: HEADROOM_PROTECT_TOOL_RESULTS=<name1,name2>
    protect_tool_results: frozenset[str] = field(default_factory=frozenset)

    # Read lifecycle management
    read_lifecycle: bool = True

    # Mechanism B: activity-based read maturation (hold fresh Reads out of
    # the provider prefix cache; compress once their file quiesces).
    # Experimental — default off. CLI: --read-maturation;
    # env: HEADROOM_READ_MATURATION=1
    read_maturation: bool = False
    # Read-maturation tuning (only meaningful when read_maturation=True).
    # Defaults mirror ReadMaturationConfig. CLI: --read-maturation-quiesce-turns,
    # --read-maturation-max-hold-turns, --read-maturation-min-size-bytes;
    # env: HEADROOM_READ_MATURATION_QUIESCE_TURNS / _MAX_HOLD_TURNS / _MIN_SIZE_BYTES.
    read_maturation_quiesce_turns: int = 5
    read_maturation_max_hold_turns: int = 25
    read_maturation_min_size_bytes: int = 2048

    # Deprecated compatibility argument. ContentRouter is always active in
    # the Python proxy; accepting this avoids breaking old config constructors
    # while keeping it out of runtime state.
    smart_routing: InitVar[bool | None] = None

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
    # What spend booked from Headroom's own token estimate (provider returned no
    # usage breakdown) does to budget enforcement. See budget_basis_policy.
    budget_estimated_basis: Literal["count", "ignore", "block"] = "count"

    # Logging
    log_requests: bool = True
    log_file: str | None = None
    log_full_messages: bool = False

    # Third-party proxy extensions (opt-in only). List of entry-point names
    # to enable from the `headroom.proxy_extension` group, or `["*"]` for
    # wildcard. Empty/None means no extensions run, even if installed.
    # CLI: --proxy-extension <name1,name2>; env: HEADROOM_PROXY_EXTENSIONS.
    proxy_extensions: list[str] | None = None

    # Compressor selection (opt-in narrowing of the built-in compressor set).
    # None (the default) leaves EVERY built-in compressor enabled — byte-
    # identical to today. When a set is given, only the named recognized
    # built-ins {smart_crusher, kompress, code_aware, search, log, tabular,
    # config, html, image} stay enabled and the rest are disabled at the
    # ContentRouterConfig `enable_*` seam; `"*"` enables all. Names that are
    # not recognized built-ins are ignored here (reserved for the
    # `headroom.compressor` registry). CLI: --compressor <name1,name2>
    # (repeatable); env: HEADROOM_COMPRESSORS.
    compressors: set[str] | None = None

    # Fallback
    fallback_enabled: bool = False
    fallback_provider: str | None = None

    # Timeouts
    request_timeout_seconds: int = 300
    connect_timeout_seconds: int = 10
    # Sending the request is a different operation from waiting for the model to
    # answer, but `write` used to inherit `request_timeout_seconds`, so pushing
    # bytes got the same 300s budget as a model thinking. That left the write
    # phase effectively unbounded in practice: when an upstream stops draining —
    # a pooled socket whose peer went away over a laptop sleep, a network change,
    # a NAT timeout — the send blocks until the OS gives up retransmitting
    # (~180-220s on macOS), which is *under* 300s, so no timeout ever fired and
    # the proxy hung, retried, and hung again (#3259).
    #
    # This bounds the WHOLE body send, not one chunk of it. httpx hands a bytes
    # body to the transport as a single write, so on HTTP/1.1 the entire upload
    # runs inside one timer -- measured: a 16MB body against a peer draining
    # steadily at ~1MB/s raises WriteTimeout at exactly the configured bound,
    # healthy peer or not. On HTTP/2 the body is split by flow control and the
    # waiting-for-window part is charged to `read`, so only real socket writes
    # count against it. The default has to clear the slower of those two while
    # staying under the OS retransmit ceiling that made the inherited 300s
    # unreachable: 150s carries a 15MB body -- the largest #3259 reports -- over
    # a ~1 Mbps uplink, and still fires well before the OS gives up at ~180s.
    write_timeout_seconds: int = 150
    # Anthropic buffered reads can legitimately run longer than the generic
    # proxy request cap. Keep the generic timeout unchanged elsewhere.
    anthropic_buffered_request_timeout_seconds: int = 600
    # How long a buffered-CCR turn holds out for full status fidelity before it
    # commits to SSE and starts a keepalive. Under the window, failures keep
    # their real HTTP status; past it, the client gets a first byte before its
    # stream-idle watchdog fires. 0 or less disables the keepalive entirely.
    # See headroom/proxy/buffered_ccr_response.py (#3079).
    buffered_ccr_grace_seconds: float = DEFAULT_BUFFERED_CCR_GRACE_SECONDS

    # Connection pool
    max_connections: int = 500
    max_keepalive_connections: int = 100
    keepalive_expiry: float = 90.0
    http2: bool = True
    http_proxy: str | None = None

    # Memory System
    memory_enabled: bool = False
    memory_backend: Literal["local", "qdrant-neo4j"] = "local"
    memory_db_path: str = ""  # Empty = auto: {cwd}/.headroom/memory.db
    # Per-project memory routing (GH #462). ``project`` (the new default)
    # gives each resolved workspace its own SQLite DB so cross-project
    # bleed becomes structurally impossible. ``user`` partitions by
    # x-headroom-user-id only. ``global`` keeps the pre-fix single-DB
    # behaviour (existing memories remain reachable here).
    memory_storage_mode: Literal["project", "user", "global"] = "project"
    memory_project_root_override: str = ""
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
    memory_qdrant_port: int = field(default_factory=_qdrant_env_port_or_default)
    memory_qdrant_api_key: str | None = field(default_factory=qdrant_env.qdrant_env_api_key)
    memory_neo4j_uri: str = "neo4j://localhost:7687"
    memory_neo4j_user: str = "neo4j"
    memory_neo4j_password: str = ""
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

    # Periodic TOIN stats logging. Enabled by default for observability, but
    # operators of long-lived proxies can disable it if TOIN stats collection
    # causes avoidable memory pressure on their platform.
    # Env: HEADROOM_PERIODIC_TOIN_STATS=0.
    periodic_toin_stats_enabled: bool = True

    # Periodic allocator trim. Long-lived proxies processing large concurrent
    # request bodies ratchet RSS through freed-but-retained allocator pages;
    # this returns them to the OS (malloc_zone_pressure_relief on macOS,
    # malloc_trim on glibc). Default-on only on macOS, where the retained-page
    # ratchet is the documented failure (#2820); an opt-in elsewhere via
    # HEADROOM_MALLOC_TRIM=1 so glibc deployments do not silently take on a
    # once-a-minute allocator purge they did not ask for. Envs:
    # HEADROOM_MALLOC_TRIM=0/1, HEADROOM_MALLOC_TRIM_INTERVAL_SECONDS.
    periodic_malloc_trim_enabled: bool = field(default_factory=lambda: sys.platform == "darwin")
    malloc_trim_interval_seconds: int = 60

    # Stateless mode — disable all filesystem writes for read-only / container deployments
    stateless: bool = False

    # Optional inbound auth. When set, non-loopback requests to the data-plane
    # routes must present this token (``Authorization: Bearer <token>`` or the
    # ``X-Headroom-Proxy-Token`` header). Loopback callers are exempt. Closes the
    # gap where a container bound to 0.0.0.0 exposes unauthenticated /v1/* routes
    # to the pod network. Env: HEADROOM_PROXY_TOKEN.
    proxy_token: str | None = None

    # Air-gap master switch — hard-disable ALL outbound network egress
    # (telemetry beacon, update check, license/usage reporter, HuggingFace model
    # downloads) for fully offline / regulated deployments. Env: HEADROOM_OFFLINE=1.
    offline: bool = False

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
    # before failing open to passthrough compression. Keeps the queue bounded
    # when all pre-upstream slots are occupied by slow/hung work.
    anthropic_pre_upstream_acquire_timeout_seconds: float = 15.0
    # Fail-open timeout for Anthropic memory-context lookup while the request
    # is still holding a pre-upstream slot. Compression already has its own
    # COMPRESSION_TIMEOUT_SECONDS guard; this bounds the memory leg too.
    anthropic_pre_upstream_memory_context_timeout_seconds: float = 2.0

    # Bound the dedicated compression threadpool. CPU-bound Rust work runs
    # here; the pool is separate from asyncio's default executor so other
    # ``asyncio.to_thread`` callers (file IO, etc.) are not contended by
    # compression bursts. ``None`` resolves to ``cpu_count or 1`` so CPU-bound
    # compression work does not oversubscribe hosts by default. Lower the cap
    # to tighten resource use on multi-tenant hosts; raise it to handle larger
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

    # Number of built-in uvicorn worker processes sharing this listen socket.
    # Kept at the end to avoid shifting existing positional constructor fields.
    # Process-local runtime hot reload is unsafe above one worker because only
    # the worker receiving the admin request would observe the update.
    worker_processes: int = 1

    def __post_init__(self, smart_routing: bool | None = None) -> None:
        if self.rollout is None:
            self.rollout = resolve_rollout()
        # ``read_maturation`` remains a concrete, already-resolved runtime
        # setting for programmatic/config-file callers.  The CLI composition
        # root derives it from this same snapshot before constructing the
        # config; rewriting it here would resolve policy a second time and
        # break explicit non-CLI configuration.
        if self.worker_processes < 1:
            raise ValueError("worker_processes must be >= 1")
        if self.retry_enabled and self.retry_max_attempts < 1:
            raise ValueError("retry_max_attempts must be >= 1 when retry_enabled=True")
        # A 0 (or negative) requests-per-minute limit divides by zero in the
        # token-bucket wait computation (rate_limit_policy.consume_from_bucket),
        # 500-ing every request. The CLI already guards this with IntRange(min=1);
        # fail fast here too so the JSON/programmatic config paths can't produce a
        # limiter that crashes at request time. Only matters when limiting is on.
        if self.rate_limit_enabled and self.rate_limit_requests_per_minute < 1:
            raise ValueError(
                "rate_limit_requests_per_minute must be >= 1 when rate_limit_enabled=True"
            )

    @property
    def provider_api_overrides(self) -> ProviderApiOverrides:
        """Return provider API URL overrides as a dedicated provider config object."""
        return ProviderApiOverrides(
            anthropic=self.anthropic_api_url,
            openai=self.openai_api_url,
            gemini=self.gemini_api_url,
            cloudcode=self.cloudcode_api_url,
            vertex=self.vertex_api_url,
        )
