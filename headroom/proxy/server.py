"""Headroom Proxy Server - Production Ready.

A full-featured LLM proxy with optimization, caching, rate limiting,
and observability.

Features:
- Context optimization (SmartCrusher, CacheAligner — live-zone-only after Phase B)
- Semantic caching (save costs on repeated queries)
- Rate limiting (token bucket)
- Retry with exponential backoff
- Cost tracking and budgets
- Request tagging and metadata
- Provider fallback
- Prometheus metrics
- Full request/response logging

Usage:
    python -m headroom.proxy.server --port 8787

    # With Claude Code:
    ANTHROPIC_BASE_URL=http://localhost:8787 claude
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from ..backends.base import Backend
    from ..cache.compression_cache import CompressionCache
    from ..memory.tracker import MemoryTracker


import httpx

try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from headroom._version import __version__
from headroom.cache.compression_feedback import get_compression_feedback
from headroom.cache.compression_store import get_compression_store
from headroom.ccr import (
    CCR_TOOL_NAME,
    # Batch processing
    CCRResponseHandler,
    CCRToolInjector,
    ContextTracker,
    ContextTrackerConfig,
    ResponseHandlerConfig,
    parse_tool_call,
)
from headroom.config import (
    CacheAlignerConfig,
    CCRConfig,
    ReadLifecycleConfig,
    SmartCrusherConfig,
)
from headroom.dashboard import get_dashboard_html
from headroom.observability import (
    LangfuseTracingConfig,
    OTelMetricsConfig,
    configure_langfuse_tracing,
    configure_otel_metrics,
    get_langfuse_tracing_status,
    get_otel_metrics_status,
    shutdown_headroom_tracing,
    shutdown_otel_metrics,
)
from headroom.pipeline import PipelineExtensionManager, PipelineStage
from headroom.providers.proxy_routes import register_provider_routes
from headroom.providers.registry import (
    DEFAULT_ANTHROPIC_API_URL,
    DEFAULT_CLOUDCODE_API_URL,
    DEFAULT_GEMINI_API_URL,
    DEFAULT_OPENAI_API_URL,
    build_proxy_provider_runtime,
    create_proxy_backend,
    format_backend_status,
)

# =============================================================================
# Extracted modules (re-exported for backward compatibility)
# =============================================================================
from headroom.proxy.cost import (
    _CACHE_ECONOMICS,  # noqa: F401
    CostTracker,  # noqa: F401
    _summarize_transforms,  # noqa: F401
    build_prefix_cache_stats,  # noqa: F401
    build_session_summary,  # noqa: F401
    merge_cost_stats,  # noqa: F401
)
from headroom.proxy.helpers import (
    COMPRESSION_TIMEOUT_SECONDS,  # noqa: F401
    MAX_COMPRESSION_CACHE_SESSIONS,  # noqa: F401
    MAX_MESSAGE_ARRAY_LENGTH,  # noqa: F401
    MAX_REQUEST_BODY_SIZE,  # noqa: F401
    MAX_SSE_BUFFER_SIZE,  # noqa: F401
    _get_image_compressor,  # noqa: F401
    _get_rtk_stats,  # noqa: F401
    _read_request_json,  # noqa: F401
    _setup_file_logging,  # noqa: F401
    is_anthropic_auth,  # noqa: F401
    jitter_delay_ms,
)
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

# Data models (extracted to headroom/proxy/models.py for maintainability)
from headroom.proxy.models import CacheEntry, ProxyConfig, RateLimitState, RequestLog  # noqa: F401
from headroom.proxy.modes import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    is_token_mode,
    normalize_proxy_mode,
)
from headroom.proxy.prometheus_metrics import PrometheusMetrics  # noqa: F401
from headroom.proxy.rate_limiter import TokenBucketRateLimiter  # noqa: F401
from headroom.proxy.request_logger import RequestLogger  # noqa: F401
from headroom.proxy.semantic_cache import SemanticCache  # noqa: F401
from headroom.proxy.warmup import WarmupRegistry
from headroom.proxy.ws_session_registry import WebSocketSessionRegistry
from headroom.subscription.base import get_quota_registry, reset_quota_registry
from headroom.subscription.codex_rate_limits import get_codex_rate_limit_state
from headroom.subscription.copilot_quota import get_copilot_quota_tracker
from headroom.subscription.tracker import (
    configure_subscription_tracker,
    get_subscription_tracker,
)
from headroom.telemetry import get_telemetry_collector
from headroom.telemetry.beacon import is_telemetry_enabled
from headroom.telemetry.toin import get_toin
from headroom.transforms import (
    CacheAligner,
    CodeAwareCompressor,
    CodeCompressorConfig,
    ContentRouter,
    ContentRouterConfig,
    SmartCrusher,
    TransformPipeline,
    is_tree_sitter_available,
)

AnyLLMBackend: Any = None
LiteLLMBackend: Any = None

fcntl: Any = None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

_build_prefix_cache_stats = build_prefix_cache_stats
_build_session_summary = build_session_summary
_merge_cost_stats = merge_cost_stats


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("headroom.proxy")

_MULTI_WORKER_CONFIG_ENV = "HEADROOM_PROXY_CONFIG_JSON"

# Env var that opts out of the Rust core deployment smoke test (Hotfix-A0).
# Default behavior: hard-fail at startup if `headroom._core` is unimportable
# (Finding #2 in HEADROOM_PROXY_LOG_FINDINGS_2026_05_03.md — production
# deployment was silently running without the Rust extension and degrading
# every compressed request to a Python-only path or a no-op).
#
# Set to the literal string "false" to start the proxy in degraded
# Python-only mode. Any other value (including unset) keeps the
# fail-loud behavior.
_RUST_CORE_REQUIRED_ENV = "HEADROOM_REQUIRE_RUST_CORE"

# sysexits.h(3) — EX_CONFIG. Process supervisors (systemd, k8s, docker)
# treat this as a deliberate configuration failure rather than a crash, so
# they won't restart-loop on a broken deployment.
_EXIT_CONFIG = 78


def _check_rust_core() -> tuple[str, str | None]:
    """Verify the Rust extension `headroom._core` is loadable at startup.

    Returns a `(status, error)` tuple:
      - ``("loaded", None)``     — `headroom._core.hello()` returned the
        expected sentinel.
      - ``("disabled", reason)`` — opt-out env var was set; proxy starts
        in Python-only degraded mode. `reason` carries the underlying
        import error (or ``None`` if the import actually succeeded).
      - ``("missing", reason)``  — never returned: this branch calls
        ``sys.exit(78)`` so the proxy refuses to start. The branch exists
        only as a typed sentinel for callers that want to reason about
        all three states (e.g. health endpoints).

    Behavior is gated by the ``HEADROOM_REQUIRE_RUST_CORE`` env var:
    any value other than ``"false"`` (case-insensitive) keeps the
    fail-loud default.
    """
    require = os.environ.get(_RUST_CORE_REQUIRED_ENV, "true").strip().lower() != "false"
    try:
        from headroom._core import hello as _rust_hello

        marker = _rust_hello()
    except Exception as exc:  # ImportError, but also any init-time PyO3 failure
        reason = f"{type(exc).__name__}: {exc}"
        if not require:
            logger.warning(
                "event=rust_core_disabled reason=%r opt_out_env=%s=false mode=python_only_degraded",
                reason,
                _RUST_CORE_REQUIRED_ENV,
            )
            return ("disabled", reason)
        # Fail loud. Print to stderr in addition to logging so operators
        # see it even if the logging handler is mis-configured.
        msg = (
            f"FATAL: Rust extension `headroom._core` not loadable.\n"
            f"    error: {reason}\n"
            f"    fix:   `make build-wheel && pip install --force-reinstall "
            f"target/wheels/headroom_*.whl`\n"
            f"    opt-out: set {_RUST_CORE_REQUIRED_ENV}=false to start in "
            f"degraded Python-only mode\n"
        )
        logger.error("event=rust_core_missing reason=%r action=exit_78", reason)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(_EXIT_CONFIG)

    # Import succeeded; sanity-check the marker so we catch a stale or
    # mis-linked .so where the symbol name resolves but returns garbage.
    if marker != "headroom-core":
        reason = f"unexpected marker {marker!r}"
        if not require:
            logger.warning(
                "event=rust_core_disabled reason=%r opt_out_env=%s=false",
                reason,
                _RUST_CORE_REQUIRED_ENV,
            )
            return ("disabled", reason)
        msg = (
            f"FATAL: Rust extension `headroom._core` is loaded but the "
            f"marker function returned {marker!r}; expected 'headroom-core'.\n"
            f"    fix:   rebuild: `make build-wheel && pip install "
            f"--force-reinstall target/wheels/headroom_*.whl`\n"
        )
        logger.error("event=rust_core_marker_mismatch marker=%r action=exit_78", marker)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(_EXIT_CONFIG)

    logger.info("event=rust_core_loaded marker=%r", marker)
    return ("loaded", None)


# Compression pipeline timeout in seconds


from headroom.proxy.handlers import (  # noqa: E402
    AnthropicHandlerMixin,
    BatchHandlerMixin,
    GeminiHandlerMixin,
    OpenAIHandlerMixin,
    StreamingMixin,
)


class HeadroomProxy(
    StreamingMixin,
    AnthropicHandlerMixin,
    OpenAIHandlerMixin,
    GeminiHandlerMixin,
    BatchHandlerMixin,
):
    """Production-ready Headroom optimization proxy."""

    ANTHROPIC_API_URL = DEFAULT_ANTHROPIC_API_URL
    OPENAI_API_URL = DEFAULT_OPENAI_API_URL
    GEMINI_API_URL = DEFAULT_GEMINI_API_URL
    CLOUDCODE_API_URL = DEFAULT_CLOUDCODE_API_URL

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.config.mode = normalize_proxy_mode(self.config.mode)
        self.pipeline_extensions = PipelineExtensionManager(
            hooks=config.hooks,
            extensions=config.pipeline_extensions,
            discover=config.discover_pipeline_extensions,
        )

        self.provider_runtime = build_proxy_provider_runtime(config)
        api_targets = self.provider_runtime.api_targets

        # Preserve the long-standing proxy compatibility surface while keeping
        # provider_runtime as the source of truth for resolved upstream targets.
        HeadroomProxy.ANTHROPIC_API_URL = api_targets.anthropic
        HeadroomProxy.OPENAI_API_URL = api_targets.openai
        HeadroomProxy.GEMINI_API_URL = api_targets.gemini
        HeadroomProxy.CLOUDCODE_API_URL = api_targets.cloudcode
        self.anthropic_provider = self.provider_runtime.pipeline_provider("anthropic")
        self.openai_provider = self.provider_runtime.pipeline_provider("openai")

        # `metrics` is hoisted ahead of transform construction so the
        # transforms can receive `self.metrics` as their compression
        # observer at __init__ time. The forcing function for catching
        # silent strategy regressions: per-strategy counters increment
        # only when wired up here, so the wiring is mandatory, not
        # something we patch in later. (See `RUST_DEV.md` audit notes.)
        self.cost_tracker = (
            CostTracker(
                budget_limit_usd=config.budget_limit_usd,
                budget_period=config.budget_period,
            )
            if config.cost_tracking_enabled
            else None
        )
        self.metrics = PrometheusMetrics(cost_tracker=self.cost_tracker)

        # Initialize transforms based on routing mode.
        #
        # Phase B PR-B1 retired the IntelligentContextManager / RollingWindow
        # message-dropping branch. Live-zone-only compression (PR-B2..B7) does
        # not drop messages — it operates on content blocks within messages —
        # so the proxy no longer needs a "context manager" transform stage.
        # Reported via metrics as `_context_manager_status = "passthrough"`.
        self._context_manager_status = "passthrough"

        if config.smart_routing:
            # Smart routing: ContentRouter handles all content types intelligently
            # It lazy-loads compressors only when needed
            router_config = ContentRouterConfig(
                enable_code_aware=config.code_aware_enabled,
                tool_profiles=config.tool_profiles,
                read_lifecycle=ReadLifecycleConfig(enabled=config.read_lifecycle),
            )
            # Token mode: allow compression of older excluded-tool results
            if is_token_mode(config.mode):
                router_config.protect_recent_reads_fraction = 0.3
            transforms = [
                CacheAligner(CacheAlignerConfig(enabled=False)),
                ContentRouter(router_config, observer=self.metrics),
            ]
            self._code_aware_status = "lazy" if config.code_aware_enabled else "disabled"
        else:
            # Legacy mode: sequential pipeline
            transforms = [
                CacheAligner(CacheAlignerConfig(enabled=False)),
                SmartCrusher(
                    SmartCrusherConfig(  # type: ignore[arg-type]
                        enabled=True,
                        min_tokens_to_crush=config.min_tokens_to_crush,
                        max_items_after_crush=config.max_items_after_crush,
                    ),
                    ccr_config=CCRConfig(
                        enabled=config.ccr_inject_tool,
                        inject_retrieval_marker=config.ccr_inject_tool,  # Add CCR markers
                    ),
                    observer=self.metrics,
                ),
            ]
            # Add CodeAware if enabled and available
            self._code_aware_status = self._setup_code_aware(config, transforms)

        self.anthropic_pipeline = TransformPipeline(
            transforms=transforms,
            provider=self.anthropic_provider,
        )
        self.openai_pipeline = TransformPipeline(
            transforms=transforms,
            provider=self.openai_provider,
        )

        # Initialize components
        self.cache = (
            SemanticCache(
                max_entries=config.cache_max_entries,
                ttl_seconds=config.cache_ttl_seconds,
            )
            if config.cache_enabled
            else None
        )

        self.rate_limiter = (
            TokenBucketRateLimiter(
                requests_per_minute=config.rate_limit_requests_per_minute,
                tokens_per_minute=config.rate_limit_tokens_per_minute,
            )
            if config.rate_limit_enabled
            else None
        )

        # `cost_tracker` and `metrics` were hoisted to before transforms so
        # ContentRouter / SmartCrusher could take `self.metrics` as their
        # compression observer at __init__ time.

        # Prefix cache tracking: freeze already-cached messages to avoid
        # invalidating the provider's prefix cache with our transforms
        from headroom.cache.prefix_tracker import PrefixFreezeConfig, SessionTrackerStore

        self.session_tracker_store = SessionTrackerStore(
            default_config=PrefixFreezeConfig(
                enabled=config.prefix_freeze_enabled,
                session_ttl_seconds=config.prefix_freeze_session_ttl,
            )
        )

        # Compression cache store for token mode (session-scoped). The dict
        # itself is mutated under `_compression_caches_lock`; the per-session
        # `CompressionCache` instances have their own internal lock guarding
        # `_cache`/`_stable_hashes`/`_first_seen` against concurrent
        # async-dispatched requests for the same session.
        self._compression_caches: dict[str, CompressionCache] = {}
        self._compression_caches_lock = threading.RLock()

        self.logger = (
            RequestLogger(
                log_file=config.log_file,
                log_full_messages=config.log_full_messages,
            )
            if config.log_requests
            else None
        )

        # Enterprise security plugin (loaded dynamically if available + licensed)
        self.security = None

        # HTTP client
        self.http_client: httpx.AsyncClient | None = None

        # Shared cold-start warmup registry (populated by startup()).
        # Holds typed slots with loaded / loading / null / error status for
        # each preloaded heavy asset. Exposed as ``proxy.warmup`` and
        # serialized by the /debug/warmup route (Unit 5).
        self.warmup: WarmupRegistry = WarmupRegistry()
        # Unit 3: live registry of Codex WS sessions. Populated by
        # ``handle_openai_responses_ws`` on accept; drained in its
        # outermost ``finally``. Consumed by ``/debug/ws-sessions``.
        self.ws_sessions: WebSocketSessionRegistry = WebSocketSessionRegistry()

        # Unit 4: bounded pre-upstream concurrency for the Anthropic HTTP
        # path. Caps how many ``handle_anthropic_messages`` calls may be
        # running deep-copy / first-stage compression / memory-context
        # lookup / upstream connect concurrently. ``/livez``, ``/readyz``,
        # ``/health``, ``/metrics``, ``/stats``, and the Codex WS path are
        # intentionally NOT gated by this semaphore.
        #
        # A value of ``0`` or negative disables the semaphore (unbounded
        # mode); this is useful for the Unit 6 counter-factual where we
        # deliberately reproduce the original starvation. The default is
        # ``max(2, min(8, os.cpu_count() or 4))``.
        _pre_upstream_cfg = config.anthropic_pre_upstream_concurrency
        if _pre_upstream_cfg is None:
            _pre_upstream_resolved = max(2, min(8, os.cpu_count() or 4))
        else:
            _pre_upstream_resolved = _pre_upstream_cfg
        self.anthropic_pre_upstream_concurrency: int = _pre_upstream_resolved
        self.anthropic_pre_upstream_acquire_timeout_seconds = float(
            config.anthropic_pre_upstream_acquire_timeout_seconds
        )
        self.anthropic_pre_upstream_memory_context_timeout_seconds = float(
            config.anthropic_pre_upstream_memory_context_timeout_seconds
        )
        if _pre_upstream_resolved > 0:
            self.anthropic_pre_upstream_sem: asyncio.Semaphore | None = asyncio.Semaphore(
                _pre_upstream_resolved
            )
        else:
            self.anthropic_pre_upstream_sem = None

        # Dedicated compression executor — see C3 in the audit followup.
        # Replaces ``asyncio.to_thread(...)`` for ``pipeline.apply()`` calls
        # so that:
        #   1. Compression work is bounded — CPU-bound Rust runs here, and
        #      bursts cannot starve other ``asyncio.to_thread`` callers
        #      sharing the loop's default executor (file IO, etc.).
        #   2. Tasks that exceed ``COMPRESSION_TIMEOUT_SECONDS`` and complete
        #      *after* the asyncio future was cancelled are counted in the
        #      ``compression_leaked_threads`` gauge — Python cannot preempt
        #      the worker, so this is the only signal that some pool slots
        #      are sitting on stuck work.
        _compression_max_cfg = config.compression_max_workers
        if _compression_max_cfg is None:
            _compression_max = min(32, (os.cpu_count() or 1) * 4)
        else:
            _compression_max = max(1, _compression_max_cfg)
        self.compression_max_workers: int = _compression_max
        self._compression_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_compression_max,
            thread_name_prefix="headroom-compress",
        )
        # Gauge: currently-running compression tasks. Mutated under
        # ``_compression_metrics_lock`` from worker threads + the asyncio
        # event loop.
        self._compression_in_flight: int = 0
        # High-water mark for in-flight count.
        self._compression_in_flight_max: int = 0
        # Counter: threads that finished AFTER their asyncio future hit the
        # timeout. Stuck-thread leak indicator.
        self._compression_leaked_threads: int = 0
        self._compression_metrics_lock = threading.Lock()

        # Backend for Anthropic API (direct, LiteLLM, or any-llm)
        # Supports: "anthropic" (direct), "bedrock", "vertex", "litellm-<provider>", or "anyllm"
        self.anthropic_backend: Backend | None = create_proxy_backend(
            backend=config.backend,
            anyllm_provider=config.anyllm_provider,
            bedrock_region=config.bedrock_region,
            logger=logger,
            anyllm_backend_cls=AnyLLMBackend,
            litellm_backend_cls=LiteLLMBackend,
        )

        # Request counter for IDs
        self._request_counter = 0
        self._request_counter_lock = asyncio.Lock()

        # CCR tool injectors (one per provider)
        self.anthropic_tool_injector = CCRToolInjector(
            provider="anthropic",
            inject_tool=config.ccr_inject_tool,
            inject_system_instructions=config.ccr_inject_system_instructions,
        )
        self.openai_tool_injector = CCRToolInjector(
            provider="openai",
            inject_tool=config.ccr_inject_tool,
            inject_system_instructions=config.ccr_inject_system_instructions,
        )

        # CCR Response Handler (handles CCR tool calls automatically)
        self.ccr_response_handler = (
            CCRResponseHandler(
                ResponseHandlerConfig(
                    enabled=True,
                    max_retrieval_rounds=config.ccr_max_retrieval_rounds,
                )
            )
            if config.ccr_handle_responses
            else None
        )

        # CCR Context Tracker (tracks compressed content across turns)
        self.ccr_context_tracker = (
            ContextTracker(
                ContextTrackerConfig(
                    enabled=True,
                    proactive_expansion=config.ccr_proactive_expansion,
                    max_proactive_expansions=config.ccr_max_proactive_expansions,
                )
            )
            if config.ccr_context_tracking
            else None
        )

        # Turn counter for context tracking
        self._turn_counter = 0

        # Memory Handler (persistent user memory)
        self.memory_handler: MemoryHandler | None = None
        if config.memory_enabled:
            # Resolve memory DB path: empty → project-scoped default
            _mem_db_path = config.memory_db_path
            if not _mem_db_path:
                _mem_dir = Path.cwd() / ".headroom"
                _mem_dir.mkdir(parents=True, exist_ok=True)
                _mem_db_path = str(_mem_dir / "memory.db")
                logger.info(f"Memory: Project-scoped DB at {_mem_db_path}")

            # PR-B6: translate the string-typed ``ProxyConfig.memory_mode``
            # into the typed ``MemoryMode`` enum. Unknown values raise
            # loudly per the no-silent-fallback policy.
            from headroom.proxy.memory_handler import MemoryMode

            try:
                _memory_mode = MemoryMode(config.memory_mode)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid memory_mode={config.memory_mode!r}; "
                    f"expected one of {[m.value for m in MemoryMode]}"
                ) from exc

            memory_config = MemoryConfig(
                enabled=True,
                backend=config.memory_backend,
                db_path=_mem_db_path,
                inject_tools=config.memory_inject_tools,
                use_native_tool=config.memory_use_native_tool,
                inject_context=config.memory_inject_context,
                top_k=config.memory_top_k,
                min_similarity=config.memory_min_similarity,
                mode=_memory_mode,
                qdrant_url=config.memory_qdrant_url,
                qdrant_host=config.memory_qdrant_host,
                qdrant_port=config.memory_qdrant_port,
                qdrant_api_key=config.memory_qdrant_api_key,
                neo4j_uri=config.memory_neo4j_uri,
                neo4j_user=config.memory_neo4j_user,
                neo4j_password=config.memory_neo4j_password,
                bridge_enabled=config.memory_bridge_enabled,
                bridge_md_paths=config.memory_bridge_md_paths,
                bridge_md_format=config.memory_bridge_md_format,
                bridge_auto_import=config.memory_bridge_auto_import,
                bridge_export_path=config.memory_bridge_export_path,
            )
            self.memory_handler = MemoryHandler(
                memory_config,
                agent_type=config.traffic_learning_agent_type,
            )

        # Usage Reporter (license validation + phone-home for managed/enterprise)
        self.usage_reporter: UsageReporter | None = None
        if config.license_key:
            from headroom.telemetry.reporter import UsageReporter

            self.usage_reporter = UsageReporter(
                license_key=config.license_key,
                cloud_url=config.license_cloud_url,
                report_interval=config.license_report_interval,
            )

        # Traffic Learner (live pattern extraction from proxy traffic)
        # Only activates with --learn flag; requires --memory for backend
        self.traffic_learner: TrafficLearner | None = None
        self.traffic_learning_agent_type: str = config.traffic_learning_agent_type
        if config.traffic_learning_enabled:
            from headroom.memory.traffic_learner import TrafficLearner

            self.traffic_learner = TrafficLearner(
                user_id=os.environ.get("HEADROOM_USER_ID", os.environ.get("USER", "default")),
                agent_type=config.traffic_learning_agent_type,
                min_evidence=config.traffic_learning_min_evidence,
            )

        # Code graph file watcher (live reindex on file changes)
        self.code_graph_watcher: CodeGraphWatcher | None = None  # type: ignore[annotation-unchecked]
        if config.code_graph_watcher:
            from headroom.graph.watcher import CodeGraphWatcher

            self.code_graph_watcher = CodeGraphWatcher(project_dir=Path.cwd())
            if self.code_graph_watcher.start():
                logger.info("Code graph: file watcher started")
            else:
                self.code_graph_watcher = None

        self.pipeline_extensions.emit(
            PipelineStage.SETUP,
            operation="proxy.setup",
            metadata={
                "mode": self.config.mode,
                "optimize": self.config.optimize,
                "backend": self.config.backend,
                "memory_enabled": self.config.memory_enabled,
            },
        )

    async def _run_compression_in_executor(
        self,
        fn,  # noqa: ANN001 — caller-supplied no-arg sync callable
        *,
        timeout: float,
    ):
        """Run a synchronous compression callable on the bounded executor
        with cancel-aware metrics.

        Replaces ``asyncio.wait_for(asyncio.to_thread(fn), timeout=...)``.

        Why a dedicated executor: the proxy's compression path is CPU-bound
        Rust work that releases the GIL via ``py.allow_threads``. Sharing
        the loop's default executor (used by ``asyncio.to_thread``) means
        a burst of slow compressions can starve unrelated ``to_thread``
        callers (file IO, etc.). The compression executor is sized
        independently via ``config.compression_max_workers``.

        Why "cancel-aware metrics": when ``asyncio.wait_for`` times out, it
        cancels the *asyncio future*. The underlying
        ``concurrent.futures.Future`` from ``run_in_executor`` cannot
        actually cancel a thread that has started — Python has no way to
        preempt running CPython bytecode or in-flight Rust calls. The
        worker keeps running to completion, ignored. We detect this by
        recording the start timestamp and incrementing
        ``_compression_leaked_threads`` from the worker's ``finally``
        block when ``elapsed > timeout``. Operators can see leaked-thread
        rate climbing in ``/stats`` before the pool fills up.

        Args:
            fn: A no-arg sync callable that runs the compression. Must not
                raise asyncio Cancellation; if it does, the wrapper still
                decrements the in-flight gauge but the leaked-thread
                counter may double-count.
            timeout: Wall-clock timeout for the asyncio side. The
                executor worker keeps running past this (Python limitation
                — see above), but at least the awaiter unblocks.

        Returns:
            Whatever ``fn()`` returns.

        Raises:
            ``asyncio.TimeoutError`` if the callable doesn't return within
            ``timeout``. Any exception raised by ``fn`` propagates
            unchanged.
        """
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        with self._compression_metrics_lock:
            self._compression_in_flight += 1
            if self._compression_in_flight > self._compression_in_flight_max:
                self._compression_in_flight_max = self._compression_in_flight

        def _wrapped():  # noqa: ANN202
            try:
                return fn()
            finally:
                elapsed = time.monotonic() - start
                with self._compression_metrics_lock:
                    self._compression_in_flight -= 1
                    if elapsed > timeout:
                        self._compression_leaked_threads += 1

        future = loop.run_in_executor(self._compression_executor, _wrapped)
        return await asyncio.wait_for(future, timeout=timeout)

    def _get_compression_cache(self, session_id: str) -> CompressionCache:
        """Get or create a CompressionCache for a session.

        Thread-safe under `_compression_caches_lock`: a concurrent pair of
        `_get_compression_cache(session_id)` calls (e.g. two async requests
        for the same conversation) must return the **same** instance,
        otherwise the per-session cache state splits and the two halves
        diverge across requests.
        """
        with self._compression_caches_lock:
            if session_id not in self._compression_caches:
                from headroom.cache.compression_cache import CompressionCache

                # Evict oldest caches if at capacity
                if len(self._compression_caches) >= MAX_COMPRESSION_CACHE_SESSIONS:
                    # Remove oldest quarter to amortize cleanup cost
                    oldest_keys = list(self._compression_caches.keys())[
                        : MAX_COMPRESSION_CACHE_SESSIONS // 4
                    ]
                    for key in oldest_keys:
                        del self._compression_caches[key]
                    logger.info(
                        "Evicted %d compression caches (exceeded %d max sessions)",
                        len(oldest_keys),
                        MAX_COMPRESSION_CACHE_SESSIONS,
                    )

                self._compression_caches[session_id] = CompressionCache()
            return self._compression_caches[session_id]

    def _setup_code_aware(self, config: ProxyConfig, transforms: list) -> str:
        """Set up code-aware compression if enabled.

        Args:
            config: Proxy configuration
            transforms: Transform list to append to

        Returns:
            Status string for logging: 'enabled', 'disabled', 'available', 'unavailable'
        """
        if config.code_aware_enabled:
            if is_tree_sitter_available():
                code_config = CodeCompressorConfig(
                    preserve_imports=True,
                    preserve_signatures=True,
                    preserve_type_annotations=True,
                )
                # CodeAware runs after the content/structure transforms.
                # Phase B PR-B1 retired the trailing context_manager so we
                # append rather than insert(-1).
                transforms.append(CodeAwareCompressor(code_config))
                return "enabled"
            else:
                logger.warning(
                    "Code-aware compression requested but tree-sitter not installed. "
                    "Install with: pip install headroom-ai[code]"
                )
                return "unavailable"
        else:
            if is_tree_sitter_available():
                return "available"  # Available but not enabled
            return "disabled"

    async def startup(self):
        """Initialize async resources."""
        self.pipeline_extensions.emit(
            PipelineStage.PRE_START,
            operation="proxy.startup",
            metadata={"port": self.config.port, "host": self.config.host},
        )
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout_seconds,
                read=self.config.request_timeout_seconds,
                write=self.config.request_timeout_seconds,
                pool=self.config.connect_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
            ),
            http2=self.config.http2,
        )
        logger.info("Headroom Proxy started")
        logger.info(f"Optimization: {'ENABLED' if self.config.optimize else 'DISABLED'}")
        self.config.mode = normalize_proxy_mode(self.config.mode)
        logger.info(f"Mode: {self.config.mode}")
        if self.config.mode == PROXY_MODE_TOKEN:
            logger.info("  Prefix freeze: re-freeze after compression")
            logger.info("  Read protection window: 30%% of excluded-tool messages")
            logger.info("  CCR TTL: extended for session lifetime")
            logger.info("  Compression cache: active")
        if self.config.mode == PROXY_MODE_CACHE:
            logger.info("  Prefix freeze: strict (all prior turns immutable)")
            logger.info("  Mutations: latest turn only")
        logger.info(f"Caching: {'ENABLED' if self.config.cache_enabled else 'DISABLED'}")
        logger.info(f"Rate Limiting: {'ENABLED' if self.config.rate_limit_enabled else 'DISABLED'}")
        logger.info(
            f"Connection Pool: max_connections={self.config.max_connections}, "
            f"max_keepalive={self.config.max_keepalive_connections}, "
            f"http2={'ENABLED' if self.config.http2 else 'DISABLED'}"
        )

        # Unit 4 pre-upstream concurrency announcement. Report the resolved
        # value (auto-detected vs. explicit) so operators can correlate
        # ``pre_upstream_wait_ms`` log lines with the configured cap.
        if self.anthropic_pre_upstream_sem is None:
            logger.info("Anthropic pre-upstream concurrency: unbounded (explicitly disabled)")
        else:
            _explicit = self.config.anthropic_pre_upstream_concurrency
            _origin = "auto-detected" if _explicit is None else "explicit"
            logger.info(
                "Anthropic pre-upstream concurrency: %d (%s)",
                self.anthropic_pre_upstream_concurrency,
                _origin,
            )
        logger.info(
            "Anthropic pre-upstream timeouts: acquire=%.1fs compression=%.1fs memory_context=%.1fs",
            self.anthropic_pre_upstream_acquire_timeout_seconds,
            float(COMPRESSION_TIMEOUT_SECONDS),
            self.anthropic_pre_upstream_memory_context_timeout_seconds,
        )

        # Smart routing status
        if self.config.smart_routing:
            logger.info("Smart Routing: ENABLED (intelligent content detection)")
        else:
            logger.info("Smart Routing: DISABLED (legacy sequential mode)")

        # Eagerly load ALL compressors, parsers, and detectors at startup
        # This eliminates cold-start latency spikes on first requests.
        # Iterate BOTH pipelines (Anthropic + OpenAI) and dedupe transforms
        # by id() so shared-transform instances never load twice. The
        # resulting status dict is merged into ``self.warmup`` so /debug/warmup
        # (Unit 5) and /readyz have a single source of truth.
        self._kompress_status = "not installed"
        eager_status: dict[str, str] = {}

        if self.config.optimize:
            logger.info("Pre-loading compressors and parsers...")
            seen_transform_ids: set[int] = set()
            pipelines = (self.anthropic_pipeline, self.openai_pipeline)
            for pipeline in pipelines:
                for transform in pipeline.transforms:
                    if id(transform) in seen_transform_ids:
                        continue
                    seen_transform_ids.add(id(transform))
                    if not hasattr(transform, "eager_load_compressors"):
                        continue
                    try:
                        transform_status = transform.eager_load_compressors()
                    except Exception as exc:
                        logger.warning(
                            "Eager preload failed for %s: %s",
                            type(transform).__name__,
                            exc,
                        )
                        continue
                    if not isinstance(transform_status, dict):
                        continue
                    # Merge: later writers win only if the key wasn't set.
                    # Preload a transform ONCE — if another pipeline also has
                    # ``eager_load_compressors`` it contributes only new keys.
                    for key, value in transform_status.items():
                        eager_status.setdefault(key, value)
                    self.warmup.merge_transform_status(transform_status)

        # Update internal status from eager loading results
        if eager_status.get("kompress") == "enabled":
            self._kompress_status = "enabled"
        if eager_status.get("code_aware") == "enabled":
            self._code_aware_status = "enabled"

        # Log component status
        if self._kompress_status == "enabled":
            logger.info("Kompress: ENABLED (ModernBERT token compressor)")
        elif self.config.optimize:
            logger.info("Kompress: not installed (pip install headroom-ai[ml] for ML compression)")

        if self._code_aware_status == "enabled":
            logger.info("Code-Aware: ENABLED (AST-based compression)")
            if "tree_sitter" in eager_status:
                logger.info(f"Tree-Sitter: {eager_status['tree_sitter']}")
        elif self._code_aware_status == "lazy":
            logger.info("Code-Aware: LAZY (will load when code content detected)")
        elif self._code_aware_status == "available":
            logger.info("Code-Aware: available but disabled (use --code-aware)")
        elif self._code_aware_status == "unavailable":
            logger.info("Code-Aware: not installed (pip install headroom-ai[code])")
        elif self._code_aware_status == "disabled":
            logger.info("Code-Aware: DISABLED")

        if eager_status.get("magika") == "enabled":
            logger.info("Magika: ENABLED (ML content detection)")

        if self.memory_handler:
            self.warmup.memory_backend.mark_loading()
            try:
                await self.memory_handler.ensure_initialized()
            except Exception as exc:  # pragma: no cover - defensive
                self.warmup.memory_backend.mark_error(str(exc))
                logger.warning("Memory: backend initialization failed (startup continues): %s", exc)
            memory_status = self.memory_handler.health_status()
            if memory_status.get("initialized"):
                self.warmup.memory_backend.mark_loaded(
                    handle=self.memory_handler,
                    backend=memory_status.get("backend"),
                )
                # Force one embed call so the ONNX graph is compiled now,
                # not lazily during the first request. Best-effort — any
                # failure is swallowed inside warmup_embedder.
                self.warmup.memory_embedder.mark_loading()
                warmed = await self.memory_handler.warmup_embedder()
                if warmed:
                    self.warmup.memory_embedder.mark_loaded()
                else:
                    # Not an error — e.g. qdrant-neo4j has no embedder slot
                    # we can reach, or the backend simply exposes no handle.
                    self.warmup.memory_embedder.mark_null()
            else:
                if self.warmup.memory_backend.status != "error":
                    self.warmup.memory_backend.mark_null()
                self.warmup.memory_embedder.mark_null()
            logger.info(
                "Memory: ENABLED "
                f"(backend={memory_status['backend']}, initialized={memory_status['initialized']})"
            )
        else:
            logger.info("Memory: DISABLED")

        # CCR status
        ccr_features = []
        if self.config.ccr_inject_tool:
            ccr_features.append("tool_injection")
        if self.config.ccr_handle_responses:
            ccr_features.append("response_handling")
        if self.config.ccr_context_tracking:
            ccr_features.append("context_tracking")
        if self.config.ccr_proactive_expansion:
            ccr_features.append("proactive_expansion")
        if ccr_features:
            logger.info(f"CCR (Compress-Cache-Retrieve): ENABLED ({', '.join(ccr_features)})")
        else:
            logger.info("CCR: DISABLED")
        logger.info(f"Savings history: {self.metrics.savings_tracker.storage_path}")

        # Reset and rebuild the quota tracker registry for this server instance.
        # reset_quota_registry() ensures a clean slate when the proxy is restarted
        # (e.g. in tests that spin up multiple app instances in the same process).
        reset_quota_registry()
        registry = get_quota_registry()
        tracker = configure_subscription_tracker(
            poll_interval_s=self.config.subscription_poll_interval_s,
            active_window_s=self.config.subscription_active_window_s,
            enabled=self.config.subscription_tracking_enabled,
        )
        registry.register(tracker)
        registry.register(get_codex_rate_limit_state())
        registry.register(get_copilot_quota_tracker())
        await registry.start_all()

        if self.config.subscription_tracking_enabled:
            logger.info(
                "Subscription tracking: ENABLED "
                f"(poll_interval={self.config.subscription_poll_interval_s}s, "
                f"active_window={self.config.subscription_active_window_s}s)"
            )
        else:
            logger.info("Subscription tracking: DISABLED")

        copilot_tracker = get_copilot_quota_tracker()
        if copilot_tracker.is_available():
            logger.info("GitHub Copilot quota tracking: ENABLED")
        else:
            logger.info(
                "GitHub Copilot quota tracking: DISABLED "
                "(set GITHUB_TOKEN or GITHUB_COPILOT_GITHUB_TOKEN to enable)"
            )

        # Log anonymous telemetry status so operators can see it in the log stream
        if is_telemetry_enabled():
            logger.info(
                "Anonymous telemetry: ENABLED (aggregate stats only — no prompts or content). "
                "Opt out: HEADROOM_TELEMETRY=off or --no-telemetry"
            )
        else:
            logger.info("Anonymous telemetry: DISABLED")

        self.pipeline_extensions.emit(
            PipelineStage.POST_START,
            operation="proxy.startup",
            metadata={
                "port": self.config.port,
                "host": self.config.host,
                "warmup": self.warmup.to_dict(),
            },
        )

    async def shutdown(self):
        """Cleanup async resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        if self.memory_handler and hasattr(self.memory_handler, "close"):
            await self.memory_handler.close()

        with contextlib.suppress(Exception):
            from headroom.models.ml_models import MLModelRegistry

            released_models = []
            released_models.extend(MLModelRegistry.unload_prefix("technique_router:"))
            released_models.extend(MLModelRegistry.unload_prefix("siglip:"))
            if released_models:
                logger.info("Released image optimizer models: %s", ", ".join(released_models))

        # Stop all quota trackers via the registry
        await get_quota_registry().stop_all()

        # Print final stats
        self._print_summary()

    def _print_summary(self):
        """Print session summary."""
        m = self.metrics
        logger.info("=" * 70)
        logger.info("HEADROOM PROXY SESSION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total requests:        {m.requests_total}")
        logger.info(f"Cached responses:      {m.requests_cached}")
        logger.info(f"Rate limited:          {m.requests_rate_limited}")
        logger.info(f"Failed:                {m.requests_failed}")
        logger.info(f"Input tokens:          {m.tokens_input_total:,}")
        logger.info(f"Output tokens:         {m.tokens_output_total:,}")
        logger.info(f"Tokens saved:          {m.tokens_saved_total:,}")
        if m.tokens_input_total > 0:
            savings_pct = (
                m.tokens_saved_total / (m.tokens_input_total + m.tokens_saved_total)
            ) * 100
            logger.info(f"Token savings:         {savings_pct:.1f}%")
        if m.latency_count > 0:
            avg_latency = m.latency_sum_ms / m.latency_count
            logger.info(f"Avg latency:           {avg_latency:.0f}ms")
        logger.info("=" * 70)

    async def _next_request_id(self) -> str:
        """Generate unique request ID."""
        async with self._request_counter_lock:
            self._request_counter += 1
            return f"hr_{int(time.time())}_{self._request_counter:06d}"

    def _extract_tags(self, headers: dict) -> dict[str, str]:
        """Extract Headroom tags from headers."""
        tags = {}
        for key, value in headers.items():
            if key.lower().startswith("x-headroom-"):
                tag_name = key.lower().replace("x-headroom-", "")
                tags[tag_name] = value
        return tags

    async def _retry_request(
        self,
        method: str,
        url: str,
        headers: dict,
        body: dict,
        stream: bool = False,
        *,
        original_body_bytes: bytes | None = None,
        body_mutated: bool = True,
        mutation_reasons: list[str] | None = None,
        request_id: str | None = None,
        forwarder_name: str = "server",
        path_for_log: str | None = None,
    ) -> httpx.Response:
        """Make request with retry and exponential backoff.

        Byte-faithful forwarding (PR-A3, fixes P0-2):
          * If ``original_body_bytes`` is provided AND ``body_mutated`` is
            ``False``, the original bytes are forwarded verbatim. SHA-256
            of upstream-received bytes equals client-sent bytes.
          * Otherwise the body dict is canonically re-serialized via
            ``serialize_body_canonical`` (compact separators, ensure_ascii=False).
          * ``HEADROOM_PROXY_PYTHON_FORWARDER_MODE=legacy_json_kwarg`` is an
            explicit operator opt-in for emergency rollback to the old
            ``httpx ... json=body`` behavior.

        The default ``body_mutated=True`` preserves backward compatibility
        for callers that still pass only ``body`` (e.g. CCR continuations
        construct their body from scratch, so canonical serialization is
        correct and original bytes do not exist).
        """
        from headroom.proxy.helpers import (
            log_outbound_request,
            prepare_outbound_body_bytes,
        )

        last_error = None
        reasons = list(mutation_reasons or [])
        outbound_bytes, source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=body_mutated,
        )
        outbound_headers = {**headers, "content-type": "application/json"}

        log_outbound_request(
            forwarder=forwarder_name,
            method=method,
            path=path_for_log or url,
            body_bytes_count=len(outbound_bytes),
            body_mutated=body_mutated,
            mutation_reasons=reasons,
            request_id=request_id,
            source=source,
        )

        for attempt in range(self.config.retry_max_attempts):
            try:
                if stream:
                    # For streaming, we return early - retry happens at higher level
                    return await self.http_client.post(  # type: ignore[union-attr]
                        url, content=outbound_bytes, headers=outbound_headers
                    )
                else:
                    response = await self.http_client.post(  # type: ignore[union-attr]
                        url, content=outbound_bytes, headers=outbound_headers
                    )

                    # Don't retry client errors (4xx)
                    if 400 <= response.status_code < 500:
                        return response

                    # Retry server errors (5xx)
                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error: {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    return response

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                last_error = e

                if not self.config.retry_enabled or attempt >= self.config.retry_max_attempts - 1:
                    raise

                # Exponential backoff with jitter
                delay_with_jitter = jitter_delay_ms(
                    self.config.retry_base_delay_ms,
                    self.config.retry_max_delay_ms,
                    attempt,
                )

                logger.warning(
                    f"Request failed (attempt {attempt + 1}), retrying in {delay_with_jitter:.0f}ms: {e}"
                )
                await asyncio.sleep(delay_with_jitter / 1000)

        raise last_error  # type: ignore[misc]


async def _log_toin_stats_periodically(interval_seconds: int = 300) -> None:
    """Background task that logs TOIN stats periodically.

    Args:
        interval_seconds: How often to log stats (default: 5 minutes).
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            toin = get_toin()
            stats = toin.get_stats()
            total_compressions = stats.get("total_compressions", 0)
            if total_compressions > 0:
                patterns = stats.get("patterns_tracked", 0)
                retrievals = stats.get("total_retrievals", 0)
                retrieval_rate = stats.get("global_retrieval_rate", 0.0)
                logger.info(
                    "TOIN: %d patterns, %d compressions, %d retrievals, %.1f%% retrieval rate",
                    patterns,
                    total_compressions,
                    retrievals,
                    retrieval_rate * 100,
                )
        except Exception as e:
            logger.debug("Failed to log TOIN stats: %s", e)


def _register_memory_components(proxy: HeadroomProxy, tracker: MemoryTracker) -> None:
    """Register all memory-tracked components with the tracker.

    This function is idempotent - it checks if components are already registered.

    Args:
        proxy: The HeadroomProxy instance.
        tracker: The MemoryTracker instance.
    """
    # Register compression store (global singleton)
    if "compression_store" not in tracker.registered_components:
        store = get_compression_store()
        tracker.register("compression_store", store.get_memory_stats)

    # Register semantic cache (instance on proxy)
    if proxy.cache and "semantic_cache" not in tracker.registered_components:
        tracker.register("semantic_cache", proxy.cache.get_memory_stats)

    # Register request logger (instance on proxy)
    if proxy.logger and "request_logger" not in tracker.registered_components:
        tracker.register("request_logger", proxy.logger.get_memory_stats)

    # Register batch context store (global singleton)
    if "batch_context_store" not in tracker.registered_components:
        try:
            from ..ccr.batch_store import get_batch_context_store

            batch_store = get_batch_context_store()
            if hasattr(batch_store, "get_memory_stats"):
                tracker.register("batch_context_store", batch_store.get_memory_stats)
        except ImportError:
            pass

    # Note: graph_store and vector_index are created per-user within the
    # LocalMemoryBackend, not as global singletons. They would need to be
    # registered when the memory system is initialized with specific backends.


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required. Install: pip install fastapi uvicorn httpx")

    from contextlib import asynccontextmanager

    # Always-on file logging to ~/.headroom/logs/ for `headroom perf` analysis.
    # Installed here (not at module import) so importing headroom.proxy.server
    # in tests or library contexts does not silently attach a RotatingFileHandler
    # to the user's live proxy.log.
    _setup_file_logging()

    config = config or ProxyConfig()
    proxy = HeadroomProxy(config)

    # Telemetry beacon (anonymous aggregate stats).
    # With uvicorn workers > 1, each worker runs the lifespan independently.
    # We must ensure only ONE beacon runs across all workers — otherwise each
    # worker creates its own beacon, spamming the telemetry table with N rows
    # per cycle instead of 1 (all reading the same /stats from the same port).
    #
    # Strategy: use a file lock to ensure only the first worker starts the
    # beacon. Other workers see the lock and skip.
    from headroom.telemetry.beacon import TelemetryBeacon

    _beacon = TelemetryBeacon(
        port=config.port if hasattr(config, "port") else 8787,
        sdk=os.environ.get("HEADROOM_SDK", "proxy").strip() or "proxy",
        backend=config.backend if hasattr(config, "backend") else "anthropic",
    )
    from headroom import paths as _hr_paths

    _beacon_lock_path = _hr_paths.beacon_lock_path(config.port)
    _beacon_lock_fd: list = [None]  # mutable holder for the lock file descriptor
    _beacon_is_owner: list = [False]

    def _try_acquire_beacon_lock() -> bool:
        """Try to acquire the beacon file lock (non-blocking).

        Returns True if this process is the beacon owner.
        """
        if not HAS_FCNTL:
            return True

        fd = None
        try:
            _beacon_lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = open(_beacon_lock_path, "w")  # noqa: SIM115
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd.write(str(os.getpid()))
            fd.flush()
            _beacon_lock_fd[0] = fd
            return True
        except OSError:
            if fd is not None:
                fd.close()
            return False

    def _release_beacon_lock() -> None:
        """Release the beacon file lock."""
        fd = _beacon_lock_fd[0]
        if fd:
            try:
                if HAS_FCNTL:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()
            except Exception:
                pass
            _beacon_lock_fd[0] = None
        try:
            _beacon_lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        # Hotfix-A0: Rust core deployment smoke test. Refuse to accept
        # traffic if the Rust extension is missing unless the operator
        # explicitly opted out with HEADROOM_REQUIRE_RUST_CORE=false. See
        # Finding #2 in HEADROOM_PROXY_LOG_FINDINGS_2026_05_03.md.
        # `_check_rust_core` either returns ("loaded"|"disabled", _) or
        # calls `sys.exit(78)` — execution past this line implies the
        # rust_core_status is recorded.
        _rust_core_status, _rust_core_error = _check_rust_core()
        app.state.rust_core_status = _rust_core_status
        app.state.rust_core_error = _rust_core_error

        configure_otel_metrics(OTelMetricsConfig.from_env(default_service_name="headroom-proxy"))
        configure_langfuse_tracing(
            LangfuseTracingConfig.from_env(default_service_name="headroom-proxy")
        )

        app.state.started_at = time.time()
        app.state.ready = False
        app.state.startup_error = None

        try:
            try:
                # Startup
                await proxy.startup()
                asyncio.create_task(_log_toin_stats_periodically())
                if proxy.usage_reporter:
                    await proxy.usage_reporter.start(proxy)
                if proxy.traffic_learner:
                    await proxy.traffic_learner.start()

                # Only start beacon if we acquire the lock (first worker wins)
                _beacon_is_owner[0] = _try_acquire_beacon_lock()
                if _beacon_is_owner[0]:
                    await _beacon.start()
                else:
                    logger.debug("Beacon: skipping (another worker owns the lock)")

                app.state.ready = True
                yield
            except Exception as exc:
                app.state.startup_error = str(exc)
                raise
        finally:
            app.state.ready = False
            # Shutdown
            if _beacon_is_owner[0]:
                await _beacon.stop()
                _release_beacon_lock()
            if proxy.usage_reporter:
                await proxy.usage_reporter.stop()
            if proxy.traffic_learner:
                await proxy.traffic_learner.stop()
            if proxy.code_graph_watcher:
                proxy.code_graph_watcher.stop()
            await proxy.shutdown()
            shutdown_headroom_tracing()
            shutdown_otel_metrics()

    app = FastAPI(
        title="Headroom Proxy",
        description="Production-ready LLM optimization proxy",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.proxy = proxy
    app.state.started_at = None
    app.state.ready = False
    app.state.startup_error = None
    # Set by the lifespan startup smoke test (`_check_rust_core`). Default
    # "missing" means lifespan hasn't run yet — anything reading /health
    # before startup completes (rare; lifespan runs before the first
    # request) sees an honest "missing" rather than a stale "loaded".
    app.state.rust_core_status = "missing"
    app.state.rust_core_error = None

    def _iso_utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _uptime_seconds() -> float:
        started_at = getattr(app.state, "started_at", None)
        if not isinstance(started_at, int | float):
            return 0.0
        return round(max(0.0, time.time() - float(started_at)), 3)

    def _component_health(
        *,
        enabled: bool,
        ready: bool,
        **details: Any,
    ) -> dict[str, Any]:
        status = "disabled" if not enabled else ("healthy" if ready else "unhealthy")
        return {
            "enabled": enabled,
            "ready": (ready if enabled else True),
            "status": status,
            **details,
        }

    def _health_checks() -> dict[str, dict[str, Any]]:
        memory_status = (
            proxy.memory_handler.health_status()
            if proxy.memory_handler
            else {
                "enabled": False,
                "backend": None,
                "initialized": False,
                "native_tool": False,
                "bridge_enabled": False,
            }
        )
        memory_enabled = bool(memory_status.get("enabled", False))
        memory_initialized = bool(memory_status.get("initialized", False))
        return {
            "startup": _component_health(
                enabled=True,
                ready=bool(getattr(app.state, "ready", False)),
                error=getattr(app.state, "startup_error", None),
            ),
            "http_client": _component_health(
                enabled=True,
                ready=proxy.http_client is not None,
            ),
            "cache": _component_health(
                enabled=config.cache_enabled,
                ready=(proxy.cache is not None),
            ),
            "rate_limiter": _component_health(
                enabled=config.rate_limit_enabled,
                ready=(proxy.rate_limiter is not None),
            ),
            "memory": _component_health(
                enabled=memory_enabled,
                ready=memory_initialized,
                backend=memory_status["backend"],
                initialized=memory_initialized,
                native_tool=bool(memory_status.get("native_tool", False)),
                bridge_enabled=bool(memory_status.get("bridge_enabled", False)),
            ),
        }

    def _runtime_payload() -> dict[str, Any]:
        ws_registry = getattr(proxy, "ws_sessions", None)
        ws_active_sessions = ws_registry.active_count() if ws_registry is not None else 0
        ws_active_relay_tasks = (
            ws_registry.active_relay_task_count() if ws_registry is not None else 0
        )
        # Snapshot compression executor metrics under their lock (gauges
        # mutated by worker threads; not safe to read without).
        with proxy._compression_metrics_lock:
            _comp_in_flight = proxy._compression_in_flight
            _comp_in_flight_max = proxy._compression_in_flight_max
            _comp_leaked = proxy._compression_leaked_threads
        return {
            "anthropic_pre_upstream": {
                "enabled": proxy.anthropic_pre_upstream_sem is not None,
                "resolved_concurrency": proxy.anthropic_pre_upstream_concurrency,
                "source": (
                    "auto" if config.anthropic_pre_upstream_concurrency is None else "explicit"
                ),
                "acquire_timeout_seconds": proxy.anthropic_pre_upstream_acquire_timeout_seconds,
                "compression_timeout_seconds": float(COMPRESSION_TIMEOUT_SECONDS),
                "memory_context_timeout_seconds": (
                    proxy.anthropic_pre_upstream_memory_context_timeout_seconds
                ),
                "codex_ws_gated": False,
            },
            "compression_executor": {
                "max_workers": proxy.compression_max_workers,
                "in_flight": _comp_in_flight,
                "in_flight_max": _comp_in_flight_max,
                "leaked_threads_total": _comp_leaked,
                "source": ("auto" if config.compression_max_workers is None else "explicit"),
            },
            "websocket_sessions": {
                "active_sessions": ws_active_sessions,
                "active_relay_tasks": ws_active_relay_tasks,
            },
        }

    def _health_payload(*, include_config: bool) -> dict[str, Any]:
        checks = _health_checks()
        ready = all(check["ready"] for check in checks.values())
        payload: dict[str, Any] = {
            "service": "headroom-proxy",
            "status": "healthy" if ready else "unhealthy",
            "ready": ready,
            "version": __version__,
            "timestamp": _iso_utc_now(),
            "uptime_seconds": _uptime_seconds(),
            "checks": checks,
            "runtime": _runtime_payload(),
            # Hotfix-A0: surface rust core load state so operators can alert
            # on `rust_core != "loaded"` (Finding #2).
            "rust_core": getattr(app.state, "rust_core_status", "missing"),
        }
        rust_core_error = getattr(app.state, "rust_core_error", None)
        if rust_core_error:
            payload["rust_core_error"] = rust_core_error
        deployment_profile = os.environ.get("HEADROOM_DEPLOYMENT_PROFILE")
        if deployment_profile:
            payload["deployment"] = {
                "profile": deployment_profile,
                "preset": os.environ.get("HEADROOM_DEPLOYMENT_PRESET"),
                "runtime": os.environ.get("HEADROOM_DEPLOYMENT_RUNTIME"),
                "supervisor": os.environ.get("HEADROOM_DEPLOYMENT_SUPERVISOR"),
                "scope": os.environ.get("HEADROOM_DEPLOYMENT_SCOPE"),
            }
        if include_config:
            payload["config"] = {
                "backend": config.backend,
                "optimize": config.optimize,
                "cache": config.cache_enabled,
                "rate_limit": config.rate_limit_enabled,
                "memory": config.memory_enabled,
                "learn": config.traffic_learning_enabled,
                "code_graph": config.code_graph_watcher,
                "pid": os.getpid(),
            }
        return payload

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # X-Headroom-Stack: SDK adapters (TS openai/anthropic/etc.) tag their
    # requests so telemetry can segment by integration surface. Registered
    # before extension middleware so any extension-level auth/guards run
    # outermost and we don't count requests they reject.
    @app.middleware("http")
    async def _record_headroom_stack(request, call_next):
        if request.url.path.startswith("/v1/"):
            stack = request.headers.get("x-headroom-stack")
            if stack:
                try:
                    proxy.metrics.record_stack(stack)
                except Exception:
                    logger.debug("record_stack failed", exc_info=True)
        return await call_next(request)

    # Third-party proxy extensions (Enterprise, custom plugins). Discovered via
    # the `headroom.proxy_extension` entry-point group, but **opt-in only**:
    # only names listed in config.proxy_extensions (CLI: --proxy-extension,
    # env: HEADROOM_PROXY_EXTENSIONS) actually get installed. Discovery alone
    # never runs third-party code. An extension that raises from its install()
    # is a deliberate fail-closed signal and aborts startup.
    from headroom.proxy.extensions import install_all as _install_extensions

    _install_extensions(app, config, enabled=getattr(config, "proxy_extensions", None))

    # Health & Metrics
    @app.get("/livez")
    async def livez():
        return JSONResponse(
            status_code=200,
            content={
                "service": "headroom-proxy",
                "status": "healthy",
                "alive": True,
                "version": __version__,
                "timestamp": _iso_utc_now(),
                "uptime_seconds": _uptime_seconds(),
            },
        )

    @app.get("/readyz")
    async def readyz():
        payload = _health_payload(include_config=False)
        return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)

    @app.get("/health")
    async def health():
        payload = _health_payload(include_config=True)
        return JSONResponse(status_code=200, content=payload)

    # Loopback-only debug introspection (Unit 5). A remote IP gets 404 —
    # debug endpoints are invisible to external scanners.
    from headroom.proxy.debug_introspection import (
        collect_tasks as _collect_tasks,
    )
    from headroom.proxy.loopback_guard import require_loopback as _require_loopback

    @app.get("/debug/tasks", dependencies=[Depends(_require_loopback)])
    async def debug_tasks(stack: bool = False):
        """Enumerate running asyncio tasks.

        Default is cheap — ``stack_depth`` is ``null`` in every entry so
        a storm snapshot does not walk 50+ coroutine frames synchronously.
        Pass ``?stack=true`` to compute ``stack_depth`` for each task
        (useful for single-shot human debugging).
        """
        ws_registry = getattr(proxy, "ws_sessions", None)
        return JSONResponse(
            status_code=200,
            content=_collect_tasks(ws_registry, with_stack_depth=stack),
        )

    @app.get("/debug/ws-sessions", dependencies=[Depends(_require_loopback)])
    async def debug_ws_sessions():
        ws_registry = getattr(proxy, "ws_sessions", None)
        snapshot = ws_registry.snapshot() if ws_registry is not None else []
        return JSONResponse(status_code=200, content=snapshot)

    @app.get("/debug/warmup", dependencies=[Depends(_require_loopback)])
    async def debug_warmup():
        warmup_registry = getattr(proxy, "warmup", None)
        payload = warmup_registry.to_dict() if warmup_registry is not None else {}
        payload["runtime"] = _runtime_payload()
        return JSONResponse(status_code=200, content=payload)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve the Headroom dashboard UI."""
        return get_dashboard_html()

    DASHBOARD_STATS_CACHE_TTL_SECONDS = 5.0
    _stats_snapshot_lock = asyncio.Lock()
    _stats_snapshot: dict[str, Any] = {"expires_at": 0.0, "value": None}

    async def _build_stats_payload() -> dict[str, Any]:
        """Build the full `/stats` response payload.

        This is the main stats endpoint - it aggregates data from all subsystems:
        - Request metrics (total, cached, failed, by model/provider)
        - Token usage and savings
        - Cost tracking
        - Canonical persisted display_session metrics for downstream dashboards
        - Compression (CCR) statistics
        - Telemetry/TOIN (data flywheel) statistics
        - Cache and rate limiter stats
        """
        m = proxy.metrics

        # Calculate average latency
        avg_latency_ms = round(m.latency_sum_ms / m.latency_count, 2) if m.latency_count > 0 else 0
        min_latency_ms = (
            round(m.latency_min_ms, 2)
            if m.latency_count > 0 and m.latency_min_ms != float("inf")
            else 0
        )
        max_latency_ms = round(m.latency_max_ms, 2) if m.latency_count > 0 else 0

        # Calculate Headroom overhead (optimization time only, excludes pass-through requests)
        avg_overhead_ms = (
            round(m.overhead_sum_ms / m.overhead_count, 2) if m.overhead_count > 0 else 0
        )
        min_overhead_ms = (
            round(m.overhead_min_ms, 2)
            if m.overhead_count > 0 and m.overhead_min_ms != float("inf")
            else 0
        )
        max_overhead_ms = round(m.overhead_max_ms, 2) if m.overhead_count > 0 else 0

        # Calculate TTFB (time to first byte)
        avg_ttfb_ms = round(m.ttfb_sum_ms / m.ttfb_count, 2) if m.ttfb_count > 0 else 0
        min_ttfb_ms = (
            round(m.ttfb_min_ms, 2) if m.ttfb_count > 0 and m.ttfb_min_ms != float("inf") else 0
        )
        max_ttfb_ms = round(m.ttfb_max_ms, 2) if m.ttfb_count > 0 else 0

        # Get compression store stats
        store = get_compression_store()
        compression_stats = store.get_stats()

        # Get telemetry/TOIN stats
        telemetry = get_telemetry_collector()
        telemetry_stats = telemetry.get_stats()

        # Get feedback loop stats
        feedback = get_compression_feedback()
        feedback_stats = feedback.get_stats()

        # Build prefix cache stats once (used in both prefix_cache and cost)
        prefix_cache_stats = _build_prefix_cache_stats(m, proxy.cost_tracker)

        # Fetch CLI filtering savings (rtk — tokens avoided before reaching context)
        cli_filtering_stats = _get_rtk_stats()
        cli_tokens_avoided = (
            cli_filtering_stats.get("tokens_saved", 0) if cli_filtering_stats else 0
        )

        # Calculate total tokens before compression
        total_tokens_before = m.tokens_input_total + m.tokens_saved_total

        # Build human-readable summary
        summary = _build_session_summary(
            proxy, m, prefix_cache_stats, cli_tokens_avoided, total_tokens_before
        )
        # DEBUG: log the summary payload for external upsert consumers
        try:
            logger.debug("/stats summary data: %r", summary)
        except Exception:
            logger.warning("Failed to log /stats summary payload")

        # Compression cache stats (token mode). Snapshot the cache list under
        # the dict lock so a concurrent eviction can't mutate the dict while
        # we iterate. Each per-session `get_stats()` is independently
        # thread-safe via the cache's own internal lock.
        compression_cache_stats: dict = {}
        if proxy.config.mode == PROXY_MODE_TOKEN and proxy._compression_caches:
            with proxy._compression_caches_lock:
                _caches_snapshot = list(proxy._compression_caches.values())
                _active_sessions = len(proxy._compression_caches)
            total_entries = 0
            total_hits = 0
            total_misses = 0
            total_tokens_saved = 0
            for cache in _caches_snapshot:
                s = cache.get_stats()
                total_entries += s.get("entries", 0)
                total_hits += s.get("hits", 0)
                total_misses += s.get("misses", 0)
                total_tokens_saved += s.get("total_tokens_saved", 0)
            compression_cache_stats = {
                "mode": PROXY_MODE_TOKEN,
                "active_sessions": _active_sessions,
                "total_entries": total_entries,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": round(total_hits / max(1, total_hits + total_misses) * 100, 1),
                "total_tokens_saved": total_tokens_saved,
            }
        else:
            compression_cache_stats = {"mode": proxy.config.mode}

        # Build unified savings summary (all layers)
        compression_tokens = m.tokens_saved_total
        cache_net_usd = prefix_cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
        total_tokens_all_layers = compression_tokens + cli_tokens_avoided
        persistent_savings = m.savings_tracker.stats_preview()
        display_session = persistent_savings.get("display_session", {})

        return {
            "summary": summary,
            "savings": {
                "total_tokens": total_tokens_all_layers,
                "by_layer": {
                    "cli_filtering": {
                        "tokens": cli_tokens_avoided,
                        "description": "Tokens avoided by CLI output filtering (rtk) before reaching context",
                    },
                    "compression": {
                        "tokens": compression_tokens,
                        "description": "Tokens removed by proxy compression (SmartCrusher, ContentRouter, etc.)",
                    },
                    "prefix_cache": {
                        "discount_usd": round(cache_net_usd, 4),
                        "description": (
                            "Cost discount from provider prefix caching. "
                            "Headroom's CacheAligner improves hit rates; "
                            "baseline caching is provider-native."
                        ),
                    },
                },
            },
            "requests": {
                "total": m.requests_total,
                "cached": m.requests_cached,
                "rate_limited": m.requests_rate_limited,
                "failed": m.requests_failed,
                "by_provider": dict(m.requests_by_provider),
                "by_model": dict(m.requests_by_model),
                "by_stack": dict(m.requests_by_stack),
            },
            "tokens": {
                "input": m.tokens_input_total,
                "output": m.tokens_output_total,
                "saved": m.tokens_saved_total,
                "cli_tokens_avoided": cli_tokens_avoided,
                "total_before_compression": total_tokens_before,
                "savings_percent": round(
                    (m.tokens_saved_total / total_tokens_before * 100)
                    if total_tokens_before > 0
                    else 0,
                    2,
                ),
            },
            "latency": {
                "average_ms": avg_latency_ms,
                "min_ms": min_latency_ms,
                "max_ms": max_latency_ms,
                "total_requests": m.latency_count,
            },
            "overhead": {
                "average_ms": avg_overhead_ms,
                "min_ms": min_overhead_ms,
                "max_ms": max_overhead_ms,
            },
            "ttfb": {
                "average_ms": avg_ttfb_ms,
                "min_ms": min_ttfb_ms,
                "max_ms": max_ttfb_ms,
            },
            "pipeline_timing": {
                name: {
                    "average_ms": round(
                        m.transform_timing_sum[name] / m.transform_timing_count[name], 2
                    ),
                    "max_ms": round(m.transform_timing_max[name], 2),
                    "count": m.transform_timing_count[name],
                }
                for name in sorted(m.transform_timing_sum.keys())
            }
            if m.transform_timing_sum
            else {},
            "compressions_by_strategy": dict(m.compressions_by_strategy),
            "tokens_saved_by_strategy": dict(m.tokens_saved_by_strategy),
            "waste_signals": dict(m.waste_signals_total) if m.waste_signals_total else {},
            "savings_history": m.savings_history[-100:],  # Last 100 data points
            "display_session": display_session,
            "persistent_savings": persistent_savings,
            "prefix_cache": prefix_cache_stats,
            "cost": _merge_cost_stats(
                proxy.cost_tracker.stats() if proxy.cost_tracker else None,
                prefix_cache_stats,
                cli_tokens_avoided=cli_tokens_avoided,
            ),
            "compression": {
                "ccr_entries": compression_stats.get("entry_count", 0),
                "ccr_max_entries": compression_stats.get("max_entries", 0),
                "original_tokens_cached": compression_stats.get("total_original_tokens", 0),
                "compressed_tokens_cached": compression_stats.get("total_compressed_tokens", 0),
                "ccr_retrievals": compression_stats.get("total_retrievals", 0),
            },
            "compression_cache": compression_cache_stats,
            "anon_telemetry_shipping": is_telemetry_enabled(),
            "telemetry": {
                "enabled": telemetry_stats.get("enabled", False),
                "total_compressions": telemetry_stats.get("total_compressions", 0),
                "total_retrievals": telemetry_stats.get("total_retrievals", 0),
                "global_retrieval_rate": round(telemetry_stats.get("global_retrieval_rate", 0), 4),
                "tool_signatures_tracked": telemetry_stats.get("tool_signatures_tracked", 0),
                "avg_compression_ratio": round(telemetry_stats.get("avg_compression_ratio", 0), 4),
                "avg_token_reduction": round(telemetry_stats.get("avg_token_reduction", 0), 4),
            },
            "otel": get_otel_metrics_status(),
            "langfuse": get_langfuse_tracing_status(),
            "feedback_loop": {
                "tools_tracked": feedback_stats.get("tools_tracked", 0),
                "total_compressions": feedback_stats.get("total_compressions", 0),
                "total_retrievals": feedback_stats.get("total_retrievals", 0),
                "global_retrieval_rate": round(feedback_stats.get("global_retrieval_rate", 0), 4),
                "tools_with_high_retrieval": sum(
                    1
                    for p in feedback_stats.get("tool_patterns", {}).values()
                    if p.get("retrieval_rate", 0) > 0.3
                ),
            },
            "toin": get_toin().get_stats(),
            "cli_filtering": cli_filtering_stats,
            "cache": await proxy.cache.stats() if proxy.cache else None,
            "rate_limiter": await proxy.rate_limiter.stats() if proxy.rate_limiter else None,
            "recent_requests": proxy.logger.get_recent(10) if proxy.logger else [],
            "log_full_messages": proxy.config.log_full_messages if proxy else False,
            **get_quota_registry().get_all_stats(),
        }

    async def _get_cached_stats_payload() -> dict[str, Any]:
        """Return a short-TTL cached `/stats` snapshot for dashboard polling."""
        now = time.monotonic()
        cached_payload = cast(dict[str, Any] | None, _stats_snapshot.get("value"))
        if cached_payload is not None and now < float(_stats_snapshot["expires_at"]):
            return cached_payload

        async with _stats_snapshot_lock:
            now = time.monotonic()
            cached_payload = cast(dict[str, Any] | None, _stats_snapshot.get("value"))
            if cached_payload is not None and now < float(_stats_snapshot["expires_at"]):
                return cached_payload

            payload = await _build_stats_payload()
            _stats_snapshot["value"] = payload
            _stats_snapshot["expires_at"] = time.monotonic() + DASHBOARD_STATS_CACHE_TTL_SECONDS
            return payload

    @app.get("/stats")
    async def stats(cached: bool = False):
        """Get comprehensive proxy statistics.

        This is the main stats endpoint - it aggregates data from all subsystems:
        - Request metrics (total, cached, failed, by model/provider)
        - Token usage and savings
        - Cost tracking
        - Canonical persisted display_session metrics for downstream dashboards
        - Compression (CCR) statistics
        - Telemetry/TOIN (data flywheel) statistics
        - Cache and rate limiter stats

        Use ``?cached=1`` for the dashboard fast path. That returns a short-TTL
        snapshot to avoid rebuilding the full payload on every UI poll.
        """
        if cached:
            return await _get_cached_stats_payload()
        return await _build_stats_payload()

    @app.get("/stats-history")
    async def stats_history(
        format: Literal["json", "csv"] = "json",
        series: Literal["history", "hourly", "daily", "weekly", "monthly"] = "history",
        history_mode: Literal["compact", "full", "none"] = "compact",
    ):
        """Get durable proxy compression history plus display-session state."""
        if format == "csv":
            filename = f"headroom-stats-history-{series}.csv"
            return Response(
                content=proxy.metrics.savings_tracker.export_csv(series=series),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        return proxy.metrics.savings_tracker.history_response(history_mode=history_mode)

    @app.get("/transformations/feed")
    async def transformations_feed(limit: int = 20):
        """Get recent message transformations for the live feed.

        Returns empty list if log_full_messages is disabled (messages are not stored).
        """
        if limit > 100:
            limit = 100

        transformations = []
        log_full_messages = proxy.config.log_full_messages if proxy else False

        if proxy and proxy.logger:
            logs = proxy.logger.get_recent_with_messages(limit)
            for log in logs:
                transformations.append(
                    {
                        "request_id": log.get("request_id"),
                        "timestamp": log.get("timestamp"),
                        "provider": log.get("provider"),
                        "model": log.get("model"),
                        "input_tokens_original": log.get("input_tokens_original"),
                        "input_tokens_optimized": log.get("input_tokens_optimized"),
                        "tokens_saved": log.get("tokens_saved"),
                        "savings_percent": log.get("savings_percent"),
                        "transforms_applied": log.get("transforms_applied", []),
                        "request_messages": log.get("request_messages"),
                        "response_content": log.get("response_content"),
                        "turn_id": log.get("turn_id"),
                    }
                )

        return {"transformations": transformations, "log_full_messages": log_full_messages}

    @app.get("/subscription-window")
    async def subscription_window():
        """Current Anthropic subscription window utilisation and Headroom contribution.

        Issue #281: the Anthropic OAuth usage API is polled every 5 minutes
        (aggressive polling risks 429s / OAuth-token flagging), so the cached
        ``utilization_pct`` lags reality by up to one poll interval. When the
        user's 5-hour window rolls over between two polls the dashboard would
        otherwise render the OLD window's percentage. We:

        1. Optionally trigger a 60s-floored singleton poll on dashboard load
           (bounded across users, well within Anthropic tolerance).
        2. Render via :meth:`SubscriptionTracker.render_state`, which
           synthesizes post-reset windows from local transcript-derived
           token counts when ``now >= window.resets_at``.
        """
        tracker = get_subscription_tracker()
        if tracker is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Subscription tracking is not enabled"},
            )
        await tracker.maybe_poll_on_demand()
        return JSONResponse(content=tracker.render_state())

    @app.get("/quota")
    async def quota():
        """Unified quota/rate-limit stats for all registered providers (Anthropic, Codex, Copilot)."""
        return JSONResponse(content=get_quota_registry().get_all_stats())

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            await proxy.metrics.export(),
            media_type="text/plain; version=0.0.4",
        )

    # Debug endpoints
    @app.get("/debug/memory")
    async def debug_memory():
        """Get detailed memory usage statistics.

        Returns memory usage for all tracked components including:
        - Process-level memory (RSS, VMS, percent)
        - Per-component memory usage and budgets
        - Cache hit/miss statistics
        - Total tracked vs target budget

        This endpoint is useful for debugging memory issues and
        monitoring memory budgets.
        """
        from ..memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()

        # Register components if not already registered
        _register_memory_components(proxy, tracker)

        report = tracker.get_report()
        return report.to_dict()

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the response cache."""
        if proxy.cache:
            await proxy.cache.clear()
            return {"status": "cleared"}
        return {"status": "cache disabled"}

    # CCR (Compress-Cache-Retrieve) endpoints
    @app.post("/v1/retrieve")
    async def ccr_retrieve(request: Request):
        """Retrieve original content from CCR compression cache.

        This is the "Retrieve" part of CCR (Compress-Cache-Retrieve).
        When SmartCrusher compresses tool outputs, the original data is cached.
        LLMs can call this endpoint to get more data if needed.

        Request body:
            hash (str): Hash key from compression marker (required)
            query (str): Optional search query to filter results

        Response:
            Full retrieval: {"hash": "...", "original_content": "...", ...}
            Search: {"hash": "...", "query": "...", "results": [...], "count": N}
        """
        data = await request.json()
        hash_key = data.get("hash")
        query = data.get("query")

        if not hash_key:
            raise HTTPException(status_code=400, detail="hash required")

        store = get_compression_store()

        if query:
            # Search within cached content
            results = store.search(hash_key, query)
            return {
                "hash": hash_key,
                "query": query,
                "results": results,
                "count": len(results),
            }
        else:
            # Return full original content
            entry = store.retrieve(hash_key)
            if entry:
                return {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_tokens": entry.original_tokens,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                    "tool_name": entry.tool_name,
                    "retrieval_count": entry.retrieval_count,
                }
            raise HTTPException(
                status_code=404, detail="Entry not found or expired (TTL: 5 minutes)"
            )

    @app.get("/v1/retrieve/stats")
    async def ccr_stats():
        """Get CCR compression store statistics."""
        store = get_compression_store()
        stats = store.get_stats()
        events = store.get_retrieval_events(limit=20)
        return {
            "store": stats,
            "recent_retrievals": [
                {
                    "hash": e.hash,
                    "query": e.query,
                    "items_retrieved": e.items_retrieved,
                    "total_items": e.total_items,
                    "tool_name": e.tool_name,
                    "retrieval_type": e.retrieval_type,
                }
                for e in events
            ],
        }

    @app.get("/v1/feedback")
    async def ccr_feedback():
        """Get CCR feedback loop statistics and learned patterns.

        This endpoint exposes the feedback loop's learned patterns for monitoring
        and debugging. It shows:
        - Per-tool retrieval rates (high = compress less aggressively)
        - Common search queries per tool
        - Queried fields (suggest what to preserve)

        Use this to understand how well compression is working and whether
        the feedback loop is adjusting appropriately.
        """
        feedback = get_compression_feedback()
        stats = feedback.get_stats()
        return {
            "feedback": stats,
            "hints_example": {
                tool_name: {
                    "hints": {
                        "max_items": hints.max_items
                        if (hints := feedback.get_compression_hints(tool_name))
                        else 15,
                        "suggested_items": hints.suggested_items if hints else None,
                        "skip_compression": hints.skip_compression if hints else False,
                        "preserve_fields": hints.preserve_fields if hints else [],
                        "reason": hints.reason if hints else "",
                    }
                }
                for tool_name in list(stats.get("tool_patterns", {}).keys())[:5]
            },
        }

    @app.get("/v1/feedback/{tool_name}")
    async def ccr_feedback_for_tool(tool_name: str):
        """Get compression hints for a specific tool.

        Returns feedback-based hints that would be used for compressing
        this tool's output.
        """
        feedback = get_compression_feedback()
        hints = feedback.get_compression_hints(tool_name)
        patterns = feedback.get_all_patterns().get(tool_name)

        return {
            "tool_name": tool_name,
            "hints": {
                "max_items": hints.max_items,
                "min_items": hints.min_items,
                "suggested_items": hints.suggested_items,
                "aggressiveness": hints.aggressiveness,
                "skip_compression": hints.skip_compression,
                "preserve_fields": hints.preserve_fields,
                "reason": hints.reason,
            },
            "pattern": {
                "total_compressions": patterns.total_compressions if patterns else 0,
                "total_retrievals": patterns.total_retrievals if patterns else 0,
                "retrieval_rate": patterns.retrieval_rate if patterns else 0.0,
                "full_retrieval_rate": patterns.full_retrieval_rate if patterns else 0.0,
                "search_rate": patterns.search_rate if patterns else 0.0,
                "common_queries": list(patterns.common_queries.keys())[:10] if patterns else [],
                "queried_fields": list(patterns.queried_fields.keys())[:10] if patterns else [],
            }
            if patterns
            else None,
        }

    # Telemetry endpoints (Data Flywheel)
    @app.get("/v1/telemetry")
    async def telemetry_stats():
        """Get telemetry statistics for the data flywheel.

        This endpoint exposes privacy-preserving telemetry data that powers
        the data flywheel - learning optimal compression strategies across
        tool types based on usage patterns.

        What's collected (anonymized):
        - Tool output structure patterns (field types, not values)
        - Compression decisions and ratios
        - Retrieval patterns (rate, type, not content)
        - Strategy effectiveness

        What's NOT collected:
        - Actual data values
        - User identifiers
        - Queries or search terms
        - File paths or tool names (hashed by default)
        """
        telemetry = get_telemetry_collector()
        return telemetry.get_stats()

    @app.get("/v1/telemetry/export")
    async def telemetry_export():
        """Export full telemetry data for aggregation.

        This endpoint exports all telemetry data in a format suitable for
        cross-user aggregation. The data is privacy-preserving - no actual
        values are included, only structural patterns and statistics.

        Use this for:
        - Building a central learning service
        - Sharing learned patterns across instances
        - Analysis and debugging
        """
        telemetry = get_telemetry_collector()
        return telemetry.export_stats()

    @app.post("/v1/telemetry/import")
    async def telemetry_import(request: Request):
        """Import telemetry data from another source.

        This allows merging telemetry from multiple sources for cross-user
        learning. The imported data is merged with existing statistics.

        Request body: Telemetry export data from /v1/telemetry/export
        """
        telemetry = get_telemetry_collector()
        data = await request.json()
        telemetry.import_stats(data)
        return {"status": "imported", "current_stats": telemetry.get_stats()}

    @app.get("/v1/telemetry/tools")
    async def telemetry_tools():
        """Get telemetry statistics for all tracked tool signatures.

        Returns statistics per tool signature (anonymized), including:
        - Compression ratios and strategy usage
        - Retrieval rates (high = compression too aggressive)
        - Learned recommendations
        """
        telemetry = get_telemetry_collector()
        all_stats = telemetry.get_all_tool_stats()
        return {
            "tool_count": len(all_stats),
            "tools": {sig_hash: stats.to_dict() for sig_hash, stats in all_stats.items()},
        }

    @app.get("/v1/telemetry/tools/{signature_hash}")
    async def telemetry_tool_detail(signature_hash: str):
        """Get detailed telemetry for a specific tool signature.

        Includes learned recommendations if enough data has been collected.
        """
        telemetry = get_telemetry_collector()
        stats = telemetry.get_tool_stats(signature_hash)
        recommendations = telemetry.get_recommendations(signature_hash)

        if stats is None:
            raise HTTPException(
                status_code=404, detail=f"No telemetry found for signature: {signature_hash}"
            )

        return {
            "signature_hash": signature_hash,
            "stats": stats.to_dict(),
            "recommendations": recommendations,
        }

    # TOIN (Tool Output Intelligence Network) endpoints
    @app.get("/v1/toin/stats")
    async def toin_stats():
        """Get overall TOIN statistics.

        Returns aggregated statistics from the Tool Output Intelligence Network,
        which learns optimal compression strategies across all tool types.

        Response includes:
        - enabled: Whether TOIN is enabled
        - patterns_tracked: Number of unique tool patterns being tracked
        - total_compressions: Total compression events recorded
        - total_retrievals: Total retrieval events recorded
        - global_retrieval_rate: Overall retrieval rate (high = compression too aggressive)
        - patterns_with_recommendations: Patterns with enough data for recommendations
        """
        toin = get_toin()
        return toin.get_stats()

    @app.get("/v1/toin/patterns")
    async def toin_patterns(limit: int = 20):
        """List TOIN patterns with most samples.

        Returns patterns sorted by sample_size descending. Use this to see
        which tool types have the most data and their learned behaviors.

        Query params:
            limit: Maximum number of patterns to return (default 20)

        Response includes for each pattern:
        - hash: Truncated tool signature hash (12 chars)
        - compressions: Total compression events
        - retrievals: Total retrieval events
        - retrieval_rate: Percentage of compressions that triggered retrieval
        - confidence: Confidence level in recommendations (0.0-1.0)
        - skip_recommended: Whether TOIN recommends skipping compression
        - optimal_max_items: Learned optimal max_items setting
        """
        toin = get_toin()
        exported = toin.export_patterns()
        patterns_data = exported.get("patterns", {})

        # Convert to list and sort by sample_size
        patterns_list = []
        for sig_hash, pattern_dict in patterns_data.items():
            sample_size = pattern_dict.get("sample_size", 0)
            total_compressions = pattern_dict.get("total_compressions", 0)
            total_retrievals = pattern_dict.get("total_retrievals", 0)
            retrieval_rate = (
                total_retrievals / total_compressions if total_compressions > 0 else 0.0
            )

            patterns_list.append(
                {
                    "hash": sig_hash[:12],
                    "compressions": total_compressions,
                    "retrievals": total_retrievals,
                    "retrieval_rate": f"{retrieval_rate:.1%}",
                    "confidence": round(pattern_dict.get("confidence", 0.0), 3),
                    "skip_recommended": pattern_dict.get("skip_compression_recommended", False),
                    "optimal_max_items": pattern_dict.get("optimal_max_items", 20),
                    "sample_size": sample_size,
                }
            )

        # Sort by sample_size descending
        patterns_list.sort(key=lambda p: p["sample_size"], reverse=True)

        # Remove sample_size from output (used only for sorting)
        for p in patterns_list:
            del p["sample_size"]

        return patterns_list[:limit]

    @app.get("/v1/toin/pattern/{hash_prefix}")
    async def toin_pattern_detail(hash_prefix: str):
        """Get detailed TOIN pattern info by hash prefix.

        Searches for a pattern where the tool signature hash starts with
        the provided prefix. Returns full pattern details if found.

        Path params:
            hash_prefix: Beginning of the tool signature hash (min 4 chars recommended)

        Response: Full pattern.to_dict() with all learned statistics and recommendations.
        """
        toin = get_toin()
        exported = toin.export_patterns()
        patterns_data = exported.get("patterns", {})

        # Search for pattern with matching hash prefix
        for sig_hash, pattern_dict in patterns_data.items():
            if sig_hash.startswith(hash_prefix):
                return pattern_dict

        raise HTTPException(
            status_code=404, detail=f"No TOIN pattern found with hash starting with: {hash_prefix}"
        )

    @app.get("/v1/retrieve/{hash_key}")
    async def ccr_retrieve_get(hash_key: str, query: str | None = None):
        """GET version of CCR retrieve for easier testing."""
        store = get_compression_store()

        if query:
            results = store.search(hash_key, query)
            return {
                "hash": hash_key,
                "query": query,
                "results": results,
                "count": len(results),
            }
        else:
            entry = store.retrieve(hash_key)
            if entry:
                return {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_tokens": entry.original_tokens,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                    "tool_name": entry.tool_name,
                    "retrieval_count": entry.retrieval_count,
                }
            raise HTTPException(status_code=404, detail="Entry not found or expired")

    # CCR Tool Call Handler - for agent frameworks to call when LLM uses headroom_retrieve
    @app.post("/v1/retrieve/tool_call")
    async def ccr_handle_tool_call(request: Request):
        """Handle a CCR tool call from an LLM response.

        This endpoint accepts tool call formats from various providers and returns
        a properly formatted tool result. Agent frameworks can use this to handle
        CCR tool calls without implementing the retrieval logic themselves.

        Request body (Anthropic format):
            {
                "tool_call": {
                    "id": "toolu_123",
                    "name": "headroom_retrieve",
                    "input": {"hash": "abc123", "query": "optional search"}
                },
                "provider": "anthropic"
            }

        Request body (OpenAI format):
            {
                "tool_call": {
                    "id": "call_123",
                    "function": {
                        "name": "headroom_retrieve",
                        "arguments": "{\"hash\": \"abc123\"}"
                    }
                },
                "provider": "openai"
            }

        Response:
            {
                "tool_result": {...},  # Formatted for the provider
                "success": true,
                "data": {...}  # Raw retrieval data
            }
        """
        data = await request.json()
        tool_call = data.get("tool_call", {})
        provider = data.get("provider", "anthropic")

        # Parse the tool call
        hash_key, query = parse_tool_call(tool_call, provider)

        if hash_key is None:
            raise HTTPException(
                status_code=400, detail=f"Invalid tool call or not a {CCR_TOOL_NAME} call"
            )

        # Perform retrieval
        store = get_compression_store()

        if query:
            results = store.search(hash_key, query)
            retrieval_data = {
                "hash": hash_key,
                "query": query,
                "results": results,
                "count": len(results),
            }
        else:
            entry = store.retrieve(hash_key)
            if entry:
                retrieval_data = {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                }
            else:
                retrieval_data = {
                    "error": "Entry not found or expired (TTL: 5 minutes)",
                    "hash": hash_key,
                }

        # Format tool result for provider
        tool_call_id = tool_call.get("id", "")
        result_content = json.dumps(retrieval_data, indent=2)

        if provider == "anthropic":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result_content,
            }
        elif provider == "openai":
            tool_result = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content,
            }
        else:
            tool_result = {
                "tool_call_id": tool_call_id,
                "content": result_content,
            }

        return {
            "tool_result": tool_result,
            "success": "error" not in retrieval_data,
            "data": retrieval_data,
        }

    # Compression-only endpoint (for TypeScript SDK and other HTTP clients)
    @app.post("/v1/compress")
    async def compress_messages(request: Request):
        return await proxy.handle_compress(request)

    register_provider_routes(app, proxy)

    return app


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_ready(item) for item in value]
    return value


def _proxy_config_payload(config: ProxyConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(config):
        value = _json_ready(getattr(config, field.name))
        try:
            json.dumps(value)
        except TypeError:
            continue
        payload[field.name] = value
    return payload


def _proxy_config_from_env() -> ProxyConfig:
    raw_config = os.environ.get(_MULTI_WORKER_CONFIG_ENV)
    if raw_config:
        try:
            return ProxyConfig(**json.loads(raw_config))
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning(
                "Invalid %s; falling back to HEADROOM_* env vars", _MULTI_WORKER_CONFIG_ENV
            )

    return ProxyConfig(
        host=_get_env_str("HEADROOM_HOST", "127.0.0.1"),
        port=_get_env_int("HEADROOM_PORT", 8787),
        openai_api_url=os.environ.get("OPENAI_TARGET_API_URL"),
        anthropic_api_url=os.environ.get("ANTHROPIC_TARGET_API_URL"),
        backend=_get_env_str("HEADROOM_BACKEND", "anthropic"),
        bedrock_region=_get_env_str("HEADROOM_BEDROCK_REGION", "us-west-2"),
        bedrock_profile=os.environ.get("AWS_PROFILE"),
        anyllm_provider=_get_env_str("HEADROOM_ANYLLM_PROVIDER", "openai"),
        max_connections=_get_env_int("HEADROOM_MAX_CONNECTIONS", 500),
        max_keepalive_connections=_get_env_int("HEADROOM_MAX_KEEPALIVE", 100),
        http2=_get_env_bool("HEADROOM_HTTP2", True),
        mode=normalize_proxy_mode(_get_env_str("HEADROOM_MODE", PROXY_MODE_TOKEN)),
    )


def create_app_from_env() -> FastAPI:
    return create_app(_proxy_config_from_env())


def _get_code_aware_banner_status(config: ProxyConfig) -> str:
    """Get code-aware compression status line for banner."""
    if config.code_aware_enabled:
        if is_tree_sitter_available():
            return "ENABLED  (AST-based)"
        else:
            return "NOT INSTALLED (pip install headroom-ai[code])"
    else:
        if is_tree_sitter_available():
            return "DISABLED (remove --no-code-aware to enable)"
        return "DISABLED"


def run_server(
    config: ProxyConfig | None = None,
    workers: int = 1,
    limit_concurrency: int = 1000,
):
    """Run the proxy server.

    Args:
        config: Proxy configuration
        workers: Number of worker processes (use N for multi-core scaling)
        limit_concurrency: Max concurrent connections before 503 response
    """
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI required. Install: pip install fastapi uvicorn httpx")
        sys.exit(1)

    config = config or ProxyConfig()
    code_aware_status = _get_code_aware_banner_status(config)

    # Format connection pool info
    pool_info = f"max={config.max_connections}, keepalive={config.max_keepalive_connections}"
    http2_status = "ENABLED" if config.http2 else "DISABLED"

    backend_status = format_backend_status(
        backend=config.backend,
        anyllm_provider=config.anyllm_provider,
        bedrock_region=config.bedrock_region,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      HEADROOM PROXY SERVER                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Version: 1.0.0                                                      ║
║  Listening: http://{config.host}:{config.port:<5}                                      ║
║  Workers: {workers:<3}  Concurrency Limit: {limit_concurrency:<5}                          ║
║  Backend: {backend_status:<59}║
╠══════════════════════════════════════════════════════════════════════╣
║  FEATURES:                                                           ║
║    Optimization:    {"ENABLED " if config.optimize else "DISABLED"}                                       ║
║    Caching:         {"ENABLED " if config.cache_enabled else "DISABLED"}   (TTL: {config.cache_ttl_seconds}s)                          ║
║    Rate Limiting:   {"ENABLED " if config.rate_limit_enabled else "DISABLED"}   ({config.rate_limit_requests_per_minute} req/min, {config.rate_limit_tokens_per_minute:,} tok/min)       ║
║    Retry:           {"ENABLED " if config.retry_enabled else "DISABLED"}   (max {config.retry_max_attempts} attempts)                       ║
║    Cost Tracking:   {"ENABLED " if config.cost_tracking_enabled else "DISABLED"}   (budget: {"$" + str(config.budget_limit_usd) + "/" + config.budget_period if config.budget_limit_usd else "unlimited"})          ║
║    Code-Aware:      {code_aware_status:<52}║
║    HTTP/2:          {http2_status:<52}║
║    Conn Pool:       {pool_info:<52}║
╠══════════════════════════════════════════════════════════════════════╣
║  USAGE:                                                              ║
║    Claude Code:   ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude     ║
║    Cursor:        Set base URL in settings                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  ENDPOINTS:                                                          ║
║    /livez                   Process liveness                         ║
║    /readyz                  Traffic readiness                        ║
║    /health                  Aggregate health                         ║
║    /stats                   Detailed statistics                      ║
║    /metrics                 Prometheus metrics                       ║
║    /cache/clear             Clear response cache                     ║
║    /v1/retrieve             CCR: Retrieve compressed content         ║
║    /v1/retrieve/stats       CCR: Compression store stats             ║
║    /v1/retrieve/tool_call   CCR: Handle LLM tool calls               ║
║    /v1/feedback             CCR: Feedback loop stats & patterns      ║
║    /v1/feedback/{{tool}}    CCR: Compression hints for a tool        ║
║    /v1/telemetry            Data flywheel: Telemetry stats           ║
║    /v1/telemetry/export     Data flywheel: Export for aggregation    ║
║    /v1/telemetry/tools      Data flywheel: Per-tool stats            ║
║    /v1/toin/stats           TOIN: Overall intelligence stats         ║
║    /v1/toin/patterns        TOIN: List learned patterns              ║
║    /v1/toin/pattern/{{hash}} TOIN: Pattern details by hash            ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    app_target: Any
    uvicorn_kwargs: dict[str, Any] = {}
    if workers > 1:
        # CCR / compression-cache / prefix-tracker / TOIN state are all
        # per-process. Round-robin across workers fragments these caches
        # and produces silent retrieval failures for `Retrieve original:
        # hash=X` markers and avoidable cache busts on the upstream
        # provider. See the "Multi-worker deployment — CCR fragmentation"
        # section in RUST_DEV.md for the full failure modes and the
        # sticky-session workaround.
        logger.warning(
            "Headroom is running with workers=%d. The in-memory CCR store, "
            "compression cache, prefix tracker, and TOIN state are all "
            "per-process; multi-worker deployments produce silent retrieval "
            "failures and avoidable cache busts when sessions land on different "
            "workers. Run --workers 1 (or place a sticky-session load balancer "
            "in front of multiple --workers 1 processes). See RUST_DEV.md → "
            "'Multi-worker deployment — CCR fragmentation'.",
            workers,
        )
        os.environ[_MULTI_WORKER_CONFIG_ENV] = json.dumps(_proxy_config_payload(config))
        app_target = "headroom.proxy.server:create_app_from_env"
        uvicorn_kwargs["factory"] = True
    else:
        app_target = create_app(config)

    uvicorn.run(
        app_target,
        host=config.host,
        port=config.port,
        log_level="warning",
        workers=workers if workers > 1 else None,  # None = single process (default)
        limit_concurrency=limit_concurrency,
        # Defense-in-depth: the loopback guard for /debug/* endpoints trusts
        # request.client.host. uvicorn's ProxyHeadersMiddleware rewrites that
        # from X-Forwarded-For when FORWARDED_ALLOW_IPS is broader than the
        # default. Disabling proxy_headers here guarantees the guard sees the
        # real peer address regardless of env.
        proxy_headers=False,
        **uvicorn_kwargs,
    )


def _get_env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_str(name: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(name, default)


def _parse_tool_profiles(cli_profiles: list[str]) -> dict[str, Any]:
    """Parse tool profiles from CLI args and HEADROOM_TOOL_PROFILES env var.

    Format: ToolName:level (e.g., Grep:conservative, Bash:moderate)
    Env var format: comma-separated (e.g., "Grep:conservative,Bash:moderate")

    Returns:
        Dict mapping tool names to CompressionProfile instances.
    """
    from headroom.config import PROFILE_PRESETS, CompressionProfile

    profiles: dict[str, CompressionProfile] = {}
    raw_entries: list[str] = list(cli_profiles)

    # Also check env var
    env_val = os.environ.get("HEADROOM_TOOL_PROFILES", "")
    if env_val:
        raw_entries.extend(e.strip() for e in env_val.split(",") if e.strip())

    for entry in raw_entries:
        if ":" not in entry:
            logger.warning("Invalid tool profile format (expected ToolName:level): %s", entry)
            continue
        tool_name, level = entry.split(":", 1)
        tool_name = tool_name.strip()
        level = level.strip().lower()

        if level in PROFILE_PRESETS:
            profiles[tool_name] = PROFILE_PRESETS[level]
        else:
            logger.warning(
                "Unknown profile level '%s' for tool '%s'. Use: conservative, moderate, aggressive",
                level,
                tool_name,
            )

    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headroom Proxy Server")

    # Server
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--openai-api-url", help=f"Custom OpenAI API URL (default: {DEFAULT_OPENAI_API_URL})"
    )
    parser.add_argument(
        "--anthropic-api-url",
        help=f"Custom Anthropic API URL (default: {DEFAULT_ANTHROPIC_API_URL})",
    )

    # Backend (anthropic direct, bedrock, openrouter, anyllm, or litellm-<provider>)
    parser.add_argument(
        "--backend",
        default="anthropic",
        help=(
            "Backend: 'anthropic' (direct), 'bedrock' (AWS), 'openrouter', "
            "'anyllm' (any-llm), or 'litellm-<provider>' (e.g., litellm-hosted_vllm, litellm-vertex)"
        ),
    )
    parser.add_argument(
        "--bedrock-region",
        default="us-west-2",
        help="AWS region for Bedrock backend (default: us-west-2)",
    )
    parser.add_argument(
        "--bedrock-profile",
        help="AWS profile for Bedrock backend (default: use default credentials)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--anyllm-provider",
        default="openai",
        help="any-llm provider: openai, anthropic, mistral, groq, ollama, bedrock, etc. (default: openai)",
    )

    # Connection pool (scalability)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=500,
        help="Max connections to upstream APIs (default: 500)",
    )
    parser.add_argument(
        "--max-keepalive", type=int, default=100, help="Max keepalive connections (default: 100)"
    )
    parser.add_argument(
        "--no-http2",
        action="store_true",
        help="Disable HTTP/2 (enabled by default for better throughput)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, use N for multi-core)",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=1000,
        help="Max concurrent connections before 503 (default: 1000)",
    )

    # Optimization
    parser.add_argument("--no-optimize", action="store_true", help="Disable optimization")
    parser.add_argument("--min-tokens", type=int, default=500, help="Min tokens to crush")
    parser.add_argument("--max-items", type=int, default=50, help="Max items after crush")
    parser.add_argument(
        "--tool-profile",
        action="append",
        default=[],
        help="Per-tool compression profile: ToolName:level (e.g., Grep:conservative, Bash:moderate, WebFetch:aggressive). "
        "Can be specified multiple times. Also settable via HEADROOM_TOOL_PROFILES env var.",
    )

    # Caching
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL seconds")

    # Rate limiting
    parser.add_argument("--no-rate-limit", action="store_true", help="Disable rate limiting")
    parser.add_argument("--rpm", type=int, default=60, help="Requests per minute")
    parser.add_argument("--tpm", type=int, default=100000, help="Tokens per minute")

    # Cost
    parser.add_argument("--budget", type=float, help="Budget limit in USD")
    parser.add_argument("--budget-period", choices=["hourly", "daily", "monthly"], default="daily")

    # Logging
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--log-messages", action="store_true", help="Log full messages")

    # Smart routing (content-aware compression)
    parser.add_argument(
        "--no-smart-routing",
        action="store_true",
        help="Disable smart routing (use legacy sequential pipeline)",
    )

    # Code-aware compression
    parser.add_argument(
        "--code-aware",
        action="store_true",
        help="Enable AST-based code compression (requires: pip install headroom-ai[code])",
    )
    parser.add_argument(
        "--no-code-aware",
        action="store_true",
        help="Disable code-aware compression",
    )

    args = parser.parse_args()

    # Environment variable defaults (HEADROOM_* prefix)
    # CLI args override env vars, env vars override ProxyConfig defaults
    env_smart_routing = _get_env_bool("HEADROOM_SMART_ROUTING", True)
    env_code_aware = _get_env_bool("HEADROOM_CODE_AWARE_ENABLED", True)
    env_optimize = _get_env_bool("HEADROOM_OPTIMIZE", True)
    env_cache = _get_env_bool("HEADROOM_CACHE_ENABLED", True)
    env_rate_limit = _get_env_bool("HEADROOM_RATE_LIMIT_ENABLED", True)

    # Determine settings: CLI flags override env vars
    # --no-X explicitly disables, --X explicitly enables, neither uses env var
    smart_routing = env_smart_routing if not args.no_smart_routing else False
    code_aware_enabled = (
        env_code_aware
        if not (args.code_aware or args.no_code_aware)
        else (args.code_aware or not args.no_code_aware)
    )
    optimize = env_optimize if not args.no_optimize else False
    cache_enabled = env_cache if not args.no_cache else False
    rate_limit_enabled = env_rate_limit if not args.no_rate_limit else False

    # Set OpenRouter API key from CLI if provided
    if hasattr(args, "openrouter_api_key") and args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key

    # Parse per-tool compression profiles from CLI and env var
    tool_profiles = _parse_tool_profiles(args.tool_profile)

    config = ProxyConfig(
        host=_get_env_str("HEADROOM_HOST", args.host),
        port=_get_env_int("HEADROOM_PORT", args.port),
        openai_api_url=_get_env_str("OPENAI_TARGET_API_URL", args.openai_api_url),
        anthropic_api_url=_get_env_str("ANTHROPIC_TARGET_API_URL", args.anthropic_api_url),
        # Backend settings
        backend=_get_env_str("HEADROOM_BACKEND", args.backend),  # type: ignore[arg-type]
        bedrock_region=_get_env_str("HEADROOM_BEDROCK_REGION", args.bedrock_region),
        bedrock_profile=args.bedrock_profile or os.environ.get("AWS_PROFILE"),
        anyllm_provider=_get_env_str("HEADROOM_ANYLLM_PROVIDER", args.anyllm_provider),
        optimize=optimize,
        min_tokens_to_crush=_get_env_int("HEADROOM_MIN_TOKENS", args.min_tokens),
        max_items_after_crush=_get_env_int("HEADROOM_MAX_ITEMS", args.max_items),
        cache_enabled=cache_enabled,
        cache_ttl_seconds=_get_env_int("HEADROOM_CACHE_TTL", args.cache_ttl),
        rate_limit_enabled=rate_limit_enabled,
        rate_limit_requests_per_minute=_get_env_int("HEADROOM_RPM", args.rpm),
        rate_limit_tokens_per_minute=_get_env_int("HEADROOM_TPM", args.tpm),
        budget_limit_usd=args.budget,
        budget_period=args.budget_period,
        log_file=_get_env_str("HEADROOM_LOG_FILE", args.log_file)
        if args.log_file
        else os.environ.get("HEADROOM_LOG_FILE"),
        log_full_messages=args.log_messages or _get_env_bool("HEADROOM_LOG_MESSAGES", False),
        smart_routing=smart_routing,
        code_aware_enabled=code_aware_enabled,
        # Connection pool settings
        max_connections=_get_env_int("HEADROOM_MAX_CONNECTIONS", args.max_connections),
        max_keepalive_connections=_get_env_int("HEADROOM_MAX_KEEPALIVE", args.max_keepalive),
        http2=not args.no_http2 and _get_env_bool("HEADROOM_HTTP2", True),
        tool_profiles=tool_profiles if tool_profiles else None,
        mode=normalize_proxy_mode(_get_env_str("HEADROOM_MODE", PROXY_MODE_TOKEN)),
    )

    # Get worker and concurrency settings
    workers = _get_env_int("HEADROOM_WORKERS", args.workers)
    limit_concurrency = _get_env_int("HEADROOM_LIMIT_CONCURRENCY", args.limit_concurrency)

    run_server(config, workers=workers, limit_concurrency=limit_concurrency)
