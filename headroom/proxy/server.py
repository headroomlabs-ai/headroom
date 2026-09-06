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
import hmac
import ipaddress
import json
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from ..backends.base import Backend
    from ..cache.compression_cache import CompressionCache
    from ..memory.tracker import MemoryTracker
    from .outcome import RequestOutcome


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
from headroom.agent_savings import DEFAULT_PROFILE, proxy_pipeline_kwargs
from headroom.cache.compression_feedback import get_compression_feedback
from headroom.cache.compression_store import format_retrieval_miss_detail, get_compression_store
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
    DEFAULT_EXCLUDE_TOOLS,
    CacheAlignerConfig,
    ReadLifecycleConfig,
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
from headroom.offline import apply_offline_env, is_offline
from headroom.pipeline import PipelineExtensionManager, PipelineStage
from headroom.providers.proxy_routes import register_provider_routes
from headroom.providers.registry import (
    DEFAULT_ANTHROPIC_API_URL,
    DEFAULT_CLOUDCODE_API_URL,
    DEFAULT_GEMINI_API_URL,
    DEFAULT_OPENAI_API_URL,
    DEFAULT_VERTEX_API_URL,
    build_proxy_provider_runtime,
    create_proxy_backend,
    format_backend_status,
    resolve_api_targets,
)
from headroom.proxy import runtime_env
from headroom.proxy.audit import is_auditable_path, record_admin_action
from headroom.proxy.auth_mode import should_stamp_codex_client
from headroom.proxy.background_compression import BackgroundCompressor
from headroom.proxy.budget_basis_policy import resolve_estimated_basis_policy
from headroom.proxy.buffered_ccr_response import DEFAULT_BUFFERED_CCR_GRACE_SECONDS

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
    COMPRESSION_CACHE_MAX_ENTRIES,
    COMPRESSION_CACHE_TTL_SECONDS,
    COMPRESSION_TIMEOUT_SECONDS,  # noqa: F401
    EAGER_PRELOAD_TIMEOUT_SECONDS,
    MAX_COMPRESSION_CACHE_SESSIONS,  # noqa: F401
    MAX_MESSAGE_ARRAY_LENGTH,  # noqa: F401
    MAX_REQUEST_BODY_SIZE,  # noqa: F401
    MAX_SSE_BUFFER_SIZE,  # noqa: F401
    RETRYABLE_OVERLOAD_STATUSES,
    _get_image_compressor,  # noqa: F401
    _read_request_json,  # noqa: F401
    _setup_file_logging,  # noqa: F401
    is_anthropic_auth,  # noqa: F401
    jitter_delay_ms,
    resolve_display_provider,
    retry_after_ms,
)
from headroom.proxy.loop_callback_failure_policy import is_known_websocket_callback_failure
from headroom.proxy.loopback_guard import is_loopback_host
from headroom.proxy.malloc_trim import trim_periodically
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

# Data models (extracted to headroom/proxy/models.py for maintainability)
from headroom.proxy.model_router import ModelRouter, ModelRouterConfig
from headroom.proxy.models import CacheEntry, ProxyConfig, RateLimitState, RequestLog  # noqa: F401
from headroom.proxy.modes import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    is_token_mode,
    normalize_proxy_mode,
)
from headroom.proxy.probe_recorder import probe_recorder_from_env
from headroom.proxy.project_context import (
    classify_project,
    set_current_project,
    strip_project_path_prefix,
)
from headroom.proxy.prometheus_metrics import PrometheusMetrics  # noqa: F401
from headroom.proxy.rate_limiter import TokenBucketRateLimiter  # noqa: F401
from headroom.proxy.request_logger import RequestLogger  # noqa: F401
from headroom.proxy.savings_tracker import LITELLM_AVAILABLE
from headroom.proxy.semantic_cache import SemanticCache  # noqa: F401
from headroom.proxy.ssl_context import build_httpx_verify
from headroom.proxy.tool_schema_savings_policy import tool_schema_saved_from_tags
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
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
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


_AGENT_LABELS: dict[str, str] = {
    "claude": "Claude",
    "claude-code": "Claude",
    "claude_cli": "Claude",
    "claude-code-cli": "Claude",
    "codex": "Codex",
    "codex-cli": "Codex",
    "cursor": "Cursor",
    "grok_build": "Grok Build",
    "copilot": "GitHub Copilot",
    "github-copilot": "GitHub Copilot",
    "aider": "Aider",
    "zed": "Zed",
    "opencode": "OpenCode",
    "openclaw": "OpenClaw",
    "gemini": "Gemini",
    "google": "Gemini",
    "vertex:google": "Gemini",
    "anthropic": "Claude",
    "openai": "OpenAI",
    "unknown": "Unidentified",
}

_AGENT_SOURCE_PRIORITY: dict[str, int] = {
    "unknown": 0,
    "provider": 1,
    "model": 2,
    "stack": 3,
    "client": 4,
}


def _normalize_agent_key(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    value = value.replace(" ", "-").replace("_", "-")
    if value.startswith("wrap-"):
        value = value.removeprefix("wrap-")
    if value in {"claude-cli", "claude-code", "claude-code-cli"}:
        return "claude-code"
    if value in {"codex-cli", "codex"}:
        return "codex"
    if value in {"github-copilot", "copilot"}:
        return "copilot"
    if value in {"google", "vertex-google", "vertex:google"}:
        return "gemini"
    return value


def _agent_label(agent_key: str) -> str:
    if agent_key in _AGENT_LABELS:
        return _AGENT_LABELS[agent_key]
    return agent_key.replace("-", " ").replace("_", " ").title()


def _classify_agent_from_log(entry: dict[str, Any]) -> tuple[str, str, str]:
    raw_tags = entry.get("tags")
    tags = raw_tags if isinstance(raw_tags, dict) else {}
    for source, candidate in (
        ("client", tags.get("client")),
        ("stack", tags.get("stack") or tags.get("headroom-stack")),
    ):
        key = _normalize_agent_key(candidate)
        if key:
            return key, _agent_label(key), source

    model = str(entry.get("model") or "").lower()
    if "codex" in model:
        return "codex", _agent_label("codex"), "model"
    if "claude" in model:
        return "claude-code", _agent_label("claude-code"), "model"
    if "gemini" in model:
        return "gemini", _agent_label("gemini"), "model"

    key = _normalize_agent_key(entry.get("provider"))
    if key:
        return key, _agent_label(key), "provider"

    return "unknown", _agent_label("unknown"), "unknown"


def _remap_provider_counts(counts: dict[str, int], config: ProxyConfig) -> dict[str, int]:
    """Relabel ``openai`` upstream counts with the configured display provider.

    Display only — the stored metrics key stays ``openai`` (issue #1533).
    Collisions (none today) are summed defensively.
    """
    out: dict[str, int] = {}
    for provider, count in counts.items():
        display = resolve_display_provider(
            provider,
            openai_api_url=config.openai_api_url,
            provider_name=config.provider_name,
        )
        out[display] = out.get(display, 0) + int(count)
    return out


def _build_agent_usage_summary(
    logs: list[dict[str, Any]],
    *,
    requests_by_provider: dict[str, int],
    requests_by_model: dict[str, int],
    global_before_tokens: int,
    global_after_tokens: int,
    global_tokens_saved: int,
    global_output_tokens: int,
) -> dict[str, Any]:
    agents: dict[str, dict[str, Any]] = {}

    def _agent_row(agent_key: str, label: str, source: str) -> dict[str, Any]:
        row = agents.setdefault(
            agent_key,
            {
                "agent": agent_key,
                "label": label,
                "source": source,
                "requests": 0,
                "before_tokens": 0,
                "after_tokens": 0,
                "output_tokens": 0,
                "tokens_saved": 0,
                "models": {},
                "providers": {},
                "has_exact_tokens": False,
            },
        )
        if _AGENT_SOURCE_PRIORITY.get(source, 0) > _AGENT_SOURCE_PRIORITY.get(
            str(row.get("source") or "unknown"), 0
        ):
            row["source"] = source
        return row

    for entry in logs:
        agent_key, label, source = _classify_agent_from_log(entry)
        row = _agent_row(agent_key, label, source)
        before = max(0, int(entry.get("input_tokens_original") or 0))
        after = max(0, int(entry.get("input_tokens_optimized") or 0))
        saved = max(0, int(entry.get("tokens_saved") or 0))
        output = max(0, int(entry.get("output_tokens") or 0))
        provider = str(entry.get("provider") or "unknown")
        model = str(entry.get("model") or "unknown")

        row["requests"] += 1
        row["before_tokens"] += before
        row["after_tokens"] += after
        row["output_tokens"] += output
        row["tokens_saved"] += saved
        row["providers"][provider] = int(row["providers"].get(provider, 0)) + 1
        row["models"][model] = int(row["models"].get(model, 0)) + 1
        if before > 0 or after > 0 or saved > 0:
            row["has_exact_tokens"] = True

    if not agents:
        inferred_model_counts: dict[str, int] = {}
        for model, count in requests_by_model.items():
            model_lower = str(model).lower()
            if "codex" in model_lower:
                key = "codex"
            elif "claude" in model_lower:
                key = "claude-code"
            elif "gemini" in model_lower:
                key = "gemini"
            else:
                continue
            inferred_model_counts[str(model)] = int(count)

        provider_request_count = sum(max(0, int(count)) for count in requests_by_provider.values())
        inferred_request_count = sum(max(0, count) for count in inferred_model_counts.values())
        use_model_fallback = (
            inferred_request_count > 0 and inferred_request_count == provider_request_count
        )

        if not use_model_fallback:
            for provider, count in requests_by_provider.items():
                key = _normalize_agent_key(provider) or "unknown"
                row = _agent_row(key, _agent_label(key), "provider")
                row["requests"] += int(count)
                row["providers"][provider] = int(row["providers"].get(provider, 0)) + int(count)
        for model, count in requests_by_model.items():
            model_lower = str(model).lower()
            if "codex" in model_lower:
                key = "codex"
            elif "claude" in model_lower:
                key = "claude-code"
            elif "gemini" in model_lower:
                key = "gemini"
            else:
                continue
            if not use_model_fallback:
                continue
            row = _agent_row(key, _agent_label(key), "model")
            row["requests"] += int(count)
            row["models"][str(model)] = int(row["models"].get(str(model), 0)) + int(count)

    rows: list[dict[str, Any]] = []
    for row in agents.values():
        before = int(row["before_tokens"])
        saved = int(row["tokens_saved"])
        after = int(row["after_tokens"])
        if before == 0 and (after > 0 or saved > 0):
            before = after + saved
        savings_percent = round((saved / before) * 100.0, 2) if before else 0.0
        row["before_tokens"] = before
        row["savings_percent"] = savings_percent
        row["after_percent"] = round((after / before) * 100.0, 2) if before else 0.0
        row["share_of_saved_percent"] = (
            round((saved / global_tokens_saved) * 100.0, 2) if global_tokens_saved else 0.0
        )
        row["share_of_requests_percent"] = 0.0
        rows.append(row)

    total_requests = sum(int(row["requests"]) for row in rows)
    for row in rows:
        row["share_of_requests_percent"] = (
            round((int(row["requests"]) / total_requests) * 100.0, 2) if total_requests else 0.0
        )

    rows.sort(
        key=lambda row: (
            int(row.get("tokens_saved", 0)),
            int(row.get("before_tokens", 0)),
            int(row.get("requests", 0)),
        ),
        reverse=True,
    )

    return {
        "agents": rows,
        "totals": {
            "requests": total_requests,
            "before_tokens": global_before_tokens,
            "after_tokens": global_after_tokens,
            "output_tokens": global_output_tokens,
            "tokens_saved": global_tokens_saved,
            "savings_percent": (
                round((global_tokens_saved / global_before_tokens) * 100.0, 2)
                if global_before_tokens
                else 0.0
            ),
        },
        "coverage": {
            "logged_requests": len(logs),
            "exact_token_rows": sum(1 for row in rows if row.get("has_exact_tokens")),
            "mode": "request_logs" if logs else "aggregate_fallback",
        },
    }


# Suppress "[transformers] PyTorch was not found" warning emitted when
# transformers is imported for availability checks (e.g. kompress ONNX probe).
# PyTorch is optional in headroom; the warning is not actionable for operators.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("headroom.proxy")


def _code_syntax_breaker_status() -> dict[str, Any]:
    """Per-language AST-compression breaker state, or {} if unavailable."""
    try:
        from headroom.transforms.code_compressor import syntax_breaker_status

        return syntax_breaker_status()
    except Exception:  # pragma: no cover - defensive; stats must not fail
        return {}


class _SuppressCancelledErrorFilter(logging.Filter):
    """Hide expected uvicorn CancelledError tracebacks during shutdown."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.ERROR and record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type is not None and issubclass(exc_type, asyncio.CancelledError):
                return False
        return True


LoopExceptionHandler = Callable[[asyncio.AbstractEventLoop, dict[str, Any]], object]


class LoopFailureDetails(TypedDict):
    message: Any | None
    exception: str | None


class LoopHealthState(TypedDict):
    status: str
    known_failures: int
    last_known_failure: LoopFailureDetails | None


class CompressionQuarantinedError(asyncio.TimeoutError):
    """Compression was skipped while a timed-out worker was still running.

    Inherit from ``asyncio.TimeoutError`` rather than the builtin directly so
    Python 3.10 callers classify quarantine bypasses exactly like executor
    timeouts. The two exception classes only became aliases in Python 3.11.
    """


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
    BedrockHandlerMixin,
    GeminiHandlerMixin,
    OpenAIHandlerMixin,
    StreamingMixin,
)


def _apply_stateless_persistence(config: ProxyConfig) -> None:
    """When the proxy runs stateless, force global persisters to in-memory so no
    files are written to the workspace.

    Covers TOIN (the always-on serving writer): it keeps learning patterns
    in-memory but never reads or writes ``toin.json``. An empty ``storage_path``
    makes the backend ``None``, which no-ops load/save/auto-save. The savings
    subsystem is handled separately via ``PrometheusMetrics(stateless=...)``.

    Note: setting ``HEADROOM_TOIN_BACKEND=none`` is NOT sufficient on its own —
    ``ToolIntelligenceNetwork`` falls back to ``config.storage_path`` when no
    backend is passed, so we must clear the path explicitly here.

    Concurrency: ``stateless`` is a per-process flag (set once at ``headroom
    proxy`` launch), never a per-request/per-session value — every session a
    process serves shares it, and two proxies with different settings run as
    separate OS processes with independent TOIN singletons. In the rare case
    where two HeadroomProxy instances with different ``stateless`` settings live
    in ONE process (e.g. tests), this fails closed: the reset forces the
    process-global TOIN in-memory, so a stateless proxy never persists (the safe
    direction). A co-resident stateful proxy would then also stop persisting
    TOIN — acceptable, since not-writing can never leak data.
    """
    if not getattr(config, "stateless", False):
        return
    from headroom.telemetry.toin import TOINConfig, get_toin, reset_toin

    # Reset first so this wins regardless of whether the singleton was already
    # created with a filesystem backend earlier in the process.
    reset_toin()
    get_toin(TOINConfig(storage_path=""))


def _provider_httpx_client_options(
    config: ProxyConfig,
    verify: Any,
) -> tuple[bool, dict[str, Any]]:
    client_kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(
            connect=config.connect_timeout_seconds,
            read=config.request_timeout_seconds,
            # Not request_timeout_seconds: see ProxyConfig.write_timeout_seconds.
            write=config.write_timeout_seconds,
            pool=config.connect_timeout_seconds,
        ),
        "limits": httpx.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_keepalive_connections,
            keepalive_expiry=config.keepalive_expiry,
        ),
        "verify": verify,
    }
    if config.http_proxy:
        client_kwargs["proxy"] = config.http_proxy
    return config.http2 and not config.http_proxy, client_kwargs


# Recognized built-in compressor names → the `ContentRouterConfig` `enable_*`
# flag that gates each one. This is the whole selection surface: a `--compressor`
# selection is mapped onto these existing flags with ZERO new dispatch logic, so
# the router's if/elif built-in dispatch stays byte-identical. Names outside this
# map (e.g. a third-party `headroom.compressor` entry point) are intentionally
# ignored here — they belong to the registry, not the built-in enable_* seam.
BUILTIN_COMPRESSOR_FLAGS: dict[str, str] = {
    "smart_crusher": "enable_smart_crusher",
    "kompress": "enable_kompress",
    "code_aware": "enable_code_aware",
    "search": "enable_search_compressor",
    "log": "enable_log_compressor",
    "tabular": "enable_tabular_compressor",
    "config": "enable_config_compressor",
    "html": "enable_html_extractor",
    "image": "enable_image_optimizer",
}


def _apply_compressor_selection(
    router_config: ContentRouterConfig,
    compressors: set[str] | None,
) -> None:
    """Narrow the built-in compressor set on ``router_config`` in place.

    ``compressors is None`` (the default) is a no-op: every ``enable_*`` flag
    keeps its dataclass default, so behavior is byte-identical to today. When a
    selection is given, each recognized built-in in :data:`BUILTIN_COMPRESSOR_FLAGS`
    is enabled iff it (or the wildcard ``"*"``) was selected, and disabled
    otherwise. Unrecognized names select no flag (reserved for the compressor
    registry) but are surfaced with a warning, because a typo'd selection is
    otherwise indistinguishable from "compress nothing" (#2384). This maps a
    selection onto the existing flags without adding any dispatch logic.
    """
    if compressors is None:
        return
    selected = {name.strip() for name in compressors if name.strip()}
    if not selected:
        return
    select_all = "*" in selected
    unmatched = sorted(selected - set(BUILTIN_COMPRESSOR_FLAGS) - {"*"})
    if unmatched:
        if select_all or selected & set(BUILTIN_COMPRESSOR_FLAGS):
            logger.warning(
                "compressor selection: %s match no built-in compressor "
                "(assumed registry names); built-ins: %s",
                ", ".join(unmatched),
                ", ".join(sorted(BUILTIN_COMPRESSOR_FLAGS)),
            )
        else:
            logger.warning(
                "compressor selection %s matches no built-in compressor — every "
                "built-in compressor is now disabled. If this is a typo, valid "
                "names are: %s (or '*' for all).",
                ", ".join(unmatched),
                ", ".join(sorted(BUILTIN_COMPRESSOR_FLAGS)),
            )
    for name, flag in BUILTIN_COMPRESSOR_FLAGS.items():
        setattr(router_config, flag, select_all or name in selected)


def _external_compressor_selection(compressors: set[str] | None) -> list[str] | None:
    """Return the selected EXTERNAL (non-built-in) compressor names, or ``None``.

    The built-in selection (:func:`_apply_compressor_selection`) consumes only
    the names in :data:`BUILTIN_COMPRESSOR_FLAGS`; every OTHER selected name is a
    third-party ``headroom.compressor`` entry point. This threads those to the
    router (via ``ContentRouterConfig.active_external_compressors``) so it can
    route matching blocks through them.

    Returns ``None`` — the router's external-dispatch branch stays inert, so the
    request path is byte-identical to today — when the selection is empty or
    contains only recognized built-in names. ``"*"`` is preserved so the router
    activates every discovered external compressor (mirroring the wildcard's
    "select everything" meaning on the built-in side).
    """
    if not compressors:
        return None
    selected = {name.strip() for name in compressors if name.strip()}
    if not selected:
        return None
    if "*" in selected:
        return ["*"]
    external = sorted(selected - set(BUILTIN_COMPRESSOR_FLAGS))
    return external or None


class HeadroomProxy(
    StreamingMixin,
    AnthropicHandlerMixin,
    OpenAIHandlerMixin,
    GeminiHandlerMixin,
    BatchHandlerMixin,
    BedrockHandlerMixin,
):
    """Production-ready Headroom optimization proxy."""

    ANTHROPIC_API_URL = DEFAULT_ANTHROPIC_API_URL
    OPENAI_API_URL = DEFAULT_OPENAI_API_URL
    GEMINI_API_URL = DEFAULT_GEMINI_API_URL
    CLOUDCODE_API_URL = DEFAULT_CLOUDCODE_API_URL
    VERTEX_API_URL = DEFAULT_VERTEX_API_URL

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.config.mode = normalize_proxy_mode(self.config.mode)
        # Record process-wide stateless mode so module-level persisters
        # (output-savings recorder, etc.) can skip workspace writes.
        from headroom import paths as _hr_paths

        _hr_paths.set_process_stateless(config.stateless)
        # Stateless: keep TOIN learning in-memory; never touch toin.json.
        _apply_stateless_persistence(self.config)
        pipeline_extensions = list(config.pipeline_extensions or [])
        probe_recorder = probe_recorder_from_env()
        if probe_recorder is not None:
            pipeline_extensions.append(probe_recorder)
        self.pipeline_extensions = PipelineExtensionManager(
            hooks=config.hooks,
            extensions=pipeline_extensions,
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
        HeadroomProxy.VERTEX_API_URL = api_targets.vertex
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
                estimated_basis_policy=config.budget_estimated_basis,
            )
            if config.cost_tracking_enabled
            else None
        )
        self.metrics = PrometheusMetrics(cost_tracker=self.cost_tracker, stateless=config.stateless)

        # Cost-aware model routing (issue #1706). Disabled unless configured, so
        # the default request path is unchanged.
        self.model_router = ModelRouter(config.model_router)

        # Initialize transforms based on routing mode.
        #
        # Phase B PR-B1 retired the IntelligentContextManager / RollingWindow
        # message-dropping branch. Live-zone-only compression (PR-B2..B7) does
        # not drop messages — it operates on content blocks within messages —
        # so the proxy no longer needs a "context manager" transform stage.
        # Reported via metrics as `_context_manager_status = "passthrough"`.
        self._context_manager_status = "passthrough"

        # ContentRouter is the single proxy routing surface. Provider handlers
        # normalize their request shapes into messages or CompressionUnits, and
        # the router chooses SmartCrusher, log/search/diff/code, or Kompress.
        profile_kwargs = proxy_pipeline_kwargs(config)
        router_config = ContentRouterConfig(
            enable_code_aware=config.code_aware_enabled,
            prefer_code_aware_for_code=_get_env_bool("HEADROOM_PREFER_CODE_AWARE_FOR_CODE", True),
            tool_profiles=config.tool_profiles,
            read_lifecycle=ReadLifecycleConfig(enabled=config.read_lifecycle),
            smart_crusher_max_items_after_crush=cast(
                int | None,
                profile_kwargs.get("max_items_after_crush"),
            ),
            smart_crusher_with_compaction=cast(
                bool,
                profile_kwargs.get("smart_crusher_with_compaction", True),
            ),
            ccr_inject_marker=config.ccr_inject_marker,
            force_kompress_all=config.force_kompress_all,
            lossless=config.lossless,
        )
        # Compressor selection (opt-in). None keeps every built-in enabled
        # (default, byte-identical to today); a selection maps the recognized
        # built-in names onto the `enable_*` flags just constructed above.
        # Runs BEFORE the disable_kompress override below so that flag stays
        # authoritative for turning Kompress off.
        _apply_compressor_selection(router_config, config.compressors)
        # External (non-built-in) `headroom.compressor` selections are ignored by
        # `_apply_compressor_selection` (they have no enable_* flag). Thread them
        # to the router here so it can route matching blocks through them; None
        # (no external selected) keeps the external-dispatch branch inert.
        router_config.active_external_compressors = _external_compressor_selection(
            config.compressors
        )
        # No-CCR lossless mode: compress tool outputs with format-native
        # lossless compaction and marker-free SmartCrusher, and suppress every
        # retrieval marker + the retrieve-tool injection so no MCP round-trip is
        # needed. Mirrors the force_kompress_all wiring precedent.
        if config.lossless:
            router_config.lossless = True
            router_config.smart_crusher_lossless_only = True
            router_config.ccr_inject_marker = False
            if hasattr(config, "ccr_inject_tool"):
                config.ccr_inject_tool = False
        if config.disable_kompress:
            router_config.enable_kompress = False
            # Opt-in restore of the legacy behaviour: send fall-through content
            # to PASSTHROUGH instead of the default KOMPRESS fallback strategy.
            if config.disable_kompress_fallback:
                router_config.fallback_strategy = CompressionStrategy.PASSTHROUGH
        # `HEADROOM_LOSSLESS_ONLY=1` routes SmartCrusher through strict
        # marker-free mode: lossless tabular compaction still applies, but
        # any path that would emit a `<<ccr:…>>` marker (row-drop or
        # opaque-blob offload) leaves the content uncompacted instead — so
        # the session needs no CCR retrieval round-trips to stay recoverable.
        if "HEADROOM_LOSSLESS_ONLY" in os.environ:
            router_config.smart_crusher_lossless_only = _get_env_bool(
                "HEADROOM_LOSSLESS_ONLY", False
            )
        # A non-None exclude_tools replaces DEFAULT_EXCLUDE_TOOLS in
        # ContentRouter, so merge rather than assign.
        if config.exclude_tools:
            router_config.exclude_tools = set(DEFAULT_EXCLUDE_TOOLS) | config.exclude_tools
        # protect_tool_results: force-merge named tools into the exclude set
        # so their results are never lossy-compressed, regardless of mode.
        if config.protect_tool_results:
            base = (
                router_config.exclude_tools
                if router_config.exclude_tools is not None
                else set(DEFAULT_EXCLUDE_TOOLS)
            )
            router_config.exclude_tools = base | config.protect_tool_results
        # Token mode: allow compression of older excluded-tool results,
        # and emit search results grouped by file (path once per file
        # instead of repeated on every match line).
        if is_token_mode(config.mode):
            router_config.protect_recent_reads_fraction = 0.3
            router_config.search_group_by_file = True
        # Note: protect_tool_results runs AFTER token mode (ordering matters).
        # It resets protect_recent_reads_fraction from 0.3→0.0, restoring
        # full protection for ALL excluded-tool results regardless of age.
        # This means naming any tool with --protect-tool-results also
        # protects Read/Glob/Grep/Write/Edit results indefinitely.
        if config.protect_tool_results:
            router_config.protect_recent_reads_fraction = 0.0
        # `--compress-user-messages` flips the router's default skip rule.
        # Off by default for prefix-cache safety; enabled for workloads where
        # user-message content dominates input (OpenAI/Azure chat with pasted
        # code/RAG context — see issue #454).
        if profile_kwargs.get("compress_user_messages"):
            router_config.skip_user_messages = False
        # Kompress (lossy ML text compression) is resolved per provider. The
        # global `disable_kompress` above is the baseline for both; a per-
        # provider override (disable_kompress_{anthropic,openai}) wins when set.
        # Only `enable_kompress` differs between providers — routing, tool
        # exclusion, and read-protection are identical — so when both resolve
        # the same we reuse ONE ContentRouter instance and the Kompress model
        # still loads once (startup warmup dedupes transforms by id()).
        base_kompress_disabled = not router_config.enable_kompress
        anthropic_kompress_disabled = (
            base_kompress_disabled
            if config.disable_kompress_anthropic is None
            else config.disable_kompress_anthropic
        )
        openai_kompress_disabled = (
            base_kompress_disabled
            if config.disable_kompress_openai is None
            else config.disable_kompress_openai
        )

        def _router_config_for(kompress_disabled: bool) -> ContentRouterConfig:
            if kompress_disabled == base_kompress_disabled:
                return router_config
            return replace(router_config, enable_kompress=not kompress_disabled)

        cache_aligner = CacheAligner(CacheAlignerConfig(enabled=False))
        anthropic_router = ContentRouter(
            _router_config_for(anthropic_kompress_disabled), observer=self.metrics
        )
        openai_router = (
            anthropic_router
            if openai_kompress_disabled == anthropic_kompress_disabled
            else ContentRouter(_router_config_for(openai_kompress_disabled), observer=self.metrics)
        )
        self._code_aware_status = "lazy" if config.code_aware_enabled else "disabled"

        _intercept_prefix: list = []
        assert config.rollout is not None
        if config.rollout.is_enabled("tool_result_interceptors"):
            from headroom.proxy.interceptors import ToolResultInterceptorTransform

            _intercept_prefix = [ToolResultInterceptorTransform()]

        self.anthropic_pipeline = TransformPipeline(
            transforms=[*_intercept_prefix, cache_aligner, anthropic_router],
            provider=self.anthropic_provider,
        )
        self.openai_pipeline = TransformPipeline(
            transforms=[*_intercept_prefix, cache_aligner, openai_router],
            provider=self.openai_provider,
        )
        # Build the DEFAULT /v1/compress pipeline now, not on first request.
        # It is a ContentRouter derived from `openai_router` (marker-free), so
        # a lazy build would land inside the bounded compression executor on a
        # cold pod's very first gateway request — paying router construction
        # and, when the ML model runs in-process, model load against the 30 s
        # compression budget. Building it here also lets startup warmup see it
        # (`_eager_preload_transforms`). Never fatal: on failure the handler
        # falls back to the original lazy path.
        try:
            self._no_ccr_pipeline()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Eager /v1/compress pipeline build failed (%s); deferring to first request",
                exc,
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
        # Ordered by last access: `_get_compression_cache` moves a session to
        # the end on every hit, so capacity eviction drops the idlest sessions
        # — whose provider prefix cache has lapsed anyway — never a busy
        # long-lived one. `_compression_cache_last_seen` drives the idle-TTL
        # sweep in `_maybe_cleanup_compression_caches`.
        self._compression_caches: OrderedDict[str, CompressionCache] = OrderedDict()
        self._compression_cache_last_seen: dict[str, float] = {}
        self._compression_caches_last_cleanup: float = time.time()
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
        # HTTP/1.1-only client for ChatGPT passthrough (Cloudflare challenges
        # our HTTP/2 fingerprint on its sensitive account endpoints).
        self.http_client_h1: httpx.AsyncClient | None = None
        self._shutdown_event: asyncio.Event | None = None

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
            _compression_max = max(1, os.cpu_count() or 1)
        else:
            _compression_max = max(1, _compression_max_cfg)
        self.compression_max_workers: int = _compression_max
        self._compression_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_compression_max,
            thread_name_prefix="headroom-compress",
        )
        # Phase 3 (#1171): off-path background compression. When enabled, a
        # cold-start-large request (frozen=0 + large live zone) forwards
        # uncompressed immediately and enqueues the compression here instead of
        # blocking the request thread under the 30s budget (which leaks a
        # non-preemptible worker -> executor saturation -> cascade). Default
        # off (opt-in), fail-open. Per-process, matching _compression_caches.
        self._background_compression_enabled: bool = os.environ.get(
            "HEADROOM_BACKGROUND_COMPRESSION", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        try:
            self._background_compression_min_tokens: int = int(
                os.environ.get("HEADROOM_BACKGROUND_COMPRESSION_MIN_TOKENS", "50000")
            )
        except ValueError:
            self._background_compression_min_tokens = 50000
        # Dedicated single thread: no-timeout background jobs never contend with
        # the request-path executor (Phase 3, #1171). Lazy -- no thread spawns
        # until the first off-path job is submitted.
        self._background_compression_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="headroom-bg-compress"
        )
        self._background_compressor = BackgroundCompressor(self._run_compression_background)
        # Gauge: currently-running compression tasks. Mutated under
        # ``_compression_metrics_lock`` from worker threads + the asyncio
        # event loop.
        self._compression_queued: int = 0
        self._compression_queued_max: int = 0
        self._compression_queue_timeouts: int = 0
        self._compression_queue_wait_seconds_total: float = 0.0
        self._compression_queue_wait_seconds_max: float = 0.0
        self._compression_in_flight: int = 0
        # High-water mark for in-flight count.
        self._compression_in_flight_max: int = 0
        self._compression_run_seconds_total: float = 0.0
        self._compression_run_seconds_max: float = 0.0
        # Counter: threads that finished AFTER their asyncio future hit the
        # timeout. Stuck-thread leak indicator.
        self._compression_leaked_threads: int = 0
        # Timeout-debt quarantine. Python cannot preempt a worker after its
        # asyncio waiter times out, so accepting more compression while that
        # worker is still running can multiply one slow call into a saturated
        # executor. New work raises immediately until every known post-timeout
        # worker has genuinely exited, then follows the caller's existing
        # compression-failure policy.
        self._compression_timed_out_in_flight: int = 0
        self._compression_timed_out_in_flight_max: int = 0
        self._compression_quarantine_activations: int = 0
        self._compression_quarantine_skips: int = 0
        # Time cap on the timeout-debt quarantine. Python cannot preempt a
        # worker, so one that never returns (a hung/pathological compression)
        # would keep ``_compression_timed_out_in_flight > 0`` forever and pin the
        # quarantine open — zeroing out ALL further compression (#2360). The
        # deadline is (re)set on every fresh timeout; once it lapses with no new
        # timeouts the leaked worker is presumed abandoned and compression
        # resumes. The bounded executor still caps thread growth, and any new
        # timeout re-arms the quarantine.
        self._compression_quarantine_deadline: float = 0.0
        self._compression_quarantine_max_seconds: float = _get_env_float(
            "HEADROOM_COMPRESSION_QUARANTINE_MAX_SECONDS", 60.0
        )
        self._compression_quarantine_releases: int = 0
        self._compression_metrics_lock = threading.Lock()

        # Backend for Anthropic API (direct, LiteLLM, or any-llm)
        # Supports: "anthropic" (direct), "bedrock", "vertex", "litellm-<provider>", or "anyllm"
        self.anthropic_backend: Backend | None = create_proxy_backend(
            backend=config.backend,
            anyllm_provider=config.anyllm_provider,
            bedrock_region=config.bedrock_region,
            bedrock_profile=config.bedrock_profile,
            logger=logger,
            openai_api_url=config.openai_api_url,
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
        if config.memory_enabled and config.stateless:
            # Persistent memory writes a SQLite DB + markdown files to disk,
            # which stateless mode forbids. Memory is cross-session learning and
            # is contradictory with an ephemeral/read-only deployment, so we
            # disable it rather than persist. Run without --stateless to use it.
            logger.warning(
                "Memory is disabled in stateless mode (it persists to disk). "
                "Run without --stateless to enable persistent memory."
            )
        elif config.memory_enabled:
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

            from headroom.memory.storage_router import MemoryStorageMode

            try:
                _storage_mode = MemoryStorageMode(config.memory_storage_mode)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid memory_storage_mode={config.memory_storage_mode!r}; "
                    f"expected one of {[m.value for m in MemoryStorageMode]}"
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
                storage_mode=_storage_mode,
                project_root_override=config.memory_project_root_override,
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

            # Migration UX (GH #462). When the user is on the new
            # project-scoped default but a legacy single-file DB exists
            # with prior memories, surface that clearly so it doesn't
            # look like an upgrade ate their data.
            if _storage_mode is MemoryStorageMode.PROJECT:
                _legacy_path = Path(_mem_db_path)
                if _legacy_path.exists() and _legacy_path.stat().st_size > 0:
                    logger.info(
                        "event=memory_storage_legacy_detected path=%s mode=project "
                        "hint=pass_--memory-storage=global_to_reach_pre-fix_memories",
                        _legacy_path,
                    )

            # The Memory Bridge binds to the single legacy backend at
            # init time; it doesn't (yet) follow per-project routing.
            # Warn so users running bridge + project mode aren't
            # surprised that only the legacy DB syncs with markdown.
            if config.memory_bridge_enabled and _storage_mode is MemoryStorageMode.PROJECT:
                logger.warning(
                    "event=memory_bridge_global_only mode=project "
                    "hint=bridge_syncs_only_the_legacy_DB_today_per-project_bridge_follow-up_planned"
                )

        # Usage Reporter (license validation + phone-home for managed/enterprise).
        # Suppressed entirely in offline mode — the air-gap switch must stop all
        # egress, including license phone-home, even when a key is configured.
        self.usage_reporter: UsageReporter | None = None
        if config.license_key and not (config.offline or is_offline()):
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
        marking the call timed out on the asyncio side and incrementing
        ``_compression_leaked_threads`` from the worker's ``finally``
        block after it eventually finishes. While any such worker remains,
        new calls raise :class:`CompressionQuarantinedError` immediately so
        callers apply the existing compression-failure policy instead of
        filling the rest of the pool with the same timeout debt. Jobs that are
        successfully cancelled before a worker starts are removed from the
        queued gauge and do not activate the quarantine. If cancellation races
        with worker startup, the now-running job is tracked as timeout debt and
        does activate it.

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
            unchanged. :class:`CompressionQuarantinedError` (an
            ``asyncio.TimeoutError`` subclass) if a prior timed-out worker is
            still running.
        """
        now = time.monotonic()
        with self._compression_metrics_lock:
            timed_out_in_flight = self._compression_timed_out_in_flight
            quarantined = timed_out_in_flight > 0 and now < self._compression_quarantine_deadline
            # Debt outlived the cap: presume the worker leaked/hung and stop
            # blocking on it. Count the release once per lapse (while debt stands
            # and the deadline has passed) so operators can see it happened.
            released = (
                timed_out_in_flight > 0
                and not quarantined
                and self._compression_quarantine_deadline > 0.0
            )
            if quarantined:
                self._compression_quarantine_skips += 1
            if released:
                self._compression_quarantine_releases += 1
                # Clear the deadline so the release is recorded once, not on
                # every subsequent request until the worker (maybe never) exits.
                self._compression_quarantine_deadline = 0.0

        if released:
            self.metrics.record_compression_quarantine("released")
            logger.warning(
                "Compression quarantine released after %.0fs cap; %d timed-out "
                "worker(s) presumed leaked. Compression resumes.",
                self._compression_quarantine_max_seconds,
                timed_out_in_flight,
            )

        if quarantined:
            self.metrics.record_compression_quarantine("skipped")
            raise CompressionQuarantinedError(
                f"compression quarantined: {timed_out_in_flight} timed-out worker(s) still running"
            )

        loop = asyncio.get_running_loop()
        queued_at = time.monotonic()
        state = {
            "queued": True,
            "started": False,
            "finished": False,
            "timed_out": False,
            "timeout_debt_recorded": False,
        }
        with self._compression_metrics_lock:
            self._compression_queued += 1
            if self._compression_queued > self._compression_queued_max:
                self._compression_queued_max = self._compression_queued

        def _record_timeout_debt_locked() -> bool:
            """Record a still-running post-timeout worker; lock must be held.

            Returns ``True`` only when this worker transitions the executor
            from clear to quarantined.
            """
            if (
                not state["timed_out"]
                or not state["started"]
                or state["finished"]
                or state["timeout_debt_recorded"]
            ):
                return False
            was_clear = self._compression_timed_out_in_flight == 0
            self._compression_timed_out_in_flight += 1
            self._compression_timed_out_in_flight_max = max(
                self._compression_timed_out_in_flight_max,
                self._compression_timed_out_in_flight,
            )
            # (Re)arm the quarantine time cap on every fresh timeout, so ongoing
            # slowness keeps quarantining while a single leaked worker cannot
            # hold it past the cap (#2360).
            self._compression_quarantine_deadline = (
                time.monotonic() + self._compression_quarantine_max_seconds
            )
            state["timeout_debt_recorded"] = True
            if was_clear:
                self._compression_quarantine_activations += 1
            return was_clear

        def _announce_quarantine() -> None:
            self.metrics.record_compression_quarantine("activated")
            logger.warning(
                "Compression worker exceeded its request deadline and is still running; "
                "new compression is quarantined until timed-out workers exit"
            )

        def _wrapped():  # noqa: ANN202
            started_at = time.monotonic()
            queue_wait = started_at - queued_at
            with self._compression_metrics_lock:
                state["started"] = True
                if state["queued"]:
                    self._compression_queued -= 1
                    state["queued"] = False
                self._compression_queue_wait_seconds_total += queue_wait
                if queue_wait > self._compression_queue_wait_seconds_max:
                    self._compression_queue_wait_seconds_max = queue_wait
                self._compression_in_flight += 1
                if self._compression_in_flight > self._compression_in_flight_max:
                    self._compression_in_flight_max = self._compression_in_flight
                quarantine_activated = _record_timeout_debt_locked()
            if quarantine_activated:
                _announce_quarantine()
            try:
                return fn()
            finally:
                elapsed = time.monotonic() - started_at
                with self._compression_metrics_lock:
                    state["finished"] = True
                    self._compression_in_flight -= 1
                    self._compression_run_seconds_total += elapsed
                    if elapsed > self._compression_run_seconds_max:
                        self._compression_run_seconds_max = elapsed
                    if state["timed_out"]:
                        self._compression_leaked_threads += 1
                    if state["timeout_debt_recorded"]:
                        self._compression_timed_out_in_flight -= 1
                        state["timeout_debt_recorded"] = False
                        quarantine_cleared = self._compression_timed_out_in_flight == 0
                    else:
                        quarantine_cleared = False
                if quarantine_cleared:
                    logger.info("Compression quarantine cleared after all timed-out workers exited")

        future = loop.run_in_executor(self._compression_executor, _wrapped)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            with self._compression_metrics_lock:
                state["timed_out"] = True
                if state["queued"]:
                    self._compression_queued -= 1
                    state["queued"] = False
                    self._compression_queue_timeouts += 1
                quarantine_activated = _record_timeout_debt_locked()
            if quarantine_activated:
                _announce_quarantine()
            raise

    async def _run_compression_background(self, fn):  # noqa: ANN001, ANN201
        """Run a compression callable on the shared executor with NO request-
        coupled deadline (Phase 3 off-path, #1171).

        Unlike ``_run_compression_in_executor`` there is no ``asyncio.wait_for``
        and no leaked-thread accounting: no caller is waiting, so a slow run
        backs up the background queue rather than starving the request executor.
        Runs on the dedicated single-thread background executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._background_compression_executor, fn)

    # How often the lazy TTL sweep in `_get_compression_cache` may run.
    _COMPRESSION_CACHE_CLEANUP_INTERVAL_SECONDS = 60.0

    def _maybe_cleanup_compression_caches(self, now: float) -> None:
        """Evict per-session compression caches idle past their TTL.

        Caller must hold `_compression_caches_lock`. Piggybacked on
        `_get_compression_cache` (the same lazy-sweep pattern as
        `PrefixCacheTrackerRegistry._maybe_cleanup`) so no background task is
        needed: any traffic at all keeps memory tracking the active-session
        window, and a fully idle process has no memory pressure worth a timer.

        A session idle longer than `COMPRESSION_CACHE_TTL_SECONDS` has
        outlived the provider prompt cache its entries protect — the default
        exceeds Anthropic's 1h extended breakpoint, the longest provider TTL
        served — so evicting it cannot bust anything: the provider already
        forgot the prefix. If the session does return, the cost is one
        cache-write turn (fail-open), which it was going to pay regardless.
        The TTL must never be set below the prefix tracker's session TTL:
        after the tracker expires, this cache's byte-identical swap is the
        only remaining protection for a still-live provider prefix.
        """
        if now - self._compression_caches_last_cleanup < (
            self._COMPRESSION_CACHE_CLEANUP_INTERVAL_SECONDS
        ):
            return
        self._compression_caches_last_cleanup = now
        # Skip sessions with a turn in flight (session_turn_lock held): popping
        # one would hand its retry a FRESH cache with a NEW lock — straggler
        # and retry then run unserialized against the same tracker, and the
        # retry's empty cache recompresses previously-returned content into
        # different bytes. An in-flight session is by definition not idle; it
        # will be swept on a later pass once genuinely quiet.
        expired = [
            sid
            for sid, seen in self._compression_cache_last_seen.items()
            if now - seen > COMPRESSION_CACHE_TTL_SECONDS
            and (cache := self._compression_caches.get(sid)) is not None
            and not cache.session_turn_lock.locked()
        ]
        for sid in expired:
            self._compression_caches.pop(sid, None)
            self._compression_cache_last_seen.pop(sid, None)
        if expired:
            logger.info(
                "Evicted %d compression caches idle > %.0fs (%d sessions remain)",
                len(expired),
                COMPRESSION_CACHE_TTL_SECONDS,
                len(self._compression_caches),
            )

    def _peek_compression_cache(self, session_id: str) -> CompressionCache | None:
        """Return the session's cache if one exists — no create, no LRU bump.

        For lookup paths that must not leave a footprint or distort access
        recency (e.g. /v1/usage taking the session turn lock): an unknown
        session answers None instead of allocating an empty cache.
        """
        with self._compression_caches_lock:
            return self._compression_caches.get(session_id)

    def _get_compression_cache(self, session_id: str) -> CompressionCache:
        """Get or create a CompressionCache for a session.

        Thread-safe under `_compression_caches_lock`: a concurrent pair of
        `_get_compression_cache(session_id)` calls (e.g. two async requests
        for the same conversation) must return the **same** instance,
        otherwise the per-session cache state splits and the two halves
        diverge across requests.

        Every access refreshes both the LRU position and the idle-TTL clock,
        so eviction — capacity or TTL — only ever hits sessions that have
        gone quiet. Losing one costs at most a single cache-write turn
        upstream; it never fails a request.
        """
        with self._compression_caches_lock:
            now = time.time()
            self._maybe_cleanup_compression_caches(now)
            cache = self._compression_caches.get(session_id)
            if cache is None:
                from headroom.cache.compression_cache import CompressionCache

                # Evict the least-recently-used quarter at capacity. The
                # OrderedDict is maintained in access order, so the front is
                # always the idlest session — never a busy long-lived one.
                # Sessions with a turn in flight (session_turn_lock held) are
                # skipped: popping one splits its lock across two cache
                # instances and desyncs the straggler from its retry (see the
                # TTL sweep's comment). If every candidate is mid-turn, no
                # eviction happens this round — briefly exceeding the cap is
                # cheaper than a guaranteed prefix bust.
                if len(self._compression_caches) >= MAX_COMPRESSION_CACHE_SESSIONS:
                    evict_count = min(
                        max(1, MAX_COMPRESSION_CACHE_SESSIONS // 4),
                        len(self._compression_caches),
                    )
                    evictable = [
                        sid
                        for sid, c in self._compression_caches.items()
                        if not c.session_turn_lock.locked()
                    ][:evict_count]
                    for sid in evictable:
                        del self._compression_caches[sid]
                        self._compression_cache_last_seen.pop(sid, None)
                    if evictable:
                        logger.info(
                            "Evicted %d least-recently-used compression caches "
                            "(exceeded %d max sessions)",
                            len(evictable),
                            MAX_COMPRESSION_CACHE_SESSIONS,
                        )

                cache = CompressionCache(max_entries=COMPRESSION_CACHE_MAX_ENTRIES)
                self._compression_caches[session_id] = cache
            else:
                self._compression_caches.move_to_end(session_id)
            self._compression_cache_last_seen[session_id] = now
            return cache

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

    def _eager_preload_transforms(self) -> tuple[dict[str, str], list[dict[str, str]]]:
        """Eagerly load every compressor/parser/detector once (dedup by ``id()``).

        Pure load: returns the merged ``eager_status`` plus the per-transform
        status dicts for the caller to merge into ``self.warmup`` on the main
        thread (``WarmupRegistry`` is not written off-thread). This runs via
        ``asyncio.to_thread`` so a slow or hung native model load cannot keep
        startup from binding the port (#790).
        """
        eager_status: dict[str, str] = {}
        transform_statuses: list[dict[str, str]] = []
        seen_transform_ids: set[int] = set()
        # The derived /v1/compress pipelines (built eagerly in __init__) own
        # their own ContentRouter instances, so dedup-by-id() does not cover
        # them via the two request pipelines — warm them explicitly or the
        # first gateway request still pays the compressor load.
        derived_pipelines = list(getattr(self, "_compress_pipeline_cache", {}).values())
        for pipeline in (self.anthropic_pipeline, self.openai_pipeline, *derived_pipelines):
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
                # Merge: later writers win only if the key wasn't set. Preload a
                # transform ONCE — if another pipeline also has
                # ``eager_load_compressors`` it contributes only new keys.
                for key, value in transform_status.items():
                    eager_status.setdefault(key, value)
                transform_statuses.append(transform_status)

        # LiteLLM's pricing tables. MEASURED 2.9-3.8s to import, and it was
        # being imported lazily ON THE EVENT LOOP during the first request:
        # emit_request_outcome -> record_request -> _estimate_compression_savings_usd
        # calls it before its own `tokens_saved <= 0` early return, so even a
        # request that saved nothing pays for it. Nothing about that is visible
        # as a failure; it just makes one unlucky user wait ~3s.
        #
        # This function already runs under asyncio.to_thread, so importing here
        # cannot delay the port bind.
        try:
            from .savings_tracker import _get_litellm_module

            eager_status.setdefault(
                "litellm", "ready" if _get_litellm_module() is not None else "not installed"
            )
        except Exception as exc:  # pricing is optional; never block startup on it
            logger.debug("LiteLLM pre-load skipped: %s", exc)
            eager_status.setdefault("litellm", "skipped")

        return eager_status, transform_statuses

    async def startup(self):
        """Initialize async resources."""
        self._get_shutdown_event().clear()
        self.pipeline_extensions.emit(
            PipelineStage.PRE_START,
            operation="proxy.startup",
            metadata={"port": self.config.port, "host": self.config.host},
        )
        # Resolve TLS verification: a custom CA bundle (corporate PKI) if one
        # is configured, else a strict-relaxed default context when
        # HEADROOM_TLS_STRICT=0, else httpx's default strict verification.
        _verify = build_httpx_verify()
        _http2, _client_kwargs = _provider_httpx_client_options(self.config, _verify)
        self.http_client = httpx.AsyncClient(http2=_http2, **_client_kwargs)
        # Reuse the primary client when HTTP/2 is already off; otherwise keep a
        # dedicated HTTP/1.1 client for ChatGPT passthrough.
        self.http_client_h1 = (
            self.http_client if not _http2 else httpx.AsyncClient(http2=False, **_client_kwargs)
        )
        logger.info("Headroom Proxy started (version %s)", __version__)
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
        # Effective compression posture, resolved (not merely requested). The
        # savings profile reaches the router through two paths — the config
        # object and the seeded process env — and `setdefault` semantics mean a
        # stale HEADROOM_* value silently overrides the profile that names it.
        # A deployment can therefore log `savings_profile=coding` while actually
        # running balanced's thresholds, and the only prior evidence was a
        # single easily-missed WARNING. Print what is actually in force so
        # "which profile am I really running?" is answerable from the banner.
        try:
            _eff = proxy_pipeline_kwargs(self.config)
            # Read cross-turn dedup off the CONSTRUCTED router, not off the env.
            # ContentRouter resolves it as `config.enable_cross_turn_dedup OR
            # $HEADROOM_DEDUPE`, so reporting the env alone would be a guess that
            # happens to be right only while nothing sets the config field. A
            # banner line exists to be trusted; it must read what was resolved.
            _dedupe: object = "unknown"
            for _t in getattr(self.anthropic_pipeline, "transforms", []):
                if isinstance(_t, ContentRouter):
                    _dedupe = bool(getattr(_t, "_cross_turn_dedup_enabled", False))
                    break
            logger.info(
                "Savings profile: %s (effective: min_tokens=%s min_chars_block=%s "
                "compress_user=%s dedupe=%s tool_search=%s)",
                self.config.savings_profile or DEFAULT_PROFILE,
                _eff.get("min_tokens_to_compress", "default"),
                _eff.get("min_chars_for_block_compression", "default(500)"),
                _eff.get("compress_user_messages", False),
                _dedupe,
                os.environ.get("HEADROOM_TOOL_SEARCH", "1") in ("1", "true", "yes", "on", "auto"),
            )
        except Exception:  # never let a banner line block startup
            logger.debug("effective savings-profile banner skipped", exc_info=True)
        logger.info(f"Caching: {'ENABLED' if self.config.cache_enabled else 'DISABLED'}")
        logger.info(f"Rate Limiting: {'ENABLED' if self.config.rate_limit_enabled else 'DISABLED'}")
        logger.info(
            f"Connection Pool: max_connections={self.config.max_connections}, "
            f"max_keepalive={self.config.max_keepalive_connections}, "
            f"http2={'ENABLED' if _http2 else 'DISABLED'}"
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

        logger.info("Smart Routing: ENABLED (ContentRouter is always active)")

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
            # Run the preload OFF the event loop with a bound. The loop body
            # already swallows per-transform Exceptions, so the only thing that
            # can still block ASGI lifespan startup (and therefore the socket
            # bind) is a hang or an uncatchable native stall during a model load
            # on Windows — the "never opens its port" failure in #790. Capping it
            # means startup always returns and uvicorn binds; on timeout the
            # transforms simply fall back to lazy loading on first use.
            transform_statuses: list[dict[str, str]] = []
            try:
                eager_status, transform_statuses = await asyncio.wait_for(
                    asyncio.to_thread(self._eager_preload_transforms),
                    timeout=EAGER_PRELOAD_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.warning(
                    "Eager preload exceeded %.0fs or failed (%s); continuing so "
                    "the proxy still binds — transforms load lazily on first use.",
                    EAGER_PRELOAD_TIMEOUT_SECONDS,
                    exc,
                )
                eager_status, transform_statuses = {}, []
            # Merge warmup status on the main thread (WarmupRegistry is not
            # written off-thread).
            for transform_status in transform_statuses:
                self.warmup.merge_transform_status(transform_status)

        # Update internal status from eager loading results
        if eager_status.get("kompress") in {"enabled", "deferred"}:
            self._kompress_status = eager_status["kompress"]
        if eager_status.get("code_aware") == "enabled":
            self._code_aware_status = "enabled"

        # Log component status
        if self._kompress_status == "enabled":
            logger.info("Kompress: ENABLED (ModernBERT token compressor)")
        elif self._kompress_status == "deferred":
            logger.info("Kompress: DEFERRED (model loads on first request)")
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
            if (
                self.config.memory_backend == "qdrant-neo4j"
                and not self.config.memory_neo4j_password
            ):
                logger.warning(
                    "NEO4J password is not set — using default credentials is insecure in production"
                )
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

        # Log local telemetry status so operators can see it in the log stream.
        # Nothing is sent externally — telemetry is collected locally only (the
        # anonymous telemetry beacon was removed); operational metrics export
        # only to your own OTEL collector via HEADROOM_OTEL_METRICS_*.
        if is_telemetry_enabled():
            logger.info(
                "Local telemetry: ENABLED (aggregate stats, local only — nothing sent "
                "externally). Opt out: HEADROOM_TELEMETRY=off or --no-telemetry"
            )
        else:
            logger.info(
                "Local telemetry: DISABLED (off by default — opt in: "
                "HEADROOM_TELEMETRY=on or --telemetry)"
            )

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
        self._get_shutdown_event().set()
        if self.http_client_h1 and self.http_client_h1 is not self.http_client:
            await self.http_client_h1.aclose()
        self.http_client_h1 = None
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

        # Persist any savings the tracker's write throttle is still holding, so
        # a graceful shutdown doesn't drop the last few requests' totals.
        with contextlib.suppress(Exception):
            self.metrics.savings_tracker.flush()

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
        # ONE headline: message compression + tool-schema deferral. Deferral can't move
        # tok_before/after (tool bytes never reach count_messages), so it used to print
        # as a separate line that read like a side metric rather than savings.
        logger.info(f"Tokens saved:          {m.tokens_saved_total + m.tool_search_saved_total:,}")
        if m.tool_search_saved_total > 0:
            logger.info(f"  messages:            {m.tokens_saved_total:,}")
            logger.info(f"  tool schemas:        {m.tool_search_saved_total:,}")
        # Active-compression ratio: savings as a fraction of what we
        # *attempted* to compress (extracted units + tool schema),
        # NOT the whole request. The full-request denominator is
        # dominated by frozen prefix bytes (instructions, user msgs,
        # prior turns) that we never touch — including them collapses
        # the headline number even on sessions where every attempted
        # compression succeeded.
        attempted = getattr(m, "attempted_input_tokens_total", 0)
        if attempted > 0:
            # `attempted` is pre-compression; savings rate is plain
            # saved / attempted.
            savings_pct = (m.tokens_saved_total / attempted) * 100
            logger.info(f"Active compression:    {savings_pct:.1f}%")
            logger.info(f"  (attempted tokens:   {attempted:,})")
        if m.tokens_input_total > 0:
            whole_request_pct = (
                m.tokens_saved_total / (m.tokens_input_total + m.tokens_saved_total)
            ) * 100
            logger.info(f"Of total wire traffic: {whole_request_pct:.2f}%")
        if m.latency_count > 0:
            avg_latency = m.latency_sum_ms / m.latency_count
            logger.info(f"Avg latency:           {avg_latency:.0f}ms")
        logger.info("=" * 70)

    async def _record_request_outcome(self, outcome: RequestOutcome) -> None:
        """Single funnel for per-request bookkeeping.

        Thin wrapper around :func:`headroom.proxy.outcome.emit_request_outcome`
        so call sites can write ``await self._record_request_outcome(outcome)``
        (idiomatic) instead of ``await emit_request_outcome(self, outcome)``.
        The real implementation lives in ``outcome.py`` as a free function so
        test dummies and provider mixins can call it without inheriting from
        ``HeadroomProxy``.

        See ``docs/superpowers/specs/P0-proxy-pipeline-audit.md`` for the
        divergence catalog this funnel collapses.
        """
        from headroom.proxy.outcome import emit_request_outcome

        # Shielded because four call sites are `finally:` blocks inside streaming
        # async generators (streaming.py:1611, :1859, :2069, openai.py:8614). A
        # client disconnect cancels that task, and the funnel now suspends partway
        # through: `record_request` commits the Prometheus counters, then awaits
        # the ledger append in a worker thread. A cancellation landing on that
        # await leaves the request counted in Prometheus but missing from the cost
        # tracker, the request log, and the PERF line `headroom perf` reads.
        #
        # The shield does not swallow the cancellation — the await below still
        # raises CancelledError, so generator teardown propagates exactly as
        # before. It only keeps the bookkeeping from being torn in half.
        await asyncio.shield(emit_request_outcome(self, outcome))

    async def _next_request_id(self) -> str:
        """Generate unique request ID."""
        async with self._request_counter_lock:
            self._request_counter += 1
            return f"hr_{int(time.time())}_{self._request_counter:06d}"

    def _extract_tags(self, headers: dict) -> dict[str, str]:
        """Backwards-compat wrapper around :func:`extract_tags`.

        Handlers call ``extract_tags(headers)`` directly. Kept here for
        any external caller still using ``proxy._extract_tags(headers)``.
        """
        from headroom.proxy.helpers import extract_tags

        return extract_tags(headers)

    def _get_shutdown_event(self) -> asyncio.Event:
        event = getattr(self, "_shutdown_event", None)
        if event is None:
            event = asyncio.Event()
            self._shutdown_event = event
        return event

    async def _wait_for_retry_delay_or_shutdown(self, delay_seconds: float) -> bool:
        try:
            await asyncio.wait_for(self._get_shutdown_event().wait(), timeout=delay_seconds)
            return True
        except asyncio.TimeoutError:
            return False

    def _shutdown_retry_response(self, method: str, url: str) -> httpx.Response:
        return httpx.Response(
            503,
            request=httpx.Request(method, url),
            headers={"content-type": "application/json", "retry-after": "0"},
            json={
                "error": {
                    "type": "shutdown",
                    "message": "Proxy is shutting down; retry backoff cancelled.",
                }
            },
        )

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
        timeout: httpx.Timeout | float | None = None,
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
        from headroom.proxy.body_forwarding import select_outbound_body
        from headroom.proxy.helpers import log_outbound_request

        last_error = None
        reasons = list(mutation_reasons or [])
        outbound = select_outbound_body(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=body_mutated,
            mutation_reasons=reasons,
        )
        outbound_bytes, source = outbound.content, outbound.source
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
            dropped_mutation_reasons=outbound.dropped_mutation_reasons,
        )

        post_kwargs: dict = {"content": outbound_bytes, "headers": outbound_headers}
        if timeout is not None:
            post_kwargs["timeout"] = timeout

        for attempt in range(self.config.retry_max_attempts):
            try:
                if stream:
                    # For streaming, we return early - retry happens at higher level
                    return await self.http_client.post(  # type: ignore[union-attr]
                        url, **post_kwargs
                    )
                else:
                    response = await self.http_client.post(  # type: ignore[union-attr]
                        url, **post_kwargs
                    )

                    # Transient overloads (429 rate-limit, 529 overloaded):
                    # retry honoring Retry-After, but return verbatim once
                    # exhausted — a clean overload signal beats a synthesized 5xx
                    # (extends #1221 to 529, Anthropic's overloaded_error).
                    if response.status_code in RETRYABLE_OVERLOAD_STATUSES:
                        if (
                            not self.config.retry_enabled
                            or attempt >= self.config.retry_max_attempts - 1
                        ):
                            return response
                        delay_ms = retry_after_ms(
                            response, self.config.retry_max_delay_ms
                        ) or jitter_delay_ms(
                            self.config.retry_base_delay_ms,
                            self.config.retry_max_delay_ms,
                            attempt,
                        )
                        logger.warning(
                            f"Upstream {response.status_code} (attempt {attempt + 1}), "
                            f"retrying in {delay_ms:.0f}ms"
                        )
                        if await self._wait_for_retry_delay_or_shutdown(delay_ms / 1000):
                            logger.info(
                                "Shutdown interrupted retry backoff for %s %s",
                                method,
                                path_for_log or "<upstream-url>",
                            )
                            return self._shutdown_retry_response(method, url)
                        continue

                    # Don't retry other client errors (4xx)
                    if 400 <= response.status_code < 500:
                        return response

                    # Retry other server errors (5xx)
                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error: {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    return response

            # httpx.TransportError covers ConnectError, the timeout family, and —
            # crucially — the protocol errors (Local/RemoteProtocolError, e.g. an
            # HTTP/2 `StreamReset`) that a poisoned shared h2 connection raises on
            # every in-flight request. Retrying drops the bad connection and
            # re-sends on a fresh one instead of collapsing to a 502. (#1639)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last_error = e

                if not self.config.retry_enabled or attempt >= self.config.retry_max_attempts - 1:
                    # On exhaustion, preserve the upstream 5xx status (e.g. 503
                    # Service Unavailable, 500, 502, 504) so the client can apply
                    # its own retry/backoff. Collapsing every exhausted 5xx into a
                    # generic 502 hides the retryable signal and makes clients give
                    # up. The 429/529 overload statuses are already returned
                    # verbatim by the RETRYABLE_OVERLOAD_STATUSES branch above and
                    # never reach here. ConnectError/TimeoutException carry no
                    # response, so those still raise.
                    if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                        return e.response
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
                if await self._wait_for_retry_delay_or_shutdown(delay_with_jitter / 1000):
                    logger.info(
                        "Shutdown interrupted retry backoff for %s %s",
                        method,
                        path_for_log or "<upstream-url>",
                    )
                    return self._shutdown_retry_response(method, url)

        if last_error is None:
            raise RuntimeError(
                "retry loop exhausted with no error recorded; retry_max_attempts must be >= 1"
            )
        raise last_error


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


def _request_is_loopback(request: Request) -> bool:
    """Return True iff the caller is on loopback by *both* peer IP and Host header.

    Mirrors the two-gate check in :func:`loopback_guard.require_loopback`
    (loopback client IP + loopback ``Host`` header, the DNS-rebinding defence)
    but returns a bool instead of raising. Endpoints use it to vary their
    payload — serving sensitive sub-blocks (upstream URLs, per-request logs)
    only to loopback callers — rather than 404ing network callers that still
    have a legitimate use for the non-sensitive aggregate fields.
    """
    from headroom.proxy.loopback_guard import is_loopback_host, is_loopback_host_header

    client = getattr(request, "client", None)
    client_host = getattr(client, "host", None) if client is not None else None
    try:
        host_header = request.headers.get("host")
    except AttributeError:
        host_header = None

    # The Host-header gate is the DNS-rebinding defence and always applies.
    if not is_loopback_host_header(host_header):
        return False

    # Genuine loopback peer (native run, or curl inside the container).
    if is_loopback_host(client_host):
        return True

    # Containerized dashboards: when Headroom runs in a bridge-network
    # container, a browser on the host reaches it via the container's
    # gateway, so ``request.client.host`` is the gateway IP, not 127.0.0.1
    # — and the per-request logs / upstream URLs get stripped even though
    # the operator is local. Treat a peer inside an operator-configured
    # trusted-gateway CIDR as loopback-equivalent. Opt-in and empty by
    # default (HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS), so this is a no-op
    # unless the operator explicitly allow-lists their container gateway.
    from headroom.proxy.forwarded_headers import (
        load_trusted_gateway_cidrs,
        peer_is_trusted_gateway,
    )

    return peer_is_trusted_gateway(client_host, load_trusted_gateway_cidrs())


def _request_can_view_dashboard_metadata(
    request: Request,
    trusted_dashboard_client_cidrs: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
) -> bool:
    """Authorize sensitive ``/stats`` metadata without widening admin access."""
    if _request_is_loopback(request):
        return True

    from headroom.proxy.forwarded_headers import peer_is_trusted_gateway, resolve_client_ip
    from headroom.proxy.loopback_guard import is_ip_literal_host_header

    try:
        host_header = request.headers.get("host")
    except AttributeError:
        return False
    if not is_ip_literal_host_header(host_header):
        return False
    # is_ip_literal_host_header() rejects a missing Host, so host_header is a str here.
    assert host_header is not None

    # CIDR authorization makes this endpoint usable by a remote dashboard, but
    # it must not let an unrelated site read sensitive metadata through a
    # victim's browser. Native CLI clients usually send neither header, so
    # absence remains valid. If either browser provenance header is present,
    # require it to identify this exact scheme/host/port.
    if not _request_has_same_origin_or_no_provenance(request, host_header):
        return False

    return peer_is_trusted_gateway(
        resolve_client_ip(request),
        trusted_dashboard_client_cidrs,
    )


def _request_has_same_origin_or_no_provenance(request: Request, host_header: str) -> bool:
    """Accept no browser provenance, otherwise require same-origin headers."""

    from headroom.proxy.forwarded_headers import trusted_forwarded_headers

    forwarded_proto = trusted_forwarded_headers(request)["proto"]
    request_scheme = forwarded_proto or request.url.scheme
    expected_origin = _normalized_http_origin(f"{request_scheme}://{host_header}")
    if expected_origin is None:
        return False

    for header_name in ("origin", "referer"):
        header_value = request.headers.get(header_name)
        if header_value and _normalized_http_origin(header_value) != expected_origin:
            return False
    return True


def _normalized_http_origin(value: str) -> tuple[str, str, int] | None:
    """Return a normalized HTTP(S) origin tuple."""

    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if port is None:
        port = 80 if scheme == "http" else 443
    return scheme, parsed.hostname.lower(), port


#: Feedback-pattern keys built verbatim from agent query text. They are useful
#: in-process for compression decisions but must never reach an HTTP response —
#: same privacy contract the TOIN endpoints were brought under in #2926/#2927.
_FEEDBACK_QUERY_TEXT_KEYS = ("common_queries", "queried_fields")


def _feedback_stats_without_query_text(stats: dict[str, Any]) -> dict[str, Any]:
    """Return ``stats`` with per-tool query text stripped from ``tool_patterns``.

    Copies only the levels it edits; the aggregate counters are shared with the
    caller's dict, which is fine because they are scalars.
    """

    patterns = stats.get("tool_patterns")
    if not isinstance(patterns, dict):
        return stats

    scrubbed: dict[str, Any] = {}
    for name, pattern in patterns.items():
        if isinstance(pattern, dict):
            scrubbed[name] = {
                key: value for key, value in pattern.items() if key not in _FEEDBACK_QUERY_TEXT_KEYS
            }
        else:
            scrubbed[name] = pattern

    return {**stats, "tool_patterns": scrubbed}


_is_known_websocket_callback_failure = is_known_websocket_callback_failure


_tool_schema_saved_from_tags = tool_schema_saved_from_tags


def read_proxy_token(headers: Mapping[str, str]) -> str | None:
    """Return the caller-supplied proxy token from request headers, or ``None``.

    Shared by the HTTP security gate and :class:`WebSocketAuthMiddleware` so the
    two transports cannot drift on what counts as a credential. Header names are
    expected to be lowercase (Starlette's ``Headers`` is case-insensitive; the
    WebSocket middleware lowercases the raw ASGI pairs itself).
    """
    raw = headers.get("x-headroom-proxy-token")
    if raw is not None:
        return str(raw) or None

    # A provider credential can legitimately occupy Authorization (for example,
    # an OAuth/subscription client). Prefer the explicit proxy header whenever
    # it is present so that the upstream credential is not mistaken for the
    # proxy credential.
    auth = str(headers.get("authorization") or "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


class WebSocketAuthMiddleware:
    """Enforce ``HEADROOM_PROXY_TOKEN`` on WebSocket handshakes.

    The HTTP security gate is registered with ``@app.middleware("http")``, which
    is a Starlette ``BaseHTTPMiddleware`` — and that class hands any scope whose
    type is not ``http`` straight to the wrapped app. WebSocket connections
    therefore never reached the gate, so every ``app.websocket(...)`` route
    accepted unauthenticated callers even with a token configured. Those routes
    are not incidental: ``/v1/responses`` and ``/v1/live`` relay to the upstream
    provider using the operator's own credentials, and they are registered
    unconditionally. ``/v1/responses`` exists on both transports, so the POST was
    authenticated while the upgrade on the very same path was not.

    Written as a raw ASGI middleware rather than folded into the gate because
    that is the only layer that sees the ``websocket`` scope at all.

    Loopback callers are exempt, matching the HTTP gate exactly (same trust
    boundary as the admin/debug routes). Credentials are read from headers only:
    the handshake carries them fine for the programmatic clients these routes
    serve, and accepting a token from the query string would put it in access
    logs and browser history.
    """

    def __init__(self, app: Any, *, proxy_token: str | None = None) -> None:
        self.app = app
        self.proxy_token = proxy_token
        # Pre-encoded for constant-time comparison, mirroring the HTTP gate:
        # compare_digest on str raises TypeError for non-ASCII input, which
        # would turn a rejected handshake into a 500.
        self.token_bytes = proxy_token.encode("utf-8") if proxy_token else b""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "websocket" or not self.proxy_token:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_host = client[0] if client else None
        if is_loopback_host(client_host):
            await self.app(scope, receive, send)
            return

        # Starlette's own Headers rather than a hand-built dict: on a repeated
        # header it returns the FIRST occurrence, which is what the HTTP gate
        # sees. Building a dict here instead took the LAST one, so the two
        # transports disagreed about which `Authorization` counted — exactly the
        # drift the shared reader below exists to prevent.
        from starlette.datastructures import Headers

        provided = read_proxy_token(Headers(scope=scope))
        if provided is not None and hmac.compare_digest(
            provided.encode("utf-8", "replace"), self.token_bytes
        ):
            await self.app(scope, receive, send)
            return

        logger.warning(
            "event=proxy_auth_rejected transport=websocket path=%s client=%s reason=%s",
            scope.get("path"),
            client_host,
            "missing_token" if provided is None else "bad_token",
        )
        # Receive the handshake before refusing it: ASGI servers send
        # ``websocket.connect`` and wait for the application to answer, and
        # answering with ``websocket.close`` *before* an accept is what refuses
        # the upgrade on the wire instead of accepting and then dropping it.
        message = await receive()
        if message["type"] == "websocket.connect":
            await send({"type": "websocket.close", "code": 1008})


class WebSocketProjectPrefixMiddleware:
    """Normalize project-prefixed WebSocket paths before route matching."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket":
            prefix_project = strip_project_path_prefix(scope)
            headers = {
                name.decode("latin-1"): value.decode("latin-1") for name, value in scope["headers"]
            }
            set_current_project(classify_project(headers) or prefix_project)
        await self.app(scope, receive, send)


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required. Install: pip install fastapi uvicorn httpx")

    from headroom.proxy.forwarded_headers import load_trusted_dashboard_client_cidrs

    # Parse once at startup so invalid operator configuration fails loudly.
    trusted_dashboard_client_cidrs = load_trusted_dashboard_client_cidrs()

    from contextlib import asynccontextmanager

    # Always-on file logging to ~/.headroom/logs/ for `headroom perf` analysis.
    # Installed here (not at module import) so importing headroom.proxy.server
    # in tests or library contexts does not silently attach a RotatingFileHandler
    # to the user's live proxy.log.
    _setup_file_logging()

    config = config or ProxyConfig()

    # Defensive re-apply of file-backed settings for embedded/non-CLI callers
    # that construct the app without going through the `headroom` CLI entrypoint
    # (which already applies them before Click parsing). setdefault keeps
    # explicit env exports authoritative; fail-open so it never blocks startup.
    try:
        from headroom import settings_store

        settings_store.apply_to_environ(settings_store.load())
    except Exception:  # noqa: BLE001 — settings load must never break startup
        pass

    # Air-gap master switch. Propagate config.offline to the env so the
    # env-based egress predicates (telemetry, update check, license) all honor
    # it, force HF/transformers offline before any model code loads, and
    # announce that every outbound path is disabled.
    if config.offline:
        os.environ.setdefault("HEADROOM_OFFLINE", "1")
    if is_offline():
        apply_offline_env()
        logger.warning(
            "event=proxy_offline_mode air-gap active — all outbound egress disabled "
            "(telemetry, update check, license reporter, HuggingFace downloads)"
        )

    proxy = HeadroomProxy(config)

    # cc-switch reconciler (opt-in: HEADROOM_CC_SWITCH_RECONCILE=1).
    # Keeps Headroom in the request path while cc-switch overwrites
    # ~/.claude/settings.json on every provider switch. See
    # headroom/proxy/cc_switch_reconciler.py for the full rationale.
    from headroom.proxy.cc_switch_reconciler import (
        CCSwitchReconciler,
        reconciler_enabled,
    )

    _cc_reconciler: CCSwitchReconciler | None = None
    if reconciler_enabled():
        _cc_proxy_port = config.port if hasattr(config, "port") else 8787

        def _set_anthropic_upstream(url: str) -> None:
            from headroom.providers.registry import _normalize_api_url

            HeadroomProxy.ANTHROPIC_API_URL = _normalize_api_url(
                url, default=DEFAULT_ANTHROPIC_API_URL
            )

        _cc_reconciler = CCSwitchReconciler(
            proxy_url=f"http://127.0.0.1:{_cc_proxy_port}",
            default_upstream=DEFAULT_ANTHROPIC_API_URL,
            set_upstream=_set_anthropic_upstream,
        )

    # Single-worker-owner lock. With uvicorn workers > 1, each worker runs the
    # lifespan independently. A file lock elects ONE owner worker so that
    # single-instance background tasks (currently the cc-switch reconciler) run
    # once across all workers instead of N times.
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
        # Arm the C-level liveness watchdog before anything that can touch
        # native code: a GIL seized during startup freezes the process just as
        # thoroughly as one seized while serving (#3178). Per worker, because
        # the timer and the GIL are both process-local.
        from headroom.proxy.hard_watchdog import start_hard_watchdog

        start_hard_watchdog()

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
        app.state.periodic_toin_stats_task = None
        app.state.periodic_malloc_trim_task = None

        try:
            try:
                previous_handler = _install_loop_exception_handler()
                # Startup
                await proxy.startup()
                if config.periodic_toin_stats_enabled:
                    app.state.periodic_toin_stats_task = asyncio.create_task(
                        _log_toin_stats_periodically()
                    )
                # Per-worker on purpose: allocator state is per-process, so
                # every worker must trim its own zones (no beacon-owner gate).
                if config.periodic_malloc_trim_enabled:
                    app.state.periodic_malloc_trim_task = asyncio.create_task(
                        trim_periodically(config.malloc_trim_interval_seconds)
                    )
                if proxy.usage_reporter:
                    await proxy.usage_reporter.start(proxy)
                if proxy.traffic_learner:
                    await proxy.traffic_learner.start()
                if proxy._background_compression_enabled:
                    await proxy._background_compressor.start()

                # Elect the single owner worker (first worker wins the lock).
                _beacon_is_owner[0] = _try_acquire_beacon_lock()

                # Only the owner worker runs the reconciler. With uvicorn
                # workers > 1 each worker runs this lifespan; without this guard
                # every worker would watch + rewrite settings.json concurrently
                # and each process would hold its own HeadroomProxy.ANTHROPIC_API_URL,
                # so workers could disagree on the upstream.
                if _cc_reconciler is not None and _beacon_is_owner[0]:
                    await _cc_reconciler.start()

                app.state.ready = True
                yield
            except Exception as exc:
                app.state.startup_error = str(exc)
                raise
        finally:
            loop: asyncio.AbstractEventLoop | None
            previous: LoopExceptionHandler | None
            try:
                loop = asyncio.get_running_loop()
                previous = previous_handler
            except RuntimeError:
                loop = None
                previous = app.state.previous_loop_exception_handler
            if loop is not None:
                loop.set_exception_handler(previous)

            app.state.ready = False
            logger.info("event=proxy_shutdown reason=signal pid=%d", os.getpid())

            async def _timed(coro: Any, *, label: str, timeout: float) -> None:
                try:
                    await asyncio.wait_for(coro, timeout=timeout)
                except Exception as exc:
                    logger.warning(
                        "event=shutdown_step_timeout_or_error label=%s timeout=%.1fs exc=%r",
                        label,
                        timeout,
                        exc,
                    )

            periodic_toin_stats_task = app.state.periodic_toin_stats_task
            if periodic_toin_stats_task is not None:
                periodic_toin_stats_task.cancel()
                await _timed(
                    asyncio.gather(periodic_toin_stats_task, return_exceptions=True),
                    label="periodic_toin_stats.stop",
                    timeout=3.0,
                )
                app.state.periodic_toin_stats_task = None

            periodic_malloc_trim_task = app.state.periodic_malloc_trim_task
            if periodic_malloc_trim_task is not None:
                periodic_malloc_trim_task.cancel()
                await _timed(
                    asyncio.gather(periodic_malloc_trim_task, return_exceptions=True),
                    label="periodic_malloc_trim.stop",
                    timeout=3.0,
                )
                app.state.periodic_malloc_trim_task = None

            if _cc_reconciler is not None:
                await _timed(_cc_reconciler.stop(), label="cc_reconciler.stop", timeout=3.0)
            if _beacon_is_owner[0]:
                _release_beacon_lock()
            if proxy.usage_reporter:
                await _timed(proxy.usage_reporter.stop(), label="usage_reporter.stop", timeout=3.0)
            if proxy.traffic_learner:
                await _timed(
                    proxy.traffic_learner.stop(), label="traffic_learner.stop", timeout=3.0
                )
            if proxy._background_compression_enabled:
                await _timed(
                    proxy._background_compressor.stop(),
                    label="background_compressor.stop",
                    timeout=3.0,
                )
            proxy._background_compression_executor.shutdown(wait=False)
            if proxy.code_graph_watcher:
                proxy.code_graph_watcher.stop()
            await _timed(proxy.shutdown(), label="proxy.shutdown", timeout=5.0)
            shutdown_headroom_tracing()
            shutdown_otel_metrics()

    app = FastAPI(
        title="Headroom Proxy",
        description="Production-ready LLM optimization proxy",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(WebSocketProjectPrefixMiddleware)
    loop_health_state: LoopHealthState = {
        "status": "healthy",
        "known_failures": 0,
        "last_known_failure": None,
    }
    app.state.proxy = proxy
    app.state.started_at = None
    app.state.ready = False
    app.state.startup_error = None
    app.state.loop_callback_health = loop_health_state
    app.state.loop_exception_handler = None
    app.state.previous_loop_exception_handler = None
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
        optional: bool = False,
        **details: Any,
    ) -> dict[str, Any]:
        if not enabled:
            status = "disabled"
        elif ready:
            status = "healthy"
        elif optional:
            # Optional/non-gating components report "degraded" rather than
            # "unhealthy" so the top-level status: "healthy" / ready: true
            # payload is not contradicted by a component-level failure label.
            status = "degraded"
        else:
            status = "unhealthy"
        result: dict[str, Any] = {
            "enabled": enabled,
            "ready": (ready if enabled else True),
            "status": status,
        }
        if optional:
            result["optional"] = True
        result.update(details)
        return result

    def _kompress_health_routers() -> list[ContentRouter]:
        routers: list[ContentRouter] = []
        for pipeline in (proxy.anthropic_pipeline, proxy.openai_pipeline):
            for transform in getattr(pipeline, "transforms", ()):
                if (
                    isinstance(transform, ContentRouter)
                    and transform.config.enable_kompress
                    and all(transform is not item for item in routers)
                ):
                    routers.append(transform)
        return routers

    def _reconcile_kompress_health() -> bool:
        routers = _kompress_health_routers()
        if not routers:
            return False

        compressors: list[Any] = []
        for router in routers:
            for name in ("_kompress", "_kompress_remote"):
                compressor = getattr(router, name, None)
                if compressor is not None and all(compressor is not item for item in compressors):
                    compressors.append(compressor)

        for compressor in compressors:
            try:
                if not compressor.is_ready():
                    continue
                backend = compressor.ready_backend()
                if not backend:
                    continue
            except Exception:
                continue
            if proxy.warmup.kompress.status == "loaded":
                return True
            proxy.warmup.kompress.mark_loaded(
                handle=compressor, backend=backend, source_status="runtime"
            )
            return True

        try:
            from headroom.transforms.kompress_compressor import HF_MODEL_ID, _kompress_cache
        except ImportError:
            return True

        cached = _kompress_cache.get(HF_MODEL_ID)
        if cached is not None:
            try:
                model, _tokenizer, backend = cached
            except (TypeError, ValueError):
                pass
            else:
                if backend and proxy.warmup.kompress.status != "loaded":
                    proxy.warmup.kompress.mark_loaded(
                        handle=model, backend=backend, source_status="runtime"
                    )
        return True

    def _health_checks() -> dict[str, dict[str, Any]]:
        kompress_enabled = _reconcile_kompress_health()
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
            "upstream": _component_health(
                enabled=os.environ.get("HEADROOM_SKIP_UPSTREAM_CHECK", "").strip() != "1",
                ready=bool(_upstream_check_cache["ok"]),
                url=_upstream_check_cache["url"],
                error=_upstream_check_cache["error"],
            ),
            "kompress": _component_health(
                enabled=kompress_enabled,
                ready=proxy.warmup.kompress.status == "loaded",
                optional=True,
                backend=proxy.warmup.kompress.info.get("backend", None),
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
            _comp_queued = proxy._compression_queued
            _comp_queued_max = proxy._compression_queued_max
            _comp_queue_timeouts = proxy._compression_queue_timeouts
            _comp_queue_wait_total = proxy._compression_queue_wait_seconds_total
            _comp_queue_wait_max = proxy._compression_queue_wait_seconds_max
            _comp_in_flight = proxy._compression_in_flight
            _comp_in_flight_max = proxy._compression_in_flight_max
            _comp_run_total = proxy._compression_run_seconds_total
            _comp_run_max = proxy._compression_run_seconds_max
            _comp_leaked = proxy._compression_leaked_threads
            _comp_timed_out_in_flight = proxy._compression_timed_out_in_flight
            _comp_timed_out_in_flight_max = proxy._compression_timed_out_in_flight_max
            _comp_quarantine_activations = proxy._compression_quarantine_activations
            _comp_quarantine_skips = proxy._compression_quarantine_skips
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
                "queued": _comp_queued,
                "queued_max": _comp_queued_max,
                "queue_timeouts_total": _comp_queue_timeouts,
                "queue_wait_seconds_total": _comp_queue_wait_total,
                "queue_wait_seconds_max": _comp_queue_wait_max,
                "running": _comp_in_flight,
                "in_flight": _comp_in_flight,
                "in_flight_max": _comp_in_flight_max,
                "run_seconds_total": _comp_run_total,
                "run_seconds_max": _comp_run_max,
                "leaked_threads_total": _comp_leaked,
                "quarantine_active": _comp_timed_out_in_flight > 0,
                "timed_out_workers": _comp_timed_out_in_flight,
                "timed_out_workers_max": _comp_timed_out_in_flight_max,
                "quarantine_activations_total": _comp_quarantine_activations,
                "quarantine_skips_total": _comp_quarantine_skips,
                "source": ("auto" if config.compression_max_workers is None else "explicit"),
            },
            "websocket_sessions": {
                "active_sessions": ws_active_sessions,
                "active_relay_tasks": ws_active_relay_tasks,
            },
        }

    def _loop_callback_payload() -> LoopHealthState:
        return {
            "status": loop_health_state["status"],
            "known_failures": loop_health_state["known_failures"],
            "last_known_failure": loop_health_state["last_known_failure"],
        }

    def _install_loop_exception_handler() -> LoopExceptionHandler | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None

        previous_handler = loop.get_exception_handler()

        def _loop_exception_handler(
            _loop: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            if _is_known_websocket_callback_failure(context):
                loop_health_state["status"] = "unhealthy"
                loop_health_state["known_failures"] += 1
                loop_health_state["last_known_failure"] = {
                    "message": context.get("message"),
                    "exception": str(context.get("exception"))
                    if context.get("exception")
                    else None,
                }
                return

            delegate_handler = app.state.previous_loop_exception_handler
            if delegate_handler is not None:
                delegate_handler(_loop, context)
                return
            _loop.default_exception_handler(context)

        loop.set_exception_handler(_loop_exception_handler)
        app.state.loop_exception_handler = _loop_exception_handler
        app.state.previous_loop_exception_handler = previous_handler
        return previous_handler

    def _health_payload(*, include_config: bool) -> dict[str, Any]:
        checks = _health_checks()
        # Kompress is an optional soft component: model downloads lazily on
        # first use, so "not ready" (cold cache) must not degrade overall health.
        ready = all(check["ready"] for name, check in checks.items() if name != "kompress")
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
            profile_kwargs = proxy_pipeline_kwargs(config)
            effective_target_ratio = cast(
                float | None,
                profile_kwargs.get("target_ratio", config.target_ratio),
            )
            payload["config"] = {
                "backend": config.backend,
                "optimize": config.optimize,
                "cache": config.cache_enabled,
                "rate_limit": config.rate_limit_enabled,
                "disable_kompress": config.disable_kompress,
                "disable_kompress_fallback": config.disable_kompress_fallback,
                "disable_kompress_anthropic": config.disable_kompress_anthropic,
                "disable_kompress_openai": config.disable_kompress_openai,
                "memory": config.memory_enabled,
                "learn": config.traffic_learning_enabled,
                "code_graph": config.code_graph_watcher,
                "anthropic_api_url": config.anthropic_api_url,
                "openai_api_url": config.openai_api_url,
                "gemini_api_url": config.gemini_api_url,
                "cloudcode_api_url": config.cloudcode_api_url,
                "vertex_api_url": config.vertex_api_url,
                "savings_profile": config.savings_profile,
                "target_ratio": effective_target_ratio,
                "target_savings_percent": (
                    round(max(0.0, min(1.0, 1.0 - float(effective_target_ratio))) * 100, 1)
                    if effective_target_ratio is not None
                    else None
                ),
                "compress_user_messages": bool(
                    profile_kwargs.get("compress_user_messages", config.compress_user_messages)
                ),
                "compress_system_messages": bool(
                    profile_kwargs.get(
                        "compress_system_messages",
                        config.compress_system_messages,
                    )
                ),
                "protect_recent": profile_kwargs.get(
                    "read_protection_window",
                    config.protect_recent,
                ),
                "protect_analysis_context": profile_kwargs.get(
                    "protect_analysis_context",
                    config.protect_analysis_context,
                ),
                "min_tokens_to_crush": profile_kwargs.get(
                    "min_tokens_to_compress",
                    config.min_tokens_to_crush,
                ),
                "max_items_after_crush": profile_kwargs.get(
                    "max_items_after_crush",
                    config.max_items_after_crush,
                ),
                "smart_crusher_with_compaction": profile_kwargs.get(
                    "smart_crusher_with_compaction",
                    config.smart_crusher_with_compaction,
                ),
                "force_kompress": bool(profile_kwargs.get("force_kompress", False)),
                "accuracy_guard": config.accuracy_guard,
                # Live (per-request) env knobs the proxy reads after startup.
                # Surfaced so `headroom wrap` can see what a reused proxy is
                # actually using and hot-sync it via /admin/runtime-env.
                "runtime_env": runtime_env.effective_runtime_env(),
                "pid": os.getpid(),
            }
        return payload

    # ---------------------------------------------------------------------------
    # Upstream connectivity check — cached to avoid hammering the upstream on
    # every /readyz poll.  Set HEADROOM_SKIP_UPSTREAM_CHECK=1 to opt out (e.g.
    # in air-gapped or test environments where the upstream isn't reachable at
    # startup time).
    # ---------------------------------------------------------------------------

    _UPSTREAM_CHECK_TTL = 30.0  # seconds
    _upstream_check_cache: dict[str, Any] = {
        "expires_at": 0.0,
        "ok": True,
        "error": None,
        "url": None,
    }
    _upstream_check_lock = asyncio.Lock()

    def _upstream_target_url() -> str:
        """Return the primary upstream base URL to probe."""
        # Use the resolved API target from the provider runtime so we respect
        # any overrides set by ProxyConfig.anthropic_api_url / env vars.
        return proxy.provider_runtime.api_targets.anthropic

    async def _check_upstream() -> None:
        """Probe the upstream API endpoint and update the cached result.

        Uses a HEAD request with a 5-second timeout — just enough to verify
        TLS + TCP reachability without triggering an inference call.
        """
        if os.environ.get("HEADROOM_SKIP_UPSTREAM_CHECK", "").strip() == "1":
            # Opt-out: treat upstream as always reachable.
            _upstream_check_cache["ok"] = True
            _upstream_check_cache["error"] = None
            _upstream_check_cache["expires_at"] = time.monotonic() + _UPSTREAM_CHECK_TTL
            return

        now = time.monotonic()
        # Fast-path: return if the cached result is still fresh (no lock needed
        # for a simple float comparison — worst case we re-check twice).
        if now < _upstream_check_cache["expires_at"]:
            return

        async with _upstream_check_lock:
            # Re-check inside the lock to handle concurrent waiters.
            if time.monotonic() < _upstream_check_cache["expires_at"]:
                return
            url = _upstream_target_url()
            _upstream_check_cache["url"] = url
            client = proxy.http_client
            if client is None:
                _upstream_check_cache["ok"] = False
                _upstream_check_cache["error"] = "proxy client not initialised"
                _upstream_check_cache["expires_at"] = time.monotonic() + _UPSTREAM_CHECK_TTL
                return
            try:
                resp = await client.head(url, timeout=5.0)
                # Any HTTP response (even 4xx/5xx) means TLS+TCP worked.
                _ = resp.status_code
                _upstream_check_cache["ok"] = True
                _upstream_check_cache["error"] = None
            except Exception as exc:  # noqa: BLE001
                _upstream_check_cache["ok"] = False
                _upstream_check_cache["error"] = str(exc)
            _upstream_check_cache["expires_at"] = time.monotonic() + _UPSTREAM_CHECK_TTL

    # CORS: scoped to localhost by default. The old wildcard origin combined
    # with allow_credentials=True let any web page the user had open read the
    # proxy's content endpoints (e.g. /v1/retrieve returns raw, uncompressed
    # tool outputs) via a cross-origin fetch to 127.0.0.1 (CWE-346).
    #
    # The default matches any loopback origin on any port via a regex, so it
    # works regardless of the --port the proxy was started on without the app
    # needing to know its own bound port (the port lives in the CLI/uvicorn
    # layer, not in ProxyConfig). Set HEADROOM_CORS_ORIGINS (comma-separated)
    # to pin an explicit allowlist for Docker or remote-dashboard deployments;
    # "*" restores the old wildcard behaviour if the operator accepts the risk.
    _default_loopback_origin_regex = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"
    _cors_origins_env = os.environ.get("HEADROOM_CORS_ORIGINS", "").strip()
    if _cors_origins_env:
        _cors_allow_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
        _cors_allow_origin_regex: str | None = None
    else:
        _cors_allow_origins = []
        _cors_allow_origin_regex = _default_loopback_origin_regex
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_allow_origins,
        allow_origin_regex=_cors_allow_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # X-Headroom-Stack: SDK adapters (TS openai/anthropic/etc.) tag their
    # requests so telemetry can segment by integration surface. Registered
    # before extension middleware so any extension-level auth/guards run
    # outermost and we don't count requests they reject.
    @app.middleware("http")
    async def _record_headroom_stack(request, call_next):
        started = time.perf_counter()
        inbound_id = f"inbound-{time.time_ns()}"
        # Project attribution: an explicit X-Headroom-Project header wins
        # (claude/codex wraps); otherwise a /p/<name> base-URL prefix (aider,
        # Copilot BYOK, Cursor — clients that cannot send custom headers).
        # The prefix strip mutates the scope, so it must happen before
        # request.url is first accessed (Starlette caches the URL).
        prefix_project = strip_project_path_prefix(request.scope)
        path = request.url.path
        method = request.method
        query = request.url.query
        headers = dict(request.headers.items())
        set_current_project(classify_project(headers) or prefix_project)
        # Path-based Codex identification: stamp X-Client: codex on the
        # Responses endpoint for callers that don't otherwise classify (e.g.
        # Codex Desktop, whose User-Agent isn't a known codex UA). Without it
        # the backend refuses oversized
        # requests with a 413 on a compression timeout, which Codex treats as a
        # hard connection failure. Mutating scope["headers"] before call_next
        # makes every downstream classify_client(headers) read "codex".
        if should_stamp_codex_client(path, headers):
            request.scope["headers"].append((b"x-client", b"codex"))
        client = getattr(request, "client", None)
        client_addr = ""
        if client is not None:
            client_host = getattr(client, "host", None)
            client_port = getattr(client, "port", None)
            client_addr = f"{client_host}:{client_port}" if client_port else str(client_host)
        try:
            proxy.metrics.record_inbound_request(method=method, path=path)
        except Exception:
            logger.debug("record_inbound_request failed", exc_info=True)
        try:
            from headroom.proxy.helpers import redact_for_wire_debug

            safe_headers = redact_for_wire_debug(headers)
        except Exception:
            safe_headers = {"redaction_error": True}
        logger.info(
            "event=proxy_inbound_request id=%s method=%s path=%s query=%s client=%s "
            "content_length=%s headers=%s",
            inbound_id,
            method,
            path,
            query,
            client_addr,
            request.headers.get("content-length", ""),
            json.dumps(safe_headers, ensure_ascii=False, default=str),
        )
        if request.url.path.startswith("/v1/"):
            stack = request.headers.get("x-headroom-stack")
            if stack:
                try:
                    proxy.metrics.record_stack(stack)
                except Exception:
                    logger.debug("record_stack failed", exc_info=True)
        try:
            response = await call_next(request)
        except asyncio.CancelledError:
            try:
                proxy.metrics.record_inbound_aborted(reason="cancelled")
            except Exception:
                logger.debug("record_inbound_aborted failed", exc_info=True)
            logger.info(
                "event=proxy_inbound_request_aborted id=%s method=%s path=%s reason=cancelled "
                "duration_ms=%.2f",
                inbound_id,
                method,
                path,
                (time.perf_counter() - started) * 1000.0,
            )
            raise
        except Exception as exc:
            try:
                proxy.metrics.record_inbound_aborted(reason=type(exc).__name__)
            except Exception:
                logger.debug("record_inbound_aborted failed", exc_info=True)
            logger.error(
                "event=proxy_inbound_request_aborted id=%s method=%s path=%s reason=%s "
                "duration_ms=%.2f",
                inbound_id,
                method,
                path,
                type(exc).__name__,
                (time.perf_counter() - started) * 1000.0,
                exc_info=True,
            )
            raise
        try:
            proxy.metrics.record_inbound_response(status_code=response.status_code)
        except Exception:
            logger.debug("record_inbound_response failed", exc_info=True)
        logger.info(
            "event=proxy_inbound_response id=%s method=%s path=%s status=%s duration_ms=%.2f",
            inbound_id,
            method,
            path,
            response.status_code,
            (time.perf_counter() - started) * 1000.0,
        )
        return response

    # ── Security gate (outermost of the HTTP middlewares) ─────────────────
    # Three concerns, kept together because they all wrap every inbound HTTP
    # request: optional inbound auth on the data plane, response security
    # headers, and an audit trail for state-mutating admin endpoints.
    # WebSocket handshakes are covered separately — see the
    # WebSocketAuthMiddleware registration just below this block.
    _proxy_token = config.proxy_token or os.environ.get("HEADROOM_PROXY_TOKEN") or None
    # Pre-encode once for constant-time comparison (compare_digest on str raises
    # TypeError for non-ASCII input, which would turn a 401 into a 500).
    _proxy_token_bytes = _proxy_token.encode("utf-8") if _proxy_token else b""
    # Health/readiness probes must stay reachable without a token so
    # orchestrators can check a container that binds non-loopback.
    _AUTH_EXEMPT_PATHS = frozenset({"/health", "/healthz", "/livez", "/readyz"})

    # Loud warning when a non-loopback bind has no token configured: that is the
    # exact shape (e.g. the Docker 0.0.0.0 image) that exposes unauthenticated
    # /v1/* routes to the surrounding network.
    if not _proxy_token and not is_loopback_host(getattr(config, "host", None)):
        logger.warning(
            "event=proxy_open_bind host=%s — proxy is bound to a non-loopback "
            "interface with no HEADROOM_PROXY_TOKEN set; the /v1/* data-plane "
            "routes are reachable WITHOUT authentication. Set HEADROOM_PROXY_TOKEN "
            "to require a bearer token from non-loopback callers.",
            getattr(config, "host", None),
        )

    def _apply_security_headers(response) -> None:
        # setdefault: never clobber a header an upstream/handler already set.
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )

    # Delegates so the HTTP gate and WebSocketAuthMiddleware read a credential
    # by exactly one rule; they guard the same token on two transports.
    _extract_proxy_token = read_proxy_token

    @app.middleware("http")
    async def _security_gate(request, call_next):
        # 1) Optional inbound auth. When a token is configured, require it on
        #    non-loopback requests; loopback callers and health probes are
        #    exempt. Loopback is the same trust boundary the admin/debug
        #    endpoints already use (see loopback_guard).
        if _proxy_token:
            path = request.url.path
            client = getattr(request, "client", None)
            client_host = getattr(client, "host", None) if client is not None else None
            if path not in _AUTH_EXEMPT_PATHS and not is_loopback_host(client_host):
                provided = _extract_proxy_token(request.headers)
                if provided is None or not hmac.compare_digest(
                    provided.encode("utf-8", "replace"), _proxy_token_bytes
                ):
                    logger.warning(
                        "event=proxy_auth_rejected path=%s client=%s reason=%s",
                        path,
                        client_host,
                        "missing_token" if provided is None else "bad_token",
                    )
                    rejection = JSONResponse(status_code=401, content={"error": "unauthorized"})
                    _apply_security_headers(rejection)
                    return rejection

        response = await call_next(request)
        _apply_security_headers(response)

        # 2) Audit trail for admin / state-mutating endpoints.
        try:
            if is_auditable_path(request.url.path):
                record_admin_action(
                    request=request,
                    action="admin_request",
                    status_code=response.status_code,
                )
        except Exception:
            logger.debug("admin audit emission failed", exc_info=True)
        return response

    # The gate above is http-only (BaseHTTPMiddleware ignores every other
    # scope), so the same token rule is applied to the `websocket` scope here.
    # Added after it, which makes it the outermost layer — an unauthenticated
    # handshake is refused before any project-prefix or routing work happens.
    app.add_middleware(WebSocketAuthMiddleware, proxy_token=_proxy_token)

    # Third-party proxy extensions (Enterprise, custom plugins). Discovered via
    # the `headroom.proxy_extension` entry-point group, but **opt-in only**:
    # only names listed in config.proxy_extensions (CLI: --proxy-extension,
    # env: HEADROOM_PROXY_EXTENSIONS) actually get installed. Discovery alone
    # never runs third-party code. An extension that raises from its install()
    # disables *itself*: `install_all` logs it, prints a SKIPPED line to stderr
    # and the proxy keeps serving without that extension (extensions.py:169-183).
    # One bad plugin must not brick the proxy. Note the consequence: a plugin
    # whose licence gate raises is absent, not fatal — the feature fails closed,
    # the process does not.
    from headroom.proxy.extensions import install_all as _install_extensions

    _install_extensions(app, config, enabled=getattr(config, "proxy_extensions", None))

    # Health & Metrics
    @app.get("/livez")
    async def livez():
        callback_state = _loop_callback_payload()
        healthy = callback_state["status"] == "healthy"
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "service": "headroom-proxy",
                "status": "healthy" if healthy else "unhealthy",
                "alive": healthy,
                "version": __version__,
                "timestamp": _iso_utc_now(),
                "uptime_seconds": _uptime_seconds(),
                "loop_health": callback_state,
            },
        )

    @app.get("/readyz")
    async def readyz():
        await _check_upstream()
        payload = _health_payload(include_config=False)
        return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)

    @app.get("/health")
    async def health(request: Request):
        await _check_upstream()
        # /health echoes upstream API URLs + backend config (the `config`
        # block). That is operational detail an external scanner should not
        # see, so include it only for loopback callers; network callers get the
        # same body as /readyz (status + checks, no config). /livez and /readyz
        # remain the unauthenticated probes for orchestration health.
        payload = _health_payload(include_config=_request_is_loopback(request))
        return JSONResponse(status_code=200, content=payload)

    # Loopback-only debug introspection (Unit 5). A remote IP gets 404 —
    # debug endpoints are invisible to external scanners.
    from headroom.proxy.debug_introspection import (
        collect_tasks as _collect_tasks,
    )
    from headroom.proxy.loopback_guard import require_loopback as _require_loopback
    from headroom.proxy.loopback_guard import require_same_origin as _require_same_origin

    def _require_loopback_or_trusted_dashboard_client(request: Request) -> None:
        """Allow loopback callers, or gateway-forwarded dashboard clients.

        Mirrors the trust chain already used by /stats and /stats-lifetime
        (see _request_can_view_dashboard_metadata) so the settings UI works
        the same way behind a reverse-proxy/gateway (issue #2466).
        """
        if not _request_can_view_dashboard_metadata(request, trusted_dashboard_client_cidrs):
            raise HTTPException(status_code=404)

    def _require_same_origin_or_trusted_dashboard_client(request: Request) -> None:
        """Same-origin CSRF guard for settings writes, trusted-dashboard aware.

        ``require_same_origin`` only accepts an ``Origin`` that itself names a
        loopback host, so a browser POST from a trusted-gateway dashboard
        client was rejected even though the paired GET routes allow that same
        caller (issue #2466). For non-loopback callers, accept an ``Origin``
        that matches this request's own Host header, provided the caller is
        already an IP-literal-Host, CIDR-trusted dashboard client. Loopback
        callers keep the stricter loopback-only origin check unchanged.
        """
        if not _request_is_loopback(request):
            origin = request.headers.get("origin")
            host_header = request.headers.get("host")
            if (
                origin
                and origin != "null"
                and host_header
                and _request_has_same_origin_or_no_provenance(request, host_header)
                and _request_can_view_dashboard_metadata(request, trusted_dashboard_client_cidrs)
            ):
                return
        _require_same_origin(request)

    @app.get("/admin/upstream", dependencies=[Depends(_require_loopback)])
    async def get_upstream():
        """Current Anthropic upstream + cc-switch reconciler state (loopback-only).

        Read-only. The upstream is mutated only via the in-process cc-switch
        reconciler (driven by ~/.claude/settings.json) — there is deliberately
        no HTTP write route, so a local process cannot redirect credential-
        bearing traffic to an arbitrary URL through this surface.
        """
        return {
            "anthropic": HeadroomProxy.ANTHROPIC_API_URL,
            "cc_switch_reconcile": _cc_reconciler is not None,
            "captured_upstream": getattr(_cc_reconciler, "current_upstream", None),
        }

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
        # Promote a deferred Kompress slot from live runtime state before we
        # serialize. Without this the registry keeps reporting the startup
        # snapshot (``status: null`` / ``source_status: deferred``) forever,
        # even while the model is loaded and compressing, unless somebody
        # happens to hit /health or /readyz first. Same read-only
        # reconciliation those endpoints run (issue #2624).
        _reconcile_kompress_health()
        warmup_registry = getattr(proxy, "warmup", None)
        payload = warmup_registry.to_dict() if warmup_registry is not None else {}
        payload["runtime"] = _runtime_payload()
        return JSONResponse(status_code=200, content=payload)

    @app.post(
        "/admin/runtime-env",
        dependencies=[Depends(_require_loopback), Depends(_require_same_origin)],
    )
    async def admin_runtime_env(request: Request):
        """Hot-reload live env knobs (the output-shaper family, the ast-grep
        read threshold) without restarting the proxy.

        Live knobs are read from the proxy's *process* environment, so a proxy
        that ``headroom wrap`` reused — rather than started — never sees values
        a user exported afterwards. Instead of a disruptive restart (cold ML
        load, dropped requests, lost caches), ``wrap`` POSTs the values here and
        the proxy applies them in memory, effective on the next request.

        Loopback-only. The body is a flat ``{ENV_NAME: "value"}`` map; unknown
        keys and non-string values are ignored. Returns what was applied plus
        the resulting live config. Last writer wins in a single-worker proxy;
        multi-worker proxies reject the update because overrides are process-local.
        """
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "expected a JSON object of {ENV_NAME: value}"},
            )
        if proxy.config.worker_processes > 1:
            return JSONResponse(
                status_code=409,
                content={
                    "error": (
                        "runtime environment hot reload is unavailable with multiple "
                        "worker processes; restart the proxy with the desired environment"
                    ),
                    "worker_processes": proxy.config.worker_processes,
                },
            )
        applied = runtime_env.set_overrides(body)
        rollout_aliases = {
            key: value for key, value in applied.items() if key == "HEADROOM_OUTPUT_SHAPER"
        }
        if rollout_aliases:
            assert proxy.config.rollout is not None
            proxy.config.rollout = proxy.config.rollout.with_legacy_env(rollout_aliases)
            async with _stats_snapshot_lock:
                _stats_snapshot["value"] = None
                _stats_snapshot["expires_at"] = 0.0
        if applied:
            logger.info("runtime-env hot-reload applied: %s", sorted(applied))
            # Record which runtime-env keys changed (the "what" of a config
            # change) in addition to the generic admin-request audit emitted by
            # the security middleware. Values are intentionally omitted — keys
            # alone avoid logging any secret values that were set.
            record_admin_action(
                request=request,
                action="runtime_env_update",
                status_code=200,
                details={"changed_keys": sorted(applied)},
            )
        return JSONResponse(
            status_code=200,
            content={
                "applied": applied,
                "runtime_env": runtime_env.effective_runtime_env(),
                "rollout": proxy.config.rollout.to_dict() if proxy.config.rollout else None,
            },
        )

    # Vendored dashboard JS (tailwind/htmx/alpine). Mounted before
    # register_provider_routes' catch-all so it is not tunneled upstream.
    from starlette.staticfiles import StaticFiles

    from headroom.dashboard import STATIC_DIR, register_static_mime_types

    # A host whose mime database maps .js to text/plain (stale Windows registry
    # entry, minimal container image) would otherwise have these served as plain
    # text, which the nosniff header above then blocks in the browser (#3179).
    register_static_mime_types()

    # check_dir=False keeps a missing assets directory from aborting proxy
    # startup: the dashboard JS 404s, but proxying itself still works.
    @app.get("/dashboard/static", include_in_schema=False)
    @app.get("/dashboard/static/", include_in_schema=False)
    async def dashboard_static_directory():
        """Keep the static directory boundary out of upstream passthrough."""
        return PlainTextResponse("Not Found", status_code=404)

    app.mount(
        "/dashboard/static",
        StaticFiles(directory=STATIC_DIR, check_dir=False),
        name="dashboard-static",
    )

    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        """Serve the Headroom dashboard UI."""
        return get_dashboard_html()

    # --- Dashboard settings API (loopback-gated, registry-validated) ---------
    # Read/write the curated HEADROOM_* knobs the settings GUI manages. Writes
    # reuse the same loopback guard as the /admin and /debug endpoints and only
    # ever touch keys in the settings_store registry allowlist. All curated
    # knobs are startup-captured, so a write always requires a restart to apply
    # (Phase 3's /settings/apply drives that).
    from headroom import settings_store

    @app.get(
        "/settings/schema", dependencies=[Depends(_require_loopback_or_trusted_dashboard_client)]
    )
    async def settings_schema(_request: Request):
        """Registry + grouped fields + effective values for the settings form."""
        schema = settings_store.to_schema()
        # Tell the UI whether this is a supervised (docker/service) install, where
        # manifest-baked knobs (HEADROOM_PORT/HEADROOM_HOST) are owned by the
        # install manifest and must be rendered read-only. Foreground proxies
        # can edit everything. Fail-open to "not supervised".
        try:
            from headroom.install import runtime as install_runtime

            _manifest, mode = install_runtime.detect_current_deployment()
            schema["supervised"] = mode != "foreground"
        except Exception:  # noqa: BLE001 — schema must render even if detection fails
            schema["supervised"] = False
        return JSONResponse(status_code=200, content=schema)

    @app.get("/settings", dependencies=[Depends(_require_loopback_or_trusted_dashboard_client)])
    async def settings_get(_request: Request):
        """Return stored (file) values only; secret fields masked."""
        return JSONResponse(status_code=200, content=settings_store.stored_values())

    @app.post(
        "/settings",
        dependencies=[
            Depends(_require_loopback_or_trusted_dashboard_client),
            Depends(_require_same_origin_or_trusted_dashboard_client),
        ],
    )
    async def settings_post(request: Request):
        """Persist settings. Unknown key -> 400; bad type/enum/range -> 422.

        Loopback-only (mutates process configuration; treated as an admin
        action). The audit log records which keys changed — never their values,
        so no secret is leaked.
        """
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("values"), dict):
            return JSONResponse(
                status_code=400,
                content={"error": "expected a JSON object with a `values` object"},
            )
        values = body["values"]
        before = settings_store.load()
        try:
            settings_store.save(values)
        except settings_store.SettingsValidationError as exc:
            if exc.unknown_keys:
                return JSONResponse(
                    status_code=400,
                    content={"error": "unknown settings key(s)", "unknown_keys": exc.unknown_keys},
                )
            return JSONResponse(
                status_code=422,
                content={"error": "invalid settings value(s)", "field_errors": exc.field_errors},
            )
        after = settings_store.load()
        changed_keys = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        record_admin_action(
            request=request,
            action="settings_update",
            status_code=200,
            details={"changed_keys": changed_keys},
        )
        return JSONResponse(
            status_code=200,
            content={"ok": True, "needs_restart": bool(changed_keys), "changed_keys": changed_keys},
        )

    @app.post(
        "/settings/apply",
        dependencies=[
            Depends(_require_loopback_or_trusted_dashboard_client),
            Depends(_require_same_origin_or_trusted_dashboard_client),
        ],
    )
    async def settings_apply(request: Request):
        """Persist settings (optional body) then restart the proxy to apply them.

        Loopback-only. Service deployments self-restart (one-click): we flush a
        202 BEFORE the detached restarter tears this process down, and the UI
        polls /health to detect the proxy returning. Docker deployments cannot
        self-restart (no docker socket in-container), so we surface the host
        command instead. Foreground `headroom proxy` returns a manual-restart
        instruction. /health is the single source of truth for "proxy is back".
        """
        from starlette.background import BackgroundTask

        from headroom.install import runtime as install_runtime

        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            body = None
        # `values` is optional — the UI may have already saved via POST /settings.
        # When present, persist it under the same validation contract.
        if isinstance(body, dict) and isinstance(body.get("values"), dict):
            try:
                settings_store.save(body["values"])
            except settings_store.SettingsValidationError as exc:
                if exc.unknown_keys:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "unknown settings key(s)",
                            "unknown_keys": exc.unknown_keys,
                        },
                    )
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "invalid settings value(s)",
                        "field_errors": exc.field_errors,
                    },
                )
        _manifest, mode = install_runtime.detect_current_deployment()
        record_admin_action(
            request=request,
            action="settings_apply",
            status_code=202 if mode == "service" else 200,
            details={"mode": mode},
        )
        if mode == "service":
            # Flush the 202 first; run the detached restart only after the
            # response body is sent (Starlette runs BackgroundTask post-response),
            # so restarting our own process never drops this response.
            return JSONResponse(
                status_code=202,
                content={"restarted": True, "mode": "service"},
                background=BackgroundTask(install_runtime.restart_current_deployment),
            )
        result = install_runtime.restart_current_deployment()
        return JSONResponse(status_code=200, content=result)

    @app.get(
        "/dashboard/settings",
        response_class=HTMLResponse,
        dependencies=[Depends(_require_loopback_or_trusted_dashboard_client)],
    )
    async def dashboard_settings():
        """Serve the Headroom settings GUI."""
        from headroom.dashboard import get_settings_html

        return get_settings_html()

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        # Registered before register_provider_routes' catch-all passthrough
        # route so browsers' automatic favicon requests for /dashboard are
        # answered locally instead of being tunneled to the wrapped upstream
        # provider (GH #1787).
        return Response(status_code=204)

    DASHBOARD_STATS_CACHE_TTL_SECONDS = 5.0
    _stats_snapshot_lock = asyncio.Lock()
    _stats_snapshot: dict[str, Any] = {"expires_at": 0.0, "value": None}

    THROUGHPUT_CACHE_TTL_SECONDS = 10.0
    _throughput_cache_lock = asyncio.Lock()
    _throughput_cache: dict[str, Any] = {"expires_at": 0.0, "value": None}

    RECENT_REQUEST_LOG_WINDOW = 100

    def _is_recent_request_number(value: Any) -> bool:
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def _recent_request_optional_number(log: dict[str, Any], key: str) -> int | float | None:
        value = log.get(key)
        return value if _is_recent_request_number(value) else None

    def _recent_request_token_accounting_status(log: dict[str, Any]) -> str:
        token_fields = (
            "input_tokens_original",
            "input_tokens_optimized",
            "tokens_saved",
            "savings_percent",
        )
        present = [_is_recent_request_number(log.get(field)) for field in token_fields]
        if all(present):
            return "complete"
        if any(present):
            return "partial"
        return "missing"

    def _build_recent_request_payload(limit: int = RECENT_REQUEST_LOG_WINDOW) -> dict[str, Any]:
        recent_request_logs = proxy.logger.get_recent(limit) if proxy.logger else []
        dashboard_recent_requests = []
        for log in reversed(recent_request_logs):
            token_accounting_status = _recent_request_token_accounting_status(log)
            dashboard_recent_requests.append(
                {
                    "request_id": log.get("request_id"),
                    "timestamp": log.get("timestamp"),
                    "provider": resolve_display_provider(
                        log.get("provider"),
                        openai_api_url=proxy.config.openai_api_url,
                        provider_name=proxy.config.provider_name,
                    ),
                    "model": log.get("model"),
                    "input_tokens_original": _recent_request_optional_number(
                        log, "input_tokens_original"
                    ),
                    "input_tokens_optimized": _recent_request_optional_number(
                        log, "input_tokens_optimized"
                    ),
                    "output_tokens": _recent_request_optional_number(log, "output_tokens"),
                    "tokens_saved": _recent_request_optional_number(log, "tokens_saved"),
                    "savings_percent": _recent_request_optional_number(log, "savings_percent"),
                    "optimization_latency_ms": _recent_request_optional_number(
                        log, "optimization_latency_ms"
                    ),
                    "total_latency_ms": _recent_request_optional_number(log, "total_latency_ms"),
                    "has_exact_tokens": token_accounting_status == "complete",
                    "token_accounting_status": token_accounting_status,
                    "transforms_applied": log.get("transforms_applied", []),
                    "savings_breakdown": log.get("savings_breakdown", []),
                    "waste_signals": log.get("waste_signals"),
                    "tool_schema_saved_tokens": _tool_schema_saved_from_tags(log.get("tags")),
                }
            )
        dashboard_recent_requests = dashboard_recent_requests[:25]
        return {
            "request_logs": recent_request_logs[-10:],
            "recent_requests": dashboard_recent_requests,
        }

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

        import time

        async with _throughput_cache_lock:
            now = time.time()
            if _throughput_cache["expires_at"] < now or _throughput_cache["value"] is None:

                def _compute_throughput():
                    from headroom.perf.analyzer import build_perf_summary, parse_log_files

                    perf_report = parse_log_files(last_n_hours=1.0)
                    return build_perf_summary(perf_report).get("throughput")

                try:
                    throughput = await asyncio.to_thread(_compute_throughput)
                    _throughput_cache["value"] = throughput
                    _throughput_cache["expires_at"] = now + THROUGHPUT_CACHE_TTL_SECONDS
                except Exception as e:
                    logger.warning("Failed to calculate throughput for stats: %s", e, exc_info=True)
                    if _throughput_cache["value"] is None:
                        _throughput_cache["value"] = None
            throughput = _throughput_cache["value"]

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

        def _pct(part: int | float, whole: int | float) -> float:
            return round((float(part) / float(whole)) * 100.0, 2) if whole else 0.0

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

        # Calculate total tokens before Headroom-side reduction.
        proxy_compression_tokens = m.tokens_saved_total
        # "All layers" must include tool-schema deferral (the tool_search layer
        # enumerated in by_layer below) — otherwise the advertised total omits it.
        all_layers_tokens_saved = proxy_compression_tokens + m.tool_search_saved_total
        total_tokens_before = m.tokens_input_total + all_layers_tokens_saved
        proxy_total_before_compression = m.tokens_input_total + proxy_compression_tokens
        # `attempted_input_tokens` is the compressible-only denominator
        # (extracted units + tool schema). The "active compression"
        # ratio is what fraction of the tokens we *tried* to compress
        # actually got compressed. Excludes prefix-frozen content
        # (user/system messages, prior turns) we never touched —
        # otherwise the ratio is dominated by content we deliberately
        # avoided changing for prefix-cache safety.
        # `attempted_input_tokens_total` is already pre-compression: it
        # accumulates `unit.tokens_before` for each eligible unit that
        # reached the router, plus the original (pre-compaction) tool
        # schema size. So the savings rate is plain `saved / attempted`
        # — adding `saved` again would double-count.
        attempted_input_tokens = getattr(m, "attempted_input_tokens_total", 0)
        # New-content denominator: what the provider actually billed as
        # non-cache-read input (uncached + cache-write tokens, summed
        # across providers from response usage). Unlike
        # `proxy_total_before_compression`, this does NOT recount the
        # full transcript on every turn — a long session's history is
        # served from prefix cache, not re-billed, so it doesn't belong
        # in a denominator that claims to measure what compression had
        # any power over. Tokens Headroom removed never reached the
        # provider at all, so they're added back to form the baseline.
        _pc_totals = prefix_cache_stats.get("totals", {})
        new_input_tokens = int(_pc_totals.get("uncached_input_tokens", 0) or 0) + int(
            _pc_totals.get("cache_write_tokens", 0) or 0
        )

        # Build human-readable summary
        summary = _build_session_summary(proxy, m, prefix_cache_stats, total_tokens_before)
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
        cache_net_usd = prefix_cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
        total_tokens_all_layers = all_layers_tokens_saved
        persistent_savings = m.savings_tracker.stats_preview()
        display_session = persistent_savings.get("display_session", {})
        recent_request_logs = proxy.logger.get_recent(10_000) if proxy.logger else []
        recent_request_payload = _build_recent_request_payload()

        # Tool-schema deferral savings: tool-definition tokens kept out of the
        # model's context by deferring heavy schemas until they're needed
        # (native tool-search injection + any registered turn-hook tools
        # rewrite). Attributed to Headroom only — see _tool_schema_saved_from_tags.
        # Aggregated over the recent request-log window.
        tool_schema_tokens = 0
        tool_schema_requests = 0
        for _ts_log in recent_request_logs:
            _ts_saved = _tool_schema_saved_from_tags(_ts_log.get("tags"))
            if _ts_saved > 0:
                tool_schema_tokens += _ts_saved
                tool_schema_requests += 1
        agent_usage = _build_agent_usage_summary(
            recent_request_logs,
            requests_by_provider=_remap_provider_counts(dict(m.requests_by_provider), proxy.config),
            requests_by_model=dict(m.requests_by_model),
            global_before_tokens=proxy_total_before_compression,
            global_after_tokens=m.tokens_input_total,
            global_tokens_saved=proxy_compression_tokens,
            global_output_tokens=m.tokens_output_total,
        )

        # Output-side reduction (counterfactual estimate from the shaper's
        # ledger). Distinct from input compression above: these are OUTPUT
        # tokens the model didn't emit because we steered verbosity / routed
        # effort down. Always labelled estimated-vs-measured + a CI so it's
        # never mistaken for an exact count. Best-effort — never break /stats.
        output_reduction: dict[str, Any] = {"available": False}
        try:
            from headroom.proxy.output_savings import get_recorder
            from headroom.proxy.output_shaper import (
                OutputShaperSettings,
                resolve_verbosity_level,
                shaper_enabled_for,
            )

            # The active steering level enables the modelled fallback, which is
            # what a deployment with no holdout and no learned baseline has --
            # i.e. every fresh install, since `learn --verbosity` needs history
            # that predates the shaper.
            _olevel: int | None = None
            try:
                _osettings = OutputShaperSettings.from_env(
                    enabled=shaper_enabled_for(getattr(proxy, "config", None))
                )
                if _osettings.enabled:
                    _olevel = resolve_verbosity_level(_osettings)[0]
            except Exception:  # pragma: no cover - defensive
                _olevel = None

            _oest = get_recorder().estimate(_olevel)
            if _oest.n_requests > 0:
                output_reduction = {
                    "available": True,
                    "method": _oest.kind,  # "measured" | "estimated" | "modelled"
                    # A modelled band is the spread between the two benchmarked
                    # models, not a sampling CI. The UI must not call it one.
                    "band_is_ci": _oest.kind != "modelled",
                    "tokens_saved": round(_oest.tokens_saved),
                    "baseline_tokens": round(_oest.baseline_tokens),
                    "reduction_percent": round(_oest.pct, 1),
                    "ci_low_percent": round(_oest.ci_low_pct, 1),
                    "ci_high_percent": round(_oest.ci_high_pct, 1),
                    "requests": _oest.n_requests,
                }
        except Exception:  # pragma: no cover - defensive
            pass

        # Model-routing section: populated only when an extension registered a
        # routing-stats provider (headroom.proxy.routing_stats); None otherwise.
        from headroom.proxy.routing_stats import get_routing_stats

        return {
            "summary": summary,
            "agent_usage": agent_usage,
            "routing": get_routing_stats(),
            "savings": {
                "total_tokens": total_tokens_all_layers,
                "per_project": persistent_savings.get("projects", {}),
                # Attribution only: these rows explain the canonical headline
                # and are not added to it again.
                "by_source": sorted(
                    (dict(row) for row in m.savings_by_source.values()),
                    key=lambda row: (-int(row.get("tokens", 0)), str(row["source"])),
                ),
                "by_layer": {
                    "compression": {
                        "tokens": proxy_compression_tokens,
                        "proxy_tokens": proxy_compression_tokens,
                        "all_layers_tokens": all_layers_tokens_saved,
                        "description": ("Tokens removed by Headroom proxy compression."),
                    },
                    "prefix_cache": {
                        "discount_usd": round(cache_net_usd, 4),
                        "description": (
                            "Cost discount from provider prefix caching. "
                            "Headroom's CacheAligner improves hit rates; "
                            "baseline caching is provider-native."
                        ),
                    },
                    "output_shaping": {
                        **output_reduction,
                        "description": (
                            "OUTPUT tokens the model didn't emit because the shaper "
                            "steered verbosity down. Counterfactual — measured "
                            "(A/B holdout), estimated (vs a learned baseline), or "
                            "modelled (a benchmark factor, when this deployment has "
                            "produced neither). Always labelled with which."
                        ),
                    },
                    "tool_search": {
                        "tokens": tool_schema_tokens,
                        "tokens_saved": tool_schema_tokens,
                        "requests": tool_schema_requests,
                        "window": len(recent_request_logs),
                        "description": (
                            "Tool-definition tokens kept out of the model's context "
                            "by deferring heavy tool schemas until they're searched "
                            "for. Counted only when Headroom performed the deferral — "
                            "not when the client (e.g. Claude Code / Codex) already "
                            "had tool search enabled. Aggregated over the recent "
                            "request window."
                        ),
                    },
                },
            },
            "requests": {
                "total": m.requests_total,
                "cached": m.requests_cached,
                "rate_limited": m.requests_rate_limited,
                "failed": m.requests_failed,
                "by_provider": _remap_provider_counts(dict(m.requests_by_provider), proxy.config),
                "by_model": dict(m.requests_by_model),
                "by_stack": dict(m.requests_by_stack),
            },
            "tokens": {
                "input": m.tokens_input_total,
                "output": m.tokens_output_total,
                "output_saved": output_reduction.get("tokens_saved", 0),
                "output_reduction_percent": output_reduction.get("reduction_percent", 0),
                "output_reduction": output_reduction,
                "saved": all_layers_tokens_saved,
                "proxy_compression_saved": proxy_compression_tokens,
                "proxy_total_before_compression": proxy_total_before_compression,
                "total_before_compression": total_tokens_before,
                "all_layers_saved": all_layers_tokens_saved,
                # Compressible-only denominator: tokens we extracted as
                # candidates + tool-schema tokens we compacted. Excludes
                # frozen-prefix content (user msgs, system prompt, prior
                # turns) that we deliberately don't touch. Already
                # pre-compression — do NOT add `tokens_saved` again.
                "proxy_attempted_tokens": attempted_input_tokens,
                # Active compression: savings as a fraction of what we
                # *tried* to compress. The number the dashboard headline
                # should show — it answers "are we doing well *when we
                # have something to compress?*" rather than diluting the
                # win by frozen-prefix bytes we never touched.
                # All-layers numerator, matching denominator: `attempted_input_tokens`
                # already counts tool schemas we COMPACTED, so pairing it with a
                # compression-only numerator undercounted every tool-heavy session.
                # Deferred schemas are added to both sides — they were attempted work
                # that succeeded completely.
                "active_savings_percent": round(
                    (
                        all_layers_tokens_saved
                        / (attempted_input_tokens + m.tool_search_saved_total)
                        * 100
                    )
                    if (attempted_input_tokens + m.tool_search_saved_total) > 0
                    else 0,
                    2,
                ),
                # Whole-request ratio kept for transparency. Heavily
                # diluted by frozen prefix on Codex-style requests
                # where most input is non-compressible by design.
                "proxy_savings_percent": round(
                    (proxy_compression_tokens / proxy_total_before_compression * 100)
                    if proxy_total_before_compression > 0
                    else 0,
                    2,
                ),
                # New-content-relative rate: savings as a fraction of the
                # input that would have newly entered context (provider-
                # billed uncached + cache-write tokens, plus the tokens
                # compression removed before they could be billed). The
                # whole-request ratios above recount the FULL transcript
                # every turn, so a 200-turn session counts its history
                # 200x into the denominator and long-running sessions
                # (1M-context models never compact) read as ~0% no
                # matter how well compression performs on new content.
                # Guarded on new_input_tokens > 0 (not the full sum): the
                # cache accumulators only see requests with cache
                # activity, so a deployment with no cache metrics (e.g.
                # Bedrock) would otherwise divide savings by themselves
                # and report ~100%. No usage data -> report 0, not a lie.
                "new_input_tokens": new_input_tokens,
                "new_input_savings_percent": round(
                    (proxy_compression_tokens / (new_input_tokens + proxy_compression_tokens) * 100)
                    if new_input_tokens > 0
                    else 0,
                    2,
                ),
                "savings_percent": round(
                    (all_layers_tokens_saved / total_tokens_before * 100)
                    if total_tokens_before > 0
                    else 0,
                    2,
                ),
                "all_layers_savings_percent": round(
                    (all_layers_tokens_saved / total_tokens_before * 100)
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
            "extension_savings": dict(m.extension_savings),
            "codex_ws": {
                "units_total": m.codex_ws_units_total,
                "units_modified_total": m.codex_ws_units_modified_total,
                "units_by_strategy": dict(m.codex_ws_units_by_strategy),
                "units_by_category": dict(m.codex_ws_units_by_category),
                "units_by_content_type": dict(m.codex_ws_units_by_content_type),
                "units_by_text_shape": dict(m.codex_ws_units_by_text_shape),
                "units_to_kompress_total": m.codex_ws_units_to_kompress_total,
                "units_kompress_attempted_total": m.codex_ws_units_kompress_attempted_total,
                "units_to_kompress_percent": _pct(
                    m.codex_ws_units_to_kompress_total,
                    m.codex_ws_units_total,
                ),
                "units_kompress_attempted_percent": _pct(
                    m.codex_ws_units_kompress_attempted_total,
                    m.codex_ws_units_total,
                ),
                "unit_elapsed_ms": {
                    "average": round(
                        m.codex_ws_unit_elapsed_ms_sum / m.codex_ws_units_total,
                        2,
                    )
                    if m.codex_ws_units_total
                    else 0.0,
                    "max": round(m.codex_ws_unit_elapsed_ms_max, 2),
                },
                "unit_bytes_sum": m.codex_ws_unit_bytes_sum,
                "unit_tokens_before_sum": m.codex_ws_unit_tokens_before_sum,
                "unit_tokens_after_sum": m.codex_ws_unit_tokens_after_sum,
                "unit_tokens_saved_sum": m.codex_ws_unit_tokens_saved_sum,
                "frames_attempted_total": m.codex_ws_frames_attempted_total,
                "frames_compressed_total": m.codex_ws_frames_compressed_total,
                "frames_failed_total": m.codex_ws_frames_failed_total,
                "frames_to_kompress_total": m.codex_ws_frames_to_kompress_total,
                "frames_kompress_attempted_total": (m.codex_ws_frames_kompress_attempted_total),
                "frames_to_kompress_percent": _pct(
                    m.codex_ws_frames_to_kompress_total,
                    m.codex_ws_frames_attempted_total,
                ),
                "frames_kompress_attempted_percent": _pct(
                    m.codex_ws_frames_kompress_attempted_total,
                    m.codex_ws_frames_attempted_total,
                ),
                "frame_elapsed_ms": {
                    "average": round(
                        m.codex_ws_frame_elapsed_ms_sum / m.codex_ws_frames_attempted_total,
                        2,
                    )
                    if m.codex_ws_frames_attempted_total
                    else 0.0,
                    "max": round(m.codex_ws_frame_elapsed_ms_max, 2),
                },
                "frame_bytes_before_sum": m.codex_ws_frame_bytes_before_sum,
                "frame_bytes_after_sum": m.codex_ws_frame_bytes_after_sum,
                "frame_attempted_tokens_sum": m.codex_ws_frame_attempted_tokens_sum,
                "frame_tokens_saved_sum": m.codex_ws_frame_tokens_saved_sum,
            },
            "waste_signals": dict(m.waste_signals_total) if m.waste_signals_total else {},
            # ContentRouter protection categories aggregated across the
            # session. Lets operators see, e.g., that 80% of messages
            # were `user_msg` (protected) and only 5% reached the
            # compressor — explains why compression rate is low and
            # whether `--compress-user-messages` would help (#454).
            "router": {
                "route_counts": dict(m.router_route_counts) if m.router_route_counts else {},
            },
            "savings_history": m.savings_history[-100:],  # Last 100 data points
            "display_session": display_session,
            # Whether LiteLLM is importable. Pricing (the "$ Saved" tile) is
            # derived entirely from LiteLLM's cost tables, and LiteLLM is gated
            # off on Python >=3.14 in pyproject — so when this is False the
            # dashboard tells the user to reinstall on 3.13 instead of just
            # showing $0.00 forever.
            "litellm_available": LITELLM_AVAILABLE,
            "persistent_savings": persistent_savings,
            "prefix_cache": prefix_cache_stats,
            "cost": _merge_cost_stats(
                proxy.cost_tracker.stats() if proxy.cost_tracker else None,
                prefix_cache_stats,
            ),
            "compression": {
                "ccr_entries": compression_stats.get("entry_count", 0),
                "ccr_max_entries": compression_stats.get("max_entries", 0),
                "original_tokens_cached": compression_stats.get("total_original_tokens", 0),
                "compressed_tokens_cached": compression_stats.get("total_compressed_tokens", 0),
                "ccr_retrievals": compression_stats.get("total_retrievals", 0),
            },
            "compression_cache": compression_cache_stats,
            # Per-language AST compression pauses. Empty on a healthy install;
            # non-empty is the explanation for a savings drop in one language.
            "code_syntax_breaker": _code_syntax_breaker_status(),
            # Always False: the anonymous telemetry beacon was removed, so no
            # telemetry is ever shipped externally (local collection only).
            "anon_telemetry_shipping": False,
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
            "proxy_inbound": proxy.metrics.inbound_snapshot(),
            "cache": await proxy.cache.stats() if proxy.cache else None,
            "rate_limiter": await proxy.rate_limiter.stats() if proxy.rate_limiter else None,
            **recent_request_payload,
            "log_full_messages": proxy.config.log_full_messages if proxy else False,
            **get_quota_registry().get_all_stats(),
            "throughput": throughput,
            # Effective state from the running process. This is the supported
            # black-box provenance surface for benchmark/qualification tools.
            "rollout": proxy.config.rollout.to_dict() if proxy.config.rollout else None,
        }

    def _dashboard_config_payload() -> dict[str, Any]:
        profile_kwargs = proxy_pipeline_kwargs(config)
        target_ratio = profile_kwargs.get("target_ratio", config.target_ratio)
        target_savings_percent = None
        if isinstance(target_ratio, int | float):
            target_savings_percent = round(max(0.0, min(1.0, 1.0 - float(target_ratio))) * 100, 1)
        return {
            "savings_profile": config.savings_profile,
            "target_ratio": target_ratio,
            "target_savings_percent": target_savings_percent,
            "compress_user_messages": bool(
                profile_kwargs.get("compress_user_messages", config.compress_user_messages)
            ),
            "compress_system_messages": bool(
                profile_kwargs.get("compress_system_messages", config.compress_system_messages)
            ),
            "protect_recent": profile_kwargs.get("read_protection_window", config.protect_recent),
            "protect_analysis_context": config.protect_analysis_context,
            "min_tokens_to_crush": profile_kwargs.get(
                "min_tokens_to_compress", config.min_tokens_to_crush
            ),
            "max_items_after_crush": profile_kwargs.get(
                "max_items_after_crush", config.max_items_after_crush
            ),
            "smart_crusher_with_compaction": profile_kwargs.get(
                "smart_crusher_with_compaction",
                config.smart_crusher_with_compaction,
            ),
            "force_kompress": bool(profile_kwargs.get("force_kompress", False)),
            "accuracy_guard": config.accuracy_guard,
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
    async def stats(request: Request, cached: bool = False):
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

        ``recent_requests`` / ``request_logs`` (per-request ids, providers,
        models, errors) and ``config`` (backend + savings profile) are embedded
        only for loopback callers — the local dashboard. Network callers still
        get the aggregate counters but never the per-request metadata.
        """
        include_sensitive = _request_can_view_dashboard_metadata(
            request,
            trusted_dashboard_client_cidrs,
        )
        if cached:
            payload = dict(await _get_cached_stats_payload())
            if include_sensitive:
                # Refresh the per-request tail on top of the cached snapshot.
                payload.update(_build_recent_request_payload())
                payload["config"] = _dashboard_config_payload()
        else:
            payload = await _build_stats_payload()
            if include_sensitive:
                payload["config"] = _dashboard_config_payload()
        if not include_sensitive:
            # _build_stats_payload bakes these in; strip for network callers.
            payload.pop("recent_requests", None)
            payload.pop("request_logs", None)
        return payload

    @app.get("/stats-lifetime")
    async def stats_lifetime(request: Request):
        """Return persisted lifetime aggregates with sensitive fields gated."""
        payload = dict(proxy.metrics.savings_tracker.lifetime_response())
        include_sensitive = _request_can_view_dashboard_metadata(
            request,
            trusted_dashboard_client_cidrs,
        )
        if not include_sensitive:
            payload.pop("projects", None)
            persistence = payload.get("persistence")
            if isinstance(persistence, dict):
                payload["persistence"] = {**persistence, "error": None}
        return payload

    @app.post(
        "/stats/reset",
        dependencies=[Depends(_require_loopback), Depends(_require_same_origin)],
    )
    async def stats_reset():
        """Reset in-memory proxy stats for local test/debug isolation."""
        await proxy.metrics.reset_runtime()
        if proxy.cost_tracker:
            proxy.cost_tracker.reset_runtime()
        async with _stats_snapshot_lock:
            _stats_snapshot["value"] = None
            _stats_snapshot["expires_at"] = 0.0
        return JSONResponse(status_code=200, content={"status": "reset"})

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

    @app.get("/transformations/feed", dependencies=[Depends(_require_loopback)])
    async def transformations_feed(limit: int = 20):
        """Get recent message transformations for the live feed.

        Loopback-only: when ``log_full_messages`` is enabled this returns the
        full request/response message bodies (prompt content and completions)
        via ``request_messages`` / ``compressed_messages`` / ``response_content``.
        With the default ``--host 0.0.0.0`` Docker bind, leaving it open would
        expose chat history to anyone able to reach the proxy port. The
        dashboard runs in the user's browser on loopback, so this gate does not
        break legitimate use.

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
                        "provider": resolve_display_provider(
                            log.get("provider"),
                            openai_api_url=proxy.config.openai_api_url,
                            provider_name=proxy.config.provider_name,
                        ),
                        "model": log.get("model"),
                        "input_tokens_original": log.get("input_tokens_original"),
                        "input_tokens_optimized": log.get("input_tokens_optimized"),
                        "tokens_saved": log.get("tokens_saved"),
                        "savings_percent": log.get("savings_percent"),
                        "transforms_applied": log.get("transforms_applied", []),
                        "request_messages": log.get("request_messages"),
                        "compressed_messages": log.get("compressed_messages"),
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
    @app.get("/debug/memory", dependencies=[Depends(_require_loopback)])
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

    @app.post(
        "/cache/clear",
        dependencies=[Depends(_require_loopback), Depends(_require_same_origin)],
    )
    async def clear_cache():
        """Clear the response cache.

        Loopback-only: this mutates server state. With the default
        ``--host 0.0.0.0`` Docker bind, an unauthenticated POST from any
        network-reachable client would otherwise let them forcibly evict the
        proxy's cached completions — a denial-of-service / cost-amplification
        lever (every cleared entry forces a fresh upstream call).
        """
        if proxy.cache:
            await proxy.cache.clear()
            return {"status": "cleared"}
        return {"status": "cache disabled"}

    # CCR (Compress-Cache-Retrieve) endpoints
    @app.post(
        "/v1/retrieve",
        dependencies=[Depends(_require_loopback), Depends(_require_same_origin)],
    )
    async def ccr_retrieve(request: Request):
        """Retrieve original content from CCR compression cache.

        This is the "Retrieve" part of CCR (Compress-Cache-Retrieve).
        When SmartCrusher compresses tool outputs, the original data is cached.
        LLMs can call this endpoint to get more data if needed.

        Request body:
            hash (str): Hash key from compression marker (required)

        Response:
            {"hash": "...", "original_content": "...", ...}
        """
        data = await request.json()
        hash_key = data.get("hash")

        if not hash_key:
            raise HTTPException(status_code=400, detail="hash required")

        store = get_compression_store()

        entry_status = store.get_entry_status(hash_key, clean_expired=True)
        if entry_status["status"] != "available":
            raise HTTPException(
                status_code=404,
                detail=format_retrieval_miss_detail(entry_status),
            )

        # Retrieval is by hash: always return the full original content.
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
            status_code=404,
            detail=format_retrieval_miss_detail(
                store.get_entry_status(hash_key, clean_expired=True)
            ),
        )

    @app.get("/v1/retrieve/stats", dependencies=[Depends(_require_loopback)])
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

    @app.get("/v1/feedback", dependencies=[Depends(_require_loopback)])
    async def ccr_feedback():
        """Get CCR feedback loop statistics and learned patterns.

        This endpoint exposes the feedback loop's learned patterns for monitoring
        and debugging. It shows:
        - Per-tool retrieval rates (high = compress less aggressively)
        - Aggregate compression/retrieval counters per tool

        Use this to understand how well compression is working and whether
        the feedback loop is adjusting appropriately.

        Loopback-guarded and query-text free for the same reason as the
        telemetry and TOIN endpoints (#2926/#2927): ``common_queries`` and
        ``queried_fields`` are built verbatim from agent search queries, so
        they stay out of the response even on the guarded path.
        """
        feedback = get_compression_feedback()
        stats = _feedback_stats_without_query_text(feedback.get_stats())
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

    @app.get("/v1/feedback/{tool_name}", dependencies=[Depends(_require_loopback)])
    async def ccr_feedback_for_tool(tool_name: str):
        """Get compression hints for a specific tool.

        Returns feedback-based hints that would be used for compressing
        this tool's output.

        Loopback-guarded, and the pattern block excludes ``common_queries``
        and ``queried_fields`` — both are raw agent query text (#2926/#2927).
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
            }
            if patterns
            else None,
        }

    # Telemetry endpoints (Data Flywheel)
    @app.get("/v1/telemetry", dependencies=[Depends(_require_loopback)])
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

    @app.get("/v1/telemetry/export", dependencies=[Depends(_require_loopback)])
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

    @app.post(
        "/v1/telemetry/import",
        dependencies=[Depends(_require_loopback), Depends(_require_same_origin)],
    )
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

    @app.get("/v1/telemetry/tools", dependencies=[Depends(_require_loopback)])
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

    @app.get("/v1/telemetry/tools/{signature_hash}", dependencies=[Depends(_require_loopback)])
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
    @app.get("/v1/toin/stats", dependencies=[Depends(_require_loopback)])
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

    @app.get("/v1/toin/patterns", dependencies=[Depends(_require_loopback)])
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

    @app.get("/v1/toin/pattern/{hash_prefix}", dependencies=[Depends(_require_loopback)])
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
                # Keep this response aligned with /v1/toin/patterns while
                # excluding query text, field semantics, and other internal
                # learning state from the detail endpoint.
                return {
                    "compressions": pattern_dict.get("total_compressions", 0),
                    "retrievals": pattern_dict.get("total_retrievals", 0),
                    "retrieval_rate": pattern_dict.get("retrieval_rate", 0.0),
                    "confidence": pattern_dict.get("confidence", 0.0),
                    "skip_recommended": pattern_dict.get("skip_compression_recommended", False),
                    "optimal_max_items": pattern_dict.get("optimal_max_items", 20),
                }

        raise HTTPException(
            status_code=404, detail=f"No TOIN pattern found with hash starting with: {hash_prefix}"
        )

    @app.get("/v1/retrieve/{hash_key}", dependencies=[Depends(_require_loopback)])
    async def ccr_retrieve_get(hash_key: str):
        """GET version of CCR retrieve for easier testing."""
        store = get_compression_store()
        entry_status = store.get_entry_status(hash_key, clean_expired=True)

        if entry_status["status"] != "available":
            raise HTTPException(
                status_code=404,
                detail=format_retrieval_miss_detail(entry_status),
            )

        # Retrieval is by hash: always return the full original content.
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
            status_code=404,
            detail=format_retrieval_miss_detail(
                store.get_entry_status(hash_key, clean_expired=True)
            ),
        )

    # CCR Tool Call Handler - for agent frameworks to call when LLM uses headroom_retrieve
    @app.post("/v1/retrieve/tool_call", dependencies=[Depends(_require_loopback)])
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
                    "input": {"hash": "abc123"}
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
        hash_key = parse_tool_call(tool_call, provider)

        if hash_key is None:
            raise HTTPException(
                status_code=400, detail=f"Invalid tool call or not a {CCR_TOOL_NAME} call"
            )

        # Perform retrieval
        store = get_compression_store()
        entry_status = store.get_entry_status(hash_key, clean_expired=True)

        if entry_status["status"] != "available":
            retrieval_data = {
                "error": format_retrieval_miss_detail(entry_status),
                "hash": hash_key,
                "status": entry_status["status"],
                "ttl_seconds": entry_status.get("ttl_seconds", entry_status["default_ttl_seconds"]),
            }
        else:
            # Retrieval is by hash: always return the full original content.
            entry = store.retrieve(hash_key)
            if entry:
                retrieval_data = {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                }
            else:
                miss_status = store.get_entry_status(hash_key, clean_expired=True)
                retrieval_data = {
                    "error": format_retrieval_miss_detail(miss_status),
                    "hash": hash_key,
                    "status": miss_status["status"],
                    "ttl_seconds": miss_status.get(
                        "ttl_seconds", miss_status["default_ttl_seconds"]
                    ),
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

    # Compression-only endpoint (for TypeScript SDK and other HTTP clients).
    # Loopback-only by default (guard added in #1537). An operator can opt in to
    # network access for an authorized in-network sidecar/gateway (e.g. Kong,
    # LiteLLM) on a trusted network by setting HEADROOM_COMPRESS_ALLOW_REMOTE=1,
    # which drops ONLY this route's loopback dependency. Inbound auth
    # (HEADROOM_PROXY_TOKEN via _security_gate) and network scoping still apply;
    # all other _require_loopback routes are unaffected. Unset/false preserves
    # today's loopback-only behavior.
    _compress_dependencies = (
        []
        if _get_env_bool("HEADROOM_COMPRESS_ALLOW_REMOTE", False)
        else [Depends(_require_loopback)]
    )

    @app.post("/v1/compress", dependencies=_compress_dependencies)
    async def compress_messages(request: Request):
        return await proxy.handle_compress(request)

    # Sidecar-mode usage relay: same exposure policy as /v1/compress — the two
    # form one contract (compress returns the bytes, usage reports what the
    # provider said about them), so they must be reachable from the same place.
    @app.post("/v1/usage", dependencies=_compress_dependencies)
    async def compress_usage(request: Request):
        return await proxy.handle_compress_usage(request)

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
        if field.name == "rollout":
            assert config.rollout is not None
            payload["_rollout_snapshot"] = config.rollout.to_internal_dict()
            continue
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
            values = json.loads(raw_config)
            if not isinstance(values, dict):
                raise TypeError("proxy config JSON must be an object")
            if "_rollout_snapshot" in values:
                rollout_value = values.pop("_rollout_snapshot")
                from headroom.rollout import RolloutSnapshot

                values["rollout"] = RolloutSnapshot.from_internal_dict(rollout_value)
            return ProxyConfig(**values)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning(
                "Invalid %s; falling back to HEADROOM_* env vars", _MULTI_WORKER_CONFIG_ENV
            )

    from headroom.rollout import resolve_rollout

    rollout = resolve_rollout()
    return ProxyConfig(
        rollout=rollout,
        host=_get_env_str("HEADROOM_HOST", "127.0.0.1"),
        port=_get_env_int("HEADROOM_PORT", 8787),
        openai_api_url=os.environ.get("OPENAI_TARGET_API_URL"),
        anthropic_api_url=os.environ.get("ANTHROPIC_TARGET_API_URL"),
        anthropic_buffered_request_timeout_seconds=_get_env_int(
            "HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS",
            600,
            min_value=1,
        ),
        write_timeout_seconds=_get_env_int(
            "HEADROOM_WRITE_TIMEOUT_SECONDS",
            150,
            min_value=1,
        ),
        buffered_ccr_grace_seconds=_get_env_float(
            "HEADROOM_BUFFERED_CCR_GRACE_SECONDS",
            DEFAULT_BUFFERED_CCR_GRACE_SECONDS,
        ),
        vertex_api_url=os.environ.get("VERTEX_TARGET_API_URL"),
        backend=_get_env_str("HEADROOM_BACKEND", "anthropic"),
        bedrock_region=_get_env_str("HEADROOM_BEDROCK_REGION", "us-west-2"),
        bedrock_profile=os.environ.get("AWS_PROFILE"),
        bedrock_api_url=os.environ.get("BEDROCK_TARGET_API_URL"),
        anyllm_provider=_get_env_str("HEADROOM_ANYLLM_PROVIDER", "openai"),
        disable_kompress=_get_env_bool("HEADROOM_DISABLE_KOMPRESS", False),
        disable_kompress_fallback=_get_env_bool("HEADROOM_DISABLE_KOMPRESS_FALLBACK", False),
        disable_kompress_anthropic=_get_env_optional_bool("HEADROOM_DISABLE_KOMPRESS_ANTHROPIC"),
        disable_kompress_openai=_get_env_optional_bool("HEADROOM_DISABLE_KOMPRESS_OPENAI"),
        force_kompress_all=_get_env_bool("HEADROOM_FORCE_KOMPRESS_ALL", False),
        lossless=_get_env_bool("HEADROOM_LOSSLESS", False),
        compress_passthrough=_get_env_bool("HEADROOM_COMPRESS_PASSTHROUGH", False),
        max_connections=_get_env_int("HEADROOM_MAX_CONNECTIONS", 500),
        max_keepalive_connections=_get_env_int("HEADROOM_MAX_KEEPALIVE", 100),
        keepalive_expiry=_get_env_float("HEADROOM_KEEPALIVE_EXPIRY", 90.0),
        http2=_get_env_bool("HEADROOM_HTTP2", True),
        http_proxy=os.environ.get("HEADROOM_HTTP_PROXY") or None,
        periodic_toin_stats_enabled=_get_env_bool("HEADROOM_PERIODIC_TOIN_STATS", True),
        periodic_malloc_trim_enabled=_get_env_bool(
            "HEADROOM_MALLOC_TRIM", sys.platform == "darwin"
        ),
        malloc_trim_interval_seconds=_get_env_int("HEADROOM_MALLOC_TRIM_INTERVAL_SECONDS", 60),
        proxy_token=os.environ.get("HEADROOM_PROXY_TOKEN") or None,
        offline=_get_env_bool("HEADROOM_OFFLINE", False),
        # Default mode is CACHE (Headroom's coding posture): delta-only compression
        # at ~0 prefix-cache busts. HEADROOM_MODE overrides.
        mode=normalize_proxy_mode(_get_env_str("HEADROOM_MODE", PROXY_MODE_CACHE)),
        # Default savings profile is "coding" so proxy_pipeline_kwargs applies its
        # posture (compress_user, protect_recent, min_tokens). HEADROOM_SAVINGS_PROFILE
        # overrides.
        savings_profile=os.environ.get("HEADROOM_SAVINGS_PROFILE") or "coding",
        read_maturation=rollout.is_enabled("read_maturation"),
        read_maturation_quiesce_turns=_get_env_int("HEADROOM_READ_MATURATION_QUIESCE_TURNS", 5),
        read_maturation_max_hold_turns=_get_env_int("HEADROOM_READ_MATURATION_MAX_HOLD_TURNS", 25),
        read_maturation_min_size_bytes=_get_env_int(
            "HEADROOM_READ_MATURATION_MIN_SIZE_BYTES", 2048
        ),
        model_router=ModelRouterConfig.from_env(
            os.environ.get("HEADROOM_MODEL_ROUTER_ENABLED"),
            os.environ.get("HEADROOM_MODEL_ROUTES"),
        ),
    )


def create_app_from_env() -> FastAPI:
    # Seed the coding-profile defaults into the process env BEFORE reading config,
    # so the uvicorn factory launch gets Headroom's out-of-box posture (cache mode,
    # tool-search, dedupe, read protection, …). setdefault → explicit env wins.
    from headroom.agent_savings import seed_proxy_env_defaults

    seed_proxy_env_defaults()
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
            return "DISABLED (--code-aware or HEADROOM_CODE_AWARE_ENABLED=1 to enable)"
        return "DISABLED  (install headroom-ai[code] to enable)"


def _configure_windows_uvicorn_loop(uvicorn_kwargs: dict[str, Any]) -> None:
    """Select a Windows-safe asyncio loop for uvicorn across versions.

    ProactorEventLoop can close the listening socket on transient AcceptEx
    failures (for example WinError 64 from keep-alive RSTs). SelectorEventLoop
    keeps accept errors scoped to the connection.

    uvicorn >= 0.36 resolves ``asyncio:SelectorEventLoop`` as a custom loop
    factory import path. Older uvicorn only accepts built-in loop names and
    raises KeyError for that value, so we set the selector policy instead.
    """
    import uvicorn as _uvicorn

    if hasattr(_uvicorn.config.Config, "get_loop_factory"):
        uvicorn_kwargs["loop"] = "asyncio:SelectorEventLoop"
    else:
        policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy_cls is not None:
            asyncio.set_event_loop_policy(policy_cls())


def run_server(
    config: ProxyConfig | None = None,
    workers: int = 1,
    limit_concurrency: int = 1000,
    print_banner: bool = True,
):
    """Run the proxy server.

    Args:
        config: Proxy configuration
        workers: Number of worker processes (use N for multi-core scaling)
        limit_concurrency: Max concurrent connections before 503 response
        print_banner: When False, skip the legacy ASCII banner. The
            Click CLI (`headroom proxy`) prints its own startup banner
            before calling this — printing a second banner here is the
            "dual banner" UX issue. Direct `python -m headroom.proxy.server`
            still gets the banner since it has no other startup output.
    """
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI required. Install: pip install fastapi uvicorn httpx")
        sys.exit(1)

    # Seed the request-time coding-profile toggles (tool-search, dedupe, read
    # protection, lossless→lossy, effort-router, block-char floor) into the
    # process env before serving, so downstream per-request readers pick them up.
    # Done here (not in the CLI command) so unit tests that mock run_server never
    # mutate os.environ. setdefault → explicit env still wins. MODE / profile are
    # already resolved into `config` above via their inline defaults.
    from headroom.agent_savings import seed_proxy_env_defaults

    seed_proxy_env_defaults()

    config = config or ProxyConfig()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    config.worker_processes = workers
    code_aware_status = _get_code_aware_banner_status(config)

    # Format connection pool info
    pool_info = f"max={config.max_connections}, keepalive={config.max_keepalive_connections}"
    http2_status = "ENABLED" if (config.http2 and not config.http_proxy) else "DISABLED"

    backend_status = format_backend_status(
        backend=config.backend,
        anyllm_provider=config.anyllm_provider,
        bedrock_region=config.bedrock_region,
    )

    # Resolve upstream API targets for display in the banner (#583).
    api_targets = resolve_api_targets(config.provider_api_overrides)

    if print_banner:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      HEADROOM PROXY SERVER                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Version: 1.0.0                                                      ║
║  Listening: http://{config.host}:{config.port:<5}                                      ║
║  Workers: {workers:<3}  Concurrency Limit: {limit_concurrency:<5}                          ║
║  Backend: {backend_status:<59}║
╠══════════════════════════════════════════════════════════════════════╣
║  UPSTREAM TARGETS:                                                   ║
║    Anthropic:  {api_targets.anthropic:<57}║
║    OpenAI:     {api_targets.openai:<57}║
║    Gemini:     {api_targets.gemini:<57}║
║    Cloud Code: {api_targets.cloudcode:<57}║
║    Vertex AI:  {api_targets.vertex:<57}║
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

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    if not any(
        isinstance(item, _SuppressCancelledErrorFilter) for item in uvicorn_error_logger.filters
    ):
        uvicorn_error_logger.addFilter(_SuppressCancelledErrorFilter())

    app_target: Any
    uvicorn_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        _configure_windows_uvicorn_loop(uvicorn_kwargs)
    if workers > 1:
        # CompressionCache and PrefixTracker are always per-worker instance vars.
        # Python CompressionStore defaults to InMemoryBackend (per-process), so
        # CCR markers written on worker A are invisible to worker B unless a
        # cross-worker backend is configured via HEADROOM_CCR_BACKEND.
        # See RUST_DEV.md -> "Multi-worker deployment -- CCR fragmentation".
        if os.environ.get("HEADROOM_CCR_BACKEND", "").strip():
            logger.warning(
                "Headroom is running with workers=%d. Compression cache, "
                "prefix tracker, TOIN state, and CostTracker are all per-process; "
                "multi-worker deployments produce avoidable cache busts and an "
                "unstable dashboard 'Proxy $ Saved' hero tile (each /stats poll "
                "hits a different worker's partial total) when sessions land on "
                "different workers. Run --workers 1 or place a sticky-session load "
                "balancer in front of multiple --workers 1 processes. "
                "See RUST_DEV.md -> 'Multi-worker deployment -- CCR fragmentation'.",
                workers,
            )
        else:
            logger.warning(
                "Headroom is running with workers=%d. The in-memory CCR store, "
                "compression cache (incl. off-path background compression), prefix "
                "tracker, TOIN state, and CostTracker are all "
                "per-process; multi-worker deployments produce silent CCR retrieval "
                "failures, avoidable cache busts, and an unstable dashboard 'Proxy $ Saved' "
                "hero tile (each /stats poll hits a different worker's partial total) when "
                "sessions land on different workers. Set HEADROOM_CCR_BACKEND=sqlite for a "
                "persistent cross-worker CCR store, run --workers 1, or place a "
                "sticky-session load balancer in front of multiple --workers 1 processes. "
                "See RUST_DEV.md -> 'Multi-worker deployment -- CCR fragmentation'.",
                workers,
            )
        os.environ[_MULTI_WORKER_CONFIG_ENV] = json.dumps(_proxy_config_payload(config))
        app_target = "headroom.proxy.server:create_app_from_env"
        uvicorn_kwargs["factory"] = True
    else:
        app_target = create_app(config)

    # "warning" keeps the default terminal quiet (uvicorn's access log is one line
    # per request). It was previously hardcoded, which left deployed proxies with
    # no way to turn request logging on: operators diagnosing a production
    # incident could not see uvicorn's view of the traffic at all, with no env var
    # and no CLI flag to change it. Overridable now; the default is unchanged.
    uvicorn_log_level = _resolve_uvicorn_log_level()

    uvicorn.run(
        app_target,
        host=config.host,
        port=config.port,
        log_level=uvicorn_log_level,
        workers=workers if workers > 1 else None,  # None = single process (default)
        limit_concurrency=limit_concurrency,
        # Defense-in-depth: the loopback guard for /debug/* endpoints trusts
        # request.client.host. uvicorn's ProxyHeadersMiddleware rewrites that
        # from X-Forwarded-For when FORWARDED_ALLOW_IPS is broader than the
        # default. Disabling proxy_headers here guarantees the guard sees the
        # real peer address regardless of env.
        proxy_headers=False,
        timeout_graceful_shutdown=10,
        **uvicorn_kwargs,
    )


def _get_env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _get_env_optional_bool(name: str) -> bool | None:
    """Tristate boolean env var: unset/empty -> None, truthy -> True, falsy -> False."""
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    return val.lower() in ("true", "1", "yes", "on")


def _get_env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    """Get integer from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        parsed = int(val)
    except ValueError:
        return default
    if min_value is not None and parsed < min_value:
        return default
    return parsed


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


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


# uvicorn rejects anything outside this set with a KeyError during startup, so an
# operator typo in HEADROOM_LOG_LEVEL must not be able to stop the proxy booting.
_UVICORN_LOG_LEVELS = frozenset({"critical", "error", "warning", "info", "debug", "trace"})
_UVICORN_LOG_LEVEL_DEFAULT = "warning"


def _resolve_uvicorn_log_level() -> str:
    """Resolve uvicorn's log level from ``HEADROOM_LOG_LEVEL``.

    Falls back to the previous hardcoded default on an unset or unrecognized
    value, warning loudly rather than failing the boot.
    """
    raw = _get_env_str("HEADROOM_LOG_LEVEL", _UVICORN_LOG_LEVEL_DEFAULT).strip().lower()
    if raw in _UVICORN_LOG_LEVELS:
        return raw
    logger.warning(
        "Ignoring unrecognized HEADROOM_LOG_LEVEL=%r; using %r. Valid values: %s",
        raw,
        _UVICORN_LOG_LEVEL_DEFAULT,
        ", ".join(sorted(_UVICORN_LOG_LEVELS)),
    )
    return _UVICORN_LOG_LEVEL_DEFAULT


def _parse_exclude_tools(cli_excludes: str | None) -> set[str]:
    """Parse extra never-compress tool names from CLI args and env var.

    Both --exclude-tools and HEADROOM_EXCLUDE_TOOLS are comma-separated
    (e.g. "WebSearch,WebFetch"). Each name is added in both original and
    lowercase form for case-insensitive matching, mirroring
    DEFAULT_EXCLUDE_TOOLS. Unset/empty -> empty set (DEFAULT_EXCLUDE_TOOLS
    used unchanged). Entries may contain glob patterns (e.g. "mcp__*"); see
    config.is_tool_excluded for the matching semantics.
    """
    raw = ",".join(s for s in (cli_excludes, os.environ.get("HEADROOM_EXCLUDE_TOOLS")) if s)
    names: set[str] = set()
    for entry in raw.split(","):
        name = entry.strip()
        if name:
            names.add(name)
            names.add(name.lower())
    return names


def _parse_csv_tools(raw: str | None) -> set[str]:
    """Parse a bare CSV tool-name string without merging HEADROOM_EXCLUDE_TOOLS."""
    names: set[str] = set()
    if not raw:
        return names
    for entry in raw.split(","):
        name = entry.strip()
        if name:
            names.add(name)
            names.add(name.lower())
    return names


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
    parser.add_argument(
        "--anthropic-buffered-request-timeout-seconds",
        type=_positive_int_arg,
        default=600,
        help=(
            "Anthropic buffered read timeout in seconds for non-streaming "
            "message and batch paths (default: 600)"
        ),
    )
    parser.add_argument(
        "--vertex-api-url",
        help=f"Custom Vertex AI regional API URL (default: {DEFAULT_VERTEX_API_URL})",
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
        "--bedrock-api-url",
        help=(
            "Custom Bedrock InvokeModel upstream for the /model/{id}/invoke "
            "passthrough routes — point at a re-signing gateway, not raw AWS "
            "(env: BEDROCK_TARGET_API_URL)"
        ),
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
        "--keepalive-expiry",
        type=float,
        default=90.0,
        help="Seconds an idle upstream keep-alive connection is kept open (default: 90)",
    )
    parser.add_argument(
        "--no-http2",
        action="store_true",
        help="Disable HTTP/2 (enabled by default for better throughput)",
    )
    parser.add_argument(
        "--http-proxy",
        help=("HTTP proxy URL for upstream provider requests only (env: HEADROOM_HTTP_PROXY)"),
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
    parser.add_argument(
        "--compress-user-messages",
        action="store_true",
        help=(
            "Opt in to compressing `user` role messages. Default is off because "
            "user content is typically the subject of the request and is part of "
            "the prefix-cache zone. Enable this for OpenAI/Azure chat workloads "
            "where the bulk of input lives in user messages (pasted content, "
            "RAG context, etc.) and you want the router to consider it eligible. "
            "Also settable via HEADROOM_COMPRESS_USER_MESSAGES=1."
        ),
    )
    parser.add_argument(
        "--disable-kompress",
        action="store_true",
        help=(
            "Disable Kompress ML compression while keeping structural compression enabled. "
            "Also settable via HEADROOM_DISABLE_KOMPRESS=1."
        ),
    )
    parser.add_argument(
        "--disable-kompress-fallback",
        action="store_true",
        help=(
            "With --disable-kompress, route fall-through content to PASSTHROUGH instead of "
            "the default KOMPRESS fallback (restores legacy --disable-kompress behaviour). "
            "Also settable via HEADROOM_DISABLE_KOMPRESS_FALLBACK=1."
        ),
    )
    parser.add_argument(
        "--disable-kompress-anthropic",
        dest="disable_kompress_anthropic",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Disable Kompress for the Anthropic pipeline only, overriding --disable-kompress. "
            "Also settable via HEADROOM_DISABLE_KOMPRESS_ANTHROPIC=1."
        ),
    )
    parser.add_argument(
        "--enable-kompress-anthropic",
        dest="disable_kompress_anthropic",
        action="store_const",
        const=False,
        help="Force-enable Kompress for the Anthropic pipeline, overriding --disable-kompress.",
    )
    parser.add_argument(
        "--disable-kompress-openai",
        dest="disable_kompress_openai",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Disable Kompress for the OpenAI/Codex pipeline only, overriding --disable-kompress. "
            "Also settable via HEADROOM_DISABLE_KOMPRESS_OPENAI=1."
        ),
    )
    parser.add_argument(
        "--enable-kompress-openai",
        dest="disable_kompress_openai",
        action="store_const",
        const=False,
        help="Force-enable Kompress for the OpenAI/Codex pipeline, overriding --disable-kompress.",
    )
    parser.add_argument(
        "--force-kompress-all",
        action="store_true",
        help=(
            "Route ALL compressible content through Kompress (kompress-v2-base), "
            "bypassing per-type compressor selection. Tool ground truth "
            "(Read/Glob/... and reversibility-gated output) is still never touched. "
            "Also settable via HEADROOM_FORCE_KOMPRESS_ALL=1."
        ),
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help=(
            "No-CCR lossless mode: compress LOG/SEARCH/DIFF tool outputs with "
            "format-native lossless compaction (and marker-free SmartCrusher) "
            "without emitting any CCR retrieval marker, so no MCP retrieve tool "
            "is needed. Also settable via HEADROOM_LOSSLESS=1."
        ),
    )
    parser.add_argument(
        "--compress-passthrough",
        action="store_true",
        help=(
            "Also compress requests that fall through to the catch-all "
            "passthrough handler (custom proxy paths not matched by a built-in "
            "API route, e.g. `/api/codex-proxy/<key>/v1/responses` behind "
            "another proxy). Applies to OpenAI Responses-shaped bodies (paths "
            "ending in `/responses`). Off by default; also settable via "
            "HEADROOM_COMPRESS_PASSTHROUGH=1."
        ),
    )
    parser.add_argument(
        "--exclude-tools",
        default=None,
        help="Comma-separated tool names whose output is never compressed, "
        "merged with the built-in defaults (e.g., WebSearch,WebFetch). "
        "Entries may use glob patterns, e.g. 'mcp__*' to exclude every MCP tool. "
        "Also settable via HEADROOM_EXCLUDE_TOOLS env var.",
    )
    parser.add_argument(
        "--protect-tool-results",
        default=None,
        help="Comma-separated tool names whose results are never lossy-compressed, "
        "merged with the built-in defaults (e.g. Bash,WebFetch). "
        "Also settable via HEADROOM_PROTECT_TOOL_RESULTS env var.",
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
    parser.add_argument(
        "--budget-estimated-basis",
        choices=["count", "ignore", "block"],
        default=None,
        help=(
            "What spend booked from Headroom's own token estimate (provider returned no "
            "usage breakdown) does to the budget: count it (default), ignore it, or block "
            "requests rather than enforce the limit on an estimate. "
            "Env: HEADROOM_BUDGET_ESTIMATED_BASIS."
        ),
    )

    # Logging
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--log-messages", action="store_true", help="Log full messages")

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
    env_code_aware = _get_env_bool("HEADROOM_CODE_AWARE_ENABLED", True)
    env_optimize = _get_env_bool("HEADROOM_OPTIMIZE", True)
    env_cache = _get_env_bool("HEADROOM_CACHE_ENABLED", True)
    env_rate_limit = _get_env_bool("HEADROOM_RATE_LIMIT_ENABLED", True)

    # Determine settings: CLI flags override env vars
    # --no-X explicitly disables, --X explicitly enables, neither uses env var
    code_aware_enabled = (
        env_code_aware
        if not (args.code_aware or args.no_code_aware)
        else (args.code_aware or not args.no_code_aware)
    )
    optimize = env_optimize if not args.no_optimize else False
    cache_enabled = env_cache if not args.no_cache else False
    rate_limit_enabled = env_rate_limit if not args.no_rate_limit else False
    disable_kompress = args.disable_kompress or _get_env_bool("HEADROOM_DISABLE_KOMPRESS", False)
    disable_kompress_fallback = args.disable_kompress_fallback or _get_env_bool(
        "HEADROOM_DISABLE_KOMPRESS_FALLBACK", False
    )
    disable_kompress_anthropic = (
        args.disable_kompress_anthropic
        if args.disable_kompress_anthropic is not None
        else _get_env_optional_bool("HEADROOM_DISABLE_KOMPRESS_ANTHROPIC")
    )
    disable_kompress_openai = (
        args.disable_kompress_openai
        if args.disable_kompress_openai is not None
        else _get_env_optional_bool("HEADROOM_DISABLE_KOMPRESS_OPENAI")
    )
    force_kompress_all = args.force_kompress_all or _get_env_bool(
        "HEADROOM_FORCE_KOMPRESS_ALL", False
    )
    lossless = getattr(args, "lossless", False) or _get_env_bool("HEADROOM_LOSSLESS", False)
    compress_passthrough = args.compress_passthrough or _get_env_bool(
        "HEADROOM_COMPRESS_PASSTHROUGH", False
    )

    # Set OpenRouter API key from CLI if provided
    if hasattr(args, "openrouter_api_key") and args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key

    # Parse per-tool compression profiles from CLI and env var
    tool_profiles = _parse_tool_profiles(args.tool_profile)
    # Parse extra never-compress tools from CLI and env var
    exclude_tools = _parse_exclude_tools(args.exclude_tools)
    protect_tool_results = _parse_csv_tools(
        args.protect_tool_results or os.environ.get("HEADROOM_PROTECT_TOOL_RESULTS")
    )

    from headroom.rollout import resolve_rollout

    rollout = resolve_rollout()
    config = ProxyConfig(
        rollout=rollout,
        host=_get_env_str("HEADROOM_HOST", args.host),
        port=_get_env_int("HEADROOM_PORT", args.port),
        openai_api_url=_get_env_str("OPENAI_TARGET_API_URL", args.openai_api_url),
        anthropic_api_url=_get_env_str("ANTHROPIC_TARGET_API_URL", args.anthropic_api_url),
        anthropic_buffered_request_timeout_seconds=_get_env_int(
            "HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS",
            args.anthropic_buffered_request_timeout_seconds,
            min_value=1,
        ),
        # Env-only here, like connect/request: this parser exposes no timeout
        # flag but the buffered one.
        write_timeout_seconds=_get_env_int(
            "HEADROOM_WRITE_TIMEOUT_SECONDS",
            150,
            min_value=1,
        ),
        vertex_api_url=_get_env_str("VERTEX_TARGET_API_URL", args.vertex_api_url),
        # Backend settings
        backend=_get_env_str("HEADROOM_BACKEND", args.backend),  # type: ignore[arg-type]
        bedrock_region=_get_env_str("HEADROOM_BEDROCK_REGION", args.bedrock_region),
        bedrock_profile=args.bedrock_profile or os.environ.get("AWS_PROFILE"),
        bedrock_api_url=_get_env_str("BEDROCK_TARGET_API_URL", args.bedrock_api_url),
        anyllm_provider=_get_env_str("HEADROOM_ANYLLM_PROVIDER", args.anyllm_provider),
        optimize=optimize,
        min_tokens_to_crush=_get_env_int("HEADROOM_MIN_TOKENS", args.min_tokens),
        max_items_after_crush=_get_env_int("HEADROOM_MAX_ITEMS", args.max_items),
        smart_crusher_with_compaction=(
            _get_env_bool("HEADROOM_SMART_CRUSHER_COMPACTION", False)
            if "HEADROOM_SMART_CRUSHER_COMPACTION" in os.environ
            else None
        ),
        cache_enabled=cache_enabled,
        cache_ttl_seconds=_get_env_int("HEADROOM_CACHE_TTL", args.cache_ttl),
        rate_limit_enabled=rate_limit_enabled,
        rate_limit_requests_per_minute=_get_env_int("HEADROOM_RPM", args.rpm),
        rate_limit_tokens_per_minute=_get_env_int("HEADROOM_TPM", args.tpm),
        budget_limit_usd=args.budget,
        budget_period=args.budget_period,
        budget_estimated_basis=cast(
            Literal["count", "ignore", "block"],
            resolve_estimated_basis_policy(args.budget_estimated_basis, os.environ),
        ),
        log_file=_get_env_str("HEADROOM_LOG_FILE", args.log_file)
        if args.log_file
        else os.environ.get("HEADROOM_LOG_FILE"),
        log_full_messages=args.log_messages or _get_env_bool("HEADROOM_LOG_MESSAGES", False),
        code_aware_enabled=code_aware_enabled,
        disable_kompress=disable_kompress,
        disable_kompress_fallback=disable_kompress_fallback,
        disable_kompress_anthropic=disable_kompress_anthropic,
        disable_kompress_openai=disable_kompress_openai,
        force_kompress_all=force_kompress_all,
        lossless=lossless,
        compress_passthrough=compress_passthrough,
        # Connection pool settings
        max_connections=_get_env_int("HEADROOM_MAX_CONNECTIONS", args.max_connections),
        max_keepalive_connections=_get_env_int("HEADROOM_MAX_KEEPALIVE", args.max_keepalive),
        keepalive_expiry=_get_env_float("HEADROOM_KEEPALIVE_EXPIRY", args.keepalive_expiry),
        http2=not args.no_http2 and _get_env_bool("HEADROOM_HTTP2", True),
        http_proxy=_get_env_str("HEADROOM_HTTP_PROXY", args.http_proxy or "") or None,
        read_maturation=rollout.is_enabled("read_maturation"),
        read_maturation_quiesce_turns=_get_env_int("HEADROOM_READ_MATURATION_QUIESCE_TURNS", 5),
        read_maturation_max_hold_turns=_get_env_int("HEADROOM_READ_MATURATION_MAX_HOLD_TURNS", 25),
        read_maturation_min_size_bytes=_get_env_int(
            "HEADROOM_READ_MATURATION_MIN_SIZE_BYTES", 2048
        ),
        tool_profiles=tool_profiles if tool_profiles else None,
        exclude_tools=exclude_tools if exclude_tools else None,
        protect_tool_results=frozenset(protect_tool_results)
        if protect_tool_results
        else frozenset(),
        mode=normalize_proxy_mode(_get_env_str("HEADROOM_MODE", PROXY_MODE_CACHE)),
        compress_user_messages=args.compress_user_messages
        or _get_env_bool("HEADROOM_COMPRESS_USER_MESSAGES", False),
        savings_profile=os.environ.get("HEADROOM_SAVINGS_PROFILE") or "coding",
        # Default 0.4 keep-ratio so the Kompress text (prose/code) path compresses
        # meaningfully out of the box; HEADROOM_TARGET_RATIO overrides.
        target_ratio=(
            float(os.environ["HEADROOM_TARGET_RATIO"])
            if os.environ.get("HEADROOM_TARGET_RATIO")
            else 0.4
        ),
        compress_system_messages=(
            _get_env_bool("HEADROOM_COMPRESS_SYSTEM_MESSAGES", False)
            if "HEADROOM_COMPRESS_SYSTEM_MESSAGES" in os.environ
            else None
        ),
        protect_recent=(
            int(os.environ["HEADROOM_PROTECT_RECENT"])
            if os.environ.get("HEADROOM_PROTECT_RECENT")
            else None
        ),
        protect_analysis_context=(
            _get_env_bool("HEADROOM_PROTECT_ANALYSIS_CONTEXT", False)
            if "HEADROOM_PROTECT_ANALYSIS_CONTEXT" in os.environ
            else None
        ),
        accuracy_guard=os.environ.get("HEADROOM_ACCURACY_GUARD") or None,
    )

    # Get worker and concurrency settings
    workers = _get_env_int("HEADROOM_WORKERS", args.workers)
    limit_concurrency = _get_env_int("HEADROOM_LIMIT_CONCURRENCY", args.limit_concurrency)

    run_server(config, workers=workers, limit_concurrency=limit_concurrency)
