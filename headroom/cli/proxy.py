"""Proxy server CLI commands."""

import os
import sys
from typing import Any

import click

from headroom.providers.registry import resolve_api_overrides, resolve_api_targets
from headroom.proxy.modes import PROXY_MODE_TOKEN, normalize_proxy_mode

from .main import main


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    envvar="HEADROOM_HOST",
    help="Host to bind to (default: 127.0.0.1, env: HEADROOM_HOST)",
)
@click.option(
    "--port",
    "-p",
    default=8787,
    type=int,
    envvar="HEADROOM_PORT",
    help="Port to bind to (default: 8787, env: HEADROOM_PORT)",
)
@click.option(
    "--workers",
    default=1,
    type=click.IntRange(min=1),
    envvar="HEADROOM_WORKERS",
    help="Number of Uvicorn worker processes (default: 1, env: HEADROOM_WORKERS)",
)
@click.option(
    "--limit-concurrency",
    default=1000,
    type=click.IntRange(min=1),
    envvar="HEADROOM_LIMIT_CONCURRENCY",
    help=(
        "Maximum concurrent connections before Uvicorn returns 503 "
        "(default: 1000, env: HEADROOM_LIMIT_CONCURRENCY)"
    ),
)
@click.option(
    "--max-connections",
    default=500,
    type=click.IntRange(min=1),
    envvar="HEADROOM_MAX_CONNECTIONS",
    help="Maximum upstream HTTP connections (default: 500, env: HEADROOM_MAX_CONNECTIONS)",
)
@click.option(
    "--max-keepalive",
    "max_keepalive_connections",
    default=100,
    type=click.IntRange(min=0),
    envvar="HEADROOM_MAX_KEEPALIVE",
    help="Maximum upstream keep-alive connections (default: 100, env: HEADROOM_MAX_KEEPALIVE)",
)
@click.option(
    "--mode",
    default=None,
    type=click.Choice(
        [
            "token",
            "cache",
            "token_mode",
            "cache_mode",
            "token_savings",
            "cost_savings",
            "token_headroom",
        ]
    ),
    help=(
        "Optimization mode: token (prioritize compression) or cache "
        "(freeze prior turns for prefix-cache stability). "
        "Legacy aliases are accepted. Default: token. Env: HEADROOM_MODE"
    ),
)
@click.option(
    "--intercept-tool-results",
    is_flag=True,
    help=(
        "Opt in to tool_result interceptors (ast-grep Read outliner, etc.). "
        "Off by default while this feature ships."
    ),
)
@click.option("--no-optimize", is_flag=True, help="Disable optimization (passthrough mode)")
@click.option("--no-cache", is_flag=True, help="Disable semantic caching")
@click.option("--no-rate-limit", is_flag=True, help="Disable rate limiting")
@click.option(
    "--proxy-extension",
    "proxy_extension",
    multiple=True,
    envvar="HEADROOM_PROXY_EXTENSIONS",
    help=(
        "Enable a registered proxy extension by entry-point name (opt-in). "
        "Repeat the flag or pass a comma-separated list. Use '*' to enable "
        "every discovered extension. Env: HEADROOM_PROXY_EXTENSIONS."
    ),
)
@click.option(
    "--no-subscription-tracking",
    is_flag=True,
    envvar="HEADROOM_NO_SUBSCRIPTION_TRACKING",
    help=(
        "Disable the Anthropic Claude Code subscription usage poller "
        "(GET /api/oauth/usage). Env: HEADROOM_NO_SUBSCRIPTION_TRACKING."
    ),
)
@click.option(
    "--subscription-poll-interval",
    type=int,
    default=None,
    envvar="HEADROOM_SUBSCRIPTION_POLL_INTERVAL",
    help=(
        "Seconds between Anthropic subscription usage polls (1–3600, default 300). "
        "Lower values give fresher /stats but risk 429s from Anthropic. "
        "Env: HEADROOM_SUBSCRIPTION_POLL_INTERVAL."
    ),
)
@click.option(
    "--retry-max-attempts",
    type=int,
    default=None,
    help="Maximum upstream retry attempts for connect/read/5xx failures (default: 3)",
)
@click.option(
    "--connect-timeout-seconds",
    type=int,
    default=None,
    help="Upstream connection timeout in seconds (default: 10)",
)
@click.option(
    "--anthropic-pre-upstream-concurrency",
    type=int,
    default=None,
    envvar="HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY",
    help=(
        "Cap the number of Anthropic HTTP requests that may run pre-upstream work "
        "(request parse / deep-copy / first compression stage / memory context / upstream connect) "
        "concurrently. Prevents cold-start replay storms from starving /livez and new Codex WS opens. "
        "Default: max(2, min(8, os.cpu_count() or 4)). "
        "Set to 0 or negative to disable (unbounded). "
        "Env: HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY."
    ),
)
@click.option(
    "--anthropic-pre-upstream-acquire-timeout-seconds",
    type=float,
    default=None,
    envvar="HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS",
    help=(
        "Fail-fast timeout for waiting on the Anthropic pre-upstream semaphore "
        "before returning 503 + Retry-After. "
        "Default: 15.0 seconds. "
        "Env: HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS."
    ),
)
@click.option(
    "--anthropic-pre-upstream-memory-context-timeout-seconds",
    type=float,
    default=None,
    envvar="HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS",
    help=(
        "Fail-open timeout for Anthropic memory-context lookup while the request "
        "still holds a pre-upstream slot. "
        "Default: 2.0 seconds. "
        "Env: HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS."
    ),
)
@click.option("--log-file", default=None, help="Path to JSONL log file")
@click.option(
    "--log-messages",
    is_flag=True,
    help="Enable full message logging (request/response content stored for live feed)",
)
@click.option(
    "--budget",
    type=float,
    default=None,
    envvar="HEADROOM_BUDGET",
    help="Daily budget limit in USD (env: HEADROOM_BUDGET)",
)
# Code graph: indexes project + watches files for live reindex via codebase-memory-mcp
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph intelligence (indexes project, watches files for live reindex via codebase-memory-mcp)",
)
# Read lifecycle (ON by default: compresses stale/superseded Read outputs)
@click.option(
    "--no-read-lifecycle",
    is_flag=True,
    help="Disable Read lifecycle management (stale/superseded Read compression)",
)
# Memory System (Multi-Provider Support)
@click.option(
    "--memory",
    is_flag=True,
    help="Enable persistent user memory. Auto-detects provider and uses appropriate tool format. "
    "Set x-headroom-user-id header for per-user memory (defaults to 'default').",
)
@click.option(
    "--memory-db-path",
    default="",
    help="Path to memory database file (default: {cwd}/.headroom/memory.db)",
)
@click.option("--no-memory-tools", is_flag=True, help="Disable automatic memory tool injection")
@click.option(
    "--no-memory-context", is_flag=True, help="Disable automatic memory context injection"
)
@click.option(
    "--memory-top-k",
    type=int,
    default=10,
    help="Number of memories to inject as context (default: 10)",
)
@click.option(
    "--memory-qdrant-url",
    default=None,
    help=(
        "Full Qdrant URL for the qdrant-neo4j backend "
        "(e.g. https://xyz.cloud.qdrant.io:6333). When set, takes precedence over "
        "--memory-qdrant-host/--memory-qdrant-port. "
        "Also reads HEADROOM_QDRANT_URL."
    ),
)
@click.option(
    "--memory-qdrant-host",
    default=None,
    help=(
        "Qdrant host for the qdrant-neo4j backend "
        "(default: localhost, also reads HEADROOM_QDRANT_HOST)"
    ),
)
@click.option(
    "--memory-qdrant-port",
    type=int,
    default=None,
    help=(
        "Qdrant port for the qdrant-neo4j backend (default: 6333, also reads HEADROOM_QDRANT_PORT)"
    ),
)
@click.option(
    "--memory-qdrant-api-key",
    default=None,
    help=("API key for hosted Qdrant (e.g. Qdrant Cloud). Also reads HEADROOM_QDRANT_API_KEY."),
)
# Traffic Learning (live pattern extraction from proxy traffic)
@click.option(
    "--learn",
    is_flag=True,
    help="Enable live traffic learning: extract error→recovery patterns, environment facts, "
    "and user preferences from proxy traffic. Implies --memory. "
    "Learned patterns are saved to agent-native memory files (MEMORY.md, .cursor/rules, AGENTS.md).",
)
@click.option(
    "--no-learn",
    is_flag=True,
    help="Explicitly disable traffic learning even when --memory is set.",
)
@click.option(
    "--min-evidence",
    type=int,
    default=None,
    envvar="HEADROOM_MIN_EVIDENCE",
    help=(
        "Minimum number of times a pattern must be observed before it is "
        "persisted to memory. Higher values reduce one-shot noise at the "
        "cost of slower learning. Default: 5. (env: HEADROOM_MIN_EVIDENCE)"
    ),
)
# Backend configuration
@click.option(
    "--backend",
    default="anthropic",
    help=(
        "API backend: 'anthropic' (direct), 'bedrock' (AWS), 'openrouter' (OpenRouter), "
        "'anyllm' (any-llm), or 'litellm-<provider>' (e.g., litellm-vertex)"
    ),
)
@click.option(
    "--anyllm-provider",
    default="openai",
    help="Provider for any-llm backend: openai, mistral, groq, ollama, etc. (default: openai)",
)
@click.option(
    "--anthropic-api-url",
    default=None,
    help="Custom Anthropic API URL for passthrough endpoints (env: ANTHROPIC_TARGET_API_URL)",
)
@click.option(
    "--openai-api-url",
    default=None,
    help="Custom OpenAI API URL for passthrough endpoints (env: OPENAI_TARGET_API_URL)",
)
@click.option(
    "--gemini-api-url",
    default=None,
    help="Custom Gemini API URL for passthrough endpoints (env: GEMINI_TARGET_API_URL)",
)
@click.option(
    "--cloudcode-api-url",
    default=None,
    help="Custom Cloud Code Assist API URL for compatibility endpoints (env: CLOUDCODE_TARGET_API_URL)",
)
@click.option(
    "--region",
    default="us-west-2",
    help="Cloud region for Bedrock/Vertex/etc (default: us-west-2)",
)
@click.option(
    "--bedrock-region",
    default=None,
    help="(deprecated, use --region) AWS region for Bedrock",
)
@click.option(
    "--bedrock-profile",
    default=None,
    help="AWS profile name for Bedrock (default: use default credentials)",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    help="Disable anonymous usage telemetry (env: HEADROOM_TELEMETRY=off)",
)
@click.option(
    "--stateless",
    is_flag=True,
    help="Disable all filesystem writes — run purely in-memory. "
    "For containerized / read-only / load-balanced deployments. "
    "(env: HEADROOM_STATELESS=true)",
)
@click.pass_context
def proxy(
    ctx: click.Context,
    mode: str | None,
    host: str,
    port: int,
    workers: int,
    limit_concurrency: int,
    max_connections: int,
    max_keepalive_connections: int,
    intercept_tool_results: bool,
    no_optimize: bool,
    no_cache: bool,
    no_rate_limit: bool,
    proxy_extension: tuple[str, ...],
    no_subscription_tracking: bool,
    subscription_poll_interval: int | None,
    retry_max_attempts: int | None,
    connect_timeout_seconds: int | None,
    anthropic_pre_upstream_concurrency: int | None,
    anthropic_pre_upstream_acquire_timeout_seconds: float | None,
    anthropic_pre_upstream_memory_context_timeout_seconds: float | None,
    log_file: str | None,
    log_messages: bool,
    budget: float | None,
    code_graph: bool,
    no_read_lifecycle: bool,
    memory: bool,
    memory_db_path: str,
    no_memory_tools: bool,
    no_memory_context: bool,
    memory_top_k: int,
    memory_qdrant_url: str | None,
    memory_qdrant_host: str | None,
    memory_qdrant_port: int | None,
    memory_qdrant_api_key: str | None,
    learn: bool,
    no_learn: bool,
    min_evidence: int | None,
    backend: str,
    anyllm_provider: str,
    anthropic_api_url: str | None,
    openai_api_url: str | None,
    gemini_api_url: str | None,
    cloudcode_api_url: str | None,
    region: str,
    bedrock_region: str | None,
    bedrock_profile: str | None,
    no_telemetry: bool,
    stateless: bool,
) -> None:
    """Start the optimization proxy server.

    \b
    Examples:
        headroom proxy                    Start proxy on port 8787
        headroom proxy --port 8080        Start proxy on port 8080
        headroom proxy --no-optimize      Passthrough mode (no optimization)

    \b
    Usage with Claude Code:
        ANTHROPIC_BASE_URL=http://localhost:8787 claude

    \b
    Usage with OpenAI-compatible clients:
        OPENAI_BASE_URL=http://localhost:8787/v1 your-app
    """
    # Import here to avoid slow startup
    try:
        from headroom.proxy.server import ProxyConfig, run_server
    except ImportError as e:
        click.echo("Error: Proxy dependencies not installed. Run: pip install headroom[proxy]")
        click.echo(f"Details: {e}")
        raise SystemExit(1) from None

    # Opt-in: turn on tool_result interceptors (ast-grep Read outline, etc.).
    # Only fetch the bundled CLI tool binaries when the feature is enabled —
    # otherwise we'd pay a network round-trip and risk a readonly-FS failure
    # for capabilities the user hasn't asked for. The TransformPipeline reads
    # this env var at construction time.
    if intercept_tool_results:
        from headroom.binaries import ensure_tools

        resolved_tools = ensure_tools()
        critical_tools = ["ast-grep"]
        missing = [t for t in critical_tools if not resolved_tools.get(t)]
        if missing:
            # User explicitly opted in — fail fast rather than silently starting
            # with non-functional interceptors. They can retry with the tool
            # installed, or drop the flag if they want pass-through behavior.
            click.secho(
                f"error: --intercept-tool-results requires tool(s) that could not "
                f"be installed: {missing}. Run `headroom tools doctor` to diagnose, "
                "or omit the flag to start the proxy without interceptors.",
                fg="red",
                err=True,
            )
            sys.exit(1)
        os.environ["HEADROOM_INTERCEPT_ENABLED"] = "1"

    provider_api_overrides = resolve_api_overrides(
        anthropic_api_url=anthropic_api_url,
        openai_api_url=openai_api_url,
        gemini_api_url=gemini_api_url,
        cloudcode_api_url=cloudcode_api_url,
        environ=os.environ,
    )

    # Resolve anyllm provider: env var takes precedence over CLI default (matches argparse path)
    effective_anyllm_provider = os.environ.get("HEADROOM_ANYLLM_PROVIDER") or anyllm_provider

    # Resolve mode: CLI flag > env var > default
    effective_mode: str = normalize_proxy_mode(
        mode or os.environ.get("HEADROOM_MODE") or PROXY_MODE_TOKEN
    )

    # Stateless mode: CLI flag or env var
    is_stateless = stateless or os.environ.get("HEADROOM_STATELESS", "").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    # Telemetry opt-out: --no-telemetry flag sets the env var
    if no_telemetry:
        os.environ["HEADROOM_TELEMETRY"] = "off"

    # Stateless mode: suppress TOIN filesystem persistence
    if is_stateless:
        os.environ["HEADROOM_TOIN_BACKEND"] = "none"

    # License key for managed/enterprise deployments (optional)
    license_key = os.environ.get("HEADROOM_LICENSE_KEY")

    # Qdrant connection for the qdrant-neo4j backend. CLI flags default
    # to None; when omitted we let ProxyConfig's default_factory resolve
    # HEADROOM_QDRANT_* env vars. Explicit CLI values win over env.
    qdrant_overrides: dict[str, Any] = {}
    if memory_qdrant_url is not None:
        qdrant_overrides["memory_qdrant_url"] = memory_qdrant_url
    if memory_qdrant_host is not None:
        qdrant_overrides["memory_qdrant_host"] = memory_qdrant_host
    if memory_qdrant_port is not None:
        qdrant_overrides["memory_qdrant_port"] = memory_qdrant_port
    if memory_qdrant_api_key is not None:
        qdrant_overrides["memory_qdrant_api_key"] = memory_qdrant_api_key

    config = ProxyConfig(
        host=host,
        port=port,
        anthropic_api_url=provider_api_overrides.anthropic,
        openai_api_url=provider_api_overrides.openai,
        gemini_api_url=provider_api_overrides.gemini,
        cloudcode_api_url=provider_api_overrides.cloudcode,
        mode=effective_mode,
        optimize=not no_optimize,
        cache_enabled=not no_cache,
        rate_limit_enabled=not no_rate_limit,
        # Flatten repeat-flag tuple AND any comma-separated values inside it.
        # `--proxy-extension a,b --proxy-extension c` and `HEADROOM_PROXY_EXTENSIONS=a,b,c`
        # both yield ["a", "b", "c"]. None when nothing was supplied.
        proxy_extensions=(
            [part.strip() for chunk in proxy_extension for part in chunk.split(",") if part.strip()]
            or None
        ),
        subscription_tracking_enabled=not no_subscription_tracking,
        subscription_poll_interval_s=(
            subscription_poll_interval if subscription_poll_interval is not None else 300
        ),
        retry_max_attempts=retry_max_attempts if retry_max_attempts is not None else 3,
        connect_timeout_seconds=connect_timeout_seconds
        if connect_timeout_seconds is not None
        else 10,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        log_file=None if is_stateless else log_file,
        log_full_messages=log_messages
        or os.environ.get("HEADROOM_LOG_MESSAGES", "").lower() in ("true", "1", "yes", "on"),
        budget_limit_usd=budget,
        # Code graph: live file watcher for incremental reindexing
        code_graph_watcher=code_graph,
        # Read lifecycle: ON by default (use --no-read-lifecycle to disable)
        read_lifecycle=not no_read_lifecycle,
        # Memory System (Multi-Provider with auto-detection)
        # --learn implies --memory (need backend for storing patterns)
        # Stateless mode disables memory (requires SQLite on disk)
        memory_enabled=False if is_stateless else (memory or (learn and not no_learn)),
        memory_db_path=memory_db_path,
        memory_inject_tools=not no_memory_tools,
        memory_inject_context=not no_memory_context,
        memory_top_k=memory_top_k,
        **qdrant_overrides,
        # Traffic Learning: only with --learn, never with --no-learn
        # Stateless mode disables learning (requires filesystem)
        traffic_learning_enabled=False if is_stateless else (learn and not no_learn),
        traffic_learning_agent_type=os.environ.get("HEADROOM_AGENT_TYPE", "unknown"),
        traffic_learning_min_evidence=min_evidence if min_evidence is not None else 5,
        # Backend (Anthropic direct, Bedrock, LiteLLM, or any-llm)
        backend=backend,
        bedrock_region=bedrock_region or region,
        bedrock_profile=bedrock_profile,
        anyllm_provider=effective_anyllm_provider,
        # License / Usage Reporting (managed/enterprise)
        license_key=license_key,
        # Stateless mode: disable all filesystem writes
        stateless=is_stateless,
        # Unit 4: bounded pre-upstream concurrency on the Anthropic HTTP
        # path. ``None`` -> HeadroomProxy computes ``max(2, min(8,
        # os.cpu_count() or 4))``; ``<= 0`` -> disabled (unbounded).
        # Precedence: CLI > env > auto-compute (click's ``envvar``
        # handles the env-var fallback).
        anthropic_pre_upstream_concurrency=anthropic_pre_upstream_concurrency,
        anthropic_pre_upstream_acquire_timeout_seconds=(
            anthropic_pre_upstream_acquire_timeout_seconds
            if anthropic_pre_upstream_acquire_timeout_seconds is not None
            else 15.0
        ),
        anthropic_pre_upstream_memory_context_timeout_seconds=(
            anthropic_pre_upstream_memory_context_timeout_seconds
            if anthropic_pre_upstream_memory_context_timeout_seconds is not None
            else 2.0
        ),
    )

    memory_status = "DISABLED"
    if config.memory_enabled:
        memory_status = "ENABLED (multi-provider)"

    license_status = "OSS (no license key)"
    if license_key:
        license_status = f"MANAGED (key={license_key[:8]}...)"

    provider_api_targets = resolve_api_targets(config.provider_api_overrides)
    anthropic_url = provider_api_targets.anthropic
    openai_url = provider_api_targets.openai
    cloudcode_url = provider_api_targets.cloudcode
    backend_section = ""

    if config.backend == "anyllm" or config.backend.startswith("anyllm-"):
        # any-llm backend
        backend_section = """
  Set credentials for your provider (e.g., OPENAI_API_KEY, MISTRAL_API_KEY)
  Providers: https://mozilla-ai.github.io/any-llm/providers/
"""
    elif config.backend != "anthropic":
        # LiteLLM backend
        from headroom.backends.litellm import get_provider_config

        provider = config.backend.replace("litellm-", "")
        provider_config = get_provider_config(provider)

        # Build usage instructions from provider config
        env_vars_str = (
            ", ".join(provider_config.env_vars) if provider_config.env_vars else "See docs"
        )
        backend_section = f"""
IMPORTANT for {provider_config.display_name} users:
  1. Set credentials: {env_vars_str}
  2. Set a dummy Anthropic key: ANTHROPIC_API_KEY="sk-ant-dummy"
     (Headroom ignores this - it uses your {provider_config.display_name} credentials)
  3. Set base URL: ANTHROPIC_BASE_URL=http://{config.host}:{config.port}"""
        if provider_config.model_format_hint:
            backend_section += f"\n  4. Use model names: {provider_config.model_format_hint}"
        backend_section += "\n"

    # Build memory section if enabled
    memory_section = ""
    if config.memory_enabled:
        memory_section = f"""
Memory (Multi-Provider):
  - Auto-detects provider from request (Anthropic, OpenAI, Gemini, etc.)
  - Anthropic: Uses native memory tool (memory_20250818) - subscription safe
  - OpenAI/Gemini/Others: Uses function calling format
  - All providers share the same semantic vector store backend
  - Set x-headroom-user-id header for per-user memory (defaults to 'default')
  - Tools: {"ENABLED" if config.memory_inject_tools else "DISABLED"}
  - Context injection: {"ENABLED" if config.memory_inject_context else "DISABLED"}
  - Database: {config.memory_db_path}
"""

    # Stateless mode warning
    stateless_line = ""
    if is_stateless:
        stateless_line = (
            "  Stateless:    YES (no filesystem writes — memory, logs, TOIN disabled)\n"
        )

    from headroom.telemetry.beacon import is_telemetry_enabled

    # Build telemetry section for the startup banner
    if is_telemetry_enabled():
        telemetry_line = (
            "  Telemetry:    ENABLED (anonymous aggregate stats)\n"
            "                Disable: HEADROOM_TELEMETRY=off or headroom proxy --no-telemetry"
        )
    else:
        telemetry_line = "  Telemetry:    DISABLED"

    # Discover proxy extensions (third-party packages registered via the
    # `headroom.proxy_extension` entry-point group). Surfaced in the banner
    # so operators can see what's available + what's currently opted-in.
    # Discovery does NOT run extension code; only the explicitly-enabled
    # set in config.proxy_extensions actually installs.
    try:
        from headroom.proxy.extensions import discover as _discover_extensions

        _ext_available = sorted(name for name, _ in _discover_extensions())
    except Exception:  # noqa: BLE001 — banner must never crash startup
        _ext_available = []
    _ext_enabled = config.proxy_extensions or []
    if not _ext_available:
        extensions_line = "  Extensions:   (none discovered)"
    elif not _ext_enabled:
        extensions_line = (
            f"  Extensions:   discovered={','.join(_ext_available)} "
            f"(opt-in: --proxy-extension <name> or HEADROOM_PROXY_EXTENSIONS=<n>)"
        )
    elif "*" in _ext_enabled:
        extensions_line = f"  Extensions:   ENABLED (wildcard) {','.join(_ext_available)}"
    else:
        extensions_line = (
            f"  Extensions:   ENABLED {','.join(sorted(_ext_enabled))} "
            f"(available: {','.join(_ext_available)})"
        )

    click.echo(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         HEADROOM PROXY                                 ║
║           The Context Optimization Layer for LLM Applications          ║
╚═══════════════════════════════════════════════════════════════════════╝

Starting proxy server...

  URL:          http://{config.host}:{config.port}
  Mode:         {config.mode}
  Optimization: {"ENABLED" if config.optimize else "DISABLED"}
  Caching:      {"ENABLED" if config.cache_enabled else "DISABLED"}
  Rate Limit:   {"ENABLED" if config.rate_limit_enabled else "DISABLED"}
  Memory:       {memory_status}
  License:      {license_status}
{extensions_line}
{stateless_line}{telemetry_line}
{backend_section}
Routing:
  /v1/messages                    → {anthropic_url}
  /v1/chat/completions            → {openai_url}
  /v1/responses                   → {openai_url}  (HTTP + WebSocket)
  /v1internal:streamGenerateContent → {cloudcode_url}

Usage:
  Claude Code:   ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude
  Codex / OpenAI: OPENAI_BASE_URL=http://{config.host}:{config.port}/v1 your-app
{memory_section}
Endpoints:
  GET  /livez      Process liveness
  GET  /readyz     Traffic readiness
  GET  /health     Aggregate health
  GET  /stats      Detailed statistics
  GET  /stats-history Durable compression history + display session
  GET  /metrics    Prometheus metrics

Press Ctrl+C to stop.
""")

    try:
        run_kwargs: dict[str, int] = {}
        if workers != 1:
            run_kwargs["workers"] = workers
        if limit_concurrency != 1000:
            run_kwargs["limit_concurrency"] = limit_concurrency
        run_server(config, **run_kwargs)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
