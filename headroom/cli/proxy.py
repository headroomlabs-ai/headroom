"""Proxy server CLI commands."""

import logging
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import click

from headroom import paths as _paths
from headroom.providers.registry import (
    resolve_api_overrides,
    resolve_api_targets,
    resolve_extra_headers,
)
from headroom.proxy.modes import PROXY_MODE_CACHE, normalize_proxy_mode

from .main import main


def ensure_proxy_dependencies() -> None:
    """Verify optional proxy extras are installed before starting or wrapping."""
    required_modules: list[str] = [
        "fastapi",
        "uvicorn",
        "httpx",
        "openai",
        "mcp",
        "magika",
        "zstandard",
        "websockets",
        "onnxruntime",
        "transformers",
        "watchdog",
    ]
    if sys.implementation.name != "pypy":
        required_modules.append("orjson")

    try:
        for module in required_modules:
            import_module(module)
    except ImportError as e:
        click.secho(
            "Error: Proxy dependencies not installed. Run: pip install headroom-ai[proxy]",
            fg="red",
            err=True,
        )
        click.secho(f"Details: {e}", fg="red", err=True)
        raise SystemExit(1) from None


# ---------------------------------------------------------------------------
# Startup log suppression.
#
# sentence_transformers makes HEAD/GET requests to HuggingFace Hub on every
# worker startup to validate the model manifest.  Each request produces an
# INFO-level httpx record and a WARNING from huggingface_hub about a missing
# HF_TOKEN.  With 8 workers this generates ~50 noisy lines per startup.
#
# Placing the suppression here (module-level in the first CLI module imported)
# ensures it is in place before sentence_transformers, huggingface_hub, or
# httpx are initialised by any downstream import or worker fork.
#
# The env vars silence the warnings.warn() path ("unauthenticated requests"
# message) which bypasses the logging system entirely.
# ---------------------------------------------------------------------------

# Env-var knobs are read by huggingface_hub before its logger hierarchy forms.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Corporate TLS-inspection support (issue #1308). When HEADROOM_TLS_STRICT=0,
# strip OpenSSL's RFC 5280 strict CA-constraint check from urllib3's context
# builder *before* huggingface_hub / requests import and cache it — otherwise
# model downloads (huggingface.co) fail with "Basic Constraints of CA cert not
# marked critical" behind Zscaler/Netskope on Python 3.13+. The proxy's own
# httpx upstream client is handled separately in proxy/server.py via
# build_httpx_verify(). No-op unless the toggle is set.
try:  # pragma: no cover - exercised via integration, not unit-importable cheaply
    from headroom.proxy.ssl_context import apply_global_tls_relaxation as _apply_tls_relax

    _apply_tls_relax()
except Exception:  # never let TLS relaxation wiring break startup
    pass

# Logger-level suppression: httpx HEAD/GET manifest checks + HF advisory msgs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# warnings.warn() path: huggingface_hub emits UserWarning for missing tokens.
warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*huggingface.*token.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# ---------------------------------------------------------------------------


def _get_env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _get_env_bool_optional(name: str) -> bool | None:
    if name not in os.environ:
        return None
    return _get_env_bool(name, False)


# libmalloc reads these before main() runs, so they cannot be set from inside
# the current process — the proxy re-execs itself once to apply them. Without
# them, freed pages from large concurrent request bodies stay resident
# (``vmmap`` shows whole "MALLOC_LARGE (empty)" regions) and long-lived proxy
# RSS only ratchets upward (#2820). Vars the operator already set are left
# untouched; HEADROOM_MALLOC_TUNING=0 disables the re-exec entirely.
_MALLOC_TUNING = {
    "MallocAggressiveMadvise": "1",  # madvise freed pages back to the OS eagerly
    "MallocLargeCache": "0",  # no death-row cache for freed large allocations
}


def _process_is_headroom_cli_entrypoint() -> bool:
    """Is this process the Headroom CLI itself, rather than an embedder?

    ``_reexec_with_malloc_tuning`` rebuilds the command line as
    ``python -m headroom.cli <argv[1:]>``. That is only a faithful
    reconstruction when the process really was started as the Headroom CLI. If
    something else invoked the ``proxy`` command in-process — pytest's
    ``CliRunner``, an embedding application, ``runpy`` — then ``argv[1:]``
    belongs to *that* program, and ``os.execv`` would replace it with a Headroom
    process parsing arguments that were never meant for us.
    """
    argv0 = Path(sys.argv[0] or "")
    if argv0.name in {"headroom", "headroom.exe"}:
        return True
    # `python -m headroom.cli` sets argv[0] to .../headroom/cli/__main__.py.
    return argv0.parts[-3:] == ("headroom", "cli", "__main__.py")


def _reexec_with_malloc_tuning() -> None:
    if sys.platform != "darwin":
        return
    if not _get_env_bool("HEADROOM_MALLOC_TUNING", True):
        return
    if os.environ.get("_HEADROOM_MALLOC_TUNED") == "1":
        return
    if not _process_is_headroom_cli_entrypoint():
        return
    missing = {k: v for k, v in _MALLOC_TUNING.items() if k not in os.environ}
    # Set the loop guard before the re-exec so the replacement process (which
    # inherits this environment) skips this path instead of re-execing forever.
    os.environ["_HEADROOM_MALLOC_TUNED"] = "1"
    if not missing:
        return
    os.environ.update(missing)
    os.execv(sys.executable, [sys.executable, "-m", "headroom.cli", *sys.argv[1:]])


def _get_env_int_optional(name: str) -> int | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        raise click.ClickException(f"{name} must be an integer, got {val!r}") from None


def _get_env_int(name: str, default: int) -> int:
    """Return the env var as an int, or ``default`` only when it is unset.

    Unlike ``_get_env_int_optional(name) or default``, an explicit ``0`` is
    preserved — ``0`` is a legitimate value (e.g. ``HEADROOM_MIN_TOKENS=0``
    means "crush every item") and ``0 or default`` would silently discard it.
    Mirrors ``headroom.proxy.server._get_env_int``.
    """
    value = _get_env_int_optional(name)
    return default if value is None else value


def _get_env_float_optional(name: str) -> float | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        raise click.ClickException(f"{name} must be a number, got {val!r}") from None


@main.command()
@click.option(
    "--port",
    "-p",
    default=8787,
    type=click.IntRange(1, 65535),
    envvar="HEADROOM_PORT",
    help="Proxy port (default: 8787, env: HEADROOM_PORT)",
)
@click.option("--no-open", is_flag=True, help="Print the URL instead of opening a browser")
def dashboard(port: int, no_open: bool) -> None:
    """Open the Headroom savings dashboard in your browser.

    Requires a running proxy (start one with `headroom proxy` or `headroom wrap ...`).
    """
    import webbrowser

    url = f"http://127.0.0.1:{port}/dashboard"
    click.echo(f"  Dashboard: {url}")
    if not no_open:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — headless/no browser: URL already printed
            pass


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
    type=click.IntRange(1, 65535),
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
    "--http2/--no-http2",
    "http2",
    default=True,
    envvar="HEADROOM_HTTP2",
    help=(
        "Use HTTP/2 to upstream providers (default: on, env: HEADROOM_HTTP2). "
        "Disable to force HTTP/1.1, which avoids shared-connection TLS corruption "
        "(SSLV3_ALERT_BAD_RECORD_MAC) when many concurrent streams are cancelled."
    ),
)
@click.option(
    "--http-proxy",
    default=None,
    envvar="HEADROOM_HTTP_PROXY",
    help=(
        "HTTP proxy URL for upstream provider requests only "
        "(HTTPS uses CONNECT; env: HEADROOM_HTTP_PROXY)."
    ),
)
@click.option(
    "--keepalive-expiry",
    "keepalive_expiry",
    default=90.0,
    type=click.FloatRange(min=0),
    envvar="HEADROOM_KEEPALIVE_EXPIRY",
    help="Seconds an idle upstream keep-alive connection is kept open (default: 90, env: HEADROOM_KEEPALIVE_EXPIRY)",
)
@click.option(
    "--mode",
    default=None,
    metavar="[token|cache]",
    type=click.Choice(
        # Canonical modes first; legacy aliases follow for backward compatibility.
        # `metavar` above hides the alias clutter from --help; users see "[token|cache]"
        # while internal callers passing "token_mode"/"cost_savings"/etc. still validate.
        [
            "token",
            "cache",
            "token_mode",
            "cache_mode",
            "token_savings",
            "cost_savings",
            "token_headroom",
        ],
        case_sensitive=False,
    ),
    help=(
        "Optimization mode (default: cache).\n"
        "  token  — prioritize compression; prior turns may be rewritten for max savings.\n"
        "  cache  — freeze prior turns to maximise provider prefix-cache hit rate.\n"
        "Legacy aliases (token_mode, token_savings, token_headroom, cache_mode, "
        "cost_savings) are still accepted. Env: HEADROOM_MODE."
    ),
)
@click.option(
    "--target-ratio",
    type=float,
    default=None,
    show_default=True,
    envvar="HEADROOM_TARGET_RATIO",
    help=(
        "Override Kompress keep-ratio for text (prose/code) compression — lower is "
        "more aggressive (e.g. 0.4 keeps ~40% of tokens). Unset (default): let "
        "Kompress decide via its own importance threshold (conservative). "
        "Env: HEADROOM_TARGET_RATIO."
    ),
)
@click.option(
    "--intercept-tool-results",
    is_flag=True,
    help=(
        "Opt in to tool_result interceptors (ast-grep Read outliner, etc.). "
        "Requires HEADROOM_ROLLOUT_CHANNEL=canary (or dev)."
    ),
)
@click.option("--no-optimize", is_flag=True, help="Disable optimization (passthrough mode)")
@click.option("--no-cache", is_flag=True, help="Disable semantic caching")
@click.option("--no-rate-limit", is_flag=True, help="Disable rate limiting")
@click.option(
    "--protect-tool-results",
    default=None,
    envvar="HEADROOM_PROTECT_TOOL_RESULTS",
    help=(
        "Comma-separated tool names whose results are never lossy-compressed, "
        "merged with the built-in defaults (e.g. Bash,WebFetch). "
        "In token mode, this also resets protect_recent_reads_fraction "
        "from 0.3 (only recent ~30% of results protected) to 0.0 (all "
        "results protected indefinitely), which prevents older Read/Glob/"
        "Grep/Write/Edit tool results from being silently compressed. "
        "Env: HEADROOM_PROTECT_TOOL_RESULTS."
    ),
)
@click.option(
    "--rpm",
    default=None,
    type=click.IntRange(min=1),
    envvar="HEADROOM_RPM",
    help="Max requests per minute. Env: HEADROOM_RPM. Default: 60.",
)
@click.option(
    "--tpm",
    default=None,
    type=click.IntRange(min=1),
    envvar="HEADROOM_TPM",
    help="Max tokens per minute. Env: HEADROOM_TPM. Default: 100000.",
)
@click.option(
    "--no-ccr",
    is_flag=True,
    envvar="HEADROOM_NO_CCR",
    help=(
        "Disable CCR entirely: no retrieval markers in compressed content AND no "
        "headroom_retrieve tool injected. Lossy compression with no recovery path "
        "(maximum savings; also right for streaming / non-MCP clients that can't "
        "resolve an injected tool). Env: HEADROOM_NO_CCR."
    ),
)
@click.option(
    "--lossless",
    is_flag=True,
    envvar="HEADROOM_LOSSLESS",
    help=(
        "No-CCR lossless mode: compress tool outputs with format-native lossless "
        "compaction (and marker-free SmartCrusher) without emitting any CCR "
        "retrieval marker, so no MCP retrieve tool is needed. Env: HEADROOM_LOSSLESS=1."
    ),
)
@click.option(
    "--ccr-inline-resolve",
    is_flag=True,
    envvar="HEADROOM_CCR_INLINE_RESOLVE",
    help=(
        "Resolve <<ccr:...>> markers inline on the response path instead of "
        "relying on the model to call headroom_retrieve. For callers with no "
        "tool-call round-trip to redeem a marker (e.g. Headroom running as a "
        "LiteLLM guardrail/proxy hop, see issue #2509). Applies to non-streaming "
        "responses only. Off by default. "
        "Env: HEADROOM_CCR_INLINE_RESOLVE."
    ),
)
@click.option(
    "--no-ccr-proactive-expansion",
    is_flag=True,
    envvar="HEADROOM_NO_CCR_PROACTIVE_EXPANSION",
    help=(
        "Disable proactive expansion of previously compressed content. "
        "Env: HEADROOM_NO_CCR_PROACTIVE_EXPANSION."
    ),
)
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
    "--compressor",
    "compressor",
    multiple=True,
    envvar="HEADROOM_COMPRESSORS",
    help=(
        "Restrict the active built-in compressors to the named set (opt-in). "
        "Repeat the flag or pass a comma-separated list; recognized names are "
        "smart_crusher, kompress, code_aware, search, log, tabular, config, "
        "html, image. Unselected built-ins are disabled; '*' selects all. "
        "Omit to keep every compressor enabled (default). "
        "Env: HEADROOM_COMPRESSORS."
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
    type=click.IntRange(min=1, max=3600),
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
    type=click.IntRange(min=1, max=10),
    default=None,
    envvar="HEADROOM_RETRY_MAX_ATTEMPTS",
    help=(
        "Maximum upstream retry attempts for connect/read/5xx failures (1–10, default: 3). "
        "Env: HEADROOM_RETRY_MAX_ATTEMPTS."
    ),
)
@click.option(
    "--retry-base-delay-ms",
    type=click.IntRange(min=0),
    default=None,
    envvar="HEADROOM_RETRY_BASE_DELAY_MS",
    help=(
        "Initial upstream retry delay in milliseconds (minimum: 0, default: 1000). "
        "Env: HEADROOM_RETRY_BASE_DELAY_MS."
    ),
)
@click.option(
    "--retry-max-delay-ms",
    type=click.IntRange(min=0),
    default=None,
    envvar="HEADROOM_RETRY_MAX_DELAY_MS",
    help=(
        "Maximum upstream retry delay in milliseconds (minimum: 0, default: 30000). "
        "Env: HEADROOM_RETRY_MAX_DELAY_MS."
    ),
)
@click.option(
    "--request-timeout-seconds",
    type=int,
    default=None,
    envvar="HEADROOM_REQUEST_TIMEOUT",
    help=(
        "Request timeout in seconds (default: 300). "
        "Useful for slow providers (eg local). "
        "Env: HEADROOM_REQUEST_TIMEOUT."
    ),
)
@click.option(
    "--connect-timeout-seconds",
    type=click.IntRange(min=1, max=300),
    default=None,
    envvar="HEADROOM_CONNECT_TIMEOUT_SECONDS",
    help=(
        "Upstream connection timeout in seconds (1–300, default: 10). "
        "Env: HEADROOM_CONNECT_TIMEOUT_SECONDS."
    ),
)
@click.option(
    "--write-timeout-seconds",
    type=click.IntRange(min=1),
    default=None,
    envvar="HEADROOM_WRITE_TIMEOUT_SECONDS",
    help=(
        "Seconds the upstream send may take before it is abandoned (default: 150). "
        "On HTTP/1.1 this bounds the whole request body, so raise it if you push "
        "large bodies over a slow link. Lower it to fail over a dead pooled "
        "connection faster; --connect-timeout-seconds only guards a fresh connect. "
        "Env: HEADROOM_WRITE_TIMEOUT_SECONDS."
    ),
)
@click.option(
    "--anthropic-buffered-request-timeout-seconds",
    type=click.IntRange(min=1),
    default=None,
    envvar="HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS",
    help=(
        "Buffered Anthropic read timeout in seconds for non-streaming "
        "message and batch paths (default: 600). "
        "Env: HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS."
    ),
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
        "before failing open to passthrough compression. "
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
@click.option(
    "--compression-max-workers",
    type=int,
    default=None,
    envvar="HEADROOM_COMPRESSION_MAX_WORKERS",
    help=(
        "Bound the dedicated compression threadpool (CPU-bound Kompress work). "
        "Default (unset): cpu_count or 1. Lower it to reduce CPU "
        "oversubscription under concurrent sessions; a value < 1 is clamped to 1. "
        "Env: HEADROOM_COMPRESSION_MAX_WORKERS."
    ),
)
@click.option(
    "--log-file",
    default=None,
    envvar="HEADROOM_LOG_FILE",
    help=(
        "Path to write request/response logs as JSONL. "
        "Each line is a JSON object with fields: timestamp, request_id, model, "
        "tokens_before, tokens_after, latency_ms, etc. "
        "Disabled in --stateless mode. Env: HEADROOM_LOG_FILE."
    ),
)
@click.option(
    "--log-messages",
    is_flag=True,
    envvar="HEADROOM_LOG_MESSAGES",
    help=(
        "Enable full message logging: request/response content is stored in the log file "
        "and served on the live feed endpoint. WARNING: may log sensitive data. "
        "Env: HEADROOM_LOG_MESSAGES."
    ),
)
@click.option(
    "--codex-wire-debug",
    is_flag=True,
    help="Enable local Codex wire snapshots and matching proxy.log frame traces.",
)
@click.option(
    "--codex-wire-debug-dir",
    default=None,
    help=(
        "Directory for Codex wire snapshots (default: "
        "~/.headroom/logs/codex_wire or workspace .headroom/logs/codex_wire)."
    ),
)
@click.option(
    "--budget",
    type=click.FloatRange(min=0.0),
    default=None,
    envvar="HEADROOM_BUDGET",
    help=(
        "Budget limit in USD per --budget-period. Requests are rejected with 429 "
        "once the limit is reached. Env: HEADROOM_BUDGET."
    ),
)
@click.option(
    "--budget-period",
    type=click.Choice(["hourly", "daily", "monthly"]),
    default="daily",
    envvar="HEADROOM_BUDGET_PERIOD",
    help=(
        "Period the --budget limit applies to. Hourly resets on a rolling hour, "
        "daily at local midnight, monthly on the 1st. Default: daily. "
        "Env: HEADROOM_BUDGET_PERIOD."
    ),
)
@click.option(
    "--budget-estimated-basis",
    type=click.Choice(["count", "ignore", "block"]),
    default="count",
    envvar="HEADROOM_BUDGET_ESTIMATED_BASIS",
    help=(
        "What to do with spend booked from Headroom's own token estimate, which "
        "happens when a provider response carries no input-token breakdown. "
        "count: it consumes the budget like measured spend (default). "
        "ignore: only provider-reported spend consumes the budget. "
        "block: refuse requests rather than enforce a hard limit on an estimate. "
        "Env: HEADROOM_BUDGET_ESTIMATED_BASIS."
    ),
)
# Code-aware compression (AST-based, requires `pip install headroom-ai[code]`).
# Pair of flags so users can override the env-var default in either direction.
# We resolve HEADROOM_CODE_AWARE_ENABLED in the body (not via Click's envvar=),
# because Click's envvar handling for paired bool flags is brittle in older
# Click versions.
@click.option(
    "--code-aware/--no-code-aware",
    "code_aware_flag",
    default=None,
    help=(
        "Enable/disable AST-based code compression. Requires the optional "
        "tree-sitter dependency: pip install headroom-ai[code]. "
        "Default: disabled. Env: HEADROOM_CODE_AWARE_ENABLED=1 to enable."
    ),
)
@click.option(
    "--disable-kompress",
    is_flag=True,
    envvar="HEADROOM_DISABLE_KOMPRESS",
    help=(
        "Disable Kompress ML compression while keeping structural compression enabled. "
        "Env: HEADROOM_DISABLE_KOMPRESS=1."
    ),
)
@click.option(
    "--disable-kompress-fallback",
    is_flag=True,
    envvar="HEADROOM_DISABLE_KOMPRESS_FALLBACK",
    help=(
        "With --disable-kompress, route fall-through content to PASSTHROUGH instead of "
        "the default KOMPRESS fallback (restores legacy --disable-kompress behaviour). "
        "Env: HEADROOM_DISABLE_KOMPRESS_FALLBACK=1."
    ),
)
@click.option(
    "--disable-kompress-anthropic/--enable-kompress-anthropic",
    "disable_kompress_anthropic",
    default=None,
    envvar="HEADROOM_DISABLE_KOMPRESS_ANTHROPIC",
    help=(
        "Disable (or --enable-) Kompress for the Anthropic pipeline only, overriding "
        "--disable-kompress. Env: HEADROOM_DISABLE_KOMPRESS_ANTHROPIC=1."
    ),
)
@click.option(
    "--disable-kompress-openai/--enable-kompress-openai",
    "disable_kompress_openai",
    default=None,
    envvar="HEADROOM_DISABLE_KOMPRESS_OPENAI",
    help=(
        "Disable (or --enable-) Kompress for the OpenAI/Codex pipeline only, overriding "
        "--disable-kompress. Env: HEADROOM_DISABLE_KOMPRESS_OPENAI=1."
    ),
)
# Code graph: indexes project + watches files for live reindex via codebase-memory-mcp.
# Only useful when the proxy is launched from a project root — it indexes the
# current working directory.
@click.option(
    "--code-graph",
    is_flag=True,
    help=(
        "Enable code graph intelligence: indexes the current working directory "
        "and watches files for live reindex via codebase-memory-mcp. Only useful "
        "when the proxy is launched from a project root."
    ),
)
# Read lifecycle (ON by default: compresses stale/superseded Read outputs)
@click.option(
    "--no-read-lifecycle",
    is_flag=True,
    help="Disable Read lifecycle management (stale/superseded Read compression)",
)
# Read maturation (Mechanism B) — experimental, OFF by default
@click.option(
    "--read-maturation",
    is_flag=True,
    envvar="HEADROOM_READ_MATURATION",
    help=(
        "EXPERIMENTAL: activity-based read maturation — hold fresh Reads "
        "out of the provider prefix cache and compress them once their "
        "file quiesces. Requires HEADROOM_ROLLOUT_CHANNEL=beta (or dev); "
        "env: HEADROOM_READ_MATURATION=1."
    ),
)
@click.option(
    "--read-maturation-quiesce-turns",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    envvar="HEADROOM_READ_MATURATION_QUIESCE_TURNS",
    help="Read maturation: mature a held Read once its file is quiet this many assistant turns.",
)
@click.option(
    "--read-maturation-max-hold-turns",
    type=click.IntRange(min=1),
    default=25,
    show_default=True,
    envvar="HEADROOM_READ_MATURATION_MAX_HOLD_TURNS",
    help="Read maturation: force-mature a Read held this many turns even if its file stays active.",
)
@click.option(
    "--read-maturation-min-size-bytes",
    type=click.IntRange(min=0),
    default=2048,
    show_default=True,
    envvar="HEADROOM_READ_MATURATION_MIN_SIZE_BYTES",
    help="Read maturation: only hold/mature Read outputs at least this many bytes.",
)
# Memory System (Multi-Provider Support)
@click.option(
    "--memory",
    is_flag=True,
    help=(
        "Enable persistent memory. Auto-detects provider and uses appropriate tool format. "
        "By default (--memory-storage=project) each workspace gets its own DB so memories "
        "from unrelated projects can never bleed in (GH #462). Override scoping with "
        "x-headroom-user-id and/or x-headroom-project-id / x-headroom-cwd request headers."
    ),
)
@click.option(
    "--memory-db-path",
    default="",
    envvar="HEADROOM_MEMORY_DB_PATH",
    help=(
        "Path to the legacy single-file memory DB (used in --memory-storage=global, "
        "and as the seed for the project-mode storage root). "
        "Default: {cwd}/.headroom/memory.db. Env: HEADROOM_MEMORY_DB_PATH."
    ),
)
@click.option(
    "--memory-storage",
    type=click.Choice(["project", "user", "global"], case_sensitive=False),
    default="project",
    show_default=True,
    help=(
        "Memory partitioning strategy. project (default): one SQLite DB per resolved "
        "workspace under <db_path_dir>/memories/projects/<basename>-<hash>/memory.db — "
        "no cross-project bleed. user: one DB per x-headroom-user-id. global: a single "
        "shared DB (pre-fix behaviour; --memory-db-path file is reused so existing "
        "memories remain reachable)."
    ),
)
@click.option(
    "--memory-project-root",
    default="",
    envvar="HEADROOM_MEMORY_PROJECT_ROOT",
    help=(
        "Override the project root used for --memory-storage=project. Useful when the "
        "client doesn't put a cwd in the system prompt or you want to force a specific "
        "workspace. Takes effect after the x-headroom-project-id and x-headroom-cwd "
        "headers. Env: HEADROOM_MEMORY_PROJECT_ROOT."
    ),
)
@click.option(
    "--no-memory-tools",
    is_flag=True,
    envvar="HEADROOM_NO_MEMORY_TOOLS",
    help=(
        "Disable automatic injection of memory_save/memory_search tools into requests. "
        "Env: HEADROOM_NO_MEMORY_TOOLS."
    ),
)
@click.option(
    "--no-memory-context",
    is_flag=True,
    envvar="HEADROOM_NO_MEMORY_CONTEXT",
    help=(
        "Disable automatic injection of relevant past memories into the system prompt. "
        "Env: HEADROOM_NO_MEMORY_CONTEXT."
    ),
)
@click.option(
    "--memory-top-k",
    type=click.IntRange(min=1, max=100),
    default=10,
    envvar="HEADROOM_MEMORY_TOP_K",
    help=(
        "Number of semantically-relevant memories to inject as context (1–100, default: 10). "
        "Env: HEADROOM_MEMORY_TOP_K."
    ),
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
    type=click.IntRange(1, 65535),
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
    type=click.IntRange(min=1),
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
    envvar="HEADROOM_BACKEND",
    help=(
        "API backend: 'anthropic' (direct), 'bedrock' (AWS), 'openrouter' (OpenRouter), "
        "'anyllm' (any-llm), or 'litellm-<provider>' (e.g., litellm-vertex). "
        "Env: HEADROOM_BACKEND."
    ),
)
@click.option(
    "--anyllm-provider",
    default="openai",
    envvar="HEADROOM_ANYLLM_PROVIDER",
    help=(
        "Provider for any-llm backend: openai, mistral, groq, ollama, etc. (default: openai). "
        "Env: HEADROOM_ANYLLM_PROVIDER."
    ),
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
    "--provider-name",
    default=None,
    help=(
        "Display name for the OpenAI-compatible upstream shown on the dashboard "
        "(e.g. 'OpenRouter'). Overrides hostname detection from --openai-api-url. "
        "Internal routing and pricing are unaffected."
    ),
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
    "--vertex-api-url",
    default=None,
    help=("Custom Vertex AI regional API URL for publisher endpoints (env: VERTEX_TARGET_API_URL)"),
)
@click.option(
    "--region",
    default="us-west-2",
    envvar="HEADROOM_REGION",
    help="Cloud region for Bedrock/Vertex/etc (default: us-west-2). Env: HEADROOM_REGION.",
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
    "--bedrock-api-url",
    default=None,
    help=(
        "Custom Bedrock InvokeModel upstream for the /model/{id}/invoke "
        "passthrough routes. Point at a re-signing gateway (LiteLLM, "
        "LocalStack), NOT raw AWS — rewriting the body breaks SigV4. "
        "(env: BEDROCK_TARGET_API_URL)"
    ),
)
@click.option(
    "--telemetry",
    is_flag=True,
    help="Opt in to anonymous usage telemetry — off by default (env: HEADROOM_TELEMETRY=on)",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    help="Force anonymous usage telemetry off (already the default; env: HEADROOM_TELEMETRY=off)",
)
@click.option(
    "--stateless",
    is_flag=True,
    help="Disable all filesystem writes — run purely in-memory. "
    "For containerized / read-only / load-balanced deployments. "
    "(env: HEADROOM_STATELESS=true)",
)
@click.option(
    "--embedding-server/--no-embedding-server",
    default=False,
    help="Run a dedicated embedding server sidecar (Option E). "
    "Shares a single ONNX embedder + HNSW index across all worker processes, "
    "saving ~600 MB RSS. Default: disabled (opt-in for testing). "
    "(env: HEADROOM_EMBEDDING_SERVER=true)",
)
@click.option(
    "--embedding-server-socket",
    default=None,
    help="Unix socket path for the embedding server sidecar. "
    "Default: /tmp/headroom-embed-{port}.sock. "
    "(env: HEADROOM_EMBEDDING_SERVER_SOCKET)",
)
@click.option(
    "--anthropic-extra-headers",
    default=None,
    help=(
        "JSON object of extra headers merged into (and overriding) headers forwarded to "
        'the Anthropic endpoint, e.g. \'{"Api-Key": "..."}\' (env: ANTHROPIC_TARGET_API_HEADERS)'
    ),
)
@click.option(
    "--openai-extra-headers",
    default=None,
    help=(
        "JSON object of extra headers merged into (and overriding) headers forwarded to "
        "the OpenAI endpoint (env: OPENAI_TARGET_API_HEADERS)"
    ),
)
@click.pass_context
def proxy(
    ctx: click.Context,
    mode: str | None,
    target_ratio: float | None,
    host: str,
    port: int,
    workers: int,
    limit_concurrency: int,
    max_connections: int,
    max_keepalive_connections: int,
    keepalive_expiry: float,
    http2: bool,
    http_proxy: str | None,
    intercept_tool_results: bool,
    no_optimize: bool,
    no_cache: bool,
    no_rate_limit: bool,
    protect_tool_results: str | None,
    rpm: int | None,
    tpm: int | None,
    no_ccr: bool,
    lossless: bool,
    ccr_inline_resolve: bool,
    no_ccr_proactive_expansion: bool,
    proxy_extension: tuple[str, ...],
    compressor: tuple[str, ...],
    no_subscription_tracking: bool,
    subscription_poll_interval: int | None,
    retry_max_attempts: int | None,
    retry_base_delay_ms: int | None,
    retry_max_delay_ms: int | None,
    request_timeout_seconds: int | None,
    connect_timeout_seconds: int | None,
    write_timeout_seconds: int | None,
    anthropic_buffered_request_timeout_seconds: int | None,
    anthropic_pre_upstream_concurrency: int | None,
    anthropic_pre_upstream_acquire_timeout_seconds: float | None,
    anthropic_pre_upstream_memory_context_timeout_seconds: float | None,
    compression_max_workers: int | None,
    log_file: str | None,
    log_messages: bool,
    codex_wire_debug: bool,
    codex_wire_debug_dir: str | None,
    budget: float | None,
    budget_period: str,
    budget_estimated_basis: str,
    code_aware_flag: bool | None,
    disable_kompress: bool,
    disable_kompress_fallback: bool,
    disable_kompress_anthropic: bool | None,
    disable_kompress_openai: bool | None,
    code_graph: bool,
    no_read_lifecycle: bool,
    read_maturation: bool,
    read_maturation_quiesce_turns: int,
    read_maturation_max_hold_turns: int,
    read_maturation_min_size_bytes: int,
    memory: bool,
    memory_db_path: str,
    memory_storage: str,
    memory_project_root: str,
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
    anthropic_extra_headers: str | None,
    openai_extra_headers: str | None,
    openai_api_url: str | None,
    provider_name: str | None,
    gemini_api_url: str | None,
    cloudcode_api_url: str | None,
    vertex_api_url: str | None,
    region: str,
    bedrock_region: str | None,
    bedrock_profile: str | None,
    bedrock_api_url: str | None,
    telemetry: bool,
    no_telemetry: bool,
    stateless: bool,
    embedding_server: bool,
    embedding_server_socket: str | None,
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
    _reexec_with_malloc_tuning()
    ensure_proxy_dependencies()

    # Import here to avoid slow startup
    from headroom.proxy.server import (
        ProxyConfig,
        _parse_csv_tools,
        _parse_exclude_tools,
        _parse_tool_profiles,
        run_server,
    )

    # Warn if --learn and --no-learn are both set (--no-learn wins, per docstring)
    if learn and no_learn:
        click.secho(
            "Warning: both --learn and --no-learn were specified; --no-learn takes precedence "
            "and traffic learning will be disabled.",
            fg="yellow",
            err=True,
        )

    # Warn on contradictory / no-op flag combinations. The resolved value still
    # applies; the warning just prevents a silently-ignored flag.
    if no_rate_limit and (rpm is not None or tpm is not None):
        click.secho(
            "Warning: --rpm/--tpm have no effect because --no-rate-limit disables rate limiting.",
            fg="yellow",
            err=True,
        )
    if no_optimize and target_ratio is not None:
        click.secho(
            "Warning: --target-ratio has no effect because --no-optimize disables compression.",
            fg="yellow",
            err=True,
        )
    if telemetry and no_telemetry:
        click.secho(
            "Warning: both --telemetry and --no-telemetry were specified; --no-telemetry "
            "takes precedence and telemetry will be disabled.",
            fg="yellow",
            err=True,
        )

    # Resolve rollout inputs once before constructing any rollout-managed
    # behavior. The immutable snapshot is injected into ProxyConfig and is also
    # what /stats later exposes.
    from headroom.rollout import resolve_rollout

    rollout_requests = []
    if intercept_tool_results:
        rollout_requests.append("tool_result_interceptors")
    if read_maturation:
        rollout_requests.append("read_maturation")
    rollout_snapshot = resolve_rollout(os.environ, requested=rollout_requests)

    if read_maturation and not rollout_snapshot.is_enabled("read_maturation"):
        click.secho(
            "error: --read-maturation is not available in the current rollout channel "
            f"({rollout_snapshot.channel.value}). Set HEADROOM_ROLLOUT_CHANNEL=beta "
            "(or dev), or use HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES=1 for an "
            "emergency override.",
            fg="red",
            err=True,
        )
        sys.exit(1)

    # Opt-in: turn on tool_result interceptors (ast-grep Read outline, etc.).
    # Only fetch the bundled CLI tool binaries when the feature is enabled —
    # otherwise we'd pay a network round-trip and risk a readonly-FS failure
    # for capabilities the user hasn't asked for. The TransformPipeline reads
    # the resolved snapshot says it is active.
    if intercept_tool_results:
        if not rollout_snapshot.is_enabled("tool_result_interceptors"):
            click.secho(
                "error: --intercept-tool-results is not available in the current "
                f"rollout channel ({rollout_snapshot.channel.value}). Set "
                "HEADROOM_ROLLOUT_CHANNEL=canary to dogfood it, or use "
                "HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES=1 for emergency override.",
                fg="red",
                err=True,
            )
            sys.exit(1)

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

    try:
        resolved_anthropic_extra_headers = resolve_extra_headers(
            anthropic_extra_headers, "ANTHROPIC_TARGET_API_HEADERS"
        )
        resolved_openai_extra_headers = resolve_extra_headers(
            openai_extra_headers, "OPENAI_TARGET_API_HEADERS"
        )
    except ValueError as exc:
        click.secho(f"error: {exc}", fg="red", err=True)
        sys.exit(1)

    provider_api_overrides = resolve_api_overrides(
        anthropic_api_url=anthropic_api_url,
        openai_api_url=openai_api_url,
        gemini_api_url=gemini_api_url,
        cloudcode_api_url=cloudcode_api_url,
        vertex_api_url=vertex_api_url,
        environ=os.environ,
    )

    # Resolve anyllm provider. An explicit --anyllm-provider flag always wins;
    # otherwise honor HEADROOM_ANYLLM_PROVIDER, which the settings store may
    # have exported into os.environ after Click parsed the option (so the
    # already-parsed param can't see it).
    _anyllm_source = click.get_current_context().get_parameter_source("anyllm_provider")
    if _anyllm_source is click.core.ParameterSource.COMMANDLINE:
        effective_anyllm_provider = anyllm_provider
    else:
        effective_anyllm_provider = os.environ.get("HEADROOM_ANYLLM_PROVIDER") or anyllm_provider

    # Resolve mode: CLI flag > env var > default. Default is CACHE (Headroom's
    # coding posture): delta-only compression at ~0 prefix-cache busts.
    effective_mode: str = normalize_proxy_mode(
        mode or os.environ.get("HEADROOM_MODE") or PROXY_MODE_CACHE
    )

    # Stateless mode: CLI flag or env var
    is_stateless = stateless or os.environ.get("HEADROOM_STATELESS", "").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    # Telemetry is opt-in (off by default). --telemetry opts in; --no-telemetry
    # forces it off. If both are passed, the explicit opt-out wins (fail-closed).
    if telemetry:
        os.environ["HEADROOM_TELEMETRY"] = "on"
    if no_telemetry:
        os.environ["HEADROOM_TELEMETRY"] = "off"

    if codex_wire_debug or codex_wire_debug_dir:
        os.environ["HEADROOM_CODEX_WIRE_DEBUG"] = "1"
        os.environ["HEADROOM_CODEX_WIRE_DEBUG_DIR"] = codex_wire_debug_dir or str(
            _paths.codex_wire_debug_dir()
        )

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
        rollout=rollout_snapshot,
        anthropic_api_url=provider_api_overrides.anthropic,
        anthropic_extra_headers=resolved_anthropic_extra_headers,
        openai_extra_headers=resolved_openai_extra_headers,
        openai_api_url=provider_api_overrides.openai,
        provider_name=provider_name,
        gemini_api_url=provider_api_overrides.gemini,
        cloudcode_api_url=provider_api_overrides.cloudcode,
        vertex_api_url=provider_api_overrides.vertex,
        mode=effective_mode,
        optimize=not no_optimize,
        cache_enabled=not no_cache,
        rate_limit_enabled=not no_rate_limit,
        rate_limit_requests_per_minute=rpm if rpm is not None else 60,
        rate_limit_tokens_per_minute=tpm if tpm is not None else 100_000,
        compress_user_messages=_get_env_bool("HEADROOM_COMPRESS_USER_MESSAGES", False),
        periodic_malloc_trim_enabled=_get_env_bool(
            "HEADROOM_MALLOC_TRIM", sys.platform == "darwin"
        ),
        malloc_trim_interval_seconds=_get_env_int("HEADROOM_MALLOC_TRIM_INTERVAL_SECONDS", 60),
        min_tokens_to_crush=_get_env_int("HEADROOM_MIN_TOKENS", 500),
        max_items_after_crush=_get_env_int("HEADROOM_MAX_ITEMS", 50),
        exclude_tools=_parse_exclude_tools(None) or None,
        protect_tool_results=frozenset(_parse_csv_tools(protect_tool_results))
        if protect_tool_results
        else frozenset(),
        tool_profiles=_parse_tool_profiles([]) or None,
        smart_crusher_with_compaction=_get_env_bool_optional("HEADROOM_SMART_CRUSHER_COMPACTION"),
        savings_profile=os.environ.get("HEADROOM_SAVINGS_PROFILE") or "coding",
        target_ratio=target_ratio,
        compress_system_messages=_get_env_bool_optional("HEADROOM_COMPRESS_SYSTEM_MESSAGES"),
        protect_recent=_get_env_int_optional("HEADROOM_PROTECT_RECENT"),
        protect_analysis_context=_get_env_bool_optional("HEADROOM_PROTECT_ANALYSIS_CONTEXT"),
        accuracy_guard=os.environ.get("HEADROOM_ACCURACY_GUARD") or None,
        # CCR opt-out: --no-ccr disables every half at once — markers in
        # content, the injected retrieve tool, AND server-side response
        # handling. Markers without a tool, or a tool without markers, are
        # useless, so it is a single switch. Default keeps CCR fully on.
        #
        # Response handling has to be part of it. The buffered stream:false
        # path keys off ``headroom_retrieve`` being present in the *request's*
        # tools, and a client can advertise that tool on its own — the bundled
        # OpenCode plugin registers it unconditionally. So with response
        # handling left on, `--no-ccr` silently kept flipping streaming turns
        # to buffered whenever history still held a redeemable marker, and the
        # documented escape hatch for the CCR buffered-stream bugs did nothing
        # for exactly the clients told to use it (#3082).
        ccr_inject_tool=not no_ccr,
        ccr_inject_marker=not no_ccr,
        ccr_handle_responses=not no_ccr,
        ccr_resolve_markers_inline=ccr_inline_resolve,
        lossless=lossless,
        ccr_proactive_expansion=not no_ccr_proactive_expansion,
        # Flatten repeat-flag tuple AND any comma-separated values inside it.
        # `--proxy-extension a,b --proxy-extension c` and `HEADROOM_PROXY_EXTENSIONS=a,b,c`
        # both yield ["a", "b", "c"]. None when nothing was supplied.
        proxy_extensions=(
            [part.strip() for chunk in proxy_extension for part in chunk.split(",") if part.strip()]
            or None
        ),
        # Same flatten-and-split shape as proxy_extensions, but a set: order
        # and duplicates don't matter for a selection. None when nothing was
        # supplied, which leaves every built-in compressor enabled.
        compressors=(
            {part.strip() for chunk in compressor for part in chunk.split(",") if part.strip()}
            or None
        ),
        subscription_tracking_enabled=not no_subscription_tracking,
        subscription_poll_interval_s=(
            subscription_poll_interval if subscription_poll_interval is not None else 300
        ),
        retry_max_attempts=retry_max_attempts if retry_max_attempts is not None else 3,
        retry_base_delay_ms=retry_base_delay_ms if retry_base_delay_ms is not None else 1000,
        retry_max_delay_ms=retry_max_delay_ms if retry_max_delay_ms is not None else 30000,
        request_timeout_seconds=request_timeout_seconds
        if request_timeout_seconds is not None and request_timeout_seconds > 0
        else 300,
        connect_timeout_seconds=connect_timeout_seconds
        if connect_timeout_seconds is not None
        else 10,
        write_timeout_seconds=write_timeout_seconds if write_timeout_seconds is not None else 150,
        anthropic_buffered_request_timeout_seconds=(
            anthropic_buffered_request_timeout_seconds
            if anthropic_buffered_request_timeout_seconds is not None
            else 600
        ),
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=keepalive_expiry,
        http2=http2,
        http_proxy=http_proxy,
        log_file=None if is_stateless else log_file,
        log_full_messages=log_messages
        or os.environ.get("HEADROOM_LOG_MESSAGES", "").lower() in ("true", "1", "yes", "on"),
        budget_limit_usd=budget,
        budget_period=cast(Literal["hourly", "daily", "monthly"], budget_period),
        budget_estimated_basis=cast(Literal["count", "ignore", "block"], budget_estimated_basis),
        # Code-aware compression resolution:
        # 1. Explicit --code-aware / --no-code-aware always wins.
        # 2. Otherwise read HEADROOM_CODE_AWARE_ENABLED (truthy = on).
        # 3. Otherwise default off — matches the prior cli/proxy.py behavior so
        #    existing users see no change unless they opt in.
        # Default ON (coding posture; consistent with the argparse server path).
        # Degrades gracefully to a no-op when tree-sitter isn't installed.
        code_aware_enabled=(
            bool(code_aware_flag)
            if code_aware_flag is not None
            else os.environ.get("HEADROOM_CODE_AWARE_ENABLED", "1").strip().lower()
            in ("true", "1", "yes", "on")
        ),
        disable_kompress=disable_kompress,
        disable_kompress_fallback=disable_kompress_fallback,
        disable_kompress_anthropic=disable_kompress_anthropic,
        disable_kompress_openai=disable_kompress_openai,
        # Optional inbound auth token + air-gap switch (env-driven).
        proxy_token=os.environ.get("HEADROOM_PROXY_TOKEN") or None,
        offline=_get_env_bool("HEADROOM_OFFLINE", False),
        # Code graph: live file watcher for incremental reindexing
        code_graph_watcher=code_graph,
        # Read lifecycle: ON by default (use --no-read-lifecycle to disable)
        read_lifecycle=not no_read_lifecycle,
        # Read maturation (Mechanism B): experimental, OFF by default
        read_maturation=rollout_snapshot.is_enabled("read_maturation"),
        read_maturation_quiesce_turns=read_maturation_quiesce_turns,
        read_maturation_max_hold_turns=read_maturation_max_hold_turns,
        read_maturation_min_size_bytes=read_maturation_min_size_bytes,
        # Memory System (Multi-Provider with auto-detection)
        # --learn implies --memory (need backend for storing patterns)
        # Stateless mode disables memory (requires SQLite on disk)
        memory_enabled=False if is_stateless else (memory or (learn and not no_learn)),
        memory_db_path=memory_db_path,
        memory_storage_mode=cast(Literal["project", "user", "global"], memory_storage.lower()),
        memory_project_root_override=memory_project_root,
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
        # CLI flag > env > unset. Matches the BEDROCK_TARGET_API_URL naming of
        # the sibling *_TARGET_API_URL passthrough overrides.
        bedrock_api_url=bedrock_api_url or os.environ.get("BEDROCK_TARGET_API_URL"),
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
        compression_max_workers=compression_max_workers,
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
    vertex_url = provider_api_targets.vertex
    from headroom.providers.registry import format_backend_usage_section

    backend_section = format_backend_usage_section(
        backend=config.backend,
        host=config.host,
        port=config.port,
    )

    # Build memory section if enabled
    memory_section = ""
    if config.memory_enabled:
        memory_section = f"""
Memory (Multi-Provider):
  - Auto-detects provider from request (Anthropic, OpenAI, Gemini, etc.)
  - Anthropic: Uses native memory tool (memory_20250818) - subscription safe
  - OpenAI/Gemini/Others: Uses function calling format
  - All providers share the same semantic vector store backend
  - Storage mode: {config.memory_storage_mode} (per-project DB by default — set x-headroom-project-id / x-headroom-cwd to override)
  - Tools: {"ENABLED" if config.memory_inject_tools else "DISABLED"}
  - Context injection: {"ENABLED" if config.memory_inject_context else "DISABLED"}
  - Database: {config.memory_db_path} (legacy / global-mode DB)
"""

    # Stateless mode warning
    stateless_line = ""
    if is_stateless:
        stateless_line = (
            "  Stateless:    YES (no filesystem writes — memory, logs, TOIN disabled)\n"
        )

    from headroom.telemetry.beacon import is_telemetry_enabled

    # Build telemetry section for the startup banner. Telemetry is opt-in
    # (off by default); the disabled line surfaces how to opt in.
    if is_telemetry_enabled():
        telemetry_line = (
            "  Telemetry:    ENABLED (anonymous aggregate stats — you opted in)\n"
            "                Disable: HEADROOM_TELEMETRY=off or headroom proxy --no-telemetry"
        )
    else:
        telemetry_line = (
            "  Telemetry:    DISABLED (opt in: HEADROOM_TELEMETRY=on or headroom proxy --telemetry)"
        )

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

    # Security posture line: inbound auth token + air-gap mode, and a loud
    # flag for the open-bind case (non-loopback host with no token).
    from headroom.proxy.loopback_guard import is_loopback_host

    _auth_on = bool(config.proxy_token or os.environ.get("HEADROOM_PROXY_TOKEN"))
    if config.offline:
        _security_status = "OFFLINE (all egress disabled)" + (
            " · inbound token REQUIRED (non-loopback)" if _auth_on else ""
        )
    elif _auth_on:
        _security_status = "inbound token REQUIRED for non-loopback callers"
    elif not is_loopback_host(config.host):
        _security_status = (
            "WARNING non-loopback bind with NO token — /v1/* is UNAUTHENTICATED "
            "(set HEADROOM_PROXY_TOKEN)"
        )
    else:
        _security_status = "loopback-only (no inbound token)"
    security_line = f"  Security:     {_security_status}"

    # Code-aware status line — same logic the inner banner uses, surfaced here
    # so the click-CLI banner is a complete picture (avoids the dual-banner
    # confusion this branch retired).
    from headroom.proxy.server import _get_code_aware_banner_status

    code_aware_line = f"  Code-Aware:   {_get_code_aware_banner_status(config)}"

    # Performance tuning section — only shown when at least one tuning var is active.
    _embed_socket = os.environ.get("HEADROOM_EMBEDDING_SERVER_SOCKET") or (
        embedding_server and (embedding_server_socket or f"/tmp/headroom-embed-{port}.sock")
    )
    _tuning_lines: list[str] = []
    if _embed_socket:
        _tuning_lines.append(f"  Embedding sidecar:       {_embed_socket}")
    if _tuning_lines:
        tuning_section = "\nPerformance Tuning:\n" + "\n".join(_tuning_lines)
    else:
        tuning_section = ""

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
{code_aware_line}
{extensions_line}
{security_line}
{stateless_line}{telemetry_line}
{backend_section}{tuning_section}

Routing:
  /v1/messages                    → {anthropic_url}
  /v1/chat/completions            → {openai_url}
  /v1/responses                   → {openai_url}  (HTTP + WebSocket)
  /v1internal:streamGenerateContent → {cloudcode_url}
  /v1/projects/.../publishers/... → {vertex_url}

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

    # Surface an "update available" notice (reads cache only; no network here).
    # Best-effort: a broken update check must never block proxy startup.
    try:
        from headroom.update_check import format_update_notice

        _update_notice = format_update_notice()
        if _update_notice:
            click.echo(f"\n{_update_notice}\n")
    except Exception:  # noqa: BLE001 — banner must never crash startup
        pass

    # -----------------------------------------------------------------------
    # Option E: start embedding server sidecar if requested
    # -----------------------------------------------------------------------
    _embed_watchdog = None
    if embedding_server:
        _embed_socket = embedding_server_socket or f"/tmp/headroom-embed-{config.port}.sock"
        # Pass socket path to all worker processes via environment variable
        os.environ["HEADROOM_EMBEDDING_SERVER_SOCKET"] = _embed_socket
        click.echo(f"  Embedding server: starting sidecar on {_embed_socket}...")

        import asyncio as _asyncio

        async def _start_embed_watchdog() -> Any:
            # Import lazily inside the guarded coroutine. The sidecar module is
            # optional and may be absent; keeping the import here lets the
            # try/except below fall back to the per-worker embedder instead of
            # crashing the proxy at startup with ModuleNotFoundError.
            from headroom.memory.adapters.watchdog import EmbeddingServerWatchdog

            wd = EmbeddingServerWatchdog(socket_path=_embed_socket)
            await wd.start()
            ok = await wd.wait_until_healthy(timeout=30.0)
            if not ok:
                click.echo(
                    "  WARNING: Embedding server did not become healthy within 30s. "
                    "Memory features may be unavailable.",
                    err=True,
                )
            else:
                click.echo("  Embedding server: ready.")
            return wd

        try:
            _embed_watchdog = _asyncio.run(_start_embed_watchdog())
        except Exception as _exc:
            click.echo(
                f"  WARNING: Failed to start embedding server sidecar: {_exc}. "
                "Falling back to per-worker embedder.",
                err=True,
            )
            os.environ.pop("HEADROOM_EMBEDDING_SERVER_SOCKET", None)

    try:
        run_kwargs: dict[str, Any] = {}
        if workers != 1:
            run_kwargs["workers"] = workers
        if limit_concurrency != 1000:
            run_kwargs["limit_concurrency"] = limit_concurrency
        # Suppress run_server's legacy banner — the click CLI already printed
        # a richer one above. Direct `python -m headroom.proxy.server` keeps
        # the legacy banner via run_server's default.
        run_kwargs["print_banner"] = False
        run_server(config, **run_kwargs)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        raise SystemExit(130) from None
    finally:
        if _embed_watchdog is not None:
            import asyncio as _asyncio2

            _asyncio2.run(_embed_watchdog.stop())
