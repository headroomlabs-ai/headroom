"""File-backed store for a curated subset of Headroom's runtime knobs (mostly ``HEADROOM_*``,

The dashboard settings GUI persists these knobs to ``settings.json`` in the
workspace dir and this module applies them to ``os.environ`` at CLI startup
with ``os.environ.setdefault`` — so an explicit shell export always wins over
the stored file. Precedence: ``export > settings.json > code default``.

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
    tier: str = "advanced"  # "basic" | "advanced" - Settings vs Advanced tab placement


# Curated registry. Env formats verified against each knob's Click option in
# headroom/cli/proxy.py (bools serialize to "1"/"0", which Click's BOOL type and
# the body-resolved HEADROOM_CODE_AWARE_ENABLED reader both accept).
SETTINGS: tuple[SettingField, ...] = (
    # --- Compression ---
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
        "Compression",
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
        "Compression",
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
        "HEADROOM_MEMORY_BACKEND",
        "memory_backend",
        "Memory backend",
        "Memory",
        "enum",
        default="local",
        choices=("local", "qdrant-neo4j"),
        help="Storage backend for agent memory. qdrant-neo4j requires the [memory-stack] extra.",
        tier="advanced",
    ),
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
    """``setdefault`` each stored value into ``os.environ`` (explicit export wins)."""
    for key, value in values.items():
        field = _BY_KEY.get(key)
        if field is None or value is None:
            continue
        os.environ.setdefault(field.env, _serialize(field, value))


def effective_values(stored: dict[str, Any] | None = None) -> dict[str, Any]:
    """The value actually active now for each knob: default ← file ← environ."""
    if stored is None:
        stored = load()
    result: dict[str, Any] = {}
    for field in SETTINGS:
        value = stored[field.key] if field.key in stored else field.default
        env_raw = os.environ.get(field.env)
        if env_raw is not None and env_raw != "":
            try:
                value = _coerce(field, env_raw)
            except (ValueError, TypeError):
                pass  # unparseable env: keep the file/default value
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

    All curated knobs are startup-captured, so every key is restart-required;
    ``needs_restart_keys`` lists them for the UI's "restart to apply" banner.
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
                "type": field.type,
                "choices": list(field.choices),
                "default": field.default,
                "help": field.help,
                "secret": field.secret,
                "manifest_managed": field.manifest_managed,
                "minimum": field.minimum,
                "maximum": field.maximum,
                "tier": field.tier,
                "env_override": bool(os.environ.get(field.env)),
                "value": _mask(field, effective.get(field.key)),
                "stored": _mask(field, stored.get(field.key)),
            }
        )
    groups: list[str] = []
    for field in SETTINGS:
        if field.group not in groups:
            groups.append(field.group)
    return {
        "groups": groups,
        "fields": fields,
        "values": {f["key"]: f["value"] for f in fields},
        "needs_restart_keys": [field.key for field in SETTINGS],
    }
