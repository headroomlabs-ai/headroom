"""Canonical filesystem contract for Headroom.

This module defines the single source of truth for where Headroom reads and
writes files. It introduces two canonical roots:

* ``HEADROOM_CONFIG_DIR`` -- read-mostly configuration (defaults to
  ``~/.headroom/config``). Holds model catalogs, plugin settings, and other
  configuration that users or admins edit.
* ``HEADROOM_WORKSPACE_DIR`` -- read-write state (defaults to ``~/.headroom``).
  Holds runtime caches, telemetry outputs, logs, savings history, memory
  databases, and anything else that the running proxy/CLI writes to.

Precedence for every per-resource helper is::

    explicit argument > per-resource env var > derived from canonical root >
    default

Adding the canonical root env vars is strictly additive: every existing
per-resource override (``HEADROOM_SAVINGS_PATH``, ``HEADROOM_TOIN_PATH``,
``HEADROOM_SUBSCRIPTION_STATE_PATH``, ``HEADROOM_MODEL_LIMITS``, ...)
continues to take precedence with identical semantics.

Implementation notes:

* Helpers return ``Path`` (never ``str``). Callers that need a string cast
  at the callsite.
* Helpers are pure -- they never call ``mkdir``. Use the ``ensure_*``
  variants when the caller needs the directory to exist.
* No caching. Every call re-reads the environment so that ``monkeypatch``
  in tests works without extra hoops.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical env var names
# ---------------------------------------------------------------------------

HEADROOM_CONFIG_DIR_ENV = "HEADROOM_CONFIG_DIR"
HEADROOM_WORKSPACE_DIR_ENV = "HEADROOM_WORKSPACE_DIR"

# ---------------------------------------------------------------------------
# Legacy per-resource env vars (kept for backward compatibility)
# ---------------------------------------------------------------------------

HEADROOM_SAVINGS_PATH_ENV = "HEADROOM_SAVINGS_PATH"
HEADROOM_SAVINGS_EVENTS_PATH_ENV = "HEADROOM_SAVINGS_EVENTS_PATH"
HEADROOM_TOIN_PATH_ENV = "HEADROOM_TOIN_PATH"
HEADROOM_SUBSCRIPTION_STATE_PATH_ENV = "HEADROOM_SUBSCRIPTION_STATE_PATH"

# ---------------------------------------------------------------------------
# Default sub-path fragments
# ---------------------------------------------------------------------------

_WORKSPACE_DIR_DEFAULT = ".headroom"
_CONFIG_DIR_DEFAULT_SUFFIX = "config"

# Resource file/sub-dir names (kept here so nothing else has to hardcode them)
_SAVINGS_FILE = "proxy_savings.json"
_TOIN_FILE = "toin.json"
_MODELS_FILE = "models.json"
_SUBSCRIPTION_FILE = "subscription_state.json"
_MEMORY_DB_FILE = "memory.db"
_MEMORIES_DIR = "memories"
_LICENSE_CACHE_FILE = "license_cache.json"
_SESSION_STATS_FILE = "session_stats.jsonl"
_SAVINGS_EVENTS_FILE = "savings_events.jsonl"
_SYNC_STATE_FILE = "sync_state.json"
_BRIDGE_STATE_FILE = "bridge_state.json"
_LOGS_DIR = "logs"
_PROXY_LOG_FILE = "proxy.log"
_DEBUG_400_DIR = "debug_400"
_CODEX_WIRE_DEBUG_DIR = "codex_wire"
_BIN_DIR = "bin"
_PROXY_CLIENTS_DIR = "clients"
_RTK_UNIX = "rtk"
_RTK_WIN = "rtk.exe"
_LEAN_CTX_UNIX = "lean-ctx"
_LEAN_CTX_WIN = "lean-ctx.exe"
_DEPLOY_DIR = "deploy"
_PLUGINS_DIR = "plugins"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _env(name: str) -> str:
    """Return a trimmed environment value, or ``""`` when unset/blank."""

    return os.environ.get(name, "").strip()


# ---------------------------------------------------------------------------
# Process-wide stateless flag
# ---------------------------------------------------------------------------
# Stateless mode forbids writes to the workspace. Many persisters are
# module-level singletons reached without a config object, so the proxy records
# the mode here once at startup and writers consult ``process_is_stateless()``.

_PROCESS_STATELESS: bool = False


def set_process_stateless(value: bool) -> None:
    """Record process-wide stateless mode (set once at proxy startup)."""

    global _PROCESS_STATELESS
    _PROCESS_STATELESS = bool(value)


def process_is_stateless() -> bool:
    """True when the process must not write to the workspace.

    True if ``set_process_stateless(True)`` was called OR the ``HEADROOM_STATELESS``
    environment variable is set, so non-proxy entrypoints honor it too.
    """

    if _PROCESS_STATELESS:
        return True
    return _env("HEADROOM_STATELESS").lower() in ("1", "true", "yes", "on")


def _resolve(explicit: str | os.PathLike[str] | None, env_var: str, derived: Path) -> Path:
    """Apply the standard precedence: explicit > env > derived.

    ``explicit`` and the env-var value are both passed through ``expanduser()``
    so that callers can pass ``"~/foo/bar"`` and have it resolve naturally.
    """

    if explicit is not None and str(explicit) != "":
        return Path(explicit).expanduser()
    env_value = _env(env_var)
    if env_value:
        return Path(env_value).expanduser()
    return derived


# ---------------------------------------------------------------------------
# Canonical roots
# ---------------------------------------------------------------------------


def workspace_dir() -> Path:
    """Return the workspace (read-write state) root directory.

    Resolution order:

    1. ``$HEADROOM_WORKSPACE_DIR`` (trimmed, tilde-expanded) if set.
    2. ``~/.headroom`` otherwise.
    """

    env_value = _env(HEADROOM_WORKSPACE_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / _WORKSPACE_DIR_DEFAULT


def config_dir() -> Path:
    """Return the config (read-mostly) root directory.

    Resolution order:

    1. ``$HEADROOM_CONFIG_DIR`` (trimmed, tilde-expanded) if set.
    2. ``$HEADROOM_WORKSPACE_DIR/config`` when the workspace env var is set
       so that a single override relocates both roots coherently.
    3. ``~/.headroom/config`` otherwise.
    """

    env_value = _env(HEADROOM_CONFIG_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    workspace_env = _env(HEADROOM_WORKSPACE_DIR_ENV)
    if workspace_env:
        return Path(workspace_env).expanduser() / _CONFIG_DIR_DEFAULT_SUFFIX
    return Path.home() / _WORKSPACE_DIR_DEFAULT / _CONFIG_DIR_DEFAULT_SUFFIX


def ensure_workspace_dir() -> Path:
    """Return :func:`workspace_dir`, creating it if it does not yet exist."""

    path = workspace_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_config_dir() -> Path:
    """Return :func:`config_dir`, creating it if it does not yet exist."""

    path = config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Per-resource helpers -- workspace bucket
# ---------------------------------------------------------------------------


def savings_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the path for the proxy savings JSON ledger."""

    return _resolve(
        explicit,
        HEADROOM_SAVINGS_PATH_ENV,
        workspace_dir() / _SAVINGS_FILE,
    )


def toin_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the path for the TOIN telemetry JSON file.

    TOIN is classified as workspace state because it is actively written by
    the running proxy (it's a compression feedback loop). The default stays
    ``~/.headroom/toin.json`` to preserve byte-for-byte backward compat.
    """

    return _resolve(
        explicit,
        HEADROOM_TOIN_PATH_ENV,
        workspace_dir() / _TOIN_FILE,
    )


def subscription_state_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the path for the subscription tracker state JSON."""

    return _resolve(
        explicit,
        HEADROOM_SUBSCRIPTION_STATE_PATH_ENV,
        workspace_dir() / _SUBSCRIPTION_FILE,
    )


def memory_db_path() -> Path:
    """Return the default memory SQLite path."""

    return workspace_dir() / _MEMORY_DB_FILE


def native_memory_dir() -> Path:
    """Return the default native-memory directory."""

    return workspace_dir() / _MEMORIES_DIR


def license_cache_path() -> Path:
    """Return the path for the cached license envelope."""

    return workspace_dir() / _LICENSE_CACHE_FILE


def session_stats_path() -> Path:
    """Return the path for the per-session stats JSONL file."""

    return workspace_dir() / _SESSION_STATS_FILE


def savings_events_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the path for the durable append-only savings event ledger.

    Unlike :func:`session_stats_path` (pruned to a short rolling window), this
    file accrues one line per compression across proxy restarts and concurrent
    MCP processes, and is the source of truth for ``headroom savings``.
    """

    return _resolve(
        explicit,
        HEADROOM_SAVINGS_EVENTS_PATH_ENV,
        workspace_dir() / _SAVINGS_EVENTS_FILE,
    )


def sync_state_path() -> Path:
    """Return the path for memory sync state."""

    return workspace_dir() / _SYNC_STATE_FILE


def bridge_state_path() -> Path:
    """Return the path for the memory bridge state."""

    return workspace_dir() / _BRIDGE_STATE_FILE


def log_dir() -> Path:
    """Return the directory for Headroom log files."""

    return workspace_dir() / _LOGS_DIR


def proxy_log_path() -> Path:
    """Return the path for the proxy log file."""

    return log_dir() / _PROXY_LOG_FILE


def debug_400_dir() -> Path:
    """Return the directory used to stash HTTP 400 debug payloads."""

    return log_dir() / _DEBUG_400_DIR


def codex_wire_debug_dir() -> Path:
    """Return the directory used for opt-in Codex wire debug captures."""

    return log_dir() / _CODEX_WIRE_DEBUG_DIR


def bin_dir() -> Path:
    """Return the directory where Headroom ships vendored binaries."""

    return workspace_dir() / _BIN_DIR


def proxy_clients_dir(port: int) -> Path:
    """Per-port dir of live wrap-client markers (one file per client PID)."""

    return workspace_dir() / _PROXY_CLIENTS_DIR / str(port)


def rtk_path() -> Path:
    """Return the path to the vendored ``rtk`` binary."""

    name = _RTK_WIN if os.name == "nt" else _RTK_UNIX
    return bin_dir() / name


def lean_ctx_path() -> Path:
    """Return the path to the vendored ``lean-ctx`` binary."""

    name = _LEAN_CTX_WIN if os.name == "nt" else _LEAN_CTX_UNIX
    return bin_dir() / name


def deploy_root() -> Path:
    """Return the root directory for persistent deployment profiles."""

    return workspace_dir() / _DEPLOY_DIR


def beacon_lock_path(port: int) -> Path:
    """Return the per-port proxy beacon lock file path."""

    return workspace_dir() / f".beacon_lock_{int(port)}"


# ---------------------------------------------------------------------------
# Per-resource helpers -- config bucket
# ---------------------------------------------------------------------------


def models_config_path() -> Path:
    """Return the default path for the models catalog JSON.

    Note: the ``HEADROOM_MODEL_LIMITS`` env var is a *content* override
    (it can hold either a JSON string or a filesystem path) and is handled
    by the provider layer. This helper only returns the default file
    location and deliberately ignores ``HEADROOM_MODEL_LIMITS``.
    """

    return config_dir() / _MODELS_FILE


# ---------------------------------------------------------------------------
# Plugin-author entry points
# ---------------------------------------------------------------------------


def plugin_config_dir(plugin_name: str) -> Path:
    """Return the config directory for a named plugin."""

    if not plugin_name or "/" in plugin_name or "\\" in plugin_name:
        raise ValueError(f"invalid plugin name: {plugin_name!r}")
    return config_dir() / _PLUGINS_DIR / plugin_name


def plugin_workspace_dir(plugin_name: str) -> Path:
    """Return the workspace directory for a named plugin."""

    if not plugin_name or "/" in plugin_name or "\\" in plugin_name:
        raise ValueError(f"invalid plugin name: {plugin_name!r}")
    return workspace_dir() / _PLUGINS_DIR / plugin_name


__all__ = [
    "HEADROOM_CONFIG_DIR_ENV",
    "HEADROOM_WORKSPACE_DIR_ENV",
    "HEADROOM_SAVINGS_PATH_ENV",
    "HEADROOM_SAVINGS_EVENTS_PATH_ENV",
    "HEADROOM_TOIN_PATH_ENV",
    "HEADROOM_SUBSCRIPTION_STATE_PATH_ENV",
    "set_process_stateless",
    "process_is_stateless",
    "config_dir",
    "workspace_dir",
    "ensure_config_dir",
    "ensure_workspace_dir",
    "savings_path",
    "toin_path",
    "subscription_state_path",
    "memory_db_path",
    "native_memory_dir",
    "license_cache_path",
    "session_stats_path",
    "savings_events_path",
    "sync_state_path",
    "bridge_state_path",
    "log_dir",
    "proxy_log_path",
    "debug_400_dir",
    "codex_wire_debug_dir",
    "bin_dir",
    "proxy_clients_dir",
    "rtk_path",
    "lean_ctx_path",
    "deploy_root",
    "beacon_lock_path",
    "models_config_path",
    "plugin_config_dir",
    "plugin_workspace_dir",
]
