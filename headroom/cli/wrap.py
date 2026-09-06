"""Wrap CLI commands to run through Headroom proxy.

Usage:
    headroom wrap claude                    # Start proxy + claude
    headroom wrap copilot -- --model ...    # Start proxy + launch GitHub Copilot CLI
    headroom wrap vscode                    # Transparently proxy VS Code Copilot
    headroom wrap vscode-claude             # Transparently proxy VS Code Claude Code
    headroom wrap codex                     # Start proxy + OpenAI Codex CLI
    headroom wrap aider                     # Start proxy + aider
    headroom wrap openclaude                # Start proxy + OpenClaude
    headroom wrap vibe                      # Start proxy + Mistral Vibe
    headroom wrap grok                      # Start proxy + Grok CLI
    headroom wrap cursor                    # Start proxy + print Cursor config instructions
    headroom wrap grok-build                # Start proxy + configure Grok Build
    headroom wrap openclaw                  # Install + configure OpenClaw plugin
    headroom wrap claude --port 9999        # Custom proxy port
    headroom wrap claude -- --model opus    # Pass args to claude
"""

from __future__ import annotations

import errno
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, NamedTuple, cast

from headroom._subprocess import pid_alive, run

# Fix Windows cp1252 encoding — box-drawing characters require UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import click

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from headroom import fsutil
from headroom._version import __version__ as _HEADROOM_VERSION
from headroom._version import normalize_release_version as _normalize_release_version
from headroom.agent_savings import (
    apply_agent_savings_env_defaults,
)
from headroom.cli.proxy import ensure_proxy_dependencies
from headroom.copilot_auth import (
    _API_TOKEN_ENV_VARS,
    _API_TOKEN_EXPIRES_AT_ENV_VAR,
    _COPILOT_OAUTH_TOKEN_ENV_VARS,
    _GENERIC_GITHUB_TOKEN_ENV_VARS,
    _REFRESH_OAUTH_TOKEN_ENV_VAR,
    CopilotSubscriptionTokenResolution,
    has_oauth_auth,
    resolve_client_bearer_token,
    resolve_copilot_api_url,
    resolve_subscription_bearer_token_details,
)
from headroom.providers.aider import build_launch_env as _build_aider_launch_env
from headroom.providers.claude import (
    REMOTE_CONTROL_BASE_URL_ENV,
    TOOL_SEARCH_DEFAULT,
    TOOL_SEARCH_ENV,
    claude_auth_conflict_message,
    claude_auth_conflict_sources,
    claude_user_settings_path,
    configure_vscode_claude_settings,
    detect_claude_code_version,
    remote_control_applies_to_auth,
    remote_control_gate_active,
    remote_control_gate_message,
    remote_control_sibling_gate_note,
    remove_vscode_claude_settings,
    vscode_claude_proxy_url,
)
from headroom.providers.claude import (
    proxy_base_url as _claude_proxy_base_url,
)
from headroom.providers.claude.runtime import TOOL_SEARCH_FOUNDRY_DEFAULT
from headroom.providers.codex import build_launch_env as _build_codex_launch_env
from headroom.providers.codex.install import codex_uses_chatgpt_auth
from headroom.providers.codex.threads import retag_to_headroom, retag_to_native
from headroom.providers.copilot import (
    build_launch_env as _build_copilot_launch_env,
)
from headroom.providers.copilot import (
    configure_vscode_proxy_settings,
    remove_vscode_proxy_settings,
    vscode_proxy_url,
    vscode_settings_path,
)
from headroom.providers.copilot import (
    copilot_model_from_args as _copilot_model_from_args_impl,
)
from headroom.providers.copilot import (
    default_wire_api_for_model as _copilot_default_wire_api_for_model_impl,
)
from headroom.providers.copilot import (
    detect_running_proxy_backend as _copilot_detect_running_proxy_backend,
)
from headroom.providers.copilot import (
    is_auto_model as _is_auto_model,
)
from headroom.providers.copilot import (
    model_configured as _copilot_model_configured_impl,
)
from headroom.providers.copilot import (
    provider_key_source as _copilot_provider_key_source,
)
from headroom.providers.copilot import (
    query_proxy_config as _copilot_query_proxy_config,
)
from headroom.providers.copilot import (
    resolve_provider_type as _copilot_resolve_provider_type,
)
from headroom.providers.copilot import (
    strip_auto_model_args as _strip_auto_model_args,
)
from headroom.providers.copilot import (
    validate_configuration as _validate_copilot_configuration,
)
from headroom.providers.cursor import render_setup_lines as _render_cursor_setup_lines
from headroom.providers.grok import (
    DEFAULT_API_URL as _GROK_DEFAULT_API_URL,
)
from headroom.providers.grok import (
    build_launch_env as _build_grok_launch_env,
)
from headroom.providers.grok_build import render_setup_lines as _render_grok_build_setup_lines
from headroom.providers.grok_build.config import (
    inject_grok_provider_config,
    restore_grok_provider_config,
)
from headroom.providers.kimi import build_launch_env as _build_kimi_launch_env
from headroom.providers.mistral_vibe import build_launch_env as _build_mistral_vibe_launch_env
from headroom.providers.omp import build_launch_env as _build_omp_launch_env
from headroom.providers.omp import inject_models_override as _inject_omp_models_override
from headroom.providers.omp import models_yml_path as _omp_models_yml_path
from headroom.providers.omp import restore_models_override as _restore_omp_models_override
from headroom.providers.openclaw import (
    OPENCLAW_NPM_PACKAGE,
)
from headroom.providers.openclaw import (
    build_plugin_entry as _build_openclaw_plugin_entry_impl,
)
from headroom.providers.openclaw import (
    build_unwrap_entry as _build_openclaw_unwrap_entry_impl,
)
from headroom.providers.openclaw import (
    decode_entry_json as _decode_openclaw_entry_json_impl,
)
from headroom.providers.openclaw import (
    normalize_gateway_provider_ids as _normalize_openclaw_gateway_provider_ids_impl,
)
from headroom.providers.opencode import build_launch_env as _build_opencode_launch_env
from headroom.providers.opencode.config import (
    _MCP_MARKER_END,  # noqa: F401
    _MCP_MARKER_START,
    _PROVIDER_MARKER_END,  # noqa: F401
    _PROVIDER_MARKER_START,
    inject_opencode_provider_config,
    opencode_config_paths,
    snapshot_opencode_config_if_unwrapped,
    strip_opencode_headroom_blocks,
)
from headroom.providers.zcode import (
    detect_upstream as _detect_zcode_upstream,
)
from headroom.providers.zcode import (
    render_setup_lines as _render_zcode_setup_lines,
)
from headroom.providers.zcode import (
    upstream_to_proxy_urls as _zcode_upstream_to_urls,
)
from headroom.proxy.project_context import with_project_prefix as _with_project_prefix

from .main import main

_COPILOT_PROXY_SEED_ENV_VARS = (
    "GITHUB_COPILOT_API_TOKEN",
    _REFRESH_OAUTH_TOKEN_ENV_VAR,
    _API_TOKEN_EXPIRES_AT_ENV_VAR,
)
_COPILOT_SUBSCRIPTION_LAUNCH_SECRET_ENV_VARS = (
    *_API_TOKEN_ENV_VARS,
    _REFRESH_OAUTH_TOKEN_ENV_VAR,
    _API_TOKEN_EXPIRES_AT_ENV_VAR,
    *_COPILOT_OAUTH_TOKEN_ENV_VARS,
    *_GENERIC_GITHUB_TOKEN_ENV_VARS,
)


def _scrub_copilot_proxy_seed_env(env: dict[str, str]) -> None:
    for key in _COPILOT_PROXY_SEED_ENV_VARS:
        env.pop(key, None)


def _scrub_copilot_subscription_launch_env(env: dict[str, str]) -> None:
    for key in _COPILOT_SUBSCRIPTION_LAUNCH_SECRET_ENV_VARS:
        env.pop(key, None)


def _read_text(path: Path) -> str:
    """Read a text file as UTF-8, falling back to the system locale encoding."""
    return fsutil.read_text(path)


def _write_text(path: Path, content: str) -> None:
    """Write a text file as UTF-8 without translating line endings (preserves CRLF)."""
    fsutil.write_text(path, content)


def _read_settings_for_write(path: Path) -> dict[str, Any]:
    """Read a Claude settings file that is about to be mutated, or refuse to write.

    Callers previously fell back to ``{}`` when the file existed but would not
    parse, then wrote that back — turning a hand-edited typo or a transient read
    error into total loss of the user's ``permissions``/``env``/``hooks``. Abort
    instead, mirroring ``mcp_registry.claude._read_json_for_write``: a malformed
    config is the user's to fix, and no Headroom feature is worth erasing it.

    An **empty** file is the one safe exception and is treated as ``{}``: there
    are no settings in it to lose, and refusing would strand the user behind a
    file they cannot see anything wrong with. A zero-byte settings.json is also
    the classic residue of an interrupted non-atomic write (the failure mode
    :func:`headroom.fsutil.write_text` now prevents), so recovering from it is
    exactly right. Anything non-empty that will not parse is treated as data.
    """
    if not path.exists():
        return {}
    try:
        raw = _read_text(path)
    except OSError as exc:
        raise click.ClickException(
            f"could not read {path} ({exc}). Fix or move it, then re-run — "
            "refusing to overwrite it and lose your settings."
        ) from exc
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"{path} is not valid JSON ({exc}). Fix or move it, then re-run — "
            "refusing to overwrite it and lose your settings."
        ) from exc
    if not isinstance(payload, dict):
        raise click.ClickException(
            f"{path} does not contain a JSON object. Fix or move it, then re-run."
        )
    return cast("dict[str, Any]", payload)


def _claude_settings_env(path: Path) -> dict[str, object]:
    """Read a Claude settings env block for preflight validation."""
    env = _read_settings_for_write(path).get("env")
    return dict(env) if isinstance(env, dict) else {}


def _raise_on_claude_auth_conflict(
    *,
    user_settings_path: Path,
    project_settings_path: Path,
    project_local_settings_path: Path,
    environ: dict[str, str],
) -> None:
    """Refuse an auth state Claude Code rejects before mutating wrap state."""
    conflict = claude_auth_conflict_sources(
        (str(user_settings_path), _claude_settings_env(user_settings_path)),
        (str(project_settings_path), _claude_settings_env(project_settings_path)),
        (str(project_local_settings_path), _claude_settings_env(project_local_settings_path)),
        ("shell environment", environ),
    )
    if conflict is not None:
        raise click.ClickException(claude_auth_conflict_message(conflict))


def _append_text(path: Path, content: str) -> None:
    """Append to a text file as UTF-8 without translating line endings."""
    fsutil.append_text(path, content)


_AGENT_SAVINGS_TARGET_AGENTS = {"claude", "codex", "cursor", "grok", "grok_build", "opencode"}
_WRAP_PROXY_TIMEOUT_ENV = "HEADROOM_WRAP_PROXY_TIMEOUT"
_WRAP_PROXY_TIMEOUT_DEFAULT_SECONDS = 45
_WRAP_PROXY_TIMEOUT_ML_DEFAULT_SECONDS = 90
_WRAP_PROXY_TIMEOUT_ML_MODULES = ("torch", "sentence_transformers", "spacy")
# Issue #746: Claude Code disables on-demand tool loading (deferral) when
# ANTHROPIC_BASE_URL is a custom host and ENABLE_TOOL_SEARCH is unset, which
# inflates the local context window by tens of K tokens. Setting the env var
# when we launch Claude Code keeps deferral on. The generic default stays
# "true" for non-Foundry sessions, while Foundry uses a dedicated compatibility
# default of "false" because its upstream does not support the deferred-tool
# shape. The key/defaults are shared with `init` and `install` via the Claude
# provider package to prevent drift.
_TOOL_SEARCH_ENV = TOOL_SEARCH_ENV
_TOOL_SEARCH_DEFAULT = TOOL_SEARCH_DEFAULT
_TOOL_SEARCH_FOUNDRY_DEFAULT = TOOL_SEARCH_FOUNDRY_DEFAULT
_AGENT_SAVINGS_WRAP_AGENTS = {"claude", "codex", "cursor", "grok", "grok_build"}

# 1M context window for `wrap claude` (#1158). Claude Code only sends the
# `context-1m` beta header — unlocking the 1M window for entitled subscription
# users — when the model id carries the `[1m]` suffix. Behind a custom
# ANTHROPIC_BASE_URL (the proxy) its `/model` picker selection does not survive,
# so `--1m` forces the suffix via ANTHROPIC_MODEL on the launched process.
_ANTHROPIC_MODEL_ENV = "ANTHROPIC_MODEL"
_CONTEXT_1M_SUFFIX = "[1m]"
_1M_MODEL_ENV = "HEADROOM_1M_MODEL"
# Fallback model for `--1m` when nothing else selects one (no ANTHROPIC_MODEL,
# no explicit --model). Overridable via HEADROOM_1M_MODEL so it can track new
# Opus releases without a code change and without pinning ANTHROPIC_MODEL
# globally (which would also change non-`--1m` sessions and override Claude
# Code's /model picker). #2937.
_DEFAULT_1M_MODEL = "claude-opus-5"
_OPENCLAUDE_INSTRUCTIONS_FILE = "CONVENTIONS.md"


def _resolve_1m_model(current: str | None) -> str:
    """Return the model id that makes Claude Code request the 1M window (#1158).

    Preserves a model the user already selected via ``ANTHROPIC_MODEL`` (only
    appending the ``[1m]`` suffix when missing). When none is set it falls back
    to ``HEADROOM_1M_MODEL`` if defined, else the built-in default Opus (#2937).
    Idempotent — a value already ending in ``[1m]`` is returned unchanged.
    """
    fallback = (os.environ.get(_1M_MODEL_ENV) or "").strip() or _DEFAULT_1M_MODEL
    base = (current or "").strip() or fallback
    return base if base.endswith(_CONTEXT_1M_SUFFIX) else f"{base}{_CONTEXT_1M_SUFFIX}"


def _apply_1m_to_claude_args(args: tuple[str, ...]) -> tuple[tuple[str, ...], str | None]:
    """Add the ``[1m]`` suffix to an explicit ``--model`` in pass-through args.

    Claude Code gives the ``--model`` CLI flag precedence over the
    ``ANTHROPIC_MODEL`` env var, so when a user passes both ``--1m`` and
    ``--model X`` the env-var suffix is silently shadowed and the session caps at
    200k (#2915). Rewriting the flag's value the same way ``_resolve_1m_model``
    rewrites the env var keeps ``--1m`` effective on the higher-precedence flag.

    Handles ``--model VALUE`` and ``--model=VALUE`` (the first occurrence only, as
    Claude Code honours the first). Idempotent via ``_resolve_1m_model``. Returns
    ``(new_args, rewritten_value)``; ``rewritten_value`` is ``None`` when no
    ``--model`` was present (the env-var path already covers that case).
    """
    out = list(args)
    for i, arg in enumerate(out):
        if arg == "--model" and i + 1 < len(out):
            rewritten = _resolve_1m_model(out[i + 1])
            out[i + 1] = rewritten
            return tuple(out), rewritten
        if arg.startswith("--model="):
            rewritten = _resolve_1m_model(arg.split("=", 1)[1])
            out[i] = f"--model={rewritten}"
            return tuple(out), rewritten
    return tuple(out), None


def _normalize_tool_search_mode(value: str) -> str:
    """Validate an ``ENABLE_TOOL_SEARCH`` value and return it normalized.

    Mirrors the values Claude Code accepts: truthy (``true``/``1``/``yes``/
    ``on``), falsy (``false``/``0``/``no``/``off``), ``auto``, or ``auto:N``
    where ``N`` is 0-100. Raises :class:`click.ClickException` on anything else
    so a typo fails loudly instead of silently leaving deferral off.
    """
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on", "false", "0", "no", "off", "auto"}:
        return normalized
    if normalized.startswith("auto:"):
        suffix = normalized[len("auto:") :]
        if suffix.isdigit() and 0 <= int(suffix) <= 100:
            return normalized
    raise click.ClickException(
        f"--tool-search must be one of: true, false, auto, auto:N (N 0-100); got {value!r}"
    )


def _configure_tool_search_env(env: dict[str, str], flag_value: str | None) -> str | None:
    """Set ``ENABLE_TOOL_SEARCH`` in ``env`` so Claude Code keeps deferring tools.

    Precedence:

    1. explicit ``--tool-search`` flag — wins (the user asked for it on the CLI),
    2. a pre-existing ``ENABLE_TOOL_SEARCH`` in the environment — respected and
       left untouched (the user's own Claude Code knob),
    3. the built-in mode-specific default (``true`` normally, ``false`` on
       Foundry).

    Returns the value written, or ``None`` when an existing environment value
    was deliberately left in place.
    """
    if flag_value is not None:
        value = _normalize_tool_search_mode(flag_value)
        env[_TOOL_SEARCH_ENV] = value
        return value
    # An empty / whitespace value counts as unset: Claude Code treats an empty
    # ENABLE_TOOL_SEARCH as absent (so deferral would stay off), so we override
    # it with the default rather than forwarding a no-op value.
    existing = env.get(_TOOL_SEARCH_ENV)
    if existing is not None and existing.strip():
        return None
    default = (
        _TOOL_SEARCH_FOUNDRY_DEFAULT if env.get("CLAUDE_CODE_USE_FOUNDRY") else _TOOL_SEARCH_DEFAULT
    )
    env[_TOOL_SEARCH_ENV] = default
    return default


# ENABLE_TOOL_SEARCH modes that turn deferral OFF. Everything else Claude Code
# accepts (true/1/yes/on/auto/auto:N) keeps on-demand tool loading active.
_TOOL_SEARCH_FALSY = {"false", "0", "no", "off"}


# Reduce-at-source: CLI tools pad tool_result output with progress bars, pager
# framing, funding/telemetry banners, and version nags — all zero-signal tokens
# the agent never acts on. Setting conservative, SAFE env defaults in the
# launched agent's environment makes those tools emit less AT THE SOURCE, so the
# proxy never has to strip them. Only knobs that can't hide diffs, errors,
# summaries, or search results are set here (no blanket --silent/--quiet).
# Opt out entirely with HEADROOM_WRAP_QUIET=0 (or false/no/off).
_QUIET_CLI_ENV = "HEADROOM_WRAP_QUIET"
_QUIET_CLI_FALSY = {"0", "false", "no", "off"}
# name -> value, injected only when the user has not already set it.
_QUIET_CLI_DEFAULTS: dict[str, str] = {
    "GIT_PAGER": "cat",  # never page (keeps full content, drops pager framing)
    "PIP_QUIET": "1",  # drop "Requirement already satisfied"/download chatter
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",  # drop the "new pip available" nag
    "npm_config_fund": "false",  # drop the funding banner
    "npm_config_audit": "false",  # drop the audit summary (not a security scan here)
    "npm_config_progress": "false",  # drop the install progress bar
}


def _quiet_cli_enabled() -> bool:
    """Quiet-CLI source defaults are on unless HEADROOM_WRAP_QUIET is falsy."""
    return os.environ.get(_QUIET_CLI_ENV, "").strip().lower() not in _QUIET_CLI_FALSY


def _configure_quiet_cli_env(env: dict[str, str]) -> list[str]:
    """Inject SAFE quiet-CLI defaults into ``env`` in place; return names set.

    No-op when ``HEADROOM_WRAP_QUIET`` is falsy. A value the user already set
    always wins (defaults are only filled when absent). ``PYTEST_ADDOPTS`` is
    *augmented* with ``-q`` rather than clobbered, so an existing value survives.
    Nothing RISKY (anything that could suppress diffs/errors/summaries/search
    output) is ever set here.
    """
    if not _quiet_cli_enabled():
        return []
    written: list[str] = []
    for name, value in _QUIET_CLI_DEFAULTS.items():
        if name not in env:
            env[name] = value
            written.append(name)
    existing = env.get("PYTEST_ADDOPTS", "")
    if "-q" not in existing.split():
        env["PYTEST_ADDOPTS"] = f"{existing} -q".strip()
        written.append("PYTEST_ADDOPTS")
    return written


def _resolved_tool_search_mode(flag_value: str | None) -> str:
    """Predict the ``ENABLE_TOOL_SEARCH`` value the launched process will get.

    Runs :func:`_configure_tool_search_env` against a throwaway copy of the
    relevant environment, so messages printed *before* the real injection (the
    Remote Control sibling note, issue #1779) apply the exact same precedence
    (flag > existing non-blank env > default) and can never drift from it.
    """
    probe: dict[str, str] = {}
    existing = os.environ.get(_TOOL_SEARCH_ENV)
    if existing is not None:
        probe[_TOOL_SEARCH_ENV] = existing
    if os.environ.get("CLAUDE_CODE_USE_FOUNDRY"):
        probe["CLAUDE_CODE_USE_FOUNDRY"] = os.environ["CLAUDE_CODE_USE_FOUNDRY"]
    written = _configure_tool_search_env(probe, flag_value)
    return written if written is not None else probe.get(_TOOL_SEARCH_ENV, "")


def _tool_search_mode_is_active(value: str) -> bool:
    """Whether an ``ENABLE_TOOL_SEARCH`` mode keeps tool deferral on (#746)."""
    return value.strip().lower() not in _TOOL_SEARCH_FALSY


def _live_wrap_module() -> Any:
    """Return the current live wrap module instance."""
    return cast(Any, sys.modules[__name__])


def _module_available(module_name: str) -> bool:
    """Return whether an optional module is installed without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _ml_wrap_extras_detected() -> bool:
    """Detect slow optional ML stacks without triggering their import cost."""

    return any(_module_available(module_name) for module_name in _WRAP_PROXY_TIMEOUT_ML_MODULES)


def _wrap_agent_savings_profile(agent_type: str) -> str | None:
    """Return the savings profile required for agent wrappers, if any."""

    if agent_type not in _AGENT_SAVINGS_WRAP_AGENTS:
        return None
    return os.environ.get("HEADROOM_SAVINGS_PROFILE") or None


def _default_wrap_proxy_timeout_seconds() -> int:
    """Return the default wrap proxy startup timeout for this environment."""

    if _ml_wrap_extras_detected():
        return _WRAP_PROXY_TIMEOUT_ML_DEFAULT_SECONDS
    return _WRAP_PROXY_TIMEOUT_DEFAULT_SECONDS


def _resolve_wrap_proxy_timeout_seconds() -> int:
    """Resolve the wrap proxy readiness timeout from env or defaults."""

    raw = os.environ.get(_WRAP_PROXY_TIMEOUT_ENV, "").strip()
    if not raw:
        return _default_wrap_proxy_timeout_seconds()

    try:
        timeout_seconds = int(raw)
    except ValueError:
        raise RuntimeError(
            f"{_WRAP_PROXY_TIMEOUT_ENV} must be a positive integer number of seconds (got {raw!r})"
        ) from None
    if timeout_seconds <= 0:
        raise RuntimeError(
            f"{_WRAP_PROXY_TIMEOUT_ENV} must be a positive integer number of seconds (got {raw!r})"
        )
    return timeout_seconds


def _print_telemetry_notice() -> None:
    """Print a telemetry notice when anonymous telemetry is enabled.

    Respects the HEADROOM_TELEMETRY and HEADROOM_TELEMETRY_WARN feature flags.
    Does nothing when telemetry or warnings are disabled.
    """
    from headroom.telemetry.beacon import format_telemetry_notice

    notice = format_telemetry_notice(prefix="  ")
    if notice:
        click.echo(notice)


# Proxy health check (reused from evals/suite_runner.py pattern)


def _check_proxy(port: int) -> bool:
    """Check if Headroom proxy is running on given port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("127.0.0.1", port))
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def _port_bind_error(port: int) -> OSError | None:
    """Return the bind error for a local proxy port, or None when it is usable."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
    except OSError as exc:
        return exc
    except OverflowError:
        return OSError(errno.EADDRNOTAVAIL, f"Port {port} out of range (0-65535)")
    return None


def _find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Find first available port >= start_port via socket.bind probe.

    Skips ports with EADDRINUSE (busy) and EACCES (reserved on Windows,
    privileged on Linux) — both indicate the port can't be bound here.
    Other OS errors (EADDRNOTAVAIL) propagate immediately.
    Raises RuntimeError when no port is found in range.
    """
    end_port = min(start_port + max_attempts, 65536)
    for port in range(start_port, end_port):
        error = _port_bind_error(port)
        if error is None:
            return port
        if error.errno not in (errno.EADDRINUSE, errno.EACCES):
            raise error
    raise RuntimeError(f"No available port found in range {start_port}-{end_port - 1}")


def _get_log_path() -> Path:
    """Get path for proxy log file."""
    from headroom import paths as _paths

    log_dir = _paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "proxy.log"


def _get_proxy_stdio_log_path() -> Path:
    """Get path for dedicated proxy stdio capture."""
    return _get_log_path().with_name("proxy-stdio.log")


def _start_proxy(
    port: int,
    *,
    learn: bool = False,
    memory: bool = False,
    agent_type: str = "unknown",
    code_graph: bool = False,
    backend: str | None = None,
    anyllm_provider: str | None = None,
    region: str | None = None,
    openai_api_url: str | None = None,
    anthropic_api_url: str | None = None,
    vertex_api_url: str | None = None,
    clear_vertex_api_url: bool = False,
    copilot_api_token: str | None = None,
    copilot_refresh_oauth_token: str | None = None,
    copilot_api_token_expires_at: float | None = None,
) -> subprocess.Popen:
    """Start Headroom proxy as a background subprocess.

    Stdout and stderr are written to a dedicated sibling file, usually
    `~/.headroom/logs/proxy-stdio.log`, to avoid pipe deadlock risk without
    competing with the rotating `proxy.log` runtime log.

    The caller is responsible for ensuring *port* is available
    (see ``_find_available_port``).
    """

    cmd = [sys.executable, "-m", "headroom.cli", "proxy", "--port", str(port)]

    # Forward HEADROOM_MODE env var so the proxy respects the user's mode choice
    headroom_mode = os.environ.get("HEADROOM_MODE")
    if headroom_mode:
        cmd.extend(["--mode", headroom_mode])

    # Forward --learn flag to proxy subprocess
    if learn:
        cmd.append("--learn")

    # Forward --memory flag to proxy subprocess
    if memory:
        cmd.append("--memory")

    # Forward --code-graph flag to proxy subprocess (live file watcher)
    if code_graph:
        cmd.append("--code-graph")

    # Forward backend configuration to proxy subprocess
    _backend = backend or os.environ.get("HEADROOM_BACKEND")
    if _backend:
        cmd.extend(["--backend", _backend])

    _anyllm = anyllm_provider or os.environ.get("HEADROOM_ANYLLM_PROVIDER")
    if _anyllm:
        cmd.extend(["--anyllm-provider", _anyllm])

    _region = region or os.environ.get("HEADROOM_REGION")
    if _region:
        cmd.extend(["--region", _region])

    if openai_api_url:
        cmd.extend(["--openai-api-url", openai_api_url])

    if anthropic_api_url:
        cmd.extend(["--anthropic-api-url", anthropic_api_url])

    if vertex_api_url:
        cmd.extend(["--vertex-api-url", vertex_api_url])

    timeout_seconds = _resolve_wrap_proxy_timeout_seconds()
    log_path = _get_log_path()
    stdio_log_path = _get_proxy_stdio_log_path()
    stdio_log_file = open(stdio_log_path, "a", encoding="utf-8")  # noqa: SIM115

    # Ensure proxy subprocess uses UTF-8 (Windows defaults to cp1252)
    proxy_env = os.environ.copy()
    _scrub_copilot_proxy_seed_env(proxy_env)
    proxy_env["PYTHONIOENCODING"] = "utf-8"
    # `python -m headroom.cli` prepends the launch cwd to sys.path, so running
    # `wrap` from a directory that contains a `headroom/` folder (most commonly a
    # clone of this repo, whose package lives at <root>/headroom/) shadows the
    # installed wheel with the raw source tree, which has no compiled
    # `headroom._core`. The proxy then dies with "No module named 'headroom._core'"
    # and wrap silently falls back to launching the client unwrapped (#2793).
    # PYTHONSAFEPATH disables that cwd prepend (Python 3.11+; a harmless no-op on
    # 3.10) so the subprocess always resolves the installed package.
    proxy_env["PYTHONSAFEPATH"] = "1"
    # Vertex AI RST_STREAMs HTTP/2 connections (error_code:2). Force HTTP/1.1
    # when wrapping a Vertex-mode client so upstream requests succeed.
    if os.environ.get("CLAUDE_CODE_USE_VERTEX") or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"):
        proxy_env.setdefault("HEADROOM_HTTP2", "false")
    # Tell the proxy which agent is being wrapped (for traffic learning output)
    if agent_type != "unknown":
        proxy_env["HEADROOM_AGENT_TYPE"] = agent_type
        proxy_env.setdefault("HEADROOM_STACK", f"wrap_{agent_type}")
    savings_profile = _wrap_agent_savings_profile(agent_type)
    if savings_profile is not None:
        apply_agent_savings_env_defaults(proxy_env, savings_profile)
    if openai_api_url:
        proxy_env["OPENAI_TARGET_API_URL"] = openai_api_url
    if anthropic_api_url:
        proxy_env["ANTHROPIC_TARGET_API_URL"] = anthropic_api_url
    if clear_vertex_api_url:
        proxy_env.pop("VERTEX_TARGET_API_URL", None)
    if vertex_api_url:
        proxy_env["VERTEX_TARGET_API_URL"] = vertex_api_url
    # Pin the wrapper-validated Copilot token for this proxy instance only.
    # Injected into the subprocess env here (not the parent's os.environ) so it
    # never leaks into shared state. The proxy's CopilotTokenProvider honours
    # GITHUB_COPILOT_API_TOKEN directly, making upstream auth deterministic.
    if copilot_api_token:
        proxy_env["GITHUB_COPILOT_API_TOKEN"] = copilot_api_token
        if openai_api_url:
            proxy_env["GITHUB_COPILOT_API_URL"] = openai_api_url
    if copilot_refresh_oauth_token:
        proxy_env["GITHUB_COPILOT_REFRESH_OAUTH_TOKEN"] = copilot_refresh_oauth_token
    if copilot_api_token_expires_at is not None:
        proxy_env["GITHUB_COPILOT_API_TOKEN_EXPIRES_AT"] = str(copilot_api_token_expires_at)

    # Detach the proxy from the launching console on Windows so an ungraceful
    # close of the owning agent (closing the terminal window, taskkill, or a
    # crash) cannot tree-kill the shared proxy out from under other live
    # clients. Without this the proxy stays in the owner's console + Job
    # object; closing that window terminates the whole tree, bypassing the
    # marker-based reference counting in ``_make_cleanup`` and breaking every
    # other ``headroom wrap`` instance routed through the same port.
    #   CREATE_NO_WINDOW         — give the proxy its OWN, invisible console.
    #                              A separate console means the parent's
    #                              CTRL_CLOSE_EVENT never reaches it, and no
    #                              stray console window pops up. DETACHED_PROCESS
    #                              also isolates the console, but for a console
    #                              subsystem exe (python.exe) it leaves the proxy
    #                              consoleless and Windows surfaces a visible
    #                              console window — closing that window killed
    #                              the proxy, defeating the whole point.
    #   CREATE_NEW_PROCESS_GROUP — isolate from the parent's Ctrl-C
    #   CREATE_BREAKAWAY_FROM_JOB— survive Job kill-on-close (Windows Terminal,
    #                              VS Code integrated terminal, conhost)
    # CREATE_NO_WINDOW / DETACHED_PROCESS / CREATE_NEW_CONSOLE are mutually
    # exclusive — pick exactly one. On POSIX, ``start_new_session`` already
    # detaches via setsid(). ``sys.platform == "win32"`` (not ``os.name ==
    # "nt"``) so mypy narrows the platform and resolves the Windows-only
    # ``subprocess`` constants below.
    _CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | _CREATE_BREAKAWAY_FROM_JOB
        )

    popen_kwargs: dict[str, Any] = {
        "stdout": stdio_log_file,
        "stderr": stdio_log_file,
        "env": proxy_env,
        "start_new_session": os.name == "posix",
        "creationflags": creationflags,
    }
    # Close the parent's copy of the stdio log handle on every exit path,
    # including when BOTH spawn attempts raise. The child keeps its own
    # inherited duplicate, so closing here never starves the proxy's logging.
    try:
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except OSError:
            # The launcher's Job object forbids breakaway. Retry without that flag;
            # CREATE_NO_WINDOW still spares the proxy from console-close events.
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = creationflags & ~_CREATE_BREAKAWAY_FROM_JOB
            proc = subprocess.Popen(cmd, **popen_kwargs)

        # Wait for proxy to be ready.
        # ML components (Kompress, Magika, Tree-sitter) load synchronously before
        # uvicorn binds the port. On slower machines this can take 20-30 seconds.
        for _i in range(timeout_seconds):
            time.sleep(1)
            if _check_proxy(port):
                click.echo(f"  Logs: {log_path}")
                return proc
            # Check if process died
            if proc.poll() is not None:
                # Read last few lines of log for error context
                try:
                    tail = _read_text(stdio_log_path)[-500:]
                except Exception:
                    tail = "(no log output)"
                raise RuntimeError(f"Proxy exited with code {proc.returncode}: {tail}")

        proc.kill()
        raise RuntimeError(
            f"Proxy failed to start on port {port} within {timeout_seconds} seconds. "
            f"Set {_WRAP_PROXY_TIMEOUT_ENV} to a larger number of seconds for slow startup."
        )
    finally:
        stdio_log_file.close()


# CLI context tools (rtk, lean-ctx) were removed from Headroom. The selector is
# kept only long enough to fail loudly: it lives in shell profiles, scripts and
# CI jobs, and silently ignoring it would look like Headroom had stopped working.
# See :mod:`headroom.context_tool_cleanup`, which uninstalls what they left behind.
_RETIRED_CONTEXT_TOOL_ENV = "HEADROOM_CONTEXT_TOOL"
_RETIRED_CONTEXT_TOOL_MESSAGE = (
    "CLI context tools (rtk, lean-ctx) have been removed from Headroom: they "
    "rewrote shell commands through a third-party binary Headroom no longer "
    "manages. Drop --context-tool / --no-context-tool and unset "
    f"{_RETIRED_CONTEXT_TOOL_ENV}; `headroom wrap` uninstalls what they left "
    "behind automatically."
)


def _retired_context_tool_callback(ctx: Any, param: Any, value: str | None) -> str | None:
    """Click eager callback: reject any surviving context-tool selection.

    Also checks the env var (the callback runs on every wrap subcommand, flag
    passed or not), so an exported ``HEADROOM_CONTEXT_TOOL`` fails with the same
    message instead of silently doing nothing.
    """
    if value is not None or os.environ.get(_RETIRED_CONTEXT_TOOL_ENV, "").strip():
        raise click.ClickException(_RETIRED_CONTEXT_TOOL_MESSAGE)
    return value


# Applied to every ``wrap`` subcommand. ``expose_value=False`` so no subcommand
# signature carries it; both spellings the flag ever had are accepted and
# rejected with one message.
_retired_context_tool_option = click.option(
    "--context-tool",
    "--no-context-tool",
    default=None,
    is_flag=False,
    flag_value="",
    metavar="TOOL",
    expose_value=False,
    is_eager=True,
    hidden=True,
    callback=_retired_context_tool_callback,
    help="Removed: CLI context tools (rtk, lean-ctx) are no longer supported.",
)


def _should_purge_context_tools(ctx: click.Context) -> bool:
    """Whether this invocation should run the retired-context-tool cleanup.

    Two exemptions, both about not doing filesystem surgery from a command the
    caller expects to be inert:

    * ``wrap selfheal`` — runs from a SessionStart hook on every new
      conversation, where rewriting ``~/.claude.json`` would race Claude Code's
      own writer for no benefit.
    * any ``--help`` invocation — help must stay read-only. Click resolves a
      subcommand's help *after* this group callback, so it cannot be detected
      from ``ctx``; scanning argv is blunt but correct, and a false positive only
      defers the cleanup to the next real run.
    """
    if ctx.invoked_subcommand == "selfheal":
        return False
    return not any(arg in ("--help", "-h") for arg in sys.argv[1:])


def _report_context_tool_purge() -> None:
    """Uninstall leftover rtk / lean-ctx state, reporting anything removed.

    Removing the integration code cannot help a machine that already ran the old
    default: the Claude ``PreToolUse`` hook, the vendored binaries and the
    injected hint-file guidance are all durable on disk. Running this once per
    ``wrap`` / ``unwrap`` invocation is what actually makes the tools go away.
    Silent when there is nothing to do — the common case once the machine-global
    half is stamped done, though the project- and config-directory-scoped half
    still runs every launch — and never fatal: a cleanup failure must not block
    launching the tool.

    Reports on **stderr**: some subcommands (``wrap/unwrap openclaw
    --prepare-only``) emit machine-readable JSON on stdout as their entire
    contract, and a human cleanup line prepended to it breaks every
    ``json.loads(stdout)`` consumer on the one run that has something to remove.
    """
    from headroom.context_tool_cleanup import purge_context_tool_artifacts

    try:
        removed = purge_context_tool_artifacts()
    except Exception as exc:  # pragma: no cover - defensive, cleanup is best-effort
        click.echo(f"Warning: could not finish removing retired CLI context tools: {exc}", err=True)
        return
    for line in removed:
        click.echo(f"Retired CLI context tool cleanup: {line}", err=True)


def _serena_instructions_opt_in() -> bool:
    """Whether Serena instruction injection into the agent's hint file is enabled.

    Injecting "prefer Serena symbol tools" guidance rewrites the user's
    ``CLAUDE.md``/``AGENTS.md``, so it is opt-in (off by default): turn it on
    with ``--serena-instructions`` (which sets ``HEADROOM_SERENA_INSTRUCTIONS=1``)
    or by exporting ``HEADROOM_SERENA_INSTRUCTIONS=1``. Serena's ``.serena/``-only
    setup (language scoping, pre-indexing) stays on by default regardless.
    """
    return os.environ.get("HEADROOM_SERENA_INSTRUCTIONS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _serena_instructions_flag_callback(ctx: Any, param: Any, value: bool) -> bool:
    """Click eager callback: ``--serena-instructions`` sets
    HEADROOM_SERENA_INSTRUCTIONS so the central gate
    (:func:`_serena_instructions_opt_in`) sees the opt-in without threading a
    param through every wrap subcommand."""
    if value:
        os.environ["HEADROOM_SERENA_INSTRUCTIONS"] = "1"
    return value


# Shared opt-in flag for Serena instruction injection, applied to the wrap
# subcommands that set up Serena. ``expose_value=False`` so no subcommand
# signature changes; it works purely through HEADROOM_SERENA_INSTRUCTIONS. Same
# approach as _code_memory_option below — set via the callback with NO ``envvar=`` so the
# settings_store drift guard doesn't flag it.
_serena_instructions_option = click.option(
    "--serena-instructions",
    is_flag=True,
    default=False,
    expose_value=False,
    is_eager=True,
    callback=_serena_instructions_flag_callback,
    help="Inject 'prefer Serena symbol tools' guidance into the agent's hint file (opt-in; off by default).",
)


# --- Code-memory MCP selection ------------------------------------------------
# The code-memory MCP is Serena by default; turn it off with --code-memory none.
# Selection flows through HEADROOM_CODE_MEMORY (set by the eager --code-memory
# callback) so it works the same on every agent without threading a param
# through each subcommand — the same approach as _serena_instructions_option above.
_CODE_MEMORY_ENV = "HEADROOM_CODE_MEMORY"
_CODE_MEMORY_SERENA = "serena"
_CODE_MEMORY_NONE = "none"
_VALID_CODE_MEMORY = {_CODE_MEMORY_SERENA, _CODE_MEMORY_NONE}


def _resolve_code_memory(kwargs: dict[str, Any]) -> str:
    """Resolve which code-memory MCP to register.

    Precedence: the explicit selector (``--code-memory`` / ``HEADROOM_CODE_MEMORY``)
    wins; otherwise the deprecated ``--serena`` / ``--no-serena`` flags map into
    it; otherwise the default is ``serena`` — mature, offline, symbol-level code
    navigation. The retired ``tokensave`` option is accepted gracefully: an
    explicit ``tokensave`` selector (or the deprecated ``--no-tokensave`` flag)
    now resolves to Serena.
    """
    env = os.environ.get(_CODE_MEMORY_ENV, "").strip().lower()
    if env == "tokensave":
        click.echo("  Note: the tokensave code-memory option was retired — using Serena instead.")
        return _CODE_MEMORY_SERENA
    if env:
        if env not in _VALID_CODE_MEMORY:
            raise click.ClickException(
                f"{_CODE_MEMORY_ENV} must be one of: {', '.join(sorted(_VALID_CODE_MEMORY))}"
            )
        return env
    if kwargs.get("no_serena"):
        return _CODE_MEMORY_NONE
    return _CODE_MEMORY_SERENA


def _code_memory_flag_callback(ctx: Any, param: Any, value: str | None) -> str | None:
    """Click eager callback: ``--code-memory X`` sets HEADROOM_CODE_MEMORY so the
    central resolver (:func:`_resolve_code_memory`) sees the choice without
    threading a param through every wrap subcommand."""
    if value:
        os.environ[_CODE_MEMORY_ENV] = value
    return value


# Shared selector applied to code-memory-capable subcommands (claude/codex/grok).
# ``expose_value=False`` so no subcommand signature changes; it flows purely
# through HEADROOM_CODE_MEMORY.
_code_memory_option = click.option(
    "--code-memory",
    type=click.Choice([_CODE_MEMORY_SERENA, _CODE_MEMORY_NONE]),
    default=None,
    expose_value=False,
    is_eager=True,
    callback=_code_memory_flag_callback,
    help=(
        "Code-memory MCP to register: 'serena' (default) or 'none'. "
        "Also set by HEADROOM_CODE_MEMORY. Replaces --serena/--no-serena."
    ),
)


# Hook-command markers Headroom manages in Claude settings.json. unwrap drops
# any hook entry whose command contains one of these. (Retired rtk / lean-ctx
# hooks are removed separately, by
# headroom.context_tool_cleanup.purge_context_tool_artifacts.)
_HEADROOM_HOOK_MARKERS = ("headroom-init-claude",)

# Env vars Headroom's init/wrap inject into Claude settings.json; unwrap removes
# them. ENABLE_TOOL_SEARCH keeps Claude Code's tool deferral on behind the proxy
# (GH #746), paired with init/wrap setting it.
_HEADROOM_ENV_KEYS = ("ANTHROPIC_BASE_URL", "ENABLE_TOOL_SEARCH")

# Stable marker embedded in the SessionStart self-heal hook that ``wrap claude``
# installs (issue #2221). Lets that hook be found (idempotent install) and
# removed (unwrap) by its command string.
_WRAP_SELFHEAL_HOOK_MARKER = "headroom-wrap-selfheal"


def _remove_claude_managed_hooks(settings_path: Path | None = None) -> bool:
    """Remove Headroom-managed entries from Claude settings.json.

    Reverses what ``headroom init claude`` adds:
      * PreToolUse / SessionStart hooks whose command contains a Headroom marker
        (``headroom-init-claude``), and
      * the ``ANTHROPIC_BASE_URL`` proxy-routing env var.
    Unrelated settings and user-authored hooks are left untouched.
    """

    path = settings_path or (Path.home() / ".claude" / "settings.json")
    if not path.exists():
        return False

    try:
        payload = json.loads(_read_text(path))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False

    changed = False

    hooks = payload.get("hooks")
    if isinstance(hooks, dict):
        for event, entries in list(hooks.items()):
            if not isinstance(entries, list):
                continue
            retained_entries: list[Any] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    retained_entries.append(entry)
                    continue
                hook_items = entry.get("hooks")
                if not isinstance(hook_items, list):
                    retained_entries.append(entry)
                    continue
                retained_hooks = [
                    item
                    for item in hook_items
                    if not (
                        isinstance(item, dict)
                        and any(
                            marker in str(item.get("command", "")).lower()
                            for marker in _HEADROOM_HOOK_MARKERS
                        )
                    )
                ]
                if len(retained_hooks) != len(hook_items):
                    changed = True
                if retained_hooks:
                    retained_entries.append({**entry, "hooks": retained_hooks})
                elif len(retained_hooks) == len(hook_items):
                    retained_entries.append(entry)
                else:
                    changed = True
            if retained_entries:
                hooks[event] = retained_entries
            else:
                del hooks[event]
                changed = True

        if hooks:
            payload["hooks"] = hooks
        else:
            payload.pop("hooks", None)

    # Remove the proxy-routing env that init/wrap injected (ANTHROPIC_BASE_URL and
    # ENABLE_TOOL_SEARCH), even when no hooks remain (the early-return bug skipped
    # this). List-comp, not any(), so every key is popped (no short-circuit).
    env = payload.get("env")
    if isinstance(env, dict):
        removed_keys = [k for k in _HEADROOM_ENV_KEYS if env.pop(k, None) is not None]
        if removed_keys:
            changed = True
            if env:
                payload["env"] = env
            else:
                payload.pop("env", None)

    if not changed:
        return False

    _write_text(path, json.dumps(payload, indent=2) + "\n")
    return True


def _foundry_upstream_url(resource: str) -> str:
    """Derive the Azure AI Foundry endpoint URL from a resource name.

    When CLAUDE_CODE_USE_FOUNDRY=1 is set, Claude Code routes requests to the
    Azure AI Services endpoint it constructs from ANTHROPIC_FOUNDRY_RESOURCE.
    If ANTHROPIC_FOUNDRY_BASE_URL is not already set in the environment,
    we derive it here so the proxy knows where to forward compressed requests.

    Azure AI Foundry (AI Services) hosts the Anthropic-format Claude API at:
      https://{resource}.services.ai.azure.com/anthropic
    This matches the URL Claude Code constructs internally from ANTHROPIC_FOUNDRY_RESOURCE,
    and what ANTHROPIC_FOUNDRY_BASE_URL must point to for the Anthropic SDK to reach Claude.
    """
    return f"https://{resource.strip()}.services.ai.azure.com/anthropic"


def _foundry_proxy_url(proxy_url: str) -> str:
    """Return the local proxy URL that Claude Code should use in Foundry mode.

    ANTHROPIC_FOUNDRY_BASE_URL is the full base URL the Anthropic SDK appends
    /v1/messages to, so it must include the /anthropic path component to match
    the Azure AI Foundry endpoint structure.  _claude_proxy_base_url() returns
    the bare http://127.0.0.1:<port> — this helper appends /anthropic so the
    proxy URL Claude Code receives mirrors the real Foundry URL shape.
    """
    return proxy_url.rstrip("/") + "/anthropic"


def _vertex_target_api_url_from_claude_env(proxy_url: str) -> str | None:
    """Return the Vertex upstream that the proxy should use for Claude Code."""
    explicit_target = os.environ.get("VERTEX_TARGET_API_URL", "").strip()
    if explicit_target:
        return (
            None
            if _normalize_proxy_api_url(explicit_target) == _normalize_proxy_api_url(proxy_url)
            else explicit_target
        )

    vertex_url = os.environ.get("ANTHROPIC_VERTEX_BASE_URL", "").strip()
    if not vertex_url:
        return None

    from headroom.providers.registry import DEFAULT_VERTEX_API_URL

    normalized_vertex_url = _normalize_proxy_api_url(vertex_url)
    if normalized_vertex_url == _normalize_proxy_api_url(DEFAULT_VERTEX_API_URL):
        return None
    if normalized_vertex_url == _normalize_proxy_api_url(proxy_url):
        return None
    return vertex_url


def _claude_wrap_base_url_env_key(*, foundry_mode: bool = False, vertex_mode: bool = False) -> str:
    if vertex_mode:
        return "ANTHROPIC_VERTEX_BASE_URL"
    if foundry_mode:
        return "ANTHROPIC_FOUNDRY_BASE_URL"
    return "ANTHROPIC_BASE_URL"


def _wrap_marker_path(settings_path: Path) -> Path:
    """Sidecar marker path for a given settings.local.json path.

    Kept out of settings.local.json itself so Headroom's own bookkeeping never
    shows up as a stray key inside a file Claude Code's config loader parses.
    """
    return settings_path.parent / ".headroom_wrap_marker.json"


def _wrap_owners_path(settings_path: Path) -> Path:
    """Sidecar recording which live wrap sessions own each settings env key.

    Separate from ``.headroom_wrap_marker.json`` on purpose: that marker
    describes a single writer and is consumed by doctor, unwrap and the
    staleness checks. Concurrency ownership is additive state, so it lives in
    its own file rather than changing a shape those readers depend on.
    """
    return settings_path.parent / ".headroom_wrap_owners.json"


def _wrap_settings_lock(settings_path: Path) -> Any:
    """Serialize settings read-modify-write across concurrent wrap sessions.

    Writing the proxy URL into ``settings.local.json`` is a read-modify-write,
    and several ``headroom wrap`` sessions in one project run it concurrently.
    The write itself is atomic, so the file never tears -- but without this the
    updates are still lost against each other (#3205).
    """
    from contextlib import nullcontext

    lock_path = settings_path.parent / ".headroom_wrap_settings.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "a+b")  # noqa: SIM115
    except OSError:
        # Matches _proxy_start_lock: a workspace that cannot hold lock state is
        # degraded, not unusable.
        return nullcontext()
    return _locked_file(lock_file)


@contextmanager
def _locked_file(lock_file: Any) -> Any:
    """Hold an exclusive OS lock on an already-open file for the block.

    Shared by ``_proxy_start_lock`` and ``_wrap_settings_lock`` -- the two
    differ only in which file they lock, and an OS-lock dance duplicated per
    call site is one place for the platform branches to drift apart.
    """
    with lock_file:
        if sys.platform == "win32":
            import msvcrt

            # msvcrt.locking operates on bytes from the current file position.
            lock_file.seek(0)
            if lock_file.read(1) == b"":
                lock_file.seek(0)
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            # LK_LOCK has implementation-dependent retry limits, and a holder
            # may legitimately take longer than that (a proxy loading ML
            # components), so use the non-blocking primitive in a loop.
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_wrap_owners(settings_path: Path) -> dict[str, Any]:
    try:
        rec = json.loads(_read_text(_wrap_owners_path(settings_path)))
    except (OSError, ValueError):
        return {}
    return rec if isinstance(rec, dict) else {}


def _write_wrap_owners(settings_path: Path, owners: dict[str, Any]) -> None:
    target = _wrap_owners_path(settings_path)
    try:
        if not owners:
            target.unlink(missing_ok=True)
            return
        _write_text(target, json.dumps(owners, indent=2) + "\n")
    except OSError:
        pass


def _live_holders(entry: Any, *, dead_ports: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
    """Holders in *entry* whose process is still provably alive.

    Reuses the same conservative liveness the proxy-client markers use: a PID
    that is gone, or that is now provably a different process, is dropped. Any
    uncertainty keeps the holder, because dropping a live owner is what causes
    a running session to be unrouted.

    ``dead_ports`` additionally drops holders whose proxy port the caller has
    *proven* dead. A wrapper process outlives its proxy after a hard reboot or
    SIGKILL of the proxy alone, and such a holder routes nothing; left in place
    it would block the #2221 self-heal from clearing a base_url that now points
    at nothing.
    """
    if not isinstance(entry, dict):
        return []
    holders = entry.get("holders")
    if not isinstance(holders, list):
        return []
    live: list[dict[str, Any]] = []
    for holder in holders:
        if not isinstance(holder, dict):
            continue
        pid = holder.get("pid")
        if not isinstance(pid, int) or not _pid_alive(pid):
            continue
        if _identity_mismatch(holder.get("start_src"), holder.get("start_time"), pid):
            continue
        port = holder.get("port")
        if isinstance(port, int) and port in dead_ports:
            continue
        live.append(holder)
    return live


def _self_holder(port: int | None) -> dict[str, Any]:
    ident = _proc_identity(os.getpid())
    return {
        "pid": os.getpid(),
        "start_src": ident[0] if ident else None,
        "start_time": ident[1] if ident else None,
        "port": port,
    }


def _claim_wrap_key(
    settings_path: Path,
    key: str,
    current_value: str | None,
    *,
    port: int | None = None,
) -> None:
    """Register this process as an owner of *key*, recording the true original.

    The first live owner records ``original``; later owners inherit it and are
    flagged ``inherited`` so their exit knows the value they happened to
    observe was not the pre-wrap one. Without that, a second wrap session
    captures the *first session's* proxy URL as the value to restore, and puts
    a dead proxy back into the file on exit (#3205).
    """
    owners = _read_wrap_owners(settings_path)
    entry = owners.get(key)
    live = _live_holders(entry)
    inherited = bool(live) and isinstance(entry, dict) and "original" in entry
    original = entry.get("original") if inherited and isinstance(entry, dict) else current_value
    me = _self_holder(port)
    me["inherited"] = inherited
    live = [h for h in live if h.get("pid") != me["pid"]]
    live.append(me)
    owners[key] = {"original": original, "holders": live}
    _write_wrap_owners(settings_path, owners)


class _KeyRelease(NamedTuple):
    """Outcome of dropping this process's claim on a settings env key."""

    should_restore: bool
    original: str | None
    trust_caller: bool
    survivor: dict[str, Any] | None


def _release_wrap_key(
    settings_path: Path,
    key: str,
    *,
    force: bool = False,
    dead_ports: frozenset[int] = frozenset(),
) -> _KeyRelease:
    """Drop this process's claim on *key*.

    ``should_restore`` is False while another live wrap session still owns the
    key -- restoring then silently unroutes a running session. ``force`` is for
    ``unwrap``, where the user is explicitly asking for their settings back:
    every claim is dropped and the restore happens regardless.

    ``trust_caller`` says whether the caller's remembered ``previous`` is its
    own first-hand observation of the pre-wrap value. True when there is no
    owner record at all (unwrap of a pre-upgrade session, and the legacy
    callers that pass the value directly), and when this process founded the
    record. False for an inheriting holder -- it remembers the *first
    session's* proxy URL, so honouring it writes a dead proxy back, the exact
    bug #3205 is about -- and false for a caller with no claim of its own,
    whose marker-derived value is second-hand where the record is not.

    ``survivor`` is a still-live holder the caller can re-point the
    single-slot wrap marker at, so an exiting session does not take the
    surviving one's #2221 self-heal record with it.
    """
    owners = _read_wrap_owners(settings_path)
    entry = owners.get(key)
    if not isinstance(entry, dict):
        return _KeyRelease(True, None, True, None)
    me = os.getpid()
    remaining = [h for h in _live_holders(entry, dead_ports=dead_ports) if h.get("pid") != me]
    original = entry.get("original")
    # Look this process's own claim up in the raw holder list, never the
    # liveness-filtered one: the caller is by definition running, and its claim
    # is what says whether the value it remembers is first-hand.
    raw = entry.get("holders")
    mine = (
        next((h for h in raw if isinstance(h, dict) and h.get("pid") == me), None)
        if isinstance(raw, list)
        else None
    )
    trust_caller = mine is not None and not mine.get("inherited")
    if remaining and not force:
        owners[key] = {"original": original, "holders": remaining}
        _write_wrap_owners(settings_path, owners)
        return _KeyRelease(False, original, trust_caller, remaining[0])
    owners.pop(key, None)
    _write_wrap_owners(settings_path, owners)
    return _KeyRelease(True, original, trust_caller, None)


def _write_wrap_marker(settings_path: Path, *, port: int, key: str, previous: str | None) -> None:
    """Best-effort record of which (pid, port, key) wrote the base_url entry.

    Lets a later wrap/doctor/unwrap invocation tell a stale leftover (writer
    process is dead or its PID was recycled) from a still-live wrap session,
    and recover the true prior value (issue #1768) instead of guessing.
    """
    try:
        ident = _proc_identity(os.getpid())
        payload = {
            "pid": os.getpid(),
            "start_src": ident[0] if ident else None,
            "start_time": ident[1] if ident else None,
            "port": port,
            "key": key,
            "previous": previous,
        }
        _write_text(_wrap_marker_path(settings_path), json.dumps(payload))
    except OSError:
        pass


def _rehome_wrap_marker(
    settings_path: Path,
    *,
    key: str,
    survivor: dict[str, Any] | None,
    original: str | None,
) -> None:
    """Hand this session's wrap marker to a session that is still running.

    The marker has one slot and the last writer wins it. When that writer exits
    while a sibling still owns the key, leaving the marker describes a dead
    process, and deleting it strips the survivor of the #2221 dead-proxy
    self-heal record. Rewrite it to describe the survivor instead, carrying the
    owner record's ``original`` as the value to restore -- the marker's own
    ``previous`` may be an earlier session's proxy URL (#3205).

    Only ever touches a marker this process wrote; a sibling's marker is
    already accurate.
    """
    marker_path = _wrap_marker_path(settings_path)
    marker = _read_wrap_marker(settings_path)
    if marker is None or marker.get("key") != key or marker.get("pid") != os.getpid():
        return
    port = survivor.get("port") if survivor is not None else None
    try:
        if survivor is None or not isinstance(port, int):
            # No survivor to hand it to, or one whose port we never recorded:
            # a marker without a usable port is worse than none.
            marker_path.unlink(missing_ok=True)
            return
        _write_text(
            marker_path,
            json.dumps(
                {
                    "pid": survivor.get("pid"),
                    "start_src": survivor.get("start_src"),
                    "start_time": survivor.get("start_time"),
                    "port": port,
                    "key": key,
                    "previous": original,
                }
            ),
        )
    except OSError:
        pass


def _read_wrap_marker(settings_path: Path) -> dict[str, Any] | None:
    marker = _wrap_marker_path(settings_path)
    try:
        rec = json.loads(_read_text(marker))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _wrap_marker_is_stale(marker: dict[str, Any]) -> bool:
    """True if ``marker`` describes a writer that is provably gone.

    Missing/invalid pid, a dead pid, or a live pid whose recorded identity no
    longer matches (PID reuse) all count as stale — the entry it describes was
    left behind by a wrap session that no longer exists.
    """
    pid = marker.get("pid")
    if not isinstance(pid, int):
        return True
    if not _pid_alive(pid):
        return True
    return _identity_mismatch(marker.get("start_src"), marker.get("start_time"), pid)


def _wrap_proxy_alive(port: int, *, attempts: int = 3, delay: float = 0.25) -> bool:
    """Retry-hardened liveness probe for a wrap proxy ``port`` (issue #2221).

    A single 1s TCP connect can spuriously fail against a live-but-busy proxy
    (full accept queue, scheduler delay). Clearing a live session's base_url on
    such a transient blip stops its cc-daemon workers from routing through the
    proxy mid-session, so the proxy is declared ALIVE on the FIRST successful
    connect and DEAD only when all ``attempts`` (spaced ~``delay`` s apart)
    fail. Returns early on the first success, so a live proxy pays no delay.
    """
    for attempt in range(attempts):
        if _check_proxy(port):
            return True
        if attempt < attempts - 1:
            time.sleep(delay)
    return False


def _wrap_marker_proxy_is_dead(marker: dict[str, Any]) -> bool:
    """True if ``marker`` records a proxy ``port`` that no longer accepts
    connections.

    Port liveness is the authoritative signal for a wrap session that vanished
    without running its cleanup (hard reboot / SIGKILL, issue #2221): the
    recorded PID is unreliable because a reboot can recycle it onto an
    unrelated live process, so a PID that still looks alive does not prove the
    proxy is up. A marker with no recorded port returns False here (fall back
    to PID-based staleness); a marker whose port IS responding is a live
    session and must never be treated as dead. Uses the retry-hardened
    ``_wrap_proxy_alive`` so a momentary blip never reads as dead.
    """
    port = marker.get("port")
    if not isinstance(port, int):
        return False
    return not _wrap_proxy_alive(port)


def _clear_wrap_marker(settings_path: Path, *, key: str) -> None:
    marker = _read_wrap_marker(settings_path)
    if marker is not None and marker.get("key") == key:
        _wrap_marker_path(settings_path).unlink(missing_ok=True)


def _check_and_clear_stale_wrap_marker(settings_path: Path, *, key: str) -> str | None:
    """If a stale wrap marker for ``key`` exists, restore its recorded prior
    value and clear the marker. Returns the restored value, or None if there
    was nothing stale to clean up.

    Called before writing a fresh base_url entry so a crashed wrap session's
    leftover doesn't get treated as this session's own state to restore later.
    """
    marker = _read_wrap_marker(settings_path)
    if marker is None or marker.get("key") != key or not _wrap_marker_is_stale(marker):
        return None
    previous = marker.get("previous")
    click.echo(
        f"headroom: clearing stale {key} left by crashed wrap session (pid {marker.get('pid')})",
        err=True,
    )
    _restore_claude_wrap_base_url(previous, settings_path=settings_path, _key_override=key)
    return previous


def _check_and_clear_dead_wrap_marker(settings_path: Path, *, key: str) -> str | None:
    """Session-start self-heal for a wrap base_url left by a dead proxy (#2221).

    Like ``_check_and_clear_stale_wrap_marker`` (PID/identity based), but also
    clears when the marker's recorded proxy PORT is no longer accepting
    connections — even if its PID still looks alive. A hard reboot / SIGKILL
    runs no signal/atexit cleanup, so the ``ANTHROPIC_BASE_URL`` persisted for
    cc-daemon conversation workers keeps pointing at a dead proxy and bricks a
    later bare ``claude`` with ConnectionRefused. Because those workers read
    settings.local.json fresh per conversation, clearing it at session start
    (before any worker reads it) also unblocks the current session.

    CRITICAL: a marker whose port IS responding is a live wrapped session and
    is never cleared. Returns the restored prior value, or None when there was
    nothing dead to clean up.
    """
    marker = _read_wrap_marker(settings_path)
    if marker is None or marker.get("key") != key:
        return None
    port = marker.get("port")
    if isinstance(port, int):
        # Port is the authoritative signal (it survives PID reuse after a
        # reboot). A single retry-hardened probe decides it: a responding port
        # is a live session (never cleared); only a port that fails the whole
        # retry window is dead. One probe here — no correlated double check.
        if _wrap_proxy_alive(port):
            return None
    elif not _wrap_marker_is_stale(marker):
        # No recorded port → fall back to PID-based staleness.
        return None
    previous = marker.get("previous")
    click.echo(
        f"headroom: clearing stale {key} left by a proxy that is no longer "
        f"running (issue #2221); restoring prior value",
        err=True,
    )
    _restore_claude_wrap_base_url(
        previous,
        settings_path=settings_path,
        _key_override=key,
        # The wrapper process can outlive its proxy (the proxy alone was
        # SIGKILLed). Its ownership claim would otherwise veto this restore and
        # leave the base_url pointing at a port proven dead just above (#3205).
        dead_ports=frozenset({port}) if isinstance(port, int) else frozenset(),
    )
    return previous


def _selfheal_dead_wrap_base_url() -> None:
    """Clear a project-local wrap base_url left pointing at a dead proxy (#2221).

    Runs at every Claude session start via the SessionStart hook that
    ``wrap claude`` installs. When ``wrap claude`` persists
    ``ANTHROPIC_BASE_URL=<proxy>`` into ``.claude/settings.local.json`` and the
    proxy later dies via hard reboot / SIGKILL, no signal/atexit cleanup fires,
    so the stale URL lingers and bricks a later bare ``claude`` with
    ConnectionRefused. cc-daemon reads settings.local.json fresh per
    conversation, so clearing it here — before any conversation worker reads
    it — also unblocks the current session.

    Must never raise: a broken self-heal must not break session startup.
    """
    try:
        settings_path = Path.cwd() / ".claude" / "settings.local.json"
        for key in (
            _claude_wrap_base_url_env_key(),
            _claude_wrap_base_url_env_key(foundry_mode=True),
            _claude_wrap_base_url_env_key(vertex_mode=True),
        ):
            _check_and_clear_dead_wrap_marker(settings_path, key=key)
    except Exception:  # noqa: BLE001 - hook must never break session startup
        pass


def _wrap_selfheal_hook_command() -> str:
    """Command string for the SessionStart self-heal hook (mirrors init hooks)."""
    from headroom.cli.init import _command_string
    from headroom.install.runtime import resolve_headroom_command

    return _command_string(
        [*resolve_headroom_command(), "wrap", "selfheal", "--marker", _WRAP_SELFHEAL_HOOK_MARKER]
    )


def _ensure_claude_wrap_selfheal_hook(settings_path: Path) -> None:
    """Install a SessionStart-only self-heal hook into settings.local.json (#2221).

    ``wrap claude`` writes the proxy base_url + a sidecar marker but installs no
    hook of its own, so a session that only ran ``wrap`` (never ``init``) had no
    reader to clear a dead-proxy URL — the reported bug. This pairs the marker
    with a SessionStart hook that runs the hidden ``wrap selfheal`` command.
    SessionStart ONLY (never PreToolUse): the self-heal must not run per Bash
    call mid-session, where a transient probe blip could clear a live session.
    Idempotent — an existing entry carrying the marker is not duplicated.
    """
    payload = _read_settings_for_write(settings_path)
    hooks = dict(payload.get("hooks") or {}) if isinstance(payload.get("hooks"), dict) else {}
    entries = (
        list(hooks.get("SessionStart") or []) if isinstance(hooks.get("SessionStart"), list) else []
    )
    already = any(
        isinstance(entry, dict)
        and isinstance(entry.get("hooks"), list)
        and any(
            isinstance(item, dict) and _WRAP_SELFHEAL_HOOK_MARKER in str(item.get("command", ""))
            for item in entry["hooks"]
        )
        for entry in entries
    )
    if already:
        return
    entries.append(
        {
            "matcher": "startup|resume",
            "hooks": [
                {
                    "type": "command",
                    "command": _wrap_selfheal_hook_command(),
                    "timeout": 10,
                }
            ],
        }
    )
    hooks["SessionStart"] = entries
    payload["hooks"] = hooks
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(settings_path, json.dumps(payload, indent=2) + "\n")


def _remove_claude_wrap_selfheal_hook(settings_path: Path) -> bool:
    """Remove the SessionStart self-heal hook that ``wrap claude`` installed (#2221).

    Mirrors ``_remove_claude_managed_hooks`` but matches only the wrap self-heal
    marker in the project-local settings.local.json. Returns True if anything
    was removed. Unrelated hooks and user-authored entries are left untouched.
    """
    if not settings_path.exists():
        return False
    try:
        payload = json.loads(_read_text(settings_path))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return False
    changed = False
    for event, entries in list(hooks.items()):
        if not isinstance(entries, list):
            continue
        retained: list[Any] = []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("hooks"), list):
                kept = [
                    item
                    for item in entry["hooks"]
                    if not (
                        isinstance(item, dict)
                        and _WRAP_SELFHEAL_HOOK_MARKER in str(item.get("command", ""))
                    )
                ]
                if len(kept) != len(entry["hooks"]):
                    changed = True
                    if kept:
                        retained.append({**entry, "hooks": kept})
                    continue
            retained.append(entry)
        if retained:
            hooks[event] = retained
        else:
            del hooks[event]
            changed = True
    if not changed:
        return False
    if hooks:
        payload["hooks"] = hooks
    else:
        payload.pop("hooks", None)
    if payload:
        _write_text(settings_path, json.dumps(payload, indent=2) + "\n")
    else:
        settings_path.unlink(missing_ok=True)
    return True


def _write_claude_wrap_base_url(
    proxy_url: str,
    *,
    foundry_mode: bool = False,
    vertex_mode: bool = False,
    settings_path: Path | None = None,
    port: int | None = None,
) -> str | None:
    """Persist proxy URL into project-local settings env key for daemon child inheritance.

    Claude Code's cc-daemon pre-forks conversation workers using spawn (not
    fork), so those workers read settings.json fresh rather than inheriting
    the daemon's environment.  Writing the mode-specific Claude base URL env
    key into the project-local settings file (.claude/settings.local.json in
    cwd) ensures every new conversation — including those started after the
    initial launch — routes through the Headroom proxy without touching the
    global user settings file or affecting sessions in other projects. Returns
    the previous value so the caller can restore it on exit (issue #951).

    When ``port`` is given, also stamps a sidecar marker recording this
    process's identity and the previous value, so a later crash can be
    detected and self-healed (issue #1768).
    """
    path = settings_path or (Path.cwd() / ".claude" / "settings.local.json")
    key = _claude_wrap_base_url_env_key(foundry_mode=foundry_mode, vertex_mode=vertex_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _wrap_settings_lock(path):
        payload = _read_settings_for_write(path)
        env_map = dict(payload.get("env") or {}) if isinstance(payload.get("env"), dict) else {}
        previous = env_map.get(key)
        # Claim before writing, so the recorded original is the value that was
        # there before *any* wrap session touched it -- not the previous
        # session's proxy URL (#3205).
        _claim_wrap_key(path, key, previous, port=port)
        env_map[key] = proxy_url
        payload["env"] = env_map
        _write_text(path, json.dumps(payload, indent=2) + "\n")
        if port is not None:
            _write_wrap_marker(path, port=port, key=key, previous=previous)
    return previous


def _write_claude_wrap_tool_search(value: str, *, settings_path: Path | None = None) -> str | None:
    """Persist the resolved tool-search mode for daemon-spawned workers.

    Claude Code workers read project settings afresh rather than inheriting
    the parent process environment (#2492). Keep this separate from the proxy
    URL crash marker: a stale tool-search mode cannot route traffic to a dead
    process, and is restored transactionally when the wrap session exits.
    """
    path = settings_path or (Path.cwd() / ".claude" / "settings.local.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _wrap_settings_lock(path):
        payload = _read_settings_for_write(path)
        env_map = dict(payload.get("env") or {}) if isinstance(payload.get("env"), dict) else {}
        previous = env_map.get(_TOOL_SEARCH_ENV)
        _claim_wrap_key(path, _TOOL_SEARCH_ENV, previous)
        env_map[_TOOL_SEARCH_ENV] = value
        payload["env"] = env_map
        _write_text(path, json.dumps(payload, indent=2) + "\n")
    return previous


def _restore_claude_wrap_tool_search(
    previous: str | None, *, settings_path: Path | None = None
) -> None:
    """Restore the project-local tool-search value written for this session."""
    _restore_claude_wrap_base_url(
        previous,
        settings_path=settings_path,
        _key_override=_TOOL_SEARCH_ENV,
    )


def _restore_claude_wrap_base_url(
    previous: str | None,
    *,
    foundry_mode: bool = False,
    vertex_mode: bool = False,
    settings_path: Path | None = None,
    _key_override: str | None = None,
    force: bool = False,
    dead_ports: frozenset[int] = frozenset(),
) -> None:
    """Restore (or remove) the env key written by _write_claude_wrap_base_url.

    Called in both the wrap-session finally block and unwrap_claude so the
    project-local settings entry is never left pointing at a dead proxy.  When
    ``previous`` is None the key is removed; when it has a value it is
    restored — preserving any URL the project already had set. Also clears
    this key's sidecar wrap marker, if any (issue #1768).

    Concurrency (#3205): while another live wrap session still owns the key,
    this is a no-op — restoring underneath a running session unroutes it. Set
    ``force`` when the user has explicitly asked for their settings back
    (``unwrap``), and ``dead_ports`` to name proxy ports already proven dead so
    holders that outlived their proxy stop counting as live.
    """
    path = settings_path or (Path.cwd() / ".claude" / "settings.local.json")
    key = _key_override or _claude_wrap_base_url_env_key(
        foundry_mode=foundry_mode, vertex_mode=vertex_mode
    )
    with _wrap_settings_lock(path):
        # Another live wrap session in this project may still be using the key.
        # Restoring underneath it silently unroutes a running session -- traffic
        # bypasses the proxy with no error anywhere (#3205).
        release = _release_wrap_key(path, key, force=force, dead_ports=dead_ports)
        if not release.should_restore:
            # The value stays, but this session's marker must not linger
            # describing a process that is gone: hand the slot to a survivor.
            _rehome_wrap_marker(path, key=key, survivor=release.survivor, original=release.original)
            return
        # The owner record holds the value from before *any* wrap session wrote.
        # Prefer the caller's own value only when the caller observed it
        # first-hand; a session that started second remembers the first
        # session's (now dead) proxy URL, and so does the marker an unwrap or a
        # self-heal reads it from.
        restore_to = previous if release.trust_caller else release.original

        if not path.exists():
            _clear_wrap_marker(path, key=key)
            return
        try:
            payload = json.loads(_read_text(path))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        env_map = payload.get("env")
        if not isinstance(env_map, dict):
            return
        if restore_to is None:
            if key not in env_map:
                _clear_wrap_marker(path, key=key)
                return
            del env_map[key]
            if env_map:
                payload["env"] = env_map
            else:
                payload.pop("env", None)
        else:
            env_map[key] = restore_to
            payload["env"] = env_map
        if payload:
            _write_text(path, json.dumps(payload, indent=2) + "\n")
        else:
            path.unlink(missing_ok=True)
        _clear_wrap_marker(path, key=key)


def _setup_headroom_mcp(
    registrar: Any, port: int, *, verbose: bool = False, force: bool = False
) -> None:
    """Register the headroom MCP server with the given agent (idempotent).

    The proxy compresses tool_result payloads and emits ``[Retrieve more:
    hash=…]`` markers. Without this registration those markers point at
    nothing — the agent has no ``headroom_retrieve`` tool to call.

    Generic across registrars: ``ClaudeRegistrar``, ``CodexRegistrar``, and
    any future agent registrar all flow through the same setup path.
    """
    from headroom.mcp_registry import build_headroom_spec, format_result

    if not registrar.detect():
        if verbose:
            click.echo(f"  MCP retrieve tool: {registrar.display_name} not detected — skipping")
        return

    proxy_url = f"http://127.0.0.1:{port}"
    spec = build_headroom_spec(proxy_url)
    result = registrar.register_server(spec, force=force)

    line = format_result(
        registrar.name,
        result,
        label="MCP retrieve tool",
        verbose=verbose,
        overwrite_hint=f"headroom mcp install --proxy-url {proxy_url} --force",
        restart_hint=f"restart {registrar.display_name} if it was already running",
    )
    if line is not None:
        click.echo(line)


def _ensure_serena_dashboard_disabled(*, verbose: bool = False) -> None:
    """Disable Serena's browser dashboard auto-open in ``~/.serena/serena_config.yml``.

    Serena opens its web dashboard in a browser tab on launch by default
    (``web_dashboard_open_on_launch: true``), so flip that off for users who run
    Serena outside Headroom. The dashboard backend still runs and stays reachable
    at http://localhost:24282/dashboard/. Other keys and comments are preserved
    via a targeted line edit rather than a YAML rewrite.

    **Never creates the file.** Verified against Serena 1.6.2.dev0
    (``serena/config/serena_config.py``): Serena autogenerates its own complete
    config only when the path does *not* exist (``if not
    os.path.exists(config_file_path): cls._generate_config_file(...)``, ~line
    1033). Once any file exists it validates instead of filling gaps, and while
    every other field falls back to a dataclass default via
    ``get_value_or_default``, a missing ``projects`` key is fatal (~line 1064):

        SerenaConfigError: `projects` key not found in Serena configuration.

    So Headroom writing its own bootstrap file bricked Serena on every machine
    without a pre-existing config — the MCP server died mid-handshake ("connection
    closed: initialize response" on Codex, bare ``MCP error -32000`` on OpenCode)
    and ``serena project index`` failed identically (#2674). Letting Serena
    generate the file is immune to Serena adding required keys later; guessing the
    schema is what caused the outage.

    Suppressing the popup does not need this file anyway: ``build_serena_spec``
    passes ``--open-web-dashboard False``, which Serena applies *after* loading
    the config (``serena/mcp.py:361`` — ``config.web_dashboard_open_on_launch =
    open_web_dashboard``), so the flag wins regardless of what is on disk.

    ``projects: []`` is still backfilled into an *existing* file, to repair
    configs an affected Headroom version already wrote.
    """
    import re

    cfg = Path.home() / ".serena" / "serena_config.yml"
    key = "web_dashboard_open_on_launch"
    if not cfg.exists():
        # Let Serena bootstrap its own valid config; the MCP flag handles the popup.
        if verbose:
            click.echo("  Serena: no serena_config.yml yet — letting Serena generate it")
        return
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError as e:
        if verbose:
            click.echo(f"  Serena: could not read serena_config.yml ({e})")
        return

    new = text
    appended: list[str] = []

    dashboard = re.compile(rf"^(\s*){re.escape(key)}:\s*\S+\s*$", re.MULTILINE)
    if dashboard.search(new):
        new = dashboard.sub(rf"\g<1>{key}: false", new)
    else:
        appended.append(f"{key}: false")

    # Repair a config left by an affected Headroom version (see #2674 above).
    if not re.search(r"^\s*projects\s*:", new, re.MULTILINE):
        appended.append("projects: []")

    if appended:
        body = new.rstrip("\n")
        new = (f"{body}\n" if body.strip() else "") + "\n".join(appended) + "\n"

    if new == text:
        return
    try:
        cfg.write_text(new, encoding="utf-8")
    except OSError as e:
        if verbose:
            click.echo(f"  Serena: could not update serena_config.yml ({e})")
        return
    if verbose:
        click.echo("  Serena: updated serena_config.yml (dashboard auto-open off)")


# Marker-fenced guidance steering the agent toward Serena's symbol tools.
# Injected only when Serena is the active code-memory engine (idempotent,
# marker-guarded).
_SERENA_MARKER = "<!-- headroom:serena-instructions -->"

SERENA_INSTRUCTIONS_BLOCK = """\
<!-- headroom:serena-instructions -->
# Serena — Symbol-First Code Navigation

Serena's MCP tools expose this project's code as a symbol graph backed by a
language server. **Prefer these tools over reading whole files** — they return
only the code you need, cutting context usage sharply. Read a file end-to-end
only when a symbol view is insufficient (non-code files, or when you need the
surrounding glue).

## Preferred workflow
- `get_symbols_overview(<file>)` — list a file's top-level symbols before opening it.
- `find_symbol(<name>)` — fetch a symbol's definition/body instead of reading the file.
- `find_referencing_symbols(<name>)` — find call sites / usages instead of grepping.
- `find_declaration(<name>)` — jump to where a symbol is defined.

## Rule
Reach for a symbol tool first; fall back to reading a whole file only when the
symbol view does not answer the question.
<!-- /headroom:serena-instructions -->
"""


def _serena_instruction_file(registrar: Any) -> Path:
    """Resolve the project instruction file the agent reads for guidance.

    Claude Code reads ``CLAUDE.md``; Codex, Grok, and OpenCode read
    ``AGENTS.md``. Both live at the project root, mirroring the RTK instruction
    targets.
    """
    name = getattr(registrar, "name", "") or ""
    filename = "CLAUDE.md" if name == "claude" else "AGENTS.md"
    return Path.cwd() / filename


def _inject_serena_instructions(file_path: Path, verbose: bool = False) -> bool:
    """Steer the agent toward Serena's symbol tools over whole-file reads.

    Opt-in (off by default): mirrors :func:`_inject_rtk_instructions` and
    early-returns unless ``--serena-instructions`` / ``HEADROOM_SERENA_INSTRUCTIONS``
    is set, so the user's hint file is left untouched by default.

    Idempotent — skips if the marker is already present. Appends to an existing
    instruction file, or creates one. Returns True once the guidance is in place.
    """
    if not _serena_instructions_opt_in():
        return False
    if file_path.exists():
        existing = _read_text(file_path)
        if _SERENA_MARKER in existing:
            if verbose:
                click.echo(f"  Serena instructions already in {file_path.name}")
            return True
        _append_text(file_path, "\n\n" + SERENA_INSTRUCTIONS_BLOCK)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(file_path, SERENA_INSTRUCTIONS_BLOCK)

    click.echo(f"  Serena instructions injected into {file_path}")
    return True


def _serena_project_skip_reason(root: Path) -> str | None:
    """Why Serena's per-project setup must not run for *root* (None = proceed).

    ``$HOME`` is never a project: scanning it walks every unrelated tree
    (Downloads, VM images, network mounts) and would write ``project.yml`` into
    Serena's own ``~/.serena`` config directory. A linked git worktree (its
    top-level ``.git`` is a file, not a directory) is an ephemeral checkout that
    would pay for its own index at a path that soon disappears.

    A project with no ``.serena/project.yml`` is skipped because the pre-index
    cannot succeed there (#2938). ``serena project index`` auto-creates the file
    when it is missing, and that auto-creation calls
    ``ProjectConfig.autogenerate(interactive=True)``, which asks one ``[y/N]``
    question per additionally-detected language server. The CLI has no
    non-interactive switch; the only way to reach the silent branch is to pass
    ``--ls/--language`` explicitly, which means Headroom guessing the project's
    languages again — exactly the hand-maintained map removed below. Serena's
    MCP server generates that file itself (non-interactively) on first start and
    indexes lazily on demand, so the pre-index simply resumes from the next
    wrap onwards.
    """
    try:
        resolved = root.resolve()
        home = Path.home().resolve()
    except OSError:
        return None
    if resolved == home:
        return "$HOME is not a project"
    if (resolved / ".git").is_file():
        return "linked git worktree"
    if not (resolved / ".serena" / "project.yml").is_file():
        return "no .serena/project.yml yet — Serena will create it and index on demand"
    return None


#: Upper bound on the synchronous pre-index. The agent does not launch until
#: this call returns, so the number is a stall budget, not just a safety net.
_SERENA_INDEX_TIMEOUT = 300
_SERENA_INDEX_TIMEOUT_ENV = "HEADROOM_SERENA_INDEX_TIMEOUT"


def _resolve_serena_index_timeout_seconds() -> int:
    """Resolve the Serena pre-index stall budget from env, else the default.

    A wrap launched from a directory Serena has already claimed re-indexes the
    whole tree on every run, and 300s of that is time the agent is not running
    (#3093). The budget is therefore tunable per environment, which also keeps
    it reachable from ``wrap ... -- agents`` sessions that take no flags.

    Unlike :func:`_resolve_wrap_proxy_timeout_seconds`, a bad value is not
    fatal here: the pre-index is best-effort, so an unusable setting falls back
    to the default rather than aborting a launch that would otherwise succeed.
    It is reported unconditionally, because a knob that looks applied but is
    not is the failure this issue is about.
    """
    raw = os.environ.get(_SERENA_INDEX_TIMEOUT_ENV, "").strip()
    if not raw:
        return _SERENA_INDEX_TIMEOUT

    timeout_seconds: int | None
    try:
        timeout_seconds = int(raw)
    except ValueError:
        timeout_seconds = None
    if timeout_seconds is None or timeout_seconds <= 0:
        click.echo(
            f"  Serena: ignoring {_SERENA_INDEX_TIMEOUT_ENV}={raw!r} "
            f"(want a positive integer number of seconds) "
            f"— using {_SERENA_INDEX_TIMEOUT}s"
        )
        return _SERENA_INDEX_TIMEOUT
    return timeout_seconds


def _kill_serena_index_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and everything it spawned (best-effort, never raises).

    ``uvx`` is a launcher: it resolves the environment and then runs the real
    ``serena`` executable as a grandchild. Killing only the direct child leaves
    that grandchild alive and reparented to PID 1, so every timed-out pre-index
    leaked one process that never exits (#2938 — the same failure mode as #615
    and #880). The child is started in its own process group precisely so the
    whole tree can be signalled here.
    """
    if sys.platform == "win32":
        # Windows has no process groups to signal for an already-wedged child;
        # ``taskkill /T`` walks the tree by parent PID instead. ``/F`` because a
        # process blocked in a read will not act on a graceful close request.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    # Backstop: if the tree kill above did not land, at least the direct child
    # goes. Then reap so the parent does not leave a zombie behind, and close
    # the capture pipes we opened so the wrap does not carry stray fds into the
    # agent it is about to exec.
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    for stream in (proc.stdout, proc.stderr, proc.stdin):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass


def _index_serena_project(*, verbose: bool = False) -> None:
    """Warm Serena's symbol cache for the current project (non-fatal).

    Runs ``serena project index`` (the same ``uvx --from serena-agent`` launch
    used to start the MCP server) in the project directory so the first symbol
    query is not paying for a cold index. Serena also indexes lazily on demand,
    so any failure here is survivable.

    This runs on the launch path, synchronously: the agent starts only once it
    returns, so the timeout below is time the user spends staring at nothing —
    ``HEADROOM_SERENA_INDEX_TIMEOUT`` resizes that budget (#3093). Two guards
    keep it bounded (#2938):

    * ``stdin`` is ``DEVNULL``. Serena prompts when it has to auto-create
      ``project.yml``, and because stdout is captured the question never
      reaches the terminal — an inherited stdin turned that into a silent,
      full-timeout hang. EOF makes it fail in about a second instead.
      ``_serena_project_skip_reason`` already keeps us out of that state; this
      is the belt-and-braces half, and it covers any future Serena prompt too.
    * The child gets its own process group so ``_kill_serena_index_tree`` can
      take out the ``uvx`` grandchild on timeout rather than orphaning it.
    """
    if shutil.which("uvx") is None:
        if verbose:
            click.echo("  Serena: uvx not found — skipping pre-index")
        return

    timeout_seconds = _resolve_serena_index_timeout_seconds()

    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "text": True,
        # ``subprocess.Popen`` directly, so the encoding defaults that
        # ``headroom._subprocess.run`` applies have to be repeated here.
        "encoding": "utf-8",
        "errors": "replace",
        "cwd": str(Path.cwd()),
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(
            [
                "uvx",
                # PyPI (prebuilt wheels), not the git source that fails to build
                # under proot-based filesystems (#2871).
                "--from",
                "serena-agent",
                "serena",
                "project",
                "index",
            ],
            **popen_kwargs,
        )
    except Exception as e:
        if verbose:
            click.echo(f"  Serena: pre-index skipped ({e})")
        return

    # Announce the wait. Indexing a large repo legitimately takes minutes and
    # the output is captured, so without this line the wrap looks hung.
    click.echo("  Serena: pre-indexing project (first run can take a while)…")
    try:
        _stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_serena_index_tree(proc)
        click.echo("  Serena: pre-index timed out (will index on demand)")
        return
    except Exception as e:
        _kill_serena_index_tree(proc)
        if verbose:
            click.echo(f"  Serena: pre-index skipped ({e})")
        return

    if proc.returncode == 0:
        click.echo("  Serena: project pre-indexed (symbol cache warmed)")
    elif verbose:
        click.echo(f"  Serena: pre-index failed ({(stderr or '')[:100]})")


def _setup_serena_mcp(
    registrar: Any, *, context: str, verbose: bool = False, force: bool = False
) -> None:
    """Register Serena MCP with the given agent (idempotent).

    A prior ``headroom wrap`` may have persisted a Serena entry built from an
    older spec — e.g. before ``--open-web-dashboard False`` was added to
    suppress the dashboard popup (#1003). ``register_server`` returns
    ``MISMATCH`` and refuses to overwrite a differing entry unless forced, so
    on its own a re-wrap leaves already-wrapped users stuck on the stale spec
    (and the popup) forever. When the ledger proves the entry currently in the
    config is one Headroom installed, force-update it to the current spec. A
    user-managed Serena (absent from our ledger) is left untouched and the
    mismatch is reported as before.
    """
    from headroom.mcp_registry import build_serena_spec, format_result
    from headroom.mcp_registry.base import RegisterStatus
    from headroom.mcp_registry.ledger import headroom_installed_matching, record_install

    if not registrar.detect():
        if verbose:
            click.echo(f"  Serena MCP: {registrar.display_name} not detected — skipping")
        return

    if shutil.which("uvx") is None:
        click.echo("  Serena MCP: uvx not found — install uv/uvx to enable Serena; skipping")
        return

    # Serena is a real launch now — make sure it won't pop a browser tab.
    _ensure_serena_dashboard_disabled(verbose=verbose)

    spec = build_serena_spec(context)
    result = registrar.register_server(spec, force=force)
    owned_drift = (
        result.status == RegisterStatus.MISMATCH
        and not force
        and headroom_installed_matching(registrar.name, registrar.get_server("serena"))
    )

    # Migrate a stale Headroom-installed entry. register_server won't overwrite
    # a differing spec without force, so an older Headroom Serena entry would
    # otherwise persist across re-wraps. Force-update it only when the ledger
    # proves Headroom installed the entry that's currently on disk — never a
    # user-managed Serena.
    if result.status == RegisterStatus.MISMATCH and not force and owned_drift:
        result = registrar.register_server(spec, force=True)
        if result.status == RegisterStatus.REGISTERED:
            click.echo("  Serena MCP: migrated previously-installed entry to current spec")

    if result.status == RegisterStatus.REGISTERED:
        record_install(registrar.name, spec)

    line = format_result(
        registrar.name,
        result,
        label="Serena MCP",
        verbose=verbose,
        overwrite_hint=(
            "run headroom wrap again"
            if owned_drift
            else "run headroom mcp reconcile --adopt"
            if registrar.name == "claude"
            else "update or remove the existing serena MCP entry, then rerun headroom wrap"
        ),
        restart_hint=f"restart {registrar.display_name} if it was already running",
    )
    if line is not None:
        click.echo(line)

    # Serena is the active engine here (we passed the detect/uvx guards): steer
    # the agent toward symbol-level tools, then warm the symbol cache. Both are
    # best-effort and non-fatal, but the pre-index is *synchronous* — the agent
    # does not launch until it returns or hits ``_SERENA_INDEX_TIMEOUT``. See
    # ``_index_serena_project`` for how that wait is kept bounded and visible.
    #
    # Headroom no longer writes ``.serena/project.yml`` language scoping. Serena
    # determines the project's languages itself during
    # ``ProjectConfig.autogenerate`` (``_determine_project_language_servers``),
    # and it records them under ``language_servers`` — ``languages`` is a legacy
    # name it migrates via ``RENAMED_FIELDS``. Our scoping therefore no-op'd on
    # any Serena-generated project.yml (wrong key, block-style list) and only did
    # anything when it created the file itself, which is the same partial-config
    # trap as #2674 — and skipped the ``project.local.yml`` sidecar Serena writes
    # alongside. Letting Serena own that file removes a hand-maintained ext→
    # language map that duplicated its detection.
    _inject_serena_instructions(_serena_instruction_file(registrar), verbose=verbose)
    skip_reason = _serena_project_skip_reason(Path.cwd())
    if skip_reason is not None:
        if verbose:
            click.echo(f"  Serena: skipping pre-index ({skip_reason})")
        return
    _index_serena_project(verbose=verbose)


def _remove_headroom_installed_serena_mcp(registrar: Any) -> str:
    """Remove Serena MCP only if the ledger proves Headroom installed it."""
    from headroom.mcp_registry.ledger import clear_install, headroom_installed_matching

    current = registrar.get_server("serena")
    if not headroom_installed_matching(registrar.name, current):
        return "not_headroom_owned"
    if registrar.unregister_server("serena"):
        clear_install(registrar.name, "serena")
        return "removed"
    return "failed"


def _disable_serena_mcp(
    registrar: Any, *, verbose: bool = False, reason: str = "--no-serena"
) -> None:
    """Actively disable a Headroom-installed Serena entry, not merely skip it.

    Serena used to be registered by default, so a prior ``headroom wrap``
    persists a ``serena`` entry into the agent's MCP config; the agent then
    keeps launching Serena on startup. Just *skipping* registration on a later
    run leaves that stale entry in place — so this removes the entry Headroom
    installed. A user-managed Serena (absent from our ledger) is reported but
    left untouched. ``reason`` is surfaced in the message (e.g. ``--no-serena``
    or ``--code-memory none`` when the user opted out).
    """
    if not registrar.detect():
        if verbose:
            click.echo(f"  Serena MCP: {registrar.display_name} not detected — skipping")
        return

    if registrar.get_server("serena") is None:
        if verbose:
            click.echo(f"  Skipping Serena MCP ({reason})")
        return

    status = _remove_headroom_installed_serena_mcp(registrar)
    if status == "removed":
        click.echo(f"  Removed previously-installed Serena MCP ({reason})")
        click.echo(f"    restart {registrar.display_name} if it was already running")
    elif status == "not_headroom_owned":
        click.echo(
            "  Serena MCP is present but user-managed — leaving it in place "
            "(--no-serena only removes entries Headroom installed)"
        )
    else:  # "failed"
        click.echo(
            "  Serena MCP: removal failed — remove the 'serena' entry from your MCP config manually"
        )


# =============================================================================
# tokensave — retired; Serena replaced it. The helpers below only clean up a
# tokensave entry a prior release installed, so upgrading users stop launching it.
# =============================================================================


def _remove_headroom_installed_tokensave_mcp(registrar: Any) -> str:
    """Remove the tokensave MCP entry only if the ledger proves Headroom installed it."""
    from headroom.mcp_registry.ledger import clear_install, headroom_installed_matching

    current = registrar.get_server("tokensave")
    if not headroom_installed_matching(registrar.name, current):
        return "not_headroom_owned"
    if registrar.unregister_server("tokensave"):
        clear_install(registrar.name, "tokensave")
        return "removed"
    return "failed"


def _disable_tokensave_mcp(registrar: Any, *, verbose: bool = False) -> None:
    """Remove a Headroom-installed tokensave MCP entry left by a prior release.

    tokensave was retired in favour of Serena. On upgrade we actively remove the
    stale ``tokensave`` entry so the agent stops launching it, and point the user
    at the leftover on-disk artifacts (we never delete files for them). A
    user-managed entry (absent from our ledger) is reported but left in place.
    """
    if not registrar.detect():
        if verbose:
            click.echo(f"  tokensave MCP: {registrar.display_name} not detected — skipping")
        return

    if registrar.get_server("tokensave") is None:
        return

    status = _remove_headroom_installed_tokensave_mcp(registrar)
    if status == "removed":
        click.echo("  Removed retired tokensave MCP (replaced by Serena)")
        click.echo(f"    restart {registrar.display_name} if it was already running")
        click.echo(
            "    leftover files are safe to delete: the 'tokensave' binary in "
            "~/.local/bin and any '.tokensave/' folder in your projects"
        )
    elif status == "not_headroom_owned":
        click.echo(
            "  tokensave MCP is present but user-managed — leaving it in place "
            "(Headroom only removes entries it installed)"
        )
    else:  # "failed"
        click.echo(
            "  tokensave MCP: removal failed — remove the 'tokensave' entry "
            "from your MCP config manually"
        )


def _setup_coding_compressor(registrar: Any, *, serena_context: str, **kwargs: Any) -> None:
    """Set up the code-memory MCP, selected via ``--code-memory`` (default serena).

    Selection (see :func:`_resolve_code_memory`):

    * ``serena`` (default) — register Serena (mature, offline, symbol-level).
    * ``none`` — register nothing.

    Either way, any Headroom-installed ``tokensave`` entry from a prior release
    is removed (tokensave was retired in favour of Serena). The deprecated
    ``--serena`` / ``--no-serena`` flags map into the selector; user-managed MCP
    entries are always left untouched (ledger).
    """
    force = bool(kwargs.get("force"))
    verbose = bool(kwargs.get("verbose"))
    selection = _resolve_code_memory(kwargs)

    # Retire any tokensave entry a prior release installed, whatever the selection.
    _disable_tokensave_mcp(registrar, verbose=verbose)

    if selection == _CODE_MEMORY_NONE:
        _disable_serena_mcp(registrar, verbose=verbose, reason="--code-memory none")
        return

    _setup_serena_mcp(registrar, context=serena_context, verbose=verbose, force=force)


_CBM_MCP_SERVER_NAME = "codebase-memory-mcp"


# Memory MCP markers
_MEMORY_MCP_MARKER = "# --- Headroom memory MCP (auto-injected) ---"
_MEMORY_MCP_END = "# --- end Headroom memory ---"
_MEMORY_AGENTS_MARKER = "<!-- headroom:memory-instructions -->"

# Codex config injection markers
_CODEX_TOP_LEVEL_MARKER = "# --- Headroom proxy (auto-injected by headroom wrap codex) ---"
_CODEX_END_MARKER = "# --- end Headroom ---"
_CODEX_MCP_MARKER = "# --- Headroom MCP server ---"
_CODEX_MCP_END = "# --- end Headroom MCP server ---"
# File name used for the pre-wrap snapshot of the Codex config file.  The
# snapshot lets `headroom unwrap codex` restore the exact prior state, even
# if the user had their own `model_provider` / `[model_providers.*]` config
# before running wrap.
_CODEX_CONFIG_BACKUP_SUFFIX = ".headroom-backup"


def _codex_home_dir() -> Path:
    """Return Codex's config directory, respecting ``CODEX_HOME`` when set."""
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser()
    return Path.home() / ".codex"


def _codex_profile_from_args(codex_args: tuple[str, ...]) -> str | None:
    """Return the profile selected by Codex CLI arguments, if any."""
    for index, argument in enumerate(codex_args):
        if argument.startswith("--profile="):
            return argument.partition("=")[2]
        if argument in {"--profile", "-p"} and index + 1 < len(codex_args):
            return codex_args[index + 1]
    return None


def _codex_model_provider_from_args(codex_args: tuple[str, ...]) -> str | None:
    """Return the last top-level model_provider CLI override, if present."""
    provider: str | None = None
    for index, argument in enumerate(codex_args):
        override: str | None = None
        if argument.startswith("--config=") or argument.startswith("-c="):
            override = argument.partition("=")[2]
        elif argument in {"--config", "-c"} and index + 1 < len(codex_args):
            override = codex_args[index + 1]
        if override is None:
            continue
        key, separator, value = override.partition("=")
        if separator and key.strip() == "model_provider":
            provider = value.strip().strip("\"'")
    return provider


def _codex_toml_value(value: Any) -> str:
    """Serialize values accepted by Codex's TOML command-line overrides."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    raise TypeError(f"unsupported Codex config override: {type(value).__name__}")


_CODEX_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _codex_dotted_key(*parts: str) -> str:
    """Dotted ``--config`` key with segments bare wherever possible.

    Codex's override parser (observed on 0.144.x) matches dotted segments
    literally and silently ignores quoted ones, so quoting every segment made
    the whole override a no-op and traffic bypassed the proxy (#2358). Quote
    only segments that are not valid TOML bare keys (e.g. a provider name
    containing a dot), where bare emission would corrupt the path.
    """
    return ".".join(
        part if _CODEX_BARE_KEY_RE.fullmatch(part) else json.dumps(part) for part in parts
    )


def _codex_session_launch_settings(
    *, port: int, codex_args: tuple[str, ...], environ: dict[str, str]
) -> tuple[tuple[str, ...], dict[str, str], list[str]]:
    """Build process-local routing while preserving the selected provider id."""
    config_file, _ = _codex_config_paths()
    try:
        config = tomllib.loads(_read_text(config_file)) if config_file.exists() else {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise click.ClickException(
            f"could not read Codex config for session routing: {exc}"
        ) from exc

    profile_name = _codex_profile_from_args(codex_args)
    profiles = config.get("profiles", {})
    profile = profiles.get(profile_name, {}) if profile_name and isinstance(profiles, dict) else {}
    provider = _codex_model_provider_from_args(codex_args) or (
        profile.get("model_provider")
        if isinstance(profile, dict) and profile.get("model_provider")
        else config.get("model_provider", "openai")
    )
    provider = str(provider)

    project = _project_name_from_cwd()
    proxy_url = _with_project_prefix(f"http://127.0.0.1:{port}/v1", project)
    overrides: list[str] = []
    env = dict(environ)
    display = [f"OPENAI_BASE_URL={proxy_url}"]
    env["OPENAI_BASE_URL"] = proxy_url

    if provider == "openai":
        overrides.append(f"openai_base_url={_codex_toml_value(proxy_url)}")
    else:
        providers = config.get("model_providers", {})
        provider_config = providers.get(provider) if isinstance(providers, dict) else None
        if not isinstance(provider_config, dict):
            raise click.ClickException(
                f"Codex provider {provider!r} cannot be redirected without changing its identity"
            )
        upstream = provider_config.get("base_url")
        if not isinstance(upstream, str) or not upstream.strip():
            raise click.ClickException(
                f"Codex custom provider {provider!r} has no upstream base_url"
            )
        prefix = ("model_providers", provider)
        overrides.extend(
            (
                f"{_codex_dotted_key(*prefix, 'base_url')}={_codex_toml_value(proxy_url)}",
                f"{_codex_dotted_key(*prefix, 'supports_websockets')}=true",
            )
        )
        env[_UPSTREAM_BASE_URL_ENV_VAR] = upstream.rstrip("/")
        display.append(f"{_UPSTREAM_BASE_URL_ENV_VAR}={upstream.rstrip('/')}")
        overrides.append(
            f"{_codex_dotted_key(*prefix, 'env_http_headers', _UPSTREAM_BASE_URL_HEADER_NAME)}="
            f"{_codex_toml_value(_UPSTREAM_BASE_URL_ENV_VAR)}"
        )

    if project and "HEADROOM_PROJECT" not in env:
        env["HEADROOM_PROJECT"] = project
    config_args = tuple(item for override in overrides for item in ("--config", override))
    return (*config_args, *codex_args), env, display


def _offer_dangling_codex_recovery(active_home: Path) -> None:
    """Offer recovery before an interactive wrap creates more Codex state."""
    if not sys.stdin.isatty():
        return
    from headroom.providers.codex.recovery import (
        discover_dangling_homes,
        recover_codex_home,
    )

    candidates = [path for path in discover_dangling_homes() if path != active_home]
    if not candidates:
        return
    click.echo("\nFound Codex state left by an earlier Headroom temporary home:")
    for candidate in candidates:
        click.echo(f"  {candidate}")
    if not click.confirm(
        "Back up both homes and recover this state before launching Codex?",
        default=True,
    ):
        click.echo("Skipped recovery. Run `headroom recover codex` to recover it later.")
        return
    for candidate in candidates:
        report = recover_codex_home(source=candidate, target=active_home)
        click.echo(f"Recovered Codex state. Backup retained at {report.backup_dir}")


def _codex_config_paths() -> tuple[Path, Path]:
    """Return ``(config_file, backup_file)`` paths for the Codex TOML config."""
    config_dir = _codex_home_dir()
    config_file = config_dir / "config.toml"
    backup_file = config_dir / f"config.toml{_CODEX_CONFIG_BACKUP_SUFFIX}"
    return config_file, backup_file


def _strip_codex_headroom_blocks(
    content: str,
    *,
    remove_mcp: bool = False,
    remove_named_mcp: bool = True,
) -> str:
    """Remove all Headroom-managed blocks from a Codex ``config.toml`` string.

    Returns the cleaned content.  Safe to call on content that never contained
    any markers — it will be returned effectively unchanged (only trailing
    whitespace is normalized).
    """
    import re

    def _remove_marker_span(text: str, start_marker: str, end_marker: str) -> str:
        while start_marker in text and end_marker in text:
            start = text.index(start_marker)
            end_idx = text.index(end_marker, start)
            if end_idx < start:
                break
            end = end_idx + len(end_marker)
            text = text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n")
        text = text.replace(start_marker + "\n", "")
        text = text.replace(end_marker + "\n", "")
        return text

    # Remove any top-level-marker → end-marker span, possibly repeated.
    content = _remove_marker_span(content, _CODEX_TOP_LEVEL_MARKER, _CODEX_END_MARKER)

    if remove_mcp:
        # Remove Headroom-managed MCP blocks written by `wrap codex`.
        content = _remove_marker_span(content, _CODEX_MCP_MARKER, _CODEX_MCP_END)
        if remove_named_mcp:
            content = re.sub(
                r"(?ms)^# --- Headroom MCP server: [^\n]+ ---\n.*?"
                r"^# --- end Headroom MCP server: [^\n]+ ---\n?",
                "",
                content,
            )
        content = _remove_marker_span(content, _MEMORY_MCP_MARKER, _MEMORY_MCP_END)

    # Strip any leftover top-level keys that older (or crashed) versions of
    # `wrap codex` may have written outside the marker block.
    content = re.sub(r'(?m)^[ \t]*model_provider[ \t]*=[ \t]*"headroom"[ \t]*\r?\n', "", content)
    content = re.sub(
        r'(?m)^[ \t]*openai_base_url[ \t]*=[ \t]*"http://127\.0\.0\.1:\d+/v1"[ \t]*\r?\n',
        "",
        content,
    )

    # Strip any orphaned `[model_providers.headroom]` table with the fields we
    # write.  We only remove it if the table is recognisably ours (base_url
    # mentions localhost and a Headroom proxy port).  This protects users who
    # happen to have a differently configured `headroom` provider.
    orphan_headroom_table = re.compile(
        r"(?ms)^\[model_providers\.headroom\][^\[]*?"
        r'base_url[ \t]*=[ \t]*"http://127\.0\.0\.1:\d+/v1"[^\[]*?'
        r"(?=^\[|\Z)"
    )
    content = orphan_headroom_table.sub("", content)

    return content.lstrip("\n").rstrip() + "\n" if content.strip() else ""


# Top-level bare keys we redirect to headroom values when the user already
# has them set.  Match the entire line (including any trailing comment) so
# we can rewrite it cleanly.  Bare keys must precede any [section] in TOML,
# so a `^` anchor combined with `^[ \t]*key` is sufficient — table lines
# start with `[`, not with the key name.
_REDIRECTABLE_KEYS: tuple[str, ...] = ("model_provider", "openai_base_url")


def _strip_existing_codex_headroom_provider_table(content: str) -> str:
    """Remove a pre-existing ``[model_providers.headroom]`` table before wrap."""
    if "[model_providers.headroom]" not in content:
        return content

    import re  # local import to match surrounding helper convention

    provider_table = re.compile(
        r"(?ms)^[ \t]*\[model_providers\.headroom\][^\n]*\n.*?(?=^[ \t]*\[|\Z)"
    )
    content = provider_table.sub("", content)
    return content.lstrip("\n").rstrip() + "\n" if content.strip() else ""


def _redirect_existing_top_level_keys(content: str, port: int) -> str:
    """Rewrite user-defined top-level keys so wrap does not create duplicates.

    Codex's ``config.toml`` rejects duplicate top-level keys (TOML spec),
    which would break ``codex`` startup after ``headroom wrap codex`` runs
    on a config that already declares its own ``model_provider`` or
    ``openai_base_url``.

    For each redirectable key, if the user's line already sets it, replace
    the value with the headroom one and append ``# was: <original-value>``
    so the user can still see and recover their previous setting.  The
    snapshot taken in ``_snapshot_codex_config_if_unwrapped`` ensures the
    pre-wrap file can be restored byte-for-byte on ``headroom unwrap
    codex``.

    Returns the modified content.  If no redirectable keys are present,
    the content is returned unchanged and the caller should fall back to
    prepending the marker-delimited top-level block (current behavior).
    """
    import re  # local import to match the module's existing convention

    if not content.strip():
        return content

    def _make_replacer(current_key: str, current_port: int) -> Callable[[re.Match[str]], str]:
        def _replace(match: re.Match[str]) -> str:
            original_value = match.group("value")
            if current_key == "model_provider":
                new_value = "headroom"
            else:  # openai_base_url
                new_value = f"http://127.0.0.1:{current_port}/v1"
            if original_value == new_value:
                return match.group(0)
            # Keep the user's original value in a trailing comment so they
            # can see what was changed.  This is metadata, not a TOML
            # duplicate.
            return f'{current_key} = "{new_value}"  # was: {original_value}'

        return _replace

    redirected = content
    for key in _REDIRECTABLE_KEYS:
        pattern = re.compile(rf'(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*"(?P<value>[^"\n]*)"[^\n]*')
        redirected = pattern.sub(_make_replacer(key, port), redirected, count=1)
    return redirected


def _has_redirectable_top_level_key(content: str, key: str) -> bool:
    """Return True if ``content`` declares ``key = "..."`` as a top-level key."""
    import re  # local import to match the module's existing convention

    pattern = re.compile(rf'(?m)^[ \t]*{key}[ \t]*=[ \t]*"[^"\n]*"')
    return pattern.search(content) is not None


def _codex_config_has_headroom_markers(content: str) -> bool:
    """Return whether a Codex config already contains wrap-owned markers."""
    managed_markers = (
        _CODEX_TOP_LEVEL_MARKER,
        _CODEX_END_MARKER,
        _CODEX_MCP_MARKER,
        _MEMORY_MCP_MARKER,
    )
    return any(marker in content for marker in managed_markers)


def _snapshot_codex_config_if_unwrapped(config_file: Path, backup_file: Path) -> None:
    """Snapshot ``config.toml`` to ``backup_file`` before the first injection.

    Called as the first step of every Headroom injection into Codex's
    ``config.toml``.  Guarantees that ``headroom unwrap codex`` can restore the
    user's original file byte-for-byte.

    Rules:

    * If the backup already exists, leave it alone — we only snapshot the
      *pre-wrap* state, so running wrap repeatedly must not clobber it.
    * If the config file doesn't exist yet, there's nothing to back up; unwrap
      will remove the file entirely instead of restoring a snapshot.
    * If the config already contains any Headroom-managed Codex marker, a wrap
      run is already active: do not snapshot the injected state.
    """
    if backup_file.exists():
        return
    if not config_file.exists():
        return
    try:
        content = _read_text(config_file)
    except OSError:
        return
    if _codex_config_has_headroom_markers(content):
        return
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, backup_file)


# Canonical casing for the proxy's per-project savings header (matched
# case-insensitively by headroom.proxy.project_context.PROJECT_HEADER).
_PROJECT_HEADER_NAME = "X-Headroom-Project"


def _project_name_from_cwd() -> str | None:
    """Project label for X-Headroom-Project: basename of the launch directory.

    Non-ASCII characters are percent-encoded (RFC 3986) so the header value
    stays within the visible-ASCII range required by RFC 7230.  The proxy
    decodes the value in sanitize_project_name before storing it.
    """
    name = Path.cwd().name.strip()
    if not name:
        return None
    return urllib.parse.quote(name, safe="-_.() ")


def _apply_project_header_env(env: dict[str, str]) -> None:
    """Inject X-Headroom-Project into ``ANTHROPIC_CUSTOM_HEADERS``.

    Claude Code reads ``ANTHROPIC_CUSTOM_HEADERS`` as newline-separated
    ``Name: value`` lines and attaches them to every API request; the
    Headroom proxy uses the X-Headroom-Project header for per-project
    savings attribution.  An existing user-supplied x-headroom-project
    header (any casing) always wins — we never duplicate or overwrite it,
    and any other user headers are preserved by appending.
    """
    project = _project_name_from_cwd()
    if not project:
        return
    header_line = f"{_PROJECT_HEADER_NAME}: {project}"
    existing = env.get("ANTHROPIC_CUSTOM_HEADERS")
    if existing:
        for line in existing.splitlines():
            name = line.split(":", 1)[0].strip()
            if name.lower() == _PROJECT_HEADER_NAME.lower():
                return  # user override wins
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"{existing}\n{header_line}"
    else:
        env["ANTHROPIC_CUSTOM_HEADERS"] = header_line


# Codex's own built-in providers plus Headroom's injected one — never treated
# as a "custom upstream to preserve" by _detect_custom_codex_upstream_base_url.
_CODEX_BUILTIN_PROVIDER_NAMES = frozenset({"openai", "anthropic", "azure", "headroom"})

# Header carrying a preserved custom upstream (freemodel.dev, LiteLLM, vLLM,
# ...) so the proxy forwards to it instead of the hardcoded OpenAI default.
# Codex's env_http_headers only accepts an env-var *name* per header (not a
# literal value), so the detected URL is exported into this env var by the
# `wrap codex` launch path — see its use in `codex()` below.
_UPSTREAM_BASE_URL_HEADER_NAME = "X-Headroom-Base-Url"
_UPSTREAM_BASE_URL_ENV_VAR = "HEADROOM_CODEX_UPSTREAM_BASE_URL"


def _codex_custom_provider_base_urls(content: str) -> dict[str, str]:
    """Return ``{provider_name: base_url}`` for user-declared custom providers.

    Excludes Codex's built-ins (``openai``/``anthropic``/``azure``) and
    Headroom's own ``headroom`` table.  A table with no ``base_url`` line, or
    one already pointing at Headroom's own localhost proxy (a leftover from a
    prior wrap this pass hasn't stripped yet), is excluded too.
    """
    import re

    tables: dict[str, str] = {}
    for match in re.finditer(
        r"(?ms)^[ \t]*\[model_providers\.(?P<name>[^\]\s]+)\][ \t]*\n"
        r"(?P<body>.*?)(?=^[ \t]*\[|\Z)",
        content,
    ):
        name = match.group("name")
        if name in _CODEX_BUILTIN_PROVIDER_NAMES:
            continue
        base_match = re.search(
            r'(?m)^[ \t]*base_url[ \t]*=[ \t]*"(?P<url>[^"\n]*)"', match.group("body")
        )
        if not base_match:
            continue
        url = base_match.group("url").strip().rstrip("/")
        if not url or url.startswith(("http://127.0.0.1", "http://localhost")):
            continue
        tables[name] = url
    return tables


def _detect_custom_codex_upstream_base_url(content: str) -> str | None:
    """Return a user-configured custom provider ``base_url`` to preserve, if any.

    Codex lets users declare OpenAI-compatible gateways (LiteLLM, vLLM,
    freemodel.dev, ...) under ``[model_providers.<name>]`` and select one via
    the top-level ``model_provider`` key. Before this, ``headroom wrap codex``
    unconditionally pointed the proxy's upstream OpenAI route at
    ``api.openai.com``, silently discarding that selection — the user's
    gateway API key then gets sent to OpenAI, which rejects it (#1614).

    If the top-level ``model_provider`` names one of the detected custom
    tables, that selection wins unambiguously. This also covers the
    already-wrapped case: once wrap has run once, the top-level key reads
    ``model_provider = "headroom"  # was: <original>`` (see
    ``_redirect_existing_top_level_keys``), so the original selection is
    recovered from that trailing comment on re-wrap / port changes.

    Falls back to the sole candidate when exactly one custom table exists
    and there is no (or no matching) top-level selection — the common case
    from the bug report, where the table is declared but selection happens
    via ``--profile`` rather than a static top-level key. Returns ``None``
    when there are multiple, un-selected candidates (ambiguous — guessing
    wrong is worse than the prior default behavior) or none at all.
    """
    import re

    candidates = _codex_custom_provider_base_urls(content)
    if not candidates:
        return None

    selected = re.search(
        r'(?m)^[ \t]*model_provider[ \t]*=[ \t]*"(?P<name>[^"\n]*)"'
        r"(?:[ \t]*#[ \t]*was:[ \t]*(?P<was>[^\r\n]*))?",
        content,
    )
    if selected:
        was = (selected.group("was") or "").strip()
        chosen = was or selected.group("name")
        if chosen in candidates:
            return candidates[chosen]

    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return None


def _inject_codex_provider_config(port: int) -> str | None:
    """Inject a Headroom model provider into Codex's config.toml.

    Two keys need to be in effect for the proxy to route all traffic:

    * ``model_provider = "headroom"`` — selects the custom provider for
      API-key mode traffic.
    * ``openai_base_url = "http://127.0.0.1:{port}/v1"`` — overrides the
      built-in ``openai`` provider's base URL.  This is the critical key for
      **subscription (ChatGPT plan) users**: Codex detects subscription auth
      and routes through the built-in ``openai`` provider regardless of
      ``model_provider``, so without this override it bypasses the proxy and
      hits ``https://chatgpt.com/backend-api/codex`` directly.

    If the user has not already declared these top-level keys, they are
    added in a marker-delimited block at the top of the file.  If the
    user *has* declared one or both, the existing lines are rewritten
    in place to the headroom values (with the previous value kept in a
    ``# was: …`` trailing comment) so the resulting file stays TOML-valid
    — TOML rejects duplicate top-level keys, which would break
    ``codex`` startup.

    Safe to call multiple times — the injected block is fully replaced on
    each call, so re-running with a different ``port`` updates the config.
    Before the first injection, the pre-wrap file is snapshotted to
    ``config.toml.headroom-backup`` so ``headroom unwrap codex``
    can restore it byte-for-byte.

    Returns the custom upstream ``base_url`` preserved from an existing
    ``[model_providers.*]`` table, if one was detected (#1614); ``None``
    otherwise. Callers that go on to launch Codex should export this value
    into ``HEADROOM_CODEX_UPSTREAM_BASE_URL`` (the injected
    ``env_http_headers`` entry maps it to the ``X-Headroom-Base-Url`` header,
    which the proxy's OpenAI HTTP handlers honor over the hardcoded
    ``api.openai.com`` default) — see its use in ``codex()`` below.
    """
    config_file, backup_file = _codex_config_paths()
    config_dir = config_file.parent

    # Detect an existing custom OpenAI-compatible provider BEFORE building the
    # injected block below, so it can be preserved as the upstream the proxy
    # forwards to instead of silently rerouting to api.openai.com (#1614).
    # Best-effort: any read/parse failure just means nothing is preserved,
    # matching prior behavior.
    custom_upstream_base_url: str | None = None
    if config_file.exists():
        try:
            custom_upstream_base_url = _detect_custom_codex_upstream_base_url(
                _read_text(config_file)
            )
        except OSError:
            custom_upstream_base_url = None

    # The injected content is split into two self-contained, marker-delimited
    # blocks: a top-level key block (at the start of the file, because bare
    # TOML keys must precede any [section]) and a provider-table block (at
    # the end).  Each block has its own matching begin/end marker pair so
    # stripping them is unambiguous and never consumes user content that
    # happens to sit between the two.  The top-level block is built
    # dynamically below — it contains only keys the user has not already
    # declared (we rewrite the existing ones in place to avoid TOML
    # duplicate-key errors).
    # Emit requires_openai_auth only for ChatGPT-OAuth users (restores the
    # account menu); omitting it for API-key users avoids forcing an OAuth
    # login (#406).
    requires_openai_auth = (
        "requires_openai_auth = true\n" if codex_uses_chatgpt_auth(config_dir / "auth.json") else ""
    )
    # Per-project savings: Codex sends the X-Headroom-Project header only
    # when the mapped env var (HEADROOM_PROJECT, set by `headroom wrap
    # codex`) exists at Codex runtime. When a custom upstream was detected,
    # add a second entry so Codex also sends X-Headroom-Base-Url — the proxy
    # forwards there instead of api.openai.com (#1614).
    env_http_headers_map = {_PROJECT_HEADER_NAME: "HEADROOM_PROJECT"}
    if custom_upstream_base_url:
        env_http_headers_map[_UPSTREAM_BASE_URL_HEADER_NAME] = _UPSTREAM_BASE_URL_ENV_VAR
    env_http_headers_toml = ", ".join(f'"{k}" = "{v}"' for k, v in env_http_headers_map.items())
    provider_section = (
        f"{_CODEX_TOP_LEVEL_MARKER}\n"
        "[model_providers.headroom]\n"
        'name = "OpenAI via Headroom proxy"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        f"supports_websockets = true\n"
        f"{requires_openai_auth}"
        # Inline table keeps the key inside this section so
        # _strip_codex_headroom_blocks removes it with the rest of the block.
        f"env_http_headers = {{ {env_http_headers_toml} }}\n"
        f"{_CODEX_END_MARKER}\n"
    )

    # The two redirectable keys and their headroom target values.
    _REDIRECT_TARGETS = {
        "model_provider": "headroom",
        "openai_base_url": f"http://127.0.0.1:{port}/v1",
    }

    def _build_top_level_block(user_content: str) -> str:
        """Build a marker-delimited block containing only the keys the user
        has not already declared at the top level.  For keys the user
        *has* declared, the in-place rewrite below handles them.
        """
        lines = [_CODEX_TOP_LEVEL_MARKER]
        for key, value in _REDIRECT_TARGETS.items():
            if _has_redirectable_top_level_key(user_content, key):
                continue
            lines.append(f'{key} = "{value}"')
        if len(lines) == 1:
            # User already declared every redirectable key — no marker
            # block needed (it would be empty).
            return ""
        lines.append(_CODEX_END_MARKER)
        return "\n".join(lines) + "\n"

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the pre-wrap state before touching anything.  No-op if the
        # config is already wrapped, is missing, or we've already snapshotted.
        _snapshot_codex_config_if_unwrapped(config_file, backup_file)

        if config_file.exists():
            content = _read_text(config_file)
            # Remove any prior Headroom-managed blocks before re-injecting so
            # the operation is idempotent and supports port changes.
            content = _strip_codex_headroom_blocks(content)
            content = _strip_existing_codex_headroom_provider_table(content)

            # Bare top-level keys must precede any [section] in TOML, and
            # TOML rejects duplicate top-level keys.  Rewrite any existing
            # top-level ``model_provider`` / ``openai_base_url`` in place
            # to the headroom values; for keys the user has not declared,
            # add them in a marker-delimited block at the top of the
            # file.  The original values are kept in a trailing ``# was:
            # <value>`` comment, and the snapshot mechanism guarantees
            # byte-for-byte restoration on unwrap.
            user_content = content.strip()
            if user_content:
                redirected = _redirect_existing_top_level_keys(user_content, port)
                top_block = _build_top_level_block(user_content)
                if top_block:
                    content = top_block + "\n" + redirected + "\n\n" + provider_section
                else:
                    content = redirected + "\n\n" + provider_section
            else:
                # Empty user content — no keys to rewrite in place; emit
                # the full marker block with both redirectable keys.
                content = (
                    f"{_CODEX_TOP_LEVEL_MARKER}\n"
                    f'model_provider = "{_REDIRECT_TARGETS["model_provider"]}"\n'
                    f'openai_base_url = "{_REDIRECT_TARGETS["openai_base_url"]}"\n'
                    f"{_CODEX_END_MARKER}\n"
                    f"\n{provider_section}"
                )
        else:
            # No config file yet — same as the empty-content path.
            content = (
                f"{_CODEX_TOP_LEVEL_MARKER}\n"
                f'model_provider = "{_REDIRECT_TARGETS["model_provider"]}"\n'
                f'openai_base_url = "{_REDIRECT_TARGETS["openai_base_url"]}"\n'
                f"{_CODEX_END_MARKER}\n"
                f"\n{provider_section}"
            )

        _write_text(config_file, content)
        click.echo(f"  Codex config: injected Headroom provider (WS + HTTP) into {config_file}")
        if custom_upstream_base_url:
            click.echo(
                f"  Codex config: preserving existing custom upstream "
                f"{custom_upstream_base_url} (from a pre-existing [model_providers.*] "
                "base_url)"
            )
        # Pull existing native threads into the headroom-provider menu so Codex's
        # history list stays whole once it routes through Headroom. Best-effort.
        retag_to_headroom(_codex_home_dir())
    except Exception as e:
        click.echo(f"  Warning: could not update Codex config: {e}")
        return None

    return custom_upstream_base_url


def _restore_codex_provider_config() -> tuple[str, Path]:
    """Undo ``_inject_codex_provider_config`` for the active Codex config file.

    Returns a tuple of ``(status, config_file)`` where status is one of:

    * ``"restored"`` — a pre-wrap backup existed and was restored; backup
      file has been removed.
    * ``"cleaned"``  — no backup existed, but the Headroom-managed block was
      found and stripped out (preserving surrounding user content).
    * ``"removed"``  — the config file only contained Headroom-managed
      content (created by wrap) and has been deleted.
    * ``"noop"``     — nothing to undo; no Headroom marker and no backup.
    """
    config_file, backup_file = _codex_config_paths()

    # Case 1: pre-wrap snapshot exists — restore it exactly.
    if backup_file.exists():
        shutil.copy2(backup_file, config_file)
        backup_file.unlink()
        return "restored", config_file

    # Case 2: no backup, but config file exists and has markers — strip them.
    if config_file.exists():
        original = _read_text(config_file)
        if _codex_config_has_headroom_markers(original):
            # Without a backup, only remove named MCP blocks when this file
            # also carries wrap-owned provider markers from a full wrap.
            remove_named_mcp = any(
                marker in original
                for marker in (
                    _CODEX_TOP_LEVEL_MARKER,
                    _CODEX_END_MARKER,
                    _CODEX_MCP_MARKER,
                    _CODEX_MCP_END,
                )
            )
            cleaned = _strip_codex_headroom_blocks(
                original,
                remove_mcp=True,
                remove_named_mcp=remove_named_mcp,
            )
            if not cleaned.strip():
                # Nothing left but Headroom content — remove the file entirely
                # so Codex falls back to its default config.
                config_file.unlink()
                return "removed", config_file
            _write_text(config_file, cleaned)
            return "cleaned", config_file

    # Nothing to undo.
    return "noop", config_file


_WRAP_BANNER_INNER_WIDTH = 47


def _print_wrap_banner(agent: str) -> None:
    """Print a centered ``HEADROOM WRAP: <AGENT>`` banner.

    Every Pattern-B wrap subcommand (proxy-only + watcher loop) used to
    inline this 3-line box by hand with hand-padded spaces, which made
    title-length changes silently miscenter the title. Compute padding
    here so adding a 9th agent just works.
    """
    title = f"HEADROOM WRAP: {agent.upper()}"
    pad_total = _WRAP_BANNER_INNER_WIDTH - len(title)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    click.echo()
    click.echo("  ╔" + "═" * _WRAP_BANNER_INNER_WIDTH + "╗")
    click.echo(f"  ║{' ' * pad_left}{title}{' ' * pad_right}║")
    click.echo("  ╚" + "═" * _WRAP_BANNER_INNER_WIDTH + "╝")
    click.echo()


def _run_proxy_only_watcher(
    *,
    agent_label: str,
    port: int,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    agent_type: str,
    print_setup_lines: Callable[[int], None],
    anthropic_api_url: str | None = None,
    openai_api_url: str | None = None,
    copilot_api_token: str | None = None,
    copilot_refresh_oauth_token: str | None = None,
    copilot_api_token_expires_at: float | None = None,
) -> None:
    """Shared scaffolding for proxy-only wrap subcommands (no child binary launch).

    Pattern-B subcommands (cursor / cline / continue) all start the proxy,
    print agent-specific configuration instructions, then block until
    Ctrl+C. This helper unifies that lifecycle so the per-agent diff is
    just the ``print_setup_lines`` callback.

    The Pattern-A subcommands (aider / copilot / codex / goose / openhands)
    launch a child binary via ``_launch_tool`` instead and never come
    through here. ``_launch_tool`` owns the proxy lifecycle on that path.
    """
    proxy_holder: list[subprocess.Popen | None] = [None]
    port_holder: list[int] = [port]
    cleanup = _make_cleanup(proxy_holder, port_holder)

    def _signal_shutdown(signum: int, frame: Any) -> None:
        cleanup(signum, frame)
        # cleanup alone leaves the watcher loop alive long enough to observe
        # the intentionally terminated proxy and report a false crash. Raise
        # into its normal Ctrl-C path so shutdown exits successfully.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_shutdown)
    signal.signal(signal.SIGTERM, _signal_shutdown)
    # Windows exposes Ctrl+Break as SIGBREAK rather than SIGINT. Test runners,
    # IDE terminals, and process supervisors commonly use Ctrl+Break to target
    # a newly created process group, so route it through the same graceful
    # cleanup path as an interactive Ctrl+C.
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_shutdown)

    try:
        _print_wrap_banner(agent_label)
        _register_proxy_client(port)
        proxy_holder[0], actual_port = _ensure_proxy(
            port,
            no_proxy,
            learn=learn,
            memory=memory,
            agent_type=agent_type,
            anthropic_api_url=anthropic_api_url,
            openai_api_url=openai_api_url,
            copilot_api_token=copilot_api_token,
            copilot_refresh_oauth_token=copilot_refresh_oauth_token,
            copilot_api_token_expires_at=copilot_api_token_expires_at,
        )
        if actual_port != port:
            _unregister_proxy_client(port)
            _register_proxy_client(actual_port)
        port_holder[0] = actual_port
        _push_runtime_env(actual_port, no_proxy)
        click.echo()
        print_setup_lines(actual_port)
        click.echo()
        click.echo("  Press Ctrl+C to stop the proxy.")
        click.echo()

        try:
            while True:
                time.sleep(1)
                proc = proxy_holder[0]
                if proc and proc.poll() is not None:
                    click.echo("  Proxy process exited unexpectedly.")
                    raise SystemExit(1)
        except KeyboardInterrupt:
            click.echo("\n  Shutting down...")
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        cleanup()


def _inject_memory_mcp_config(user_id: str) -> None:
    """Register headroom memory as an MCP server in Codex's config.toml.

    Idempotent — replaces existing section if present.
    """
    import sys

    config_file, _ = _codex_config_paths()
    config_dir = config_file.parent

    # Use forward slashes in TOML paths (works on all platforms, avoids
    # backslash escaping issues on Windows)
    python_bin = sys.executable.replace("\\", "/")
    mcp_section = (
        f"\n{_MEMORY_MCP_MARKER}\n"
        f"[mcp_servers.headroom_memory]\n"
        f'command = "{python_bin}"\n'
        f'args = ["-m", "headroom.memory.mcp_server", "--user", "{user_id}"]\n'
        f"startup_timeout_sec = 30\n"
        f"tool_timeout_sec = 30\n"
        f"{_MEMORY_MCP_END}\n"
    )

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot pre-wrap state before touching config.toml so `unwrap codex`
        # can fully restore it even when only `--memory` (not a full provider
        # injection) was used.
        _, backup_file = _codex_config_paths()
        _snapshot_codex_config_if_unwrapped(config_file, backup_file)

        if config_file.exists():
            content = _read_text(config_file)
            if _MEMORY_MCP_MARKER in content:
                start = content.index(_MEMORY_MCP_MARKER)
                end = content.index(_MEMORY_MCP_END) + len(_MEMORY_MCP_END)
                content = content[:start].rstrip("\n") + mcp_section + content[end:].lstrip("\n")
            else:
                content = content.rstrip() + "\n" + mcp_section
        else:
            content = mcp_section

        _write_text(config_file, content)
        click.echo(f"  Memory MCP: registered in {config_file}")
    except Exception as e:
        click.echo(f"  Warning: could not register memory MCP: {e}")


def _inject_memory_agents_md(file_path: Path) -> bool:
    """Inject memory usage guidance into AGENTS.md.

    Idempotent — skips if marker already present.
    """
    memory_block = (
        f"{_MEMORY_AGENTS_MARKER}\n"
        "## Memory\n\n"
        "Use the `headroom_memory` MCP server for persistent cross-session knowledge.\n\n"
        "**Before** answering questions about prior decisions, conventions, project context,\n"
        "architecture, user preferences, org info, codenames, debugging history, or anything\n"
        "from past sessions — call `memory_search` first.\n\n"
        "**After** making durable decisions, discovering conventions, or learning important\n"
        "facts — call `memory_save` to persist them for future sessions.\n\n"
        "Memory is your first source of truth for anything not visible in the current conversation.\n"
    )

    if file_path.exists():
        existing = _read_text(file_path)
        if _MEMORY_AGENTS_MARKER in existing:
            return True  # Already injected
        _append_text(file_path, "\n\n" + memory_block)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(file_path, memory_block)

    click.echo(f"  Memory guidance injected into {file_path.name}")
    return True


def _resolve_copilot_provider_type(backend: str | None, provider_type: str) -> str:
    """Resolve Copilot BYOK provider type for the current proxy backend."""
    return _copilot_resolve_provider_type(backend, provider_type)


def _query_proxy_config(port: int) -> dict[str, Any] | None:
    """Query the running proxy's feature configuration via /health.

    Returns a dict with keys like backend, optimize, cache, rate_limit,
    memory, learn, code_graph, pid.  Returns None if unreachable or the
    response lacks a config block.
    """
    return _copilot_query_proxy_config(port)


def _query_proxy_health(port: int) -> dict[str, Any] | None:
    """Query the running proxy's full /health payload."""
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _proxy_health_config(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the config block from a Headroom /health payload."""
    if payload is None:
        return None
    config = payload.get("config")
    return config if isinstance(config, dict) else None


def _env_bool_value(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _agent_savings_config_mismatches(
    running_config: dict[str, Any],
    agent_type: str,
) -> list[str]:
    """Return restart reasons when a running proxy lacks target agent savings."""

    if agent_type not in _AGENT_SAVINGS_TARGET_AGENTS:
        return []

    if _wrap_agent_savings_profile(agent_type) is None:
        return []

    desired_env = os.environ.copy()
    apply_agent_savings_env_defaults(desired_env)
    checks: tuple[tuple[str, str, str, str], ...] = (
        ("HEADROOM_SAVINGS_PROFILE", "savings_profile", "savings-profile", "str"),
        ("HEADROOM_TARGET_RATIO", "target_ratio", "target-ratio", "float"),
        (
            "HEADROOM_COMPRESS_USER_MESSAGES",
            "compress_user_messages",
            "compress-user-messages",
            "bool",
        ),
        (
            "HEADROOM_COMPRESS_SYSTEM_MESSAGES",
            "compress_system_messages",
            "compress-system-messages",
            "bool",
        ),
        ("HEADROOM_PROTECT_RECENT", "protect_recent", "protect-recent", "int"),
        (
            "HEADROOM_PROTECT_ANALYSIS_CONTEXT",
            "protect_analysis_context",
            "protect-analysis-context",
            "bool",
        ),
        ("HEADROOM_MIN_TOKENS", "min_tokens_to_crush", "min-tokens", "int"),
        ("HEADROOM_MAX_ITEMS", "max_items_after_crush", "max-items", "int"),
        (
            "HEADROOM_SMART_CRUSHER_COMPACTION",
            "smart_crusher_with_compaction",
            "smart-crusher-compaction",
            "bool",
        ),
        ("HEADROOM_ACCURACY_GUARD", "accuracy_guard", "accuracy-guard", "str"),
    )

    mismatches: list[str] = []
    for env_key, config_key, label, value_type in checks:
        expected = desired_env.get(env_key)
        if expected is None:
            continue
        actual = running_config.get(config_key)
        try:
            if value_type == "float":
                matches = actual is not None and abs(float(actual) - float(expected)) < 1e-9
            elif value_type == "int":
                matches = actual is not None and int(actual) == int(expected)
            elif value_type == "bool":
                matches = actual is not None and bool(actual) is _env_bool_value(expected)
            else:
                matches = str(actual or "").strip().lower() == expected.strip().lower()
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches.append(label)

    return mismatches


def _proxy_active_session_count(payload: dict[str, Any] | None) -> int:
    """Return active session count from /health runtime metadata."""
    if payload is None:
        return 0
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return 0
    websocket_sessions = runtime.get("websocket_sessions")
    if not isinstance(websocket_sessions, dict):
        return 0
    counts = []
    for key in ("active_sessions", "active_relay_tasks"):
        value = websocket_sessions.get(key, 0)
        if isinstance(value, int):
            counts.append(value)
    return max(counts, default=0)


def _normalize_proxy_api_url(url: object) -> str | None:
    """Normalize configured upstream URLs for running-proxy comparisons."""
    if not isinstance(url, str):
        return None
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized or None


def _proxy_version(payload: dict[str, Any] | None) -> str | None:
    """Return the running proxy version when it exposes one."""
    if payload is None:
        return None
    version = payload.get("version")
    return version if isinstance(version, str) and version else None


def _proxy_needs_version_restart(payload: dict[str, Any] | None) -> bool:
    """Return True when a running Headroom proxy uses a different package version."""
    running_version = _proxy_version(payload)
    running_release = _normalize_release_version(running_version)
    # -dev is a display marker for source builds; compare the base release so a
    # dev CLI still restarts a stale proxy on a real version difference.
    current_release = _normalize_release_version(_HEADROOM_VERSION.removesuffix("-dev"))
    return (
        running_release is not None
        and current_release is not None
        and running_release != current_release
    )


def _detect_running_proxy_backend(port: int) -> str | None:
    """Read the backend of an already-running proxy from its health endpoint."""
    return _copilot_detect_running_proxy_backend(port)


def _kill_proxy_by_pid(pid: int, port: int) -> bool:
    """Terminate a proxy process by PID and wait for the port to free up.

    Sends SIGTERM first, falls back to SIGKILL after 5 seconds.
    Returns True if the port is free afterwards, False otherwise.
    """
    if sys.platform == "win32":
        # ``os.kill(..., SIGTERM)`` only targets one Windows process.  The
        # native proxy launcher can own a serving child, so terminating the
        # reported PID alone may leave that child bound to the port.  Walk the
        # verified Headroom process tree, matching the existing Serena cleanup
        # strategy used elsewhere in this module.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:
            pass
        for _ in range(50):
            time.sleep(0.1)
            if not _check_proxy(port):
                return True
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        click.echo(f"  Warning: No permission to kill proxy PID {pid}")
        return False
    except (ProcessLookupError, OSError, SystemError):
        pass

    # Wait for port to free (up to 5 seconds)
    for _ in range(50):
        time.sleep(0.1)
        if not _check_proxy(port):
            return True

    # SIGTERM didn't work — escalate to SIGKILL (Unix) or terminate (Windows)
    try:
        _kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(pid, _kill_signal)
    except (ProcessLookupError, PermissionError, OSError, SystemError):
        pass

    for _ in range(20):
        time.sleep(0.1)
        if not _check_proxy(port):
            return True

    return False


def _stop_local_proxy_for_unwrap(port: int) -> str:
    """Stop a local Headroom proxy for durable unwrap commands.

    Returns a status string:
      * ``"stopped"``: a Headroom proxy was identified and stopped.
      * ``"not_running"``: nothing is listening on the requested port.
      * ``"unidentified"``: something is listening, but it did not expose
        Headroom's health/config payload, so we did not kill it.
      * ``"no_pid"``: the service looked like Headroom but did not expose a PID.
      * ``"failed"``: a PID was found but the port stayed bound after stop.
    """

    if not _check_proxy(port):
        return "not_running"

    running_config = _query_proxy_config(port)
    if running_config is None:
        return "unidentified"

    proxy_pid = running_config.get("pid")
    if proxy_pid is None:
        return "no_pid"

    try:
        pid = int(proxy_pid)
    except (TypeError, ValueError):
        return "no_pid"

    return "stopped" if _kill_proxy_by_pid(pid, port) else "failed"


def _manifest_targets_claude(manifest: Any) -> bool:
    targets = getattr(manifest, "targets", None)
    if isinstance(targets, list) and any(
        str(target).strip().lower() == "claude" for target in targets
    ):
        return True
    tool_envs = getattr(manifest, "tool_envs", None)
    if isinstance(tool_envs, dict) and any(
        str(name).strip().lower() == "claude" for name in tool_envs
    ):
        return True
    mutations = getattr(manifest, "mutations", None)
    if isinstance(mutations, list):
        for mutation in mutations:
            if str(getattr(mutation, "target", "")).strip().lower() == "claude":
                return True
    return False


def _can_unwrap_stop_persistent_manifest(manifest: Any) -> bool:
    if not _manifest_targets_claude(manifest):
        return False
    supervisor_kind = str(getattr(manifest, "supervisor_kind", "")).strip().lower()
    return supervisor_kind in {"", "none", "service"}


def _same_port_claude_env_keys(port: int) -> list[str]:
    matches: list[str] = []
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
    ):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        try:
            parsed = urllib.parse.urlparse(raw)
        except Exception:
            continue
        try:
            parsed_port = parsed.port
        except ValueError:
            continue
        if parsed_port != port:
            continue
        host = (parsed.hostname or "").strip().lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            continue
        matches.append(key)
    return matches


def _stop_persistent_manifest_for_claude_unwrap(manifest: Any) -> str | None:
    from headroom.cli.install import _deactivate_deployment_mutations, _stop_deployment

    try:
        _deactivate_deployment_mutations(manifest)
        _stop_deployment(manifest)
        return None
    except Exception as exc:
        return str(exc)


def _unwrap_claude_route_cleanup(port: int) -> dict[str, Any]:
    manifest = _find_persistent_manifest(port)
    env_keys = _same_port_claude_env_keys(port)
    if manifest is not None:
        if _can_unwrap_stop_persistent_manifest(manifest):
            error = _stop_persistent_manifest_for_claude_unwrap(manifest)
            if error is None:
                return {
                    "kind": "persistent_stopped",
                    "manifest": manifest,
                    "env_keys": env_keys,
                }
            return {
                "kind": "persistent_failed",
                "manifest": manifest,
                "env_keys": env_keys,
                "error": error,
            }
        return {
            "kind": "persistent_residue",
            "manifest": manifest,
            "env_keys": env_keys,
        }
    return {
        "kind": "local",
        "status": _stop_local_proxy_for_unwrap(port),
        "env_keys": env_keys,
    }


def _echo_claude_unwrap_route_cleanup(result: dict[str, Any], port: int) -> bool:
    kind = str(result.get("kind") or "")
    env_keys = [str(key) for key in result.get("env_keys", []) if isinstance(key, str)]
    clean = True
    if kind == "local":
        status = str(result.get("status") or "failed")
        _echo_unwrap_proxy_stop_status(status, port)
        clean = status in {"stopped", "not_running"}
    elif kind == "persistent_stopped":
        manifest = result["manifest"]
        click.echo(
            f"  Stopped Claude-owned persistent deployment '{manifest.profile}' on port {port}."
        )
    elif kind == "persistent_residue":
        manifest = result["manifest"]
        click.echo(
            "  Warning: same-port persistent deployment "
            f"'{manifest.profile}' still owns port {port}; left it running because it is not "
            "clearly Claude-targeted."
        )
        click.echo(f"  To stop it, run `headroom install stop --profile {manifest.profile}`.")
        click.echo(
            f"  To remove it completely, run `headroom install remove --profile {manifest.profile}`."
        )
        clean = False
    elif kind == "persistent_failed":
        manifest = result["manifest"]
        click.echo(
            "  Warning: failed to stop Claude-owned persistent deployment "
            f"'{manifest.profile}' on port {port}: {result.get('error')}"
        )
        click.echo(f"  Retry with `headroom install stop --profile {manifest.profile}`.")
        clean = False
    if env_keys:
        click.echo(
            "  Warning: current shell still exports "
            + ", ".join(env_keys)
            + f" for port {port}; restart Claude and your shell or unset those variables."
        )
        clean = False
    return clean


def _echo_unwrap_proxy_stop_status(status: str, port: int) -> None:
    """Print a human-readable proxy stop result for unwrap commands."""

    if status == "stopped":
        click.echo(f"  Stopped local Headroom proxy on port {port}.")
    elif status == "not_running":
        click.echo(f"  No local Headroom proxy detected on port {port}.")
    elif status == "unidentified":
        click.echo(
            f"  Warning: port {port} is in use, but it did not look like Headroom; left it running."
        )
    elif status == "no_pid":
        click.echo(
            f"  Warning: Headroom proxy on port {port} did not expose a PID; left it running."
        )
    else:
        click.echo(f"  Warning: failed to stop Headroom proxy on port {port}; stop it manually.")


def _find_persistent_manifest(port: int) -> Any:
    """Return a matching persistent deployment manifest for the requested port."""
    from headroom.install.state import list_manifests

    manifests = [manifest for manifest in list_manifests() if manifest.port == port]
    manifests.sort(key=lambda manifest: (manifest.profile != "default", manifest.profile))
    return manifests[0] if manifests else None


def _recover_persistent_proxy(port: int) -> bool:
    """Start or recover a matching persistent deployment for the requested port."""
    from headroom.install.health import probe_ready
    from headroom.install.models import InstallPreset, SupervisorKind
    from headroom.install.runtime import start_detached_agent, start_persistent_docker, wait_ready
    from headroom.install.supervisors import start_supervisor

    manifest = _find_persistent_manifest(port)
    if manifest is None:
        return False

    if probe_ready(manifest.health_url):
        click.echo(f"  Reusing persistent deployment '{manifest.profile}' on port {port}")
        return True

    if manifest.supervisor_kind == SupervisorKind.TASK.value:
        click.echo(
            f"  Warning: task-based deployment '{manifest.profile}' cannot be auto-recovered via wrap"
        )
        return False

    click.echo(f"  Recovering persistent deployment '{manifest.profile}' on port {port}...")
    try:
        if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
            start_persistent_docker(manifest)
        elif manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            start_supervisor(manifest)
        else:
            start_detached_agent(manifest.profile)
    except Exception as exc:
        click.echo(
            f"  Warning: could not recover persistent deployment '{manifest.profile}': {exc}"
        )
        return False

    if wait_ready(manifest, timeout_seconds=45):
        click.echo(f"  Recovered persistent deployment '{manifest.profile}' on port {port}")
        return True

    click.echo(f"  Warning: persistent deployment '{manifest.profile}' did not become ready")
    return False


def _restart_persistent_proxy(manifest: Any, port: int) -> bool:
    """Restart a persistent deployment after an idle stale-version detection."""
    from headroom.install.models import InstallPreset, SupervisorKind
    from headroom.install.runtime import (
        start_detached_agent,
        start_persistent_docker,
        stop_runtime,
        wait_ready,
    )
    from headroom.install.supervisors import start_supervisor

    click.echo(
        f"  Restarting persistent deployment '{manifest.profile}' "
        f"with Headroom {_HEADROOM_VERSION}..."
    )
    try:
        if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
            stop_runtime(manifest)
            start_persistent_docker(manifest)
        elif manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            # start_supervisor performs the platform-native restart operation:
            # systemd restart, launchctl kickstart -k, or sc.exe start.
            start_supervisor(manifest)
        else:
            stop_runtime(manifest)
            start_detached_agent(manifest.profile)
    except Exception as exc:
        click.echo(
            f"  Warning: could not restart persistent deployment '{manifest.profile}': {exc}"
        )
        return False

    if wait_ready(manifest, timeout_seconds=45):
        click.echo(f"  Restarted persistent deployment '{manifest.profile}' on port {port}")
        return True

    click.echo(f"  Warning: persistent deployment '{manifest.profile}' did not become ready")
    return False


def _copilot_model_configured(copilot_args: tuple[str, ...], env: dict[str, str]) -> bool:
    """Return True when Copilot BYOK model selection is configured."""
    return _copilot_model_configured_impl(copilot_args, env)


def _copilot_model_from_args(copilot_args: tuple[str, ...], env: dict[str, str]) -> str | None:
    """Resolve the Copilot model from command-line args or environment."""
    return _copilot_model_from_args_impl(copilot_args, env)


def _copilot_default_wire_api_for_model(model: str | None) -> str:
    """Return the default OpenAI-compatible wire API for a Copilot model."""
    return _copilot_default_wire_api_for_model_impl(model)


def _build_copilot_native_launch_env(
    *, port: int, environ: dict[str, str], project: str | None
) -> tuple[dict[str, str], list[str]]:
    from headroom.providers.copilot.wrap import build_native_launch_env

    return build_native_launch_env(port=port, environ=environ, project=project)


def _native_api_url_supported(*, environ: Mapping[str, str] | None = None) -> bool | None:
    from headroom.providers.copilot.wrap import native_api_url_supported

    return native_api_url_supported(environ=environ)


def _should_use_copilot_oauth(
    *,
    backend: str | None,
    provider_type: str,
    env: dict[str, str],
    force_subscription: bool = False,
) -> bool:
    """Prefer a reusable Copilot OAuth session when the requested routing supports it."""
    if force_subscription:
        return True
    if env.get("COPILOT_PROVIDER_API_KEY") or env.get("COPILOT_PROVIDER_BEARER_TOKEN"):
        return False
    if provider_type == "anthropic":
        return False

    effective_backend = backend or os.environ.get("HEADROOM_BACKEND")
    if effective_backend not in (None, "", "anthropic"):
        return False

    return has_oauth_auth()


def _push_runtime_env(port: int, no_proxy: bool) -> None:
    """Hot-sync this session's live env knobs to the proxy on ``port``.

    Live knobs (the output-shaper family, the ast-grep read threshold) are read
    from the *proxy's* process environment. A proxy we reused — rather than
    started — would otherwise ignore values exported in this shell, since its
    environment was snapshotted when it first launched. Pushing them to
    ``/admin/runtime-env`` applies them in memory with no disruptive restart.

    Best-effort: a silent no-op when nothing is explicitly set, when there is no
    proxy (``--no-proxy``), when the proxy is unreachable, or when it predates
    the endpoint (older build returns 404).
    """
    if no_proxy:
        return
    from headroom.proxy import runtime_env as _rt

    payload = _rt.explicit_env(os.environ)
    if not payload:
        return

    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/admin/runtime-env",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            response.read()
    except (OSError, urllib.error.URLError, ValueError):
        return
    click.echo(f"  Synced output settings to proxy: {', '.join(sorted(payload))}")


def _ensure_proxy_unlocked(
    port: int,
    no_proxy: bool,
    *,
    learn: bool = False,
    memory: bool = False,
    agent_type: str = "unknown",
    code_graph: bool = False,
    backend: str | None = None,
    anyllm_provider: str | None = None,
    region: str | None = None,
    openai_api_url: str | None = None,
    anthropic_api_url: str | None = None,
    vertex_api_url: str | None = None,
    clear_vertex_api_url: bool = False,
    copilot_api_token: str | None = None,
    copilot_refresh_oauth_token: str | None = None,
    copilot_api_token_expires_at: float | None = None,
) -> tuple[subprocess.Popen | None, int]:
    """Start or verify proxy. Returns (process_handle, actual_port).

    The public ``_ensure_proxy`` wrapper serializes callers per port before
    entering this function. Keeping the implementation separate makes the
    lock boundary explicit and ensures every health/configuration check runs
    under the same startup critical section.
    """
    helpers = _live_wrap_module()
    copilot_subscription_seed_requested = (
        bool(copilot_api_token)
        or bool(copilot_refresh_oauth_token)
        or copilot_api_token_expires_at is not None
    )
    # --no-proxy reuses an already-running proxy, so backend/region/provider
    # flags (which only apply when we start one) would be silently dropped.
    if no_proxy and (backend or anyllm_provider or region):
        click.echo(
            "  Warning: --backend/--region/--anyllm-provider have no effect with --no-proxy "
            "(reusing the existing proxy)."
        )
    if not no_proxy:
        manifest = helpers._find_persistent_manifest(port)
        isolated_copilot_subscription_proxy = copilot_subscription_seed_requested and (
            manifest is not None or helpers._check_proxy(port)
        )
        if isolated_copilot_subscription_proxy:
            click.echo(
                "  Copilot subscription seeds are session-specific; "
                "starting a dedicated local proxy instance for this wrap session."
            )
        if not isolated_copilot_subscription_proxy and manifest is not None:
            from headroom.install.health import probe_ready

            if probe_ready(manifest.health_url):
                health_payload = helpers._query_proxy_health(port)
                if helpers._proxy_needs_version_restart(health_payload):
                    running_version = helpers._proxy_version(health_payload) or "unknown"
                    active_sessions = helpers._proxy_active_session_count(health_payload)
                    other_wrappers = helpers._live_proxy_clients(port, exclude_self=True)
                    if active_sessions > 0 or other_wrappers:
                        detail = (
                            f"{active_sessions} active session(s)"
                            if active_sessions > 0
                            else f"{len(other_wrappers)} attached wrapper(s)"
                        )
                        click.echo(
                            f"  Proxy on port {port} is running Headroom {running_version}; "
                            f"current CLI is {_HEADROOM_VERSION}."
                        )
                        click.echo(
                            f"  Leaving it running because {detail} "
                            "are still attached; it will be restarted when idle."
                        )
                        return None, port
                    if helpers._restart_persistent_proxy(manifest, port):
                        return None, port
                    raise click.ClickException(
                        f"Persistent deployment '{manifest.profile}' on port {port} "
                        f"is running stale Headroom {running_version} and could not be restarted."
                    )
                # Check if the running proxy has the features we need.
                # Without this, a persistent deployment started for one use case
                # (e.g. --backend anthropic) would be silently reused for another
                # (e.g. --subscription --provider-type openai) causing auth failures.
                running_config = helpers._proxy_health_config(health_payload)
                if running_config is None:
                    running_config = helpers._query_proxy_config(port)
                if running_config is not None:
                    missing = []
                    if memory and not running_config.get("memory"):
                        missing.append("memory")
                    if learn and not running_config.get("learn"):
                        missing.append("learn")
                    if code_graph and not running_config.get("code_graph"):
                        missing.append("code_graph")
                    if copilot_subscription_seed_requested:
                        missing.append("copilot-subscription-auth")
                    if openai_api_url:
                        running_openai_url = _normalize_proxy_api_url(
                            running_config.get("openai_api_url")
                        )
                        requested_openai_url = _normalize_proxy_api_url(openai_api_url)
                        if running_openai_url != requested_openai_url:
                            missing.append("openai-api-url")
                    if not missing:
                        click.echo(f"  Proxy already running on port {port}")
                        click.echo(f"  Dashboard:    http://127.0.0.1:{port}/dashboard")
                        return None, port
                # Features mismatch or config unavailable — fall through to
                # the non-persistent path which handles proxy restart.
            else:
                if helpers._recover_persistent_proxy(port):
                    # If the caller requested feature-sensitive config (e.g.
                    # openai_api_url for Copilot subscription), continue into
                    # the shared running-proxy checks below so mismatch-driven
                    # restart logic can run. For plain recover-only calls,
                    # preserve the historical fast return.
                    if not any(
                        (
                            memory,
                            learn,
                            code_graph,
                            openai_api_url,
                            copilot_subscription_seed_requested,
                        )
                    ):
                        return None, port
                    if not helpers._check_proxy(port):
                        return None, port

                    # A freshly recovered persistent proxy may not expose
                    # a full config payload yet. In feature-sensitive flows
                    # (e.g. Copilot subscription), treat missing or mismatched
                    # config as restart-required and refresh the persistent
                    # deployment directly instead of silently reusing it.
                    health_payload = helpers._query_proxy_health(port)
                    running_config = helpers._proxy_health_config(health_payload)
                    if running_config is None:
                        running_config = helpers._query_proxy_config(port)

                    if running_config is None:
                        click.echo(
                            f"  Recovered persistent deployment '{manifest.profile}' "
                            "did not expose config; restarting with requested features..."
                        )
                        if helpers._restart_persistent_proxy(manifest, port):
                            return None, port
                        raise click.ClickException(
                            f"Persistent deployment '{manifest.profile}' on port {port} "
                            "could not be restarted after recovery."
                        )

                    missing = []
                    if memory and not running_config.get("memory"):
                        missing.append("memory")
                    if learn and not running_config.get("learn"):
                        missing.append("learn")
                    if code_graph and not running_config.get("code_graph"):
                        missing.append("code-graph")
                    if copilot_subscription_seed_requested:
                        missing.append("copilot-subscription-auth")
                    if openai_api_url:
                        running_openai_url = _normalize_proxy_api_url(
                            running_config.get("openai_api_url")
                        )
                        requested_openai_url = _normalize_proxy_api_url(openai_api_url)
                        if running_openai_url != requested_openai_url:
                            missing.append("openai-api-url")

                    if missing:
                        flags_str = ", ".join(f"--{f}" for f in missing)
                        click.echo(
                            f"  Recovered persistent deployment '{manifest.profile}' is missing: "
                            f"{flags_str}; restarting..."
                        )
                        if helpers._restart_persistent_proxy(manifest, port):
                            return None, port
                        raise click.ClickException(
                            f"Persistent deployment '{manifest.profile}' on port {port} "
                            "could not be restarted with requested features."
                        )
                    return None, port
                elif helpers._check_proxy(port):
                    raise click.ClickException(
                        f"Persistent deployment '{manifest.profile}' on port {port} is not healthy."
                    )
            click.echo(
                f"  Warning: persistent deployment '{manifest.profile}' on port {port} "
                "is stale; starting a fresh proxy instead."
            )

        if not isolated_copilot_subscription_proxy and helpers._check_proxy(port):
            # Proxy is running — check if it has the features we need
            needs_restart = False
            health_payload = helpers._query_proxy_health(port)
            running_config = helpers._proxy_health_config(health_payload)
            if running_config is None:
                running_config = helpers._query_proxy_config(port)

            if helpers._proxy_needs_version_restart(health_payload):
                running_version = helpers._proxy_version(health_payload) or "unknown"
                active_sessions = helpers._proxy_active_session_count(health_payload)
                other_wrappers = helpers._live_proxy_clients(port, exclude_self=True)
                if active_sessions > 0 or other_wrappers:
                    # active_sessions only counts Codex WebSocket relay; the
                    # marker list also covers HTTP wrap clients. Either means a
                    # live session is attached, so don't restart the shared
                    # proxy out from under it — defer until idle.
                    detail = (
                        f"{active_sessions} active session(s)"
                        if active_sessions > 0
                        else f"{len(other_wrappers)} attached wrapper(s)"
                    )
                    click.echo(
                        f"  Proxy on port {port} is running Headroom {running_version}; "
                        f"current CLI is {_HEADROOM_VERSION}."
                    )
                    click.echo(
                        f"  Leaving it running because {detail} "
                        "are still attached; it will be restarted when idle."
                    )
                    return None, port

                click.echo(
                    f"  Proxy on port {port} is running Headroom {running_version}; "
                    f"restarting with {_HEADROOM_VERSION}..."
                )
                proxy_pid = running_config.get("pid") if running_config is not None else None
                if proxy_pid is None:
                    raise click.ClickException(
                        f"Proxy on port {port} is stale but did not expose a PID. "
                        "Stop it manually and retry."
                    )
                if not helpers._kill_proxy_by_pid(int(proxy_pid), port):
                    raise click.ClickException(
                        f"Failed to stop stale proxy (PID {proxy_pid}) on port {port}. "
                        "Stop it manually and retry."
                    )
                needs_restart = True

            if running_config is not None:
                missing = []
                if memory and not running_config.get("memory"):
                    missing.append("memory")
                if learn and not running_config.get("learn"):
                    missing.append("learn")
                if code_graph and not running_config.get("code_graph"):
                    missing.append("code_graph")
                if copilot_subscription_seed_requested:
                    missing.append("copilot-subscription-auth")
                expected_savings_profile = helpers._wrap_agent_savings_profile(agent_type)
                if (
                    expected_savings_profile is not None
                    and running_config.get("savings_profile") != expected_savings_profile
                ):
                    missing.append("savings-profile")
                if openai_api_url:
                    running_openai_url = _normalize_proxy_api_url(
                        running_config.get("openai_api_url")
                    )
                    requested_openai_url = _normalize_proxy_api_url(openai_api_url)
                    if running_openai_url != requested_openai_url:
                        missing.append("openai-api-url")
                if vertex_api_url or clear_vertex_api_url:
                    running_vertex_url = _normalize_proxy_api_url(
                        running_config.get("vertex_api_url")
                    )
                    requested_vertex_url = _normalize_proxy_api_url(vertex_api_url)
                    if running_vertex_url != requested_vertex_url:
                        missing.append("vertex-api-url")

                if missing:
                    flags_str = ", ".join(
                        f if f.startswith("--") else f"--{f.replace('_', '-')}" for f in missing
                    )
                    other_wrappers = helpers._live_proxy_clients(port, exclude_self=True)
                    if other_wrappers:
                        # Another wrapper is attached to this proxy; restarting it
                        # to add flags would drop their in-flight requests. Reuse
                        # the running proxy as-is rather than disrupt them.
                        click.echo(
                            f"  Proxy on port {port} is missing: {flags_str}, but "
                            f"{len(other_wrappers)} other wrapper(s) are attached."
                        )
                        click.echo(
                            "  Leaving it running to avoid disrupting them; this "
                            "session will use the existing proxy as-is."
                        )
                    else:
                        needs_restart = True
                        click.echo(f"  Proxy on port {port} is missing: {flags_str}")
                        click.echo("  Restarting proxy with upgraded configuration...")

                        # Merge: keep features the running proxy already has
                        memory = memory or bool(running_config.get("memory"))
                        learn = learn or bool(running_config.get("learn"))
                        code_graph = code_graph or bool(running_config.get("code_graph"))

                        proxy_pid = running_config.get("pid")
                        if proxy_pid is not None:
                            if not helpers._kill_proxy_by_pid(int(proxy_pid), port):
                                raise click.ClickException(
                                    f"Failed to stop existing proxy (PID {proxy_pid}) on port {port}. "
                                    "Stop it manually and retry."
                                )
                        else:
                            click.echo(
                                "  Warning: Running proxy does not expose PID. "
                                "Cannot restart automatically."
                            )
                            click.echo(
                                f"  Please stop the proxy on port {port} manually "
                                f"and rerun with {flags_str}."
                            )
                            return None, port

            if not needs_restart:
                click.echo(f"  Proxy already running on port {port}")
                click.echo(f"  Dashboard:    http://127.0.0.1:{port}/dashboard")
                return None, port

        # Start (or restart) the proxy with the requested flags.
        # Subscription-seeded sessions must not claim the shared port even if
        # it is momentarily free; the persistent install or shared proxy still
        # owns that slot semantically.
        port_search_start = port + 1 if isolated_copilot_subscription_proxy else port
        try:
            actual_port = helpers._find_available_port(port_search_start)
        except OSError as e:
            raise click.ClickException(f"Port {port} is unavailable: {e}") from e
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e

        if actual_port != port:
            if isolated_copilot_subscription_proxy:
                click.echo(
                    f"  Port {port} is reserved for the shared proxy; "
                    f"using port {actual_port} for this dedicated session instead."
                )
            else:
                click.echo(f"  Port {port} is in use, using port {actual_port} instead.")

        click.echo(f"  Starting Headroom proxy on port {actual_port}...")
        try:
            proc = cast(
                subprocess.Popen[Any],
                _live_wrap_module()._start_proxy(
                    actual_port,
                    learn=learn,
                    memory=memory,
                    agent_type=agent_type,
                    code_graph=code_graph,
                    backend=backend,
                    anyllm_provider=anyllm_provider,
                    region=region,
                    openai_api_url=openai_api_url,
                    anthropic_api_url=anthropic_api_url,
                    vertex_api_url=vertex_api_url,
                    clear_vertex_api_url=clear_vertex_api_url,
                    copilot_api_token=copilot_api_token,
                    copilot_refresh_oauth_token=copilot_refresh_oauth_token,
                    copilot_api_token_expires_at=copilot_api_token_expires_at,
                ),
            )
            click.echo(f"  Proxy ready on http://127.0.0.1:{actual_port}")
            click.echo(f"  Dashboard:    http://127.0.0.1:{actual_port}/dashboard")
            return proc, actual_port
        except RuntimeError as e:
            click.echo(f"  Error: {e}")
            raise SystemExit(1) from e
    else:
        if not helpers._check_proxy(port):
            click.echo(f"  Warning: No proxy detected on port {port}")
        elif vertex_api_url or clear_vertex_api_url:
            health_payload = helpers._query_proxy_health(port)
            running_config = helpers._proxy_health_config(health_payload)
            if running_config is None:
                running_config = helpers._query_proxy_config(port)
            running_vertex_url = (
                _normalize_proxy_api_url(running_config.get("vertex_api_url"))
                if running_config is not None
                else None
            )
            requested_vertex_url = _normalize_proxy_api_url(vertex_api_url)
            if running_vertex_url != requested_vertex_url:
                click.echo(
                    "  Warning: --no-proxy is set, but the running proxy does not "
                    "advertise the requested Vertex target. Requests may still go "
                    "to the proxy's existing Vertex upstream."
                )
        return None, port


@contextmanager
def _proxy_start_lock(port: int) -> Any:
    """Serialize wrap proxy startup across processes sharing a port.

    A proxy can spend tens of seconds loading optional ML components before it
    binds its socket. Without this lock, two concurrent ``headroom wrap``
    commands both see an unavailable health endpoint, choose the same port,
    and race to spawn a listener. The lock is deliberately held through the
    health/configuration checks and startup, then released once the proxy is
    ready (or startup fails). Lock files are retained so an interrupted
    process cannot create an inode-replacement race for another waiter.
    """
    from headroom import paths as _paths

    lock_path = _paths.proxy_start_lock_path(port)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "a+b")  # noqa: SIM115
    except OSError:
        # Locking is a race-prevention enhancement, not a reason to make wrap
        # unusable when a read-only/custom workspace cannot hold state. The
        # existing port bind remains the final safety check in that degraded
        # environment.
        yield
        return
    with _locked_file(lock_file):
        yield


@wraps(_ensure_proxy_unlocked)
def _ensure_proxy(
    port: int,
    no_proxy: bool,
    **kwargs: Any,
) -> tuple[subprocess.Popen | None, int]:
    """Start or reuse a proxy without racing another wrap on the same port."""
    if no_proxy:
        return _ensure_proxy_unlocked(port, no_proxy, **kwargs)
    with _proxy_start_lock(port):
        # Re-checking is part of the lock boundary: a concurrent wrapper may
        # have finished startup while this caller was waiting for the lock.
        return _ensure_proxy_unlocked(port, no_proxy, **kwargs)


def _client_marker_path(port: int) -> Path:
    """Path to this process's wrap-client marker for ``port``."""
    from headroom import paths as _paths

    d = _paths.proxy_clients_dir(port)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{os.getpid()}.json"


def _proc_identity(pid: int) -> tuple[str, float] | None:
    """Best-effort ``(source, start_time)`` identity for a PID.

    Used to defeat PID reuse: a marker is only trusted while the live PID is
    *the same process* that wrote it. Returns ``None`` when start time can't be
    determined (e.g. macOS without psutil), in which case callers fall back to
    existence-only liveness — no regression, just no reuse protection there.

    The ``source`` tag ("psutil" vs "proc") guards against comparing values in
    different units; we only compare like-for-like.
    """
    try:
        import psutil  # type: ignore[import-untyped]  # optional dependency; portable when present

        return ("psutil", psutil.Process(pid).create_time())
    except Exception:
        pass
    # Linux fallback: field 22 of /proc/<pid>/stat is starttime in clock ticks
    # since boot — a stable per-process value. `comm` (field 2) may contain
    # spaces/parens, so split after the final ')'.
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            fields = fh.read().rpartition(b")")[2].split()
        return ("proc", float(fields[19]))
    except (OSError, IndexError, ValueError):
        return None


def _register_proxy_client(port: int) -> None:
    """Register this wrap process as a live client of the shared proxy.

    Best-effort: a failed write just means our marker is missing, and the
    liveness pruning in :func:`_live_proxy_clients` is the real safety net.
    """
    try:
        payload: dict[str, Any] = {"pid": os.getpid(), "started_at": time.time()}
        ident = _proc_identity(os.getpid())
        if ident is not None:
            payload["start_src"], payload["start_time"] = ident
        _write_text(_client_marker_path(port), json.dumps(payload))
    except OSError:
        pass


def _unregister_proxy_client(port: int) -> None:
    """Remove this process's client marker (idempotent)."""
    try:
        _client_marker_path(port).unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` names a live process.

    Thin wrapper over the shared Windows-safe helper so the marker-cleanup path
    and the install/runtime status path use one liveness probe (see #1544).
    """
    return pid_alive(pid)


def _identity_mismatch(src: Any, recorded: Any, pid: int) -> bool:
    """True only if ``pid``'s current identity *provably* differs from the
    recorded ``(src, recorded)`` identity (i.e. the PID was recycled).

    Conservative by design: any uncertainty (unknown/legacy identity, unknown
    start time, mismatched source) returns ``False`` — never claim a mismatch
    without proof, since the caller uses this to decide whether to trust or
    discard state tied to a live PID.
    """
    if not isinstance(src, str) or not isinstance(recorded, int | float):
        return False  # legacy / identity-less record — can't tell
    ident = _proc_identity(pid)
    if ident is None or ident[0] != src:
        return False  # can't compare like-for-like — don't claim mismatch
    # Start times are stable per process; >1s apart means a different process.
    return abs(ident[1] - float(recorded)) > 1.0


def _marker_pid_reused(marker: Path, pid: int) -> bool:
    """True only if the live ``pid`` is *provably* a different process than the
    one that wrote ``marker`` (i.e. the PID was recycled after a crash).
    """
    try:
        rec = json.loads(_read_text(marker))
    except (OSError, ValueError):
        return False
    return _identity_mismatch(rec.get("start_src"), rec.get("start_time"), pid)


def _live_proxy_clients(port: int, *, exclude_self: bool = True) -> list[int]:
    """Live wrap-client PIDs for ``port``, pruning stale markers as we go."""
    from headroom import paths as _paths

    d = _paths.proxy_clients_dir(port)
    if not d.exists():
        return []
    me = os.getpid()
    live: list[int] = []
    for marker in d.glob("*.json"):
        try:
            pid = int(marker.stem)
        except ValueError:
            continue
        # Stale if the PID is gone, or recycled by an unrelated process.
        if not _pid_alive(pid) or _marker_pid_reused(marker, pid):
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        if not (exclude_self and pid == me):
            live.append(pid)
    return live


def _make_cleanup(proxy_proc_holder: list, port: int | list[int] = 8787) -> Any:
    """Create a cleanup function that terminates the proxy on exit.

    Only kills the proxy when no other live headroom-wrapped clients remain,
    tracked via per-PID marker files in ``paths.proxy_clients_dir(port)``.

    ``port`` can be an ``int`` or a ``list[int]``.  When a port fallback occurs
    (``_ensure_proxy`` ups the port because the requested one is busy), the
    caller can update ``port[0]`` in-place and the closure picks it up.
    """

    def _other_clients_exist() -> bool:
        p = port[0] if isinstance(port, list) else port
        return len(_live_proxy_clients(p, exclude_self=True)) > 0

    def cleanup(signum: int | None = None, frame: Any = None) -> None:
        p = port[0] if isinstance(port, list) else port
        _unregister_proxy_client(p)
        proc = proxy_proc_holder[0] if proxy_proc_holder else None
        if proc:
            if _other_clients_exist():
                # Other clients still using the proxy — leave it running.
                return
            # Snapshot the serving PID before terminating the launcher.  On
            # Windows the detached serving child can briefly make /health
            # unavailable while the launcher exits, causing the later safety
            # probe to classify our own listener as "unidentified" and leave
            # it orphaned.  We still verify it through Headroom's health
            # payload before trusting the PID.
            serving_pid: int | None = None
            if sys.platform == "win32" and _check_proxy(p):
                running_config = _query_proxy_config(p)
                try:
                    serving_pid = int(running_config["pid"]) if running_config else None
                except (KeyError, TypeError, ValueError):
                    serving_pid = None
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            # On Windows the proxy launcher can exit while its detached
            # serving child remains alive (the native runtime uses a child
            # process).  The detachment is intentional so an ungraceful
            # terminal close cannot disrupt other wrappers, but a graceful
            # Ctrl+C from the last wrapper must still stop the listener.
            if sys.platform == "win32" and _check_proxy(p):
                stop_status = _stop_local_proxy_for_unwrap(p)
                if stop_status == "unidentified" and serving_pid is not None:
                    stop_status = "stopped" if _kill_proxy_by_pid(serving_pid, p) else "failed"
                if stop_status not in {"stopped", "not_running"}:
                    click.echo(
                        f"  Warning: proxy on port {p} remained running "
                        f"after shutdown ({stop_status})."
                    )

    return cleanup


def _ignore_child_sigint(signum: int | None = None, frame: Any = None) -> None:
    """Keep the wrapper alive when Ctrl-C is intended for the child CLI."""

    return None


def _exit_on_signal(signum: int | None = None, frame: Any = None) -> None:
    """Unwind on SIGTERM/SIGHUP so the ``finally`` block actually runs.

    Registering ``cleanup`` itself as the handler did not achieve what its call
    site documented. A Python signal handler that returns normally does not
    unwind the stack -- under PEP 475 the interrupted ``waitpid`` is simply
    retried -- so the ``finally`` that restores ``settings.local.json`` never
    ran, while the handler had already terminated the proxy underneath a child
    that was still alive. Raising SystemExit reverses that: the settings are
    restored and cleanup runs exactly once, from ``finally`` (#3205).
    """
    raise SystemExit(128 + int(signum or 0))


def _launch_tool(
    binary: str,
    args: tuple,
    env: dict[str, str],
    port: int,
    no_proxy: bool,
    tool_label: str,
    env_vars_display: list[str],
    *,
    learn: bool = False,
    memory: bool = False,
    agent_type: str = "unknown",
    code_graph: bool = False,
    backend: str | None = None,
    anyllm_provider: str | None = None,
    region: str | None = None,
    openai_api_url: str | None = None,
    anthropic_api_url: str | None = None,
    copilot_api_token: str | None = None,
    copilot_refresh_oauth_token: str | None = None,
    copilot_api_token_expires_at: float | None = None,
    configure_launch: Callable[
        [int, tuple, dict[str, str], list[str]],
        tuple[tuple, dict[str, str], list[str]],
    ]
    | None = None,
) -> None:
    """Common logic: start proxy, launch tool, clean up."""
    proxy_holder: list[subprocess.Popen | None] = [None]
    port_holder: list[int] = [port]
    cleanup = _make_cleanup(proxy_holder, port_holder)
    signal.signal(signal.SIGINT, _ignore_child_sigint)
    signal.signal(signal.SIGTERM, _exit_on_signal)

    try:
        click.echo()
        padded = f"HEADROOM WRAP: {tool_label}".center(47)
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo(f"  ║{padded}║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        _register_proxy_client(port)
        proxy_holder[0], actual_port = _ensure_proxy(
            port,
            no_proxy,
            learn=learn,
            memory=memory,
            agent_type=agent_type,
            code_graph=code_graph,
            backend=backend,
            anyllm_provider=anyllm_provider,
            region=region,
            openai_api_url=openai_api_url,
            anthropic_api_url=anthropic_api_url,
            copilot_api_token=copilot_api_token,
            copilot_refresh_oauth_token=copilot_refresh_oauth_token,
            copilot_api_token_expires_at=copilot_api_token_expires_at,
        )
        if actual_port != port:
            _unregister_proxy_client(port)
            _register_proxy_client(actual_port)
        port_holder[0] = actual_port
        _push_runtime_env(actual_port, no_proxy)

        # If port fell back, update environment URLs to point at the actual port.
        if actual_port != port:
            for k, v in dict(env).items():
                env[k] = v.replace(f"127.0.0.1:{port}", f"127.0.0.1:{actual_port}")

        if configure_launch is not None:
            args, env, env_vars_display = configure_launch(actual_port, args, env, env_vars_display)

        # Reduce-at-source: fill in SAFE quiet-CLI env defaults for the launched
        # agent (git/npm/pip/pytest emit less noise), unless the user opted out.
        # Applies to every wrapped tool since they all launch through here.
        _quiet_written = _configure_quiet_cli_env(env)

        click.echo()
        click.echo(f"  Launching {tool_label} (API routed through Headroom)...")
        for var in env_vars_display:
            click.echo(f"  {var}")
        if _quiet_written:
            click.echo(
                f"  Quiet CLI defaults: {', '.join(_quiet_written)} (opt out: {_QUIET_CLI_ENV}=0)"
            )
        if args:
            click.echo(f"  Extra args: {' '.join(args)}")
        _print_telemetry_notice()
        click.echo()

        result = subprocess.run([binary, *args], env=env)
        raise SystemExit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        cleanup()


def _run_checked(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    action: str,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess and raise a ClickException with actionable context on failure."""
    try:
        return run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise click.ClickException(f"{action} failed: command not found: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        details = stderr or stdout or f"exit code {e.returncode}"
        raise click.ClickException(f"{action} failed: {details}") from e


def _resolve_openclaw_extensions_dir(openclaw_bin: str) -> Path:
    """Resolve OpenClaw extension root from active config file path."""
    result = _run_checked([openclaw_bin, "config", "file"], action="openclaw config file")
    lines = result.stdout.strip().splitlines()
    config_path_str = lines[-1].strip() if lines else ""
    if not config_path_str:
        raise click.ClickException(
            "Unable to resolve OpenClaw config path from `openclaw config file`."
        )
    config_path = Path(config_path_str).expanduser()
    return config_path.parent / "extensions"


def _normalize_openclaw_gateway_provider_ids(provider_ids: tuple[str, ...] | None) -> list[str]:
    """Normalize configured OpenClaw provider ids, defaulting to openai-codex."""
    return _normalize_openclaw_gateway_provider_ids_impl(provider_ids)


def _read_openclaw_config_value(openclaw_bin: str, path: str) -> Any | None:
    """Read an OpenClaw config value when present, returning None on missing paths."""
    result = run(
        [openclaw_bin, "config", "get", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output:
        return None

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def _decode_openclaw_entry_json(raw_value: str | None) -> Any | None:
    """Decode a JSON payload captured from `openclaw config get` when available."""
    return _decode_openclaw_entry_json_impl(raw_value)


def _build_openclaw_plugin_entry(
    *,
    existing_entry: Any,
    proxy_port: int,
    startup_timeout_ms: int,
    python_path: str | None,
    no_auto_start: bool,
    gateway_provider_ids: tuple[str, ...] | None,
    enabled: bool,
) -> dict[str, object]:
    """Merge managed Headroom plugin settings with any existing entry payload."""
    return _build_openclaw_plugin_entry_impl(
        existing_entry=existing_entry,
        proxy_port=proxy_port,
        startup_timeout_ms=startup_timeout_ms,
        python_path=python_path,
        no_auto_start=no_auto_start,
        gateway_provider_ids=gateway_provider_ids,
        enabled=enabled,
    )


def _build_openclaw_unwrap_entry(existing_entry: Any) -> dict[str, object]:
    """Disable the managed plugin while preserving unrelated user config."""
    return _build_openclaw_unwrap_entry_impl(existing_entry)


def _write_openclaw_plugin_entry(openclaw_bin: str, entry: dict[str, object]) -> None:
    """Persist the Headroom plugin config entry."""
    _run_checked(
        [
            openclaw_bin,
            "config",
            "set",
            "plugins.entries.headroom",
            json.dumps(entry, separators=(",", ":")),
            "--strict-json",
        ],
        action="openclaw config set plugins.entries.headroom",
    )


def _set_openclaw_context_engine_slot(openclaw_bin: str, engine_id: str) -> None:
    """Persist the selected OpenClaw context engine slot."""
    _run_checked(
        [
            openclaw_bin,
            "config",
            "set",
            "plugins.slots.contextEngine",
            json.dumps(engine_id),
            "--strict-json",
        ],
        action="openclaw config set plugins.slots.contextEngine",
    )


def _restart_or_start_openclaw_gateway(openclaw_bin: str) -> tuple[str, str]:
    """Restart the gateway when running, otherwise start it."""
    restart_result = run(
        [openclaw_bin, "gateway", "restart"],
        capture_output=True,
        text=True,
    )
    if restart_result.returncode == 0:
        output = restart_result.stdout.strip() or restart_result.stderr.strip()
        return "restarted", output

    start_result = _run_checked(
        [openclaw_bin, "gateway", "start"],
        action="openclaw gateway start",
    )
    output = start_result.stdout.strip() or start_result.stderr.strip()
    return "started", output


def _copy_openclaw_plugin_into_extensions(
    *,
    plugin_dir: Path,
    openclaw_bin: str,
) -> Path:
    """Fallback install path when `openclaw plugins install` is blocked on linked source."""
    dist_dir = plugin_dir / "dist"
    if not dist_dir.exists():
        raise click.ClickException(
            f"Plugin dist folder missing at {dist_dir}. Build the plugin first."
        )
    hook_shim_dir = plugin_dir / "hook-shim"
    if not hook_shim_dir.exists():
        raise click.ClickException(
            f"Plugin hook-shim folder missing at {hook_shim_dir}. Build the plugin first."
        )

    extensions_dir = _resolve_openclaw_extensions_dir(openclaw_bin)
    target_dir = extensions_dir / "headroom"
    target_dist = target_dir / "dist"
    target_hook_shim = target_dir / "hook-shim"
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_dist.exists():
        shutil.rmtree(target_dist)
    if target_hook_shim.exists():
        shutil.rmtree(target_hook_shim)
    shutil.copytree(dist_dir, target_dist)
    shutil.copytree(hook_shim_dir, target_hook_shim)

    for filename in ("openclaw.plugin.json", "package.json", "README.md"):
        source = plugin_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)

    return target_dir


@main.group()
@click.pass_context
def wrap(ctx: click.Context) -> None:
    """Wrap CLI tools to run through Headroom.

    \b
    Starts a Headroom proxy, configures the environment, and launches
    the target tool so all API calls route through Headroom automatically.

    \b
    Supported tools (one Click subcommand per tool):
        headroom wrap claude              # Claude Code (Anthropic)
        headroom wrap codex               # OpenAI Codex CLI
        headroom wrap copilot -- --model claude-sonnet-4-20250514
        headroom wrap vscode             # VS Code Copilot (preserves model picker)
        headroom wrap vscode-claude      # VS Code Claude Code extension
        headroom wrap aider               # Aider
        headroom wrap openclaude          # OpenClaude
        headroom wrap vibe                # Mistral Vibe
        headroom wrap grok                # Grok CLI (xAI)
        headroom wrap cursor              # Cursor (prints config instructions)
        headroom wrap grok-build          # Grok Build (updates ~/.grok/config.toml)
        headroom wrap cline               # Cline (VS Code; prints config instructions)
        headroom wrap continue            # Continue (VS Code/JetBrains; injects systemMessage)
        headroom wrap goose               # Goose (Block) CLI
        headroom wrap openhands           # OpenHands CLI
        headroom wrap openclaw            # OpenClaw plugin bootstrap
        headroom wrap opencode            # OpenCode CLI
        headroom wrap omp                 # Oh My Pi CLI
        headroom wrap zcode               # ZCode desktop app setup

    \b
    `wrap` vs `proxy`:
        - `headroom wrap <tool>` — convenience: starts the proxy for you,
          sets the right env vars, and launches the wrapped CLI.
        - `headroom proxy` — just the proxy. Use this with any
          OpenAI/Anthropic-compatible client by setting
          ANTHROPIC_BASE_URL / OPENAI_BASE_URL yourself.

    \b
    `openclaw` is a separate tool — different from opencode.
    """
    if _should_purge_context_tools(ctx):
        _report_context_tool_purge()


@main.group()
@click.pass_context
def unwrap(ctx: click.Context) -> None:
    """Undo durable Headroom wrapping for supported tools."""
    if _should_purge_context_tools(ctx):
        _report_context_tool_purge()


@wrap.command("selfheal", hidden=True)
@click.option("--marker", default=None, hidden=True)
def wrap_selfheal(marker: str | None) -> None:
    """Session-start self-heal for a wrap base_url left by a dead proxy (#2221).

    Installed as a SessionStart-only hook by ``wrap claude`` so a session that
    only ran ``wrap`` (never ``init``) still recovers a stale ``ANTHROPIC_BASE_URL``
    when its proxy died without cleanup. Best-effort and never raises.
    """
    del marker
    _selfheal_dead_wrap_base_url()


# =============================================================================
# Claude Code
# =============================================================================

# Hostnames that mean "this machine" — used to avoid feeding the proxy its own
# URL back as the upstream when the user already had ANTHROPIC_BASE_URL pointed
# at a previous Headroom instance.
_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _detect_inbound_anthropic_upstream(port: int) -> str | None:
    """Return a pre-set ANTHROPIC_BASE_URL that is NOT this proxy, else None.

    Issue #1353: users who already route Claude Code at a LiteLLM (or any
    custom Anthropic-compatible) gateway via ``ANTHROPIC_BASE_URL`` lose that
    routing when they run ``headroom wrap claude`` — wrap overwrites
    ANTHROPIC_BASE_URL with the local proxy URL and silently forwards to
    api.anthropic.com instead of the LiteLLM URL the user configured. Treat
    the pre-existing value as the proxy's upstream so compression layers on
    top of the user's gateway instead of replacing it. URLs pointing at this
    proxy instance are ignored to avoid a self-referential forwarding loop.
    """

    base_url = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
    if not base_url:
        return None
    try:
        parsed = urllib.parse.urlsplit(base_url)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname in _LOCAL_HOSTNAMES:
        try:
            parsed_port = parsed.port
        except ValueError:
            return None
        if parsed_port == port:
            return None
    return base_url


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@_serena_instructions_option
@click.option(
    # no "-p" short alias here: claude's own -p/--print must fall through to CLAUDE_ARGS
    "--port",
    default=8787,
    type=click.IntRange(1, 65535),
    help="Proxy port (default: 8787)",
)
@click.option(
    "--no-mcp",
    is_flag=True,
    help="Skip headroom MCP server registration (compression markers will be unactionable)",
)
@_code_memory_option
@click.option(
    "--no-tokensave",
    is_flag=True,
    hidden=True,
    help="Deprecated and ignored: tokensave was retired; Serena is the default code memory.",
)
@click.option(
    "--serena",
    is_flag=True,
    hidden=True,
    help="Deprecated: use --code-memory serena. Force the Serena MCP compressor on.",
)
@click.option(
    "--no-serena",
    is_flag=True,
    hidden=True,
    help="Deprecated: use --code-memory none. Register no code-memory MCP.",
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable the proxy's live code-graph file watcher for the current project.",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to MEMORY.md)"
)
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--tool-search",
    "tool_search",
    default=None,
    metavar="MODE",
    help=(
        "Keep Claude Code's on-demand tool loading (deferral) active through the "
        "proxy. MODE is true (default), auto, auto:N, or false. Without it, a "
        "custom ANTHROPIC_BASE_URL makes Claude Code load every tool schema "
        "eagerly, inflating local context (issue #746). A pre-set "
        "ENABLE_TOOL_SEARCH env var is respected."
    ),
)
@click.option(
    "--backend",
    default=None,
    help="API backend for the proxy: 'anthropic' (default), 'litellm-vertex_ai', etc. "
    "(env: HEADROOM_BACKEND). For Vertex, prefer CLAUDE_CODE_USE_VERTEX=1 (native, "
    "keeps your GCP auth) over a litellm backend.",
)
@click.option(
    "--region",
    default=None,
    help="Cloud region for Vertex/Bedrock backends (env: HEADROOM_REGION).",
)
@click.option(
    "--1m",
    "context_1m",
    is_flag=True,
    help=(
        "Preserve the 1M context window. Behind a custom ANTHROPIC_BASE_URL "
        "Claude Code drops the context-1m beta header and caps at 200k; this "
        "sets ANTHROPIC_MODEL=<opus>[1m] on the launched process so the 1M "
        "window activates through the proxy (issue #1158)."
    ),
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def claude(
    port: int,
    no_mcp: bool,
    no_tokensave: bool,
    serena: bool,
    no_serena: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    tool_search: str | None,
    backend: str | None,
    region: str | None,
    context_1m: bool,
    verbose: bool,
    prepare_only: bool,
    claude_args: tuple,
) -> None:
    """Launch Claude Code through Headroom proxy.

    \b
    Sets ANTHROPIC_BASE_URL to route all Anthropic API calls through Headroom.
    All unknown flags are passed through to claude (e.g. --resume, --model).

    \b
    Examples:
        headroom wrap claude                    # Start everything (Serena code memory)
        headroom wrap claude --memory           # With persistent memory
        headroom wrap claude --resume <id>      # Resume a session
        headroom wrap claude -- -p              # Claude in print mode
        headroom wrap claude --no-mcp           # Skip MCP retrieve tool registration
        headroom wrap claude --code-memory none # No code-memory MCP
        headroom wrap claude --1m               # Preserve the 1M context window
    """
    if prepare_only:
        return

    claude_bin = shutil.which("claude")
    if not claude_bin:
        click.echo("Error: 'claude' not found in PATH.")
        click.echo("Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
        raise SystemExit(1)

    # Validate --tool-search up front so a typo fails before we start the proxy.
    if tool_search is not None:
        tool_search = _normalize_tool_search_mode(tool_search)

    proxy_holder: list[subprocess.Popen | None] = [None]
    _saved_base_url: list[str | None] = [None]  # previous settings.json value for restore
    _tool_search_not_written = object()
    _saved_tool_search: list[object | str | None] = [_tool_search_not_written]
    _settings_foundry: list[bool] = [False]
    port_holder: list[int] = [port]
    _settings_vertex: list[bool] = [False]
    # Bind before the try so the finally can always reference it. It is otherwise
    # only assigned inside the try (after _ensure_proxy, which can raise), so an
    # early proxy-start failure would make the finally raise UnboundLocalError,
    # masking the real error and skipping cleanup(). Mirrors the holders above.
    _wrap_settings_path = Path.cwd() / ".claude" / "settings.local.json"
    _raise_on_claude_auth_conflict(
        user_settings_path=claude_user_settings_path(),
        project_settings_path=Path.cwd() / ".claude" / "settings.json",
        project_local_settings_path=_wrap_settings_path,
        environ=dict(os.environ),
    )
    cleanup = _make_cleanup(proxy_holder, port_holder)
    signal.signal(signal.SIGINT, _ignore_child_sigint)
    signal.signal(signal.SIGTERM, _exit_on_signal)
    if hasattr(signal, "SIGHUP"):
        # Terminal close / tmux kill-session sends SIGHUP, not SIGTERM — without
        # this, the finally block's base_url restore never runs (issue #1768).
        signal.signal(signal.SIGHUP, _exit_on_signal)

    # Memory sync BEFORE proxy startup — sync headroom DB ↔ Claude's files
    if memory:
        try:
            mem_dir = Path.cwd() / ".headroom"
            mem_dir.mkdir(parents=True, exist_ok=True)
            _sync_db = str(mem_dir / "memory.db")
            _sync_user = os.environ.get("USER", os.environ.get("USERNAME", "default"))

            click.echo(f"  Syncing memory (user={_sync_user})...")
            sync_result = run(
                [
                    sys.executable,
                    "-m",
                    "headroom.memory.sync",
                    "--db",
                    _sync_db,
                    "--user",
                    _sync_user,
                    "--agent",
                    "claude",
                    "--force",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if sync_result.returncode == 0 and sync_result.stdout.strip():
                import json as _json

                stats = _json.loads(sync_result.stdout.strip().split("\n")[-1])
                imp, exp, ms = stats["imported"], stats["exported"], stats["ms"]
                if imp or exp:
                    click.echo(f"  Memory synced: {imp} imported, {exp} exported ({ms}ms)")
                else:
                    click.echo(f"  Memory: up to date ({ms}ms)")
            elif sync_result.returncode != 0:
                click.echo(f"  Warning: memory sync error: {sync_result.stderr[-200:]}")
        except Exception as e:
            click.echo(f"  Warning: memory sync failed: {e}")

    try:
        click.echo()
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo("  ║            HEADROOM WRAP: CLAUDE              ║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        # Detect Foundry mode: Claude Code uses ANTHROPIC_FOUNDRY_BASE_URL instead of
        # ANTHROPIC_BASE_URL when CLAUDE_CODE_USE_FOUNDRY=1 is set.
        # Users typically set ANTHROPIC_FOUNDRY_RESOURCE (the resource name) rather
        # than the full ANTHROPIC_FOUNDRY_BASE_URL.  When the URL is absent we derive
        # it from the resource name so the proxy has an upstream to forward to.
        foundry_upstream = None
        if os.environ.get("CLAUDE_CODE_USE_FOUNDRY"):
            foundry_upstream = os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL")
            if not foundry_upstream:
                resource = os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE", "").strip()
                if resource:
                    foundry_upstream = _foundry_upstream_url(resource)

        # Detect Vertex mode: with CLAUDE_CODE_USE_VERTEX=1, Claude Code IGNORES
        # ANTHROPIC_BASE_URL and authenticates to Google Vertex with GCP ADC. The
        # documented way to route its Vertex :rawPredict / :streamRawPredict
        # traffic through a gateway is ANTHROPIC_VERTEX_BASE_URL. Point it at
        # Headroom and the proxy compresses the request, then forwards to the
        # real regional Vertex host (derived per-request from the path's
        # location) using Claude Code's own ADC token — no API key, no creds held
        # by Headroom. This is the turnkey Vertex compression path.
        use_vertex = bool(os.environ.get("CLAUDE_CODE_USE_VERTEX"))
        proxy_url = _claude_proxy_base_url(port)
        vertex_upstream = _vertex_target_api_url_from_claude_env(proxy_url) if use_vertex else None

        # Issue #1353: when none of the explicit override channels (Foundry,
        # Vertex, ANTHROPIC_TARGET_API_URL) apply, inherit a pre-existing
        # ANTHROPIC_BASE_URL as the upstream so wrap doesn't silently revert
        # the user's LiteLLM/custom-gateway routing back to api.anthropic.com.
        custom_upstream: str | None = None
        if (
            not foundry_upstream
            and not use_vertex
            and not os.environ.get("ANTHROPIC_TARGET_API_URL")
        ):
            custom_upstream = _detect_inbound_anthropic_upstream(port)

        upstream_for_proxy = foundry_upstream or custom_upstream

        _register_proxy_client(port)
        proxy_holder[0], actual_port = _ensure_proxy(
            port,
            no_proxy,
            learn=learn,
            memory=memory,
            agent_type="claude",
            code_graph=code_graph,
            backend=backend,
            region=region,
            anthropic_api_url=upstream_for_proxy,
            vertex_api_url=vertex_upstream,
            clear_vertex_api_url=use_vertex and vertex_upstream is None,
        )
        if actual_port != port:
            _unregister_proxy_client(port)
            _register_proxy_client(actual_port)
        port_holder[0] = actual_port
        _push_runtime_env(actual_port, no_proxy)

        if not no_mcp:
            from headroom.mcp_registry import ClaudeRegistrar

            _setup_headroom_mcp(ClaudeRegistrar(), actual_port, verbose=verbose)
        elif verbose:
            click.echo("  Skipping MCP retrieve tool (--no-mcp)")

        # Coding-task compressor: Serena (retires any legacy tokensave entry).
        from headroom.mcp_registry import CLAUDE_SERENA_CONTEXT, ClaudeRegistrar

        _setup_coding_compressor(
            ClaudeRegistrar(),
            serena_context=CLAUDE_SERENA_CONTEXT,
            serena=serena,
            no_serena=no_serena,
            no_tokensave=no_tokensave,
            verbose=verbose,
        )

        proxy_url = _claude_proxy_base_url(actual_port)
        click.echo()
        click.echo("  Launching Claude Code (API routed through Headroom)...")
        if use_vertex:
            click.echo(
                f"  Vertex mode: ANTHROPIC_VERTEX_BASE_URL={proxy_url} "
                "→ compress, then forward to Vertex with your GCP ADC token"
            )
        elif foundry_upstream:
            click.echo(
                f"  Foundry mode: ANTHROPIC_FOUNDRY_BASE_URL={_foundry_proxy_url(proxy_url)} → upstream {foundry_upstream}"
            )
        elif custom_upstream:
            click.echo(f"  ANTHROPIC_BASE_URL={proxy_url} → upstream {custom_upstream}")
        else:
            click.echo(f"  ANTHROPIC_BASE_URL={proxy_url}")
            # Issue #1779: Claude Code 2.1.196+ deterministically disables
            # first-party Remote Control (/rc) behind a custom ANTHROPIC_BASE_URL.
            # Warn accurately — but only for subscription sessions that ever had
            # RC (skip API-key/cloud auth) and only when the installed version is
            # at/after the gate (or unknown). The gate is upstream; Headroom
            # cannot restore RC, so this is a launch-time notice, not a fix.
            # Detecting the version shells out to `claude --version`, so skip that
            # subprocess for auth modes we would never warn about anyway.
            _cc_version = (
                detect_claude_code_version(claude_bin)
                if remote_control_applies_to_auth(os.environ)
                else None
            )
            if remote_control_gate_active(proxy_url, os.environ, _cc_version):
                click.echo(
                    "  "
                    + remote_control_gate_message(
                        f"the wrapped Claude session's {REMOTE_CONTROL_BASE_URL_ENV}",
                        version=_cc_version,
                    )
                )
                # Session-accurate sibling co-report: reflect what THIS launch
                # actually does with #746/#1158 (never claim deferral is on for
                # a --tool-search false session, never advise --1m twice).
                click.echo(
                    "  "
                    + remote_control_sibling_gate_note(
                        tool_search_active=_tool_search_mode_is_active(
                            _resolved_tool_search_mode(tool_search)
                        ),
                        context_1m_enabled=context_1m,
                    )
                )
        if claude_args:
            click.echo(f"  Extra args: {' '.join(claude_args)}")
        _print_telemetry_notice()
        click.echo()

        env = os.environ.copy()
        if use_vertex:
            # Claude Code stays in Vertex mode (keeps CLAUDE_CODE_USE_VERTEX,
            # ANTHROPIC_VERTEX_PROJECT_ID, CLOUD_ML_REGION, ADC — all inherited);
            # we only redirect its Vertex endpoint to Headroom.
            env["ANTHROPIC_VERTEX_BASE_URL"] = proxy_url
        elif foundry_upstream:
            # ANTHROPIC_FOUNDRY_BASE_URL is the base URL the Anthropic SDK
            # appends /v1/messages to.  The real Foundry URL includes /anthropic,
            # so the proxy URL must mirror that structure.
            env["ANTHROPIC_FOUNDRY_BASE_URL"] = _foundry_proxy_url(proxy_url)
        else:
            env["ANTHROPIC_BASE_URL"] = proxy_url

        # Issue #951: write to settings.json so daemon-spawned conversation
        # workers (which read settings.json fresh rather than inheriting the
        # daemon's environment) also route through Headroom.
        _settings_vertex[0] = bool(use_vertex)
        _settings_foundry[0] = bool(foundry_upstream) and not _settings_vertex[0]
        # _wrap_settings_path is bound before the try (above) so the finally is
        # always safe; the value is unchanged here.
        _check_and_clear_stale_wrap_marker(
            _wrap_settings_path,
            key=_claude_wrap_base_url_env_key(
                foundry_mode=_settings_foundry[0], vertex_mode=_settings_vertex[0]
            ),
        )
        _saved_base_url[0] = _write_claude_wrap_base_url(
            (
                _foundry_proxy_url(proxy_url)
                if _settings_foundry[0]
                else env["ANTHROPIC_VERTEX_BASE_URL"]
                if _settings_vertex[0]
                else proxy_url
            ),
            foundry_mode=_settings_foundry[0],
            vertex_mode=_settings_vertex[0],
            settings_path=_wrap_settings_path,
            port=port,
        )
        # Issue #2221: pair the marker just written with a reader. wrap installs
        # no hook of its own, so a session that only ran `wrap` (never `init`)
        # had nothing to clear a dead-proxy base_url. SessionStart-only.
        _ensure_claude_wrap_selfheal_hook(_wrap_settings_path)

        # Per-project savings attribution: tag every request with the launch
        # directory's name via X-Headroom-Project (user override wins).
        _apply_project_header_env(env)

        # Issue #746: keep Claude Code's on-demand tool loading on through the
        # proxy so tool schemas are not eagerly materialized into local context.
        _tool_search_value = _configure_tool_search_env(env, tool_search)
        _resolved_tool_search_value = env.get(_TOOL_SEARCH_ENV, "")
        _saved_tool_search[0] = _write_claude_wrap_tool_search(
            _resolved_tool_search_value,
            settings_path=_wrap_settings_path,
        )
        if _tool_search_value is not None:
            # Describe what the written value actually does: --tool-search
            # false/0/no/off turns deferral OFF, and the banner must say so
            # rather than repeat "kept on" (issue #1779 accuracy rule).
            _tool_search_state = (
                "on-demand tool loading kept on"
                if _tool_search_mode_is_active(_tool_search_value)
                else "on-demand tool loading DISABLED per your setting"
            )
            click.echo(
                f"  {_TOOL_SEARCH_ENV}={_tool_search_value} ({_tool_search_state}; issue #746)"
            )
        elif verbose:
            click.echo(
                f"  {_TOOL_SEARCH_ENV}={env.get(_TOOL_SEARCH_ENV)} "
                "(using your existing environment value)"
            )

        # Issue #1158: opt-in 1M context window. Claude Code only sends the
        # context-1m beta header when the model id carries the [1m] suffix, so
        # force it via ANTHROPIC_MODEL on the launched process.
        if context_1m:
            env[_ANTHROPIC_MODEL_ENV] = _resolve_1m_model(env.get(_ANTHROPIC_MODEL_ENV))
            # An explicit pass-through --model outranks ANTHROPIC_MODEL in Claude
            # Code, so add the suffix there too or the env var is silently
            # shadowed and the window stays 200k (#2915). Report what will
            # actually take effect rather than the shadowed env value.
            claude_args, _model_flag_1m = _apply_1m_to_claude_args(claude_args)
            if _model_flag_1m is not None:
                click.echo(f"  --model {_model_flag_1m} (1M context window; issue #1158)")
            else:
                click.echo(
                    f"  {_ANTHROPIC_MODEL_ENV}={env[_ANTHROPIC_MODEL_ENV]} "
                    "(1M context window; issue #1158)"
                )

        result = subprocess.run([claude_bin, *claude_args], env=env)
        raise SystemExit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        if _saved_tool_search[0] is not _tool_search_not_written:
            _restore_claude_wrap_tool_search(
                cast(str | None, _saved_tool_search[0]),
                settings_path=_wrap_settings_path,
            )
        _restore_claude_wrap_base_url(
            _saved_base_url[0],
            foundry_mode=_settings_foundry[0],
            vertex_mode=_settings_vertex[0],
            settings_path=_wrap_settings_path,
        )
        cleanup()


# =============================================================================
# Claude Code (unwrap)
# =============================================================================


def _warn_if_proxy_env_leaked(port: int) -> None:
    """Issue #2238: surface a proxy URL that survived unwrap in the live shell.

    ``unwrap_claude`` restores settings.local.json, but if ``ANTHROPIC_BASE_URL``
    (or the Foundry/Vertex equivalents) was exported into the current shell or a
    persistent profile, it outlives the JSON edit and Claude keeps trying to reach
    the (now unwrapped) proxy, failing with a connection error. The user previously
    had to discover ``Remove-Item Env:ANTHROPIC_BASE_URL`` by hand — emit it here.
    """
    proxy_host = f"127.0.0.1:{port}"
    leaked = []
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_FOUNDRY_BASE_URL", "ANTHROPIC_VERTEX_BASE_URL"):
        value = os.environ.get(name, "").strip()
        if proxy_host in value:
            leaked.append((name, value))
    if not leaked:
        return
    click.echo("  ⚠ Headroom's proxy URL is still exported in this shell's environment:")
    for name, value in leaked:
        click.echo(f"      {name}={value}")
    click.echo(
        "    Claude will keep routing through the (now unwrapped) proxy and fail to connect."
    )
    click.echo("    Clear it for the current shell, then restart Claude Code:")
    click.echo("      PowerShell:  Remove-Item Env:ANTHROPIC_BASE_URL")
    click.echo("      bash/zsh:    unset ANTHROPIC_BASE_URL")
    click.echo(
        "    If it reappears after restart, remove it from your shell profile "
        "(e.g. $PROFILE / ~/.bashrc / ~/.zshrc)."
    )


@unwrap.command("claude")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
@click.option("--keep-mcp", is_flag=True, help="Keep Headroom MCP registrations")
def unwrap_claude(
    port: int,
    no_stop_proxy: bool,
    keep_mcp: bool,
) -> None:
    """Undo durable setup from ``headroom wrap claude``."""
    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║          HEADROOM UNWRAP: CLAUDE              ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()

    if not keep_mcp:
        from headroom.mcp_registry import ClaudeRegistrar

        registrar = ClaudeRegistrar()
        if registrar.detect():
            removed_headroom = registrar.unregister_server("headroom")
            removed_code_graph = registrar.unregister_server(_CBM_MCP_SERVER_NAME)
            tokensave_status = _remove_headroom_installed_tokensave_mcp(registrar)
            serena_status = _remove_headroom_installed_serena_mcp(registrar)
            if removed_headroom:
                click.echo("  Removed Headroom MCP retrieve tool from Claude.")
            else:
                click.echo("  Headroom MCP retrieve tool was not registered in Claude.")
            if removed_code_graph:
                click.echo("  Removed legacy codebase-memory-mcp code graph server from Claude.")
            if tokensave_status == "removed":
                click.echo("  Removed Headroom-installed tokensave MCP server from Claude.")
            elif tokensave_status == "failed":
                click.echo(
                    "  tokensave MCP server matched Headroom ledger but could not be removed."
                )
            if serena_status == "removed":
                click.echo("  Removed Headroom-installed Serena MCP server from Claude.")
            elif serena_status == "failed":
                click.echo("  Serena MCP server matched Headroom ledger but could not be removed.")
        else:
            click.echo("  Claude Code not detected; skipped MCP cleanup.")
    else:
        click.echo("  Kept Claude MCP registrations (--keep-mcp).")

    if _remove_claude_managed_hooks():
        click.echo("  Removed Headroom-managed hooks and proxy env from settings.json.")
    else:
        click.echo("  No Headroom-managed hooks found in settings.json.")

    _unwrap_settings_path = Path.cwd() / ".claude" / "settings.local.json"
    if _remove_claude_wrap_selfheal_hook(_unwrap_settings_path):
        click.echo("  Removed Headroom wrap self-heal SessionStart hook (issue #2221).")
    for _foundry, _vertex in ((False, False), (True, False), (False, True)):
        _key = _claude_wrap_base_url_env_key(foundry_mode=_foundry, vertex_mode=_vertex)
        _marker = _read_wrap_marker(_unwrap_settings_path)
        _prior = (
            _marker.get("previous") if _marker is not None and _marker.get("key") == _key else None
        )
        _restore_claude_wrap_base_url(
            _prior,
            foundry_mode=_foundry,
            vertex_mode=_vertex,
            settings_path=_unwrap_settings_path,
            # unwrap is the user asking for their settings back, so it drops
            # every wrap session's claim rather than deferring to a live
            # sibling and silently doing nothing (#3205).
            force=True,
        )

    # Issue #2238: unwrap restores settings.local.json, but a proxy URL that was
    # exported into the live shell (or a persistent profile) survives unwrap and
    # leaves Claude unable to reach the real API ("connection error" until the
    # user manually runs `Remove-Item Env:ANTHROPIC_BASE_URL`). Warn loudly and
    # give the exact per-shell fix instead of leaving the user to discover it.
    _warn_if_proxy_env_leaked(port)

    click.echo()
    clean_unwrap = True
    if no_stop_proxy:
        click.echo("  Kept proxy stop disabled (--no-stop-proxy).")
        clean_unwrap = False
    else:
        clean_unwrap = _echo_claude_unwrap_route_cleanup(_unwrap_claude_route_cleanup(port), port)
    if clean_unwrap:
        click.echo("✓ Claude is no longer durably wrapped by Headroom.")
    else:
        click.echo(
            "  Claude local wrap settings were removed, but effective routing residue remains."
        )
    click.echo()


# =============================================================================
# GitHub Copilot CLI
# =============================================================================


def _require_copilot_subscription_resolution() -> CopilotSubscriptionTokenResolution:
    resolution = resolve_subscription_bearer_token_details()
    if resolution is None:
        raise click.ClickException(
            "GitHub Copilot subscription mode requires a reusable GitHub/Copilot bearer "
            "token, but none could be resolved. Run `headroom copilot-auth login` first, or set "
            "GITHUB_COPILOT_TOKEN / GITHUB_COPILOT_GITHUB_TOKEN."
        )
    return resolution


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--backend",
    default=None,
    help="API backend for the proxy: 'anthropic', 'anyllm', 'litellm-vertex', etc. (env: HEADROOM_BACKEND)",
)
@click.option(
    "--anyllm-provider",
    default=None,
    help="Provider for any-llm backend: openai, mistral, groq, etc. (env: HEADROOM_ANYLLM_PROVIDER)",
)
@click.option(
    "--region", default=None, help="Cloud region for Bedrock/Vertex (env: HEADROOM_REGION)"
)
@click.option(
    "--provider-type",
    type=click.Choice(["auto", "anthropic", "openai"]),
    default="auto",
    show_default=True,
    help="Copilot BYOK provider mode. 'auto' uses anthropic for the default proxy backend and openai for translated backends.",
)
@click.option(
    "--wire-api",
    type=click.Choice(["completions", "responses"]),
    default=None,
    help="OpenAI-compatible Copilot wire API. Defaults to 'completions' when provider-type resolves to openai.",
)
@click.option(
    "--subscription",
    is_flag=True,
    help=(
        "Experimental: route GitHub-authenticated Copilot CLI traffic through Headroom "
        "without requiring a provider API key."
    ),
)
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--native",
    is_flag=True,
    help=(
        "Route Copilot's own GitHub-authenticated API through Headroom instead of "
        "the single-model BYOK override. Keeps native model aliases and /model switching."
    ),
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("copilot_args", nargs=-1, type=click.UNPROCESSED)
def copilot(
    port: int,
    no_proxy: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    provider_type: str,
    wire_api: str | None,
    subscription: bool,
    memory: bool,
    native: bool,
    verbose: bool,
    copilot_args: tuple[str, ...],
) -> None:
    """Launch GitHub Copilot CLI through Headroom proxy.

    \b
    Configures Copilot CLI BYOK provider variables so Copilot routes through
    the local Headroom proxy. In auto mode, the wrapper uses Anthropic-style
    routing for the stock proxy backend and OpenAI-compatible routing for
    translated backends such as any-llm and LiteLLM.

    \b
    Examples:
        headroom wrap copilot -- --model claude-sonnet-4-20250514
        headroom wrap copilot --backend anyllm --anyllm-provider groq -- --model gpt-4o
        headroom wrap copilot --provider-type openai --wire-api responses -- --model gpt-5.4
        headroom wrap copilot --subscription -- --model gpt-4.1

    \b
    Copilot hosted API (--subscription and the implicit OAuth path) routes to the
    generic host https://api.githubcopilot.com, which serves the full model set.
    Enterprise / data-residency accounts provisioned on a dedicated host pin it
    explicitly with GITHUB_COPILOT_API_URL (the override flows through to upstream).
    See TESTING-copilot-subscription.md for details.
    """
    copilot_bin = shutil.which("copilot")
    if not copilot_bin:
        click.echo("Error: 'copilot' not found in PATH.")
        click.echo(
            "Install GitHub Copilot CLI: "
            "https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli"
        )
        raise SystemExit(1)

    explicit_subscription = subscription
    effective_backend = backend or os.environ.get("HEADROOM_BACKEND")
    if _check_proxy(port):
        running_backend = _detect_running_proxy_backend(port)
        if effective_backend and running_backend and effective_backend != running_backend:
            raise click.ClickException(
                f"Proxy already running on port {port} with backend '{running_backend}'. "
                f"Stop it or rerun with --backend {running_backend}."
            )
        effective_backend = running_backend or effective_backend

    if native:
        subscription = True
        if provider_type == "anthropic":
            raise click.ClickException(
                "--native does not use the BYOK provider override; drop --provider-type anthropic."
            )
        if wire_api is not None:
            raise click.ClickException(
                "--native selects the wire per request; drop the BYOK-only --wire-api option."
            )

    effective_provider_type = _resolve_copilot_provider_type(effective_backend, provider_type)
    if subscription:
        if effective_backend not in (None, "", "anthropic"):
            raise click.ClickException(
                "--subscription routes to GitHub Copilot's hosted API and cannot be combined "
                "with translated backends such as anyllm or litellm-*."
            )
        if provider_type == "anthropic":
            raise click.ClickException(
                "--subscription uses Copilot's OpenAI-compatible hosted API path; "
                "do not combine it with --provider-type anthropic."
            )
        effective_provider_type = "openai"
    _validate_copilot_configuration(
        provider_type=effective_provider_type,
        wire_api=wire_api,
        backend=effective_backend,
    )

    env = os.environ.copy()
    _scrub_copilot_proxy_seed_env(env)
    openai_api_url: str | None = None
    copilot_proxy_token: str | None = None
    copilot_refresh_oauth_token: str | None = None
    copilot_api_token_expires_at: float | None = None
    client_bearer: str | None = None
    subscription_resolution: CopilotSubscriptionTokenResolution | None = None
    anthropic_api_url: str | None = None
    use_copilot_oauth = _should_use_copilot_oauth(
        backend=effective_backend,
        provider_type=provider_type,
        env=env,
        force_subscription=subscription,
    )
    # Without a provider key, the old implicit OAuth lane still configured
    # Copilot as a one-model BYOK client. Native aliases (and runtime /model
    # switches) were then forwarded literally and rejected by GitHub (#1910).
    # Explicit --subscription remains on its existing fixed-wire behavior;
    # implicit GitHub OAuth uses Copilot's own routing automatically.
    if use_copilot_oauth and not explicit_subscription:
        native = True

    if use_copilot_oauth:
        if subscription:
            subscription_resolution = _require_copilot_subscription_resolution()
            client_bearer = subscription_resolution.token
        else:
            client_bearer = resolve_client_bearer_token()
        if not client_bearer:
            raise click.ClickException(
                "GitHub Copilot subscription mode requires a reusable GitHub/Copilot bearer "
                "token, but none could be resolved. Run `headroom copilot-auth login` first, or set "
                "GITHUB_COPILOT_TOKEN / GITHUB_COPILOT_GITHUB_TOKEN."
            )

        if native:
            openai_api_url = (
                subscription_resolution.api_url
                if subscription_resolution is not None
                else resolve_copilot_api_url(client_bearer)
            )
            env, env_vars_display = _build_copilot_native_launch_env(
                port=port,
                environ=env,
                project=_project_name_from_cwd(),
            )
            env["GITHUB_COPILOT_API_URL"] = openai_api_url
            env["OPENAI_TARGET_API_URL"] = openai_api_url
            env["ANTHROPIC_TARGET_API_URL"] = openai_api_url
            anthropic_api_url = openai_api_url
            copilot_proxy_token = client_bearer
            if subscription_resolution is not None:
                copilot_refresh_oauth_token = subscription_resolution.refresh_oauth_token
                copilot_api_token_expires_at = subscription_resolution.api_token_expires_at
            support = _native_api_url_supported(environ=os.environ)
            if support is False:
                raise click.ClickException(
                    "This Copilot CLI build does not reference COPILOT_API_URL; refusing "
                    "a native launch that could silently bypass Headroom."
                )
            if support is None and verbose:
                click.echo("  Note: could not verify this Copilot CLI's COPILOT_API_URL support.")
        else:
            selected_model = _copilot_model_from_args(copilot_args, env)

        # ``--model auto`` is a Copilot-internal routing token that the BYOK
        # API rejects with ``400 The requested model is not supported``.  In
        # subscription/OAuth mode we route to the real Copilot hosted API, so
        # Copilot's own native auto-selection works fine — we just need to
        # strip the ``--model auto`` flag before launch so Copilot doesn't
        # forward it to the provider endpoint.
        if not native and _is_auto_model(selected_model):
            copilot_args = _strip_auto_model_args(copilot_args)
            selected_model = None
            click.echo(
                "  Note: '--model auto' is not forwarded to the Copilot API "
                "(it would cause a 400). Removed it; Copilot will use its own "
                "automatic model selection."
            )

        if not native:
            env_wire_api = env.get("COPILOT_PROVIDER_WIRE_API")
            effective_wire_api = wire_api or (
                env_wire_api
                if env_wire_api in {"completions", "responses"}
                else _copilot_default_wire_api_for_model(selected_model)
            )
            env["COPILOT_PROVIDER_TYPE"] = "openai"
            # Per-project savings: the Copilot CLI cannot send custom headers, so
            # the project rides as a /p/<name> base-URL prefix the proxy strips.
            env["COPILOT_PROVIDER_BASE_URL"] = _with_project_prefix(
                f"http://127.0.0.1:{port}/v1", _project_name_from_cwd()
            )
            env["COPILOT_PROVIDER_WIRE_API"] = effective_wire_api
            env["COPILOT_PROVIDER_BEARER_TOKEN"] = client_bearer
            env["GITHUB_COPILOT_USE_TOKEN_EXCHANGE"] = "false"
            env.pop("COPILOT_PROVIDER_API_KEY", None)
            # Hand the exact token we resolved (and, for --subscription, validated
            # against GitHub) to the proxy explicitly via copilot_proxy_token below.
            # The proxy pins it as GITHUB_COPILOT_API_TOKEN, so upstream auth is
            # deterministic instead of the proxy re-running unvalidated discovery
            # (read_cached_oauth_token returns the *first* candidate, which may not
            # be the one the wrapper approved → environment-dependent 401s). Passing
            # it as a launch argument — rather than mutating this process's global
            # os.environ — keeps the token off shared state and out of unrelated
            # code paths.
            copilot_proxy_token = client_bearer
            if subscription_resolution is not None:
                copilot_refresh_oauth_token = subscription_resolution.refresh_oauth_token
                copilot_api_token_expires_at = subscription_resolution.api_token_expires_at
            env_vars_display = [
                "COPILOT_PROVIDER_TYPE=openai",
                f"COPILOT_PROVIDER_BASE_URL={env['COPILOT_PROVIDER_BASE_URL']}",
                f"COPILOT_PROVIDER_WIRE_API={effective_wire_api}",
                (
                    "COPILOT_AUTH_MODE=github-subscription-experimental"
                    if subscription
                    else "COPILOT_AUTH_MODE=github-oauth"
                ),
            ]
            # Non-subscription OAuth keeps upstream's generic-host policy from
            # #610. Subscription mode can use the endpoint returned by the Copilot
            # token exchange, which is how Business accounts advertise their API
            # host without requiring users to configure it manually.
            openai_api_url = (
                subscription_resolution.api_url
                if subscription_resolution is not None
                else resolve_copilot_api_url(client_bearer)
            )
            env["GITHUB_COPILOT_API_URL"] = openai_api_url
            env["OPENAI_TARGET_API_URL"] = openai_api_url
            env_vars_display.append(f"COPILOT_PROVIDER_API_URL={openai_api_url}")
    else:
        env, env_vars_display = _build_copilot_launch_env(
            port=port,
            provider_type=effective_provider_type,
            wire_api=wire_api,
            environ=env,
            project=_project_name_from_cwd(),
        )

        if not env.get("COPILOT_PROVIDER_API_KEY"):
            src = _copilot_provider_key_source(effective_provider_type)
            click.echo(
                f"\n  Error: Copilot BYOK mode requires a provider API key.\n"
                f"  `headroom wrap copilot` uses Copilot's BYOK mode, which bypasses GitHub's\n"
                f"  Copilot API and routes requests directly to the model provider through the\n"
                f"  Headroom proxy. A GitHub Copilot subscription alone is not sufficient.\n\n"
                f"  Set one of:\n"
                f"    export {src}=sk-...          # recommended\n"
                f"    export COPILOT_PROVIDER_API_KEY=sk-...  # also works\n"
            )
            raise SystemExit(1)

    if not subscription and not native and not _copilot_model_configured(copilot_args, env):
        # Distinguish between "--model auto" (wrong model for BYOK) and
        # genuinely missing model (no --model flag at all).
        raw_model = _copilot_model_from_args(copilot_args, env)
        if _is_auto_model(raw_model):
            click.echo(
                "  Error: '--model auto' is not supported in Copilot BYOK mode.\n"
                "  BYOK routes to an external provider (Anthropic/OpenAI) which\n"
                "  does not recognise 'auto' as a model name — the request will\n"
                "  fail with a 400 error.\n"
                "  Options:\n"
                "    • Use a concrete model: --model gpt-4o\n"
                "    • Use subscription mode for native auto-routing:\n"
                "      headroom wrap copilot --subscription -- --model auto"
            )
            raise SystemExit(1)
        else:
            click.echo(
                "  Note: Copilot BYOK requires a model. Pass `--model <name>` "
                "or set `COPILOT_MODEL` / `COPILOT_PROVIDER_MODEL_ID`."
            )

    _launch_tool(
        binary=copilot_bin,
        args=copilot_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="COPILOT",
        env_vars_display=env_vars_display,
        learn=False,
        memory=memory,
        agent_type="copilot",
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        openai_api_url=openai_api_url,
        anthropic_api_url=anthropic_api_url,
        copilot_api_token=copilot_proxy_token,
        copilot_refresh_oauth_token=copilot_refresh_oauth_token,
        copilot_api_token_expires_at=copilot_api_token_expires_at,
    )


# =============================================================================
# GitHub Copilot CLI (unwrap)
# =============================================================================


@wrap.command("vscode")
@click.option("--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--settings-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override settings.json path (Insiders, VSCodium, portable profiles)",
)
@click.option(
    "--configure/--no-configure",
    default=True,
    help="Safely add/update Headroom's transparent Copilot proxy settings",
)
def vscode_copilot(
    port: int,
    memory: bool,
    settings_file: Path | None,
    configure: bool,
) -> None:
    """Run Headroom for GitHub Copilot inside Visual Studio Code.

    Transparently overrides Copilot's proxy and CAPI endpoints, preserving the
    model selected in VS Code. It does not edit Codex settings.
    """
    resolution = _require_copilot_subscription_resolution()
    target_settings = settings_file or vscode_settings_path()

    def _print_setup(actual_port: int) -> None:
        if configure:
            action = configure_vscode_proxy_settings(
                target_settings,
                vscode_proxy_url(actual_port, _project_name_from_cwd()),
            )
            click.echo(f"  VS Code Copilot proxy settings {action}: {target_settings}")
            click.echo(
                "  Keep using Copilot's normal model picker; the selected model is preserved."
            )
            return
        click.echo("  Add these user settings to VS Code:")
        click.echo(
            f'  "github.copilot.advanced.debug.overrideProxyUrl": "{vscode_proxy_url(actual_port, _project_name_from_cwd())}",'
        )
        click.echo(
            f'  "github.copilot.advanced.debug.overrideCapiUrl": "{vscode_proxy_url(actual_port, _project_name_from_cwd())}"'
        )

    _run_proxy_only_watcher(
        agent_label="VS CODE COPILOT",
        port=port,
        no_proxy=False,
        learn=False,
        memory=memory,
        agent_type="copilot",
        print_setup_lines=_print_setup,
        openai_api_url=resolution.api_url,
        copilot_api_token=resolution.token,
        copilot_refresh_oauth_token=resolution.refresh_oauth_token,
        copilot_api_token_expires_at=resolution.api_token_expires_at,
    )


@unwrap.command("vscode")
@click.option(
    "--settings-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override settings.json path",
)
def unwrap_vscode_copilot(settings_file: Path | None) -> None:
    """Remove only Headroom's transparent VS Code Copilot proxy settings."""
    target_settings = settings_file or vscode_settings_path()
    if remove_vscode_proxy_settings(target_settings):
        click.echo(f"Removed Headroom Copilot proxy settings from {target_settings}")
    else:
        click.echo(f"No Headroom Copilot proxy settings found in {target_settings}")


# =============================================================================
# Claude Code for VS Code
# =============================================================================


@wrap.command("vscode-claude")
@click.option("--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--settings-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override Claude Code user settings.json path",
)
@click.option(
    "--configure/--no-configure",
    default=True,
    help="Safely add/update Claude Code's proxy environment settings",
)
def vscode_claude(
    port: int,
    memory: bool,
    settings_file: Path | None,
    configure: bool,
) -> None:
    """Route VS Code's official Claude Code extension through Headroom.

    Run this from your project, reload VS Code after first setup, and keep this
    command running while using Claude Code. Authentication and model selection
    remain unchanged. Run `headroom unwrap vscode-claude` to restore settings.
    """
    target_settings = settings_file or claude_user_settings_path()

    def _print_setup(actual_port: int) -> None:
        proxy_url = vscode_claude_proxy_url(actual_port, _project_name_from_cwd())
        if configure:
            action = configure_vscode_claude_settings(target_settings, proxy_url)
            click.echo(f"  VS Code Claude Code proxy settings {action}: {target_settings}")
            click.echo("  Next: Reload VS Code, then use the Claude Code panel.")
            click.echo("  Keep this command running. Press Ctrl+C to stop the proxy.")
            click.echo("  Authentication and the selected Claude model are preserved.")
            click.echo("  Undo later with: headroom unwrap vscode-claude")
            click.echo("  Guide: https://docs.headroomlabs.ai/docs/vscode-claude-code")
            return
        click.echo(f"  Add these values under 'env' in {target_settings}:")
        click.echo(f'  "ANTHROPIC_BASE_URL": "{proxy_url}",')
        click.echo(f'  "{_TOOL_SEARCH_ENV}": "{_TOOL_SEARCH_DEFAULT}"')

    _run_proxy_only_watcher(
        agent_label="VS CODE CLAUDE",
        port=port,
        no_proxy=False,
        learn=False,
        memory=memory,
        agent_type="claude",
        print_setup_lines=_print_setup,
    )


@unwrap.command("vscode-claude")
@click.option(
    "--settings-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override Claude Code user settings.json path",
)
def unwrap_vscode_claude(settings_file: Path | None) -> None:
    """Restore settings saved by `headroom wrap vscode-claude`.

    Reload the VS Code window afterward. If setup used --settings-file, pass the
    same path here.
    """
    target_settings = settings_file or claude_user_settings_path()
    if remove_vscode_claude_settings(target_settings):
        click.echo(f"Restored Claude Code settings in {target_settings}")
        click.echo("Reload the VS Code window to apply the restored settings.")
    else:
        click.echo(f"No Headroom VS Code Claude settings found for {target_settings}")


# =============================================================================
# GitHub Copilot CLI (unwrap)
# =============================================================================


@unwrap.command("copilot")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_copilot(port: int, no_stop_proxy: bool) -> None:
    """Undo durable setup from ``headroom wrap copilot``."""
    if not no_stop_proxy:
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)


# =============================================================================
# OpenAI Codex CLI
# =============================================================================


def _prepare_codex_wrap_state(
    *,
    port: int,
    no_mcp: bool,
    no_tokensave: bool,
    serena: bool,
    no_serena: bool,
    memory: bool,
    verbose: bool,
    persistent_routing: bool = True,
) -> None:
    """Prepare the active Codex home for a wrap or prepare-only invocation."""
    # Ensure the Codex home exists before anything tries to detect or write it.
    # ``CodexRegistrar.detect()`` is just ``self._codex_dir.is_dir()``, and until
    # now the directory was created as a *side effect* of injecting the rtk
    # guidance into ``$CODEX_HOME/AGENTS.md``. With the CLI context tools removed
    # nothing creates it, so on a machine where Codex is installed but has never
    # been launched, detect() returned False and Headroom silently skipped the
    # MCP registration — no ``headroom_retrieve``, so every compression marker
    # the proxy emits would be unresolvable. (Latent since #2344 made rtk
    # opt-in; the wrap e2e only masked it by exporting HEADROOM_RTK=1.)
    _codex_home_dir().mkdir(parents=True, exist_ok=True)

    # Snapshot Codex config.toml BEFORE any wrap-time mutation so
    # `headroom unwrap codex` can restore the user's pre-wrap state
    # byte-for-byte. The snapshot is a no-op if the backup already exists
    # or if the file already has Headroom markers, so this is safe to
    # call repeatedly. Crucially this must run before MCP install, which
    # writes its marker block to the same file.
    if persistent_routing:
        _codex_config_file, _codex_backup_file = _codex_config_paths()
        _snapshot_codex_config_if_unwrapped(_codex_config_file, _codex_backup_file)

    # Register headroom MCP server in Codex config.toml so Codex can
    # call headroom_retrieve on compression markers from the proxy.
    if not no_mcp:
        from headroom.mcp_registry import CodexRegistrar

        # Codex starts a long-lived local MCP subprocess from config.toml.
        # If a previous wrap used another port, retrieval can silently point
        # at the wrong proxy while model traffic uses the right one.
        _setup_headroom_mcp(CodexRegistrar(), port, verbose=verbose, force=True)
    elif verbose:
        click.echo("  Skipping MCP retrieve tool (--no-mcp)")

    # Coding-task compressor: Serena (retires any legacy tokensave entry). Codex
    # starts long-lived MCP subprocesses from config.toml, so force re-registration.
    from headroom.mcp_registry import CodexRegistrar

    _setup_coding_compressor(
        CodexRegistrar(),
        serena_context="codex",
        serena=serena,
        no_serena=no_serena,
        no_tokensave=no_tokensave,
        verbose=verbose,
        force=True,
    )

    # Setup memory MCP server for Codex (native tool integration)
    if memory:
        click.echo("  Setting up memory for Codex...")
        mem_dir = Path.cwd() / ".headroom"
        mem_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(mem_dir / "memory.db")
        mem_user = os.environ.get("USER", os.environ.get("USERNAME", "default"))

        # Register MCP server in Codex config
        _inject_memory_mcp_config(mem_user)

        # Inject memory guidance into project AGENTS.md
        agents_md = Path.cwd() / "AGENTS.md"
        _inject_memory_agents_md(agents_md)

        # Sync Claude's memories → DB so MCP search finds them
        try:
            import asyncio

            from headroom.memory.sync import _build_sync_backend, sync_import
            from headroom.memory.sync_adapters.claude_code import (
                ClaudeCodeAdapter,
                get_claude_memory_dir,
            )

            claude_memory_dir = get_claude_memory_dir()

            async def _import_claude_memories() -> int:
                backend = _build_sync_backend(db_path)
                await backend._ensure_initialized()
                adapter = ClaudeCodeAdapter(claude_memory_dir)
                count = await sync_import(backend, adapter, mem_user)
                await backend.close()
                return count

            imported = asyncio.run(_import_claude_memories())
            if imported:
                click.echo(f"  Memory: imported {imported} memories from Claude")
        except Exception as e:
            click.echo(f"  Warning: Claude memory import failed: {e}")

    # Inject Headroom provider into Codex config so WebSocket traffic also
    # routes through the proxy.  Codex ignores OPENAI_BASE_URL for its WS
    # transport unless a custom provider declares supports_websockets = true.
    # NOTE: this must run BEFORE _inject_memory_mcp_config because it rewrites
    # the config file.  Re-inject MCP config after if memory is enabled.
    if persistent_routing:
        _inject_codex_provider_config(port)
        if memory:
            _inject_memory_mcp_config(os.environ.get("USER", os.environ.get("USERNAME", "default")))


def _run_codex_wrap(
    *,
    port: int,
    no_mcp: bool,
    no_tokensave: bool,
    serena: bool,
    no_serena: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    codex_args: tuple,
) -> None:
    """Execute the Codex wrap flow against the durable Codex home."""
    if not no_proxy:
        ensure_proxy_dependencies()

    if prepare_only:
        _prepare_codex_wrap_state(
            port=port,
            no_mcp=no_mcp,
            no_tokensave=no_tokensave,
            serena=serena,
            no_serena=no_serena,
            memory=memory,
            verbose=verbose,
        )
        return

    codex_bin = shutil.which("codex")
    if not codex_bin:
        click.echo("Error: 'codex' not found in PATH.")
        click.echo("Install Codex CLI: npm install -g @openai/codex")
        raise SystemExit(1)

    active_codex_home = _codex_home_dir()
    _offer_dangling_codex_recovery(active_codex_home)
    _prepare_codex_wrap_state(
        port=port,
        no_mcp=no_mcp,
        no_tokensave=no_tokensave,
        serena=serena,
        no_serena=no_serena,
        memory=memory,
        verbose=verbose,
        persistent_routing=False,
    )

    env, env_vars_display = _build_codex_launch_env(port, os.environ)
    env["CODEX_HOME"] = str(active_codex_home)

    def configure_codex_launch(
        actual_port: int,
        current_args: tuple,
        current_env: dict[str, str],
        current_display: list[str],
    ) -> tuple[tuple, dict[str, str], list[str]]:
        del current_display
        return _codex_session_launch_settings(
            port=actual_port,
            codex_args=current_args,
            environ=current_env,
        )

    _launch_tool(
        binary=codex_bin,
        args=codex_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="CODEX",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="codex",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        configure_launch=configure_codex_launch,
    )


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@_serena_instructions_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option(
    "--no-mcp",
    is_flag=True,
    help="Skip headroom MCP server registration (compression markers will be unactionable)",
)
@_code_memory_option
@click.option(
    "--no-tokensave",
    is_flag=True,
    hidden=True,
    help="Deprecated and ignored: tokensave was retired; Serena is the default code memory.",
)
@click.option(
    "--serena",
    is_flag=True,
    hidden=True,
    help="Deprecated: use --code-memory serena. Force the Serena MCP compressor on.",
)
@click.option(
    "--no-serena",
    is_flag=True,
    hidden=True,
    help="Deprecated: use --code-memory none. Register no code-memory MCP.",
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable the proxy's live code-graph file watcher for the current project.",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to AGENTS.md)"
)
@click.option(
    "--backend",
    default=None,
    help="API backend for the proxy: 'anthropic', 'anyllm', 'litellm-vertex', etc. (env: HEADROOM_BACKEND)",
)
@click.option(
    "--anyllm-provider",
    default=None,
    help="Provider for any-llm backend: openai, mistral, groq, etc. (env: HEADROOM_ANYLLM_PROVIDER)",
)
@click.option(
    "--region", default=None, help="Cloud region for Bedrock/Vertex (env: HEADROOM_REGION)"
)
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("codex_args", nargs=-1, type=click.UNPROCESSED)
def codex(
    port: int,
    no_mcp: bool,
    no_tokensave: bool,
    serena: bool,
    no_serena: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    codex_args: tuple,
) -> None:
    """Launch OpenAI Codex CLI through Headroom proxy.

    \b
    Sets OPENAI_BASE_URL to route all OpenAI API calls through Headroom.
    Also registers the headroom MCP server in the active Codex config file
    so Codex can call ``headroom_retrieve`` on compression markers.

    \b
    Examples:
        headroom wrap codex                         # Start proxy + mcp + codex
        headroom wrap codex -- "fix the bug"        # Pass prompt to codex
        headroom wrap codex --no-mcp                # Skip MCP retrieve tool registration
        headroom wrap codex --code-memory none      # No code-memory MCP
        headroom wrap codex --port 9999             # Custom proxy port
        headroom wrap codex --backend anyllm --anyllm-provider groq
    """
    return _run_codex_wrap(
        port=port,
        no_mcp=no_mcp,
        no_tokensave=no_tokensave,
        serena=serena,
        no_serena=no_serena,
        code_graph=code_graph,
        no_proxy=no_proxy,
        learn=learn,
        memory=memory,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        verbose=verbose,
        prepare_only=prepare_only,
        codex_args=codex_args,
    )


# =============================================================================
# Aider
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--backend", default=None, help="API backend: 'anthropic', 'anyllm', 'litellm-vertex', etc."
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Bedrock/Vertex")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("aider_args", nargs=-1, type=click.UNPROCESSED)
def aider(
    port: int,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    aider_args: tuple,
) -> None:
    """Launch aider through Headroom proxy.

    \b
    Sets OPENAI_API_BASE to route all API calls through Headroom.

    \b
    Examples:
        headroom wrap aider                              # Start proxy + aider
        headroom wrap aider -- --model gpt-4o            # Use GPT-4o
        headroom wrap aider -- --model claude-sonnet-4   # Use Claude
        headroom wrap aider --backend litellm-vertex --region us-central1
    """
    if prepare_only:
        return

    aider_bin = shutil.which("aider")
    if not aider_bin:
        click.echo("Error: 'aider' not found in PATH.")
        click.echo("Install aider: pip install aider-chat")
        raise SystemExit(1)

    env, env_vars_display = _build_aider_launch_env(
        port, os.environ, project=_project_name_from_cwd()
    )

    _launch_tool(
        binary=aider_bin,
        args=aider_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="AIDER",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="aider",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# OpenClaude
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--backend", default=None, help="API backend: 'anthropic', 'anyllm', 'litellm-vertex', etc."
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Bedrock/Vertex")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("openclaude_args", nargs=-1, type=click.UNPROCESSED)
def openclaude(
    port: int,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    openclaude_args: tuple,
) -> None:
    """Launch OpenClaude through Headroom proxy.

    \b
    OpenClaude is a prose-format coding CLI (like Aider / Cline); it speaks
    OpenAI- and Anthropic-compatible HTTP, so wrap routes both base URLs
    through the local proxy — same env shape as `wrap aider`.

    \b
    Examples:
        headroom wrap openclaude                         # Start proxy + openclaude
        headroom wrap openclaude -- --model gpt-4o       # Pass args to openclaude
    """
    if prepare_only:
        return

    openclaude_bin = shutil.which("openclaude")
    if not openclaude_bin:
        click.echo("Error: 'openclaude' not found in PATH.")
        click.echo("Install OpenClaude before running `headroom wrap openclaude`.")
        raise SystemExit(1)

    env, env_vars_display = _build_aider_launch_env(
        port, os.environ, project=_project_name_from_cwd()
    )

    _launch_tool(
        binary=openclaude_bin,
        args=openclaude_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="OPENCLAUDE",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="openclaude",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# Mistral Vibe
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("vibe_args", nargs=-1, type=click.UNPROCESSED)
def vibe(
    port: int,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
    vibe_args: tuple,
) -> None:
    """Launch Mistral Vibe through Headroom proxy.

    \b
    Sets VIBE_PROVIDERS to route all Mistral API calls through Headroom.

    \b
    Examples:
        headroom wrap vibe                         # Start proxy + vibe
        headroom wrap vibe -- "fix the bug"        # Pass prompt to vibe
        headroom wrap vibe --port 9999             # Custom proxy port
    """
    if prepare_only:
        return

    vibe_bin = shutil.which("vibe")
    if not vibe_bin:
        click.echo("Error: 'vibe' not found in PATH.")
        click.echo("Install Mistral Vibe: https://github.com/mistralai/mistral-vibe")
        raise SystemExit(1)

    env, env_vars_display = _build_mistral_vibe_launch_env(
        port, os.environ, project=_project_name_from_cwd()
    )

    _launch_tool(
        binary=vibe_bin,
        args=vibe_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="VIBE",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="vibe",
        code_graph=code_graph,
        openai_api_url="https://api.mistral.ai",
    )


# =============================================================================
# Kimi CLI
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--kimi-api-url",
    default="https://api.kimi.com/coding/v1",
    help="Upstream Kimi coding endpoint (default: https://api.kimi.com/coding/v1)",
)
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("kimi_args", nargs=-1, type=click.UNPROCESSED)
def kimi(
    port: int,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    kimi_api_url: str,
    prepare_only: bool,
    kimi_args: tuple,
) -> None:
    """Launch Kimi CLI through Headroom proxy.

    \b
    Sets KIMI_CODE_BASE_URL for managed Kimi Code and KIMI_BASE_URL for legacy
    kimi-cli to route OpenAI-compatible /chat/completions traffic through
    Headroom. Managed Kimi Code needs one `/login` after the proxy URL changes
    so its OAuth slot matches that URL; legacy kimi-cli keeps its existing login.

    \b
    Examples:
        headroom wrap kimi                         # Start proxy + kimi
        headroom wrap kimi -- -m kimi-for-coding   # Pass args to kimi
        headroom wrap kimi --port 9999             # Custom proxy port
        headroom wrap kimi --kimi-api-url https://api.moonshot.ai/v1
    """
    if prepare_only:
        return

    kimi_bin = shutil.which("kimi") or shutil.which("kimi-cli")
    if not kimi_bin:
        click.echo("Error: 'kimi' (or 'kimi-cli') not found in PATH.")
        click.echo("Install Kimi CLI: https://github.com/MoonshotAI/kimi-cli")
        raise SystemExit(1)

    project = _project_name_from_cwd()
    env, env_vars_display = _build_kimi_launch_env(port, os.environ, project=project)

    def configure_kimi_launch(
        actual_port: int,
        current_args: tuple,
        current_env: dict[str, str],
        current_display: list[str],
    ) -> tuple[tuple, dict[str, str], list[str]]:
        del current_display
        updated_env, updated_display = _build_kimi_launch_env(
            actual_port, current_env, project=project
        )
        return current_args, updated_env, updated_display

    _launch_tool(
        binary=kimi_bin,
        args=kimi_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="KIMI",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="kimi",
        code_graph=code_graph,
        openai_api_url=kimi_api_url,
        configure_launch=configure_kimi_launch,
    )


# =============================================================================
# Grok CLI
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@_serena_instructions_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-mcp", is_flag=True, help="Skip headroom MCP server registration")
@_code_memory_option
@click.option(
    "--no-tokensave",
    is_flag=True,
    hidden=True,
    help="Deprecated and ignored: tokensave was retired; Serena is the default code memory.",
)
@click.option(
    "--serena",
    is_flag=True,
    hidden=True,
    help="Deprecated: use --code-memory serena. Force the Serena MCP compressor on.",
)
@click.option(
    "--no-serena",
    is_flag=True,
    hidden=True,
    help="Deprecated: use --code-memory none. Register no code-memory MCP.",
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable the proxy's live code-graph file watcher for the current project.",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--backend",
    default=None,
    help="API backend for the proxy: 'anthropic' (default), 'litellm-xai', etc.",
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Vertex/Bedrock backends")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("grok_args", nargs=-1, type=click.UNPROCESSED)
def grok(
    port: int,
    no_mcp: bool,
    no_tokensave: bool,
    serena: bool,
    no_serena: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    grok_args: tuple,
) -> None:
    """Launch Grok CLI through Headroom proxy.

    \b
    Sets ``GROK_MODELS_BASE_URL`` so Grok routes inference traffic through
    Headroom while preserving native settings and authentication routing.
    Registers the headroom MCP server in ``~/.grok/config.toml`` so Grok can
    call ``headroom_retrieve`` on compression markers.

    \b
    Examples:
        headroom wrap grok                         # Start proxy + grok
        headroom wrap grok -- -p "fix the bug"     # Pass prompt to grok
        headroom wrap grok --no-mcp                # Skip MCP retrieve tool registration
        headroom wrap grok --port 9999             # Custom proxy port
    """
    if not no_mcp:
        from headroom.mcp_registry import GrokRegistrar

        _setup_headroom_mcp(GrokRegistrar(), port, verbose=verbose, force=True)
    elif verbose:
        click.echo("  Skipping MCP retrieve tool (--no-mcp)")

    from headroom.mcp_registry import GrokRegistrar

    _setup_coding_compressor(
        GrokRegistrar(),
        serena_context="grok",
        serena=serena,
        no_serena=no_serena,
        no_tokensave=no_tokensave,
        verbose=verbose,
        force=True,
    )

    if prepare_only:
        return

    grok_bin = shutil.which("grok")
    if not grok_bin:
        click.echo("Error: 'grok' not found in PATH.")
        click.echo("Install Grok CLI: https://docs.x.ai/docs/grok-cli")
        raise SystemExit(1)

    env, env_vars_display = _build_grok_launch_env(
        port, os.environ, project=_project_name_from_cwd()
    )

    _launch_tool(
        binary=grok_bin,
        args=grok_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="GROK",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="grok",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        openai_api_url=_GROK_DEFAULT_API_URL,
    )


# =============================================================================
# Cursor
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to .cursor/rules/)"
)
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
def cursor(
    port: int,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
) -> None:
    """Start Headroom proxy for use with Cursor.

    \b
    Cursor reads its API configuration from its settings UI, not from
    environment variables. This command starts the proxy and prints the Cursor
    settings.

    \b
    After running this command, open Cursor and configure:
        Settings > Models > OpenAI API Key > Advanced > Override Base URL

    \b
    Example:
        headroom wrap cursor                # Start proxy + Cursor settings
        headroom wrap cursor --port 9999    # Custom proxy port
    """
    if prepare_only:
        return

    def _print_cursor_setup(actual_port: int) -> None:
        for line in _render_cursor_setup_lines(actual_port, project=_project_name_from_cwd()):
            click.echo(line)

    _run_proxy_only_watcher(
        agent_label="cursor",
        port=port,
        no_proxy=no_proxy,
        learn=learn,
        memory=memory,
        agent_type="cursor",
        print_setup_lines=_print_cursor_setup,
    )


# =============================================================================
# Grok Build
# =============================================================================


@wrap.command("grok-build", context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
def grok_build(
    port: int,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
) -> None:
    """Start Headroom proxy for use with Grok Build.

    \b
    Grok Build reads model endpoints from ``~/.grok/config.toml``. This
    command starts the proxy (upstream ``https://api.x.ai``, same as
    ``wrap grok``), injects a Headroom-managed ``[model.grok-build]``
    override, and prints next steps.

    \b
    Example:
        headroom wrap grok-build
        headroom wrap grok-build --port 9999
    """
    project = _project_name_from_cwd()
    if prepare_only:
        try:
            config_file = inject_grok_provider_config(port, project=project)
            click.echo(f"  Grok config: injected Headroom proxy override into {config_file}")
        except Exception as e:
            click.echo(f"  Warning: could not update Grok config: {e}")
        return

    def _print_grok_build_setup(actual_port: int) -> None:
        try:
            config_file = inject_grok_provider_config(actual_port, project=project)
            click.echo(f"  Grok config: injected Headroom proxy override into {config_file}")
            click.echo()
        except Exception as e:
            click.echo(f"  Warning: could not update Grok config: {e}")
            click.echo()
        for line in _render_grok_build_setup_lines(actual_port, project=project):
            click.echo(line)

    # Client hop is local proxy via config.toml; upstream must be xAI (not
    # the OpenAI default). Omitting this caused 401s with Grok auth headers.
    _run_proxy_only_watcher(
        agent_label="grok-build",
        port=port,
        no_proxy=no_proxy,
        learn=learn,
        memory=memory,
        agent_type="grok_build",
        print_setup_lines=_print_grok_build_setup,
        openai_api_url=_GROK_DEFAULT_API_URL,
    )


# =============================================================================
# Cline (VS Code extension)
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
def cline(
    port: int,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
) -> None:
    """Start Headroom proxy for use with Cline (VS Code extension).

    \b
    Cline is a VS Code extension that reads its API configuration from the
    VS Code settings UI, not from environment variables. This command starts
    the proxy and prints the Cline settings the user should configure.

    \b
    After running this command, open Cline's settings in VS Code and configure
    the API Base URL to point at the local Headroom proxy.

    \b
    Uninstall: there is no ``headroom unwrap cline`` subcommand — nothing is
    written to the project.

    \b
    Examples:
        headroom wrap cline                  # Start proxy + Cline settings
        headroom wrap cline --port 9999      # Custom proxy port
    """
    if prepare_only:
        return

    def _print_cline_setup(actual_port: int) -> None:
        anthropic_base = _claude_proxy_base_url(actual_port)
        openai_base = f"http://127.0.0.1:{actual_port}/v1"
        click.echo("  Configure Cline in VS Code:")
        click.echo("    Settings > Cline > API Provider")
        click.echo(f"    Anthropic Base URL: {anthropic_base}")
        click.echo(f"    OpenAI Compatible Base URL: {openai_base}")

    _run_proxy_only_watcher(
        agent_label="cline",
        port=port,
        no_proxy=no_proxy,
        learn=learn,
        memory=memory,
        agent_type="cline",
        print_setup_lines=_print_cline_setup,
    )


# =============================================================================
# ZCode (zcode.z.ai desktop app)
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
def zcode(
    port: int,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
) -> None:
    """Start Headroom proxy for use with ZCode (zcode.z.ai desktop app).

    \b
    ZCode is a desktop Electron app that reads its API configuration from
    the settings UI, not from environment variables. This command starts the
    proxy and prints the ZCode settings the user should configure.

    \b
    After running this command, open ZCode and configure:
        Settings > Model Settings > Add Provider
        Set OpenAI Base URL and/or Anthropic Base URL to the proxy URLs.

    \b
    Example:
        headroom wrap zcode                # Start proxy + ZCode settings
        headroom wrap zcode --port 9999    # Custom proxy port
    """
    if prepare_only:
        return

    upstream = _detect_zcode_upstream()
    anthropic_url, openai_url = _zcode_upstream_to_urls(upstream)

    def _print_zcode_setup(actual_port: int = port) -> None:
        click.echo(f"  Detected provider: {upstream.provider_name}")
        click.echo(f"  Upstream: {upstream.base_url}")
        click.echo()
        for line in _render_zcode_setup_lines(actual_port):
            click.echo(line)

    _run_proxy_only_watcher(
        agent_label="zcode",
        port=port,
        no_proxy=no_proxy,
        learn=learn,
        memory=memory,
        agent_type="zcode",
        print_setup_lines=_print_zcode_setup,
        anthropic_api_url=anthropic_url,
        openai_api_url=openai_url,
    )


# =============================================================================
# Continue (VS Code / JetBrains extension)
# =============================================================================


@wrap.command("continue", context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Path to Continue config.json (default: ./.continue/config.json)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
def continue_dev(
    port: int,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    config_path: Path | None,
    verbose: bool,
    prepare_only: bool,
) -> None:
    """Start Headroom proxy for use with Continue (VS Code / JetBrains).

    \b
    Continue reads its model configuration from .continue/config.json (a JSON
    document with a top-level ``systemMessage`` and a ``models`` array). This
    command starts the proxy and prints the per-model ``apiBase`` the user
    should configure manually.

    \b
    Continue is an IDE extension — its API base URL is configured per-model
    in config.json (or via the IDE UI), not via environment variables. The
    config file is overridable via --config.

    \b
    Uninstall: there is no ``headroom unwrap continue`` subcommand — the config
    file is left untouched.

    \b
    Examples:
        headroom wrap continue                # Start proxy + print apiBase
        headroom wrap continue --port 9999    # Custom proxy port
        headroom wrap continue --config path/to/config.json
    """
    config_file = config_path or (Path.cwd() / ".continue" / "config.json")

    if prepare_only:
        return

    def _print_continue_setup(actual_port: int) -> None:
        anthropic_base = _claude_proxy_base_url(actual_port)
        openai_base = f"http://127.0.0.1:{actual_port}/v1"
        click.echo("  Configure Continue in your IDE:")
        click.echo(f"    Edit {config_file} and set, per model:")
        click.echo(f'      "apiBase": "{openai_base}"          # OpenAI-compatible models')
        click.echo(f'      "apiBase": "{anthropic_base}"       # Anthropic models')

    _run_proxy_only_watcher(
        agent_label="continue",
        port=port,
        no_proxy=no_proxy,
        learn=learn,
        memory=memory,
        agent_type="continue",
        print_setup_lines=_print_continue_setup,
    )


# =============================================================================
# Goose (Block)
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--backend", default=None, help="API backend: 'anthropic', 'anyllm', 'litellm-vertex', etc."
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Bedrock/Vertex")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("goose_args", nargs=-1, type=click.UNPROCESSED)
def goose(
    port: int,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    goose_args: tuple,
) -> None:
    """Launch Goose (Block) CLI through Headroom proxy.

    \b
    Sets OPENAI_BASE_URL and ANTHROPIC_BASE_URL to route Goose's API calls
    through Headroom.

    \b
    Uninstall: there is no ``headroom unwrap goose`` subcommand — nothing is
    written to the project.

    \b
    Examples:
        headroom wrap goose                          # Start proxy + goose
        headroom wrap goose -- session               # Start a Goose session
        headroom wrap goose -- --provider anthropic  # Pass args to goose
    """
    if prepare_only:
        return

    goose_bin = shutil.which("goose")
    if not goose_bin:
        click.echo("Error: 'goose' not found in PATH.")
        click.echo("Install Goose: https://block.github.io/goose/")
        raise SystemExit(1)

    # Goose accepts OpenAI- and Anthropic-compatible providers; route both.
    env = os.environ.copy()
    openai_base = f"http://127.0.0.1:{port}/v1"
    anthropic_base = _claude_proxy_base_url(port)
    env["OPENAI_BASE_URL"] = openai_base
    env["OPENAI_API_BASE"] = openai_base
    env["ANTHROPIC_BASE_URL"] = anthropic_base
    env_vars_display = [
        f"OPENAI_BASE_URL={openai_base}",
        f"ANTHROPIC_BASE_URL={anthropic_base}",
    ]

    _launch_tool(
        binary=goose_bin,
        args=goose_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="GOOSE",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="goose",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# OpenHands
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--backend", default=None, help="API backend: 'anthropic', 'anyllm', 'litellm-vertex', etc."
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Bedrock/Vertex")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("openhands_args", nargs=-1, type=click.UNPROCESSED)
def openhands(
    port: int,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    openhands_args: tuple,
) -> None:
    """Launch OpenHands CLI through Headroom proxy.

    \b
    Sets OPENAI_BASE_URL / ANTHROPIC_BASE_URL to route OpenHands' API calls
    through Headroom. Nothing is written to disk, so there is nothing to undo.

    \b
    Examples:
        headroom wrap openhands                # Start proxy + openhands
        headroom wrap openhands -- --task ...  # Pass args to openhands
    """
    if prepare_only:
        return

    openhands_bin = shutil.which("openhands")
    if not openhands_bin:
        click.echo("Error: 'openhands' not found in PATH.")
        click.echo("Install OpenHands: https://docs.all-hands.dev/")
        raise SystemExit(1)

    env = os.environ.copy()
    openai_base = f"http://127.0.0.1:{port}/v1"
    anthropic_base = _claude_proxy_base_url(port)
    env["OPENAI_BASE_URL"] = openai_base
    env["OPENAI_API_BASE"] = openai_base
    env["ANTHROPIC_BASE_URL"] = anthropic_base
    # Also set LLM_BASE_URL for OpenHands' generic LLM provider config.
    env["LLM_BASE_URL"] = openai_base
    env_vars_display = [
        f"OPENAI_BASE_URL={openai_base}",
        f"ANTHROPIC_BASE_URL={anthropic_base}",
        f"LLM_BASE_URL={openai_base}",
    ]
    _launch_tool(
        binary=openhands_bin,
        args=openhands_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="OPENHANDS",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="openhands",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# OpenClaw
# =============================================================================


@wrap.command("openclaw")
@_retired_context_tool_option
@click.option(
    "--plugin-path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Path to local OpenClaw plugin source directory (advanced/dev override)",
)
@click.option(
    "--plugin-spec",
    default=OPENCLAW_NPM_PACKAGE,
    show_default=True,
    help="NPM plugin spec for OpenClaw install (used when --plugin-path is omitted)",
)
@click.option(
    "--skip-build",
    is_flag=True,
    help="Skip npm install/build in local source mode (--plugin-path)",
)
@click.option(
    "--copy",
    is_flag=True,
    help="Install by copying plugin path instead of using --link",
)
@click.option(
    "--proxy-port", default=8787, type=click.IntRange(1, 65535), help="Headroom proxy port"
)
@click.option("--startup-timeout-ms", default=20000, type=int, help="Proxy startup timeout")
@click.option(
    "--gateway-provider-id",
    "gateway_provider_ids",
    multiple=True,
    help="OpenClaw provider id to route through Headroom (repeatable; default: openai-codex)",
)
@click.option(
    "--python-path",
    default=None,
    help="Optional Python executable for proxy launcher fallback",
)
@click.option(
    "--no-auto-start",
    is_flag=True,
    help="Disable plugin auto-start of local headroom proxy",
)
@click.option(
    "--no-restart",
    is_flag=True,
    help="Do not restart OpenClaw gateway at the end",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.option("--existing-entry-json", default=None, hidden=True)
def openclaw(
    plugin_path: Path | None,
    plugin_spec: str,
    skip_build: bool,
    copy: bool,
    proxy_port: int,
    startup_timeout_ms: int,
    gateway_provider_ids: tuple[str, ...],
    python_path: str | None,
    no_auto_start: bool,
    no_restart: bool,
    verbose: bool,
    prepare_only: bool,
    existing_entry_json: str | None,
) -> None:
    """Install and configure Headroom OpenClaw plugin in one command.

    \b
    What this command does:
      1. Installs OpenClaw plugin from npm (or local --plugin-path)
      2. Builds plugin source if --plugin-path is used
      3. Writes minimal plugin config and sets contextEngine slot
      4. Validates config
      5. Restarts OpenClaw gateway (unless --no-restart)

    \b
    Example:
      headroom wrap openclaw
      headroom wrap openclaw --plugin-path C:\\git\\headroom\\plugins\\openclaw
    """
    if prepare_only:
        entry = _build_openclaw_plugin_entry(
            existing_entry=_decode_openclaw_entry_json(existing_entry_json),
            proxy_port=proxy_port,
            startup_timeout_ms=startup_timeout_ms,
            python_path=python_path,
            no_auto_start=no_auto_start,
            gateway_provider_ids=gateway_provider_ids,
            enabled=True,
        )
        click.echo(json.dumps(entry, separators=(",", ":")))
        return

    openclaw_bin = shutil.which("openclaw")
    if not openclaw_bin:
        raise click.ClickException("'openclaw' not found in PATH. Install OpenClaw CLI first.")

    plugin_dir = plugin_path.resolve() if plugin_path else None
    local_source_mode = plugin_dir is not None
    if plugin_dir:
        if not plugin_dir.exists():
            raise click.ClickException(f"Plugin path not found: {plugin_dir}.")
        if not (plugin_dir / "package.json").exists():
            raise click.ClickException(f"Invalid plugin path (missing package.json): {plugin_dir}")
        if not (plugin_dir / "openclaw.plugin.json").exists():
            raise click.ClickException(
                f"Invalid plugin path (missing openclaw.plugin.json): {plugin_dir}"
            )

    npm_bin = shutil.which("npm")
    if local_source_mode and not skip_build and not npm_bin:
        raise click.ClickException(
            "'npm' not found in PATH. Install Node/npm or rerun with --skip-build."
        )

    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║           HEADROOM WRAP: OPENCLAW             ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()
    if local_source_mode:
        click.echo(f"  Plugin source: local ({plugin_dir})")
    else:
        click.echo(f"  Plugin source: npm ({plugin_spec})")

    if local_source_mode and not skip_build:
        click.echo("  Building OpenClaw plugin (npm install + npm run build)...")
        _run_checked([npm_bin or "npm", "install"], cwd=plugin_dir, action="npm install")
        _run_checked([npm_bin or "npm", "run", "build"], cwd=plugin_dir, action="npm run build")
    elif not local_source_mode and skip_build:
        click.echo("  Skipping build: npm install mode does not build local source.")

    effective_python_path = python_path
    if effective_python_path is None and not no_auto_start and sys.executable:
        effective_python_path = sys.executable

    existing_entry = _read_openclaw_config_value(openclaw_bin, "plugins.entries.headroom")
    entry = _build_openclaw_plugin_entry(
        existing_entry=existing_entry,
        proxy_port=proxy_port,
        startup_timeout_ms=startup_timeout_ms,
        python_path=effective_python_path,
        no_auto_start=no_auto_start,
        gateway_provider_ids=gateway_provider_ids,
        enabled=True,
    )

    install_cmd = [
        openclaw_bin,
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
    ]
    if local_source_mode:
        if copy:
            install_cmd.append(str(plugin_dir))
            install_cwd = None
        else:
            install_cmd.extend(["--link", "."])
            install_cwd = plugin_dir
    else:
        install_cmd.append(plugin_spec)
        install_cwd = None

    click.echo("  Installing OpenClaw plugin with required unsafe-install flag...")
    install_result = run(
        install_cmd,
        cwd=str(install_cwd) if install_cwd else None,
        capture_output=True,
        text=True,
    )
    if install_result.returncode != 0:
        combined_error = "\n".join(
            x for x in [install_result.stderr.strip(), install_result.stdout.strip()] if x
        )
        plugin_already_exists = "plugin already exists" in combined_error.lower()
        linked_install_bug = (
            "also not a valid hook pack" in combined_error.lower()
            and "--dangerously-force-unsafe-install" in " ".join(install_cmd)
        )
        if plugin_already_exists:
            click.echo("  Plugin already installed; continuing with configuration/update steps.")
        elif linked_install_bug and local_source_mode and plugin_dir is not None:
            click.echo(
                "  OpenClaw linked-path install bug detected; applying extension-path fallback..."
            )
            target_dir = _copy_openclaw_plugin_into_extensions(
                plugin_dir=plugin_dir,
                openclaw_bin=openclaw_bin,
            )
            click.echo(f"  Fallback plugin copy completed: {target_dir}")
        else:
            details = combined_error or f"exit code {install_result.returncode}"
            raise click.ClickException(f"openclaw plugins install failed: {details}")
    elif verbose and install_result.stdout.strip():
        click.echo(install_result.stdout.strip())

    # Write the managed plugin entry only after a successful (or recoverable)
    # install, so a hard install failure leaves no stale
    # plugins.entries.headroom config behind.
    click.echo("  Writing plugin configuration...")
    _write_openclaw_plugin_entry(openclaw_bin, entry)

    _set_openclaw_context_engine_slot(openclaw_bin, "headroom")
    _run_checked(
        [openclaw_bin, "config", "validate"],
        action="openclaw config validate",
    )

    if no_restart:
        click.echo("  Skipping gateway restart (--no-restart).")
        click.echo(
            "  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply plugin changes."
        )
    else:
        click.echo("  Applying plugin changes to OpenClaw gateway...")
        gateway_action, gateway_output = _restart_or_start_openclaw_gateway(openclaw_bin)
        click.echo(f"  Gateway {gateway_action}.")
        if verbose and gateway_output:
            click.echo(gateway_output)

    inspect_result = _run_checked(
        [openclaw_bin, "plugins", "inspect", "headroom"],
        action="openclaw plugins inspect headroom",
    )
    if verbose and inspect_result.stdout.strip():
        click.echo(inspect_result.stdout.strip())

    click.echo()
    click.echo("✓ OpenClaw is configured to use Headroom context compression.")
    click.echo("  Plugin: headroom")
    click.echo("  Slot:   plugins.slots.contextEngine = headroom")
    click.echo()


# =============================================================================
# OpenCode
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@_serena_instructions_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-mcp", is_flag=True, help="Skip headroom MCP server registration")
@click.option("--no-serena", is_flag=True, help="Skip Serena MCP server registration")
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--copilot-subscription",
    is_flag=True,
    help="Route headroom/* models through the authenticated GitHub Copilot subscription",
)
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--backend", default=None, help="API backend: 'anthropic', 'anyllm', 'litellm-vertex', etc."
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Bedrock/Vertex")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("opencode_args", nargs=-1, type=click.UNPROCESSED)
def opencode(
    port: int,
    no_mcp: bool,
    no_serena: bool,
    code_graph: bool,
    no_proxy: bool,
    copilot_subscription: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    opencode_args: tuple,
) -> None:
    """Launch OpenCode through Headroom proxy.

    \b
    Sets OPENCODE_CONFIG_CONTENT to route all OpenCode API calls through
    Headroom. Configures a headroom provider via @ai-sdk/openai-compatible.
    Also sets OPENAI_BASE_URL and ANTHROPIC_BASE_URL as fallbacks.

    \b
    Examples:
        headroom wrap opencode                         # Start proxy + opencode
        headroom wrap opencode -- "fix the bug"        # Pass prompt to opencode
        headroom wrap opencode --no-mcp                # Skip MCP retrieve tool registration
        headroom wrap opencode --no-serena             # Skip Serena MCP registration
        headroom wrap opencode --port 9999             # Custom proxy port
        headroom wrap opencode --backend anyllm --anyllm-provider groq
        headroom wrap opencode --copilot-subscription # Use a GitHub Copilot subscription
    """
    subscription_resolution = None
    if copilot_subscription:
        effective_backend = backend or os.environ.get("HEADROOM_BACKEND")
        if effective_backend not in (None, "", "anthropic"):
            raise click.ClickException(
                "--copilot-subscription cannot be combined with translated backends "
                "such as anyllm or litellm-*; use the anthropic backend."
            )
        if no_proxy:
            raise click.ClickException(
                "--copilot-subscription cannot be combined with --no-proxy because "
                "it requires a private seeded proxy."
            )
        if prepare_only:
            raise click.ClickException(
                "--copilot-subscription cannot be combined with --prepare-only because "
                "it requires a running private seeded proxy."
            )
        subscription_resolution = _require_copilot_subscription_resolution()

    # Verify the opencode binary exists BEFORE mutating any config. Otherwise a
    # missing binary leaves headroom MCP/Serena/memory entries in the user's
    # opencode config and an injected AGENTS.md, then errors with no cleanup --
    # the config-before-verify anti-pattern (#1614). Siblings (claude, codex,
    # goose, omp) already check first. `--prepare-only` intentionally writes
    # config without launching, so it is exempt.
    opencode_bin: str | None = None
    if not prepare_only:
        opencode_bin = shutil.which("opencode")
        if not opencode_bin:
            click.echo("Error: 'opencode' not found in PATH.")
            click.echo("Install OpenCode: https://opencode.ai")
            raise SystemExit(1)

    # Snapshot OpenCode config.json BEFORE any wrap-time mutation so
    # `headroom unwrap opencode` can restore the user's pre-wrap state.
    _opencode_config_file, _opencode_backup_file = opencode_config_paths()
    snapshot_opencode_config_if_unwrapped(_opencode_config_file, _opencode_backup_file)

    # Register headroom MCP server in OpenCode config so OpenCode can
    # call headroom_retrieve on compression markers from the proxy.
    if not no_mcp:
        from headroom.mcp_registry import OpencodeRegistrar

        _setup_headroom_mcp(OpencodeRegistrar(), port, verbose=verbose, force=True)
    elif verbose:
        click.echo("  Skipping MCP retrieve tool (--no-mcp)")

    if not no_serena:
        from headroom.mcp_registry import OpencodeRegistrar

        # Serena ships no "opencode" context (only agent/codex/claude-code/ide/…);
        # passing --context opencode crashes Serena on launch (#1549/#1572). Use
        # the generic "agent" context, which OpenCode is.
        _setup_serena_mcp(OpencodeRegistrar(), context="agent", verbose=verbose, force=True)
    else:
        from headroom.mcp_registry import OpencodeRegistrar

        _disable_serena_mcp(OpencodeRegistrar(), verbose=verbose)

    # Setup memory MCP server for OpenCode (native tool integration)
    if memory:
        click.echo("  Setting up memory for OpenCode...")
        mem_dir = Path.cwd() / ".headroom"
        mem_dir.mkdir(parents=True, exist_ok=True)
        mem_user = os.environ.get("USER", os.environ.get("USERNAME", "default"))
        _inject_memory_mcp_config(mem_user)
        agents_md = Path.cwd() / "AGENTS.md"
        _inject_memory_agents_md(agents_md)

    if prepare_only:
        inject_opencode_provider_config(port)
        return

    # Past the prepare-only return the launch path always ran the binary check
    # above, so opencode_bin is resolved.
    assert opencode_bin is not None

    # Register our proxy client marker BEFORE _ensure_proxy so that another
    # wrapper's cleanup sees us as an active client and doesn't terminate a
    # shared proxy during the startup gap.
    _register_proxy_client(port)

    # Resolve port before config injection so the provider block and MCP
    # URL both point at the port the proxy will actually be on.
    _opencode_proxy, actual_port = _ensure_proxy(
        port,
        no_proxy,
        learn=learn,
        memory=memory,
        agent_type="opencode",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        openai_api_url=(subscription_resolution.api_url if subscription_resolution else None),
        copilot_api_token=(subscription_resolution.token if subscription_resolution else None),
        copilot_refresh_oauth_token=(
            subscription_resolution.refresh_oauth_token if subscription_resolution else None
        ),
        copilot_api_token_expires_at=(
            subscription_resolution.api_token_expires_at if subscription_resolution else None
        ),
    )

    try:
        # If the proxy fell back to a different port, move our marker so
        # cleanup tracking stays accurate and update MCP config.
        if actual_port != port:
            _unregister_proxy_client(port)
            _register_proxy_client(actual_port)
            if not no_mcp:
                from headroom.mcp_registry import OpencodeRegistrar

                _setup_headroom_mcp(OpencodeRegistrar(), actual_port, verbose=verbose, force=True)

        launch_environ = os.environ.copy()
        if subscription_resolution is not None:
            _scrub_copilot_subscription_launch_env(launch_environ)
        env, env_vars_display = _build_opencode_launch_env(
            actual_port, launch_environ, project=_project_name_from_cwd(), include_mcp=not no_mcp
        )

        # Inject Headroom provider into OpenCode config so traffic routes through proxy.
        inject_opencode_provider_config(actual_port)
        if memory:
            mem_dir = Path.cwd() / ".headroom"
            _inject_memory_mcp_config(
                os.environ.get("USER", os.environ.get("USERNAME", "default")),
            )

        # Proxy already started by _ensure_proxy above; tell _launch_tool to
        # skip duplicate startup.
        _launch_tool(
            binary=opencode_bin,
            args=opencode_args,
            env=env,
            port=actual_port,
            no_proxy=True,
            tool_label="OPENCODE",
            env_vars_display=env_vars_display,
            learn=learn,
            memory=memory,
            agent_type="opencode",
            code_graph=code_graph,
            backend=backend,
            anyllm_provider=anyllm_provider,
            region=region,
        )
    finally:
        if _opencode_proxy and _opencode_proxy.poll() is None:
            _other = _live_proxy_clients(actual_port, exclude_self=True)
            if not _other:
                _opencode_proxy.terminate()
                try:
                    _opencode_proxy.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _opencode_proxy.kill()


def _opencode_home_dir() -> Path:
    """Return the OpenCode home/config directory."""
    env_path = os.environ.get("OPENCODE_HOME", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".config" / "opencode"


# =============================================================================
# OpenCode (unwrap)
# =============================================================================


@unwrap.command("opencode")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_opencode(port: int, no_stop_proxy: bool) -> None:
    """Undo ``headroom wrap opencode`` edits to the active OpenCode config file.

    Behaviour:

    * If a pre-wrap backup (``opencode.json.headroom-backup``) exists, the
      original file is restored byte-for-byte and the backup is removed.
    * Otherwise, if the config file still contains the Headroom-managed
      block, that block is stripped out and the rest of the file is
      preserved.
    * If the config only ever contained Headroom-written content, the file
      is removed entirely so OpenCode falls back to its defaults.
    * If neither a backup nor a Headroom block is present, this is a safe
      no-op.
    """
    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║         HEADROOM UNWRAP: OPENCODE             ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()

    config_file, backup_file = opencode_config_paths()

    if backup_file.exists():
        try:
            shutil.copy2(backup_file, config_file)
            backup_file.unlink()
            click.echo(f"  Restored prior {config_file} from pre-wrap backup.")
            status = "restored"
        except OSError as exc:
            raise click.ClickException(
                f"could not restore OpenCode config from backup: {exc}"
            ) from exc
    elif config_file.exists():
        content = _read_text(config_file)
        if _PROVIDER_MARKER_START in content or _MCP_MARKER_START in content:
            cleaned = strip_opencode_headroom_blocks(content)
            if cleaned.strip():
                _write_text(config_file, cleaned + "\n")
                click.echo(f"  Removed Headroom block from {config_file}; other content preserved.")
                status = "cleaned"
            else:
                config_file.unlink()
                click.echo(f"  Removed {config_file} (contained only Headroom-written config).")
                status = "removed"
        else:
            click.echo(f"  Nothing to undo: {config_file} has no Headroom wrap markers.")
            status = "noop"
    else:
        click.echo(f"  Nothing to undo: {config_file} does not exist.")
        status = "noop"

    # Remove Serena MCP if it was installed by Headroom.
    # Also remove the headroom MCP server itself.
    from headroom.mcp_registry import OpencodeRegistrar

    opencode_registrar = OpencodeRegistrar()
    if opencode_registrar.detect():
        if opencode_registrar.unregister_server("headroom"):
            click.echo("  Removed Headroom MCP server from OpenCode.")
        serena_status = _remove_headroom_installed_serena_mcp(opencode_registrar)
        if serena_status == "removed":
            click.echo("  Removed Headroom-installed Serena MCP server from OpenCode.")
        elif serena_status == "failed":
            click.echo("  Serena MCP server matched Headroom ledger but could not be removed.")

    click.echo()
    click.echo("✓ OpenCode is no longer routed through the Headroom proxy.")
    if not no_stop_proxy and status != "noop":
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()


@unwrap.command("openclaw")
@click.option(
    "--proxy-port", default=8787, type=click.IntRange(1, 65535), help="Headroom proxy port"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
@click.option("--no-restart", is_flag=True, help="Do not restart OpenClaw gateway at the end")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.option("--existing-entry-json", default=None, hidden=True)
def unwrap_openclaw(
    proxy_port: int,
    no_stop_proxy: bool,
    no_restart: bool,
    verbose: bool,
    prepare_only: bool,
    existing_entry_json: str | None,
) -> None:
    """Disable the Headroom OpenClaw plugin and restore the legacy engine slot."""
    if prepare_only:
        click.echo(
            json.dumps(
                _build_openclaw_unwrap_entry(_decode_openclaw_entry_json(existing_entry_json)),
                separators=(",", ":"),
            )
        )
        return

    openclaw_bin = shutil.which("openclaw")
    if not openclaw_bin:
        raise click.ClickException("'openclaw' not found in PATH. Install OpenClaw CLI first.")

    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║          HEADROOM UNWRAP: OPENCLAW            ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()
    click.echo("  Disabling Headroom plugin and removing engine mapping...")

    existing_entry = _read_openclaw_config_value(openclaw_bin, "plugins.entries.headroom")
    entry = _build_openclaw_unwrap_entry(existing_entry)
    _write_openclaw_plugin_entry(openclaw_bin, entry)
    _set_openclaw_context_engine_slot(openclaw_bin, "legacy")
    _run_checked(
        [openclaw_bin, "config", "validate"],
        action="openclaw config validate",
    )

    if no_restart:
        click.echo("  Skipping gateway restart (--no-restart).")
        click.echo(
            "  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply unwrap changes."
        )
    else:
        click.echo("  Applying unwrap changes to OpenClaw gateway...")
        gateway_action, gateway_output = _restart_or_start_openclaw_gateway(openclaw_bin)
        click.echo(f"  Gateway {gateway_action}.")
        if verbose and gateway_output:
            click.echo(gateway_output)

    if verbose:
        inspect_result = _run_checked(
            [openclaw_bin, "plugins", "inspect", "headroom"],
            action="openclaw plugins inspect headroom",
        )
        if inspect_result.stdout.strip():
            click.echo(inspect_result.stdout.strip())

    click.echo()
    click.echo("✓ OpenClaw Headroom wrap removed.")
    click.echo("  Plugin: headroom (installed, disabled)")
    click.echo("  Slot:   plugins.slots.contextEngine = legacy")
    if not no_stop_proxy:
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(proxy_port), proxy_port)
    click.echo()


# =============================================================================
# OpenAI Codex CLI (unwrap)
# =============================================================================


@unwrap.command("grok-build")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_grok_build(port: int, no_stop_proxy: bool) -> None:
    """Undo ``headroom wrap grok-build`` edits to the active Grok config file."""
    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║         HEADROOM UNWRAP: GROK BUILD          ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()

    try:
        status, config_file = restore_grok_provider_config()
    except Exception as e:  # pragma: no cover - filesystem-level errors
        raise click.ClickException(f"could not unwrap Grok Build config: {e}") from e

    if status == "restored":
        click.echo(f"  Restored prior {config_file} from pre-wrap backup.")
    elif status == "cleaned":
        click.echo(f"  Removed Headroom block from {config_file}; other content preserved.")
    elif status == "removed":
        click.echo(f"  Removed {config_file} (contained only Headroom-written config).")
    else:
        if not os.environ.get("GROK_HOME"):
            click.echo(
                "  Warning: found no Headroom wrap markers in the default Grok config. "
                "If you wrapped Grok Build with GROK_HOME, rerun unwrap with the same "
                "environment variable, e.g. GROK_HOME=/path/to/grok-home "
                "headroom unwrap grok-build."
            )
        click.echo(f"  Nothing to undo: {config_file} has no Headroom wrap markers.")

    click.echo()
    click.echo("✓ Grok Build is no longer routed through the Headroom proxy.")
    if not no_stop_proxy and status != "noop":
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()


@unwrap.command("codex")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_codex(port: int, no_stop_proxy: bool) -> None:
    """Undo ``headroom wrap codex`` edits to the active Codex config file.

    Behaviour:

    * If a pre-wrap backup (``config.toml.headroom-backup``) exists, the
      original file is restored byte-for-byte and the backup is removed.
    * Otherwise, if the config file still contains the Headroom-managed
      block, that block is stripped out and the rest of the file is
      preserved.
    * If the config only ever contained Headroom-written content, the file
      is removed entirely so Codex falls back to its defaults.
    * If neither a backup nor a Headroom block is present, this is a safe
      no-op (the user either never wrapped that config, or already unwrapped
      it). When ``CODEX_HOME`` is unset, print a warning hint because Headroom
      may be looking at the default config while Codex was wrapped with a
      custom home.
    """
    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║           HEADROOM UNWRAP: CODEX              ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()

    try:
        status, config_file = _restore_codex_provider_config()
    except Exception as e:  # pragma: no cover - filesystem-level errors
        raise click.ClickException(f"could not unwrap Codex config: {e}") from e

    if status == "restored":
        click.echo(f"  Restored prior {config_file} from pre-wrap backup.")
    elif status == "cleaned":
        click.echo(f"  Removed Headroom block from {config_file}; other content preserved.")
    elif status == "removed":
        click.echo(f"  Removed {config_file} (contained only Headroom-written config).")
    else:
        if not os.environ.get("CODEX_HOME"):
            click.echo(
                "  Warning: found no Headroom wrap markers in the default Codex config. "
                "If you wrapped Codex with CODEX_HOME, rerun unwrap with the same "
                "environment variable, e.g. CODEX_HOME=/path/to/codex-home "
                "headroom unwrap codex."
            )
        click.echo(f"  Nothing to undo: {config_file} has no Headroom wrap markers.")

    # tokensave and Serena are each written as their own [mcp_servers.<name>]
    # table with Headroom markers, separate from the provider block handled
    # above — a "cleaned" restore leaves them behind. Remove them explicitly
    # (only if we installed them), mirroring unwrap_claude. Runs after the
    # restore so a backup-restore that already dropped them is a safe no-op.
    from headroom.mcp_registry import CodexRegistrar

    codex_registrar = CodexRegistrar()
    if codex_registrar.detect():
        tokensave_status = _remove_headroom_installed_tokensave_mcp(codex_registrar)
        if tokensave_status == "removed":
            click.echo("  Removed Headroom-installed tokensave MCP server from Codex.")
        elif tokensave_status == "failed":
            click.echo("  tokensave MCP server matched Headroom ledger but could not be removed.")

        serena_status = _remove_headroom_installed_serena_mcp(codex_registrar)
        if serena_status == "removed":
            click.echo("  Removed Headroom-installed Serena MCP server from Codex.")
        elif serena_status == "failed":
            click.echo("  Serena MCP server matched Headroom ledger but could not be removed.")

    if status in {"restored", "cleaned", "removed"}:
        # Hand the threads back to the native-provider menu so the full history
        # stays visible once Codex no longer routes through Headroom. Best-effort.
        retag_to_native(_codex_home_dir())

    click.echo()
    click.echo("✓ Codex is no longer routed through the Headroom proxy.")
    if not no_stop_proxy and status != "noop":
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()


# =============================================================================
# Oh My Pi (omp)
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@_retired_context_tool_option
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("omp_args", nargs=-1, type=click.UNPROCESSED)
def omp(
    port: int,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
    omp_args: tuple,
) -> None:
    """Launch Oh My Pi (omp) through Headroom proxy.

    \b
    Points omp's built-in `anthropic` provider at Headroom by injecting a
    marker-fenced `providers.anthropic.baseUrl` override into
    ~/.omp/agent/models.yml (pre-wrap file backed up byte-for-byte; undo with
    `headroom unwrap omp`). omp resolves its Anthropic chat endpoint from
    models.yml — ANTHROPIC_BASE_URL only affects its web-search helper — and a
    same-ID override keeps omp's bundled model catalog and stored credentials.
    omp's other providers (OpenAI-direct, Gemini, ...) keep their normal
    endpoints; route those via your own custom provider in models.yml.

    \b
    Examples:
        headroom wrap omp                       # Start proxy + omp
        headroom wrap omp -- -p "fix the bug"   # omp in non-interactive print mode
        headroom wrap omp -- --model opus       # Pick a model (fuzzy match)
        headroom unwrap omp                     # Restore pre-wrap models.yml
    """
    if prepare_only:
        _inject_omp_models_override(port, _project_name_from_cwd())
        return

    omp_bin = shutil.which("omp")
    if not omp_bin:
        click.echo("Error: 'omp' not found in PATH.")
        click.echo("Install Oh My Pi: npm install -g @oh-my-pi/pi-coding-agent")
        raise SystemExit(1)

    env, env_vars_display = _build_omp_launch_env(
        port, os.environ, project=_project_name_from_cwd()
    )

    # Durable endpoint redirect (survives omp-spawned child sessions, which
    # re-read models.yml rather than inheriting a parent env) — same durable
    # wrap + backup + unwrap contract as the Codex config.toml injection.
    models_file, _ = _inject_omp_models_override(port, _project_name_from_cwd())
    click.echo(f"  models.yml override written: {models_file}")

    _launch_tool(
        binary=omp_bin,
        args=omp_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="OMP",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="omp",
        code_graph=code_graph,
    )


@unwrap.command("omp")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_omp(port: int, no_stop_proxy: bool) -> None:
    """Undo ``headroom wrap omp`` edits to omp's models.yml.

    Restores the byte-for-byte pre-wrap backup when one exists, or removes the
    wrap-created file when there was no models.yml before wrapping. A
    models.yml the wrap does not manage is never touched.
    """
    status = _restore_omp_models_override()
    if status == "restored":
        click.echo(f"  Restored pre-wrap models.yml from backup: {_omp_models_yml_path()}")
    elif status == "removed":
        click.echo(f"  Removed wrap-created models.yml: {_omp_models_yml_path()}")
    else:
        click.echo("  No Headroom-managed models.yml found — nothing to restore.")

    click.echo()
    click.echo("✓ omp is no longer routed through the Headroom proxy.")
    if not no_stop_proxy and status != "noop":
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()


# =============================================================================
# Grok CLI (unwrap)
# =============================================================================


@unwrap.command("grok")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_grok(port: int, no_stop_proxy: bool) -> None:
    """Undo durable ``headroom wrap grok`` MCP and guidance edits.

    Grok inference routing is session-scoped via ``GROK_MODELS_BASE_URL`` and
    does not require config restoration. Native settings and authentication
    routing stay unchanged. This command removes Headroom MCP servers from
    ``~/.grok/config.toml``.
    """
    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║            HEADROOM UNWRAP: GROK              ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()

    from headroom.mcp_registry import GrokRegistrar

    grok_registrar = GrokRegistrar()
    removed_any = False
    if grok_registrar.detect():
        tokensave_status = _remove_headroom_installed_tokensave_mcp(grok_registrar)
        if tokensave_status == "removed":
            click.echo("  Removed Headroom-installed tokensave MCP server from Grok.")
            removed_any = True
        elif tokensave_status == "failed":
            click.echo("  tokensave MCP server matched Headroom ledger but could not be removed.")

        serena_status = _remove_headroom_installed_serena_mcp(grok_registrar)
        if serena_status == "removed":
            click.echo("  Removed Headroom-installed Serena MCP server from Grok.")
            removed_any = True
        elif serena_status == "failed":
            click.echo("  Serena MCP server matched Headroom ledger but could not be removed.")

        if grok_registrar.unregister_server("headroom"):
            click.echo("  Removed Headroom MCP server from Grok config.")
            removed_any = True

    if not removed_any:
        click.echo("  Nothing to undo: no Headroom MCP markers found.")

    click.echo()
    click.echo("✓ Grok is no longer configured for Headroom MCP retrieval.")
    click.echo("  Start Grok without `headroom wrap grok` so API traffic skips the proxy.")
    if not no_stop_proxy and removed_any:
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()


# =============================================================================
# ZCode (unwrap)
# =============================================================================


@unwrap.command("zcode")
@click.option(
    "--port", "-p", default=8787, type=click.IntRange(1, 65535), help="Proxy port (default: 8787)"
)
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_zcode(port: int, no_stop_proxy: bool) -> None:
    """Stop routing ZCode through Headroom.

    ``wrap zcode`` only prints settings for ZCode's UI (it writes no ZCode
    config), so there is nothing to restore — this stops the local proxy.
    """
    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║           HEADROOM UNWRAP: ZCODE             ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()

    click.echo("  Remove the Headroom base URLs from ZCode's provider settings.")
    click.echo()
    click.echo("✓ ZCode is no longer routed through the Headroom proxy.")
    if not no_stop_proxy:
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()
