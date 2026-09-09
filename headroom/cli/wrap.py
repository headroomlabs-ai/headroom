"""Wrap Claude Code to run through the Headroom proxy.

Usage:
    headroom wrap claude                    # Start proxy + claude
    headroom wrap vscode-claude             # Transparently proxy VS Code Claude Code
    headroom wrap claude --port 9999        # Custom proxy port
    headroom wrap claude -- --model opus    # Pass args to claude
    headroom unwrap claude                  # Restore settings
"""

from __future__ import annotations

import errno
import importlib.util
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable
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

from headroom import fsutil
from headroom._version import __version__ as _HEADROOM_VERSION
from headroom._version import normalize_release_version as _normalize_release_version
from headroom.agent_savings import (
    apply_agent_savings_env_defaults,
)
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

from .main import main


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


_AGENT_SAVINGS_TARGET_AGENTS = {"claude"}
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
_AGENT_SAVINGS_WRAP_AGENTS = {"claude"}

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

    timeout_seconds = _resolve_wrap_proxy_timeout_seconds()
    log_path = _get_log_path()
    stdio_log_path = _get_proxy_stdio_log_path()
    stdio_log_file = open(stdio_log_path, "a", encoding="utf-8")  # noqa: SIM115

    # Ensure proxy subprocess uses UTF-8 (Windows defaults to cp1252)
    proxy_env = os.environ.copy()
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


# Codex config injection markers


# File name used for the pre-wrap snapshot of the Codex config file.  The
# snapshot lets `headroom unwrap codex` restore the exact prior state, even
# if the user had their own `model_provider` / `[model_providers.*]` config
# before running wrap.


# Top-level bare keys we redirect to headroom values when the user already
# has them set.  Match the entire line (including any trailing comment) so
# we can rewrite it cleanly.  Bare keys must precede any [section] in TOML,
# so a `^` anchor combined with `^[ \t]*key` is sufficient — table lines
# start with `[`, not with the key name.


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


# Header carrying a preserved custom upstream (freemodel.dev, LiteLLM, vLLM,
# ...) so the proxy forwards to it instead of the hardcoded OpenAI default.
# Codex's env_http_headers only accepts an env-var *name* per header (not a
# literal value), so the detected URL is exported into this env var by the
# `wrap codex` launch path — see its use in `codex()` below.


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
) -> None:
    """Shared scaffolding for proxy-only wrap subcommands (no child binary launch).

    Used by ``wrap vscode-claude``: start the proxy, print the setup
    instructions, then block until Ctrl+C.
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


def _query_proxy_config(port: int) -> dict[str, Any] | None:
    """Query the running proxy's feature configuration via /health.

    Returns a dict with keys like backend, optimize, cache, rate_limit,
    memory, learn, code_graph, pid.  Returns None if unreachable or the
    response lacks a config block.
    """
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None

    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    return config


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
) -> tuple[subprocess.Popen | None, int]:
    """Start or verify proxy. Returns (process_handle, actual_port).

    The public ``_ensure_proxy`` wrapper serializes callers per port before
    entering this function. Keeping the implementation separate makes the
    lock boundary explicit and ensures every health/configuration check runs
    under the same startup critical section.
    """
    helpers = _live_wrap_module()
    # --no-proxy reuses an already-running proxy, so backend/region/provider
    # flags (which only apply when we start one) would be silently dropped.
    if no_proxy and (backend or anyllm_provider or region):
        click.echo(
            "  Warning: --backend/--region/--anyllm-provider have no effect with --no-proxy "
            "(reusing the existing proxy)."
        )
    if not no_proxy:
        manifest = helpers._find_persistent_manifest(port)
        if manifest is not None:
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

        if helpers._check_proxy(port):
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
        port_search_start = port
        try:
            actual_port = helpers._find_available_port(port_search_start)
        except OSError as e:
            raise click.ClickException(f"Port {port} is unavailable: {e}") from e
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e

        if actual_port != port:
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


@main.group()
@click.pass_context
def wrap(ctx: click.Context) -> None:
    """Wrap Claude Code to run through Headroom.

    \b
    Starts a Headroom proxy, configures the environment, and launches
    Claude Code so all API calls route through Headroom automatically.

    \b
    Supported tools:
        headroom wrap claude              # Claude Code CLI
        headroom wrap vscode-claude       # VS Code Claude Code extension

    \b
    `wrap` vs `proxy`:
        - `headroom wrap claude` — convenience: starts the proxy for you,
          sets the right env vars, and launches Claude Code.
        - `headroom proxy` — just the proxy. Point Claude Code at it by
          setting ANTHROPIC_BASE_URL yourself.
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

        # This build routes Claude Code only to the direct Anthropic API
        # (Vertex / Azure Foundry / Bedrock gateway modes were removed).
        if os.environ.get("CLAUDE_CODE_USE_VERTEX") or os.environ.get("CLAUDE_CODE_USE_FOUNDRY"):
            raise click.ClickException(
                "CLAUDE_CODE_USE_VERTEX / CLAUDE_CODE_USE_FOUNDRY are not supported by this "
                "build of Headroom; unset them to route through the direct Anthropic API."
            )

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
        env["ANTHROPIC_BASE_URL"] = proxy_url

        # Issue #951: write to settings.json so daemon-spawned conversation
        # workers (which read settings.json fresh rather than inheriting the
        # daemon's environment) also route through Headroom.
        _settings_vertex[0] = False
        _settings_foundry[0] = False
        # _wrap_settings_path is bound before the try (above) so the finally is
        # always safe; the value is unchanged here.
        _check_and_clear_stale_wrap_marker(
            _wrap_settings_path,
            key=_claude_wrap_base_url_env_key(),
        )
        _saved_base_url[0] = _write_claude_wrap_base_url(
            proxy_url,
            foundry_mode=False,
            vertex_mode=False,
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


# =============================================================================
# GitHub Copilot CLI (unwrap)
# =============================================================================


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


# =============================================================================
# OpenAI Codex CLI
# =============================================================================


# =============================================================================
# Aider
# =============================================================================


# =============================================================================
# OpenClaude
# =============================================================================


# =============================================================================
# Mistral Vibe
# =============================================================================


# =============================================================================
# Kimi CLI
# =============================================================================


# =============================================================================
# Grok CLI
# =============================================================================


# =============================================================================
# Cursor
# =============================================================================


# =============================================================================
# Grok Build
# =============================================================================


# =============================================================================
# Cline (VS Code extension)
# =============================================================================


# =============================================================================
# ZCode (zcode.z.ai desktop app)
# =============================================================================


# =============================================================================
# Continue (VS Code / JetBrains extension)
# =============================================================================


# =============================================================================
# Goose (Block)
# =============================================================================


# =============================================================================
# OpenHands
# =============================================================================


# =============================================================================
# OpenClaw
# =============================================================================


# =============================================================================
# OpenCode
# =============================================================================


# =============================================================================
# OpenCode (unwrap)
# =============================================================================


# =============================================================================
# OpenAI Codex CLI (unwrap)
# =============================================================================


# =============================================================================
# Oh My Pi (omp)
# =============================================================================


# =============================================================================
# Grok CLI (unwrap)
# =============================================================================


# =============================================================================
# ZCode (unwrap)
# =============================================================================


