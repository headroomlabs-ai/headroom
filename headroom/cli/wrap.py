"""Wrap CLI commands to run through Headroom proxy.

Usage:
    headroom wrap claude                    # Start proxy + context tool + claude
    headroom wrap copilot -- --model ...    # Start proxy + launch GitHub Copilot CLI
    headroom wrap codex                     # Start proxy + OpenAI Codex CLI
    headroom wrap aider                     # Start proxy + aider
    headroom wrap cursor                    # Start proxy + print Cursor config instructions
    headroom wrap openclaw                  # Install + configure OpenClaw plugin
    headroom wrap hermes                    # Start proxy + Hermes (Nous Research) agent
    headroom wrap claude --no-context-tool  # Without CLI context-tool setup
    headroom wrap claude --port 9999        # Custom proxy port
    headroom wrap claude -- --model opus    # Pass args to claude
"""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

# Fix Windows cp1252 encoding — box-drawing characters require UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import click

from headroom._version import __version__ as _HEADROOM_VERSION
from headroom.copilot_auth import DEFAULT_API_URL as COPILOT_API_URL
from headroom.copilot_auth import has_oauth_auth, resolve_client_bearer_token
from headroom.providers.aider import build_launch_env as _build_aider_launch_env
from headroom.providers.claude import proxy_base_url as _claude_proxy_base_url
from headroom.providers.codex import build_launch_env as _build_codex_launch_env
from headroom.providers.copilot import (
    build_launch_env as _build_copilot_launch_env,
)
from headroom.providers.copilot import (
    detect_running_proxy_backend as _copilot_detect_running_proxy_backend,
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
    validate_configuration as _validate_copilot_configuration,
)
from headroom.providers.cursor import render_setup_lines as _render_cursor_setup_lines
from headroom.providers.hermes import build_launch_env as _build_hermes_launch_env
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

from .main import main

_CONTEXT_TOOL_ENV = "HEADROOM_CONTEXT_TOOL"
_CONTEXT_TOOL_RTK = "rtk"
_CONTEXT_TOOL_LEAN_CTX = "lean-ctx"
_VALID_CONTEXT_TOOLS = {_CONTEXT_TOOL_RTK, _CONTEXT_TOOL_LEAN_CTX}


def _live_wrap_module() -> Any:
    """Return the current live wrap module instance."""
    return cast(Any, sys.modules[__name__])


def _selected_context_tool() -> str:
    """Return the configured CLI context tool.

    RTK remains the default for backward compatibility. Set
    ``HEADROOM_CONTEXT_TOOL=lean-ctx`` to let lean-ctx configure the supported
    coding agent instead.
    """

    raw = os.environ.get(_CONTEXT_TOOL_ENV, "").strip().lower().replace("_", "-")
    if not raw:
        return _CONTEXT_TOOL_RTK
    if raw == "leanctx":
        raw = _CONTEXT_TOOL_LEAN_CTX
    if raw not in _VALID_CONTEXT_TOOLS:
        raise click.ClickException(
            f"{_CONTEXT_TOOL_ENV} must be one of: {', '.join(sorted(_VALID_CONTEXT_TOOLS))}"
        )
    return raw


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


def _get_log_path() -> Path:
    """Get path for proxy log file."""
    from headroom import paths as _paths

    log_dir = _paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "proxy.log"


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
) -> subprocess.Popen:
    """Start Headroom proxy as a background subprocess.

    Logs are written to ~/.headroom/logs/proxy.log to avoid pipe buffer
    deadlocks (macOS pipe buffer is ~64KB — a busy proxy fills it quickly,
    blocking the process).
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

    log_path = _get_log_path()
    log_file = open(log_path, "a")  # noqa: SIM115

    # Ensure proxy subprocess uses UTF-8 (Windows defaults to cp1252)
    proxy_env = os.environ.copy()
    proxy_env["PYTHONIOENCODING"] = "utf-8"

    # Tell the proxy which agent is being wrapped (for traffic learning output)
    if agent_type != "unknown":
        proxy_env["HEADROOM_AGENT_TYPE"] = agent_type
        proxy_env.setdefault("HEADROOM_STACK", f"wrap_{agent_type}")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        env=proxy_env,
        start_new_session=os.name == "posix",
    )

    # Wait for proxy to be ready (up to 45 seconds).
    # ML components (Kompress, Magika, Tree-sitter) load synchronously before
    # uvicorn binds the port. On slower machines this can take 20-30 seconds.
    for _i in range(45):
        time.sleep(1)
        if _check_proxy(port):
            click.echo(f"  Logs: {log_path}")
            return proc
        # Check if process died
        if proc.poll() is not None:
            log_file.close()
            # Read last few lines of log for error context
            try:
                tail = log_path.read_text()[-500:]
            except Exception:
                tail = "(no log output)"
            raise RuntimeError(f"Proxy exited with code {proc.returncode}: {tail}")

    proc.kill()
    log_file.close()
    raise RuntimeError(f"Proxy failed to start on port {port} within 45 seconds")


def _setup_rtk(verbose: bool = False) -> Path | None:
    """Ensure rtk is installed and hooks are registered."""
    from headroom.rtk import get_rtk_path
    from headroom.rtk.installer import ensure_rtk, register_claude_hooks

    rtk_path = get_rtk_path()

    if rtk_path:
        if verbose:
            click.echo(f"  rtk found at {rtk_path}")
    else:
        click.echo("  Downloading rtk (Rust Token Killer)...")
        rtk_path = ensure_rtk()
        if rtk_path:
            click.echo(f"  rtk installed at {rtk_path}")
        else:
            click.echo("  rtk download failed — continuing without it")
            return None

    # Register hooks (idempotent)
    if register_claude_hooks(rtk_path):
        if verbose:
            click.echo("  rtk hooks registered in Claude Code")
    else:
        click.echo("  rtk hook registration failed — continuing without it")

    return rtk_path


def _setup_lean_ctx_agent(agent: str, verbose: bool = False) -> Path | None:
    """Run lean-ctx agent setup for the requested coding tool."""

    from headroom.lean_ctx import get_lean_ctx_path
    from headroom.lean_ctx.installer import ensure_lean_ctx

    lean_ctx = get_lean_ctx_path()
    if not lean_ctx:
        click.echo("  Downloading lean-ctx...")
        lean_ctx = ensure_lean_ctx()
    if not lean_ctx:
        click.echo("  lean-ctx download failed — continuing without it")
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="headroom-lean-ctx-") as setup_cwd:
            # lean-ctx writes project-local files when initialized from a git
            # checkout. Run from a non-project directory so setup is limited to
            # home-scoped agent config such as ~/.codex or ~/.claude.
            result = subprocess.run(
                [str(lean_ctx), "init", "--agent", agent],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=setup_cwd,
            )
    except Exception as e:
        click.echo(f"  lean-ctx setup failed — continuing without it: {e}")
        return None

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        click.echo(f"  lean-ctx setup failed — continuing without it{suffix}")
        return None

    if verbose:
        detail = result.stdout.strip()
        if detail:
            click.echo(f"  lean-ctx configured for {agent}: {detail}")
        else:
            click.echo(f"  lean-ctx configured for {agent}")
    return lean_ctx


def _remove_claude_rtk_hooks(settings_path: Path | None = None) -> bool:
    """Remove Headroom/rtk-managed Claude hook entries from settings.json.

    `rtk init --global --auto-patch` installs a Claude PreToolUse hook that
    points at an ``rtk-rewrite`` script. Unwrap should remove that hook without
    touching unrelated Claude settings or user-authored hooks.
    """

    path = settings_path or (Path.home() / ".claude" / "settings.json")
    if not path.exists():
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
                    isinstance(item, dict) and "rtk-rewrite" in str(item.get("command", "")).lower()
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

    if not changed:
        return False

    if hooks:
        payload["hooks"] = hooks
    else:
        payload.pop("hooks", None)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


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


def _setup_serena_mcp(
    registrar: Any, *, context: str, verbose: bool = False, force: bool = False
) -> None:
    """Register Serena MCP with the given agent (idempotent)."""
    from headroom.mcp_registry import build_serena_spec, format_result
    from headroom.mcp_registry.base import RegisterStatus
    from headroom.mcp_registry.ledger import record_install

    if not registrar.detect():
        if verbose:
            click.echo(f"  Serena MCP: {registrar.display_name} not detected — skipping")
        return

    if shutil.which("uvx") is None:
        click.echo("  Serena MCP: uvx not found — install uv/uvx to enable Serena; skipping")
        return

    spec = build_serena_spec(context)
    result = registrar.register_server(spec, force=force)
    if result.status == RegisterStatus.REGISTERED:
        record_install(registrar.name, spec)

    line = format_result(
        registrar.name,
        result,
        label="Serena MCP",
        verbose=verbose,
        overwrite_hint="update or remove the existing serena MCP entry, then rerun headroom wrap",
        restart_hint=f"restart {registrar.display_name} if it was already running",
    )
    if line is not None:
        click.echo(line)


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


_CBM_MCP_SERVER_NAME = "codebase-memory-mcp"


def _register_cbm_mcp_server(cbm_bin: str) -> None:
    """Register codebase-memory-mcp as an MCP server in Claude Code.

    Uses ``claude mcp add`` so the tools appear in ``/mcp`` automatically.
    Idempotent — skips if already registered.
    """
    claude_cli = shutil.which("claude")
    if not claude_cli:
        return

    # Check if already registered
    check = subprocess.run(
        [claude_cli, "mcp", "get", _CBM_MCP_SERVER_NAME],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return  # Already registered

    result = subprocess.run(
        [claude_cli, "mcp", "add", _CBM_MCP_SERVER_NAME, "-s", "user", "--", cbm_bin],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        click.echo(f"  Code graph: registered {_CBM_MCP_SERVER_NAME} MCP server")
    else:
        pass  # Non-critical — tools won't appear in /mcp but graph still works


def _setup_code_graph(verbose: bool = False) -> bool:
    """Ensure codebase-memory-mcp is installed, registered as MCP server, and project is indexed.

    codebase-memory-mcp builds a knowledge graph of the codebase using
    tree-sitter, enabling the LLM to query code structure (call chains,
    function definitions, impact analysis) instead of reading entire files.

    Steps:
    1. Download the binary if not already present.
    2. Register as an MCP server in Claude Code (``claude mcp add``).
    3. Index the current project (fast, idempotent).

    With Claude Code's MCP Tool Search, the 14 graph tools add ~200 tokens
    overhead per request (not the full ~1,915) — they're lazy-loaded.

    Returns True if graph is ready, False if setup failed.
    """
    from headroom.graph.installer import ensure_cbm, get_cbm_path

    cbm_path = get_cbm_path()
    if not cbm_path:
        click.echo("  Code graph: downloading codebase-memory-mcp...")
        cbm_path = ensure_cbm()
        if cbm_path:
            click.echo(f"  Code graph: installed at {cbm_path}")
        else:
            click.echo("  Code graph: download failed — skipping")
            return False

    cbm_bin = str(cbm_path)

    # Register as MCP server so tools appear in /mcp
    _register_cbm_mcp_server(cbm_bin)

    # Index current project (fast — ~1s for most repos, idempotent)
    project_dir = str(Path.cwd())
    try:
        result = subprocess.run(
            [
                cbm_bin,
                "cli",
                "index_repository",
                json.dumps({"repo_path": project_dir, "mode": "fast"}),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Parse node/edge counts from output
            for line in result.stdout.splitlines():
                if '"nodes"' in line:
                    try:
                        # Parse the JSON response to extract node/edge counts
                        import re

                        m_nodes = re.search(r'"nodes":(\d+)', line)
                        m_edges = re.search(r'"edges":(\d+)', line)
                        if m_nodes and m_edges:
                            nodes = int(m_nodes.group(1))
                            edges = int(m_edges.group(1))
                            click.echo(
                                f"  Code graph: indexed ({nodes:,} symbols, "
                                f"{edges:,} relationships)"
                            )
                        else:
                            click.echo("  Code graph: indexed")
                    except (ValueError, AttributeError):
                        click.echo("  Code graph: indexed")
                    return True
            click.echo("  Code graph: indexed")
            return True
        else:
            if verbose:
                click.echo(f"  Code graph: indexing failed ({result.stderr[:100]})")
            return False
    except subprocess.TimeoutExpired:
        click.echo("  Code graph: indexing timed out (will complete in background)")
        return False
    except Exception as e:
        if verbose:
            click.echo(f"  Code graph: setup failed ({e})")
        return False


# rtk instructions for tools without hook support (Codex, Cursor, Aider).
# These get injected into AGENTS.md / .cursorrules so the LLM voluntarily
# uses rtk-prefixed commands. Kept concise to minimize instruction overhead.
RTK_INSTRUCTIONS_BLOCK = """\
<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->
"""

# Marker used to detect if instructions are already injected
_RTK_MARKER = "<!-- headroom:rtk-instructions -->"

# Memory MCP markers
_MEMORY_MCP_MARKER = "# --- Headroom memory MCP (auto-injected) ---"
_MEMORY_MCP_END = "# --- end Headroom memory ---"
_MEMORY_AGENTS_MARKER = "<!-- headroom:memory-instructions -->"

# Codex config injection markers
_CODEX_TOP_LEVEL_MARKER = "# --- Headroom proxy (auto-injected by headroom wrap codex) ---"
_CODEX_END_MARKER = "# --- end Headroom ---"
_CODEX_MCP_MARKER = "# --- Headroom MCP server ---"
_CODEX_MCP_END = "# --- end Headroom MCP server ---"
# File name used for the pre-wrap snapshot of ~/.codex/config.toml.  The
# snapshot lets `headroom unwrap codex` restore the exact prior state, even
# if the user had their own `model_provider` / `[model_providers.*]` config
# before running wrap.
_CODEX_CONFIG_BACKUP_SUFFIX = ".headroom-backup"


def _codex_config_paths() -> tuple[Path, Path]:
    """Return ``(config_file, backup_file)`` paths for the Codex TOML config."""
    config_dir = Path.home() / ".codex"
    config_file = config_dir / "config.toml"
    backup_file = config_dir / f"config.toml{_CODEX_CONFIG_BACKUP_SUFFIX}"
    return config_file, backup_file


def _strip_codex_headroom_blocks(content: str, *, remove_mcp: bool = False) -> str:
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
    * If the config already contains a Headroom marker, a wrap run is already
      active: do not snapshot the injected state.
    """
    if backup_file.exists():
        return
    if not config_file.exists():
        return
    try:
        content = config_file.read_text()
    except OSError:
        return
    if _CODEX_TOP_LEVEL_MARKER in content or _CODEX_END_MARKER in content:
        return
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, backup_file)


def _ensure_rtk_binary(verbose: bool = False) -> Path | None:
    """Ensure rtk binary is installed (download if needed). No hook registration."""
    from headroom.rtk import get_rtk_path
    from headroom.rtk.installer import ensure_rtk

    rtk_path = get_rtk_path()

    if rtk_path:
        if verbose:
            click.echo(f"  rtk found at {rtk_path}")
        return rtk_path

    click.echo("  Downloading rtk (Rust Token Killer)...")
    rtk_path = ensure_rtk()
    if rtk_path:
        click.echo(f"  rtk installed at {rtk_path}")
        return rtk_path

    click.echo("  rtk download failed — continuing without it")
    return None


def _prepare_wrap_rtk(verbose: bool = False, *, label: str | None = None) -> Path | None:
    """Ensure rtk is present for host-bridged wrap flows without host-specific setup."""
    if label:
        click.echo(f"  Preparing rtk for {label}...")
    return _ensure_rtk_binary(verbose=verbose)


def _inject_codex_provider_config(port: int) -> None:
    """Inject a Headroom model provider into Codex's config.toml.

    Two keys are written in the top-level block:

    * ``model_provider = "headroom"`` — selects the custom provider for
      API-key mode traffic.
    * ``openai_base_url = "http://127.0.0.1:{port}/v1"`` — overrides the
      built-in ``openai`` provider's base URL.  This is the critical key for
      **subscription (ChatGPT plan) users**: Codex detects subscription auth
      and routes through the built-in ``openai`` provider regardless of
      ``model_provider``, so without this override it bypasses the proxy and
      hits ``https://chatgpt.com/backend-api/codex`` directly.

    Safe to call multiple times — the injected block is fully replaced on
    each call, so re-running with a different ``port`` updates the config.
    Before the first injection, the pre-wrap file is snapshotted to
    ``~/.codex/config.toml.headroom-backup`` so ``headroom unwrap codex``
    can restore it byte-for-byte.
    """
    config_file, backup_file = _codex_config_paths()
    config_dir = config_file.parent

    # The injected content is split into two self-contained, marker-delimited
    # blocks: a top-level key block (at the start of the file, because bare
    # TOML keys must precede any [section]) and a provider-table block (at
    # the end).  Each block has its own matching begin/end marker pair so
    # stripping them is unambiguous and never consumes user content that
    # happens to sit between the two.
    top_level_block = (
        f"{_CODEX_TOP_LEVEL_MARKER}\n"
        f'model_provider = "headroom"\n'
        f'openai_base_url = "http://127.0.0.1:{port}/v1"\n'
        f"{_CODEX_END_MARKER}\n"
    )
    provider_section = (
        f"{_CODEX_TOP_LEVEL_MARKER}\n"
        "[model_providers.headroom]\n"
        'name = "OpenAI via Headroom proxy"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        f"supports_websockets = true\n"
        f"{_CODEX_END_MARKER}\n"
    )

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the pre-wrap state before touching anything.  No-op if the
        # config is already wrapped, is missing, or we've already snapshotted.
        _snapshot_codex_config_if_unwrapped(config_file, backup_file)

        if config_file.exists():
            content = config_file.read_text()
            # Remove any prior Headroom-managed blocks before re-injecting so
            # the operation is idempotent and supports port changes.
            content = _strip_codex_headroom_blocks(content)

            # Place the top-level key block at the very beginning of the file
            # (bare TOML keys must precede any [section]) and the provider
            # table at the end.  User content, if any, sits between them.
            user_content = content.strip()
            if user_content:
                content = top_level_block + "\n" + user_content + "\n\n" + provider_section
            else:
                content = top_level_block + "\n" + provider_section
        else:
            content = top_level_block + "\n" + provider_section

        config_file.write_text(content)
        click.echo(f"  Codex config: injected Headroom provider (WS + HTTP) into {config_file}")
    except Exception as e:
        click.echo(f"  Warning: could not update Codex config: {e}")


def _restore_codex_provider_config() -> tuple[str, Path]:
    """Undo ``_inject_codex_provider_config`` for ``~/.codex/config.toml``.

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
        original = config_file.read_text()
        if _CODEX_TOP_LEVEL_MARKER in original or _CODEX_END_MARKER in original:
            cleaned = _strip_codex_headroom_blocks(original, remove_mcp=True)
            if not cleaned.strip():
                # Nothing left but Headroom content — remove the file entirely
                # so Codex falls back to its default config.
                config_file.unlink()
                return "removed", config_file
            config_file.write_text(cleaned)
            return "cleaned", config_file

    # Nothing to undo.
    return "noop", config_file


def _emit_wrap_interrupted(agent: str, marker_path: Path | None) -> None:
    """Log a clear interruption message after a partial wrap setup.

    Called when a wrap subcommand catches ``KeyboardInterrupt`` between marker
    injection and proxy startup. The marker file (if any) is left on disk —
    re-running the same ``headroom wrap <agent>`` command is idempotent and
    safe.
    """
    if marker_path is not None:
        click.echo(
            f"\n  Wrap was interrupted; marker file at {marker_path} is on "
            f"disk. Rerun `headroom wrap {agent}` to retry — it's idempotent."
        )
    else:
        click.echo(
            f"\n  Wrap was interrupted before any on-disk changes. Rerun "
            f"`headroom wrap {agent}` to retry — it's idempotent."
        )


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


def _setup_context_tool_for_agent(
    *,
    agent: str,
    agent_display: str,
    marker_path: Path | None,
    on_rtk_ready: Callable[[Path], object] | None = None,
    rtk_required: bool = False,
    verbose: bool = False,
) -> Path | None:
    """Run the rtk-or-lean-ctx context-tool setup with KeyboardInterrupt handling.

    Replaces the ``try / except KeyboardInterrupt / rtk-vs-lean-ctx fork``
    each wrap subcommand was inlining. Returns the rtk binary path if rtk
    mode was selected and the install succeeded; otherwise ``None``.

    Args:
        agent: Internal agent name (used by ``_setup_lean_ctx_agent`` and
            the interrupt message).
        agent_display: User-facing capitalized name for the echo lines
            (e.g. ``"Cline"``, ``"OpenHands"``).
        marker_path: Optional path to report on Ctrl+C interruption. Pass
            ``None`` for env-only agents that never touch disk (openhands).
        on_rtk_ready: Optional callback invoked with the rtk binary path
            when rtk install succeeds. Use to inject into a marker file
            (e.g. ``.clinerules``, ``.goosehints``, ``config.json``).
        rtk_required: If ``True`` and rtk install fails, raise
            ``SystemExit(1)`` instead of silently falling through. Use this
            for env-var-only agents (openhands) where there is no
            fallback marker file to write to.
        verbose: Forwarded to the rtk/lean-ctx installers.
    """
    try:
        if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
            click.echo(f"  Setting up lean-ctx for {agent_display}...")
            _setup_lean_ctx_agent(agent, verbose=verbose)
            return None
        click.echo(f"  Setting up rtk for {agent_display}...")
        rtk_path = _ensure_rtk_binary(verbose=verbose)
        if not rtk_path:
            if rtk_required:
                click.echo(
                    "  Error: rtk install failed; refusing to inject "
                    f"context-tool guidance for {agent_display} without rtk. "
                    "Install rtk manually and re-run, or pass --no-context-tool "
                    "to skip rtk."
                )
                raise SystemExit(1)
            return None
        if on_rtk_ready is not None:
            on_rtk_ready(rtk_path)
        return rtk_path
    except KeyboardInterrupt:
        _emit_wrap_interrupted(
            agent, marker_path if (marker_path and marker_path.exists()) else None
        )
        raise SystemExit(130) from None


def _run_proxy_only_watcher(
    *,
    agent_label: str,
    port: int,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    agent_type: str,
    print_setup_lines: Callable[[], None],
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
    cleanup = _make_cleanup(proxy_holder, port)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        _print_wrap_banner(agent_label)
        proxy_holder[0] = _ensure_proxy(
            port, no_proxy, learn=learn, memory=memory, agent_type=agent_type
        )
        click.echo()
        print_setup_lines()
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


def _inject_rtk_instructions(file_path: Path, verbose: bool = False) -> bool:
    """Inject rtk instructions into a file (AGENTS.md, .cursorrules, etc.).

    Idempotent — skips if marker already present. Appends to existing content.
    Returns True if instructions were written.
    """
    if file_path.exists():
        existing = file_path.read_text()
        if _RTK_MARKER in existing:
            if verbose:
                click.echo(f"  rtk instructions already in {file_path.name}")
            return True
        # Append to existing file
        with open(file_path, "a") as f:
            f.write("\n\n" + RTK_INSTRUCTIONS_BLOCK)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(RTK_INSTRUCTIONS_BLOCK)

    click.echo(f"  rtk instructions injected into {file_path}")
    return True


def _inject_memory_mcp_config(db_path: str, user_id: str) -> None:
    """Register headroom memory as an MCP server in Codex's config.toml.

    Idempotent — replaces existing section if present.
    """
    import sys

    config_dir = Path.home() / ".codex"
    config_file = config_dir / "config.toml"

    # Use forward slashes in TOML paths (works on all platforms, avoids
    # backslash escaping issues on Windows)
    python_bin = sys.executable.replace("\\", "/")
    db_path_toml = db_path.replace("\\", "/")
    mcp_section = (
        f"\n{_MEMORY_MCP_MARKER}\n"
        f"[mcp_servers.headroom_memory]\n"
        f'command = "{python_bin}"\n'
        f'args = ["-m", "headroom.memory.mcp_server", "--db", "{db_path_toml}", "--user", "{user_id}"]\n'
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
            content = config_file.read_text()
            if _MEMORY_MCP_MARKER in content:
                start = content.index(_MEMORY_MCP_MARKER)
                end = content.index(_MEMORY_MCP_END) + len(_MEMORY_MCP_END)
                content = content[:start].rstrip("\n") + mcp_section + content[end:].lstrip("\n")
            else:
                content = content.rstrip() + "\n" + mcp_section
        else:
            content = mcp_section

        config_file.write_text(content)
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
        existing = file_path.read_text()
        if _MEMORY_AGENTS_MARKER in existing:
            return True  # Already injected
        with open(file_path, "a") as f:
            f.write("\n\n" + memory_block)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(memory_block)

    click.echo(f"  Memory guidance injected into {file_path.name}")
    return True


def _apply_rtk_to_systemmessage_field(
    container: dict[str, Any],
    location_label: str,
    verbose: bool = False,
) -> tuple[bool, bool]:
    """Apply the RTK block to ``container["systemMessage"]`` in place.

    Returns ``(changed, ok)``:

    * ``changed`` is ``True`` if the field was written (or rewritten) on this
      call. ``False`` for idempotent skips and for refusals.
    * ``ok`` is ``True`` for "RTK guidance is now present (or refused-safely)",
      ``False`` only for refusals where the user must intervene. Callers
      surface ``not ok`` as a warning to the user.

    Refusal cases (loud, no silent overwrite):

    * ``systemMessage`` exists and is **not a string** (dict / list / number).
      We never clobber user data of an unknown shape. The user must remove or
      clear the field before re-running.
    """
    existing_msg = container.get("systemMessage")

    if isinstance(existing_msg, str) and _RTK_MARKER in existing_msg:
        if verbose:
            click.echo(f"  rtk instructions already in {location_label}")
        return False, True

    if existing_msg is None or (isinstance(existing_msg, str) and not existing_msg.strip()):
        container["systemMessage"] = RTK_INSTRUCTIONS_BLOCK
        return True, True

    if isinstance(existing_msg, str):
        container["systemMessage"] = existing_msg.rstrip() + "\n\n" + RTK_INSTRUCTIONS_BLOCK
        return True, True

    # Non-string, non-null value present — refuse loudly. We will not clobber
    # user data of unknown shape.
    click.echo(
        f"  Warning: {location_label} systemMessage is not a string "
        f"(type={type(existing_msg).__name__}); refusing to overwrite. "
        "To opt in, remove or clear the existing systemMessage value and re-run."
    )
    return False, False


def _inject_continue_rtk_systemmessage(config_file: Path, verbose: bool = False) -> bool:
    """Inject the rtk instructions block into Continue's ``.continue/config.json``.

    Continue's schema supports both a top-level ``systemMessage`` string and a
    per-model ``systemMessage`` on each entry in the ``models`` array. The
    per-model value, when set, overrides the top-level one — so users with
    per-model configs would otherwise silently get no RTK guidance. This
    helper writes the RTK block into **every** ``systemMessage`` site:

    * top-level ``systemMessage``
    * each ``models[i].systemMessage`` where ``models[i]`` is a dict

    The RTK marker (``<!-- headroom:rtk-instructions -->``) is the idempotency
    token: if a prior ``systemMessage`` already contains the marker we leave
    that site alone. If the existing value is a non-empty string we append
    with a separator. If the existing value is **non-string** (dict / list /
    number) we refuse loudly and leave it untouched — we do not clobber user
    data of unknown shape. To opt in to overwrite, the user must clear the
    existing value first.

    The config file is read/written as JSON. Malformed JSON is left untouched
    and the helper returns ``False``. Note: Continue's modern config is
    YAML-first; users on the YAML schema should configure systemMessage
    through that file instead — this helper only handles the JSON variant.

    Returns ``True`` if injection succeeded (or was already idempotent at
    every site); ``False`` if any site refused or the file was malformed.
    """
    if config_file.exists():
        try:
            content = config_file.read_text()
        except OSError as exc:
            click.echo(f"  Warning: could not read {config_file}: {exc}")
            return False
        if not content.strip():
            data: dict[str, Any] = {}
        else:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                click.echo(
                    f"  Warning: {config_file} is not valid JSON ({exc.msg}); "
                    "not modifying — fix the file manually before re-running."
                )
                return False
            if not isinstance(parsed, dict):
                click.echo(
                    f"  Warning: {config_file} top-level value is not an object; "
                    "Continue expects a JSON object — leaving file untouched."
                )
                return False
            data = parsed
    else:
        data = {}

    any_changed = False
    all_ok = True

    # 1. Top-level systemMessage.
    changed, ok = _apply_rtk_to_systemmessage_field(
        data, location_label=f"{config_file.name} (top-level)", verbose=verbose
    )
    any_changed = any_changed or changed
    all_ok = all_ok and ok

    # 2. Per-model systemMessage. Continue's models[] entry overrides the
    # top-level value when set, so we must visit each one.
    models = data.get("models")
    if isinstance(models, list):
        for idx, model in enumerate(models):
            if not isinstance(model, dict):
                continue
            label = f"{config_file.name} models[{idx}]"
            if isinstance(model.get("title"), str):
                label = f"{config_file.name} models[{idx}] ({model['title']})"
            changed_i, ok_i = _apply_rtk_to_systemmessage_field(
                model, location_label=label, verbose=verbose
            )
            any_changed = any_changed or changed_i
            all_ok = all_ok and ok_i

    if any_changed:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps(data, indent=2) + "\n")
        click.echo(f"  rtk instructions injected into {config_file}")
    elif all_ok and verbose:
        # Idempotent re-run with no refusals — nothing to do.
        click.echo(f"  rtk instructions already present in {config_file.name}")

    return all_ok


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


def _proxy_version(payload: dict[str, Any] | None) -> str | None:
    """Return the running proxy version when it exposes one."""
    if payload is None:
        return None
    version = payload.get("version")
    return version if isinstance(version, str) and version else None


def _proxy_needs_version_restart(payload: dict[str, Any] | None) -> bool:
    """Return True when a running Headroom proxy uses a different package version."""
    running_version = _proxy_version(payload)
    return (
        running_version is not None
        and running_version != "unknown"
        and _HEADROOM_VERSION != "unknown"
        and running_version != _HEADROOM_VERSION
    )


def _detect_running_proxy_backend(port: int) -> str | None:
    """Read the backend of an already-running proxy from its health endpoint."""
    return _copilot_detect_running_proxy_backend(port)


def _kill_proxy_by_pid(pid: int, port: int) -> bool:
    """Terminate a proxy process by PID and wait for the port to free up.

    Sends SIGTERM first, falls back to SIGKILL after 5 seconds.
    Returns True if the port is free afterwards, False otherwise.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Already gone
    except PermissionError:
        click.echo(f"  Warning: No permission to kill proxy PID {pid}")
        return False

    # Wait for port to free (up to 5 seconds)
    for _ in range(50):
        time.sleep(0.1)
        if not _check_proxy(port):
            return True

    # SIGTERM didn't work — escalate to SIGKILL (Unix) or terminate (Windows)
    try:
        _kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(pid, _kill_signal)
    except (ProcessLookupError, PermissionError):
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


def _should_use_copilot_oauth(
    *,
    backend: str | None,
    provider_type: str,
    env: dict[str, str],
) -> bool:
    """Prefer a reusable Copilot OAuth session when the requested routing supports it."""
    if env.get("COPILOT_PROVIDER_API_KEY") or env.get("COPILOT_PROVIDER_BEARER_TOKEN"):
        return False
    if provider_type == "anthropic":
        return False

    effective_backend = backend or os.environ.get("HEADROOM_BACKEND")
    if effective_backend not in (None, "", "anthropic"):
        return False

    return has_oauth_auth()


def _ensure_proxy(
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
) -> subprocess.Popen | None:
    """Start or verify proxy. Returns process handle if we started it."""
    helpers = _live_wrap_module()
    if not no_proxy:
        manifest = helpers._find_persistent_manifest(port)
        if manifest is not None:
            from headroom.install.health import probe_ready

            if probe_ready(manifest.health_url):
                health_payload = helpers._query_proxy_health(port)
                if helpers._proxy_needs_version_restart(health_payload):
                    running_version = helpers._proxy_version(health_payload) or "unknown"
                    active_sessions = helpers._proxy_active_session_count(health_payload)
                    if active_sessions > 0:
                        click.echo(
                            f"  Proxy on port {port} is running Headroom {running_version}; "
                            f"current CLI is {_HEADROOM_VERSION}."
                        )
                        click.echo(
                            f"  Leaving it running because {active_sessions} active session(s) "
                            "are still attached; it will be restarted when idle."
                        )
                        return None
                    if helpers._restart_persistent_proxy(manifest, port):
                        return None
                    raise click.ClickException(
                        f"Persistent deployment '{manifest.profile}' on port {port} "
                        f"is running stale Headroom {running_version} and could not be restarted."
                    )
                click.echo(f"  Proxy already running on port {port}")
                return None
            if helpers._recover_persistent_proxy(port):
                return None
            if helpers._check_proxy(port):
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
                if active_sessions > 0:
                    click.echo(
                        f"  Proxy on port {port} is running Headroom {running_version}; "
                        f"current CLI is {_HEADROOM_VERSION}."
                    )
                    click.echo(
                        f"  Leaving it running because {active_sessions} active session(s) "
                        "are still attached; it will be restarted when idle."
                    )
                    return None

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

                if missing:
                    needs_restart = True
                    flags_str = ", ".join(f"--{f.replace('_', '-')}" for f in missing)
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
                        return None

            if not needs_restart:
                click.echo(f"  Proxy already running on port {port}")
                return None

        # Start (or restart) the proxy with the requested flags
        click.echo(f"  Starting Headroom proxy on port {port}...")
        try:
            proc = cast(
                subprocess.Popen[Any],
                _live_wrap_module()._start_proxy(
                    port,
                    learn=learn,
                    memory=memory,
                    agent_type=agent_type,
                    code_graph=code_graph,
                    backend=backend,
                    anyllm_provider=anyllm_provider,
                    region=region,
                    openai_api_url=openai_api_url,
                ),
            )
            click.echo(f"  Proxy ready on http://127.0.0.1:{port}")
            return proc
        except RuntimeError as e:
            click.echo(f"  Error: {e}")
            raise SystemExit(1) from e
    else:
        if not helpers._check_proxy(port):
            click.echo(f"  Warning: No proxy detected on port {port}")
        return None


def _make_cleanup(proxy_proc_holder: list, port: int = 8787) -> Any:
    """Create a cleanup function that terminates the proxy on exit.

    Only kills the proxy if no other headroom-wrapped clients are using it.
    Checks by looking for other processes with ANTHROPIC_BASE_URL or
    OPENAI_BASE_URL pointing at our port.
    """

    def _other_clients_exist() -> bool:
        """Check if other processes are using this proxy."""
        try:
            # Count headroom wrap processes (excluding ourselves)
            result = subprocess.run(
                ["pgrep", "-f", f"127.0.0.1:{port}"],
                capture_output=True,
                text=True,
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            my_pid = str(os.getpid())
            other_pids = [p for p in pids if p != my_pid]
            return len(other_pids) > 0
        except Exception:
            return False  # If we can't check, assume no others

    def cleanup(signum: int | None = None, frame: Any = None) -> None:
        proc = proxy_proc_holder[0] if proxy_proc_holder else None
        if proc and proc.poll() is None:
            if _other_clients_exist():
                # Other clients still using the proxy — leave it running
                return
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    return cleanup


def _ignore_child_sigint(signum: int | None = None, frame: Any = None) -> None:
    """Keep the wrapper alive when Ctrl-C is intended for the child CLI."""

    return None


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
) -> None:
    """Common logic: start proxy, launch tool, clean up."""
    proxy_holder: list[subprocess.Popen | None] = [None]
    cleanup = _make_cleanup(proxy_holder, port)
    signal.signal(signal.SIGINT, _ignore_child_sigint)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        padded = f"HEADROOM WRAP: {tool_label}".center(47)
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo(f"  ║{padded}║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(
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
        )

        if code_graph:
            _setup_code_graph(verbose=False)

        click.echo()
        click.echo(f"  Launching {tool_label} (API routed through Headroom)...")
        for var in env_vars_display:
            click.echo(f"  {var}")
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
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
    result = subprocess.run(
        [openclaw_bin, "config", "get", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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
    restart_result = subprocess.run(
        [openclaw_bin, "gateway", "restart"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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
def wrap() -> None:
    """Wrap CLI tools to run through Headroom.

    \b
    Starts a Headroom proxy, configures the environment, and launches
    the target tool so all API calls route through Headroom automatically.

    \b
    Supported tools (one Click subcommand per tool):
        headroom wrap claude              # Claude Code (Anthropic)
        headroom wrap codex               # OpenAI Codex CLI
        headroom wrap copilot -- --model claude-sonnet-4-20250514
        headroom wrap aider               # Aider
        headroom wrap cursor              # Cursor (prints config instructions)
        headroom wrap cline               # Cline (VS Code; prints config instructions)
        headroom wrap continue            # Continue (VS Code/JetBrains; injects systemMessage)
        headroom wrap goose               # Goose (Block) CLI
        headroom wrap openhands           # OpenHands CLI
        headroom wrap openclaw            # OpenClaw plugin bootstrap

    \b
    `wrap` vs `proxy`:
        - `headroom wrap <tool>` — convenience: starts the proxy for you,
          sets the right env vars, and launches the wrapped CLI.
        - `headroom proxy` — just the proxy. Use this with any
          OpenAI/Anthropic-compatible client by setting
          ANTHROPIC_BASE_URL / OPENAI_BASE_URL yourself.

    \b
    Note: `headroom wrap opencode` does NOT exist. For opencode, run
    `headroom proxy` and point opencode at it via OPENAI_BASE_URL.
    `openclaw` is a separate tool — different from opencode.
    """


@main.group()
def unwrap() -> None:
    """Undo durable Headroom wrapping for supported tools."""


# =============================================================================
# Claude Code
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
)
@click.option(
    "--no-mcp",
    is_flag=True,
    help="Skip headroom MCP server registration (compression markers will be unactionable)",
)
@click.option("--no-serena", is_flag=True, help="Skip Serena MCP server registration")
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to MEMORY.md)"
)
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def claude(
    port: int,
    no_rtk: bool,
    no_mcp: bool,
    no_serena: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
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
        headroom wrap claude                    # Start everything
        headroom wrap claude --memory           # With persistent memory
        headroom wrap claude --resume <id>      # Resume a session
        headroom wrap claude -- -p              # Claude in print mode
        headroom wrap claude --code-graph        # With code graph intelligence
        headroom wrap claude --no-context-tool  # Skip CLI context-tool setup
        headroom wrap claude --no-mcp           # Skip MCP retrieve tool registration
        headroom wrap claude --no-serena        # Skip Serena MCP registration
    """
    if prepare_only:
        if not no_rtk:
            if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
                _setup_lean_ctx_agent("claude", verbose=verbose)
            else:
                _prepare_wrap_rtk(verbose=verbose, label="Claude")
        return

    claude_bin = shutil.which("claude")
    if not claude_bin:
        click.echo("Error: 'claude' not found in PATH.")
        click.echo("Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
        raise SystemExit(1)

    # Setup rtk before launching (Claude-specific)
    proxy_holder: list[subprocess.Popen | None] = [None]
    cleanup = _make_cleanup(proxy_holder, port)
    signal.signal(signal.SIGINT, _ignore_child_sigint)
    signal.signal(signal.SIGTERM, cleanup)

    # Memory sync BEFORE proxy startup — sync headroom DB ↔ Claude's files
    if memory:
        try:
            import subprocess as _sp

            mem_dir = Path.cwd() / ".headroom"
            mem_dir.mkdir(parents=True, exist_ok=True)
            _sync_db = str(mem_dir / "memory.db")
            _sync_user = os.environ.get("USER", os.environ.get("USERNAME", "default"))

            click.echo(f"  Syncing memory (user={_sync_user})...")
            sync_result = _sp.run(
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

        proxy_holder[0] = _ensure_proxy(
            port, no_proxy, learn=learn, memory=memory, agent_type="claude", code_graph=code_graph
        )

        if not no_rtk:
            if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
                click.echo("  Setting up lean-ctx...")
                _setup_lean_ctx_agent("claude", verbose=verbose)
            else:
                click.echo("  Setting up rtk...")
                _setup_rtk(verbose=verbose)
        elif verbose:
            click.echo("  Skipping CLI context tool (--no-context-tool)")

        if not no_mcp:
            from headroom.mcp_registry import ClaudeRegistrar

            _setup_headroom_mcp(ClaudeRegistrar(), port, verbose=verbose)
        elif verbose:
            click.echo("  Skipping MCP retrieve tool (--no-mcp)")

        if not no_serena:
            from headroom.mcp_registry import ClaudeRegistrar

            _setup_serena_mcp(ClaudeRegistrar(), context="claude-code", verbose=verbose)
        elif verbose:
            click.echo("  Skipping Serena MCP (--no-serena)")

        if code_graph:
            _setup_code_graph(verbose=verbose)

        click.echo()
        click.echo("  Launching Claude Code (API routed through Headroom)...")
        click.echo(f"  ANTHROPIC_BASE_URL={_claude_proxy_base_url(port)}")
        if claude_args:
            click.echo(f"  Extra args: {' '.join(claude_args)}")
        _print_telemetry_notice()
        click.echo()

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = _claude_proxy_base_url(port)

        result = subprocess.run([claude_bin, *claude_args], env=env)
        raise SystemExit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        cleanup()


# =============================================================================
# Claude Code (unwrap)
# =============================================================================


@unwrap.command("claude")
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
@click.option("--keep-mcp", is_flag=True, help="Keep Headroom MCP registrations")
@click.option("--keep-rtk", is_flag=True, help="Keep rtk Claude hooks")
def unwrap_claude(
    port: int,
    no_stop_proxy: bool,
    keep_mcp: bool,
    keep_rtk: bool,
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
            serena_status = _remove_headroom_installed_serena_mcp(registrar)
            if removed_headroom:
                click.echo("  Removed Headroom MCP retrieve tool from Claude.")
            else:
                click.echo("  Headroom MCP retrieve tool was not registered in Claude.")
            if removed_code_graph:
                click.echo("  Removed code graph MCP server from Claude.")
            if serena_status == "removed":
                click.echo("  Removed Headroom-installed Serena MCP server from Claude.")
            elif serena_status == "failed":
                click.echo("  Serena MCP server matched Headroom ledger but could not be removed.")
        else:
            click.echo("  Claude Code not detected; skipped MCP cleanup.")
    else:
        click.echo("  Kept Claude MCP registrations (--keep-mcp).")

    if not keep_rtk:
        if _remove_claude_rtk_hooks():
            click.echo("  Removed rtk Claude hook from settings.json.")
        else:
            click.echo("  No rtk Claude hook found in settings.json.")
    else:
        click.echo("  Kept rtk Claude hooks (--keep-rtk).")

    click.echo()
    click.echo("✓ Claude is no longer durably wrapped by Headroom.")
    if not no_stop_proxy:
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()


# =============================================================================
# GitHub Copilot CLI
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
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
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("copilot_args", nargs=-1, type=click.UNPROCESSED)
def copilot(
    port: int,
    no_rtk: bool,
    no_proxy: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    provider_type: str,
    wire_api: str | None,
    memory: bool,
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
        headroom wrap copilot --no-context-tool -- --prompt "explain this file"
    """
    copilot_bin = shutil.which("copilot")
    if not copilot_bin:
        click.echo("Error: 'copilot' not found in PATH.")
        click.echo(
            "Install GitHub Copilot CLI: "
            "https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli"
        )
        raise SystemExit(1)

    effective_backend = backend or os.environ.get("HEADROOM_BACKEND")
    if _check_proxy(port):
        running_backend = _detect_running_proxy_backend(port)
        if effective_backend and running_backend and effective_backend != running_backend:
            raise click.ClickException(
                f"Proxy already running on port {port} with backend '{running_backend}'. "
                f"Stop it or rerun with --backend {running_backend}."
            )
        effective_backend = running_backend or effective_backend

    effective_provider_type = _resolve_copilot_provider_type(effective_backend, provider_type)
    _validate_copilot_configuration(
        provider_type=effective_provider_type,
        wire_api=wire_api,
        backend=effective_backend,
    )

    if not no_rtk:
        if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
            click.echo("  Setting up lean-ctx for Copilot...")
            _setup_lean_ctx_agent("copilot", verbose=verbose)
        else:
            click.echo("  Setting up rtk for Copilot...")
            rtk_path = _ensure_rtk_binary(verbose=verbose)
            if rtk_path:
                copilot_instructions = Path.cwd() / ".github" / "copilot-instructions.md"
                _inject_rtk_instructions(copilot_instructions, verbose=verbose)

    env = os.environ.copy()
    openai_api_url: str | None = None
    if _should_use_copilot_oauth(
        backend=effective_backend,
        provider_type=provider_type,
        env=env,
    ):
        client_bearer = resolve_client_bearer_token()
        if not client_bearer:
            raise click.ClickException(
                "GitHub Copilot auth was detected but no reusable bearer token could be resolved."
            )

        effective_wire_api = wire_api or "completions"
        env["COPILOT_PROVIDER_TYPE"] = "openai"
        env["COPILOT_PROVIDER_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["COPILOT_PROVIDER_WIRE_API"] = effective_wire_api
        env["COPILOT_PROVIDER_BEARER_TOKEN"] = client_bearer
        env.pop("COPILOT_PROVIDER_API_KEY", None)
        env_vars_display = [
            "COPILOT_PROVIDER_TYPE=openai",
            f"COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:{port}/v1",
            f"COPILOT_PROVIDER_WIRE_API={effective_wire_api}",
            "COPILOT_AUTH_MODE=github-oauth",
        ]
        openai_api_url = COPILOT_API_URL
    else:
        env, env_vars_display = _build_copilot_launch_env(
            port=port,
            provider_type=effective_provider_type,
            wire_api=wire_api,
            environ=env,
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

    if not _copilot_model_configured(copilot_args, env):
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
    )


# =============================================================================
# OpenAI Codex CLI
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
)
@click.option(
    "--no-mcp",
    is_flag=True,
    help="Skip headroom MCP server registration (compression markers will be unactionable)",
)
@click.option("--no-serena", is_flag=True, help="Skip Serena MCP server registration")
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
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
    no_rtk: bool,
    no_mcp: bool,
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
    Sets up the selected CLI context tool so Codex uses token-optimized
    commands (60-90% savings on shell output). Also
    registers the headroom MCP server in ~/.codex/config.toml so Codex
    can call ``headroom_retrieve`` on compression markers.

    \b
    Examples:
        headroom wrap codex                         # Start proxy + context tool + mcp + codex
        headroom wrap codex -- "fix the bug"        # Pass prompt to codex
        headroom wrap codex --no-context-tool       # Skip CLI context-tool setup
        headroom wrap codex --no-mcp                # Skip MCP retrieve tool registration
        headroom wrap codex --no-serena             # Skip Serena MCP registration
        headroom wrap codex --port 9999             # Custom proxy port
        headroom wrap codex --backend anyllm --anyllm-provider groq
    """
    # Snapshot ~/.codex/config.toml BEFORE any wrap-time mutation so
    # `headroom unwrap codex` can restore the user's pre-wrap state
    # byte-for-byte. The snapshot is a no-op if the backup already exists
    # or if the file already has Headroom markers, so this is safe to
    # call repeatedly. Crucially this must run before MCP install, which
    # writes its marker block to the same file.
    _codex_config_file, _codex_backup_file = _codex_config_paths()
    _snapshot_codex_config_if_unwrapped(_codex_config_file, _codex_backup_file)

    # Setup CLI context tool for Codex.
    if not no_rtk:
        if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
            click.echo("  Setting up lean-ctx for Codex...")
            _setup_lean_ctx_agent("codex", verbose=verbose)
        else:
            click.echo("  Setting up rtk for Codex...")
            rtk_path = _ensure_rtk_binary(verbose=verbose)
            if rtk_path:
                # Inject into project AGENTS.md (Codex reads this automatically)
                agents_md = Path.cwd() / "AGENTS.md"
                _inject_rtk_instructions(agents_md, verbose=verbose)

                # Also inject into global ~/.codex/AGENTS.md
                global_agents = Path.home() / ".codex" / "AGENTS.md"
                _inject_rtk_instructions(global_agents, verbose=verbose)

    # Register headroom MCP server in ~/.codex/config.toml so Codex can
    # call headroom_retrieve on compression markers from the proxy.
    if not no_mcp:
        from headroom.mcp_registry import CodexRegistrar

        # Codex starts a long-lived local MCP subprocess from config.toml.
        # If a previous wrap used another port, retrieval can silently point
        # at the wrong proxy while model traffic uses the right one.
        _setup_headroom_mcp(CodexRegistrar(), port, verbose=verbose, force=True)
    elif verbose:
        click.echo("  Skipping MCP retrieve tool (--no-mcp)")

    if not no_serena:
        from headroom.mcp_registry import CodexRegistrar

        _setup_serena_mcp(CodexRegistrar(), context="codex", verbose=verbose, force=True)
    elif verbose:
        click.echo("  Skipping Serena MCP (--no-serena)")

    # Setup memory MCP server for Codex (native tool integration)
    if memory:
        click.echo("  Setting up memory for Codex...")
        mem_dir = Path.cwd() / ".headroom"
        mem_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(mem_dir / "memory.db")
        mem_user = os.environ.get("USER", os.environ.get("USERNAME", "default"))

        # Register MCP server in Codex config
        _inject_memory_mcp_config(db_path, mem_user)

        # Inject memory guidance into project AGENTS.md
        agents_md = Path.cwd() / "AGENTS.md"
        _inject_memory_agents_md(agents_md)

        # Sync Claude's memories → DB so MCP search finds them
        try:
            import asyncio

            from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
            from headroom.memory.sync import sync_import
            from headroom.memory.sync_adapters.claude_code import (
                ClaudeCodeAdapter,
                get_claude_memory_dir,
            )

            claude_memory_dir = get_claude_memory_dir()

            async def _import_claude_memories() -> int:
                config = LocalBackendConfig(db_path=db_path)
                backend = LocalBackend(config)
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

    if prepare_only:
        _inject_codex_provider_config(port)
        return

    codex_bin = shutil.which("codex")
    if not codex_bin:
        click.echo("Error: 'codex' not found in PATH.")
        click.echo("Install Codex CLI: npm install -g @openai/codex")
        raise SystemExit(1)

    env, env_vars_display = _build_codex_launch_env(port, os.environ)

    # Inject Headroom provider into Codex config so WebSocket traffic also
    # routes through the proxy.  Codex ignores OPENAI_BASE_URL for its WS
    # transport unless a custom provider declares supports_websockets = true.
    # NOTE: this must run BEFORE _inject_memory_mcp_config because it rewrites
    # the config file.  Re-inject MCP config after if memory is enabled.
    _inject_codex_provider_config(port)
    if memory:
        mem_dir = Path.cwd() / ".headroom"
        _inject_memory_mcp_config(
            str(mem_dir / "memory.db"),
            os.environ.get("USER", os.environ.get("USERNAME", "default")),
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
    )


# =============================================================================
# Aider
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
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
    no_rtk: bool,
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
    Sets up the selected CLI context tool so aider uses token-optimized commands.

    \b
    Examples:
        headroom wrap aider                              # Start proxy + context tool + aider
        headroom wrap aider -- --model gpt-4o            # Use GPT-4o
        headroom wrap aider -- --model claude-sonnet-4   # Use Claude
        headroom wrap aider --no-context-tool            # Skip CLI context-tool setup
        headroom wrap aider --backend litellm-vertex --region us-central1
    """
    # Setup CLI context tool for aider.
    if not no_rtk:
        if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
            click.echo("  Setting up lean-ctx for aider...")
            _setup_lean_ctx_agent("aider", verbose=verbose)
        else:
            click.echo("  Setting up rtk for aider...")
            rtk_path = _ensure_rtk_binary(verbose=verbose)
            if rtk_path:
                # aider reads CONVENTIONS.md from project root
                conventions = Path.cwd() / "CONVENTIONS.md"
                _inject_rtk_instructions(conventions, verbose=verbose)

    if prepare_only:
        return

    aider_bin = shutil.which("aider")
    if not aider_bin:
        click.echo("Error: 'aider' not found in PATH.")
        click.echo("Install aider: pip install aider-chat")
        raise SystemExit(1)

    env, env_vars_display = _build_aider_launch_env(port, os.environ)

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
# Cursor
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
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
    no_rtk: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
) -> None:
    """Start Headroom proxy for use with Cursor.

    \b
    Cursor reads its API configuration from its settings UI, not from
    environment variables. This command starts the proxy, sets up the selected
    CLI context tool, and prints the Cursor settings.

    \b
    After running this command, open Cursor and configure:
        Settings > Models > OpenAI API Key > Advanced > Override Base URL

    \b
    Example:
        headroom wrap cursor                # Start proxy + context-tool instructions
        headroom wrap cursor --no-context-tool  # Proxy only, no CLI context tool
        headroom wrap cursor --port 9999    # Custom proxy port
    """
    cursorrules: Path | None = Path.cwd() / ".cursorrules" if not no_rtk else None
    if not no_rtk:
        _setup_context_tool_for_agent(
            agent="cursor",
            agent_display="Cursor",
            marker_path=cursorrules,
            on_rtk_ready=lambda _rtk: _inject_rtk_instructions(
                cast(Path, cursorrules), verbose=verbose
            ),
            verbose=verbose,
        )

    if prepare_only:
        return

    def _print_cursor_setup() -> None:
        for line in _render_cursor_setup_lines(port):
            click.echo(line)
        if not no_rtk:
            click.echo()
            if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
                click.echo("  lean-ctx configured for Cursor")
            else:
                click.echo("  rtk instructions injected into .cursorrules")
            click.echo("  Cursor will use token-optimized commands automatically.")

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
# Cline (VS Code extension)
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
def cline(
    port: int,
    no_rtk: bool,
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
    the proxy, sets up the selected CLI context tool (injecting RTK guidance
    into .clinerules at the project root), and prints the Cline settings the
    user should configure.

    \b
    After running this command, open Cline's settings in VS Code and configure
    the API Base URL to point at the local Headroom proxy.

    \b
    Uninstall: there is no ``headroom unwrap cline`` subcommand. To remove the
    injected guidance, hand-edit ``.clinerules`` at the project root and
    delete everything between ``<!-- headroom:rtk-instructions -->`` and
    ``<!-- /headroom:rtk-instructions -->`` (inclusive). If ``lean-ctx`` mode
    is selected, the lean-ctx agent name ``cline`` may not be recognized by
    the local lean-ctx binary; a warning is printed in that case and setup
    is skipped silently.

    \b
    Examples:
        headroom wrap cline                  # Start proxy + .clinerules instructions
        headroom wrap cline --no-context-tool # Proxy only, no CLI context tool
        headroom wrap cline --port 9999      # Custom proxy port
    """
    # Pre-compute the marker path so the KeyboardInterrupt handler can report
    # its location even if the interrupt fires before _inject_rtk_instructions
    # returns (e.g., during the inner _ensure_rtk_binary download).
    clinerules: Path | None = Path.cwd() / ".clinerules" if not no_rtk else None
    if not no_rtk:
        _setup_context_tool_for_agent(
            agent="cline",
            agent_display="Cline",
            marker_path=clinerules,
            on_rtk_ready=lambda _rtk: _inject_rtk_instructions(
                cast(Path, clinerules), verbose=verbose
            ),
            verbose=verbose,
        )

    if prepare_only:
        return

    def _print_cline_setup() -> None:
        anthropic_base = _claude_proxy_base_url(port)
        openai_base = f"http://127.0.0.1:{port}/v1"
        click.echo("  Configure Cline in VS Code:")
        click.echo("    Settings > Cline > API Provider")
        click.echo(f"    Anthropic Base URL: {anthropic_base}")
        click.echo(f"    OpenAI Compatible Base URL: {openai_base}")
        if not no_rtk:
            click.echo()
            if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
                click.echo("  lean-ctx configured for Cline")
            else:
                click.echo("  rtk instructions injected into .clinerules")
            click.echo("  Cline will use token-optimized commands automatically.")

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
# Continue (VS Code / JetBrains extension)
# =============================================================================


@wrap.command("continue", context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
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
    no_rtk: bool,
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
    command starts the proxy, sets up the selected CLI context tool by
    extending ``systemMessage`` with RTK guidance, and prints the per-model
    ``apiBase`` the user should configure manually.

    \b
    Continue is an IDE extension — its API base URL is configured per-model
    in config.json (or via the IDE UI), not via environment variables. The
    config file is overridable via --config.

    \b
    Note: Continue's modern config is YAML-first (``.continue/config.yaml``).
    This helper only writes the JSON variant. Users on the YAML schema should
    configure ``systemMessage`` through that file by hand.

    \b
    Per-model handling: Continue overrides top-level ``systemMessage`` with
    per-model ``systemMessage`` when set, so this command also injects into
    each ``models[i].systemMessage`` if the ``models`` array is present.
    Existing non-string ``systemMessage`` values are NEVER overwritten — the
    command warns loudly and leaves them in place. To opt in, clear the
    existing value first.

    \b
    Uninstall: there is no ``headroom unwrap continue`` subcommand. To remove
    the injected guidance, hand-edit ``.continue/config.json`` and delete
    everything between ``<!-- headroom:rtk-instructions -->`` and
    ``<!-- /headroom:rtk-instructions -->`` (inclusive) from every
    ``systemMessage`` field — both top-level and inside ``models[*]``. If
    ``lean-ctx`` mode is selected, the lean-ctx agent name ``continue`` may
    not be recognized by the local lean-ctx binary; a warning is printed in
    that case and setup is skipped silently.

    \b
    Examples:
        headroom wrap continue                # Start proxy + inject systemMessage
        headroom wrap continue --no-context-tool   # Proxy only
        headroom wrap continue --port 9999    # Custom proxy port
        headroom wrap continue --config path/to/config.json
    """
    config_file = config_path or (Path.cwd() / ".continue" / "config.json")

    if not no_rtk:
        _setup_context_tool_for_agent(
            agent="continue",
            agent_display="Continue",
            marker_path=config_file,
            on_rtk_ready=lambda _rtk: _inject_continue_rtk_systemmessage(
                config_file, verbose=verbose
            ),
            verbose=verbose,
        )

    if prepare_only:
        return

    def _print_continue_setup() -> None:
        anthropic_base = _claude_proxy_base_url(port)
        openai_base = f"http://127.0.0.1:{port}/v1"
        click.echo("  Configure Continue in your IDE:")
        click.echo(f"    Edit {config_file} and set, per model:")
        click.echo(f'      "apiBase": "{openai_base}"          # OpenAI-compatible models')
        click.echo(f'      "apiBase": "{anthropic_base}"       # Anthropic models')
        if not no_rtk:
            click.echo()
            if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
                click.echo("  lean-ctx configured for Continue")
            else:
                click.echo(f"  rtk instructions injected into {config_file.name} systemMessage")
            click.echo("  Continue will use token-optimized commands automatically.")

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
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
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
    no_rtk: bool,
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
    through Headroom. Sets up the selected CLI context tool by injecting RTK
    guidance into .goosehints at the project root (Goose reads this file as
    extra system context).

    \b
    Uninstall: there is no ``headroom unwrap goose`` subcommand. To remove the
    injected guidance, hand-edit ``.goosehints`` at the project root and
    delete everything between ``<!-- headroom:rtk-instructions -->`` and
    ``<!-- /headroom:rtk-instructions -->`` (inclusive). If ``lean-ctx`` mode
    is selected, the lean-ctx agent name ``goose`` may not be recognized by
    the local lean-ctx binary; a warning is printed in that case and setup
    is skipped silently.

    \b
    Examples:
        headroom wrap goose                          # Start proxy + context tool + goose
        headroom wrap goose -- session               # Start a Goose session
        headroom wrap goose -- --provider anthropic  # Pass args to goose
        headroom wrap goose --no-context-tool        # Skip CLI context-tool setup
    """
    # Goose reads .goosehints from the project root as extra context.
    # Pre-compute the marker path so the KeyboardInterrupt handler can report
    # its location even if the interrupt fires before _inject_rtk_instructions
    # returns (e.g., during the inner _ensure_rtk_binary download).
    goosehints: Path | None = Path.cwd() / ".goosehints" if not no_rtk else None
    if not no_rtk:
        _setup_context_tool_for_agent(
            agent="goose",
            agent_display="Goose",
            marker_path=goosehints,
            on_rtk_ready=lambda _rtk: _inject_rtk_instructions(
                cast(Path, goosehints), verbose=verbose
            ),
            verbose=verbose,
        )

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
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
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
    no_rtk: bool,
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
    through Headroom. Instructions are injected via the
    ``OPENHANDS_INSTRUCTIONS`` environment variable at launch time so the
    on-disk OpenHands config is left untouched.

    \b
    The ``OPENHANDS_INSTRUCTIONS`` value injected by this command contains the
    ``<!-- headroom:rtk-instructions -->`` marker. To uninstall, simply do not
    set ``OPENHANDS_INSTRUCTIONS`` in the parent shell — this command never
    writes to disk, so nothing to clean up. If ``lean-ctx`` mode is selected,
    the lean-ctx agent name ``openhands`` may not be recognized by the local
    lean-ctx binary; a warning is printed in that case and rtk-style guidance
    falls through.

    \b
    Examples:
        headroom wrap openhands                # Start proxy + context tool + openhands
        headroom wrap openhands -- --task ...  # Pass args to openhands
        headroom wrap openhands --no-context-tool
    """
    # openhands never writes to disk — its rtk guidance ships via the
    # OPENHANDS_INSTRUCTIONS env var below — so marker_path is None and
    # rtk_required gates the env-only path: without an rtk binary there
    # is no fallback marker file to fall through to.
    rtk_path: Path | None = None
    if not no_rtk:
        rtk_path = _setup_context_tool_for_agent(
            agent="openhands",
            agent_display="OpenHands",
            marker_path=None,
            rtk_required=True,
            verbose=verbose,
        )

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
    if not no_rtk and rtk_path:
        # Inject rtk guidance via env var so OpenHands picks it up as the
        # session's instruction prefix. Appending instead of overwriting
        # any pre-existing OPENHANDS_INSTRUCTIONS so user-supplied content
        # is preserved. The marker check guards against double-injection
        # when the user inherits an env var that already has the rtk block.
        existing_instructions = env.get("OPENHANDS_INSTRUCTIONS", "")
        if _RTK_MARKER in existing_instructions:
            # Already injected — pre-existing env var contains marker.
            pass
        elif existing_instructions.strip():
            env["OPENHANDS_INSTRUCTIONS"] = (
                existing_instructions.rstrip() + "\n\n" + RTK_INSTRUCTIONS_BLOCK
            )
        else:
            env["OPENHANDS_INSTRUCTIONS"] = RTK_INSTRUCTIONS_BLOCK

    env_vars_display = [
        f"OPENAI_BASE_URL={openai_base}",
        f"ANTHROPIC_BASE_URL={anthropic_base}",
        f"LLM_BASE_URL={openai_base}",
    ]
    if not no_rtk and "OPENHANDS_INSTRUCTIONS" in env:
        env_vars_display.append("OPENHANDS_INSTRUCTIONS=<rtk instructions injected>")

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
# Hermes (Nous Research)
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-context-tool",
    "--no-rtk",
    "no_rtk",
    is_flag=True,
    help="Skip CLI context-tool setup",
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
@click.argument("hermes_args", nargs=-1, type=click.UNPROCESSED)
def hermes(
    port: int,
    no_rtk: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
    hermes_args: tuple,
) -> None:
    """Launch Hermes (Nous Research) agent through Headroom proxy.

    \b
    Sets OPENAI_BASE_URL and ANTHROPIC_BASE_URL to route Hermes' API calls
    through Headroom. Sets up the selected CLI context tool by injecting RTK
    guidance into CONVENTIONS.md at the project root.

    \b
    Examples:
        headroom wrap hermes                         # Start proxy + context tool + hermes
        headroom wrap hermes -- --model hermes-3     # Pass args to hermes
        headroom wrap hermes --no-context-tool       # Skip CLI context-tool setup
        headroom wrap hermes --backend litellm-vertex --region us-central1
    """
    if not no_rtk:
        if _selected_context_tool() == _CONTEXT_TOOL_LEAN_CTX:
            click.echo("  Setting up lean-ctx for hermes...")
            _setup_lean_ctx_agent("hermes", verbose=verbose)
        else:
            click.echo("  Setting up rtk for hermes...")
            rtk_path = _ensure_rtk_binary(verbose=verbose)
            if rtk_path:
                conventions = Path.cwd() / "CONVENTIONS.md"
                _inject_rtk_instructions(conventions, verbose=verbose)

    if prepare_only:
        return

    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        click.echo("Error: 'hermes' not found in PATH.")
        click.echo("Install Hermes: https://hermes-agent.nousresearch.com/")
        raise SystemExit(1)

    env, env_vars_display = _build_hermes_launch_env(port, os.environ)

    _launch_tool(
        binary=hermes_bin,
        args=hermes_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="HERMES",
        env_vars_display=env_vars_display,
        learn=learn,
        memory=memory,
        agent_type="hermes",
        code_graph=code_graph,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# OpenClaw
# =============================================================================


@wrap.command("openclaw")
@click.option(
    "--plugin-path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Path to local OpenClaw plugin source directory (advanced/dev override)",
)
@click.option(
    "--plugin-spec",
    default="headroom-ai/openclaw",
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
@click.option("--proxy-port", default=8787, type=int, help="Headroom proxy port")
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

    click.echo("  Writing plugin configuration...")
    _write_openclaw_plugin_entry(openclaw_bin, entry)

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
    install_result = subprocess.run(
        install_cmd,
        cwd=str(install_cwd) if install_cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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


@unwrap.command("openclaw")
@click.option("--proxy-port", default=8787, type=int, help="Headroom proxy port")
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


@unwrap.command("codex")
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-stop-proxy", is_flag=True, help="Do not stop the local Headroom proxy")
def unwrap_codex(port: int, no_stop_proxy: bool) -> None:
    """Undo ``headroom wrap codex`` edits to ``~/.codex/config.toml``.

    Behaviour:

    * If a pre-wrap backup (``config.toml.headroom-backup``) exists, the
      original file is restored byte-for-byte and the backup is removed.
    * Otherwise, if the config file still contains the Headroom-managed
      block, that block is stripped out and the rest of the file is
      preserved.
    * If the config only ever contained Headroom-written content, the file
      is removed entirely so Codex falls back to its defaults.
    * If neither a backup nor a Headroom block is present, this is a safe
      no-op (the user either never wrapped, or already unwrapped).
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
        click.echo(f"  Nothing to undo: {config_file} has no Headroom wrap markers.")

    click.echo()
    click.echo("✓ Codex is no longer routed through the Headroom proxy.")
    if not no_stop_proxy:
        _echo_unwrap_proxy_stop_status(_stop_local_proxy_for_unwrap(port), port)
    click.echo()
