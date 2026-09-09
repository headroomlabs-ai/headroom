"""Durable agent initialization commands (Claude Code only)."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from hashlib import sha1
from pathlib import Path
from typing import Any

import click

from headroom._subprocess import run
from headroom.install.models import ConfigScope, InstallPreset, RuntimeKind, SupervisorKind
from headroom.install.paths import claude_settings_path, validate_profile_name
from headroom.install.planner import build_manifest
from headroom.install.runtime import (
    acquire_runtime_start_lock,
    resolve_headroom_command,
    runtime_status,
    start_detached_agent,
    start_persistent_docker,
    stop_runtime,
    wait_ready,
)
from headroom.install.state import ManifestError, load_manifest, save_manifest
from headroom.install.supervisors import start_supervisor
from headroom.providers.claude import TOOL_SEARCH_DEFAULT, TOOL_SEARCH_ENV
from headroom.providers.claude.runtime import TOOL_SEARCH_FOUNDRY_DEFAULT

from .main import main

logger = logging.getLogger(__name__)

_VERBOSE_HANDLER_ATTR = "_headroom_init_verbose_handler"

_GLOBAL_PROFILE = "init-user"
_CLAUDE_HOOK_MARKER = "headroom-init-claude"
_SUPPORTED_TARGETS = ("claude",)
_LOCAL_TARGETS = {"claude"}
_GLOBAL_TARGETS = {"claude"}
_STARTUP_READY_TIMEOUT_SECONDS = 15


def _command_string(parts: list[str]) -> str:
    if os.name == "nt":
        # Normalize backslash paths to forward slashes so hook commands
        # work when Claude Code executes them via Git Bash (#724).
        parts = [p.replace("\\", "/") for p in parts]
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def _hook_command(*parts: str) -> str:
    return _command_string([*resolve_headroom_command(), "init", "hook", "ensure", *parts])


def _powershell_matcher() -> str:
    return "Bash|PowerShell" if os.name == "nt" else "Bash"


def _enable_verbose_logging() -> None:
    """Attach a stderr handler to the init logger at DEBUG level.

    Idempotent: calling this multiple times in one process (e.g. when nested
    subcommands are invoked) leaves exactly one handler attached. Does NOT
    mutate stdout; all verbose output goes to stderr so ``headroom init``
    can still be composed in pipes that consume stdout.
    """

    if getattr(logger, _VERBOSE_HANDLER_ATTR, None) is not None:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("[headroom init] %(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    setattr(logger, _VERBOSE_HANDLER_ATTR, handler)


def _local_profile(cwd: Path | None = None) -> str:
    root = (cwd or Path.cwd()).resolve()
    slug = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in root.name.lower()).strip(
        "-"
    )
    digest = sha1(str(root).encode("utf-8")).hexdigest()[:8]
    return validate_profile_name(f"init-{slug or 'repo'}-{digest}")


def _runtime_profile(global_scope: bool, cwd: Path | None = None) -> str:
    return _GLOBAL_PROFILE if global_scope else _local_profile(cwd)


def _claude_scope_path(global_scope: bool) -> Path:
    if global_scope:
        return claude_settings_path()
    return Path.cwd() / ".claude" / "settings.local.json"


def _json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as e:
        # This is a user-owned file (e.g. ~/.claude/settings.json) that the
        # callers read-merge-write. Returning {} would make the following
        # _write_json overwrite it, silently discarding the user's settings;
        # letting the raw JSONDecodeError propagate crashes `headroom init`
        # with a traceback. Abort with an actionable message so the user can
        # fix the JSON (or move it aside) without losing it.
        raise click.ClickException(
            f"{path} contains invalid JSON ({e}); fix it and re-run, or move it aside."
        ) from e
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    logger.debug("write json: %s (keys=%s)", path, sorted(payload.keys()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_claude_hooks(path: Path, profile: str, port: int) -> None:
    logger.debug("ensure claude hooks: %s (profile=%s, port=%s)", path, profile, port)
    payload = _json_file(path)
    env_map = dict(payload.get("env") or {}) if isinstance(payload.get("env"), dict) else {}
    env_map["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    # GH #746: with a custom ANTHROPIC_BASE_URL and ENABLE_TOOL_SEARCH unset,
    # Claude Code stops deferring MCP/system tool schemas and materializes them
    # all into its context window — overflowing it (breaks sub-agent spawns,
    # forces constant compaction). Keep deferral on; respect a user-set value.
    # Shares the TOOL_SEARCH_* constants with `wrap` and `install`.
    tool_search_default = (
        TOOL_SEARCH_FOUNDRY_DEFAULT
        if os.environ.get("CLAUDE_CODE_USE_FOUNDRY")
        else TOOL_SEARCH_DEFAULT
    )
    env_map.setdefault(TOOL_SEARCH_ENV, tool_search_default)
    payload["env"] = env_map

    hooks = dict(payload.get("hooks") or {}) if isinstance(payload.get("hooks"), dict) else {}
    command = _hook_command("--profile", profile)
    for event, matcher in (
        ("SessionStart", "startup|resume"),
        ("PreToolUse", _powershell_matcher()),
    ):
        entries = list(hooks.get(event) or []) if isinstance(hooks.get(event), list) else []
        retained: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                retained.append(entry)
                continue
            hook_items = entry.get("hooks")
            if not isinstance(hook_items, list):
                retained.append(entry)
                continue
            has_headroom = any(
                isinstance(item, dict)
                and item.get("command")
                and _CLAUDE_HOOK_MARKER in str(item.get("command"))
                for item in hook_items
            )
            if not has_headroom:
                retained.append(entry)
        retained.append(
            {
                "matcher": matcher,
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{command} --marker {_CLAUDE_HOOK_MARKER}",
                        "timeout": 15,
                    }
                ],
            }
        )
        hooks[event] = retained
    payload["hooks"] = hooks
    _write_json(path, payload)


def _manifest_changed(existing: Any, *, port: int, memory: bool) -> bool:
    return any(
        [
            getattr(existing, "port", port) != port,
            getattr(existing, "memory_enabled", memory) != memory,
        ]
    )


def _ensure_runtime_manifest(
    *,
    global_scope: bool,
    targets: list[str],
    port: int,
    memory: bool,
) -> str:
    profile = _runtime_profile(global_scope)
    try:
        existing = load_manifest(profile)
    except ManifestError as e:
        # Recover from a corrupt manifest by overwriting it rather than crashing.
        click.echo(f"Warning: {e}; overwriting.")
        existing = None
    merged_targets = sorted(set(existing.targets if existing else []).union(targets))
    manifest = build_manifest(
        profile=profile,
        preset=InstallPreset.PERSISTENT_TASK.value,
        runtime_kind=RuntimeKind.PYTHON.value,
        scope=ConfigScope.USER.value,
        provider_mode="manual",
        targets=merged_targets,
        port=port,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=memory,
        telemetry_enabled=False,
        image="",
    )
    manifest.supervisor_kind = SupervisorKind.NONE.value
    manifest.artifacts = []
    manifest.mutations = existing.mutations if existing else []
    if existing is not None and _manifest_changed(existing, port=port, memory=memory):
        try:
            stop_runtime(existing)
        except Exception:
            pass
    save_manifest(manifest)
    return profile


def _marketplace_source() -> str:
    override = os.environ.get("HEADROOM_MARKETPLACE_SOURCE")
    if override:
        return override
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / ".claude-plugin" / "marketplace.json").exists():
        return str(repo_root)
    return "headroomlabs-ai/headroom"


def _run_checked(command: list[str], *, action: str) -> None:
    logger.debug("subprocess [%s]: %s", action, _command_string(command))
    result = run(
        command,
        capture_output=True,
        text=True,
    )
    logger.debug(
        "subprocess [%s] exit=%s stdout=%r stderr=%r",
        action,
        result.returncode,
        result.stdout[:200],
        result.stderr[:200],
    )
    if result.returncode == 0:
        return
    detail = "\n".join(part for part in (result.stderr.strip(), result.stdout.strip()) if part)
    if "already" in detail.lower() or "exists" in detail.lower():
        logger.debug(
            "subprocess [%s] non-zero exit tolerated ('already'/'exists' detected)", action
        )
        return
    raise click.ClickException(f"{action} failed: {detail or result.returncode}")


def _install_claude_marketplace(scope: str) -> None:
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise click.ClickException("'claude' not found in PATH. Install Claude Code first.")
    source = _marketplace_source()
    _run_checked(
        [claude_bin, "plugin", "marketplace", "add", source], action="claude marketplace add"
    )
    _run_checked(
        [claude_bin, "plugin", "install", "headroom@headroom-marketplace", "--scope", scope],
        action="claude plugin install",
    )


@contextmanager
def _suppress_hook_output() -> Iterator[None]:
    """Keep best-effort hook recovery from emitting invalid hook output."""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _ensure_profile_running(profile: str) -> None:
    # Best-effort hook path: a corrupt manifest must not crash the session.
    try:
        manifest = load_manifest(profile)
    except ManifestError:
        return
    if manifest is None:
        return
    with _suppress_hook_output():
        if wait_ready(manifest, timeout_seconds=1):
            return
        try:
            with acquire_runtime_start_lock(manifest.profile) as acquired:
                if not acquired:
                    return
                if wait_ready(manifest, timeout_seconds=1):
                    return
                if runtime_status(manifest) == "running":
                    if wait_ready(manifest, timeout_seconds=_STARTUP_READY_TIMEOUT_SECONDS):
                        return
                    stop_runtime(manifest)
                if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
                    start_persistent_docker(manifest)
                elif manifest.supervisor_kind == SupervisorKind.SERVICE.value:
                    start_supervisor(manifest)
                else:
                    start_detached_agent(manifest.profile)
                wait_ready(manifest, timeout_seconds=45)
        except Exception:
            return


def _probe_init_targets(global_scope: bool) -> list[tuple[str, str | None]]:
    """Return ``[(target, which_result)]`` for every in-scope supported target.

    ``which_result`` is the absolute path reported by :func:`shutil.which`, or
    ``None`` when the binary is not on PATH. Callers use the list both to
    build an auto-detected target list and to produce a diagnostic error
    message when nothing was found.
    """

    allowed = _GLOBAL_TARGETS if global_scope else _LOCAL_TARGETS
    logger.debug(
        "detect_init_targets: global_scope=%s allowed=%s",
        global_scope,
        sorted(allowed),
    )
    probes: list[tuple[str, str | None]] = []
    for target in _SUPPORTED_TARGETS:
        if target not in allowed:
            continue
        path = shutil.which(target)
        logger.debug("detect_init_targets: shutil.which(%r) -> %s", target, path or "None")
        probes.append((target, path))
    return probes


def detect_init_targets(global_scope: bool) -> list[str]:
    """Return agent names in scope for which a binary was found on PATH."""

    return [name for name, path in _probe_init_targets(global_scope) if path]


def _format_empty_detection_error(global_scope: bool) -> str:
    """Build the error message shown when no in-scope targets were detected."""

    probes = _probe_init_targets(global_scope)
    scope_flag = "-g" if global_scope else ""
    scope_label = "user" if global_scope else "local"

    lines: list[str] = [
        f"No supported {scope_label}-scope agents were found on PATH.",
        "",
        "Headroom probed the following agents via shutil.which():",
    ]
    for name, path in probes:
        status = f"found at {path}" if path else "not found"
        lines.append(f"  - {name}: {status}")

    lines.extend(
        [
            "",
            f"The {scope_flag or '--local (no flag)'} option is still supported; "
            "headroom init just needs to know which agent to target.",
            "Install the agent you want first, then re-run with an explicit target:",
            "",
        ]
    )
    for name, _path in probes:
        flag = " -g" if global_scope else ""
        lines.append(f"  headroom init{flag} {name}")

    lines.extend(
        [
            "",
            "Tip: run `headroom init --help` to see all options.",
        ]
    )
    return "\n".join(lines)


def _init_claude(*, global_scope: bool, profile: str, port: int) -> None:
    _ensure_claude_hooks(_claude_scope_path(global_scope), profile, port)
    _install_claude_marketplace("user" if global_scope else "local")
    click.echo(f"Configured Claude Code ({'user' if global_scope else 'local'} scope).")
    click.echo("Restart Claude Code to activate Headroom hooks and provider routing.")


def _run_init_targets(
    *,
    targets: list[str],
    global_scope: bool,
    port: int,
    memory: bool,
) -> None:
    logger.debug(
        "run_init_targets: targets=%s global_scope=%s port=%s memory=%s",
        targets,
        global_scope,
        port,
        memory,
    )
    profile = _ensure_runtime_manifest(
        global_scope=global_scope,
        targets=targets,
        port=port,
        memory=memory,
    )
    logger.debug("run_init_targets: using profile=%s", profile)
    for target in targets:
        logger.debug("run_init_targets: dispatching -> %s", target)
        if target == "claude":
            _init_claude(global_scope=global_scope, profile=profile, port=port)

    _install_headroom_mcp_for_targets(targets=targets, port=port)


def _install_headroom_mcp_for_targets(*, targets: list[str], port: int) -> None:
    """Install the headroom MCP server into each detected target agent."""
    from headroom.mcp_registry import format_results, install_everywhere

    proxy_url = f"http://127.0.0.1:{port}"
    results = install_everywhere(proxy_url=proxy_url, agents=targets)
    if not results:
        return

    lines = format_results(
        results,
        verbose=True,
        overwrite_hint=f"headroom mcp install --proxy-url {proxy_url} --force",
    )
    if lines:
        click.echo("\nMCP retrieve tool:")
        for line in lines:
            click.echo(line)


@main.group(invoke_without_command=True)
@click.option("-g", "--global", "global_scope", is_flag=True, help="Install for the current user.")
@click.option(
    "--port",
    default=8787,
    type=click.IntRange(1, 65535),
    show_default=True,
    help="Headroom proxy port.",
)
@click.option("--memory", is_flag=True, help="Enable persistent memory in the proxy runtime.")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Emit debug-level diagnostics to stderr (flag values, shutil.which results, "
    "file paths touched, subprocess invocations and exit codes).",
)
@click.pass_context
def init(
    ctx: click.Context,
    global_scope: bool,
    port: int,
    memory: bool,
    verbose: bool,
) -> None:
    """Install durable Headroom integration for Claude Code."""
    if verbose:
        _enable_verbose_logging()
    logger.debug(
        "init: global_scope=%s port=%s memory=%s invoked_subcommand=%s",
        global_scope,
        port,
        memory,
        ctx.invoked_subcommand,
    )
    if ctx.invoked_subcommand is not None:
        ctx.obj = {
            "global_scope": global_scope,
            "port": port,
            "memory": memory,
            "verbose": verbose,
        }
        return

    targets = detect_init_targets(global_scope)
    if not targets:
        logger.debug("init: detect_init_targets returned empty; exiting with guided error")
        raise click.ClickException(_format_empty_detection_error(global_scope))
    logger.debug("init: detected targets=%s", targets)
    _run_init_targets(
        targets=targets,
        global_scope=global_scope,
        port=port,
        memory=memory,
    )


def _ctx_value(ctx: click.Context, key: str) -> Any:
    return (ctx.obj or {}).get(key)


@init.command("claude")
@click.pass_context
def init_claude(ctx: click.Context) -> None:
    """Install Claude Code durable hooks and provider routing."""
    _run_init_targets(
        targets=["claude"],
        global_scope=bool(_ctx_value(ctx, "global_scope")),
        port=int(_ctx_value(ctx, "port") or 8787),
        memory=bool(_ctx_value(ctx, "memory")),
    )


@init.group("hook", hidden=True)
def init_hook() -> None:
    """Internal hook helpers."""


@init_hook.command("ensure")
@click.option("--profile", default=None, help="Explicit deployment profile to ensure.")
@click.option("--marker", default=None, hidden=True)
def init_hook_ensure(profile: str | None, marker: str | None) -> None:
    """Best-effort ensure used by installed agent hooks."""
    del marker

    def _has_manifest(name: str) -> bool:
        # Best-effort: a corrupt manifest must not crash the session-start hook.
        try:
            return load_manifest(name) is not None
        except ManifestError:
            return False

    profiles: list[str] = []
    if profile:
        profiles.append(profile)
    else:
        local_profile = _local_profile()
        if _has_manifest(local_profile):
            profiles.append(local_profile)
        elif _has_manifest(_GLOBAL_PROFILE):
            profiles.append(_GLOBAL_PROFILE)
    for name in profiles:
        _ensure_profile_running(name)
