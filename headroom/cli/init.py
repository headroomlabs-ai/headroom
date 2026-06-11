"""Durable agent initialization commands."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any

import click

from headroom.install.models import ConfigScope, InstallPreset, RuntimeKind, SupervisorKind
from headroom.install.paths import claude_settings_path, codex_config_path, validate_profile_name
from headroom.install.planner import build_manifest
from headroom.install.providers import _apply_unix_env_scope, _apply_windows_env_scope
from headroom.install.runtime import (
    acquire_runtime_start_lock,
    resolve_headroom_command,
    runtime_status,
    start_detached_agent,
    start_persistent_docker,
    stop_runtime,
    wait_ready,
)
from headroom.install.state import load_manifest, save_manifest
from headroom.install.supervisors import start_supervisor

from .main import main

logger = logging.getLogger(__name__)

_VERBOSE_HANDLER_ATTR = "_headroom_init_verbose_handler"

_GLOBAL_PROFILE = "init-user"
_CLAUDE_HOOK_MARKER = "headroom-init-claude"
_COPILOT_HOOK_MARKER = "headroom-init-copilot"
_CODEX_HOOK_MARKER = "headroom-init-codex"
_CODEX_PROVIDER_MARKER_START = "# --- Headroom init provider ---"
_CODEX_PROVIDER_MARKER_END = "# --- end Headroom init provider ---"
_CODEX_FEATURE_MARKER_START = "# --- Headroom init features ---"
_CODEX_FEATURE_MARKER_END = "# --- end Headroom init features ---"
_SUPPORTED_TARGETS = ("claude", "copilot", "codex", "openclaw")
_LOCAL_TARGETS = {"claude", "codex"}
_GLOBAL_TARGETS = {"claude", "copilot", "codex", "openclaw"}
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


def _copilot_config_path() -> Path:
    return Path.home() / ".copilot" / "config.json"


def _codex_hooks_path(global_scope: bool) -> Path:
    return (Path.home() if global_scope else Path.cwd()) / ".codex" / "hooks.json"


def _claude_scope_path(global_scope: bool) -> Path:
    if global_scope:
        return claude_settings_path()
    return Path.cwd() / ".claude" / "settings.local.json"


def _codex_scope_path(global_scope: bool) -> Path:
    if global_scope:
        return codex_config_path()
    return Path.cwd() / ".codex" / "config.toml"


def _json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    payload = json.loads(content)
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


def _ensure_copilot_hooks(path: Path, profile: str) -> None:
    logger.debug("ensure copilot hooks: %s (profile=%s)", path, profile)
    payload = _json_file(path)
    hooks = dict(payload.get("hooks") or {}) if isinstance(payload.get("hooks"), dict) else {}
    command = f"{_hook_command('--profile', profile)} --marker {_COPILOT_HOOK_MARKER}"
    for event in ("SessionStart", "PreToolUse"):
        entries = list(hooks.get(event) or []) if isinstance(hooks.get(event), list) else []
        retained = [
            entry
            for entry in entries
            if not (
                isinstance(entry, dict) and _COPILOT_HOOK_MARKER in str(entry.get("command", ""))
            )
        ]
        retained.append({"type": "command", "command": command, "cwd": ".", "timeout": 15})
        hooks[event] = retained
    payload["hooks"] = hooks
    _write_json(path, payload)


def _replace_marker_block(
    content: str, marker_start: str, marker_end: str, block: str, *, at_root: bool = False
) -> str:
    if marker_start in content and marker_end in content:
        start = content.index(marker_start)
        end = content.index(marker_end) + len(marker_end)
        content = content[:start].rstrip() + "\n\n" + content[end:].lstrip()
    block = block.strip()
    if at_root:
        # The block carries top-level keys, so it must sit above the first table
        # header; appended after a table (e.g. [features]) TOML scopes those keys
        # into that table and Codex rejects the config (#260).
        lines = content.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                head = "\n".join(lines[:index]).rstrip()
                tail = "\n".join(lines[index:]).lstrip("\n")
                prefix = f"{head}\n\n" if head else ""
                return (f"{prefix}{block}\n\n{tail}").rstrip() + "\n"
    return (content.rstrip() + "\n\n" + block + "\n").lstrip()


def _strip_codex_init_block(content: str) -> str:
    """Remove all Headroom init-managed blocks and orphan keys from a Codex config.toml string."""
    import re

    # Remove any provider marker → end marker span, possibly repeated.
    while _CODEX_PROVIDER_MARKER_START in content and _CODEX_PROVIDER_MARKER_END in content:
        start = content.index(_CODEX_PROVIDER_MARKER_START)
        end_idx = content.index(_CODEX_PROVIDER_MARKER_END, start)
        if end_idx < start:
            break
        end = end_idx + len(_CODEX_PROVIDER_MARKER_END)
        content = content[:start].rstrip("\n") + "\n" + content[end:].lstrip("\n")

    # Remove stale unpaired markers.
    content = content.replace(_CODEX_PROVIDER_MARKER_START + "\n", "")
    content = content.replace(_CODEX_PROVIDER_MARKER_END + "\n", "")

    # Strip any orphan top-level keys that a crashed or partial write may have
    # left outside the marker block.
    content = re.sub(r'(?m)^[ \t]*model_provider[ \t]*=[ \t]*"headroom"[ \t]*\r?\n', "", content)
    content = re.sub(
        r'(?m)^[ \t]*openai_base_url[ \t]*=[ \t]*"http://127\.0\.0\.1:\d+/v1"[ \t]*\r?\n',
        "",
        content,
    )

    # Strip any orphaned [model_providers.headroom] table that is recognisably ours.
    orphan_headroom_table = re.compile(
        r"(?ms)^\[model_providers\.headroom\][^\[]*?"
        r'base_url[ \t]*=[ \t]*"http://127\.0\.0\.1:\d+/v1"[^\[]*?'
        r"(?=^\[|\Z)"
    )
    content = orphan_headroom_table.sub("", content)

    return content.lstrip("\n").rstrip() + "\n" if content.strip() else ""


def _ensure_codex_provider(path: Path, port: int) -> None:
    import re

    logger.debug("ensure codex provider block: %s (port=%s)", path, port)
    block = (
        f"{_CODEX_PROVIDER_MARKER_START}\n"
        'model_provider = "headroom"\n'
        f'openai_base_url = "http://127.0.0.1:{port}/v1"\n\n'
        "[model_providers.headroom]\n"
        'name = "Headroom init proxy"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        "supports_websockets = true\n"
        f"{_CODEX_PROVIDER_MARKER_END}"
    )
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    # init owns model_provider/openai_base_url: drop any prior assignment (any
    # value, including one an older version mis-scoped under a table) so we
    # replace it instead of emitting a duplicate top-level key (#260).
    content = re.sub(r"(?m)^[ \t]*model_provider[ \t]*=.*\r?\n", "", content)
    content = re.sub(r"(?m)^[ \t]*openai_base_url[ \t]*=.*\r?\n", "", content)
    # The provider block carries top-level keys (model_provider, openai_base_url),
    # so it must land at the document root rather than after a trailing table (#260).
    content = _replace_marker_block(
        content, _CODEX_PROVIDER_MARKER_START, _CODEX_PROVIDER_MARKER_END, block, at_root=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _ensure_codex_feature_flag(path: Path) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    if _CODEX_FEATURE_MARKER_START in content and _CODEX_FEATURE_MARKER_END in content:
        block = f"{_CODEX_FEATURE_MARKER_START}\ncodex_hooks = true\n{_CODEX_FEATURE_MARKER_END}"
        content = _replace_marker_block(
            content,
            _CODEX_FEATURE_MARKER_START,
            _CODEX_FEATURE_MARKER_END,
            block,
        )
    elif "[features]" in content:
        lines = content.splitlines()
        inserted = False
        for index, line in enumerate(lines):
            if line.strip() != "[features]":
                continue
            section_end = index + 1
            while section_end < len(lines) and not (
                lines[section_end].startswith("[") and lines[section_end].endswith("]")
            ):
                if "codex_hooks" in lines[section_end]:
                    inserted = True
                    break
                section_end += 1
            if not inserted:
                lines[index + 1 : index + 1] = [
                    _CODEX_FEATURE_MARKER_START,
                    "codex_hooks = true",
                    _CODEX_FEATURE_MARKER_END,
                ]
                inserted = True
            break
        content = "\n".join(lines).rstrip() + "\n"
        if not inserted:
            content = (
                content.rstrip()
                + "\n\n[features]\n"
                + _CODEX_FEATURE_MARKER_START
                + "\n"
                + "codex_hooks = true\n"
                + _CODEX_FEATURE_MARKER_END
                + "\n"
            )
    else:
        content = (
            content.rstrip()
            + "\n\n[features]\n"
            + _CODEX_FEATURE_MARKER_START
            + "\n"
            + "codex_hooks = true\n"
            + _CODEX_FEATURE_MARKER_END
            + "\n"
        ).lstrip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _ensure_codex_hooks(path: Path, profile: str) -> None:
    logger.debug("ensure codex hooks: %s (profile=%s)", path, profile)
    command = f"{_hook_command('--profile', profile)} --marker {_CODEX_HOOK_MARKER}"
    payload = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume",
                    "hooks": [{"type": "command", "command": command, "timeout": 15}],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": command, "timeout": 15}],
                }
            ],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _manifest_changed(
    existing: Any,
    *,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    memory: bool,
) -> bool:
    return any(
        [
            getattr(existing, "port", port) != port,
            getattr(existing, "backend", backend) != backend,
            getattr(existing, "anyllm_provider", anyllm_provider) != anyllm_provider,
            getattr(existing, "region", region) != region,
            getattr(existing, "memory_enabled", memory) != memory,
        ]
    )


def _ensure_runtime_manifest(
    *,
    global_scope: bool,
    targets: list[str],
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    memory: bool,
) -> str:
    profile = _runtime_profile(global_scope)
    existing = load_manifest(profile)
    merged_targets = sorted(set(existing.targets if existing else []).union(targets))
    manifest = build_manifest(
        profile=profile,
        preset=InstallPreset.PERSISTENT_TASK.value,
        runtime_kind=RuntimeKind.PYTHON.value,
        scope=ConfigScope.USER.value,
        provider_mode="manual",
        targets=merged_targets,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode="token",
        memory_enabled=memory,
        telemetry_enabled=True,
        image="ghcr.io/chopratejas/headroom:latest",
    )
    manifest.supervisor_kind = SupervisorKind.NONE.value
    manifest.artifacts = []
    manifest.mutations = existing.mutations if existing else []
    if existing is not None and _manifest_changed(
        existing,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        memory=memory,
    ):
        try:
            stop_runtime(existing)
        except Exception:
            pass
    save_manifest(manifest)
    return profile


def _env_manifest(values: dict[str, str]) -> Any:
    return build_manifest(
        profile="init-env",
        preset=InstallPreset.PERSISTENT_TASK.value,
        runtime_kind=RuntimeKind.PYTHON.value,
        scope=ConfigScope.USER.value,
        provider_mode="manual",
        targets=["copilot"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=False,
        telemetry_enabled=True,
        image="ghcr.io/chopratejas/headroom:latest",
    )


def _apply_user_env(values: dict[str, str]) -> None:
    manifest = _env_manifest(values)
    manifest.base_env = {}
    manifest.tool_envs = {"copilot": values}
    scope = "windows" if os.name == "nt" else "unix"
    logger.debug("apply user env scope=%s keys=%s", scope, sorted(values.keys()))
    if os.name == "nt":
        _apply_windows_env_scope(manifest)
    else:
        _apply_unix_env_scope(manifest)


def _resolve_copilot_env(port: int, backend: str) -> dict[str, str]:
    if backend == "anthropic":
        return {
            "COPILOT_PROVIDER_TYPE": "anthropic",
            "COPILOT_PROVIDER_BASE_URL": f"http://127.0.0.1:{port}",
        }
    return {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "COPILOT_PROVIDER_WIRE_API": "completions",
    }


def _marketplace_source() -> str:
    override = os.environ.get("HEADROOM_MARKETPLACE_SOURCE")
    if override:
        return override
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / ".claude-plugin" / "marketplace.json").exists():
        return str(repo_root)
    return "chopratejas/headroom"


def _run_checked(command: list[str], *, action: str) -> None:
    logger.debug("subprocess [%s]: %s", action, _command_string(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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


def _install_copilot_marketplace() -> None:
    copilot_bin = shutil.which("copilot")
    if not copilot_bin:
        raise click.ClickException("'copilot' not found in PATH. Install GitHub Copilot CLI first.")
    source = _marketplace_source()
    _run_checked(
        [copilot_bin, "plugin", "marketplace", "add", source],
        action="copilot marketplace add",
    )
    _run_checked(
        [copilot_bin, "plugin", "install", "headroom@headroom-marketplace"],
        action="copilot plugin install",
    )


def _ensure_profile_running(profile: str) -> None:
    manifest = load_manifest(profile)
    if manifest is None:
        return
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
    """Build the error message shown when no in-scope targets were detected.

    Lists every agent that was probed, what ``shutil.which`` returned, and
    confirms how to proceed explicitly — including that the ``-g`` / ``--global``
    flag the user tried is still valid.
    """

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


def _init_copilot(*, global_scope: bool, profile: str, port: int, backend: str) -> None:
    if not global_scope:
        raise click.ClickException(
            "Copilot durable init currently requires -g (current-user scope)."
        )
    _ensure_copilot_hooks(_copilot_config_path(), profile)
    _apply_user_env(_resolve_copilot_env(port, backend))
    _install_copilot_marketplace()
    click.echo("Configured GitHub Copilot CLI (user scope).")
    click.echo("Restart Copilot CLI to activate Headroom hooks and provider routing.")


def _init_codex(*, global_scope: bool, profile: str, port: int) -> None:
    config_path = _codex_scope_path(global_scope)
    _ensure_codex_provider(config_path, port)
    _ensure_codex_feature_flag(config_path)
    _ensure_codex_hooks(_codex_hooks_path(global_scope), profile)
    click.echo(f"Configured Codex ({'user' if global_scope else 'local'} scope).")
    if os.name == "nt":
        click.echo(
            "Codex hooks are currently disabled upstream on Windows; provider routing was still installed."
        )
    click.echo("Restart Codex to activate Headroom configuration.")


def _init_openclaw(*, global_scope: bool, port: int) -> None:
    if not global_scope:
        raise click.ClickException(
            "OpenClaw durable init currently requires -g (current-user scope)."
        )
    command = [*resolve_headroom_command(), "wrap", "openclaw", "--proxy-port", str(port)]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_init_targets(
    *,
    targets: list[str],
    global_scope: bool,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    memory: bool,
) -> None:
    logger.debug(
        "run_init_targets: targets=%s global_scope=%s port=%s backend=%s memory=%s",
        targets,
        global_scope,
        port,
        backend,
        memory,
    )
    runtime_targets = [target for target in targets if target != "openclaw"]
    profile = _ensure_runtime_manifest(
        global_scope=global_scope,
        targets=runtime_targets,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        memory=memory,
    )
    logger.debug("run_init_targets: using profile=%s", profile)
    for target in targets:
        logger.debug("run_init_targets: dispatching -> %s", target)
        if target == "claude":
            _init_claude(global_scope=global_scope, profile=profile, port=port)
        elif target == "copilot":
            _init_copilot(global_scope=global_scope, profile=profile, port=port, backend=backend)
        elif target == "codex":
            _init_codex(global_scope=global_scope, profile=profile, port=port)
        elif target == "openclaw":
            _init_openclaw(global_scope=global_scope, port=port)

    # Register the headroom MCP server with every targeted agent that has
    # a registrar implemented. Wave 1 covers Claude Code; subsequent waves
    # add Cursor / Codex / Continue / Cline / Windsurf / Goose without
    # touching the call sites.
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
@click.option("--port", default=8787, type=int, show_default=True, help="Headroom proxy port.")
@click.option("--backend", default="anthropic", show_default=True, help="Proxy backend.")
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backends.")
@click.option("--region", default=None, help="Cloud region for Bedrock / Vertex style backends.")
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
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    memory: bool,
    verbose: bool,
) -> None:
    """Install durable Headroom integrations for supported agents."""
    if verbose:
        _enable_verbose_logging()
    logger.debug(
        "init: global_scope=%s port=%s backend=%s anyllm_provider=%s region=%s memory=%s "
        "invoked_subcommand=%s",
        global_scope,
        port,
        backend,
        anyllm_provider,
        region,
        memory,
        ctx.invoked_subcommand,
    )
    if ctx.invoked_subcommand is not None:
        ctx.obj = {
            "global_scope": global_scope,
            "port": port,
            "backend": backend,
            "anyllm_provider": anyllm_provider,
            "region": region,
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
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
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
        backend=str(_ctx_value(ctx, "backend") or "anthropic"),
        anyllm_provider=_ctx_value(ctx, "anyllm_provider"),
        region=_ctx_value(ctx, "region"),
        memory=bool(_ctx_value(ctx, "memory")),
    )


@init.command("copilot")
@click.pass_context
def init_copilot(ctx: click.Context) -> None:
    """Install GitHub Copilot CLI durable hooks and provider routing."""
    _run_init_targets(
        targets=["copilot"],
        global_scope=bool(_ctx_value(ctx, "global_scope")),
        port=int(_ctx_value(ctx, "port") or 8787),
        backend=str(_ctx_value(ctx, "backend") or "anthropic"),
        anyllm_provider=_ctx_value(ctx, "anyllm_provider"),
        region=_ctx_value(ctx, "region"),
        memory=bool(_ctx_value(ctx, "memory")),
    )


@init.command("codex")
@click.pass_context
def init_codex(ctx: click.Context) -> None:
    """Install Codex durable hooks and provider routing."""
    _run_init_targets(
        targets=["codex"],
        global_scope=bool(_ctx_value(ctx, "global_scope")),
        port=int(_ctx_value(ctx, "port") or 8787),
        backend=str(_ctx_value(ctx, "backend") or "anthropic"),
        anyllm_provider=_ctx_value(ctx, "anyllm_provider"),
        region=_ctx_value(ctx, "region"),
        memory=bool(_ctx_value(ctx, "memory")),
    )


@init.command("openclaw")
@click.pass_context
def init_openclaw(ctx: click.Context) -> None:
    """Install the durable OpenClaw Headroom plugin."""
    _run_init_targets(
        targets=["openclaw"],
        global_scope=bool(_ctx_value(ctx, "global_scope")),
        port=int(_ctx_value(ctx, "port") or 8787),
        backend=str(_ctx_value(ctx, "backend") or "anthropic"),
        anyllm_provider=_ctx_value(ctx, "anyllm_provider"),
        region=_ctx_value(ctx, "region"),
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
    profiles: list[str] = []
    if profile:
        profiles.append(profile)
    else:
        local_profile = _local_profile()
        if load_manifest(local_profile) is not None:
            profiles.append(local_profile)
        elif load_manifest(_GLOBAL_PROFILE) is not None:
            profiles.append(_GLOBAL_PROFILE)
    for name in profiles:
        _ensure_profile_running(name)
