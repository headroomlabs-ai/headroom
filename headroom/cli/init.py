"""Durable agent initialization commands."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal, cast

from headroom._subprocess import run
from headroom._version import __version__ as _HEADROOM_VERSION

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import click

from headroom.install.models import (
    ArtifactRecord,
    ConfigScope,
    InstallPreset,
    RuntimeKind,
    SupervisorKind,
)
from headroom.install.paths import (
    claude_settings_path,
    codex_config_path,
    manifest_path,
    profile_root,
    unix_ensure_script_path,
    unix_run_script_path,
    validate_profile_name,
    windows_ensure_cmd_path,
    windows_ensure_script_path,
    windows_run_cmd_path,
    windows_run_script_path,
)
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
from headroom.install.state import ManifestError, delete_manifest, load_manifest, save_manifest
from headroom.install.supervisors import (
    install_supervisor,
    remove_supervisor,
    start_supervisor,
)
from headroom.providers.claude import TOOL_SEARCH_DEFAULT, TOOL_SEARCH_ENV
from headroom.providers.claude.runtime import TOOL_SEARCH_FOUNDRY_DEFAULT
from headroom.providers.codex.install import codex_uses_chatgpt_auth
from headroom.providers.codex.threads import retag_to_headroom
from headroom.providers.pi_extension import (
    ensure_extension_config,
    ensure_host_package,
    extension_config_path,
    extension_release_version,
    inspect_host_package,
    remove_owned_extension_config,
    remove_owned_host_package,
)

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
_SUPPORTED_TARGETS = ("claude", "copilot", "codex", "openclaw", "pi", "omp")
_LOCAL_TARGETS = {"claude", "codex"}
_GLOBAL_TARGETS = {"claude", "copilot", "codex", "openclaw", "pi", "omp"}
_NATIVE_EXTENSION_TARGETS = {"pi", "omp"}
_STARTUP_READY_TIMEOUT_SECONDS = 15
_TOML_TABLE_HEADER_RE = re.compile(r"^[ \t]*(?:\[\[[^\]\r\n]+\]\]|\[[^\]\r\n]+\])[ \t]*(?:#.*)?$")
_TOML_FEATURES_NAME_RE = r"(?:features|\"features\"|'features')"
_TOML_CODEX_HOOKS_NAME_RE = r"(?:codex_hooks|\"codex_hooks\"|'codex_hooks')"
_CODEX_FEATURES_TABLE_RE = re.compile(
    rf"^[ \t]*\[[ \t]*{_TOML_FEATURES_NAME_RE}[ \t]*\][ \t]*(?:#.*)?$"
)
_CODEX_FEATURES_DOTTED_LEGACY_RE = re.compile(
    rf"^[ \t]*{_TOML_FEATURES_NAME_RE}[ \t]*\.[ \t]*{_TOML_CODEX_HOOKS_NAME_RE}[ \t]*="
)
_CODEX_FEATURES_LEGACY_KEY_RE = re.compile(rf"^[ \t]*{_TOML_CODEX_HOOKS_NAME_RE}[ \t]*=")


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
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as e:
        # This is a user-owned file (e.g. ~/.claude/settings.json or Codex's
        # hooks.json) that the callers read-merge-write. Returning {} would make
        # the following _write_json overwrite it, silently discarding the user's
        # settings; letting the raw JSONDecodeError propagate crashes `headroom
        # init` with a traceback. Abort with an actionable message so the user
        # can fix the JSON (or move it aside) without losing it.
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
    content = _remove_marker_block(content, marker_start, marker_end)
    block = block.strip()
    if at_root:
        # The block carries top-level keys, so it must sit above the first table
        # header; appended after a table (e.g. [features]) TOML scopes those keys
        # into that table and Codex rejects the config (#260).
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if _TOML_TABLE_HEADER_RE.search(line):
                head = "\n".join(lines[:index]).rstrip()
                tail = "\n".join(lines[index:]).lstrip("\n")
                prefix = f"{head}\n\n" if head else ""
                return (f"{prefix}{block}\n\n{tail}").rstrip() + "\n"
    return (content.rstrip() + "\n\n" + block + "\n").lstrip()


def _remove_marker_block(content: str, marker_start: str, marker_end: str) -> str:
    if marker_start not in content or marker_end not in content:
        return content
    start = content.index(marker_start)
    end = content.index(marker_end) + len(marker_end)
    return content[:start].rstrip() + "\n\n" + content[end:].lstrip()


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
    # Emit requires_openai_auth only for ChatGPT-OAuth users (restores the
    # account menu); omitting it for API-key users avoids forcing an OAuth
    # login (#406).
    requires_openai_auth = (
        "requires_openai_auth = true\n"
        if codex_uses_chatgpt_auth(path.parent / "auth.json")
        else ""
    )
    block = (
        f"{_CODEX_PROVIDER_MARKER_START}\n"
        'model_provider = "headroom"\n'
        f'openai_base_url = "http://127.0.0.1:{port}/v1"\n\n'
        "[model_providers.headroom]\n"
        'name = "Headroom init proxy"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        "supports_websockets = true\n"
        f"{requires_openai_auth}"
        f"{_CODEX_PROVIDER_MARKER_END}"
    )
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    # init owns the ROOT-level model_provider/openai_base_url: drop any prior
    # root assignment so we replace it instead of emitting a duplicate top-level
    # key (#260). Scope the strip to the document root (everything before the
    # first table header) -- these keys also appear legitimately inside
    # [profiles.*] tables as per-profile overrides, and stripping them there
    # silently reroutes the user's profiles to the injected "headroom" default.
    _first_table = re.search(r"(?m)^[ \t]*\[", content)
    _split = _first_table.start() if _first_table else len(content)
    root, rest = content[:_split], content[_split:]
    root = re.sub(r"(?m)^[ \t]*model_provider[ \t]*=.*\r?\n", "", root)
    root = re.sub(r"(?m)^[ \t]*openai_base_url[ \t]*=.*\r?\n", "", root)
    content = root + rest
    # The provider block carries top-level keys (model_provider, openai_base_url),
    # so it must land at the document root rather than after a trailing table (#260).
    content = _replace_marker_block(
        content, _CODEX_PROVIDER_MARKER_START, _CODEX_PROVIDER_MARKER_END, block, at_root=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    # Codex filters its history menu by the active model_provider, so existing
    # native threads vanish once we switch to "headroom". Retag them to match the
    # active provider so the history stays whole (#961), mirroring the install
    # (providers.codex.install) and wrap (cli.wrap) paths. The revert direction is
    # handled by `headroom unwrap codex`.
    retag_to_headroom(path.parent)


def _codex_feature_block() -> str:
    return f"{_CODEX_FEATURE_MARKER_START}\nhooks = true\n{_CODEX_FEATURE_MARKER_END}"


def _codex_dotted_feature_block() -> str:
    return f"{_CODEX_FEATURE_MARKER_START}\nfeatures.hooks = true\n{_CODEX_FEATURE_MARKER_END}"


def _codex_features_table_index(lines: list[str]) -> int | None:
    return next(
        (index for index, line in enumerate(lines) if _CODEX_FEATURES_TABLE_RE.search(line)),
        None,
    )


def _codex_features(content: str) -> dict[str, Any] | None:
    if not content.strip():
        return None
    try:
        parsed = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return None
    features = parsed.get("features")
    return features if isinstance(features, dict) else None


def _codex_features_has_hooks(content: str) -> bool:
    features = _codex_features(content)
    if features is None:
        # Keep init resilient for already-invalid user configs; this fallback
        # only needs to avoid adding a second obvious hooks line.
        lines = content.splitlines()
        features_index = _codex_features_table_index(lines)
        if features_index is None:
            return False
        for line in lines[features_index + 1 :]:
            if _TOML_TABLE_HEADER_RE.search(line):
                break
            if re.search(r"^[ \t]*hooks[ \t]*=", line):
                return True
        return False

    return "hooks" in features


def _strip_codex_legacy_feature_flag(content: str) -> str:
    lines = content.splitlines(keepends=True)
    retained: list[str] = []
    in_features = False
    in_root = True

    for line in lines:
        if _TOML_TABLE_HEADER_RE.search(line):
            in_root = False
            in_features = bool(_CODEX_FEATURES_TABLE_RE.search(line))
            retained.append(line)
            continue
        if (in_root and _CODEX_FEATURES_DOTTED_LEGACY_RE.search(line)) or (
            in_features and _CODEX_FEATURES_LEGACY_KEY_RE.search(line)
        ):
            continue
        retained.append(line)

    return "".join(retained)


def _ensure_codex_feature_flag(path: Path) -> None:
    """Ensure Codex's ``[features].hooks`` flag is enabled in config.toml.

    ``hooks`` is the canonical key. ``codex_hooks`` was the original key name and
    still resolves as a deprecated alias, but Codex >= 0.129 emits a deprecation
    warning for it (renamed in openai/codex#20522). Any legacy
    ``[features].codex_hooks`` line is removed, whether inside or outside our
    marker block, so a migrated config drops the deprecated key and never
    collides with a duplicate ``hooks`` key. A user-managed ``hooks`` value
    outside our marker block is left untouched.
    """
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    # Drop the deprecated alias key from [features]. Mirrors the top-level key
    # cleanup in _ensure_codex_provider (#260) so re-running init migrates a
    # legacy config rather than producing a duplicate `hooks` key, while leaving
    # unrelated user tables untouched.
    content = _strip_codex_legacy_feature_flag(content)
    if _CODEX_FEATURE_MARKER_START in content and _CODEX_FEATURE_MARKER_END in content:
        # init owns its marker block; remove it first, then reinsert under the
        # correct TOML scope below.
        content = _remove_marker_block(
            content, _CODEX_FEATURE_MARKER_START, _CODEX_FEATURE_MARKER_END
        )

    if _codex_features_has_hooks(content):
        # A user-managed `[features].hooks` key already exists outside our
        # marker block; respect their value. Clearing the legacy key above was
        # the only work.
        pass
    else:
        lines = content.splitlines()
        features_index = _codex_features_table_index(lines)
        if features_index is not None:
            # Leading blank line matches the normalisation _replace_marker_block
            # applies on later runs, so re-running init is byte-idempotent.
            lines[features_index + 1 : features_index + 1] = [
                "",
                *_codex_feature_block().splitlines(),
            ]
            content = "\n".join(lines).rstrip() + "\n"
        elif _codex_features(content) is not None:
            # The user expressed [features] via dotted keys, so adding a new
            # table would duplicate it. Keep this key at the document root.
            content = _replace_marker_block(
                content,
                _CODEX_FEATURE_MARKER_START,
                _CODEX_FEATURE_MARKER_END,
                _codex_dotted_feature_block(),
                at_root=True,
            )
        else:
            content = (
                content.rstrip() + "\n\n[features]\n\n" + _codex_feature_block() + "\n"
            ).lstrip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _ensure_codex_hooks(path: Path, profile: str) -> None:
    logger.debug("ensure codex hooks: %s (profile=%s)", path, profile)
    command = f"{_hook_command('--profile', profile)} --marker {_CODEX_HOOK_MARKER}"
    # Read-merge-write rather than overwrite: the previous version wrote a fresh
    # payload wholesale, destroying any user-managed hooks (and other top-level
    # keys) in codex hooks.json. Merge per event and dedup on the Headroom
    # marker, matching _ensure_claude_hooks / _ensure_copilot_hooks.
    payload = _json_file(path)
    hooks = dict(payload.get("hooks") or {}) if isinstance(payload.get("hooks"), dict) else {}
    for event, matcher in (
        ("SessionStart", "startup|resume"),
        ("PreToolUse", "Bash"),
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
                and _CODEX_HOOK_MARKER in str(item.get("command"))
                for item in hook_items
            )
            if not has_headroom:
                retained.append(entry)
        retained.append(
            {
                "matcher": matcher,
                "hooks": [{"type": "command", "command": command, "timeout": 15}],
            }
        )
        hooks[event] = retained
    payload["hooks"] = hooks
    _write_json(path, payload)


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


def _build_runtime_manifest(
    *,
    profile: str,
    existing: Any | None,
    targets: list[str],
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    memory: bool,
) -> Any:
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
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )
    # The planner intentionally recognizes only proxy/provider targets. Native
    # hosts share this manifest, so restore the merged ownership list afterward.
    manifest.targets = merged_targets
    manifest.supervisor_kind = (
        getattr(existing, "supervisor_kind", SupervisorKind.NONE.value)
        if existing
        else SupervisorKind.NONE.value
    )
    manifest.artifacts = list(getattr(existing, "artifacts", [])) if existing else []
    manifest.mutations = existing.mutations if existing else []
    return manifest


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
    try:
        existing = load_manifest(profile)
    except ManifestError as e:
        # Keep the legacy init behavior; native init preflights corrupt ownership
        # state before calling this path and never overwrites it.
        click.echo(f"Warning: {e}; overwriting.")
        existing = None
    manifest = _build_runtime_manifest(
        profile=profile,
        existing=existing,
        targets=targets,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        memory=memory,
    )
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


def _artifact_key(artifact: ArtifactRecord) -> tuple[str, str]:
    return artifact.kind, artifact.path


def _upsert_artifacts(manifest: Any, records: list[ArtifactRecord]) -> None:
    replacements = {_artifact_key(record): record for record in records}
    manifest.artifacts = [
        artifact for artifact in manifest.artifacts if _artifact_key(artifact) not in replacements
    ] + list(replacements.values())


def _snapshot_manifest(profile: str) -> tuple[bool, bytes, Any | None]:
    path = manifest_path(profile)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return False, b"", load_manifest(profile)
    try:
        manifest = load_manifest(profile)
    except ManifestError as exc:
        raise click.ClickException(
            f"Native Pi/OMP init cannot use corrupt ownership state: {exc}. "
            "Fix or move the manifest, then retry."
        ) from exc
    return True, content, manifest


def _restore_manifest_snapshot(profile: str, existed: bool, content: bytes) -> None:
    path = manifest_path(profile)
    if not existed:
        root = profile_root(profile)
        shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            raise click.ClickException(
                f"Could not remove newly created ownership state for profile {profile}."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if path.read_bytes() != content:
        raise click.ClickException(
            f"Could not verify restored ownership state for profile {profile}."
        )


def _save_manifest_verified(manifest: Any) -> None:
    save_manifest(manifest)
    persisted = load_manifest(manifest.profile)
    if (
        persisted is None
        or persisted.targets != manifest.targets
        or persisted.artifacts != manifest.artifacts
    ):
        raise click.ClickException(
            f"Could not verify persisted ownership state for profile {manifest.profile}."
        )


def _task_file_paths(manifest: Any) -> set[Path]:
    paths = {
        Path(artifact.path)
        for artifact in manifest.artifacts
        if artifact.kind in {"script", "service-unit", "cron", "plist"}
    }
    if sys.platform == "win32":
        paths.update(
            {
                windows_run_script_path(manifest.profile),
                windows_run_cmd_path(manifest.profile),
                windows_ensure_script_path(manifest.profile),
                windows_ensure_cmd_path(manifest.profile),
            }
        )
    else:
        paths.update(
            {
                unix_run_script_path(manifest.profile),
                unix_ensure_script_path(manifest.profile),
            }
        )
    return paths


def _snapshot_files(paths: set[Path]) -> dict[Path, tuple[bytes, int] | None]:
    snapshot: dict[Path, tuple[bytes, int] | None] = {}
    for path in paths:
        try:
            snapshot[path] = (path.read_bytes(), path.stat().st_mode & 0o777)
        except FileNotFoundError:
            snapshot[path] = None
    return snapshot


def _restore_files(snapshot: dict[Path, tuple[bytes, int] | None]) -> None:
    for path, prior in snapshot.items():
        if prior is None:
            path.unlink(missing_ok=True)
            continue
        content, mode = prior
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(mode)


def _snapshot_extension_config() -> tuple[bool, bytes, int | None]:
    path = extension_config_path().absolute()
    try:
        return True, path.read_bytes(), path.stat().st_mode & 0o777
    except FileNotFoundError:
        return False, b"", None


def _expected_extension_config(snapshot: tuple[bool, bytes, int | None], port: int) -> bytes | None:
    _existed, content, _mode = snapshot
    try:
        payload = json.loads(content) if content else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    payload["baseUrl"] = f"http://127.0.0.1:{port}"
    return (json.dumps(payload, indent=2) + "\n").encode()


def _restore_extension_config_snapshot(
    snapshot: tuple[bool, bytes, int | None],
    *,
    expected_managed: bytes | None,
) -> None:
    existed, content, mode = snapshot
    path = extension_config_path().absolute()
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None

    if current == content if existed else current is None:
        return
    if expected_managed is None or current != expected_managed:
        raise click.ClickException(
            f"Pi extension config at {path} changed during the failed init; "
            "the concurrent file was preserved."
        )
    if not existed:
        path.unlink()
        return
    path.write_bytes(content)
    if mode is not None:
        path.chmod(mode)


def _command_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value or "").encode("utf-8", errors="surrogateescape")


def _scheduler_snapshot_error(platform: str, stderr: Any) -> click.ClickException:
    detail = _command_bytes(stderr).decode("utf-8", errors="replace").strip()
    return click.ClickException(
        f"Could not snapshot {platform} scheduler state: {detail or 'unknown query failure'}."
    )


def _snapshot_scheduler(manifest: Any) -> tuple[str, Any]:
    if sys.platform.startswith("linux"):
        result = run(["crontab", "-l"], capture_output=True)
        if result.returncode == 0:
            return "linux", (True, _command_bytes(result.stdout))
        detail = _command_bytes(result.stderr).lower()
        if result.returncode == 1 and b"no crontab for" in detail:
            return "linux", (False, b"")
        raise _scheduler_snapshot_error("Linux crontab", result.stderr)
    if sys.platform == "darwin":
        plist = Path.home() / "Library" / "LaunchAgents" / f"com.headroom.{manifest.profile}.plist"
        try:
            content = plist.read_bytes()
        except FileNotFoundError:
            content = None
        domain = f"gui/{os.getuid()}"
        result = run(
            ["launchctl", "print", f"{domain}/com.headroom.{manifest.profile}"],
            capture_output=True,
        )
        if result.returncode == 0:
            loaded = True
        else:
            detail = _command_bytes(result.stderr).lower()
            if result.returncode == 113 and b"could not find service" in detail:
                loaded = False
            else:
                raise _scheduler_snapshot_error("macOS launchd", result.stderr)
        return "darwin", (plist, content, loaded, domain)
    if sys.platform.startswith("win"):
        tasks: dict[str, bytes | None] = {}
        for suffix in ("startup", "health"):
            name = f"{manifest.service_name}-{suffix}"
            result = run(
                ["schtasks", "/Query", "/TN", name, "/XML"],
                capture_output=True,
            )
            if result.returncode == 0:
                tasks[name] = _command_bytes(result.stdout)
                continue
            detail = _command_bytes(result.stderr).lower()
            if result.returncode == 1 and b"cannot find the file specified" in detail:
                tasks[name] = None
                continue
            raise _scheduler_snapshot_error("Windows Task Scheduler", result.stderr)
        return "windows", tasks
    return "none", None


def _restore_windows_task(name: str, xml: bytes) -> None:
    temp = Path(tempfile.mkstemp(suffix=".xml")[1])
    try:
        temp.write_bytes(xml)
        run(["schtasks", "/Create", "/TN", name, "/XML", str(temp), "/F"], check=True)
    finally:
        temp.unlink(missing_ok=True)


def _restore_scheduler(snapshot: tuple[str, Any]) -> None:
    platform, state = snapshot
    if platform == "linux":
        existed, content = state
        if existed:
            run(["crontab", "-"], input=content, check=True)
        else:
            result = run(["crontab", "-r"], capture_output=True)
            if result.returncode != 0:
                detail = _command_bytes(result.stderr).lower()
                if result.returncode != 1 or b"no crontab for" not in detail:
                    raise _scheduler_snapshot_error("Linux crontab rollback", result.stderr)
    elif platform == "darwin":
        plist, content, was_loaded, domain = state
        result = run(
            ["launchctl", "bootout", f"{domain}/{plist.stem}"],
            capture_output=True,
        )
        if result.returncode != 0:
            detail = _command_bytes(result.stderr).lower()
            if result.returncode != 3 or b"no such process" not in detail:
                raise _scheduler_snapshot_error("macOS launchd rollback", result.stderr)
        if content is None:
            plist.unlink(missing_ok=True)
        else:
            plist.parent.mkdir(parents=True, exist_ok=True)
            plist.write_bytes(content)
        if was_loaded and content is not None:
            run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    elif platform == "windows":
        for name, xml in state.items():
            if xml is None:
                result = run(
                    ["schtasks", "/Delete", "/TN", name, "/F"],
                    capture_output=True,
                )
                if result.returncode != 0:
                    detail = _command_bytes(result.stderr).lower()
                    if result.returncode != 1 or b"cannot find the file specified" not in detail:
                        raise _scheduler_snapshot_error(
                            "Windows Task Scheduler rollback", result.stderr
                        )
            else:
                _restore_windows_task(name, xml)


def _restore_native_package(
    *,
    host: Literal["pi", "omp"],
    binary: str,
    previous: Any,
    current_artifact: ArtifactRecord | None,
) -> None:
    if current_artifact is None:
        return
    if previous is None:
        result = remove_owned_host_package(host, binary, current_artifact)
        if result == "preserved":
            raise click.ClickException(f"Could not roll back the {host} package install.")
        return
    ensure_host_package(host, binary, previous.version, current_artifact)


def _wait_runtime_status(manifest: Any, expected: str, timeout_seconds: int = 10) -> bool:
    for _ in range(timeout_seconds):
        if runtime_status(manifest) == expected:
            return True
        time.sleep(1)
    return runtime_status(manifest) == expected


def _wait_runtime_transition(manifest: Any, expected: str, before: str) -> bool:
    current = runtime_status(manifest)
    # stop_runtime removes the PID file before SIGTERM is necessarily observed,
    # so a first "stopped" result after a running snapshot is not authoritative.
    if expected == "stopped" and before == "running" and current == "stopped":
        time.sleep(1)
        return runtime_status(manifest) == "stopped"
    if current == expected:
        return True
    return _wait_runtime_status(manifest, expected)


def _start_profile_strict_locked(manifest: Any) -> None:
    if not getattr(manifest, "preset", None):
        start_detached_agent(manifest.profile)
        return
    if runtime_status(manifest) != "running":
        start_detached_agent(manifest.profile)
    if not wait_ready(manifest, timeout_seconds=45):
        raise click.ClickException(
            f"Headroom runtime for profile {manifest.profile} did not become ready."
        )


def _start_profile_strict(manifest: Any) -> None:
    with acquire_runtime_start_lock(manifest.profile) as acquired:
        if not acquired:
            raise click.ClickException(
                f"Headroom runtime for profile {manifest.profile} is already being started."
            )
        _start_profile_strict_locked(manifest)


def _restore_runtime_state(
    manifest: Any, status: str, was_ready: bool, *, assume_start_lock: bool = False
) -> None:
    if getattr(manifest, "preset", None):
        current_status = runtime_status(manifest)
        stop_runtime(manifest)
        if not _wait_runtime_transition(manifest, "stopped", current_status):
            raise click.ClickException(
                f"Could not restore stopped runtime state for profile {manifest.profile}."
            )
    if status == "stopped":
        return
    if status != "running":
        raise click.ClickException(
            f"Cannot restore unknown runtime state {status!r} for profile {manifest.profile}."
        )
    if was_ready:
        if assume_start_lock:
            _start_profile_strict_locked(manifest)
        else:
            _start_profile_strict(manifest)
        return
    start_detached_agent(manifest.profile)
    if not _wait_runtime_status(manifest, "running"):
        raise click.ClickException(
            f"Could not restore running runtime state for profile {manifest.profile}."
        )


def _init_native_hosts(
    *,
    hosts: list[tuple[Literal["pi", "omp"], str]],
    release: str,
    manifest: Any,
    manifest_snapshot: tuple[bool, bytes, Any | None],
    port: int,
) -> None:
    _manifest_existed, _manifest_bytes, previous_manifest = manifest_snapshot
    package_states: dict[str, Any] = {
        host: inspect_host_package(host, binary) for host, binary in hosts
    }
    config_snapshot = _snapshot_extension_config()
    expected_config_bytes = _expected_extension_config(config_snapshot, port)
    task_files_snapshot = _snapshot_files(_task_file_paths(manifest))
    scheduler_snapshot = _snapshot_scheduler(manifest)
    prior_runtime_status = (
        runtime_status(previous_manifest)
        if previous_manifest is not None and getattr(previous_manifest, "preset", None)
        else "stopped"
    )
    prior_runtime_ready = bool(
        previous_manifest is not None
        and prior_runtime_status == "running"
        and wait_ready(previous_manifest, timeout_seconds=1)
    )
    previous_config = next(
        (
            artifact
            for artifact in (previous_manifest.artifacts if previous_manifest else [])
            if artifact.kind == "pi-extension-config"
        ),
        None,
    )
    with acquire_runtime_start_lock(manifest.profile) as acquired:
        if not acquired:
            raise click.ClickException(f"Profile {manifest.profile} is already being initialized.")
        _init_native_hosts_locked(
            hosts=hosts,
            release=release,
            manifest=manifest,
            manifest_snapshot=manifest_snapshot,
            port=port,
            package_states=package_states,
            config_snapshot=config_snapshot,
            expected_config_bytes=expected_config_bytes,
            task_files_snapshot=task_files_snapshot,
            scheduler_snapshot=scheduler_snapshot,
            prior_runtime_status=prior_runtime_status,
            prior_runtime_ready=prior_runtime_ready,
            previous_config=previous_config,
        )


def _init_native_hosts_locked(
    *,
    hosts: list[tuple[Literal["pi", "omp"], str]],
    release: str,
    manifest: Any,
    manifest_snapshot: tuple[bool, bytes, Any | None],
    port: int,
    package_states: dict[str, Any],
    config_snapshot: tuple[bool, bytes, int | None],
    expected_config_bytes: bytes | None,
    task_files_snapshot: dict[Path, tuple[bytes, int] | None],
    scheduler_snapshot: tuple[str, Any],
    prior_runtime_status: str,
    prior_runtime_ready: bool,
    previous_config: ArtifactRecord | None,
) -> None:
    manifest_existed, manifest_bytes, previous_manifest = manifest_snapshot
    installed_packages: list[tuple[Literal["pi", "omp"], str, ArtifactRecord]] = []
    config_managed_bytes: bytes | None = None

    try:
        if previous_manifest is not None and _manifest_changed(
            previous_manifest,
            port=getattr(manifest, "port", port),
            backend=getattr(manifest, "backend", "anthropic"),
            anyllm_provider=getattr(manifest, "anyllm_provider", None),
            region=getattr(manifest, "region", None),
            memory=getattr(manifest, "memory_enabled", False),
        ):
            try:
                stop_runtime(previous_manifest)
            except Exception:
                pass
        config_artifact = ensure_extension_config(port, previous_config)
        try:
            config_managed_bytes = extension_config_path().absolute().read_bytes()
        except FileNotFoundError:
            config_managed_bytes = None

        for host, binary in hosts:
            previous_artifact = next(
                (
                    artifact
                    for artifact in (previous_manifest.artifacts if previous_manifest else [])
                    if artifact.kind == "pi-extension-package" and artifact.path == host
                ),
                None,
            )
            package_artifact = ensure_host_package(host, binary, release, previous_artifact)
            installed_packages.append((host, binary, package_artifact))

        native_targets = [host for host, _binary in hosts]
        final_targets = sorted(
            set(previous_manifest.targets if previous_manifest else []).union(native_targets)
        )
        provisional = deepcopy(manifest)
        provisional.targets = list(previous_manifest.targets if previous_manifest else [])
        provisional.artifacts = list(previous_manifest.artifacts if previous_manifest else [])
        provisional.supervisor_kind = SupervisorKind.TASK.value
        _save_manifest_verified(provisional)
        task_artifacts = install_supervisor(provisional)
        _start_profile_strict_locked(provisional)

        manifest.targets = final_targets
        manifest.supervisor_kind = SupervisorKind.TASK.value
        _upsert_artifacts(
            manifest,
            [
                config_artifact,
                *(artifact for _host, _binary, artifact in installed_packages),
                *task_artifacts,
            ],
        )
        _save_manifest_verified(manifest)
    except BaseException as initiating_error:
        rollback_errors: list[str] = []
        for host, binary, artifact in reversed(installed_packages):
            try:
                _restore_native_package(
                    host=host,
                    binary=binary,
                    previous=package_states[host],
                    current_artifact=artifact,
                )
            except BaseException as rollback_error:
                rollback_errors.append(str(rollback_error))
        # The failing helper owns its internal rollback. Verify that contract.
        installed_hosts = {host for host, _binary, _artifact in installed_packages}
        for host, binary in hosts:
            if host in installed_hosts:
                continue
            try:
                if inspect_host_package(host, binary) != package_states[host]:
                    rollback_errors.append(
                        f"{host} package helper did not restore its pre-init state."
                    )
            except BaseException as rollback_error:
                rollback_errors.append(str(rollback_error))
        for restore in (
            lambda: _restore_extension_config_snapshot(
                config_snapshot,
                expected_managed=config_managed_bytes or expected_config_bytes,
            ),
            lambda: remove_supervisor(manifest),
            lambda: _restore_scheduler(scheduler_snapshot),
            lambda: _restore_files(task_files_snapshot),
            lambda: _restore_manifest_snapshot(manifest.profile, manifest_existed, manifest_bytes),
            lambda: (
                _restore_runtime_state(
                    previous_manifest,
                    prior_runtime_status,
                    prior_runtime_ready,
                    assume_start_lock=True,
                )
                if previous_manifest is not None
                else stop_runtime(manifest)
                if getattr(manifest, "preset", None) is not None
                else None
            ),
        ):
            try:
                restore()
            except BaseException as rollback_error:
                rollback_errors.append(str(rollback_error))
        message = str(initiating_error)
        if rollback_errors:
            message += " Rollback also failed: " + "; ".join(rollback_errors)
        raise click.ClickException(message) from initiating_error


def _init_native_host(*, host: Literal["pi", "omp"], binary: str, profile: str, port: int) -> None:
    release = extension_release_version(_HEADROOM_VERSION)
    manifest_snapshot = _snapshot_manifest(profile)
    manifest = manifest_snapshot[2]
    if manifest is None:
        raise click.ClickException(f"Deployment profile {profile!r} was not created.")
    _init_native_hosts(
        hosts=[(host, binary)],
        release=release,
        manifest=deepcopy(manifest),
        manifest_snapshot=manifest_snapshot,
        port=port,
    )


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
        image="ghcr.io/headroomlabs-ai/headroom:latest",
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
    native_targets = _NATIVE_EXTENSION_TARGETS.intersection(targets)
    if not global_scope and native_targets:
        raise click.ClickException("Durable Pi/OMP init requires -g (current-user scope).")

    native_preflight: dict[str, tuple[str, str]] = {}
    native_manifest_snapshot: tuple[bool, bytes, Any | None] | None = None
    native_manifest: Any | None = None
    profile = _runtime_profile(global_scope)
    if native_targets:
        release = extension_release_version(_HEADROOM_VERSION)
        for target in sorted(native_targets):
            binary = shutil.which(target)
            if not binary:
                raise click.ClickException(f"'{target}' not found in PATH. Install {target} first.")
            native_preflight[target] = (binary, release)
        native_manifest_snapshot = _snapshot_manifest(profile)
        native_manifest = _build_runtime_manifest(
            profile=profile,
            existing=native_manifest_snapshot[2],
            targets=[target for target in targets if target != "openclaw"],
            port=port,
            backend=backend,
            anyllm_provider=anyllm_provider,
            region=region,
            memory=memory,
        )
    else:
        profile = _ensure_runtime_manifest(
            global_scope=global_scope,
            targets=[target for target in targets if target != "openclaw"],
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
    if native_targets:
        if native_manifest_snapshot is None or native_manifest is None:
            raise click.ClickException("Native Pi/OMP init transaction was not prepared.")
        native_hosts: list[tuple[Literal["pi", "omp"], str]] = [
            (cast(Literal["pi", "omp"], target), native_preflight[target][0])
            for target in targets
            if target in native_targets
        ]
        _init_native_hosts(
            hosts=native_hosts,
            release=next(iter(native_preflight.values()))[1],
            manifest=native_manifest,
            manifest_snapshot=native_manifest_snapshot,
            port=port,
        )

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
@click.option(
    "--port",
    default=8787,
    type=click.IntRange(1, 65535),
    show_default=True,
    help="Headroom proxy port.",
)
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
    if anyllm_provider and backend != "anyllm":
        click.echo(
            f"Warning: --anyllm-provider is ignored unless --backend anyllm "
            f"(got --backend {backend})."
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


def _remove_native_target(host: Literal["pi", "omp"], global_scope: bool) -> None:
    if not global_scope:
        raise click.ClickException("Durable Pi/OMP removal requires -g (current-user scope).")

    try:
        manifest = load_manifest(_GLOBAL_PROFILE)
    except ManifestError as exc:
        raise click.ClickException(
            f"Cannot remove durable {host} integration: {exc}. Fix or move the manifest, then retry."
        ) from exc
    if manifest is None or host not in manifest.targets:
        click.echo(f"No durable {host} integration is installed.")
        return

    package = next(
        (
            artifact
            for artifact in manifest.artifacts
            if artifact.kind == "pi-extension-package" and artifact.path == host
        ),
        None,
    )
    removing_last_native = not (_NATIVE_EXTENSION_TARGETS.intersection(manifest.targets) - {host})
    if removing_last_native:
        with acquire_runtime_start_lock(manifest.profile) as acquired:
            if not acquired:
                raise click.ClickException(
                    f"Profile {manifest.profile} is already being initialized."
                )
            _remove_native_target_locked(host, manifest, package)
        return
    _remove_native_target_locked(host, manifest, package)


def _remove_native_target_locked(host: Literal["pi", "omp"], manifest: Any, package: Any) -> None:
    if package is not None:
        binary = shutil.which(host)
        if not binary and package.metadata.get("owned") is True:
            raise click.ClickException(
                f"'{host}' not found in PATH; cannot safely remove its managed package."
            )
        remove_owned_host_package(host, binary or "", package)

    remaining_targets = [target for target in manifest.targets if target != host]
    if _NATIVE_EXTENSION_TARGETS.intersection(remaining_targets):
        manifest.targets = remaining_targets
        if package is not None:
            manifest.artifacts.remove(package)
        _save_manifest_verified(manifest)
        click.echo(f"Removed durable {host} integration.")
        return

    config = next(
        (artifact for artifact in manifest.artifacts if artifact.kind == "pi-extension-config"),
        None,
    )
    if config is not None:
        remove_owned_extension_config(config)
    task_paths = _task_file_paths(manifest)
    remove_supervisor(manifest)
    manifest.targets = remaining_targets
    if package is not None:
        manifest.artifacts.remove(package)
    for path in task_paths:
        path.unlink(missing_ok=True)
    manifest.artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.kind
        not in {
            "pi-extension-config",
            "script",
            "service-unit",
            "cron",
            "crontab",
            "plist",
            "windows-service",
            "windows-task",
        }
    ]
    manifest.supervisor_kind = SupervisorKind.NONE.value

    if manifest.targets:
        _save_manifest_verified(manifest)
    else:
        stop_runtime(manifest)
        delete_manifest(manifest.profile)
        if profile_root(manifest.profile).exists():
            raise click.ClickException(
                f"Could not verify deletion of profile {manifest.profile!r}."
            )
    click.echo(f"Removed durable {host} integration.")


@init.group("remove")
def init_remove() -> None:
    """Remove Headroom-managed durable agent integrations."""


@init_remove.command("pi")
@click.pass_context
def init_remove_pi(ctx: click.Context) -> None:
    """Remove the durable Pi Headroom extension."""
    _remove_native_target("pi", bool(_ctx_value(ctx, "global_scope")))


@init_remove.command("omp")
@click.pass_context
def init_remove_omp(ctx: click.Context) -> None:
    """Remove the durable OMP Headroom extension."""
    _remove_native_target("omp", bool(_ctx_value(ctx, "global_scope")))


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


@init.command("pi")
@click.pass_context
def init_pi(ctx: click.Context) -> None:
    """Install the durable Pi Headroom extension for the current user."""
    _run_init_targets(
        targets=["pi"],
        global_scope=bool(_ctx_value(ctx, "global_scope")),
        port=int(_ctx_value(ctx, "port") or 8787),
        backend=str(_ctx_value(ctx, "backend") or "anthropic"),
        anyllm_provider=_ctx_value(ctx, "anyllm_provider"),
        region=_ctx_value(ctx, "region"),
        memory=bool(_ctx_value(ctx, "memory")),
    )


@init.command("omp")
@click.pass_context
def init_omp(ctx: click.Context) -> None:
    """Install the durable OMP Headroom extension for the current user."""
    _run_init_targets(
        targets=["omp"],
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
