"""Undo the retired CLI context tools on machines that already have them.

Headroom used to download ``rtk`` (Rust Token Killer) and ``lean-ctx``, register
their Claude Code ``PreToolUse`` hooks, symlink the managed binaries onto
``PATH``, and inject a marker-fenced instruction block ("always prefix shell
commands with ``rtk``" / "prefer ``ctx_read`` over ``Read``") into a dozen agent
hint files. Both integrations have been removed from Headroom.

Deleting the code is not enough: everything above is *durable state on the
user's disk*. Left alone, the Claude hooks keep rewriting every Bash command
through binaries Headroom no longer manages, and the injected guidance keeps
telling agents to use tools that may not resolve. So ``headroom wrap`` /
``headroom unwrap`` call :func:`purge_context_tool_artifacts` once per run to
remove what earlier versions installed.

Everything here is idempotent, best-effort and deliberately conservative:

* only files Headroom installed (or caused a context tool to install) are
  deleted;
* ``~/.local/bin/{rtk,lean-ctx}`` is unlinked only when it is a symlink into
  Headroom's own bin directory — a user's own build is never touched;
* a JSON config that does not parse is reported and **skipped**, never
  overwritten (a hand-edited typo must not cost the user their settings);
* the tools' own backups of *config* files (``~/.claude.json.lean-ctx.bak`` and
  friends) are left in place — they hold the user's real settings history. Only
  backups of the hook scripts being deleted are cleaned up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from headroom import fsutil, paths

# Substrings that identify a hook command as one of the retired tools'.
# ``rtk init --auto-patch`` writes ``~/.claude/hooks/rtk-rewrite.sh`` and shells
# out to ``rtk rewrite``; ``lean-ctx init --agent`` writes ``lean-ctx-rewrite.sh``
# / ``lean-ctx-redirect.sh`` plus ``-native`` variants that exec ``lean-ctx hook
# rewrite``. All are specific enough that a user-authored hook cannot match by
# accident.
_HOOK_COMMAND_MARKERS = (
    "rtk-rewrite",
    "rtk rewrite",
    "lean-ctx-rewrite",
    "lean-ctx-redirect",
    "lean-ctx hook",
)

# Fence Headroom wrapped its injected guidance in. Both tools shared it — the
# lean-ctx block was written inside the same ``rtk-instructions`` markers.
_FENCE_START = "<!-- headroom:rtk-instructions -->"
_FENCE_END = "<!-- /headroom:rtk-instructions -->"

# Managed binary names (Windows ships the .exe).
_BINARY_NAMES = ("rtk", "rtk.exe", "lean-ctx", "lean-ctx.exe")

# Hook scripts the tools generated, relative to ``~/.claude/hooks``. Each is
# also cleaned up in its ``.lean-ctx.bak`` form: lean-ctx copies the previous
# script aside on every re-init, so the backups pile up alongside the originals.
_HOOK_SCRIPTS = (
    "rtk-rewrite.sh",
    ".rtk-hook.sha256",
    "lean-ctx-rewrite.sh",
    "lean-ctx-redirect.sh",
    "lean-ctx-rewrite-native",
    "lean-ctx-redirect-native",
)

# MCP server entries the tools registered, and the config files holding them.
# lean-ctx registers itself as an MCP server during ``lean-ctx init``; rtk never
# did, but it is matched too so a stale hand-added entry is cleaned up as well.
_MCP_SERVER_NAMES = ("lean-ctx", "lean_ctx", "rtk")


def purge_context_tool_artifacts() -> list[str]:
    """Remove every rtk / lean-ctx artifact an earlier Headroom version installed.

    Returns human-readable descriptions of what was removed — plus a line for
    any config that had to be skipped because the user must fix it by hand. An
    empty list means there was nothing to do, which is the steady state after
    the first run.
    """
    home = Path.home()
    project = Path.cwd()
    report: list[str] = []

    # 1. Hook registrations (Claude Code's settings.json, Cursor's hooks.json).
    for config in (home / ".claude" / "settings.json", home / ".cursor" / "hooks.json"):
        report += _purge_hook_config(config)

    # 2. The generated hook scripts, their integrity digests and stale backups.
    hooks_dir = home / ".claude" / "hooks"
    report += _remove_files(
        *(hooks_dir / name for name in _HOOK_SCRIPTS),
        *(hooks_dir / f"{name}.lean-ctx.bak" for name in _HOOK_SCRIPTS),
    )

    # 3. The PATH symlinks, then the managed binaries they pointed at.
    for name in ("rtk", "lean-ctx"):
        report += _remove_managed_path_link(home / ".local" / "bin" / name)
    report += _remove_files(*(paths.bin_dir() / name for name in _BINARY_NAMES))

    # 4. MCP server registrations (lean-ctx registers itself during init).
    report += _purge_mcp_entries(home / ".claude.json", "mcpServers")
    report += _purge_mcp_entries(_opencode_home(home) / "opencode.json", "mcp")

    # 5. Marker-fenced guidance in every hint file the wrap harnesses wrote to.
    for hint_file in _instruction_files(home, project):
        report += _purge_fenced_block(hint_file)
    report += _purge_continue_system_messages(project / ".continue" / "config.json")

    return report


def _instruction_files(home: Path, project: Path) -> list[Path]:
    """Hint files the wrap subcommands injected the context-tool block into.

    De-duplicated: ``CODEX_HOME`` / ``OPENCODE_HOME`` can point at the project
    directory, and a file must not be reported twice.
    """
    candidates = (
        home / ".claude" / "CLAUDE.md",
        _codex_home(home) / "AGENTS.md",
        _opencode_home(home) / "AGENTS.md",
        project / "AGENTS.md",
        project / ".github" / "copilot-instructions.md",
        # Shared by `wrap aider` and `wrap openclaude`.
        project / "CONVENTIONS.md",
        project / ".clinerules",
        project / ".goosehints",
        project / ".cursorrules",
    )
    return list(dict.fromkeys(candidates))


def _codex_home(home: Path) -> Path:
    """Codex's config directory, respecting ``CODEX_HOME``."""
    configured = os.environ.get("CODEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else home / ".codex"


def _opencode_home(home: Path) -> Path:
    """OpenCode's config directory, respecting ``OPENCODE_HOME``."""
    configured = os.environ.get("OPENCODE_HOME", "").strip()
    return Path(configured).expanduser() if configured else home / ".config" / "opencode"


# --- hook registrations -------------------------------------------------------


def _references_context_tool(entry: Any) -> bool:
    """Whether a hook entry's command is one a retired context tool registered."""
    if not isinstance(entry, dict):
        return False
    command = str(entry.get("command", "")).lower()
    return any(marker in command for marker in _HOOK_COMMAND_MARKERS)


def _prune_hooks(hooks: Any) -> tuple[Any, bool]:
    """Drop retired-tool entries from a ``hooks`` mapping; return ``(pruned, changed)``.

    Handles both shapes Headroom's installers produced: Claude Code nests
    ``hooks.<Event>[].hooks[].command`` while Cursor's ``hooks.json`` puts the
    command directly on the event entry. Anything unrecognised is passed
    through untouched, and user-authored entries always survive.
    """
    if not isinstance(hooks, dict):
        return hooks, False

    changed = False
    pruned: dict[str, Any] = {}
    for event, entries in hooks.items():
        if not isinstance(entries, list):
            pruned[event] = entries
            continue

        retained: list[Any] = []
        for entry in entries:
            # Cursor shape: the command sits on the entry itself.
            if _references_context_tool(entry):
                changed = True
                continue
            # Claude shape: a matcher entry holding a list of hooks.
            inner = entry.get("hooks") if isinstance(entry, dict) else None
            if isinstance(inner, list):
                kept_inner = [item for item in inner if not _references_context_tool(item)]
                if len(kept_inner) != len(inner):
                    changed = True
                    if not kept_inner:
                        # The matcher existed only to hold the retired tool's hook.
                        continue
                    entry = {**entry, "hooks": kept_inner}
            retained.append(entry)

        if not entries:
            pruned[event] = entries
        elif retained:
            pruned[event] = retained
        else:
            # Every entry for this event was ours — drop the empty event.
            changed = True

    return pruned, changed


def _purge_hook_config(path: Path) -> list[str]:
    """Remove retired-tool hook registrations from a JSON hook config."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(fsutil.read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"skipped {path} (unreadable JSON: {exc}) — remove any stale hook by hand"]
    if not isinstance(payload, dict):
        return [f"skipped {path} (not a JSON object) — remove any stale hook by hand"]

    hooks, changed = _prune_hooks(payload.get("hooks"))
    if not changed:
        return []
    if hooks:
        payload["hooks"] = hooks
    else:
        payload.pop("hooks", None)
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")
    return [f"removed the retired context-tool hook from {path}"]


# --- MCP registrations --------------------------------------------------------


def _purge_mcp_entries(path: Path, container_key: str) -> list[str]:
    """Drop retired-tool MCP server entries from a client config.

    ``lean-ctx init`` registers lean-ctx as an MCP server in the harness's own
    config — Claude Code keeps them under ``mcpServers``, OpenCode under ``mcp``.
    Only the exactly-named entries are removed; every other server, and every
    unrelated top-level key, is preserved byte-for-byte.
    """
    if not path.is_file():
        return []
    try:
        payload = json.loads(fsutil.read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"skipped {path} (unreadable JSON: {exc}) — remove any stale MCP entry by hand"]
    if not isinstance(payload, dict):
        return [f"skipped {path} (not a JSON object) — remove any stale MCP entry by hand"]

    servers = payload.get(container_key)
    if not isinstance(servers, dict):
        return []
    removed = [name for name in _MCP_SERVER_NAMES if name in servers]
    if not removed:
        return []
    for name in removed:
        servers.pop(name, None)
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")
    return [f"removed the {', '.join(removed)} MCP server entry from {path}"]


# --- binaries -----------------------------------------------------------------


def _remove_files(*targets: Path) -> list[str]:
    """Delete Headroom-managed files, reporting each removal."""
    report: list[str] = []
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        try:
            target.unlink()
        except OSError as exc:
            report.append(f"could not remove {target} ({exc})")
            continue
        report.append(f"removed {target}")
    return report


def _remove_managed_path_link(link: Path) -> list[str]:
    """Unlink ``link`` only if it is a symlink into Headroom's bin directory.

    A real file, or a symlink to a user's own build, is left alone — the PATH
    entry is not ours to reclaim unless we created it.
    """
    if not link.is_symlink():
        return []
    try:
        target = link.resolve()
        managed_dir = paths.bin_dir().resolve()
    except OSError:
        return []
    if managed_dir not in target.parents:
        return []
    return _remove_files(link)


# --- marker-fenced guidance ---------------------------------------------------


def _strip_fences(content: str) -> str | None:
    """Remove every fenced context-tool block from ``content``.

    Returns the cleaned text (``""`` when nothing but the block was in the
    file), or ``None`` when no complete fence was present.
    """
    cleaned = content
    found = False
    while True:
        start = cleaned.find(_FENCE_START)
        if start < 0:
            break
        end = cleaned.find(_FENCE_END, start)
        if end < 0:
            break
        end += len(_FENCE_END)
        prefix = cleaned[:start].rstrip()
        suffix = cleaned[end:].lstrip("\n")
        cleaned = "\n\n".join(part for part in (prefix, suffix) if part)
        found = True

    if not found:
        return None
    return cleaned.rstrip() + "\n" if cleaned.strip() else ""


def _purge_fenced_block(path: Path) -> list[str]:
    """Strip the context-tool guidance block from a hint file (deleting an empty file)."""
    if not path.is_file():
        return []
    try:
        content = fsutil.read_text(path)
    except OSError:
        return []
    cleaned = _strip_fences(content)
    if cleaned is None:
        return []
    if cleaned:
        fsutil.write_text(path, cleaned)
        return [f"removed the context-tool guidance block from {path}"]
    return _remove_files(path)


def _purge_continue_system_messages(path: Path) -> list[str]:
    """Strip the context-tool block from every ``systemMessage`` in Continue's config.

    Continue supports a top-level ``systemMessage`` plus a per-model override on
    each ``models[]`` entry, and wrap injected into all of them. A field left
    empty by the strip is removed so the config returns to "not set".
    """
    if not path.is_file():
        return []
    try:
        payload = json.loads(fsutil.read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"skipped {path} (unreadable JSON: {exc}) — remove any stale guidance by hand"]
    if not isinstance(payload, dict):
        return [f"skipped {path} (not a JSON object) — remove any stale guidance by hand"]

    containers: list[dict[str, Any]] = [payload]
    models = payload.get("models")
    if isinstance(models, list):
        containers += [model for model in models if isinstance(model, dict)]

    changed = False
    for container in containers:
        message = container.get("systemMessage")
        if not isinstance(message, str):
            continue
        cleaned = _strip_fences(message)
        if cleaned is None:
            continue
        changed = True
        if cleaned.strip():
            container["systemMessage"] = cleaned
        else:
            container.pop("systemMessage", None)

    if not changed:
        return []
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")
    return [f"removed the context-tool guidance from {path}"]
