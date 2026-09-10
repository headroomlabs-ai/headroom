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
``headroom unwrap`` call :func:`purge_context_tool_artifacts` on every run to
remove what earlier versions installed — machine-global artifacts (hooks,
binaries, Claude Code's MCP registration) are removed once per workspace and
then stamped done (see the stamp below), while project- and config-directory-
scoped guidance (``CODEX_HOME`` / ``OPENCODE_HOME`` hint files, Continue's
config) is inspected on every launch, since a later launch can sit in a
different project or point at a different ``CODEX_HOME`` / ``OPENCODE_HOME``.

Everything here is idempotent, best-effort and deliberately conservative:

* only files Headroom installed (or caused a context tool to install) are
  deleted. An MCP entry's ``command`` or a hook script's body counts as
  Headroom's only when it names a path inside :func:`paths.bin_dir`
  (:func:`_references_managed_bin` — which is where the matching rules and
  the reasons behind them live);
* a hook entry is Headroom's when it names such a path directly, or — the
  common case, since a hook command names a script rather than the binary —
  when it names one of the ``~/.claude/hooks`` scripts already classified as
  Headroom's, whose verdict it inherits
  (:func:`_references_context_tool`, :func:`_names_a_managed_script`). A
  Cursor ``hooks.json`` entry naming a script under ``~/.cursor`` cannot
  inherit a verdict this way, since the map covers ``~/.claude/hooks`` only;
  it is still caught when its ``command`` names the managed directory;
* ``.rtk-hook.sha256`` is never read for its own provenance (it holds a hex
  digest, not a path) and instead inherits ``rtk-rewrite.sh``'s
  classification; a ``<name>.lean-ctx.bak`` backup inherits ``<name>``'s
  (:func:`_classify_hook_scripts`);
* a hook script that exists but cannot be read is classified unknown —
  deleted by nothing, and named in the report so the user can remove it by
  hand;
* ``~/.local/bin/{rtk,lean-ctx}`` is unlinked only when it is a symlink into
  Headroom's own bin directory — a user's own build is never touched;
* a JSON config that does not parse is reported and **skipped**, never
  overwritten (a hand-edited typo must not cost the user their settings);
* the tools' own backups of *config* files (``~/.claude.json.lean-ctx.bak`` and
  friends) are left in place — they hold the user's real settings history. Only
  backups of the hook scripts proven to be Headroom's are cleaned up;
* two cases cannot be decided at all, and are accepted as limits rather than
  fixed:

  * ``get_lean_ctx_path`` used to check ``PATH`` before Headroom's own bin
    directory, so on a machine that already had ``lean-ctx`` on ``PATH``,
    the tool that ran was the user's own, and the config it wrote looks
    exactly like config the user wrote by hand. That leftover survives the
    purge — it still points at a binary that exists, so nothing dangles;
  * an rtk hook written *after* #1698 execs a bare ``rtk`` and never mentions
    :func:`paths.bin_dir`, so it reads exactly like a hook a user wrote by
    hand, and ``rtk-rewrite.sh``, its ``.rtk-hook.sha256`` and its
    ``settings.json`` entry all survive while step 3 removes the managed
    binary — leaving a hook that silently no-ops (#487, #1698). Earlier
    hooks are decidable: Headroom patched the absolute managed path into
    them (``_patch_rtk_hook_absolute_path``, removed by #1698), so the
    window this misses is rtk setups run between #1698 and the tools'
    removal in #2677;
* the marker-fenced guidance block is the one step with no provenance check
  to make — ``<!-- headroom:rtk-instructions -->`` is Headroom's own fence,
  and no third party writes it.

Removing the retired integration's machine-global footprint — hook
registrations, hook scripts, PATH symlinks, managed binaries and Claude
Code's own MCP registration — is a one-time migration: the first completed
run of that half stamps ``.context-tools-purged`` beside the managed bin
directory, and every later run skips that half outright. Without the stamp
this would keep rewriting the same machine-wide files on every ``wrap``
invocation forever, and a user who installs one of these tools *after* the
migration would have Headroom auditing files at each launch for a leftover
that cannot exist there.

Project- and config-directory-scoped state is not covered by that stamp: a
later invocation can sit in a different project, or point ``CODEX_HOME`` /
``OPENCODE_HOME`` somewhere the stamped run never inspected, and whatever
guidance an earlier Headroom left behind there is still worth removing — so
those steps run on every invocation instead (:func:`_purge_invocation_scoped`).
"""

from __future__ import annotations

import json
import os
import posixpath
import re
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

# rtk's integrity digest never names a path (see ``_classify_hook_scripts``)
# and rtk-rewrite.sh is the script it authenticates.
_RTK_DIGEST_NAME = ".rtk-hook.sha256"
_RTK_SCRIPT_NAME = "rtk-rewrite.sh"

# MCP server entries the tools registered, and the config files holding them.
# lean-ctx registers itself as an MCP server during ``lean-ctx init``; rtk never
# did, but it is matched too so a stale hand-added entry is cleaned up as well.
_MCP_SERVER_NAMES = ("lean-ctx", "lean_ctx", "rtk")

# Report-line prefixes meaning "this one is not settled" — a config that would
# not parse, a script that would not read, a file that would not unlink. Such a
# run leaves a leftover behind, so it must not be stamped as the completed
# migration. Emitted by _purge_hook_config, _purge_mcp_entries,
# _purge_fenced_block, _purge_continue_system_messages and _remove_files.
_DEFERRED_PREFIXES = ("skipped ", "could not remove ")


def purge_context_tool_artifacts() -> list[str]:
    """Remove every rtk / lean-ctx artifact an earlier Headroom version installed.

    Returns human-readable descriptions of what was removed — plus a line for
    any config that had to be skipped because the user must fix it by hand.
    Machine-global cleanup (hooks, binaries, Claude Code's MCP registration)
    runs once and is then skipped via the stamp below; project- and config-
    directory-scoped cleanup (hint files, ``CODEX_HOME`` / ``OPENCODE_HOME``,
    Continue's config) runs on every call, so a later call in a different
    project or a repointed ``CODEX_HOME`` / ``OPENCODE_HOME`` can still report
    something even after the global half is long since stamped done.
    """
    marker = _purge_marker()
    home = Path.home()
    project = Path.cwd()
    report: list[str] = []

    if not marker.exists():
        # Classify every hook script's provenance once, up front: both step 1
        # (is a settings.json entry pointing at *our* script?) and step 2 (is
        # the script itself ours?) need the same answer, and each file is
        # read once.
        hooks_dir = home / ".claude" / "hooks"
        hook_classification = _classify_hook_scripts(hooks_dir)

        # 1. Hook registrations (Claude Code's settings.json, Cursor's hooks.json).
        for config in (home / ".claude" / "settings.json", home / ".cursor" / "hooks.json"):
            report += _purge_hook_config(config, hooks_dir, hook_classification)

        # 2. The generated hook scripts, their integrity digests and stale
        # backups — only the ones proven to reference Headroom's managed bin
        # directory.
        managed_names = [name for name in _HOOK_SCRIPTS if hook_classification.get(name)]
        report += _remove_files(
            *(hooks_dir / name for name in managed_names),
            *(hooks_dir / f"{name}.lean-ctx.bak" for name in managed_names),
        )
        for name in _HOOK_SCRIPTS:
            if hook_classification.get(name, False) is not None:
                continue
            if name == _RTK_DIGEST_NAME:
                report.append(
                    f"skipped {hooks_dir / name} (inherits {_RTK_SCRIPT_NAME}'s unreadable verdict)"
                    " — remove any stale hook script by hand"
                )
            else:
                report.append(
                    f"skipped {hooks_dir / name} (could not read to verify it was Headroom's)"
                    " — remove any stale hook script by hand"
                )

        # 3. The PATH symlinks, then the managed binaries they pointed at.
        for name in ("rtk", "lean-ctx"):
            report += _remove_managed_path_link(home / ".local" / "bin" / name)
        report += _remove_files(*(paths.bin_dir() / name for name in _BINARY_NAMES))

        # 4. Claude Code's own MCP server registration (lean-ctx registers
        # itself during init). OpenCode's is invocation-scoped — see below.
        report += _purge_mcp_entries(home / ".claude.json", "mcpServers")

        # Only a global half that settled everything is the completed
        # migration. One that could not read a script or parse a config left
        # a leftover behind, and the user needs both the reminder on the next
        # launch and the cleanup once the permissions or the typo are fixed.
        # A deferral in the invocation-scoped half below must not withhold
        # this stamp — that half re-runs every time regardless, so nothing is
        # lost by stamping the global half done now.
        if not any(line.startswith(_DEFERRED_PREFIXES) for line in report):
            try:
                # Cleanup must not become the first mutation on a pristine
                # machine. In particular, ``wrap <missing-tool>`` validates
                # the binary after the wrap-group migration hook; creating
                # ``~/.headroom`` merely to stamp an empty scan violates that
                # command's no-side-effects-on-failure contract. Established
                # Headroom installs already have the state directory and get
                # the one-time fast path; clean machines cheaply rescan until
                # some real Headroom state exists.
                if marker.parent.is_dir():
                    marker.touch()
            except OSError:
                pass  # Unwritable workspace: the purge simply runs again next time.

    report += _purge_invocation_scoped(home, project)
    return report


def _purge_invocation_scoped(home: Path, project: Path) -> list[str]:
    """Steps the one-time stamp must never withhold.

    ``OPENCODE_HOME``'s config, the hint files in ``project`` /
    ``CODEX_HOME`` / ``OPENCODE_HOME``, and Continue's config are all a
    function of *this* invocation's cwd and environment, not of the machine —
    a later run can sit in a different project or point ``CODEX_HOME`` /
    ``OPENCODE_HOME`` somewhere the global-half stamp never inspected. Each
    step is cheap and side-effect-free when nothing matches, so re-running
    them on every invocation costs a handful of reads in the steady state.
    """
    report: list[str] = []
    report += _purge_mcp_entries(_opencode_home(home) / "opencode.json", "mcp")
    for hint_file in _instruction_files(home, project):
        report += _purge_fenced_block(hint_file)
    report += _purge_continue_system_messages(project / ".continue" / "config.json")
    return report


def _purge_marker() -> Path:
    """Path of the "already migrated" stamp.

    Derived from :func:`paths.bin_dir` rather than ``workspace_dir`` so it
    cannot escape a temporary tree through ``HEADROOM_WORKSPACE_DIR``.
    """
    return paths.bin_dir().parent / ".context-tools-purged"


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


# Characters that can never be part of a path: a match ending right before
# one of these (or at end-of-string) sits at a real word boundary. Quotes,
# `=`/`:` (`export PATH="<dir>:$PATH"`, `BIN=<dir>/x`), `;`/`,`/`|` (command
# joiners) and `()` (subshells) all end a path the same way whitespace does.
_PATH_BOUNDARY_CHARS = frozenset("\"'=:;,()|")

# Splits a command into path-shaped tokens on whitespace plus the same
# boundary punctuation above — used by _names_a_managed_script, which (unlike
# _references_managed_bin's needle-anchored scan) tokenizes the whole command.
_PATH_TOKEN_SPLIT = re.compile(r"[\s" + re.escape("".join(_PATH_BOUNDARY_CHARS)) + r"]+")


def _norm_path_text(value: str) -> str:
    """Case-fold ``value`` and give it one separator, so paths compare as text."""
    return os.path.normcase(value).replace("\\", "/")


def _references_managed_bin(text: str) -> bool:
    """Whether ``text`` names a path inside Headroom's managed bin directory.

    A hook command or script body is free text we don't control, so this
    scans ``text`` for raw occurrences of the managed directory — the
    unresolved and resolved bin directory, and its ``~``-relative form
    (home-relative, since a script may reference it unexpanded) — matched
    against the text as given and as ``expanduser``'d, case-folded, with both
    path separators. Deliberately *not* tokenized on whitespace first: a
    quoted or ``$HOME``-derived path can itself contain a space, and slicing
    the text into words before searching would sever it.

    A hit is only a real reference at a path boundary on *both* ends. The
    character immediately before the match, if any, must be whitespace or a
    :data:`_PATH_BOUNDARY_CHARS` character, or the match is just the tail of
    some longer, unrelated path segment (e.g. ``/prefix<bin_dir>/lean-ctx``)
    and is rejected. The match must then be immediately followed by
    end-of-string or a :data:`_PATH_BOUNDARY_CHARS` character (an exact
    reference, e.g. a bare ``PATH=<dir>`` export), or by ``/`` — in which
    case the run of characters up to the next boundary is lexically
    normalized (``.``/``..`` collapsed) and re-compared, so neither a sibling
    directory like ``bin-backup``/``binfoo`` nor a ``bin/../evil`` traversal
    can borrow the managed prefix.
    # ponytail: boundary-aware substring scan, not a shell parse — upgrade to
    # shlex if a command ever embeds a managed path it does not execute.
    """
    try:
        bin_dir = paths.bin_dir()
        resolved_bin_dir = bin_dir.resolve()
    except OSError:
        return False

    needles = {_norm_path_text(str(bin_dir)), _norm_path_text(str(resolved_bin_dir))}
    home = Path.home()
    for base in (bin_dir, resolved_bin_dir):
        try:
            needles.add(_norm_path_text(f"~/{base.relative_to(home).as_posix()}"))
        except ValueError:
            pass

    for haystack in (text, os.path.expanduser(text)):
        normalized_haystack = _norm_path_text(haystack)
        if any(_names_managed_dir(normalized_haystack, needle) for needle in needles):
            return True
    return False


def _names_managed_dir(haystack: str, needle: str) -> bool:
    """Whether a normalized ``haystack`` names ``needle`` at a path boundary
    on both ends: the character right before the match, if any, must be
    whitespace or a :data:`_PATH_BOUNDARY_CHARS` character too, or a
    user-owned path that merely has the managed directory as a substring
    (e.g. ``/prefix<bin_dir>/lean-ctx``) would be misread as naming it.
    """
    search_from = 0
    while True:
        index = haystack.find(needle, search_from)
        if index < 0:
            return False
        end = index + len(needle)
        search_from = index + 1  # keep scanning; occurrences may overlap
        if index > 0 and not (
            haystack[index - 1].isspace() or haystack[index - 1] in _PATH_BOUNDARY_CHARS
        ):
            continue
        following = haystack[end : end + 1]
        if not following or following.isspace() or following in _PATH_BOUNDARY_CHARS:
            return True
        if following != "/":
            continue
        tail_end = end
        while tail_end < len(haystack) and not (
            haystack[tail_end].isspace() or haystack[tail_end] in _PATH_BOUNDARY_CHARS
        ):
            tail_end += 1
        candidate = posixpath.normpath(haystack[index:tail_end])
        if candidate == needle or candidate.startswith(needle + "/"):
            return True


def _classify_hook_scripts(hooks_dir: Path) -> dict[str, bool | None]:
    """Classify each existing ``_HOOK_SCRIPTS`` file by whether it is Headroom's.

    ``True`` — the file exists and its body references the managed bin
    directory. ``False`` — it exists and does not. ``None`` — it exists but
    could not be read, so its provenance is unprovable. A basename with no
    file on disk is simply absent from the map. Each file is read at most once.

    ``.rtk-hook.sha256`` holds a hex digest that can never reference a path,
    so it is never read; it inherits ``rtk-rewrite.sh``'s classification.
    """
    classification: dict[str, bool | None] = {}
    for name in _HOOK_SCRIPTS:
        if name == _RTK_DIGEST_NAME:
            continue
        path = hooks_dir / name
        if not path.is_file():
            continue
        try:
            body = fsutil.read_text(path)
        except OSError:
            classification[name] = None
            continue
        classification[name] = _references_managed_bin(body)

    if (hooks_dir / _RTK_DIGEST_NAME).is_file():
        classification[_RTK_DIGEST_NAME] = classification.get(_RTK_SCRIPT_NAME, False)

    return classification


def _references_context_tool(
    entry: Any, hooks_dir: Path, hook_classification: dict[str, bool | None]
) -> bool:
    """Whether a hook entry is one a retired context tool registered.

    A command-marker hit alone is not enough — a user could author a script
    with a matching name. It must also either name the managed bin directory
    directly, or name — as a path resolving to that exact file, not merely
    sharing its basename, so a same-named script of the user's own in a
    different directory is never caught — a hook script in ``hooks_dir`` the
    classification map marks ``True`` (see :func:`_names_a_managed_script`).
    """
    if not isinstance(entry, dict):
        return False
    command = str(entry.get("command", ""))
    if not any(marker in command.lower() for marker in _HOOK_COMMAND_MARKERS):
        return False
    if _references_managed_bin(command):
        return True
    return _names_a_managed_script(command, hooks_dir, hook_classification)


def _names_a_managed_script(
    command: str, hooks_dir: Path, hook_classification: dict[str, bool | None]
) -> bool:
    """Whether ``command`` names, by absolute path, a hook script the map marks ``True``.

    A real command is rarely the bare script path: ``bash <script>`` wraps
    it, a shell often quotes it, and it may carry a redundant ``./`` segment.
    ``command`` is split into path-shaped tokens on the same boundary
    punctuation :data:`_PATH_BOUNDARY_CHARS` (and whitespace) mark as *not*
    part of a path, each token is lexically normalized, and compared against
    the script's absolute path only.

    Deliberately not resolved against ``~``/:func:`Path.home` or against the
    process's working directory: a *relative* hook command in
    ``~/.claude/settings.json`` is resolved by the harness against the
    project's cwd, not against home — this module never writes or inspects a
    project-relative path, so treating one as if it named a home-relative
    script would delete a project-local hook this purge has no business
    touching. A relative token, and a ``$VAR``-style unexpanded reference
    (e.g. ``$HOME/...``), are both simply not recognised — unprovable, so
    kept, per the guard's own rule.
    """
    tokens = {
        token
        for haystack in (command, os.path.expanduser(command))
        for token in _PATH_TOKEN_SPLIT.split(haystack)
        if token
    }
    normalized_tokens = {posixpath.normpath(_norm_path_text(token)) for token in tokens}
    return any(
        verdict is True and _norm_path_text(str(hooks_dir / name)) in normalized_tokens
        for name, verdict in hook_classification.items()
    )


def _prune_hooks(
    hooks: Any, hooks_dir: Path, hook_classification: dict[str, bool | None]
) -> tuple[Any, bool]:
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
            if _references_context_tool(entry, hooks_dir, hook_classification):
                changed = True
                continue
            # Claude shape: a matcher entry holding a list of hooks.
            inner = entry.get("hooks") if isinstance(entry, dict) else None
            if isinstance(inner, list):
                kept_inner = [
                    item
                    for item in inner
                    if not _references_context_tool(item, hooks_dir, hook_classification)
                ]
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


def _purge_hook_config(
    path: Path, hooks_dir: Path, hook_classification: dict[str, bool | None]
) -> list[str]:
    """Remove retired-tool hook registrations from a JSON hook config."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(fsutil.read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"skipped {path} (unreadable JSON: {exc}) — remove any stale hook by hand"]
    if not isinstance(payload, dict):
        return [f"skipped {path} (not a JSON object) — remove any stale hook by hand"]

    hooks, changed = _prune_hooks(payload.get("hooks"), hooks_dir, hook_classification)
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
    A name match alone is not enough — a user can register their own server
    under the same name — so an entry is removed only when its ``command``
    also names Headroom's managed bin directory. Every other server, and
    every unrelated top-level key, is preserved byte-for-byte.
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
    removed = [
        name
        for name in _MCP_SERVER_NAMES
        if isinstance(servers.get(name), dict)
        and isinstance(servers[name].get("command"), str)
        and _references_managed_bin(servers[name]["command"])
    ]
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
