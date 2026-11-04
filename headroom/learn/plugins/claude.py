"""Claude Code plugin for headroom learn.

Reads conversation logs from ~/.claude/projects/ (JSONL format).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path, PureWindowsPath

from .._shared import classify_error, claude_config_dir, is_error_content
from ..base import ConversationScanner, LearnPlugin
from ..models import (
    ErrorCategory,
    ProjectInfo,
    SessionData,
    SessionEvent,
    ToolCall,
)
from ..writer import ClaudeCodeWriter, ContextWriter
from ._paths import path_exists as _path_exists

logger = logging.getLogger(__name__)


class ClaudeCodePlugin(LearnPlugin, ConversationScanner):
    """Reads Claude Code conversation logs from ~/.claude/projects/.

    Claude Code stores conversations as JSONL files with these line types:
    - type="assistant": message.content[] has tool_use blocks (name, input, id)
    - type="user": message.content[] has tool_result blocks (tool_use_id, content)
    """

    def __init__(self, claude_dir: Path | None = None):
        self.claude_dir = claude_dir or claude_config_dir()
        self.projects_dir = self.claude_dir / "projects"

    # --- LearnPlugin identity ---

    @property
    def name(self) -> str:
        return "claude"

    @property
    def display_name(self) -> str:
        return "Claude Code"

    @property
    def description(self) -> str:
        return "Claude Code (~/.claude/)"

    def detect(self) -> bool:
        return self.projects_dir.exists() and any(self.projects_dir.iterdir())

    def create_writer(self) -> ContextWriter:
        return ClaudeCodeWriter()

    # --- ConversationScanner interface ---

    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects under ~/.claude/projects/."""
        if not self.projects_dir.exists():
            return []

        projects = []
        for entry in sorted(self.projects_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            project_path = _decode_project_path(entry.name)
            if project_path is None:
                win = re.match(r"^-?([A-Za-z])--?(.+)$", entry.name)
                if win:
                    drive = win.group(1).upper()
                    tokens = [p for p in win.group(2).split("-") if p]
                    project_path = Path(f"{drive}:\\" + "\\".join(tokens))
                else:
                    stripped = entry.name.lstrip("-")
                    project_path = Path("/" + stripped.replace("-", "/"))

            name = _project_display_name(project_path, entry.name)

            context_file = None
            if _path_exists(project_path):
                claude_md = project_path / "CLAUDE.md"
                if _path_exists(claude_md):
                    context_file = claude_md

            # `entry` itself stats fine (its parent is ours) but a project dir
            # left behind by a root-run session is not traversable, so stat-ing
            # anything under it raises PermissionError. Treat that as absent.
            memory_dir = entry / "memory"
            memory_file = memory_dir / "MEMORY.md" if _path_exists(memory_dir) else None
            if memory_file and not _path_exists(memory_file):
                memory_file = None

            jsonl_files = list(entry.glob("*.jsonl"))
            if not jsonl_files:
                continue

            session_project_path = self._project_path_from_session_cwd(jsonl_files)
            if session_project_path is not None:
                project_path = session_project_path
                name = _project_display_name(project_path, entry.name)

            projects.append(
                ProjectInfo(
                    name=name,
                    project_path=project_path,
                    data_path=entry,
                    context_file=context_file,
                    memory_file=memory_file,
                )
            )

        return self._merge_worktrees(projects)

    def _merge_worktrees(self, projects: list[ProjectInfo]) -> list[ProjectInfo]:
        """Fold each linked git worktree into the project of its main checkout.

        Claude Code files every working directory's sessions under its own
        folder, so each worktree (a Conductor workspace, ``.claude/worktrees/*``)
        would otherwise be its own project: a handful of sessions, too thin to
        cross a pattern threshold, with learnings written into a checkout that
        is deleted with the workspace. The merged project keeps the main
        checkout's path and memory folder, which Claude Code's auto-memory
        shares across a repo's worktrees, and scans every member's sessions.
        A worktree whose checkout is gone has nothing to resolve it from and
        stays separate.
        """
        groups: dict[Path, list[tuple[ProjectInfo, Path | None]]] = {}
        for project in projects:
            root = _main_worktree_root(project.project_path)
            try:
                key = root or project.project_path.resolve()
            except OSError:
                key = project.project_path
            groups.setdefault(key, []).append((project, root))

        merged: list[ProjectInfo] = []
        for key, members in groups.items():
            if all(root is None for _, root in members):
                merged.extend(project for project, _ in members)
                continue
            main = next((project for project, root in members if root is None), None)
            # A repo worked on only through worktrees has no session folder of
            # its own; use the one Claude Code would give its main checkout.
            data_path = (
                main.data_path
                if main
                else self.projects_dir / re.sub(r"[^A-Za-z0-9]", "-", str(key))
            )
            project_path = main.project_path if main else key
            claude_md = project_path / "CLAUDE.md"
            memory_file = data_path / "memory" / "MEMORY.md"
            merged.append(
                dataclasses.replace(
                    main or members[0][0],
                    name=_project_display_name(project_path, data_path.name),
                    project_path=project_path,
                    data_path=data_path,
                    context_file=claude_md if _path_exists(claude_md) else None,
                    memory_file=memory_file if _path_exists(memory_file) else None,
                    # The session folder can be a subdirectory of the checkout;
                    # record the checkout root too so selecting the worktree
                    # itself, or a path elsewhere in it, still finds this project.
                    worktree_paths=list(
                        dict.fromkeys(
                            path
                            for p, root in members
                            if root is not None
                            for path in (p.project_path, _checkout_root(p.project_path))
                            if path is not None
                        )
                    ),
                    extra_data_paths=[p.data_path for p, _ in members if p is not main],
                )
            )
        return merged

    @staticmethod
    def _project_path_from_session_cwd(jsonl_files: list[Path]) -> Path | None:
        for jsonl_path in sorted(jsonl_files):
            try:
                with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        cwd = event.get("cwd")
                        if isinstance(cwd, str) and cwd:
                            project_path = Path(cwd)
                            if project_path.exists():
                                return project_path
            except (OSError, UnicodeDecodeError):
                continue
        return None

    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all conversation JSONL files for a project.

        Claude Code writes the main session at ``<project>/<uuid>.jsonl`` and
        nests the transcripts it spawns under ``<project>/<uuid>/subagents/**``
        (subagents) and ``.../subagents/workflows/**`` (workflow agents). Each
        nested transcript is its own context window with its own token spend, so
        by default we descend into them. Pass ``include_subagents=False`` to
        restrict to top-level main sessions only.
        """
        file_sources: list[tuple[Path, str]] = []
        for data_path in (project.data_path, *project.extra_data_paths):
            if include_subagents:
                jsonl_files = sorted(data_path.rglob("*.jsonl"))
            else:
                jsonl_files = sorted(data_path.glob("*.jsonl"))
            file_sources.extend((f, self._classify_source(data_path, f)) for f in jsonl_files)
        if not file_sources:
            return []
        jsonl_files = [f for f, _ in file_sources]

        if max_workers <= 1 or len(jsonl_files) <= 1:
            return [
                s
                for f, src in file_sources
                if (s := self._scan_session(f, source=src)) and s.tool_calls
            ]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        sessions: list[SessionData] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._scan_session, f, src): f for f, src in file_sources}
            for future in as_completed(futures):
                session = future.result()
                if session and session.tool_calls:
                    sessions.append(session)
        return sessions

    @staticmethod
    def _classify_source(data_path: Path, jsonl_path: Path) -> str:
        """Tag a transcript as main / subagent / workflow from its path depth."""
        parts = jsonl_path.relative_to(data_path).parts
        if len(parts) == 1:
            return "main"
        if "workflows" in parts:
            return "workflow"
        return "subagent"

    def _scan_session(self, jsonl_path: Path, source: str = "main") -> SessionData | None:
        """Scan a single JSONL conversation file."""
        session_id = jsonl_path.stem
        tool_uses: dict[str, tuple[str, dict]] = {}
        tool_calls: list[ToolCall] = []
        events: list[SessionEvent] = []
        total_input_tokens = 0
        total_output_tokens = 0
        msg_index = 0

        try:
            with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_index += 1
                    line_type = d.get("type", "")
                    ts = d.get("timestamp", None)

                    if line_type == "assistant":
                        self._extract_tool_uses(d, tool_uses)
                        # `get("message", {})` returns None for an explicit
                        # {"message": null} line (the default only applies to a
                        # missing key); `.get` on None then raises AttributeError,
                        # which the OSError/UnicodeDecodeError guard does not catch
                        # — so one malformed line crashed the whole learn run.
                        usage = (d.get("message") or {}).get("usage", {})
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_input_tokens += usage.get("cache_read_input_tokens", 0)
                        total_input_tokens += usage.get("cache_creation_input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                    elif line_type == "user":
                        self._extract_tool_results(d, tool_uses, tool_calls, events, msg_index, ts)
                        self._extract_user_events(d, events, msg_index, ts)

        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Failed to read %s: %s", jsonl_path, e)
            return None

        for tc in tool_calls:
            if not any(e.type == "tool_call" and e.tool_call is tc for e in events):
                events.append(SessionEvent(type="tool_call", msg_index=tc.msg_index, tool_call=tc))
        events.sort(key=lambda e: e.msg_index)

        return SessionData(
            session_id=session_id,
            tool_calls=tool_calls,
            events=events,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            source=source,
        )

    def _extract_tool_uses(self, d: dict, tool_uses: dict[str, tuple[str, dict]]) -> None:
        """Extract tool_use blocks from an assistant message."""
        msg = d.get("message") or {}
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tc_id = block.get("id", "")
            name = block.get("name", "")
            inp = block.get("input", {})
            if tc_id and name:
                tool_uses[tc_id] = (name, inp if isinstance(inp, dict) else {})

    def _extract_tool_results(
        self,
        d: dict,
        tool_uses: dict[str, tuple[str, dict]],
        tool_calls: list[ToolCall],
        events: list[SessionEvent],
        msg_index: int,
        timestamp: str | None = None,
    ) -> None:
        """Extract tool_result blocks from a user message and match to tool_uses."""
        msg = d.get("message") or {}
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            tc_id = block.get("tool_use_id", "")
            result_content = block.get("content", "")
            if not isinstance(result_content, str):
                result_content = str(result_content)

            if tc_id not in tool_uses:
                continue

            name, inp = tool_uses[tc_id]

            explicit_error = block.get("is_error", False)
            detected_error = is_error_content(result_content)
            is_err = explicit_error or detected_error

            error_cat = classify_error(result_content) if is_err else ErrorCategory.UNKNOWN

            tc = ToolCall(
                name=name,
                tool_call_id=tc_id,
                input_data=inp,
                output=result_content,
                is_error=is_err,
                error_category=error_cat,
                msg_index=msg_index,
                output_bytes=len(result_content.encode("utf-8")),
            )
            tool_calls.append(tc)
            events.append(
                SessionEvent(
                    type="tool_call", msg_index=msg_index, timestamp=timestamp, tool_call=tc
                )
            )

            if name in ("Agent", "agent"):
                tool_result_meta = d.get("toolUseResult", {})
                if isinstance(tool_result_meta, dict):
                    events.append(
                        SessionEvent(
                            type="agent_summary",
                            msg_index=msg_index,
                            timestamp=timestamp,
                            agent_id=tool_result_meta.get("agentId", ""),
                            agent_tool_count=tool_result_meta.get("totalToolUseCount", 0),
                            agent_tokens=tool_result_meta.get("totalTokens", 0),
                            agent_duration_ms=tool_result_meta.get("totalDurationMs", 0),
                            agent_prompt=tool_result_meta.get("prompt", "")[:200],
                        )
                    )

    def _extract_user_events(
        self,
        d: dict,
        events: list[SessionEvent],
        msg_index: int,
        timestamp: str | None = None,
    ) -> None:
        """Extract user text messages and interruptions from a user line."""
        msg = d.get("message") or {}
        content = msg.get("content", "")

        if isinstance(content, str) and content.strip():
            events.append(
                SessionEvent(
                    type="user_message",
                    msg_index=msg_index,
                    timestamp=timestamp,
                    text=content[:500],
                )
            )
            return

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if "[Request interrupted by user" in text:
                        events.append(
                            SessionEvent(
                                type="interruption",
                                msg_index=msg_index,
                                timestamp=timestamp,
                                text=text[:200],
                            )
                        )


# =============================================================================
# Path Decode Helpers (Claude Code specific)
# =============================================================================


def _decode_windows_path(drive: str, parts: list[str]) -> Path | None:
    """Reconstruct a Windows path from drive letter + dash-split tokens.

    Empty tokens (from consecutive dashes in the encoded name) are dropped so
    the literal join never produces doubled separators.
    """
    tokens = [p for p in parts if p]
    if not tokens:
        return None
    win_path = Path(f"{drive}:\\" + "\\".join(tokens))
    if _path_exists(win_path):
        return win_path
    drive_root = Path(f"{drive}:\\")
    if _path_exists(drive_root):
        result = _greedy_path_decode(drive_root, tokens)
        if result:
            return result
    if tokens[0].lower() == "users":
        return win_path
    return None


def _decode_project_path(escaped_name: str) -> Path | None:
    """Decode a Claude Code escaped project path."""
    # Windows paths are encoded without a leading dash: "C:\Users\x" becomes
    # "C--Users-x" (":" and "\" each collapse to "-"). Older callers also pass
    # the legacy "-C-Users-x" form; accept both.
    win = re.match(r"^-?([A-Za-z])--?(.+)$", escaped_name)
    if win:
        result = _decode_windows_path(win.group(1).upper(), win.group(2).split("-"))
        if result is not None:
            return result
        if not escaped_name.startswith("-"):
            return None

    if not escaped_name.startswith("-"):
        return None

    parts = escaped_name[1:].split("-")
    if len(parts) < 2:
        return None

    simple = Path("/" + escaped_name[1:].replace("-", "/"))
    if _path_exists(simple):
        return simple

    if len(parts) < 3:
        return None

    if parts[0] in ("Users", "home") and len(parts) > 2:
        # Start the greedy decode at the mount root so a home-directory
        # component containing '.', '-' or '_' (e.g. "first.last", encoded as
        # "first-last") is matched as a single directory by tokenisation,
        # instead of being split into "/Users/first/last". Falls back to the
        # legacy single-token assumption if the rooted walk finds nothing.
        result = _greedy_path_decode(Path(f"/{parts[0]}"), parts[1:])
        if result is not None:
            return result
        base = Path(f"/{parts[0]}/{parts[1]}")
        return _greedy_path_decode(base, parts[2:])

    return None


def _checkout_root(path: Path) -> Path | None:
    """The nearest directory at or above ``path`` holding a ``.git`` entry."""
    try:
        return next((d for d in (path, *path.parents) if _path_exists(d / ".git")), None)
    except OSError:
        return None


def _main_worktree_root(path: Path) -> Path | None:
    """Return the main checkout of the linked git worktree ``path`` is in.

    Returns None for a main checkout, a submodule, a bare repo, or anything
    outside git. Reads git's own pointer files rather than running git.
    """
    # A relative path (a Windows path decoded on POSIX) would walk up into the
    # process's working directory and match whatever repo that sits in.
    if not path.is_absolute():
        return None
    try:
        checkout = _checkout_root(path)
        if checkout is None:
            return None
        # A main checkout's .git is a directory, so this read fails there.
        pointer = (checkout / ".git").read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir:"):
            return None
        gitdir = checkout / pointer[len("gitdir:") :].strip()
        # Only a linked worktree's gitdir has `commondir` (a submodule's does not).
        common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    except (OSError, UnicodeDecodeError):
        return None
    # A bare repo has no checkout to merge into.
    return common.parent if common.name == ".git" else None


def _project_display_name(project_path: Path, fallback: str) -> str:
    """Return a human project name for POSIX and Windows-style decoded paths."""
    rendered = str(project_path)
    if re.match(r"^[A-Za-z]:[\\/]", rendered):
        return PureWindowsPath(rendered).name or fallback
    if project_path == Path("/"):
        return fallback
    return project_path.name or fallback


def _greedy_path_decode(base: Path, parts: list[str]) -> Path | None:
    """Greedily decode remaining path parts using real child directories."""
    if not parts:
        return base if _path_exists(base) else None

    if not _path_exists(base) or not base.is_dir():
        return None

    try:
        entries = list(base.iterdir())
    except OSError:
        return None

    # Windows profiles routinely contain reparse-point junctions (e.g.
    # "AppData\Local\Temporary Internet Files") that raise PermissionError on
    # is_dir(). Skip those entries individually instead of letting one
    # inaccessible sibling abort the whole listing — and thus every project
    # path that happens to walk through this directory.
    children = []
    for entry in entries:
        try:
            if entry.is_dir():
                children.append(entry)
        except OSError:
            continue
    children.sort()

    for child in children:
        for tokenization in _component_tokenizations(child.name):
            n_tokens = len(tokenization)
            if parts[:n_tokens] != tokenization:
                continue

            result = _greedy_path_decode(child, parts[n_tokens:])
            if result:
                return result

    return None


def _component_tokenizations(component: str) -> list[list[str]]:
    """Return possible escaped token sequences for a real path component."""
    tokenizations: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(tokens: list[str]) -> None:
        key = tuple(tokens)
        if tokens and key not in seen:
            seen.add(key)
            tokenizations.append(tokens)

    add([component])

    for separator in (" ", "-", ".", "_", None):
        if separator is None:
            tokens = [token for token in re.split(r"[-.\s_]", component) if token]
        else:
            tokens = [token for token in component.split(separator) if token]
        add(tokens)

    if component.startswith(".") and len(component) > 1:
        hidden_component = component[1:]
        add(["", hidden_component])
        for separator in (" ", "-", ".", "_", None):
            if separator is None:
                tokens = [token for token in re.split(r"[-.\s_]", hidden_component) if token]
            else:
                tokens = [token for token in hidden_component.split(separator) if token]
            add(["", *tokens])

    return tokenizations


# Module-level instance for auto-discovery by the plugin registry
plugin = ClaudeCodePlugin()
