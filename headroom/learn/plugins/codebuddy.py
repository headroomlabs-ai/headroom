"""CodeBuddy plugin for headroom learn.

Reads conversation logs from ~/.codebuddy/projects/ (JSONL format).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PureWindowsPath

from .._shared import classify_error, is_error_content
from ..base import ConversationScanner, LearnPlugin
from ..models import (
    ErrorCategory,
    ProjectInfo,
    SessionData,
    SessionEvent,
    ToolCall,
)
from ..writer import CodeBuddyWriter, ContextWriter

logger = logging.getLogger(__name__)


class CodeBuddyPlugin(LearnPlugin, ConversationScanner):
    """Reads CodeBuddy conversation logs from ~/.codebuddy/projects/.

    CodeBuddy stores conversations as JSONL files with these line types:
    - type="assistant": message.content[] has tool_use blocks (name, input, id)
    - type="user": message.content[] has tool_result blocks (tool_use_id, content)
    """

    def __init__(self, codebuddy_dir: Path | None = None):
        self.codebuddy_dir = codebuddy_dir or Path.home() / ".codebuddy"
        self.projects_dir = self.codebuddy_dir / "projects"

    # --- LearnPlugin identity ---

    @property
    def name(self) -> str:
        return "codebuddy"

    @property
    def display_name(self) -> str:
        return "CodeBuddy"

    @property
    def description(self) -> str:
        return "CodeBuddy (~/.codebuddy/)"

    def detect(self) -> bool:
        return self.projects_dir.exists() and any(self.projects_dir.iterdir())

    def create_writer(self) -> ContextWriter:
        return CodeBuddyWriter()

    # --- ConversationScanner interface ---

    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects under ~/.codebuddy/projects/."""
        if not self.projects_dir.exists():
            return []

        projects = []
        for entry in sorted(self.projects_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            project_path = _decode_project_path(entry.name)
            if project_path is None:
                fallback_parts = entry.name[1:].split("-")
                if len(fallback_parts[0]) == 1 and fallback_parts[0].isalpha():
                    drive = fallback_parts[0].upper()
                    project_path = Path(f"{drive}:\\" + "\\".join(fallback_parts[1:]))
                else:
                    project_path = Path("/" + entry.name[1:].replace("-", "/"))

            name = _project_display_name(project_path, entry.name)

            context_file = None
            if project_path.exists():
                cb_md = project_path / "CODEBUDDY.md"
                if cb_md.exists():
                    context_file = cb_md

            memory_dir = entry / "memory"
            memory_file = memory_dir / "MEMORY.md" if memory_dir.exists() else None
            if memory_file and not memory_file.exists():
                memory_file = None

            jsonl_files = list(entry.glob("*.jsonl"))
            if not jsonl_files:
                continue

            projects.append(
                ProjectInfo(
                    name=name,
                    project_path=project_path,
                    data_path=entry,
                    context_file=context_file,
                    memory_file=memory_file,
                )
            )

        return projects

    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all conversation JSONL files for a project."""
        jsonl_files = sorted(project.data_path.glob("*.jsonl"))
        if not jsonl_files:
            return []

        if max_workers <= 1 or len(jsonl_files) <= 1:
            return [s for f in jsonl_files if (s := self._scan_session(f)) and s.tool_calls]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        sessions: list[SessionData] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._scan_session, f): f for f in jsonl_files}
            for future in as_completed(futures):
                session = future.result()
                if session and session.tool_calls:
                    sessions.append(session)
        return sessions

    def _scan_session(self, jsonl_path: Path) -> SessionData | None:
        """Scan a single JSONL conversation file."""
        session_id = jsonl_path.stem
        tool_uses: dict[str, tuple[str, dict]] = {}
        tool_calls: list[ToolCall] = []
        events: list[SessionEvent] = []
        total_input_tokens = 0
        total_output_tokens = 0
        msg_index = 0

        try:
            with open(jsonl_path) as f:
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
                        usage = d.get("message", {}).get("usage", {})
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
        )

    def _extract_tool_uses(self, d: dict, tool_uses: dict[str, tuple[str, dict]]) -> None:
        """Extract tool_use blocks from an assistant message."""
        msg = d.get("message", {})
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
        msg = d.get("message", {})
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

    def _extract_user_events(
        self,
        d: dict,
        events: list[SessionEvent],
        msg_index: int,
        timestamp: str | None = None,
    ) -> None:
        """Extract user text messages and interruptions from a user line."""
        msg = d.get("message", {})
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
# Path Decode Helpers (CodeBuddy specific)
# =============================================================================


def _decode_project_path(escaped_name: str) -> Path | None:
    """Decode a CodeBuddy escaped project path."""
    if not escaped_name.startswith("-"):
        return None

    parts = escaped_name[1:].split("-")
    if len(parts) < 2:
        return None

    if len(parts[0]) == 1 and parts[0].isalpha():
        drive = parts[0].upper()
        win_path = Path(f"{drive}:\\" + "\\".join(parts[1:]))
        if win_path.exists():
            return win_path

    simple = Path("/" + escaped_name[1:].replace("-", "/"))
    if simple.exists():
        return simple

    if len(parts) < 3:
        return None

    if parts[0] in ("Users", "home") and len(parts) > 2:
        base = Path(f"/{parts[0]}/{parts[1]}")
        return _greedy_path_decode(base, parts[2:])

    return None


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
        return base if base.exists() else None

    if not base.exists() or not base.is_dir():
        return None

    try:
        children = sorted(child for child in base.iterdir() if child.is_dir())
    except OSError:
        return None

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

    for separator in ("-", ".", "_", None):
        if separator is None:
            tokens = [token for token in re.split(r"[-._]", component) if token]
        else:
            tokens = [token for token in component.split(separator) if token]
        add(tokens)

    if component.startswith(".") and len(component) > 1:
        hidden_component = component[1:]
        add(["", hidden_component])
        for separator in ("-", ".", "_", None):
            if separator is None:
                tokens = [token for token in re.split(r"[-._]", hidden_component) if token]
            else:
                tokens = [token for token in hidden_component.split(separator) if token]
            add(["", *tokens])

    return tokenizations


# Module-level instance for auto-discovery by the plugin registry
plugin = CodeBuddyPlugin()
