"""Grok CLI plugin for headroom learn.

Reads session logs from ~/.grok/sessions/<workspace>/<session-id>/updates.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import unquote

from .._shared import classify_error, is_error_content, normalize_tool_name
from ..base import ConversationScanner, LearnPlugin
from ..models import ErrorCategory, ProjectInfo, SessionData, ToolCall
from ..writer import ContextWriter, GrokWriter

logger = logging.getLogger(__name__)


class GrokPlugin(LearnPlugin, ConversationScanner):
    """Reads Grok CLI session logs from ~/.grok/sessions/."""

    def __init__(self, grok_dir: Path | None = None):
        self.grok_dir = grok_dir or Path.home() / ".grok"
        self.sessions_dir = self.grok_dir / "sessions"

    @property
    def name(self) -> str:
        return "grok"

    @property
    def display_name(self) -> str:
        return "Grok CLI"

    @property
    def description(self) -> str:
        return "Grok CLI (~/.grok/sessions/)"

    def detect(self) -> bool:
        if not self.sessions_dir.exists():
            return False
        return any(self.sessions_dir.rglob("updates.jsonl"))

    def create_writer(self) -> ContextWriter:
        return GrokWriter()

    def discover_projects(self) -> list[ProjectInfo]:
        if not self.sessions_dir.exists():
            return []

        projects: list[ProjectInfo] = []
        for workspace_dir in sorted(self.sessions_dir.iterdir()):
            if not workspace_dir.is_dir():
                continue
            session_files = list(workspace_dir.glob("*/updates.jsonl"))
            if not session_files:
                continue

            decoded = unquote(workspace_dir.name)
            project_path = Path(decoded) if decoded.startswith("/") else Path.cwd()
            agents_md = project_path / "AGENTS.md"
            grok_md = project_path / "GROK.md"

            projects.append(
                ProjectInfo(
                    name=workspace_dir.name,
                    project_path=project_path,
                    data_path=workspace_dir,
                    context_file=grok_md
                    if grok_md.exists()
                    else agents_md
                    if agents_md.exists()
                    else None,
                )
            )
        return projects

    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        del include_subagents
        session_files = sorted(project.data_path.glob("*/updates.jsonl"))
        if not session_files:
            return []

        if max_workers <= 1 or len(session_files) <= 1:
            return [s for f in session_files if (s := self._scan_session(f)) and s.tool_calls]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        sessions: list[SessionData] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._scan_session, f): f for f in session_files}
            for future in as_completed(futures):
                session = future.result()
                if session and session.tool_calls:
                    sessions.append(session)
        return sessions

    def _scan_session(self, jsonl_path: Path) -> SessionData | None:
        session_id = jsonl_path.parent.name
        pending_calls: dict[str, tuple[str, dict]] = {}
        tool_calls: list[ToolCall] = []
        msg_index = 0

        try:
            with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    params = entry.get("params", {})
                    if not isinstance(params, dict):
                        continue
                    update = params.get("update", {})
                    if not isinstance(update, dict):
                        continue

                    session_update = update.get("sessionUpdate", "")
                    if session_update == "tool_call":
                        msg_index += 1
                        call_id = str(update.get("toolCallId", ""))
                        title = str(update.get("title", "tool"))
                        raw_input = update.get("rawInput", {})
                        if not isinstance(raw_input, dict):
                            raw_input = {"raw": raw_input}
                        name = normalize_tool_name(title)
                        if call_id:
                            pending_calls[call_id] = (name, raw_input)
                        continue

                    if session_update != "tool_call_update":
                        continue

                    status = update.get("status")
                    if status not in ("completed", "failed"):
                        continue

                    call_id = str(update.get("toolCallId", ""))
                    if call_id not in pending_calls:
                        continue

                    msg_index += 1
                    name, inp = pending_calls[call_id]
                    result_content = _extract_tool_output(update)
                    is_err = status == "failed" or is_error_content(result_content)
                    error_cat = classify_error(result_content) if is_err else ErrorCategory.UNKNOWN

                    tool_calls.append(
                        ToolCall(
                            name=name,
                            tool_call_id=call_id,
                            input_data=inp,
                            output=result_content,
                            is_error=is_err,
                            error_category=error_cat,
                            msg_index=msg_index,
                            output_bytes=len(result_content.encode("utf-8")),
                        )
                    )
        except OSError as exc:
            logger.debug("Failed to read Grok session %s: %s", jsonl_path, exc)
            return None

        if not tool_calls:
            return None
        return SessionData(session_id=session_id, tool_calls=tool_calls)


def _extract_tool_output(update: dict) -> str:
    raw_output = update.get("rawOutput")
    if isinstance(raw_output, dict):
        for key in ("output_for_prompt", "output", "FileContent"):
            value = raw_output.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                content = value.get("content")
                if isinstance(content, str) and content.strip():
                    return content

    content_blocks = update.get("content")
    if isinstance(content_blocks, list):
        parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            inner = block.get("content")
            if isinstance(inner, dict):
                text = inner.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    return ""


plugin = GrokPlugin()
