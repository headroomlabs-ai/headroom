"""OpenCode plugin for headroom learn.

Reads conversation data from the OpenCode SQLite database at
``~/.local/share/opencode/opencode.db``.

OpenCode stores messages and tool parts in two tables:
- ``message``: one row per turn (user/assistant), with JSON ``data``
- ``part``: one row per content part; tool parts have ``data.type == "tool"``

A tool part looks like::

    {
        "type": "tool",
        "tool": "bash",
        "callID": "toolu_01...",
        "state": {
            "status": "completed" | "error",
            "input": { ... },
            "output": "..."
        }
    }

``headroom learn opencode`` mines these for errors and writes corrections to
the project's ``AGENTS.md`` file (OpenCode's native rules file).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .._shared import classify_error, is_error_content, normalize_tool_name
from ..base import ConversationScanner, LearnPlugin
from ..models import (
    ErrorCategory,
    ProjectInfo,
    SessionData,
    SessionEvent,
    ToolCall,
)
from ..writer import CodexWriter, ContextWriter

logger = logging.getLogger(__name__)

_OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"

# Tool part status values that indicate failure.
_ERROR_STATUSES = {"error", "failed", "aborted"}


class OpenCodePlugin(LearnPlugin, ConversationScanner):
    """Read OpenCode sessions from the SQLite database.

    OpenCode stores all conversation data in a single SQLite file which makes
    discovery fast — one file, many projects.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _OPENCODE_DB

    # ------------------------------------------------------------------
    # LearnPlugin identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "opencode"

    @property
    def display_name(self) -> str:
        return "OpenCode"

    @property
    def description(self) -> str:
        return "OpenCode (~/.local/share/opencode/opencode.db)"

    def detect(self) -> bool:
        return self._db_path.exists()

    def create_writer(self) -> ContextWriter:
        # Re-use CodexWriter — it writes to AGENTS.md, which is exactly what
        # OpenCode reads.
        return CodexWriter()

    # ------------------------------------------------------------------
    # ConversationScanner interface
    # ------------------------------------------------------------------

    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects that have at least one session."""
        if not self.detect():
            return []

        try:
            conn = sqlite3.connect(str(self._db_path))
        except sqlite3.Error as exc:
            logger.warning("Cannot open OpenCode DB %s: %s", self._db_path, exc)
            return []

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT p.id, p.name, p.worktree "
                "FROM project p "
                "WHERE EXISTS (SELECT 1 FROM session s WHERE s.project_id = p.id)"
            )
            projects: list[ProjectInfo] = []
            for row in cursor.fetchall():
                proj_id, proj_name, worktree = row
                worktree_path = Path(worktree) if worktree else Path("~").expanduser()
                agents_md = worktree_path / "AGENTS.md"
                projects.append(
                    ProjectInfo(
                        name=proj_name or worktree_path.name or proj_id,
                        project_path=worktree_path,
                        data_path=self._db_path.parent,
                        context_file=agents_md if agents_md.exists() else None,
                    )
                )
        finally:
            conn.close()

        return projects

    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all sessions for a project and return normalized tool calls."""
        if not self.detect():
            return []

        try:
            conn = sqlite3.connect(str(self._db_path))
        except sqlite3.Error as exc:
            logger.warning("Cannot open OpenCode DB: %s", exc)
            return []

        try:
            return self._scan_project_sessions(conn, project)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_project_sessions(
        self, conn: sqlite3.Connection, project: ProjectInfo
    ) -> list[SessionData]:
        cursor = conn.cursor()

        # Find the project ID from the worktree path.
        cursor.execute(
            "SELECT id FROM project WHERE worktree = ?",
            (str(project.project_path),),
        )
        row = cursor.fetchone()
        if row is None:
            return []
        project_db_id: str = row[0]

        # Get all sessions for this project.
        cursor.execute(
            "SELECT id, time_created FROM session WHERE project_id = ? ORDER BY time_created DESC LIMIT 200",
            (project_db_id,),
        )
        session_rows = cursor.fetchall()

        sessions: list[SessionData] = []
        for session_id, time_created in session_rows:
            session = self._scan_session(conn, session_id, time_created)
            if session is not None and session.tool_calls:
                sessions.append(session)

        return sessions

    def _scan_session(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        time_created: int | None,
    ) -> SessionData | None:
        cursor = conn.cursor()

        # Get all tool parts for this session ordered by creation time.
        cursor.execute(
            """
            SELECT p.data, p.time_created
            FROM   part p
            JOIN   message m ON p.message_id = m.id
            WHERE  m.session_id = ?
            AND    p.data LIKE '%"type"%tool%'
            ORDER  BY p.time_created
            """,
            (session_id,),
        )
        rows = cursor.fetchall()

        tool_calls: list[ToolCall] = []
        events: list[SessionEvent] = []

        for idx, (data_raw, part_time) in enumerate(rows):
            try:
                data = json.loads(data_raw)
            except json.JSONDecodeError:
                continue

            if data.get("type") != "tool":
                continue

            raw_tool_name = str(data.get("tool") or "unknown")
            tool_name = (
                "Bash" if raw_tool_name.lower() == "bash" else normalize_tool_name(raw_tool_name)
            )
            call_id = str(data.get("callID") or f"oc_{session_id}_{idx}")
            state_raw = data.get("state")
            state: dict = state_raw if isinstance(state_raw, dict) else {}
            status = str(state.get("status") or "unknown")
            input_raw = state.get("input")
            input_data: dict = input_raw if isinstance(input_raw, dict) else {}
            output: str = str(state.get("output") or "")

            # Detect truncated output pointer.
            if output == "...output truncated..." or output.startswith("...output truncated"):
                truncated_ref = state.get("outputRef") or state.get("outputFile")
                if truncated_ref:
                    try:
                        output = Path(truncated_ref).read_text(errors="replace")
                    except OSError:
                        pass

            is_error = status in _ERROR_STATUSES or is_error_content(output)
            error_cat = classify_error(output) if is_error else ErrorCategory.UNKNOWN

            tc = ToolCall(
                name=tool_name,
                tool_call_id=call_id,
                input_data=input_data,
                output=output,
                is_error=is_error,
                error_category=error_cat,
                msg_index=idx,
                output_bytes=len(output.encode()),
            )
            tool_calls.append(tc)
            part_ts = str(part_time) if part_time is not None else None
            events.append(
                SessionEvent(type="tool_call", msg_index=idx, timestamp=part_ts, tool_call=tc)
            )

        if not tool_calls:
            return None

        ts: datetime | None = None
        if time_created is not None:
            try:
                ts = datetime.fromtimestamp(time_created / 1000, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                pass

        return SessionData(
            session_id=session_id,
            tool_calls=tool_calls,
            events=events,
            timestamp=ts,
        )


# Module-level instance — auto-discovered by the learn plugin registry.
plugin = OpenCodePlugin()
