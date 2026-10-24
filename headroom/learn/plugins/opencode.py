"""OpenCode plugin for headroom learn.

Reads conversation data from the OpenCode SQLite database at
``~/.local/share/opencode/opencode.db`` or
``~/.local/share/opencode/opencode-local.db``. Set
``HEADROOM_OPENCODE_DB`` to force a specific database path.

Supports both schemas:
- Legacy (opencode v1): ``message`` + ``part`` tables, where tool parts look
  like ``{"type": "tool", "tool": "bash", "callID": ..., "state":
  {"status": ..., "input": {...}, "output": "..."}}``.
- Current (opencode v2, >= 2026-03): ``session_v2`` + ``session_message``
  tables, where each message row holds JSON ``data`` with a ``content`` list
  and tool entries look like ``{"type": "tool", "id": ..., "name": "bash",
  "state": {"status": ..., "input": {...}, "content": [{"type": "text",
  "text": "..."}], "error": {...}}}``.

``headroom learn opencode`` mines these for errors and writes corrections to
the project's ``AGENTS.md`` file (OpenCode's native rules file).
"""

from __future__ import annotations

import json
import logging
import os
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
from ._paths import path_exists

logger = logging.getLogger(__name__)

_OPENCODE_DIR = Path.home() / ".local" / "share" / "opencode"
_OPENCODE_DB = _OPENCODE_DIR / "opencode.db"
_OPENCODE_DB_ENV = "HEADROOM_OPENCODE_DB"
_OPENCODE_DB_CANDIDATES = ("opencode-local.db", "opencode.db")

# Tool part status values that indicate failure.
_ERROR_STATUSES = {"error", "failed", "aborted"}


def _tables(conn: sqlite3.Connection) -> set[str]:
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {r[0] for r in cur.fetchall()}
    except sqlite3.Error:
        return set()


def _session_table(tables: set[str]) -> str | None:
    # Prefer the v2 schema used by current opencode releases, fall back to
    # the legacy table name.
    if "session_v2" in tables:
        return "session_v2"
    if "session" in tables:
        return "session"
    return None


def _extract_output(state: dict) -> str:
    """Extract human-readable tool output from either schema shape.

    Combines all available sources: the legacy ``state.output`` string, the
    v2 ``state.content`` text blocks, and ``state.error`` details. The error
    is always included when present — even alongside content blocks — so a
    failed call keeps its failure message for error classification.
    """
    parts: list[str] = []
    # Legacy shape: state.output is a plain string.
    if isinstance(state.get("output"), str) and state.get("output"):
        parts.append(str(state.get("output") or ""))
    # Current shape: state.content is a list of {type, text} blocks.
    content = state.get("content")
    if isinstance(content, list) and content:
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            t = item.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
            elif item.get("type") != "text":
                parts.append(json.dumps(item)[:500])
    elif content is not None and not isinstance(content, list):
        parts.append(str(content))
    # Error shape: state.error = {type, message}. Surface it as output so
    # error classification sees the message even when content is empty —
    # or when content holds only pre-failure progress output.
    err = state.get("error")
    if isinstance(err, dict) and err:
        etype = str(err.get("type") or "error")
        msg = str(err.get("message") or "")
        combined = f"{etype}: {msg}".strip(": ").strip()
        parts.append(combined if combined else json.dumps(err)[:2000])
    elif isinstance(err, str) and err:
        parts.append(err)
    return "\n".join(p for p in parts if p)


class OpenCodePlugin(LearnPlugin, ConversationScanner):
    """Read OpenCode sessions from the SQLite database.

    OpenCode stores all conversation data in a single SQLite file which makes
    discovery fast — one file, many projects.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = self._resolve_db_path(db_path)

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
        return (
            "OpenCode (~/.local/share/opencode/{opencode.db,opencode-local.db}; "
            "HEADROOM_OPENCODE_DB overrides)"
        )

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
            tables = _tables(conn)
            sess_tbl = _session_table(tables)
            if sess_tbl is None:
                logger.warning("OpenCode DB %s has no session table", self._db_path)
                return []
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"SELECT p.id, p.name, p.worktree "
                    f"FROM project p "
                    f"WHERE EXISTS (SELECT 1 FROM {sess_tbl} s WHERE s.project_id = p.id)"
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                cursor.execute(
                    f"SELECT p.id, NULL, p.worktree "
                    f"FROM project p "
                    f"WHERE EXISTS (SELECT 1 FROM {sess_tbl} s WHERE s.project_id = p.id)"
                )
                rows = cursor.fetchall()
            projects: list[ProjectInfo] = []
            for row in rows:
                proj_id, proj_name, worktree = row
                worktree_path = Path(worktree) if worktree else Path("~").expanduser()
                agents_md = worktree_path / "AGENTS.md"
                projects.append(
                    ProjectInfo(
                        name=proj_name or worktree_path.name or proj_id,
                        project_path=worktree_path,
                        data_path=self._db_path.parent,
                        context_file=agents_md if path_exists(agents_md) else None,
                    )
                )
        except sqlite3.Error as exc:
            logger.warning("Cannot discover OpenCode projects: %s", exc)
            return []
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
        tables = _tables(conn)
        sess_tbl = _session_table(tables)
        if sess_tbl is None:
            return []

        cursor.execute(
            "SELECT id FROM project WHERE worktree = ?",
            (str(project.project_path),),
        )
        row = cursor.fetchone()
        if row is None:
            return []
        project_db_id: str = row[0]

        cursor.execute(
            f"SELECT id, time_created FROM {sess_tbl} WHERE project_id = ? ORDER BY time_created DESC LIMIT 200",
            (project_db_id,),
        )
        session_rows = cursor.fetchall()

        sessions: list[SessionData] = []
        # The message layout follows the session schema: v2 sessions live in
        # session_message, legacy ones in part/message. Resolve once per scan
        # instead of per session (up to 200 sqlite_master lookups per project).
        prefer_v2 = sess_tbl == "session_v2"
        for session_id, time_created in session_rows:
            try:
                session = self._scan_session(
                    conn, session_id, time_created, tables=tables, prefer_v2=prefer_v2
                )
            except sqlite3.Error as exc:
                logger.warning("Cannot scan OpenCode session %s: %s", session_id, exc)
                continue
            if session is not None and session.tool_calls:
                sessions.append(session)

        return sessions

    @staticmethod
    def _resolve_db_path(db_path: Path | None) -> Path:
        if db_path is not None:
            return db_path

        override = os.getenv(_OPENCODE_DB_ENV)
        if override is not None:
            return Path(override).expanduser()

        ranked_candidates: list[tuple[int, int, Path]] = []
        for name in _OPENCODE_DB_CANDIDATES:
            candidate = _OPENCODE_DIR / name
            try:
                mtime_ns = candidate.stat().st_mtime_ns
            except OSError:
                continue
            ranked_candidates.append((mtime_ns, 1 if name == "opencode.db" else 0, candidate))

        if ranked_candidates:
            return max(ranked_candidates)[2]

        return _OPENCODE_DB

    def _scan_session(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        time_created: int | None,
        tables: set[str] | None = None,
        prefer_v2: bool = True,
    ) -> SessionData | None:
        """Scan one session, trying the matching message layout first.

        ``prefer_v2`` follows the session schema the ID was discovered from.
        When the preferred layout yields no tool calls (partially migrated
        databases can hold both families), the other layout is tried before
        giving up, so mixed-schema sessions don't silently lose their calls.
        """
        if tables is None:
            tables = _tables(conn)
        attempts = []
        if "session_message" in tables:
            attempts.append((True, self._scan_session_v2))
        if "part" in tables and "message" in tables:
            attempts.append((False, self._scan_session_legacy))
        if not attempts:
            logger.warning("OpenCode DB has neither session_message nor part/message tables")
            return None
        attempts.sort(key=lambda attempt: attempt[0] != prefer_v2)
        for _, scan in attempts:
            session = scan(conn, session_id, time_created)
            if session is not None and session.tool_calls:
                return session
        return None

    def _scan_session_v2(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        time_created: int | None,
    ) -> SessionData | None:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT data, time_created FROM session_message "
            "WHERE session_id = ? AND data LIKE '%\"type\"%tool%' "
            "ORDER BY seq",
            (session_id,),
        )
        rows = cursor.fetchall()

        tool_calls: list[ToolCall] = []
        events: list[SessionEvent] = []
        idx = 0
        for data_raw, msg_time in rows:
            try:
                data = json.loads(data_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            # data_raw is TEXT; a stored "null"/"[]" parses to a non-dict.
            if not isinstance(data, dict):
                continue
            content = data.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict) or item.get("type") != "tool":
                    continue
                raw_tool_name = str(item.get("name") or item.get("tool") or "unknown")
                tool_name = (
                    "Bash" if raw_tool_name.lower() == "bash" else normalize_tool_name(raw_tool_name)
                )
                call_id = str(item.get("id") or item.get("callID") or f"oc_{session_id}_{idx}")
                state_raw = item.get("state")
                state: dict = state_raw if isinstance(state_raw, dict) else {}
                status = str(state.get("status") or "unknown")
                input_raw = state.get("input")
                input_data: dict = input_raw if isinstance(input_raw, dict) else {}
                output: str = _extract_output(state)

                if output == "...output truncated..." or output.startswith("...output truncated"):
                    truncated_ref = (
                        state.get("outputRef") or state.get("outputFile") or state.get("outputRefFile")
                    )
                    if truncated_ref:
                        try:
                            output = Path(str(truncated_ref)).read_text(errors="replace")
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
                part_ts = str(msg_time) if msg_time is not None else None
                events.append(
                    SessionEvent(type="tool_call", msg_index=idx, timestamp=part_ts, tool_call=tc)
                )
                idx += 1

        if not tool_calls:
            return None

        ts = _to_datetime(time_created)
        return SessionData(
            session_id=session_id,
            tool_calls=tool_calls,
            events=events,
            timestamp=ts,
        )

    def _scan_session_legacy(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        time_created: int | None,
    ) -> SessionData | None:
        cursor = conn.cursor()

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

            raw_tool_name = str(data.get("tool") or data.get("name") or "unknown")
            tool_name = (
                "Bash" if raw_tool_name.lower() == "bash" else normalize_tool_name(raw_tool_name)
            )
            call_id = str(data.get("callID") or data.get("id") or f"oc_{session_id}_{idx}")
            state_raw = data.get("state")
            state: dict = state_raw if isinstance(state_raw, dict) else {}
            status = str(state.get("status") or "unknown")
            input_raw = state.get("input")
            input_data: dict = input_raw if isinstance(input_raw, dict) else {}
            output: str = _extract_output(state)

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

        ts = _to_datetime(time_created)
        return SessionData(
            session_id=session_id,
            tool_calls=tool_calls,
            events=events,
            timestamp=ts,
        )


def _to_datetime(time_created: int | None) -> datetime | None:
    if time_created is None:
        return None
    try:
        return datetime.fromtimestamp(time_created / 1000, tz=timezone.utc)
    except (OSError, OverflowError, ValueError, TypeError):
        return None


# Module-level instance — auto-discovered by the learn plugin registry.
plugin = OpenCodePlugin()
