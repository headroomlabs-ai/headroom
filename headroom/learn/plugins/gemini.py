"""Google Gemini CLI plugin for headroom learn.

Reads session logs from ~/.gemini/tmp/<project_hash>/chats/ (JSON and JSONL).
"""

from __future__ import annotations

import json
import logging
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
from ..writer import ContextWriter, GeminiWriter
from ._paths import path_exists

logger = logging.getLogger(__name__)


def _optional_id(part: dict) -> str | None:
    """Return a call/response ID, treating malformed (non-string) IDs as missing."""
    value = part.get("id")
    return value if isinstance(value, str) else None


class GeminiPlugin(LearnPlugin, ConversationScanner):
    """Reads Google Gemini CLI session logs from ~/.gemini/tmp/<project>/chats/.

    Gemini CLI stores sessions as JSON or JSONL files with messages in the
    Gemini API format:
    - role: "user" or "model"
    - parts[]: array containing text, functionCall, or functionResponse objects

    Tool calls use:
    - functionCall: {name, args}  (in model messages)
    - functionResponse: {name, response}  (in user messages)
    """

    def __init__(self, gemini_dir: Path | None = None):
        self.gemini_dir = gemini_dir or Path.home() / ".gemini"
        self.tmp_dir = self.gemini_dir / "tmp"

    # --- LearnPlugin identity ---

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini CLI"

    @property
    def description(self) -> str:
        return "Google Gemini CLI (~/.gemini/)"

    def detect(self) -> bool:
        if not self.tmp_dir.exists():
            return False
        return bool(
            any(self.tmp_dir.rglob("session-*.json")) or any(self.tmp_dir.rglob("session-*.jsonl"))
        )

    def create_writer(self) -> ContextWriter:
        return GeminiWriter()

    # --- ConversationScanner interface ---

    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects with Gemini session data."""
        if not self.tmp_dir.exists():
            return []

        projects = []
        for project_dir in sorted(self.tmp_dir.iterdir()):
            if not project_dir.is_dir():
                continue

            chats_dir = project_dir / "chats"
            if not chats_dir.exists():
                continue

            session_files = list(chats_dir.glob("session-*.json")) + list(
                chats_dir.glob("session-*.jsonl")
            )
            if not session_files:
                continue

            project_path = self._detect_project_path(session_files[0])

            gemini_md = None
            if project_path and path_exists(project_path):
                candidate = project_path / "GEMINI.md"
                if path_exists(candidate):
                    gemini_md = candidate

            projects.append(
                ProjectInfo(
                    name=project_path.name if project_path else project_dir.name,
                    project_path=project_path or Path.cwd(),
                    data_path=chats_dir,
                    context_file=gemini_md,
                    memory_file=None,
                )
            )

        return projects

    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all Gemini session files for a project.

        ``include_subagents`` is accepted for a uniform plugin contract but is a
        no-op: Gemini stores sessions flat, with no nested transcript hierarchy.
        """
        session_files = sorted(project.data_path.glob("session-*.json")) + sorted(
            project.data_path.glob("session-*.jsonl")
        )
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

    def _scan_session(self, session_path: Path) -> SessionData | None:
        """Parse a single Gemini session file (JSON or JSONL)."""
        if session_path.suffix == ".jsonl":
            return self._scan_jsonl_session(session_path)
        return self._scan_json_session(session_path)

    def _scan_json_session(self, json_path: Path) -> SessionData | None:
        """Parse a Gemini JSON session file."""
        try:
            with open(json_path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to read Gemini session %s: %s", json_path, e)
            return None

        session_id = json_path.stem

        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict):
            messages = data.get("messages", data.get("history", data.get("contents", []))) or []
            session_id = str(data.get("id", data.get("session_id", session_id)))
        else:
            return None

        return self._parse_messages(session_id, messages)

    def _scan_jsonl_session(self, jsonl_path: Path) -> SessionData | None:
        """Parse a Gemini JSONL session file."""
        session_id = jsonl_path.stem
        messages: list[dict] = []

        try:
            with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    entry_type = entry.get("type", "")

                    if entry_type == "session_metadata":
                        session_id = entry.get("id", entry.get("session_id", session_id))
                        continue

                    role = entry.get("role", "")
                    if not role:
                        if entry_type in ("user", "gemini", "model"):
                            role = "model" if entry_type == "gemini" else entry_type
                        else:
                            continue

                    parts = entry.get("parts", [])
                    if parts:
                        messages.append({"role": role, "parts": parts})

        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Failed to read Gemini session %s: %s", jsonl_path, e)
            return None

        return self._parse_messages(session_id, messages)

    def _parse_messages(self, session_id: str, messages: list) -> SessionData | None:
        """Parse Gemini API format messages into normalized SessionData."""
        # name -> pending calls as (provider id, normalized name, args, msg index, ordinal)
        tool_calls_pending: dict[str, list[tuple[str | None, str, dict, int, int]]] = {}
        # name -> msg index of the latest model turn that called it
        last_call_msg: dict[str, int] = {}
        # (msg index, name) -> calls seen so far, to keep same-turn call ids distinct
        call_ordinals: dict[tuple[int, str], int] = {}
        tool_calls: list[ToolCall] = []
        events: list[SessionEvent] = []
        msg_index = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            msg_index += 1
            role = msg.get("role", "")
            parts = msg.get("parts", [])

            usage = msg.get("usageMetadata", msg.get("usage", {}))
            if isinstance(usage, dict):
                # Gemini's promptTokenCount is the FULL input token count and
                # cachedContentTokenCount is the cached SUBSET of it, so adding
                # both double-counts the cached input. Likewise totalTokenCount
                # == promptTokenCount + candidatesTokenCount, so
                # (totalTokenCount - promptTokenCount) is just candidatesTokenCount
                # again — adding it on top double-counts the output. Count the
                # prompt as input and the candidates as output, once each.
                total_input_tokens += usage.get("promptTokenCount", 0)
                total_output_tokens += usage.get("candidatesTokenCount", 0)

            if not isinstance(parts, list):
                continue

            # Calls answered by ID somewhere in this message are reserved for that
            # exact match, so an earlier ID-less response can't take them by FIFO.
            reserved_ids = {
                (part["functionResponse"].get("name", ""), _optional_id(part["functionResponse"]))
                for part in parts
                if isinstance(part, dict)
                and isinstance(part.get("functionResponse"), dict)
                and _optional_id(part["functionResponse"]) is not None
            }

            for part in parts:
                if not isinstance(part, dict):
                    continue

                if role == "user" and "text" in part:
                    text = part["text"]
                    if isinstance(text, str) and text.strip():
                        events.append(
                            SessionEvent(
                                type="user_message",
                                msg_index=msg_index,
                                text=text[:500],
                            )
                        )

                if "functionCall" in part:
                    fc = part["functionCall"]
                    if isinstance(fc, dict):
                        name = fc.get("name", "")
                        args = fc.get("args", {})
                        if not isinstance(args, dict):
                            args = {}
                        normalized_name = normalize_tool_name(name)
                        if name:
                            ordinal = call_ordinals.get((msg_index, name), 0)
                            call_ordinals[(msg_index, name)] = ordinal + 1
                            last_call_msg[name] = msg_index
                            tool_calls_pending.setdefault(name, []).append(
                                (_optional_id(fc), normalized_name, args, msg_index, ordinal)
                            )

                if "functionResponse" in part:
                    fr = part["functionResponse"]
                    if isinstance(fr, dict):
                        name = fr.get("name", "")
                        response = fr.get("response", {})

                        if isinstance(response, dict):
                            result_content = response.get("output", response.get("result", ""))
                            if not isinstance(result_content, str):
                                result_content = json.dumps(response)
                        elif isinstance(response, str):
                            result_content = response
                        else:
                            result_content = str(response)

                        pending = tool_calls_pending.get(name, [])
                        response_id = _optional_id(fr)
                        # Prefer an exact ID match; use FIFO only when either side lacks an ID.
                        match_idx = next(
                            (
                                i
                                for i, (call_id, *_) in enumerate(pending)
                                if response_id is not None and call_id == response_id
                            ),
                            None,
                        )
                        if match_idx is None:
                            # FIFO only within the latest turn that called this tool, so an
                            # unanswered call from an earlier turn can't take a later result.
                            match_idx = next(
                                (
                                    i
                                    for i, (call_id, _, _, idx, _) in enumerate(pending)
                                    if idx == last_call_msg.get(name)
                                    and (response_id is None or call_id is None)
                                    and (call_id is None or (name, call_id) not in reserved_ids)
                                ),
                                None,
                            )
                        ordinal = 0
                        if match_idx is not None:
                            _, normalized_name, args, call_idx, ordinal = pending.pop(match_idx)
                        else:
                            normalized_name = normalize_tool_name(name)
                            args = {}
                            call_idx = msg_index

                        call_id = f"{session_id}_{call_idx}_{name}"
                        if ordinal:
                            # Same-name calls in one turn need distinct ids, or loop
                            # detection drops them as replays.
                            call_id = f"{call_id}_{ordinal}"
                        is_err = is_error_content(result_content)
                        error_cat = (
                            classify_error(result_content) if is_err else ErrorCategory.UNKNOWN
                        )

                        tc = ToolCall(
                            name=normalized_name,
                            tool_call_id=call_id,
                            input_data=args,
                            output=result_content,
                            is_error=is_err,
                            error_category=error_cat,
                            msg_index=msg_index,
                            output_bytes=len(result_content.encode("utf-8")),
                        )
                        tool_calls.append(tc)
                        events.append(
                            SessionEvent(
                                type="tool_call",
                                msg_index=msg_index,
                                tool_call=tc,
                            )
                        )

        events.sort(key=lambda e: e.msg_index)

        return SessionData(
            session_id=session_id,
            tool_calls=tool_calls,
            events=events,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

    def _detect_project_path(self, session_path: Path) -> Path | None:
        """Try to detect the project path from a session file (JSON or JSONL)."""
        # A `.jsonl` session is a stream of one JSON object per line, so
        # `json.load` on the whole file raises JSONDecodeError on the second
        # line and detection silently fell back to cwd — writing the learned
        # insights to the wrong project and missing its GEMINI.md. Read JSONL
        # line-by-line like the sibling `_scan_jsonl_session` (and the Claude
        # plugin's `_project_path_from_session_cwd`) do.
        if session_path.suffix == ".jsonl":
            return self._detect_project_path_jsonl(session_path)

        try:
            with open(session_path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if isinstance(data, dict):
            return self._project_path_from_entry(data)

        return None

    def _detect_project_path_jsonl(self, session_path: Path) -> Path | None:
        try:
            with open(session_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    found = self._project_path_from_entry(entry)
                    if found is not None:
                        return found
        except (OSError, UnicodeDecodeError):
            return None
        return None

    @staticmethod
    def _project_path_from_entry(entry: dict) -> Path | None:
        project_path = entry.get("projectPath", entry.get("project_path", ""))
        if project_path and path_exists(Path(project_path)):
            return Path(project_path)
        cwd = entry.get("cwd", entry.get("workingDirectory", ""))
        if cwd and path_exists(Path(cwd)):
            return Path(cwd)
        return None


# Module-level instance for auto-discovery by the plugin registry
plugin = GeminiPlugin()
