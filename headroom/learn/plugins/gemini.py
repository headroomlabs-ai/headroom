"""Google Gemini CLI & Antigravity plugin for headroom learn.

Reads session logs from ~/.gemini/tmp/<project_hash>/chats/ (JSON and JSONL),
as well as Google Antigravity CLI and IDE brain transcripts (~/.gemini/antigravity-*/brain/).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
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


class GeminiPlugin(LearnPlugin, ConversationScanner):
    """Reads Google Gemini CLI & Antigravity session logs from ~/.gemini/.

    Supports:
    - Gemini CLI sessions in ~/.gemini/tmp/<project>/chats/ (JSON and JSONL).
    - Antigravity CLI & IDE transcripts in ~/.gemini/antigravity-*/brain/.
    """

    def __init__(self, gemini_dir: Path | None = None):
        self.gemini_dir = gemini_dir or Path.home() / ".gemini"
        self.tmp_dir = self.gemini_dir / "tmp"
        self.antigravity_cli_dir = self.gemini_dir / "antigravity-cli"
        self.antigravity_ide_dir = self.gemini_dir / "antigravity-ide"

    # --- LearnPlugin identity ---

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini / Antigravity"

    @property
    def description(self) -> str:
        return "Google Gemini CLI & Antigravity (~/.gemini/)"

    def detect(self) -> bool:
        if self.tmp_dir.exists() and (
            any(self.tmp_dir.rglob("session-*.json")) or any(self.tmp_dir.rglob("session-*.jsonl"))
        ):
            return True
        for agy_dir in (self.antigravity_cli_dir, self.antigravity_ide_dir):
            brain_dir = agy_dir / "brain"
            if brain_dir.exists() and any(brain_dir.rglob("transcript.jsonl")):
                return True
        return False

    def create_writer(self) -> ContextWriter:
        return GeminiWriter()

    # --- ConversationScanner interface ---

    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects with Gemini or Antigravity session data."""
        discovered: dict[Path, dict] = {}
        unmapped_projects: list[ProjectInfo] = []

        # 1. Legacy Gemini CLI projects (~/.gemini/tmp/<project>/chats/)
        if self.tmp_dir.exists():
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
                if project_path:
                    entry = discovered.setdefault(
                        project_path,
                        {
                            "name": project_path.name,
                            "project_path": project_path,
                            "data_paths": [],
                        },
                    )
                    if chats_dir not in entry["data_paths"]:
                        entry["data_paths"].append(chats_dir)
                else:
                    unmapped_projects.append(
                        ProjectInfo(
                            name=project_dir.name,
                            project_path=Path.cwd(),
                            data_path=chats_dir,
                            data_paths=[chats_dir],
                            context_file=None,
                            memory_file=None,
                        )
                    )

        # 2. Antigravity CLI and IDE projects (~/.gemini/antigravity-*/brain/)
        for agy_dir in (self.antigravity_cli_dir, self.antigravity_ide_dir):
            brain_dir = agy_dir / "brain"
            if not brain_dir.exists():
                continue

            transcripts = sorted(brain_dir.rglob("transcript.jsonl"))
            if not transcripts:
                continue

            for tf in transcripts:
                detected_path = self._detect_antigravity_project_path(tf)
                if detected_path:
                    entry = discovered.setdefault(
                        detected_path,
                        {
                            "name": detected_path.name,
                            "project_path": detected_path,
                            "data_paths": [],
                        },
                    )
                    if brain_dir not in entry["data_paths"]:
                        entry["data_paths"].append(brain_dir)

        projects: list[ProjectInfo] = []
        for proj_path, info in discovered.items():
            context_file = None
            if path_exists(proj_path):
                candidate_gemini = proj_path / "GEMINI.md"
                candidate_agents = proj_path / "AGENTS.md"
                if path_exists(candidate_gemini):
                    context_file = candidate_gemini
                elif path_exists(candidate_agents):
                    context_file = candidate_agents

            data_paths = info["data_paths"]
            primary_data_path = data_paths[0] if data_paths else proj_path
            projects.append(
                ProjectInfo(
                    name=info["name"],
                    project_path=proj_path,
                    data_path=primary_data_path,
                    context_file=context_file,
                    memory_file=None,
                    data_paths=data_paths,
                )
            )

        projects.extend(unmapped_projects)
        return projects

    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all session files for a project (Gemini CLI and/or Antigravity)."""
        paths = project.data_paths if project.data_paths else [project.data_path]
        scan_items: list[tuple[Path, Callable[[Path], SessionData | None]]] = []
        seen_files: set[Path] = set()

        for p in paths:
            if not p.exists():
                continue

            # 1. Gemini CLI sessions
            if p.is_file() and p.name.startswith("session-") and p.suffix in (".json", ".jsonl"):
                if p not in seen_files:
                    seen_files.add(p)
                    scan_items.append((p, self._scan_session))
            elif p.is_dir() and (
                p.name == "chats" or any(p.glob("session-*.json")) or any(p.glob("session-*.jsonl"))
            ):
                gemini_files = sorted(p.glob("session-*.json")) + sorted(p.glob("session-*.jsonl"))
                for gf in gemini_files:
                    if gf not in seen_files:
                        seen_files.add(gf)
                        scan_items.append((gf, self._scan_session))

            # 2. Antigravity transcripts
            if p.is_file() and p.name == "transcript.jsonl":
                if p not in seen_files:
                    detected = self._detect_antigravity_project_path(p)
                    if detected is None or detected == project.project_path:
                        seen_files.add(p)
                        scan_items.append((p, self._scan_antigravity_transcript))
            elif p.is_dir():
                transcripts = sorted(p.rglob("transcript.jsonl"))
                for tf in transcripts:
                    if tf not in seen_files:
                        detected = self._detect_antigravity_project_path(tf)
                        if detected == project.project_path or (
                            detected is None and len(transcripts) == 1
                        ):
                            seen_files.add(tf)
                            scan_items.append((tf, self._scan_antigravity_transcript))

        if not scan_items:
            return []

        def _run_scanner(
            item: tuple[Path, Callable[[Path], SessionData | None]],
        ) -> SessionData | None:
            fpath, fn = item
            return fn(fpath)

        if max_workers <= 1 or len(scan_items) <= 1:
            return [s for item in scan_items if (s := _run_scanner(item)) and s.tool_calls]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        sessions: list[SessionData] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_scanner, item): item for item in scan_items}
            for future in as_completed(futures):
                session = future.result()
                if session and session.tool_calls:
                    sessions.append(session)
        return sessions

    @staticmethod
    def _detect_antigravity_project_path(transcript_path: Path) -> Path | None:
        """Extract project path from an Antigravity transcript."""
        try:
            with open(transcript_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        step = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(step, dict):
                        continue

                    tool_calls = step.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            args = tc.get("args")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}
                            if isinstance(args, dict):
                                for key in ("Cwd", "SearchDirectory", "DirectoryPath"):
                                    val = args.get(key)
                                    if isinstance(val, str) and val.strip():
                                        p = Path(val.strip("\"'"))
                                        if path_exists(p):
                                            return p

                    content = step.get("content")
                    if isinstance(content, str):
                        m = re.search(r"([A-Za-z]:[\\/][^\r\n\t]+?|/[^\r\n\t]+?)\s*->", content)
                        if m:
                            p = Path(m.group(1).strip().strip("[]'\""))
                            if path_exists(p):
                                return p
                        m_ws = re.search(r"Workspace:\s*([^\r\n]+)", content)
                        if m_ws:
                            p = Path(m_ws.group(1).strip().strip("[]'\""))
                            if path_exists(p):
                                return p
        except (OSError, UnicodeDecodeError):
            pass
        return None

    @staticmethod
    def _extract_antigravity_identity(transcript_path: Path) -> str:
        """Derive a stable session identity from the conversation and source."""
        parts = transcript_path.parts
        source = ""
        conv_id = ""

        if "brain" in parts:
            b_idx = parts.index("brain")
            if b_idx > 0:
                source = parts[b_idx - 1]
            if b_idx + 1 < len(parts):
                conv_id = parts[b_idx + 1]

        if not conv_id:
            cur = transcript_path.parent
            if cur.name == "logs":
                cur = cur.parent
            if cur.name == ".system_generated":
                cur = cur.parent
            conv_id = cur.name if cur.name else transcript_path.stem

        if source and conv_id:
            return f"{source}_{conv_id}"
        return conv_id or transcript_path.stem

    def _scan_antigravity_transcript(self, transcript_path: Path) -> SessionData | None:
        """Parse an Antigravity transcript.jsonl file into SessionData."""
        tool_calls: list[ToolCall] = []
        events: list[SessionEvent] = []
        session_id = self._extract_antigravity_identity(transcript_path)
        session_timestamp: datetime | None = None

        steps: list = []
        try:
            with open(transcript_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        steps.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Failed to read Antigravity transcript %s: %s", transcript_path, e)
            return None

        if not steps:
            return None

        msg_idx = 0
        pending_tool_calls: list[ToolCall] = []

        for step in steps:
            if not isinstance(step, dict):
                continue

            step_type = step.get("type", "")
            source = step.get("source", "")
            content = step.get("content", "")
            created_at = step.get("created_at")

            if session_timestamp is None and isinstance(created_at, str):
                try:
                    session_timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # User input
            if step_type == "USER_INPUT" or source in ("USER_EXPLICIT", "USER_SYSTEM"):
                if isinstance(content, str) and content.strip():
                    m = re.search(r"<USER_REQUEST>\s*([\s\S]*?)\s*</USER_REQUEST>", content)
                    user_text = m.group(1).strip() if m else content.strip()
                    events.append(
                        SessionEvent(
                            type="user_message",
                            msg_index=msg_idx,
                            text=user_text,
                            timestamp=created_at,
                        )
                    )
                    msg_idx += 1

            # Model tool calls
            elif step.get("tool_calls"):
                tcs = step.get("tool_calls")
                if isinstance(tcs, list):
                    for tc_item in tcs:
                        if not isinstance(tc_item, dict):
                            continue
                        raw_name = str(tc_item.get("name", "unknown"))
                        tool_name = normalize_tool_name(raw_name)
                        call_id = str(tc_item.get("id") or f"ag_{session_id}_{len(tool_calls)}")
                        raw_args = tc_item.get("args") or {}
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                raw_args = {"raw": raw_args}
                        elif not isinstance(raw_args, dict):
                            raw_args = {}

                        tc = ToolCall(
                            name=tool_name,
                            tool_call_id=call_id,
                            input_data=raw_args,
                            output="",
                            is_error=False,
                            error_category=ErrorCategory.UNKNOWN,
                            msg_index=msg_idx,
                        )
                        pending_tool_calls.append(tc)
                        tool_calls.append(tc)
                        events.append(
                            SessionEvent(
                                type="tool_call",
                                msg_index=msg_idx,
                                timestamp=created_at,
                                tool_call=tc,
                            )
                        )
                        msg_idx += 1

            # Output / error matching
            elif pending_tool_calls:
                status = step.get("status", "")
                is_err = status == "ERROR"
                tc = pending_tool_calls.pop(0)
                if isinstance(content, str):
                    tc.output = content
                    tc.output_bytes = len(content.encode("utf-8", errors="replace"))
                    if is_error_content(content):
                        is_err = True
                if is_err:
                    tc.is_error = True
                    tc.error_category = classify_error(tc.output or "")

        return SessionData(
            session_id=session_id,
            tool_calls=tool_calls,
            events=events,
            timestamp=session_timestamp,
        )

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
        tool_calls_pending: dict[str, tuple[str, dict, int]] = {}
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
                            tool_calls_pending[name] = (normalized_name, args, msg_index)

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

                        if name in tool_calls_pending:
                            normalized_name, args, call_idx = tool_calls_pending.pop(name)
                        else:
                            normalized_name = normalize_tool_name(name)
                            args = {}
                            call_idx = msg_index

                        call_id = f"{session_id}_{call_idx}_{name}"
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
