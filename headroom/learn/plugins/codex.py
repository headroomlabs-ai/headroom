"""OpenAI Codex CLI plugin for headroom learn.

Reads session logs from ~/.codex/sessions/ (JSON and JSONL formats).
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
    ToolCall,
)
from ..writer import CodexWriter, ContextWriter

logger = logging.getLogger(__name__)


class CodexPlugin(LearnPlugin, ConversationScanner):
    """Reads OpenAI Codex CLI session logs from ~/.codex/sessions/.

    Codex stores sessions as JSON files with:
    - session.id, session.timestamp, session.instructions
    - items[]: array of message/function_call/function_call_output/reasoning objects

    function_call items have: name, call_id, arguments (JSON string)
    function_call_output items have: call_id, output (string or JSON string)
    """

    def __init__(self, codex_dir: Path | None = None):
        self.codex_dir = codex_dir or Path.home() / ".codex"
        self.sessions_dir = self.codex_dir / "sessions"

    # --- LearnPlugin identity ---

    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex CLI"

    @property
    def description(self) -> str:
        return "OpenAI Codex CLI (~/.codex/)"

    def detect(self) -> bool:
        if not self.sessions_dir.exists():
            return False
        return bool(
            any(self.sessions_dir.rglob("*.json")) or any(self.sessions_dir.rglob("*.jsonl"))
        )

    def create_writer(self) -> ContextWriter:
        return CodexWriter()

    # --- ConversationScanner interface ---

    def _iter_session_files(self, root: Path | None = None) -> list[Path]:
        """Return all known Codex session files, including nested rollouts."""
        search_root = root or self.sessions_dir
        session_files = list(search_root.rglob("*.json")) + list(search_root.rglob("*.jsonl"))
        return sorted(path for path in session_files if path.is_file())

    def discover_projects(self) -> list[ProjectInfo]:
        """Codex doesn't organize by project — return a single 'codex' project."""
        if not self.sessions_dir.exists():
            return []

        session_files = self._iter_session_files()
        if not session_files:
            return []

        agents_md = self.codex_dir / "AGENTS.md"
        instructions_md = self.codex_dir / "instructions.md"

        return [
            ProjectInfo(
                name="codex",
                project_path=Path.cwd(),
                data_path=self.sessions_dir,
                context_file=agents_md if agents_md.exists() else None,
                memory_file=instructions_md if instructions_md.exists() else None,
            )
        ]

    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all Codex session JSON files.

        ``include_subagents`` is accepted for a uniform plugin contract but is a
        no-op: Codex stores sessions flat, with no nested transcript hierarchy.
        """
        session_files = self._iter_session_files(project.data_path)
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

    def _scan_session(self, json_path: Path) -> SessionData | None:
        """Parse a single Codex session file."""
        if json_path.suffix == ".jsonl":
            return self._scan_jsonl_session(json_path)
        return self._scan_json_session(json_path)

    def _scan_json_session(self, json_path: Path) -> SessionData | None:
        """Parse a single Codex session file."""
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to read Codex session %s: %s", json_path, e)
            return None

        session_info = data.get("session", {})
        session_id = session_info.get("id", json_path.stem)
        items = data.get("items", [])

        if not items:
            return None

        func_calls: dict[str, tuple[str, dict]] = {}
        tool_calls: list[ToolCall] = []
        msg_index = 0

        for item in items:
            msg_index += 1
            item_type = item.get("type", "")

            if item_type == "function_call":
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                raw_args = item.get("arguments", "")
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    parsed = raw_args
                else:
                    parsed = {"raw": str(raw_args)}

                # Codex-specific: extract command from shell args
                if name == "shell" and "command" in parsed:
                    cmd = parsed["command"]
                    if isinstance(cmd, list):
                        parsed["command"] = cmd[-1] if cmd else ""
                    name = "Bash"
                else:
                    name = normalize_tool_name(name)

                if call_id:
                    func_calls[call_id] = (name, parsed)

            elif item_type == "function_call_output":
                call_id = item.get("call_id", "")
                output_raw = item.get("output", "")

                if call_id not in func_calls:
                    continue

                name, inp = func_calls[call_id]
                result_content = _parse_codex_output(output_raw)

                is_err = is_error_content(result_content)
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

        return SessionData(session_id=session_id, tool_calls=tool_calls)

    def _scan_jsonl_session(self, jsonl_path: Path) -> SessionData | None:
        """Parse a modern Codex rollout session stored as JSONL."""
        session_id = jsonl_path.stem
        func_calls: dict[str, tuple[str, dict]] = {}
        tool_calls: list[ToolCall] = []
        msg_index = 0

        try:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if entry.get("type") == "session_meta":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict):
                            session_id = payload.get("id", session_id)
                        continue

                    if entry.get("type") != "response_item":
                        continue

                    payload = entry.get("payload", {})
                    if not isinstance(payload, dict):
                        continue

                    msg_index += 1
                    item_type = payload.get("type", "")

                    if item_type in ("function_call", "custom_tool_call"):
                        call_id = payload.get("call_id", "")
                        name = payload.get("name", "")
                        parsed = _parse_codex_arguments(payload)
                        name, parsed = _normalize_codex_tool(name, parsed)
                        if call_id and name:
                            func_calls[call_id] = (name, parsed)
                        continue

                    if item_type not in ("function_call_output", "custom_tool_call_output"):
                        continue

                    call_id = payload.get("call_id", "")
                    if call_id not in func_calls:
                        continue

                    name, inp = func_calls[call_id]
                    result_content = _parse_codex_output(payload.get("output", ""))
                    is_err = is_error_content(result_content)
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

        except OSError as e:
            logger.debug("Failed to read Codex session %s: %s", jsonl_path, e)
            return None

        if not tool_calls:
            return None

        return SessionData(session_id=session_id, tool_calls=tool_calls)


# =============================================================================
# Codex-specific Helpers
# =============================================================================


def _parse_codex_arguments(payload: dict) -> dict:
    """Parse arguments for either legacy or rollout Codex tool calls."""
    raw_args = payload.get("arguments", payload.get("input", ""))
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {"raw": raw_args}
        except (json.JSONDecodeError, TypeError):
            return {"raw": raw_args}
    if isinstance(raw_args, dict):
        return raw_args
    return {"raw": str(raw_args)}


def _normalize_codex_tool(name: str, parsed: dict) -> tuple[str, dict]:
    """Normalize modern Codex tool names to the cross-agent schema."""
    if name == "shell" and "command" in parsed:
        cmd = parsed["command"]
        if isinstance(cmd, list):
            parsed["command"] = cmd[-1] if cmd else ""
        return "Bash", parsed

    if name == "exec_command" and "cmd" in parsed:
        parsed = dict(parsed)
        parsed["command"] = parsed.get("cmd", "")
        return "Bash", parsed

    return normalize_tool_name(name), parsed


def _parse_codex_output(output_raw: object) -> str:
    """Parse tool output from Codex rollout records."""
    if isinstance(output_raw, str):
        try:
            parsed_out = json.loads(output_raw)
        except (json.JSONDecodeError, TypeError):
            return output_raw

        if isinstance(parsed_out, dict):
            if "output" in parsed_out:
                return str(parsed_out["output"])
            return json.dumps(parsed_out)
        return output_raw

    return str(output_raw)


# Module-level instance for auto-discovery by the plugin registry
plugin = CodexPlugin()
