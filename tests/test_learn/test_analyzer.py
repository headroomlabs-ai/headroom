"""Tests for session analyzer — digest builder and LLM-based analysis."""

import io
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from headroom.learn.analyzer import (
    SessionAnalyzer,
    _build_digest,
    _call_cli_llm,
    _call_llm,
    _detect_default_model,
    _parse_llm_response,
    _resolve_timeout_secs,
    _strip_fenced_json,
)
from headroom.learn.models import (
    AnalysisResult,
    ErrorCategory,
    ProjectInfo,
    RecommendationTarget,
    SessionData,
    SessionEvent,
    ToolCall,
)


def _project() -> ProjectInfo:
    return ProjectInfo(
        name="test-project",
        project_path=Path("/tmp/test-project"),
        data_path=Path("/tmp/test-data"),
    )


def _tc(
    name: str = "Bash",
    input_data: dict | None = None,
    output: str = "ok",
    is_error: bool = False,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    msg_index: int = 0,
    output_bytes: int = 0,
) -> ToolCall:
    return ToolCall(
        name=name,
        tool_call_id=f"tc_{msg_index}",
        input_data=input_data or {},
        output=output,
        is_error=is_error,
        error_category=error_category,
        msg_index=msg_index,
        output_bytes=output_bytes or len(output),
    )


# =============================================================================
# Digest Builder Tests
# =============================================================================


class TestDigestBuilder:
    def test_includes_project_info(self):
        project = _project()
        sessions = [SessionData(session_id="s1", tool_calls=[_tc()])]
        digest = _build_digest(project, sessions)
        assert "test-project" in digest
        assert "/tmp/test-project" in digest

    def test_includes_session_stats(self):
        sessions = [
            SessionData(
                session_id="abc123",
                tool_calls=[_tc(msg_index=0), _tc(msg_index=1, is_error=True, output="Error!")],
            )
        ]
        digest = _build_digest(_project(), sessions)
        assert "abc123" in digest
        assert "2 calls" in digest
        assert "1 failure" in digest

    def test_includes_tool_call_details(self):
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/src/foo.py"},
                        output="contents",
                        msg_index=0,
                    ),
                    _tc(
                        name="Bash",
                        input_data={"command": "python3 run.py"},
                        output="ModuleNotFoundError",
                        is_error=True,
                        error_category=ErrorCategory.MODULE_NOT_FOUND,
                        msg_index=1,
                    ),
                ],
            )
        ]
        digest = _build_digest(_project(), sessions)
        assert "/src/foo.py" in digest
        assert "python3 run.py" in digest
        assert "ERROR" in digest
        assert "ModuleNotFoundError" in digest

    def test_includes_user_messages(self):
        tc = _tc(msg_index=0)
        events = [
            SessionEvent(type="tool_call", msg_index=0, tool_call=tc),
            SessionEvent(type="user_message", msg_index=1, text="Use uv run instead"),
        ]
        sessions = [SessionData(session_id="s1", tool_calls=[tc], events=events)]
        digest = _build_digest(_project(), sessions)
        assert "USER:" in digest
        assert "Use uv run instead" in digest

    def test_includes_subagent_summaries(self):
        events = [
            SessionEvent(
                type="agent_summary",
                msg_index=0,
                agent_tool_count=150,
                agent_tokens=60000,
                agent_prompt="Explore all test files",
            ),
        ]
        sessions = [SessionData(session_id="s1", events=events)]
        digest = _build_digest(_project(), sessions)
        assert "SUBAGENT" in digest
        assert "150 tool calls" in digest
        assert "Explore all test files" in digest

    def test_includes_interruptions(self):
        events = [
            SessionEvent(
                type="interruption",
                msg_index=0,
                text="[Request interrupted by user]",
            ),
        ]
        sessions = [SessionData(session_id="s1", events=events)]
        digest = _build_digest(_project(), sessions)
        assert "INTERRUPTED" in digest

    def test_empty_sessions(self):
        digest = _build_digest(_project(), [])
        assert "0 sessions" in digest or "test-project" in digest


# =============================================================================
# Prior Patterns Injection Tests
# =============================================================================


_MARKER_BLOCK = (
    "<!-- headroom:learn:start -->\n"
    "## Headroom Learned Patterns\n"
    "*Auto-generated by `headroom learn` on 2026-04-01 — do not edit manually*\n"
    "\n"
    "### Large Files\n"
    "- `src/App.tsx` is very large (~40k tokens) — use offset/limit reads\n"
    "- `src/lib.rs` frequently exceeds 10k tokens\n"
    "\n"
    "<!-- headroom:learn:end -->"
)


def _project_with_files(
    tmp_path: Path, claude_md_text: str | None, memory_md_text: str | None
) -> ProjectInfo:
    """Build a ProjectInfo pointing at temp CLAUDE.md / MEMORY.md files."""
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "memory").mkdir(parents=True)

    context_file: Path | None = None
    if claude_md_text is not None:
        context_file = proj_dir / "CLAUDE.md"
        context_file.write_text(claude_md_text)

    memory_file: Path | None = None
    if memory_md_text is not None:
        memory_file = data_dir / "memory" / "MEMORY.md"
        memory_file.write_text(memory_md_text)

    return ProjectInfo(
        name="proj",
        project_path=proj_dir,
        data_path=data_dir,
        context_file=context_file,
        memory_file=memory_file,
    )


class TestPriorPatternsInjection:
    """The digest should include the prior marker block so the LLM can emit
    COMPLETE updated sections instead of condensed deltas that reference
    now-dropped siblings (the "X is also large — same rule as Y, Z" bug)."""

    def test_digest_includes_prior_block_from_claude_md(self, tmp_path):
        project = _project_with_files(
            tmp_path, claude_md_text=f"# Project\n\n{_MARKER_BLOCK}\n", memory_md_text=None
        )
        digest = _build_digest(project, [])
        assert "Prior Learned Patterns" in digest
        assert "### Large Files" in digest
        assert "App.tsx" in digest

    def test_digest_includes_prior_block_from_memory_md(self, tmp_path):
        project = _project_with_files(
            tmp_path, claude_md_text=None, memory_md_text=f"{_MARKER_BLOCK}\n"
        )
        digest = _build_digest(project, [])
        assert "Prior Learned Patterns" in digest
        assert "MEMORY.md" in digest
        assert "### Large Files" in digest

    def test_digest_omits_section_when_no_files_exist(self, tmp_path):
        project = _project_with_files(tmp_path, claude_md_text=None, memory_md_text=None)
        digest = _build_digest(project, [])
        assert "Prior Learned Patterns" not in digest
        assert "<!-- headroom:learn" not in digest

    def test_digest_omits_section_when_file_has_no_marker_block(self, tmp_path):
        """CLAUDE.md exists but has no headroom block → no prior section emitted."""
        project = _project_with_files(
            tmp_path,
            claude_md_text="# Project\n\nJust a regular readme, no headroom block.\n",
            memory_md_text=None,
        )
        digest = _build_digest(project, [])
        assert "Prior Learned Patterns" not in digest

    def test_digest_surfaces_both_files_when_both_present(self, tmp_path):
        project = _project_with_files(
            tmp_path,
            claude_md_text=f"# Project\n\n{_MARKER_BLOCK}\n",
            memory_md_text=f"{_MARKER_BLOCK}\n",
        )
        digest = _build_digest(project, [])
        assert digest.count("### Large Files") >= 2  # once per file
        assert "CLAUDE.md" in digest
        assert "MEMORY.md" in digest

    @patch("headroom.learn.analyzer._call_llm")
    def test_analyze_passes_prior_block_through_to_llm(self, mock_call_llm: MagicMock, tmp_path):
        """End-to-end: SessionAnalyzer.analyze() → _call_llm receives digest
        containing the prior marker block content."""
        mock_call_llm.return_value = {"context_file_rules": [], "memory_file_rules": []}
        project = _project_with_files(
            tmp_path, claude_md_text=f"# Project\n\n{_MARKER_BLOCK}\n", memory_md_text=None
        )
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[_tc(msg_index=0, is_error=True, output="error")],
            )
        ]

        SessionAnalyzer(model="test-model").analyze(project, sessions)

        mock_call_llm.assert_called_once()
        digest_arg = mock_call_llm.call_args[0][0]
        assert "Prior Learned Patterns" in digest_arg
        assert "App.tsx" in digest_arg


# =============================================================================
# LLM Response Parser Tests
# =============================================================================


class TestLLMResponseParser:
    def test_parses_context_file_rules(self):
        raw = {
            "context_file_rules": [
                {
                    "section": "Environment",
                    "content": "- Use `uv run python` instead of `python3`",
                    "estimated_tokens_saved": 800,
                    "evidence_count": 5,
                }
            ],
            "memory_file_rules": [],
        }
        recs = _parse_llm_response(raw)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.CONTEXT_FILE
        assert recs[0].section == "Environment"
        assert "uv run python" in recs[0].content
        assert recs[0].estimated_tokens_saved == 800
        assert recs[0].evidence_count == 5

    def test_parses_memory_file_rules(self):
        raw = {
            "context_file_rules": [],
            "memory_file_rules": [
                {
                    "section": "User Preferences",
                    "content": "- Do not auto-execute curl commands",
                    "estimated_tokens_saved": 500,
                    "evidence_count": 3,
                }
            ],
        }
        recs = _parse_llm_response(raw)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.MEMORY_FILE
        assert "curl" in recs[0].content

    def test_sorts_by_token_savings(self):
        raw = {
            "context_file_rules": [
                {
                    "section": "Paths",
                    "content": "- Use correct paths",
                    "estimated_tokens_saved": 200,
                    "evidence_count": 2,
                },
                {
                    "section": "Environment",
                    "content": "- Use uv",
                    "estimated_tokens_saved": 1000,
                    "evidence_count": 5,
                },
            ],
            "memory_file_rules": [],
        }
        recs = _parse_llm_response(raw)
        assert recs[0].estimated_tokens_saved == 1000
        assert recs[1].estimated_tokens_saved == 200

    def test_handles_missing_fields(self):
        raw = {
            "context_file_rules": [
                {"section": "Env", "content": "- stuff"},
                {"section": "", "content": ""},  # should be skipped
                {"not_a_real_field": True},  # should be skipped
            ],
            "memory_file_rules": [],
        }
        recs = _parse_llm_response(raw)
        assert len(recs) == 1

    def test_handles_empty_response(self):
        recs = _parse_llm_response({})
        assert recs == []

    def test_handles_non_dict_entries(self):
        raw = {"context_file_rules": ["not a dict", 42], "memory_file_rules": []}
        recs = _parse_llm_response(raw)
        assert recs == []


# =============================================================================
# Full Analyzer Integration Tests (mocked LLM)
# =============================================================================


class TestSessionAnalyzer:
    def test_empty_sessions_no_llm_call(self):
        """No failures + no events → no LLM call, empty result."""
        analyzer = SessionAnalyzer()
        result = analyzer.analyze(_project(), [])
        assert result.total_calls == 0
        assert result.total_failures == 0
        assert result.recommendations == []

    @patch("headroom.learn.analyzer._call_llm")
    def test_calls_llm_with_digest(self, mock_call_llm: MagicMock):
        mock_call_llm.return_value = {
            "context_file_rules": [
                {
                    "section": "Environment",
                    "content": "- Use uv run python",
                    "estimated_tokens_saved": 800,
                    "evidence_count": 3,
                }
            ],
            "memory_file_rules": [],
        }

        analyzer = SessionAnalyzer(model="test-model")
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(msg_index=0, is_error=True, output="ModuleNotFoundError"),
                    _tc(msg_index=1),
                ],
            )
        ]
        result = analyzer.analyze(_project(), sessions)

        mock_call_llm.assert_called_once()
        assert result.total_calls == 2
        assert result.total_failures == 1
        assert len(result.recommendations) == 1
        assert "uv run python" in result.recommendations[0].content

    @patch("headroom.learn.analyzer._call_llm")
    def test_handles_llm_failure_gracefully(self, mock_call_llm: MagicMock):
        mock_call_llm.side_effect = RuntimeError("API key not set")

        analyzer = SessionAnalyzer(model="test-model")
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[_tc(msg_index=0, is_error=True, output="error")],
            )
        ]
        result = analyzer.analyze(_project(), sessions)

        # Stats should still work, just no recommendations
        assert result.total_calls == 1
        assert result.total_failures == 1
        assert result.recommendations == []

    @patch("headroom.learn.analyzer._call_llm")
    def test_passes_events_to_digest(self, mock_call_llm: MagicMock):
        """User messages and subagent events should appear in the digest."""
        mock_call_llm.return_value = {"context_file_rules": [], "memory_file_rules": []}

        tc = _tc(msg_index=0, is_error=True, output="error")
        events = [
            SessionEvent(type="tool_call", msg_index=0, tool_call=tc),
            SessionEvent(type="user_message", msg_index=1, text="use venv python"),
        ]
        sessions = [SessionData(session_id="s1", tool_calls=[tc], events=events)]

        analyzer = SessionAnalyzer(model="test-model")
        analyzer.analyze(_project(), sessions)

        # Check that the digest passed to the LLM includes user message
        call_args = mock_call_llm.call_args
        digest = call_args[0][0]  # first positional arg
        assert "use venv python" in digest


# =============================================================================
# Model Auto-Detection
# =============================================================================


class TestDetectDefaultModel:
    def test_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert _detect_default_model() == "claude-sonnet-4-6"

    def test_openai_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert _detect_default_model() == "gpt-4o"

    def test_gemini_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        assert _detect_default_model() == "gemini/gemini-flash-latest"

    def test_anthropic_preferred_over_openai(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert _detect_default_model() == "claude-sonnet-4-6"

    def test_no_keys_no_cli_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr("headroom.learn.analyzer.shutil.which", lambda _name: None)

        with pytest.raises(RuntimeError, match="No LLM API key found"):
            _detect_default_model()

    def test_cli_fallback_claude(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(
            "headroom.learn.analyzer.shutil.which",
            lambda name: f"/usr/bin/{name}" if name == "claude" else None,
        )
        assert _detect_default_model() == "claude-cli"

    def test_cli_fallback_gemini(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(
            "headroom.learn.analyzer.shutil.which",
            lambda name: f"/usr/bin/{name}" if name == "gemini" else None,
        )
        assert _detect_default_model() == "gemini-cli"

    def test_cli_fallback_codex(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(
            "headroom.learn.analyzer.shutil.which",
            lambda name: f"/usr/bin/{name}" if name == "codex" else None,
        )
        assert _detect_default_model() == "codex-cli"

    def test_api_key_preferred_over_cli(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(
            "headroom.learn.analyzer.shutil.which",
            lambda name: f"/usr/bin/{name}" if name == "claude" else None,
        )
        assert _detect_default_model() == "claude-sonnet-4-6"

    def test_env_var_selects_gemini(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HEADROOM_LEARN_CLI", "gemini")
        assert _detect_default_model() == "gemini-cli"

    def test_env_var_selects_codex(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HEADROOM_LEARN_CLI", "codex")
        assert _detect_default_model() == "codex-cli"

    def test_env_var_invalid_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HEADROOM_LEARN_CLI", "unknown-tool")
        with pytest.raises(ValueError, match="not a supported CLI"):
            _detect_default_model()

    def test_api_key_preferred_over_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("HEADROOM_LEARN_CLI", "gemini")
        assert _detect_default_model() == "claude-sonnet-4-6"

    def test_env_var_preferred_over_auto_detect(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HEADROOM_LEARN_CLI", "codex")
        monkeypatch.setattr(
            "headroom.learn.analyzer.shutil.which",
            lambda name: f"/usr/bin/{name}" if name == "claude" else None,
        )
        # codex selected via env var, even though claude is in PATH
        assert _detect_default_model() == "codex-cli"


# =============================================================================
# CLI LLM Backend
# =============================================================================


class TestStripFencedJson:
    def test_raw_json(self):
        result = _strip_fenced_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _strip_fenced_json(raw)
        assert result == {"key": "value"}

    def test_fenced_no_language_tag(self):
        raw = '```\n{"key": "value"}\n```'
        result = _strip_fenced_json(raw)
        assert result == {"key": "value"}

    def test_whitespace_padding(self):
        raw = '  \n```json\n{"key": "value"}\n```\n  '
        result = _strip_fenced_json(raw)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _strip_fenced_json("not json at all")


def _fake_claude_popen(
    *,
    stdout_lines: list[str],
    stderr_lines: list[str] | None = None,
    returncode: int = 0,
    stdout_delay: float = 0.0,
) -> MagicMock:
    """Build a Popen mock factory for the streaming claude-cli path.

    Returns a MagicMock that, when called as ``Popen(cmd, ...)``, yields a
    fake process whose stdout/stderr behave like line-iterable text streams.
    Each stdout line is sleep(*stdout_delay*)-gated to let tests simulate slow
    or hung processes.
    """
    if stderr_lines is None:
        stderr_lines = []

    def _make_iter(lines: list[str], delay: float):
        def _gen():
            for line in lines:
                if delay:
                    time.sleep(delay)
                yield line

        return _gen()

    factory = MagicMock()

    def _construct(*args, **kwargs):
        proc = MagicMock()
        proc.stdin = io.StringIO()
        proc.stdout = _make_iter(stdout_lines, stdout_delay)
        proc.stderr = _make_iter(stderr_lines, 0.0)
        proc.returncode = returncode
        proc.wait = MagicMock(return_value=returncode)
        proc.kill = MagicMock()
        proc.poll = MagicMock(return_value=returncode)
        return proc

    factory.side_effect = _construct
    return factory


def _stream_event(event_type: str, **fields) -> str:
    return json.dumps({"type": event_type, **fields}) + "\n"


def _result_event(text: str) -> str:
    return _stream_event("result", subtype="success", is_error=False, result=text)


class TestCallCliLlm:
    def test_claude_cli_streams_and_parses_result_event(self):
        stdout = [
            _stream_event("system", subtype="init"),
            _stream_event("assistant", message={"content": "thinking..."}),
            _result_event('{"context_file_rules": [], "memory_file_rules": []}'),
        ]
        with patch(
            "headroom.learn.analyzer.subprocess.Popen", _fake_claude_popen(stdout_lines=stdout)
        ) as popen:
            result = _call_cli_llm("test digest", "claude-cli")
        assert result == {"context_file_rules": [], "memory_file_rules": []}
        cmd = popen.call_args[0][0]
        assert cmd == ["claude", "-p", "--output-format", "stream-json", "--verbose"]

    def test_claude_cli_parses_fenced_result(self):
        stdout = [
            _result_event('```json\n{"context_file_rules": [], "memory_file_rules": []}\n```'),
        ]
        with patch(
            "headroom.learn.analyzer.subprocess.Popen", _fake_claude_popen(stdout_lines=stdout)
        ):
            result = _call_cli_llm("test digest", "claude-cli")
        assert result == {"context_file_rules": [], "memory_file_rules": []}

    def test_claude_cli_idle_timeout_kills_hang(self, monkeypatch):
        import threading as _threading

        monkeypatch.setenv("HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS", "1")

        # An iterator that never yields and never EOFs — simulates a hung CLI.
        # The pump thread blocks in __next__, so no events reach the watchdog.
        blocked = _threading.Event()  # never set

        class _HangingStream:
            def __iter__(self):
                return self

            def __next__(self):
                blocked.wait(timeout=10)
                raise StopIteration

        def _construct(*args, **kwargs):
            proc = MagicMock()
            proc.stdin = io.StringIO()
            proc.stdout = _HangingStream()
            proc.stderr = _HangingStream()
            proc.returncode = 0
            proc.wait = MagicMock(return_value=0)
            proc.kill = MagicMock(side_effect=lambda: blocked.set())
            proc.poll = MagicMock(return_value=None)
            return proc

        popen = MagicMock(side_effect=_construct)
        with patch("headroom.learn.analyzer.subprocess.Popen", popen):
            with pytest.raises(RuntimeError, match="produced no output"):
                _call_cli_llm("test digest", "claude-cli")

    def test_claude_cli_hard_cap_kills_continuous_chatter(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "1")
        monkeypatch.setenv("HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS", "10")

        # Continuous output every 50ms so idle never fires; hard cap should.
        chatter = [_stream_event("assistant", message={"i": i}) for i in range(1000)]
        popen = _fake_claude_popen(stdout_lines=chatter, stdout_delay=0.05)
        with patch("headroom.learn.analyzer.subprocess.Popen", popen):
            with pytest.raises(RuntimeError, match="exceeded the 1s hard cap"):
                _call_cli_llm("test digest", "claude-cli")

    def test_claude_cli_missing_result_event_raises(self):
        stdout = [_stream_event("assistant", message={"content": "no result"})]
        with patch(
            "headroom.learn.analyzer.subprocess.Popen", _fake_claude_popen(stdout_lines=stdout)
        ):
            with pytest.raises(RuntimeError, match="did not emit a final `result` event"):
                _call_cli_llm("test digest", "claude-cli")

    def test_claude_cli_nonzero_exit_raises(self):
        popen = _fake_claude_popen(
            stdout_lines=[],
            stderr_lines=["Error: auth required\n"],
            returncode=1,
        )
        with patch("headroom.learn.analyzer.subprocess.Popen", popen):
            with pytest.raises(RuntimeError, match="failed.*exit 1"):
                _call_cli_llm("test digest", "claude-cli")

    def test_claude_cli_unparseable_result_raises_with_context(self):
        stdout = [_result_event("This is not JSON at all")]
        with patch(
            "headroom.learn.analyzer.subprocess.Popen", _fake_claude_popen(stdout_lines=stdout)
        ):
            with pytest.raises(RuntimeError, match="unparseable output"):
                _call_cli_llm("test digest", "claude-cli")

    def test_claude_cli_not_installed_raises(self):
        popen = MagicMock(side_effect=FileNotFoundError("No such file or directory: 'claude'"))
        with patch("headroom.learn.analyzer.subprocess.Popen", popen):
            with pytest.raises(RuntimeError, match="not found in PATH"):
                _call_cli_llm("test digest", "claude-cli")

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_codex_cli_uses_exec(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"context_file_rules": [], "memory_file_rules": []}',
            stderr="",
        )
        result = _call_cli_llm("test digest", "codex-cli")
        assert result == {"context_file_rules": [], "memory_file_rules": []}
        cmd = mock_run.call_args[0][0]
        assert cmd == ["codex", "exec"]

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_gemini_cli_uses_p_flag(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"context_file_rules": [], "memory_file_rules": []}',
            stderr="",
        )
        _call_cli_llm("test digest", "gemini-cli")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["gemini", "-p"]

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_codex_nonzero_exit_raises(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: auth required",
        )
        with pytest.raises(RuntimeError, match="failed.*exit 1"):
            _call_cli_llm("test digest", "codex-cli")

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_codex_stderr_truncated_in_error(self, mock_run: MagicMock):
        long_stderr = "x" * 5000
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr=long_stderr,
        )
        with pytest.raises(RuntimeError) as exc_info:
            _call_cli_llm("test digest", "codex-cli")
        assert long_stderr not in str(exc_info.value)

    def test_unknown_cli_model_raises(self):
        with pytest.raises(ValueError, match="Unknown CLI model"):
            _call_cli_llm("test digest", "unknown-cli")

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_codex_not_installed_raises(self, mock_run: MagicMock):
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'codex'")
        with pytest.raises(RuntimeError, match="not found in PATH"):
            _call_cli_llm("test digest", "codex-cli")

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_codex_timeout_raises_runtime_error(self, mock_run: MagicMock):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["codex", "exec"], timeout=300)
        with pytest.raises(RuntimeError, match="did not respond within"):
            _call_cli_llm("test digest", "codex-cli")

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_codex_timeout_honors_env_override(self, mock_run: MagicMock, monkeypatch):
        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "42")
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        _call_cli_llm("test digest", "codex-cli")
        assert mock_run.call_args.kwargs["timeout"] == 42

    @patch("headroom.learn.analyzer.subprocess.run")
    def test_codex_unparseable_output_raises_with_context(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="This is not JSON at all",
            stderr="",
        )
        with pytest.raises(RuntimeError, match="unparseable output"):
            _call_cli_llm("test digest", "codex-cli")


class TestParseStreamEvent:
    def test_returns_none_for_empty_line(self):
        from headroom.learn.analyzer import _parse_stream_event

        assert _parse_stream_event("") is None
        assert _parse_stream_event("   \n") is None

    def test_returns_none_for_invalid_json(self):
        from headroom.learn.analyzer import _parse_stream_event

        assert _parse_stream_event("not json at all") is None
        assert _parse_stream_event("{unclosed") is None

    def test_returns_none_for_non_dict_json(self):
        from headroom.learn.analyzer import _parse_stream_event

        assert _parse_stream_event('"a string"') is None
        assert _parse_stream_event("[1, 2, 3]") is None

    def test_parses_valid_event(self):
        from headroom.learn.analyzer import _parse_stream_event

        assert _parse_stream_event('{"type": "result", "result": "x"}') == {
            "type": "result",
            "result": "x",
        }


class TestClaudeCliEdgeCases:
    """Coverage for less-traveled branches in the streaming claude-cli path."""

    def test_non_string_result_field_falls_through_to_missing(self):
        # `result` event present but the `result` field is a dict, not a string.
        # The watchdog should not store it as final_result, so the path raises
        # the "did not emit a final result event" error.
        stdout = [_stream_event("result", subtype="success", result={"unexpected": "shape"})]
        with patch(
            "headroom.learn.analyzer.subprocess.Popen", _fake_claude_popen(stdout_lines=stdout)
        ):
            with pytest.raises(RuntimeError, match="did not emit a final `result` event"):
                _call_cli_llm("test digest", "claude-cli")

    def test_stderr_on_success_is_logged_not_raised(self, caplog):
        import logging

        stdout = [_result_event('{"context_file_rules": [], "memory_file_rules": []}')]
        stderr_warning = "deprecation: --foo will be removed in v2\n"
        popen = _fake_claude_popen(stdout_lines=stdout, stderr_lines=[stderr_warning])
        with caplog.at_level(logging.DEBUG, logger="headroom.learn.analyzer"):
            with patch("headroom.learn.analyzer.subprocess.Popen", popen):
                result = _call_cli_llm("test digest", "claude-cli")
        assert result == {"context_file_rules": [], "memory_file_rules": []}
        assert any("CLI stderr (exit 0)" in rec.message for rec in caplog.records)

    def test_non_result_stdout_lines_are_buffered_into_snippet_on_failure(self):
        # If only assistant/system events arrive (no result), the missing-result
        # error should include a snippet from stdout.
        stdout = [
            _stream_event("system", subtype="init"),
            _stream_event("assistant", message={"content": "thinking..."}),
        ]
        with patch(
            "headroom.learn.analyzer.subprocess.Popen", _fake_claude_popen(stdout_lines=stdout)
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _call_cli_llm("test digest", "claude-cli")
        message = str(exc_info.value)
        assert "did not emit a final `result` event" in message
        assert "thinking" in message  # stdout snippet was included

    def test_resolve_timeout_logs_warning_for_invalid(self, caplog, monkeypatch):
        import logging

        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "abc")
        with caplog.at_level(logging.WARNING, logger="headroom.learn.analyzer"):
            assert _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", 300) == 300
        assert any(
            "Invalid HEADROOM_LEARN_CLI_TIMEOUT_SECS" in rec.message for rec in caplog.records
        )

    def test_resolve_timeout_logs_warning_for_non_positive(self, caplog, monkeypatch):
        import logging

        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "-5")
        with caplog.at_level(logging.WARNING, logger="headroom.learn.analyzer"):
            assert _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", 300) == 300
        assert any("must be positive" in rec.message for rec in caplog.records)


class TestResolveTimeoutSecs:
    def test_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", raising=False)
        assert _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", 300) == 300

    def test_uses_default_when_empty(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "")
        assert _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", 300) == 300

    def test_uses_default_for_non_integer(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "not-a-number")
        assert _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", 300) == 300

    def test_uses_default_for_non_positive(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "0")
        assert _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", 300) == 300

    def test_returns_overridden_value(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_LEARN_CLI_TIMEOUT_SECS", "777")
        assert _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", 300) == 777


class TestCallLlmRouting:
    @patch("headroom.learn.analyzer._call_cli_llm")
    def test_routes_cli_model_to_cli_backend(self, mock_cli: MagicMock):
        mock_cli.return_value = {"context_file_rules": [], "memory_file_rules": []}
        result = _call_llm("test digest", "claude-cli")
        mock_cli.assert_called_once_with("test digest", "claude-cli")
        assert result == {"context_file_rules": [], "memory_file_rules": []}

    @patch("headroom.learn.analyzer._call_cli_llm")
    def test_routes_codex_cli(self, mock_cli: MagicMock):
        mock_cli.return_value = {}
        _call_llm("digest", "codex-cli")
        mock_cli.assert_called_once_with("digest", "codex-cli")


# =============================================================================
# Legacy Compatibility
# =============================================================================


class TestFailureAnalyzerCompat:
    @patch("headroom.learn.analyzer._call_llm")
    def test_legacy_alias_works(self, mock_call_llm: MagicMock):
        from headroom.learn.analyzer import FailureAnalyzer

        mock_call_llm.return_value = {"context_file_rules": [], "memory_file_rules": []}

        analyzer = FailureAnalyzer()
        result = analyzer.analyze(_project(), [])
        assert isinstance(result, AnalysisResult)
