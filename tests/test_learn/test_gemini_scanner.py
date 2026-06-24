"""Unit tests for GeminiScanner — Google Gemini CLI session parsing.

Tests use synthetic session data in tmp directories, no real Gemini data needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.learn.models import ProjectInfo, Recommendation, RecommendationTarget
from headroom.learn.scanner import GeminiScanner
from headroom.learn.writer import GeminiWriter

# =============================================================================
# Helpers
# =============================================================================


def _make_gemini_session(
    messages: list[dict],
    session_id: str = "session-2026-04-09T10-00-abc123",
) -> dict:
    """Wrap messages in a Gemini session JSON structure."""
    return {
        "id": session_id,
        "messages": messages,
    }


def _write_json_session(chats_dir: Path, data: dict, name: str = "session-test.json") -> Path:
    """Write a session JSON file to a chats directory."""
    path = chats_dir / name
    path.write_text(json.dumps(data))
    return path


def _write_jsonl_session(
    chats_dir: Path, records: list[dict], name: str = "session-test.jsonl"
) -> Path:
    """Write a session JSONL file to a chats directory."""
    path = chats_dir / name
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _setup_gemini_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create ~/.gemini/tmp/<project>/chats/ directory structure."""
    gemini_dir = tmp_path / ".gemini"
    project_dir = gemini_dir / "tmp" / "abc123"
    chats_dir = project_dir / "chats"
    chats_dir.mkdir(parents=True)
    return gemini_dir, chats_dir


# =============================================================================
# Project Discovery
# =============================================================================


class TestProjectDiscovery:
    def test_no_gemini_dir(self, tmp_path):
        scanner = GeminiScanner(gemini_dir=tmp_path / ".gemini")
        assert scanner.discover_projects() == []

    def test_empty_tmp_dir(self, tmp_path):
        (tmp_path / ".gemini" / "tmp").mkdir(parents=True)
        scanner = GeminiScanner(gemini_dir=tmp_path / ".gemini")
        assert scanner.discover_projects() == []

    def test_no_session_files(self, tmp_path):
        chats_dir = tmp_path / ".gemini" / "tmp" / "proj1" / "chats"
        chats_dir.mkdir(parents=True)
        scanner = GeminiScanner(gemini_dir=tmp_path / ".gemini")
        assert scanner.discover_projects() == []

    def test_discovers_project_with_sessions(self, tmp_path):
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        session = _make_gemini_session(
            [
                {"role": "user", "parts": [{"text": "hello"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "read_file", "args": {"path": "/tmp/test.py"}}},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "read_file",
                                "response": {"output": "print('hello')"},
                            }
                        },
                    ],
                },
            ]
        )
        _write_json_session(chats_dir, session)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        assert len(projects) == 1
        assert projects[0].data_path == chats_dir


# =============================================================================
# JSON Session Parsing
# =============================================================================


class TestJsonSessionParsing:
    def test_basic_tool_call(self, tmp_path):
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        session = _make_gemini_session(
            [
                {"role": "user", "parts": [{"text": "read the config file"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"path": "/app/config.yaml"},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "read_file",
                                "response": {"output": "port: 8080\nhost: localhost"},
                            }
                        },
                    ],
                },
                {"role": "model", "parts": [{"text": "The config has port 8080."}]},
            ]
        )
        _write_json_session(chats_dir, session)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        assert len(sessions) == 1
        assert len(sessions[0].tool_calls) == 1
        tc = sessions[0].tool_calls[0]
        assert tc.name == "Read"  # Normalized from read_file
        assert tc.output == "port: 8080\nhost: localhost"
        assert not tc.is_error

    def test_multiple_tool_calls(self, tmp_path):
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        session = _make_gemini_session(
            [
                {"role": "user", "parts": [{"text": "find and read test files"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "search_files",
                                "args": {"pattern": "test_*.py"},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "search_files",
                                "response": {"output": "tests/test_main.py\ntests/test_utils.py"},
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"path": "tests/test_main.py"},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "read_file",
                                "response": {"output": "def test_main(): pass"},
                            }
                        },
                    ],
                },
            ]
        )
        _write_json_session(chats_dir, session)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        assert len(sessions[0].tool_calls) == 2
        assert sessions[0].tool_calls[0].name == "Glob"  # search_files → Glob
        assert sessions[0].tool_calls[1].name == "Read"  # read_file → Read

    def test_error_detection(self, tmp_path):
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        session = _make_gemini_session(
            [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "read_file", "args": {"path": "/missing.txt"}}},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "read_file",
                                "response": {
                                    "output": "FileNotFoundError: No such file or directory: '/missing.txt'"
                                },
                            }
                        },
                    ],
                },
            ]
        )
        _write_json_session(chats_dir, session)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        tc = sessions[0].tool_calls[0]
        assert tc.is_error
        assert tc.error_category.value == "file_not_found"

    def test_shell_command_normalized(self, tmp_path):
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        session = _make_gemini_session(
            [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "run_shell_command",
                                "args": {"command": "ls -la"},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "run_shell_command",
                                "response": {
                                    "output": "total 0\ndrwxr-xr-x 2 user user 64 Apr 9 10:00 ."
                                },
                            }
                        },
                    ],
                },
            ]
        )
        _write_json_session(chats_dir, session)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        assert sessions[0].tool_calls[0].name == "Bash"

    def test_user_messages_extracted(self, tmp_path):
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        session = _make_gemini_session(
            [
                {"role": "user", "parts": [{"text": "What files are in this project?"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "search_files", "args": {"pattern": "*"}}},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "search_files",
                                "response": {"output": "main.py"},
                            }
                        },
                    ],
                },
                {"role": "user", "parts": [{"text": "Now run the tests"}]},
            ]
        )
        _write_json_session(chats_dir, session)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        user_events = [e for e in sessions[0].events if e.type == "user_message"]
        assert len(user_events) == 2
        assert "What files" in user_events[0].text
        assert "run the tests" in user_events[1].text

    def test_array_format_messages(self, tmp_path):
        """Sessions stored as bare array of messages (no wrapper object)."""
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        messages = [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "write_file",
                            "args": {"path": "test.py", "content": "pass"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "write_file",
                            "response": {"output": "File written"},
                        }
                    },
                ],
            },
        ]
        _write_json_session(chats_dir, messages)  # Write array directly

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        assert len(sessions) == 1
        assert sessions[0].tool_calls[0].name == "Write"

    def test_no_tool_calls_returns_empty(self, tmp_path):
        """Session with only text (no tool calls) produces no tool_calls."""
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        session = _make_gemini_session(
            [
                {"role": "user", "parts": [{"text": "What is Python?"}]},
                {"role": "model", "parts": [{"text": "Python is a programming language."}]},
            ]
        )
        _write_json_session(chats_dir, session)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        # No tool calls → session filtered out
        assert len(sessions) == 0


# =============================================================================
# JSONL Session Parsing
# =============================================================================


class TestJsonlSessionParsing:
    def test_basic_jsonl_session(self, tmp_path):
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        records = [
            {"type": "session_metadata", "id": "ses-001"},
            {"role": "user", "parts": [{"text": "list files"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "run_shell_command", "args": {"command": "ls"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "run_shell_command",
                            "response": {"output": "main.py\ntest.py"},
                        }
                    },
                ],
            },
        ]
        _write_jsonl_session(chats_dir, records)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        assert len(sessions) == 1
        assert sessions[0].tool_calls[0].name == "Bash"
        assert "main.py" in sessions[0].tool_calls[0].output

    def test_jsonl_type_field_roles(self, tmp_path):
        """JSONL records where role is in the 'type' field (user/gemini)."""
        gemini_dir, chats_dir = _setup_gemini_dir(tmp_path)
        records = [
            {
                "type": "gemini",
                "parts": [
                    {"functionCall": {"name": "read_file", "args": {"path": "README.md"}}},
                ],
            },
            {
                "type": "user",
                "parts": [
                    {"functionResponse": {"name": "read_file", "response": {"output": "# Hello"}}},
                ],
            },
        ]
        _write_jsonl_session(chats_dir, records)

        scanner = GeminiScanner(gemini_dir=gemini_dir)
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        assert len(sessions) == 1
        assert sessions[0].tool_calls[0].name == "Read"


# =============================================================================
# Tool Name Normalization
# =============================================================================


class TestToolNameNormalization:
    def test_all_known_names(self):
        from headroom.learn._shared import normalize_tool_name

        assert normalize_tool_name("run_shell_command") == "Bash"
        assert normalize_tool_name("shell") == "Bash"
        assert normalize_tool_name("execute_command") == "Bash"
        assert normalize_tool_name("read_file") == "Read"
        assert normalize_tool_name("read_many_files") == "Read"
        assert normalize_tool_name("write_file") == "Write"
        assert normalize_tool_name("write_new_file") == "Write"
        assert normalize_tool_name("create_file") == "Write"
        assert normalize_tool_name("edit_file") == "Edit"
        assert normalize_tool_name("replace_in_file") == "Edit"
        assert normalize_tool_name("search_files") == "Glob"
        assert normalize_tool_name("find_files") == "Glob"
        assert normalize_tool_name("grep") == "Grep"
        assert normalize_tool_name("search_text") == "Grep"
        assert normalize_tool_name("list_directory") == "Glob"

    def test_unknown_name_preserved(self):
        from headroom.learn._shared import normalize_tool_name

        assert normalize_tool_name("custom_tool") == "custom_tool"


# =============================================================================
# Writer Integration
# =============================================================================


class TestGeminiWriter:
    def test_writes_to_gemini_md(self, tmp_path):
        proj = ProjectInfo(name="gemini-test", project_path=tmp_path, data_path=tmp_path)
        recs = [
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section="Commands",
                content="- Use `python -m pytest`",
                confidence=0.9,
                evidence_count=5,
            ),
        ]

        writer = GeminiWriter()
        result = writer.write(recs, proj, dry_run=False)

        assert len(result.files_written) == 1
        assert result.files_written[0].name == "GEMINI.md"
        content = (tmp_path / "GEMINI.md").read_text()
        assert "python -m pytest" in content

    def test_empty_recs_no_write(self, tmp_path):
        proj = ProjectInfo(name="clean", project_path=tmp_path, data_path=tmp_path)
        writer = GeminiWriter()
        result = writer.write([], proj, dry_run=False)
        assert result.files_written == []
        assert not (tmp_path / "GEMINI.md").exists()

    def test_dry_run(self, tmp_path):
        proj = ProjectInfo(name="test", project_path=tmp_path, data_path=tmp_path)
        recs = [
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section="Test",
                content="- test",
                confidence=0.8,
                evidence_count=3,
            ),
        ]

        writer = GeminiWriter()
        result = writer.write(recs, proj, dry_run=True)

        assert result.dry_run is True
        assert len(result.files_written) == 1
        # Dry run should NOT create the file
        assert not (tmp_path / "GEMINI.md").exists()
