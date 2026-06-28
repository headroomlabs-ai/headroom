"""Tests for the OpenCode learn scanner."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from headroom.learn.models import ErrorCategory
from headroom.learn.plugins.opencode import OpenCodePlugin
from headroom.learn.registry import get_registry, reset_registry
from headroom.learn.writer import CodexWriter


def _create_opencode_db(db_path: Path, project_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE project (
                id TEXT PRIMARY KEY,
                name TEXT,
                worktree TEXT
            );
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                time_created INTEGER
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT,
                data TEXT,
                time_created INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO project (id, name, worktree) VALUES (?, ?, ?)",
            ("project-1", "Headroom", str(project_path)),
        )
        conn.execute(
            "INSERT INTO session (id, project_id, time_created) VALUES (?, ?, ?)",
            ("session-1", "project-1", 1_700_000_000_000),
        )
        conn.execute(
            "INSERT INTO message (id, session_id) VALUES (?, ?)",
            ("message-1", "session-1"),
        )
        conn.execute(
            "INSERT INTO part (id, message_id, data, time_created) VALUES (?, ?, ?, ?)",
            (
                "part-1",
                "message-1",
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "call-1",
                        "state": {
                            "status": "error",
                            "input": {"command": "pytest"},
                            "output": "Error: command failed with exit code 1",
                        },
                    }
                ),
                1_700_000_000_001,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_opencode_plugin_discovers_projects_and_scans_tool_failures(tmp_path: Path) -> None:
    project_path = tmp_path / "repo"
    project_path.mkdir()
    (project_path / "AGENTS.md").write_text("# Existing context\n", encoding="utf-8")
    db_path = tmp_path / "opencode.db"
    _create_opencode_db(db_path, project_path)

    plugin = OpenCodePlugin(db_path=db_path)

    projects = plugin.discover_projects()
    assert len(projects) == 1
    assert projects[0].name == "Headroom"
    assert projects[0].project_path == project_path
    assert projects[0].context_file == project_path / "AGENTS.md"

    sessions = plugin.scan_project(projects[0])
    assert len(sessions) == 1
    assert sessions[0].session_id == "session-1"
    assert sessions[0].timestamp is not None

    tool_call = sessions[0].tool_calls[0]
    assert tool_call.name == "Bash"
    assert tool_call.tool_call_id == "call-1"
    assert tool_call.input_data == {"command": "pytest"}
    assert tool_call.is_error is True
    assert tool_call.error_category == ErrorCategory.RUNTIME_ERROR


def test_opencode_plugin_uses_agents_writer(tmp_path: Path) -> None:
    plugin = OpenCodePlugin(db_path=tmp_path / "missing.db")

    assert plugin.detect() is False
    assert isinstance(plugin.create_writer(), CodexWriter)


def test_opencode_plugin_is_discovered_by_registry() -> None:
    reset_registry()
    try:
        assert "opencode" in get_registry()
    finally:
        reset_registry()
