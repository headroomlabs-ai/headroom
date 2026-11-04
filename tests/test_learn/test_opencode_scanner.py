"""Tests for the OpenCode learn scanner."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

import headroom.learn.plugins.opencode as opencode_module
from headroom.learn.models import ErrorCategory
from headroom.learn.plugins.opencode import OpenCodePlugin
from headroom.learn.registry import get_registry, reset_registry
from headroom.learn.writer import CodexWriter


def _create_opencode_db(
    db_path: Path,
    project_path: Path,
    *,
    project_id: str = "project-1",
    project_name: str = "Headroom",
    session_id: str = "session-1",
    tool_command: str = "pytest",
) -> None:
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
            (project_id, project_name, str(project_path)),
        )
        conn.execute(
            "INSERT INTO session (id, project_id, time_created) VALUES (?, ?, ?)",
            (session_id, project_id, 1_700_000_000_000),
        )
        conn.execute(
            "INSERT INTO message (id, session_id) VALUES (?, ?)",
            ("message-1", session_id),
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
                            "input": {"command": tool_command},
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


def _patch_default_paths(monkeypatch: pytest.MonkeyPatch, db_dir: Path) -> tuple[Path, Path]:
    canonical_db = db_dir / "opencode.db"
    local_db = db_dir / "opencode-local.db"
    monkeypatch.setattr(opencode_module, "_OPENCODE_DIR", db_dir, raising=False)
    monkeypatch.setattr(opencode_module, "_OPENCODE_DB", canonical_db, raising=False)
    return canonical_db, local_db


def _set_mtime(path: Path, *, seconds: int) -> None:
    os.utime(path, ns=(seconds * 1_000_000_000, seconds * 1_000_000_000))


def _write_agents_file(project_path: Path) -> None:
    project_path.mkdir()
    (project_path / "AGENTS.md").write_text("# Existing context\n", encoding="utf-8")


def test_opencode_plugin_explicit_path_discovers_projects_and_scans_tool_failures(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "repo"
    _write_agents_file(project_path)
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


def test_opencode_plugin_newer_local_database_wins_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = tmp_path / "repo"
    _write_agents_file(project_path)
    db_dir = tmp_path / "opencode"
    db_dir.mkdir()
    canonical_db, local_db = _patch_default_paths(monkeypatch, db_dir)
    _create_opencode_db(
        canonical_db,
        project_path,
        project_name="Canonical",
        tool_command="canonical-command",
    )
    _create_opencode_db(
        local_db,
        project_path,
        project_name="Local",
        tool_command="local-command",
    )
    _set_mtime(canonical_db, seconds=1)
    _set_mtime(local_db, seconds=2)

    plugin = OpenCodePlugin()

    assert plugin.detect() is True
    projects = plugin.discover_projects()
    assert len(projects) == 1
    assert projects[0].name == "Local"

    sessions = plugin.scan_project(projects[0])
    assert sessions[0].tool_calls[0].input_data == {"command": "local-command"}


def test_opencode_plugin_canonical_only_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = tmp_path / "repo"
    _write_agents_file(project_path)
    db_dir = tmp_path / "opencode"
    db_dir.mkdir()
    canonical_db, _ = _patch_default_paths(monkeypatch, db_dir)
    _create_opencode_db(
        canonical_db,
        project_path,
        project_name="Canonical",
        tool_command="canonical-command",
    )

    plugin = OpenCodePlugin()

    assert plugin.detect() is True
    projects = plugin.discover_projects()
    assert len(projects) == 1
    assert projects[0].name == "Canonical"
    assert plugin._db_path == canonical_db


def test_opencode_plugin_local_only_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = tmp_path / "repo"
    _write_agents_file(project_path)
    db_dir = tmp_path / "opencode"
    db_dir.mkdir()
    _, local_db = _patch_default_paths(monkeypatch, db_dir)
    _create_opencode_db(
        local_db,
        project_path,
        project_name="Local",
        tool_command="local-command",
    )

    plugin = OpenCodePlugin()

    assert plugin.detect() is True
    projects = plugin.discover_projects()
    assert len(projects) == 1
    assert projects[0].name == "Local"
    assert plugin._db_path == local_db


def test_opencode_plugin_equal_mtime_prefers_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = tmp_path / "repo"
    _write_agents_file(project_path)
    db_dir = tmp_path / "opencode"
    db_dir.mkdir()
    canonical_db, local_db = _patch_default_paths(monkeypatch, db_dir)
    _create_opencode_db(
        canonical_db,
        project_path,
        project_name="Canonical",
        tool_command="canonical-command",
    )
    _create_opencode_db(
        local_db,
        project_path,
        project_name="Local",
        tool_command="local-command",
    )
    _set_mtime(canonical_db, seconds=1)
    _set_mtime(local_db, seconds=1)

    plugin = OpenCodePlugin()

    assert plugin._db_path == canonical_db
    projects = plugin.discover_projects()
    assert len(projects) == 1
    assert projects[0].name == "Canonical"


def test_opencode_plugin_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_path = tmp_path / "repo"
    _write_agents_file(project_path)
    db_dir = tmp_path / "opencode"
    db_dir.mkdir()
    canonical_db, local_db = _patch_default_paths(monkeypatch, db_dir)
    override_db = db_dir / "override.db"
    _create_opencode_db(
        canonical_db,
        project_path,
        project_name="Canonical",
        tool_command="canonical-command",
    )
    _create_opencode_db(
        local_db,
        project_path,
        project_name="Local",
        tool_command="local-command",
    )
    _create_opencode_db(
        override_db,
        project_path,
        project_name="Override",
        tool_command="override-command",
    )
    _set_mtime(canonical_db, seconds=1)
    _set_mtime(local_db, seconds=2)
    monkeypatch.setenv(opencode_module._OPENCODE_DB_ENV, str(override_db))

    plugin = OpenCodePlugin()

    assert plugin._db_path == override_db
    projects = plugin.discover_projects()
    assert len(projects) == 1
    assert projects[0].name == "Override"


def test_opencode_plugin_missing_override_stays_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = tmp_path / "repo"
    _write_agents_file(project_path)
    db_dir = tmp_path / "opencode"
    db_dir.mkdir()
    canonical_db, local_db = _patch_default_paths(monkeypatch, db_dir)
    missing_db = db_dir / "override-missing.db"
    _create_opencode_db(canonical_db, project_path, project_name="Canonical")
    _create_opencode_db(local_db, project_path, project_name="Local")
    _set_mtime(canonical_db, seconds=1)
    _set_mtime(local_db, seconds=2)
    monkeypatch.setenv(opencode_module._OPENCODE_DB_ENV, str(missing_db))

    plugin = OpenCodePlugin()

    assert plugin._db_path == missing_db
    assert plugin.detect() is False
    assert plugin.discover_projects() == []


def test_opencode_plugin_no_default_database_detects_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_dir = tmp_path / "opencode"
    db_dir.mkdir()
    canonical_db, _ = _patch_default_paths(monkeypatch, db_dir)

    plugin = OpenCodePlugin()

    assert plugin._db_path == canonical_db
    assert plugin.detect() is False
    assert plugin.discover_projects() == []


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
