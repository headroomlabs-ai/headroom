"""learn plugins must not crash when a session-derived path can't be stat'd.

Session files record project paths written on other machines / by other users.
``Path.exists()`` calls ``os.stat``, which raises ``PermissionError`` (not
``False``) when a parent directory isn't stat-able, and unhandled that aborts
the whole ``learn`` command (issue #2443). The claude plugin already guards
this; the gemini and grok plugins share the same shape and must too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.learn.plugins._paths import path_exists


class _BoomPath(type(Path())):  # type: ignore[misc]
    """A Path whose ``exists()`` raises PermissionError, like a restricted stat."""

    def exists(self) -> bool:  # noqa: D401
        raise PermissionError(13, "Permission denied", str(self))


def test_path_exists_swallows_permission_error() -> None:
    assert path_exists(_BoomPath("/home/other/project")) is False


def test_path_exists_true_for_real_path(tmp_path: Path) -> None:
    assert path_exists(tmp_path) is True


def test_gemini_detect_project_path_survives_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.learn.plugins.gemini import GeminiPlugin

    session = tmp_path / "session-1.json"
    session.write_text(json.dumps({"projectPath": "/restricted/project"}), encoding="utf-8")

    # os.stat on the recorded path raises PermissionError for a restricted parent.
    def _boom(self: Path) -> bool:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "exists", _boom)

    plugin = GeminiPlugin(gemini_dir=tmp_path)
    # Must treat the unreadable path as absent, not raise.
    assert plugin._detect_project_path(session) is None


def test_grok_discover_projects_survives_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.learn.plugins.grok import GrokPlugin

    workspace = tmp_path / "sessions" / "%2Fhome%2Fu%2Fproj"
    (workspace / "s1").mkdir(parents=True)
    (workspace / "s1" / "updates.jsonl").write_text("{}\n", encoding="utf-8")

    real_exists = Path.exists

    def _boom_for_context_files(self: Path) -> bool:
        # Only the GROK.md / AGENTS.md probes under the decoded project path blow
        # up; the real session dirs still resolve so discovery reaches them.
        if self.name in {"GROK.md", "AGENTS.md"}:
            raise PermissionError(13, "Permission denied", str(self))
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", _boom_for_context_files)

    plugin = GrokPlugin(grok_dir=tmp_path)
    projects = plugin.discover_projects()
    # Discovery completes; the unreadable context file is simply treated as absent.
    assert len(projects) == 1
    assert projects[0].context_file is None


def test_opencode_discover_projects_survives_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from headroom.learn.plugins.opencode import OpenCodePlugin

    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE project (id TEXT PRIMARY KEY, name TEXT, worktree TEXT);
            CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, time_created INTEGER);
            """
        )
        conn.execute(
            "INSERT INTO project (id, name, worktree) VALUES (?, ?, ?)",
            ("p1", "Proj", "/restricted/worktree"),
        )
        conn.execute(
            "INSERT INTO session (id, project_id, time_created) VALUES (?, ?, ?)",
            ("s1", "p1", 1_700_000_000_000),
        )
        conn.commit()
    finally:
        conn.close()

    # The recorded worktree is behind a restricted parent: the AGENTS.md probe
    # stats it. Only that probe blows up so the DB open still works.
    real_exists = Path.exists

    def _boom(self: Path) -> bool:
        if self.name == "AGENTS.md":
            raise PermissionError(13, "Permission denied", str(self))
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", _boom)

    plugin = OpenCodePlugin(db_path=db_path)
    projects = plugin.discover_projects()
    assert len(projects) == 1
    assert projects[0].context_file is None
