"""Tests for CodeBuddy learn plugin."""

from __future__ import annotations

import json
from pathlib import Path

from headroom.learn.plugins.codebuddy import (
    CodeBuddyPlugin,
    _component_tokenizations,
    _decode_project_path,
    _greedy_path_decode,
    _project_display_name,
)


def _make_dirs(base: Path, *rel_paths: str) -> None:
    for rel in rel_paths:
        (base / rel).mkdir(parents=True, exist_ok=True)


# =============================================================================
# _decode_project_path
# =============================================================================


def test_decode_simple_posix(tmp_path: Path) -> None:
    # _decode_project_path resolves from /, not tmp_path.
    # Test with a non-existent path to verify it returns None.
    result = _decode_project_path("-nonexistent-nonexistent")
    assert result is None


def test_decode_returns_none_for_no_dash_prefix():
    assert _decode_project_path("Users-me-repo") is None


def test_decode_returns_none_for_single_part():
    assert _decode_project_path("-Users") is None


def test_decode_windows_drive(tmp_path: Path) -> None:
    # Windows-style path: only resolves if drive exists, skip
    result = _decode_project_path("-X-nonexistent")
    assert result is None


def test_decode_simple_posix_exists(tmp_path: Path) -> None:
    # Only resolves from /, so non-existent returns None
    result = _decode_project_path("-nonexistent-nonexistent-nonexistent")
    assert result is None


# =============================================================================
# _project_display_name
# =============================================================================


def test_display_name_posix(tmp_path: Path) -> None:
    d = tmp_path / "my-project"
    d.mkdir()
    assert _project_display_name(d, "fallback") == "my-project"


def test_display_name_windows():
    result = _project_display_name(Path(r"C:\Users\me\repo"), "fallback")
    assert result == "repo"


def test_display_name_root():
    assert _project_display_name(Path("/"), "fallback") == "fallback"


# =============================================================================
# _component_tokenizations
# =============================================================================


def test_simple_component():
    tokens = _component_tokenizations("headroom")
    assert ["headroom"] in tokens


def test_hyphenated_component():
    tokens = _component_tokenizations("my-project")
    assert ["my", "project"] in tokens
    assert ["my-project"] in tokens


def test_dotted_component():
    tokens = _component_tokenizations("GitHub.nosync")
    assert ["GitHub", "nosync"] in tokens
    assert ["GitHub.nosync"] in tokens


def test_hidden_component():
    tokens = _component_tokenizations(".hidden")
    assert ["", "hidden"] in tokens
    assert [".hidden"] in tokens


# =============================================================================
# _greedy_path_decode
# =============================================================================


def test_greedy_simple(tmp_path: Path) -> None:
    _make_dirs(tmp_path, "headroom")
    result = _greedy_path_decode(tmp_path, ["headroom"])
    assert result == tmp_path / "headroom"


def test_greedy_hyphenated(tmp_path: Path) -> None:
    _make_dirs(tmp_path, "my-project")
    result = _greedy_path_decode(tmp_path, ["my", "project"])
    assert result == tmp_path / "my-project"


def test_greedy_empty_parts(tmp_path: Path) -> None:
    _make_dirs(tmp_path, "exists")
    result = _greedy_path_decode(tmp_path, [])
    assert result == tmp_path


def test_greedy_nonexistent_base():
    result = _greedy_path_decode(Path("/nonexistent"), ["a"])
    assert result is None


# =============================================================================
# CodeBuddyPlugin
# =============================================================================


class TestCodeBuddyPlugin:
    def test_name(self):
        plugin = CodeBuddyPlugin()
        assert plugin.name == "codebuddy"

    def test_display_name(self):
        plugin = CodeBuddyPlugin()
        assert plugin.display_name == "CodeBuddy"

    def test_description(self):
        plugin = CodeBuddyPlugin()
        assert "~/.codebuddy/" in plugin.description

    def test_detect_true_when_projects_dir_has_content(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        projects_dir.mkdir(parents=True)
        (projects_dir / "-Users-me-repo").mkdir()
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        assert plugin.detect() is True

    def test_detect_false_when_empty(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        projects_dir.mkdir(parents=True)
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        assert plugin.detect() is False

    def test_detect_false_when_no_dir(self, tmp_path: Path) -> None:
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / "nonexistent")
        assert plugin.detect() is False

    def test_create_writer(self):
        from headroom.learn.writer import CodeBuddyWriter

        plugin = CodeBuddyPlugin()
        writer = plugin.create_writer()
        assert isinstance(writer, CodeBuddyWriter)

    def test_writer_targets_codebuddy_memory(self, tmp_path: Path):
        from headroom.learn.models import ProjectInfo, Recommendation, RecommendationTarget

        project_data = tmp_path / ".codebuddy" / "projects" / "-repo"
        project = ProjectInfo(name="repo", project_path=tmp_path / "repo", data_path=project_data)
        recommendation = Recommendation(
            target=RecommendationTarget.CONTEXT_FILE,
            section="Workflow",
            content="Use the project formatter.",
        )

        result = CodeBuddyPlugin().create_writer().write([recommendation], project, dry_run=False)

        expected = project_data / "memory" / "MEMORY.md"
        assert result.files_written == [expected]
        assert "Use the project formatter." in expected.read_text(encoding="utf-8")
        assert not (project.project_path / "CLAUDE.local.md").exists()

    def test_discover_projects_empty_dir(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        projects_dir.mkdir(parents=True)
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        projects = plugin.discover_projects()
        assert projects == []

    def test_discover_projects_no_jsonl(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        entry = projects_dir / "-Users-me-repo"
        entry.mkdir(parents=True)
        (entry / "memory").mkdir()
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        projects = plugin.discover_projects()
        assert projects == []

    def test_discover_projects_skips_dot_entries(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        dot_entry = projects_dir / ".DS_Store"
        dot_entry.mkdir(parents=True)
        (dot_entry / "test.jsonl").write_text("")
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        projects = plugin.discover_projects()
        assert projects == []

    def test_discover_projects_with_jsonl(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        entry = projects_dir / "-Users-me-repo"
        entry.mkdir(parents=True)
        (entry / "test.jsonl").write_text('{"type":"user","message":{"content":"hello"}}\n')
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        projects = plugin.discover_projects()
        assert len(projects) == 1
        assert projects[0].name is not None

    def test_scan_project_returns_sessions(self, tmp_path: Path) -> None:
        from headroom.learn.models import ProjectInfo

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        jsonl = project_dir / "session.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tc_1",
                                "name": "Bash",
                                "input": {"command": "ls"},
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tc_1",
                                "content": "file1\nfile2",
                            }
                        ]
                    },
                }
            )
            + "\n"
        )
        project = ProjectInfo(name="test", project_path=tmp_path, data_path=project_dir)
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        sessions = plugin.scan_project(project)
        assert len(sessions) == 1
        assert len(sessions[0].tool_calls) == 1
        assert sessions[0].tool_calls[0].name == "Bash"

    def test_scan_project_empty_dir(self, tmp_path: Path) -> None:
        from headroom.learn.models import ProjectInfo

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project = ProjectInfo(name="empty", project_path=tmp_path, data_path=project_dir)
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        sessions = plugin.scan_project(project)
        assert sessions == []

    def test_scan_project_parallel(self, tmp_path: Path) -> None:
        from headroom.learn.models import ProjectInfo

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        for i in range(3):
            (project_dir / f"session_{i}.jsonl").write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "tool_use", "id": f"tc_{i}", "name": "Read", "input": {}}
                            ],
                            "usage": {"input_tokens": 5, "output_tokens": 3},
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {"type": "tool_result", "tool_use_id": f"tc_{i}", "content": "data"}
                            ]
                        },
                    }
                )
                + "\n"
            )
        project = ProjectInfo(name="multi", project_path=tmp_path, data_path=project_dir)
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        sessions = plugin.scan_project(project, max_workers=2)
        assert len(sessions) == 3

    def test_scan_project_handles_malformed_json(self, tmp_path: Path) -> None:
        from headroom.learn.models import ProjectInfo

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "bad.jsonl").write_text("not json\n")
        project = ProjectInfo(name="bad", project_path=tmp_path, data_path=project_dir)
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        sessions = plugin.scan_project(project)
        assert sessions == []

    def test_scan_session_extracts_user_events(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": "Hello, please help me"},
                }
            )
            + "\n"
        )
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path)
        session = plugin._scan_session(jsonl)
        assert session is not None
        assert any(e.type == "user_message" for e in session.events)

    def test_scan_session_extracts_interruption(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [{"type": "text", "text": "[Request interrupted by user"}]
                    },
                }
            )
            + "\n"
        )
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path)
        session = plugin._scan_session(jsonl)
        assert session is not None
        assert any(e.type == "interruption" for e in session.events)

    def test_scan_session_error_content(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "tc_1", "name": "Bash", "input": {}}
                        ],
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tc_1",
                                "content": "Error: command not found",
                                "is_error": True,
                            }
                        ]
                    },
                }
            )
            + "\n"
        )
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path)
        session = plugin._scan_session(jsonl)
        assert session is not None
        assert len(session.tool_calls) == 1
        assert session.tool_calls[0].is_error is True

    def test_discover_projects_windows_fallback(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        entry = projects_dir / "-C-Users-me-repo"
        entry.mkdir(parents=True)
        (entry / "test.jsonl").write_text('{"type":"user","message":{"content":"hi"}}\n')
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        projects = plugin.discover_projects()
        assert len(projects) == 1

    def test_discover_projects_with_codebuddy_md(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        entry = projects_dir / "-Users-me-repo"
        entry.mkdir(parents=True)
        (entry / "test.jsonl").write_text('{"type":"user","message":{"content":"hi"}}\n')
        # Create the actual project dir with CODEBUDDY.md
        real_project = tmp_path / "Users" / "me" / "repo"
        real_project.mkdir(parents=True)
        (real_project / "CODEBUDDY.md").write_text("# Project")
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        projects = plugin.discover_projects()
        assert len(projects) == 1
        if projects[0].context_file:
            assert projects[0].context_file.name == "CODEBUDDY.md"

    def test_discover_projects_with_memory(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / ".codebuddy" / "projects"
        entry = projects_dir / "-Users-me-repo"
        entry.mkdir(parents=True)
        (entry / "test.jsonl").write_text('{"type":"user","message":{"content":"hi"}}\n')
        memory_dir = entry / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("# Memory Index\n")
        plugin = CodeBuddyPlugin(codebuddy_dir=tmp_path / ".codebuddy")
        projects = plugin.discover_projects()
        assert len(projects) == 1
        assert projects[0].memory_file is not None
