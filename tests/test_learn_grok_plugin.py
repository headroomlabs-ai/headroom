from __future__ import annotations

import json
import sys
from pathlib import Path

from headroom.learn.plugins.grok import GrokPlugin


def test_grok_plugin_detects_updates_jsonl(tmp_path: Path) -> None:
    grok_dir = tmp_path / ".grok"
    session_dir = grok_dir / "sessions" / "%2Ftmp%2Fproject" / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "updates.jsonl").write_text("{}\n", encoding="utf-8")

    plugin = GrokPlugin(grok_dir=grok_dir)

    assert plugin.detect() is True


def test_grok_plugin_scans_tool_calls(tmp_path: Path) -> None:
    grok_dir = tmp_path / ".grok"
    session_dir = grok_dir / "sessions" / "%2Ftmp%2Fproject" / "session-1"
    session_dir.mkdir(parents=True)

    lines = [
        {
            "params": {
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call-1",
                    "title": "Shell",
                    "rawInput": {"command": "false"},
                }
            }
        },
        {
            "params": {
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "failed",
                    "rawOutput": {"output_for_prompt": "Exit code: 1"},
                }
            }
        },
    ]
    (session_dir / "updates.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )

    plugin = GrokPlugin(grok_dir=grok_dir)
    projects = plugin.discover_projects()
    assert len(projects) == 1

    sessions = plugin.scan_project(projects[0])
    assert len(sessions) == 1
    assert len(sessions[0].tool_calls) == 1
    assert sessions[0].tool_calls[0].is_error is True


def test_grok_plugin_resolves_absolute_workspace_path(tmp_path: Path) -> None:
    # The workspace dir name is a URL-encoded absolute cwd. It must resolve to
    # that path, not fall back to the process cwd. The Windows branch is the
    # real guard for the fix (a drive-letter path does not start with "/"); the
    # POSIX branch confirms no regression. Detection uses Path.is_absolute().
    grok_dir = tmp_path / ".grok"
    if sys.platform == "win32":
        workspace = "C%3A%5Cproj%5Capp"
        expected = Path(r"C:\proj\app")
    else:
        workspace = "%2Ftmp%2Fproj%2Fapp"
        expected = Path("/tmp/proj/app")
    session_dir = grok_dir / "sessions" / workspace / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "updates.jsonl").write_text("{}\n", encoding="utf-8")

    plugin = GrokPlugin(grok_dir=grok_dir)
    projects = plugin.discover_projects()

    assert len(projects) == 1
    assert projects[0].project_path == expected
