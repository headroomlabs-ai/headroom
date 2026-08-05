from __future__ import annotations

import json
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
