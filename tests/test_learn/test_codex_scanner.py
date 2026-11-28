"""Codex rollout scanner behavior on shell tool results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.learn.plugins.codex import CodexPlugin


@pytest.mark.parametrize(
    ("exit_code", "output", "failed"),
    [
        (7, "", True),
        (-15, "", True),
        (0, "checks passed", False),
        (0, "Process exited with code 7\n", False),
    ],
)
def test_codex_scan_classifies_exec_exit_status(
    tmp_path: Path, exit_code: int, output: str, failed: bool
) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    rollout = sessions / "rollout.jsonl"
    records = [
        {"type": "session_meta", "payload": {"id": "exec-demo"}},
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "command-1",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "pre-commit run"}),
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "command-1",
                "output": (
                    "Chunk ID: 904f13\n"
                    "Wall time: 0.001 seconds\n"
                    f"Process exited with code {exit_code}\n"
                    f"Original token count: {len(output.split())}\n"
                    f"Output:\n{output}"
                ),
            },
        },
    ]
    rollout.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8", newline="\n")

    plugin = CodexPlugin(codex_dir=tmp_path)
    (project,) = plugin.discover_projects()
    (session,) = plugin.scan_project(project)
    assert session.session_id == "exec-demo"
    assert session.failure_count == int(failed)
