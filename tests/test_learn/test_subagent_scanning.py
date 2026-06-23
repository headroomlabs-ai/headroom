"""The Claude scanner must descend into subagent and workflow transcripts.

Claude Code writes a main session at ``<project>/<uuid>.jsonl`` and nests the
transcripts it spawns under ``<project>/<uuid>/subagents/**`` (subagents) and
``.../subagents/workflows/**`` (workflow agents). Each nested transcript is a
separate context window with its own token spend and its own tool-call
failures, so ``headroom learn`` must see them — not just the top-level session.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.learn.models import ProjectInfo
from headroom.learn.plugins.claude import ClaudeCodePlugin


def _write_session(path: Path, out: str = "x" * 400) -> None:
    """Write a minimal Claude Code session: one tool_use paired with a result."""
    lines = [
        {
            "type": "assistant",
            "message": {
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "Read",
                        "input": {"file_path": "/a.py"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "u1", "content": out}]},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines))


def test_scan_project_discovers_subagent_and_workflow_transcripts(tmp_path: Path) -> None:
    _write_session(tmp_path / "main-uuid.jsonl")
    _write_session(tmp_path / "main-uuid" / "subagents" / "agent-1.jsonl")
    _write_session(tmp_path / "main-uuid" / "subagents" / "workflows" / "wf_1" / "agent-2.jsonl")

    plugin = ClaudeCodePlugin()
    project = ProjectInfo(name="p", project_path=tmp_path, data_path=tmp_path)
    sessions = plugin.scan_project(project, max_workers=1)

    assert len(sessions) == 3
    assert sorted(s.source for s in sessions) == ["main", "subagent", "workflow"]


def test_main_only_restricts_to_top_level(tmp_path: Path) -> None:
    _write_session(tmp_path / "main-uuid.jsonl")
    _write_session(tmp_path / "main-uuid" / "subagents" / "agent-1.jsonl")

    plugin = ClaudeCodePlugin()
    project = ProjectInfo(name="p", project_path=tmp_path, data_path=tmp_path)
    sessions = plugin.scan_project(project, max_workers=1, include_subagents=False)

    assert len(sessions) == 1
    assert sessions[0].source == "main"


def test_subagents_found_in_parallel_scan(tmp_path: Path) -> None:
    # Multiple files force the ThreadPool path; nested transcripts must still appear.
    _write_session(tmp_path / "main-a.jsonl")
    _write_session(tmp_path / "main-b.jsonl")
    _write_session(tmp_path / "main-a" / "subagents" / "agent-1.jsonl")

    plugin = ClaudeCodePlugin()
    project = ProjectInfo(name="p", project_path=tmp_path, data_path=tmp_path)
    sessions = plugin.scan_project(project, max_workers=4)

    assert len(sessions) == 3
    assert sum(1 for s in sessions if s.source == "subagent") == 1
