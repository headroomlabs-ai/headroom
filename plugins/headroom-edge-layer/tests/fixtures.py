"""Synthetic Claude Code session JSONL builder, shared by the test suite.

Mirrors the shape `benchmarks/claude_session_mode_benchmark.py`'s
`load_session_replay()` expects: one JSON object per line, `type` in
{"user", "assistant"}, an `assistant` line carries `requestId` and
`message.usage`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SessionBuilder:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._ts = 0

    def _next_ts(self) -> str:
        self._ts += 1
        return f"2026-01-01T00:{self._ts:02d}:00Z"

    def user_text(self, text: str) -> "SessionBuilder":
        self._lines.append(
            json.dumps({"type": "user", "timestamp": self._next_ts(), "message": {"role": "user", "content": text}})
        )
        return self

    def user_tool_result(self, tool_use_id: str, text: str) -> "SessionBuilder":
        content = [{"type": "tool_result", "tool_use_id": tool_use_id, "content": [{"type": "text", "text": text}]}]
        self._lines.append(
            json.dumps({"type": "user", "timestamp": self._next_ts(), "message": {"role": "user", "content": content}})
        )
        return self

    def assistant(
        self,
        request_id: str,
        content: list[dict[str, Any]],
        *,
        input_tokens: int = 100,
        output_tokens: int = 10,
        cache_read_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> "SessionBuilder":
        usage: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_write_5m_tokens + cache_write_1h_tokens,
        }
        if cache_write_5m_tokens or cache_write_1h_tokens:
            usage["cache_creation"] = {
                "ephemeral_5m_input_tokens": cache_write_5m_tokens,
                "ephemeral_1h_input_tokens": cache_write_1h_tokens,
            }
        self._lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": self._next_ts(),
                    "requestId": request_id,
                    "message": {"role": "assistant", "model": model, "content": content, "usage": usage},
                }
            )
        )
        return self

    def tool_use_text(self, tool_use_id: str, tool_name: str, tool_input: dict[str, Any], text: str = "") -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.append({"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input})
        return blocks

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n")
        return path


def numbered_file_lines(n: int, *, prefix: str = "def line_{i}(): pass") -> str:
    return "\n".join(f"   {i}\t{prefix.format(i=i)}" for i in range(1, n + 1))


def grep_hits(path: str, n: int) -> str:
    return "\n".join(f"{path}:{i}:def line_{i}(): pass  # matches bar" for i in range(1, n + 1))


def build_basic_session(tmp_path: Path, *, project: str = "-test-project", session_id: str = "session1") -> Path:
    """One session: a read, a grep, a repeat read of the same file, a large
    web fetch, and a final no-tool-use assistant turn. Matches the shape
    manually verified against census.py during development.
    """
    builder = SessionBuilder()
    file_lines = numbered_file_lines(50)

    builder.user_text("Fix the bug in foo.py")
    builder.assistant(
        "req1",
        builder.tool_use_text("tu1", "Read", {"file_path": "/repo/foo.py"}, "Let me look at the file"),
        input_tokens=500,
        output_tokens=20,
        cache_write_5m_tokens=100,
    )
    builder.user_tool_result("tu1", file_lines)

    builder.assistant(
        "req2",
        builder.tool_use_text("tu2", "Grep", {"pattern": "bar"}, "Now let me search for bar usages"),
        input_tokens=700,
        output_tokens=15,
        cache_read_tokens=500,
        cache_write_5m_tokens=30,
        cache_write_1h_tokens=20,
    )
    builder.user_tool_result("tu2", grep_hits("foo.py", 60))

    builder.assistant(
        "req3",
        builder.tool_use_text("tu3", "Read", {"file_path": "/repo/foo.py"}, "Let me re-read foo.py to confirm"),
        input_tokens=900,
        output_tokens=10,
        cache_read_tokens=1200,
    )
    builder.user_tool_result("tu3", file_lines)  # identical repeat read

    web_text = "\n".join(f"Paragraph {i} of documentation about widgets and gadgets." for i in range(1, 400))
    builder.assistant(
        "req4",
        builder.tool_use_text("tu4", "WebFetch", {"url": "https://example.com/docs"}, "Let me check the docs online"),
        input_tokens=1000,
        output_tokens=12,
        cache_read_tokens=1800,
    )
    builder.user_tool_result("tu4", web_text)

    builder.assistant(
        "req5",
        [{"type": "text", "text": "Fixed it — the bug was an off-by-one error."}],
        input_tokens=2000,
        output_tokens=40,
        cache_read_tokens=2500,
    )

    return builder.write(tmp_path / project / f"{session_id}.jsonl")
