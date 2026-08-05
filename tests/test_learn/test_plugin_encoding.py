"""Regression tests for #1202 — the ``learn`` session scanners must read agent
transcripts as UTF-8 with replacement, so a stray non-UTF-8 byte cannot abort
(or silently drop) a scan.

``0x9d`` is undefined in cp1252 *and* an invalid UTF-8 start byte, so a bare
``open()`` fails on it regardless of the host locale. Before the fix this made
the Codex JSONL scanner raise ``UnicodeDecodeError`` (the scan caught only
``OSError``), aborting the whole cross-agent run, while the Claude scanner
caught it and silently dropped the session.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.learn.models import SessionData
from headroom.learn.plugins.claude import ClaudeCodePlugin
from headroom.learn.plugins.codex import CodexPlugin


def _stray_byte_line() -> bytes:
    # A line that is neither valid UTF-8 nor decodable in cp1252.
    return b"\x9d arrow \xe2\x86\x92 junk\n"


def test_claude_scan_recovers_session_with_stray_byte(tmp_path: Path) -> None:
    jsonl = tmp_path / "session.jsonl"
    valid = json.dumps(
        {"type": "assistant", "message": {"usage": {"input_tokens": 5}}, "text": "em — arrow →"}
    )
    jsonl.write_bytes(valid.encode() + b"\n" + _stray_byte_line())

    result = ClaudeCodePlugin(claude_dir=tmp_path)._scan_session(jsonl)

    # Before the fix this returned None (session silently dropped); now the
    # valid line is read and the stray-byte line is skipped, not fatal.
    assert result is not None
    assert result.session_id == "session"
    assert result.total_input_tokens == 5


def test_codex_jsonl_scan_does_not_crash_on_stray_byte(tmp_path: Path) -> None:
    jsonl = tmp_path / "rollout.jsonl"
    meta = json.dumps({"type": "session_meta", "payload": {"id": "abc"}})
    jsonl.write_bytes(meta.encode() + b"\n" + _stray_byte_line())

    # Before the fix this raised UnicodeDecodeError and aborted the run.
    result = CodexPlugin()._scan_jsonl_session(jsonl)

    assert result is None or isinstance(result, SessionData)
