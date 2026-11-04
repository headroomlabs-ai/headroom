from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from headroom.subscription import session_tracking


def test_compute_window_tokens_reads_recent_entries_from_large_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timestamp = "2026-01-01T00:00:00Z"
    recent_entry = {
        "timestamp": timestamp,
        "message": {
            "model": "claude-opus-4-1",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        },
    }
    recent_line = json.dumps(recent_entry).encode() + b"\n"
    max_file_bytes = len(recent_line) + 8
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"x" * max_file_bytes + b"\n" + recent_line)

    monkeypatch.setattr(session_tracking, "_MAX_FILE_BYTES", max_file_bytes)
    monkeypatch.setattr(session_tracking, "find_transcript_files", lambda: [transcript])

    entry_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    tokens = session_tracking.compute_window_tokens(entry_ts - 1, entry_ts + 1)

    assert tokens.input == 11
    assert tokens.output == 7
    assert tokens.weighted_token_equivalent == 36.0
    assert tokens.by_model == {
        "claude-opus-4-1": {
            "input": 11,
            "output": 7,
            "cache_reads": 0,
            "cache_writes_5m": 0,
            "cache_writes_1h": 0,
            "cache_writes_total": 0,
        }
    }


def test_read_transcript_lines_discards_partial_initial_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    complete_line = b'{"marker":"recent"}\n'
    max_file_bytes = len(complete_line) + 5
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"partial-old-entry\n" + complete_line)
    monkeypatch.setattr(session_tracking, "_MAX_FILE_BYTES", max_file_bytes)

    assert session_tracking._read_transcript_lines(transcript) == ['{"marker":"recent"}']


def test_read_transcript_lines_preserves_line_at_tail_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tail = b'{"marker":"first"}\n{"marker":"second"}\n'
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b'{"marker":"old"}\n' + tail)
    monkeypatch.setattr(session_tracking, "_MAX_FILE_BYTES", len(tail))

    assert session_tracking._read_transcript_lines(transcript) == [
        '{"marker":"first"}',
        '{"marker":"second"}',
    ]


def test_read_transcript_lines_preserves_small_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text('{"marker":"first"}\n\n{"marker":"second"}\n')
    monkeypatch.setattr(session_tracking, "_MAX_FILE_BYTES", 1024)

    assert session_tracking._read_transcript_lines(transcript) == [
        '{"marker":"first"}',
        '{"marker":"second"}',
    ]


def test_compute_window_tokens_dedups_usage_by_message_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A single response stored across multiple transcript lines (one per content
    block, same message.usage) must be counted once, not per line (#2340)."""
    timestamp = "2026-01-01T00:00:00Z"
    dup = {
        "timestamp": timestamp,
        "message": {
            "id": "msg_dup",
            "model": "claude-opus-4-1",
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
    }
    other = {
        "timestamp": timestamp,
        "message": {
            "id": "msg_other",
            "model": "claude-opus-4-1",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    }
    noid = {
        "timestamp": timestamp,
        "message": {"model": "claude-opus-4-1", "usage": {"input_tokens": 1, "output_tokens": 1}},
    }
    lines = [dup, dup, dup, other, noid]
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("\n".join(json.dumps(e) for e in lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(session_tracking, "find_transcript_files", lambda: [transcript])

    entry_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    tokens = session_tracking.compute_window_tokens(entry_ts - 1, entry_ts + 1)

    # msg_dup counted once (100), msg_other (5), id-less line (1) -> 106, NOT 306.
    assert tokens.input == 106
    assert tokens.output == 13
    assert tokens.by_model["claude-opus-4-1"]["input"] == 106
