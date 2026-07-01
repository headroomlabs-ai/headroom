"""Tests for the `headroom inspect` command (issue #1267).

The command reads the proxy's loopback ``/transformations/feed`` endpoint and
renders original-vs-compressed content. Tests stub ``probe_json`` so no proxy
is required.

Tests invoke the real top-level CLI (``main``) so the shipped command path —
including subcommand registration — is exercised, not just the command object.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli.inspect import _extract_text, _role


def _run(args: list[str]):
    from headroom.cli.main import main

    return CliRunner().invoke(main, ["inspect", *args])


def test_extract_text_handles_str_and_blocks() -> None:
    assert _extract_text("hello") == "hello"
    assert _extract_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    # Nested tool_result content.
    assert _extract_text([{"type": "tool_result", "content": [{"text": "x"}]}]) == "x"
    # Unknown block falls back to JSON, never silently dropped.
    assert "foo" in _extract_text([{"weird": "foo"}])
    assert _extract_text(None) == ""


def test_role_extraction() -> None:
    assert _role({"role": "user"}) == "user"
    assert _role("not a dict") == "?"


def test_no_proxy_errors_cleanly() -> None:
    with patch("headroom.install.health.probe_json", return_value=None):
        result = _run([])
    assert result.exit_code != 0
    assert "No reachable proxy" in result.output


def test_log_messages_disabled_hint() -> None:
    payload = {"transformations": [], "log_full_messages": False}
    with patch("headroom.install.health.probe_json", return_value=payload):
        result = _run([])
    assert result.exit_code != 0
    assert "--log-messages" in result.output


def test_empty_feed_message() -> None:
    payload = {"transformations": [], "log_full_messages": True}
    with patch("headroom.install.health.probe_json", return_value=payload):
        result = _run([])
    assert result.exit_code == 0
    assert "No requests recorded" in result.output


def _feed_payload() -> dict:
    return {
        "log_full_messages": True,
        "transformations": [
            {
                "request_id": "req-1",
                "model": "gpt-4o",
                "input_tokens_original": 100,
                "input_tokens_optimized": 40,
                "tokens_saved": 60,
                "savings_percent": 60.0,
                "transforms_applied": ["SmartCrusher"],
                "request_messages": [
                    {"role": "user", "content": "line one\nline two\nline three"},
                ],
                "compressed_messages": [
                    {"role": "user", "content": "line one\nline three"},
                ],
            }
        ],
    }


def test_text_render_shows_diff_and_header() -> None:
    with patch("headroom.install.health.probe_json", return_value=_feed_payload()):
        result = _run([])
    assert result.exit_code == 0
    out = result.output
    assert "req-1" in out
    assert "gpt-4o" in out
    assert "SmartCrusher" in out
    # The removed line shows up on the original side of the diff.
    assert "line two" in out


def test_json_format_emits_raw_feed() -> None:
    with patch("headroom.install.health.probe_json", return_value=_feed_payload()):
        result = _run(["--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed[0]["request_id"] == "req-1"
