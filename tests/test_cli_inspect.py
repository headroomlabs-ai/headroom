"""Tests for the `headroom inspect` command (issue #1267).

The command reads the proxy's loopback ``/transformations/feed`` endpoint and
renders original-vs-compressed content. Tests stub ``probe_json`` so no proxy
is required.

Tests invoke the real top-level CLI (``main``) so the shipped command path —
including subcommand registration — is exercised, not just the command object.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from headroom import paths as paths_mod
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


def test_inspect_uses_the_live_wrap_marker_port(tmp_path: Path, monkeypatch) -> None:
    from headroom.cli import port_discovery

    workspace = tmp_path / "workspace"
    marker = workspace / "clients" / "8788" / "4242.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"pid": 4242, "started_at": 100.0}), encoding="utf-8")
    monkeypatch.setattr(paths_mod, "workspace_dir", lambda: workspace)
    monkeypatch.setattr(port_discovery, "pid_alive", lambda pid: pid == 4242)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)
    monkeypatch.delenv("HEADROOM_PORT_DISCOVERY", raising=False)

    requested_urls: list[str] = []

    def probe(url: str, timeout: float = 1.0):
        requested_urls.append(url)
        if url.endswith("/health"):
            return {"service": "headroom-proxy"}
        return _feed_payload()

    with patch("headroom.install.health.probe_json", side_effect=probe):
        result = _run([])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8788/health" in requested_urls
    assert "http://127.0.0.1:8788/transformations/feed?limit=1" in requested_urls
