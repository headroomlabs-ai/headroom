"""Tests for wrap.py's workspace-root registration: marker cwd/session_token
fields, the 0700 hardening, and session-token header injection."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from headroom import paths
from headroom.cli import wrap as wrap_cli


def _marker_payload(port: int, pid: int) -> dict:
    path = paths.proxy_clients_dir(port) / f"{pid}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_register_proxy_client_records_real_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    try:
        wrap_cli._register_proxy_client(8787)
        payload = _marker_payload(8787, os.getpid())
        # os.getcwd() resolves symlinks (e.g. macOS /tmp -> /private/tmp);
        # compare against the same resolution, not the raw tmp_path string.
        assert payload["cwd"] == os.getcwd()
        assert "session_token" not in payload  # no token passed -- optional
    finally:
        wrap_cli._unregister_proxy_client(8787)


def test_register_proxy_client_records_session_token_when_given(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    try:
        wrap_cli._register_proxy_client(8788, session_token="tok-xyz")
        payload = _marker_payload(8788, os.getpid())
        assert payload["session_token"] == "tok-xyz"
    finally:
        wrap_cli._unregister_proxy_client(8788)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode bits")
def test_client_marker_directory_is_hardened_to_0700(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    try:
        wrap_cli._register_proxy_client(8789)
        mode = stat.S_IMODE(paths.proxy_clients_dir(8789).stat().st_mode)
        assert mode == 0o700
    finally:
        wrap_cli._unregister_proxy_client(8789)


def test_apply_session_token_header_env_sets_fresh_header():
    env: dict[str, str] = {}
    wrap_cli._apply_session_token_header_env(env, "tok-123")
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Headroom-Session-Token: tok-123"


def test_apply_session_token_header_env_appends_to_existing_headers():
    env = {"ANTHROPIC_CUSTOM_HEADERS": "X-Some-Other: value"}
    wrap_cli._apply_session_token_header_env(env, "tok-123")
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == (
        "X-Some-Other: value\nX-Headroom-Session-Token: tok-123"
    )


def test_apply_session_token_header_env_noop_for_blank_token():
    env: dict[str, str] = {}
    wrap_cli._apply_session_token_header_env(env, "")
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env


def test_claude_launch_uses_one_token_for_marker_and_header():
    """Same token must reach both the marker and the header, or lookups
    never match."""
    import secrets as _secrets

    token = _secrets.token_urlsafe(32)
    env: dict[str, str] = {}
    wrap_cli._apply_session_token_header_env(env, token)
    header_line = env["ANTHROPIC_CUSTOM_HEADERS"]
    assert token in header_line
    assert header_line.startswith(f"{wrap_cli._SESSION_TOKEN_HEADER_NAME}:")
