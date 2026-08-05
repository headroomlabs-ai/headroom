from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.cli import mcp as mcp_cli


def _write_servers(path: Path, servers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def test_find_registration_in_claude_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claude_json = tmp_path / ".claude.json"
    _write_servers(
        claude_json,
        {
            "headroom": {
                "command": "headroom",
                "args": ["mcp", "serve"],
                "env": {"HEADROOM_PROXY_URL": "http://x:1"},
            }
        },
    )
    monkeypatch.setattr(mcp_cli, "CLAUDE_JSON_PATH", claude_json)
    monkeypatch.setattr(mcp_cli, "MCP_CONFIG_PATH", tmp_path / ".claude" / "mcp.json")
    monkeypatch.chdir(tmp_path)

    found = mcp_cli.find_headroom_registration()
    assert found is not None
    path, cfg = found
    assert path == claude_json
    assert cfg["env"]["HEADROOM_PROXY_URL"] == "http://x:1"


def test_find_registration_falls_back_to_mcp_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp_json = tmp_path / "mcp.json"
    _write_servers(mcp_json, {"headroom": {"command": "headroom"}})
    monkeypatch.setattr(mcp_cli, "CLAUDE_JSON_PATH", tmp_path / ".claude.json")  # absent
    monkeypatch.setattr(mcp_cli, "MCP_CONFIG_PATH", mcp_json)
    monkeypatch.chdir(tmp_path)

    found = mcp_cli.find_headroom_registration()
    assert found is not None and found[0] == mcp_json


def test_find_registration_prefers_claude_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_json = tmp_path / ".claude.json"
    mcp_json = tmp_path / "mcp.json"
    _write_servers(claude_json, {"headroom": {"command": "a"}})
    _write_servers(mcp_json, {"headroom": {"command": "b"}})
    monkeypatch.setattr(mcp_cli, "CLAUDE_JSON_PATH", claude_json)
    monkeypatch.setattr(mcp_cli, "MCP_CONFIG_PATH", mcp_json)
    monkeypatch.chdir(tmp_path)

    found = mcp_cli.find_headroom_registration()
    assert found is not None and found[0] == claude_json  # ~/.claude.json takes precedence


def test_find_registration_none_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_cli, "CLAUDE_JSON_PATH", tmp_path / ".claude.json")
    monkeypatch.setattr(mcp_cli, "MCP_CONFIG_PATH", tmp_path / "mcp.json")
    monkeypatch.chdir(tmp_path)

    assert mcp_cli.find_headroom_registration() is None


def test_find_registration_skips_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_json = tmp_path / ".claude.json"
    claude_json.write_text("{ not valid json", encoding="utf-8")
    mcp_json = tmp_path / "mcp.json"
    _write_servers(mcp_json, {"headroom": {"command": "ok"}})
    monkeypatch.setattr(mcp_cli, "CLAUDE_JSON_PATH", claude_json)
    monkeypatch.setattr(mcp_cli, "MCP_CONFIG_PATH", mcp_json)
    monkeypatch.chdir(tmp_path)

    found = mcp_cli.find_headroom_registration()
    assert found is not None and found[0] == mcp_json  # malformed file skipped, next match used
