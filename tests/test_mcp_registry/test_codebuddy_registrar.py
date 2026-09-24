"""Tests for the CodeBuddy MCP registrar."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.codebuddy import CodeBuddyRegistrar


def _make_registrar(
    tmp_path: Path,
    *,
    cli: str | None = "/usr/local/bin/codebuddy",
) -> CodeBuddyRegistrar:
    """Build a registrar pointed at ``tmp_path`` as $HOME."""
    return CodeBuddyRegistrar(codebuddy_cli=cli, home_dir=tmp_path)


def _spec() -> ServerSpec:
    return ServerSpec(
        name="headroom",
        command="headroom",
        args=("mcp", "serve"),
        env={},
    )


# ----------------------------------------------------------------------
# detect()
# ----------------------------------------------------------------------


def test_detect_true_when_cli_present(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    assert reg.detect() is True


def test_detect_true_when_only_codebuddy_dir_exists(tmp_path: Path) -> None:
    (tmp_path / ".codebuddy").mkdir()
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.detect() is True


def test_detect_false_when_neither_present(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.detect() is False


# ----------------------------------------------------------------------
# get_server() — file-based reads
# ----------------------------------------------------------------------


def test_get_server_returns_none_when_unregistered(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.get_server("headroom") is None


def test_get_server_reads_config(tmp_path: Path) -> None:
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": "headroom",
                        "args": ["mcp", "serve"],
                        "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"},
                    }
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli=None)
    got = reg.get_server("headroom")
    assert got is not None
    assert got.command == "headroom"
    assert got.args == ("mcp", "serve")
    assert got.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"}


# ----------------------------------------------------------------------
# register_server() — happy paths
# ----------------------------------------------------------------------


def test_register_via_cli_calls_codebuddy_mcp_add(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result) as run_mock:
        result = reg.register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    cmds = [call.args[0] for call in run_mock.call_args_list]
    add_cmd = next(c for c in cmds if "add" in c)
    assert add_cmd[:6] == [
        "/usr/local/bin/codebuddy",
        "mcp",
        "add",
        "headroom",
        "-s",
        "user",
    ]


def test_register_via_cli_includes_env(tmp_path: Path) -> None:
    spec = ServerSpec(
        name="headroom",
        command="headroom",
        args=("mcp", "serve"),
        env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"},
    )
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result) as run_mock:
        reg.register_server(spec)
    add_cmd = next(c for c in [call.args[0] for call in run_mock.call_args_list] if "add" in c)
    assert "-e" in add_cmd
    e_idx = add_cmd.index("-e")
    assert add_cmd[e_idx + 1] == "HEADROOM_PROXY_URL=http://127.0.0.1:9000"


def test_register_writes_file_when_no_cli(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli=None)
    result = reg.register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    data = json.loads(cfg.read_text())
    assert "headroom" in data["mcpServers"]
    assert data["mcpServers"]["headroom"]["command"] == "headroom"
    assert data["mcpServers"]["headroom"]["args"] == ["mcp", "serve"]


# ----------------------------------------------------------------------
# register_server() — already / mismatch / force
# ----------------------------------------------------------------------


def test_register_already_when_spec_matches(tmp_path: Path) -> None:
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps({"mcpServers": {"headroom": {"command": "headroom", "args": ["mcp", "serve"]}}})
    )
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    with patch("subprocess.run") as run_mock:
        result = reg.register_server(_spec())
    assert result.status == RegisterStatus.ALREADY
    run_mock.assert_not_called()


def test_register_mismatch_when_spec_differs_no_force(tmp_path: Path) -> None:
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": "headroom",
                        "args": ["mcp", "serve"],
                        "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
                    }
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    with patch("subprocess.run") as run_mock:
        result = reg.register_server(_spec())
    assert result.status == RegisterStatus.MISMATCH
    assert "env" in (result.detail or "")
    run_mock.assert_not_called()


def test_register_force_overwrites_mismatch(tmp_path: Path) -> None:
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": "headroom-old",
                        "args": ["mcp", "serve"],
                    }
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    fake_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_ok) as run_mock:
        result = reg.register_server(_spec(), force=True)
    assert result.status == RegisterStatus.REGISTERED
    cmds = [call.args[0] for call in run_mock.call_args_list]
    assert any("remove" in c for c in cmds)
    assert any("add" in c for c in cmds)


# ----------------------------------------------------------------------
# CLI failure paths
# ----------------------------------------------------------------------


def test_register_cli_failure_falls_back_to_file(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    fail = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="codebuddy: error")
    with patch("subprocess.run", return_value=fail):
        result = reg.register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert "headroom" in data["mcpServers"]


# ----------------------------------------------------------------------
# unregister
# ----------------------------------------------------------------------


def test_unregister_via_cli(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/codebuddy")
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=ok) as run_mock:
        assert reg.unregister_server("headroom") is True
    cmd = run_mock.call_args_list[0].args[0]
    assert cmd[:5] == ["/usr/local/bin/codebuddy", "mcp", "remove", "headroom", "-s"]
    assert cmd[5] == "user"


def test_unregister_via_file_when_no_cli(tmp_path: Path) -> None:
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {"command": "headroom", "args": ["mcp", "serve"]},
                    "other": {"command": "other"},
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.unregister_server("headroom") is True
    data = json.loads(cfg.read_text())
    assert "headroom" not in data["mcpServers"]
    assert "other" in data["mcpServers"]


def test_unregister_returns_false_when_absent(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.unregister_server("headroom") is False


# ----------------------------------------------------------------------
# Robustness: bad JSON should not crash
# ----------------------------------------------------------------------


@pytest.mark.parametrize("contents", ["", "not json", "{", "[]"])
def test_get_server_robust_to_bad_json(tmp_path: Path, contents: str) -> None:
    cfg = tmp_path / ".codebuddy" / ".mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(contents)
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.get_server("headroom") is None
