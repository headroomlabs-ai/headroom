"""Tests for the Claude Code MCP registrar."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.claude import ClaudeRegistrar
from headroom.mcp_registry.install import build_headroom_spec

_RESOLVED_COMMAND = ("/usr/bin/python", "-m", "headroom.cli")
_RESOLVED_ARGS = ("-m", "headroom.cli", "mcp", "serve")


def _make_registrar(
    tmp_path: Path,
    *,
    cli: str | None = "/usr/local/bin/claude",
) -> ClaudeRegistrar:
    """Build a registrar pointed at ``tmp_path`` as $HOME."""
    return ClaudeRegistrar(claude_cli=cli, home_dir=tmp_path)


def _spec() -> ServerSpec:
    return ServerSpec(
        name="headroom",
        command="/usr/bin/python",
        args=("-m", "headroom.cli", "mcp", "serve"),
        env={},
    )


def _install_spec(monkeypatch: pytest.MonkeyPatch) -> ServerSpec:
    monkeypatch.setattr(
        "headroom.mcp_registry.install.resolve_headroom_command",
        lambda: list(_RESOLVED_COMMAND),
    )
    return build_headroom_spec()


# ----------------------------------------------------------------------
# detect()
# ----------------------------------------------------------------------


def test_detect_true_when_cli_present(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    assert reg.detect() is True


def test_detect_true_when_only_claude_dir_exists(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.detect() is True


def test_detect_true_when_only_modern_config_exists(tmp_path: Path) -> None:
    (tmp_path / ".claude.json").write_text("{}")
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


def test_get_server_reads_modern_config(tmp_path: Path) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": _RESOLVED_COMMAND[0],
                        "args": list(_RESOLVED_ARGS),
                        "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"},
                    }
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli=None)
    got = reg.get_server("headroom")
    assert got is not None
    assert got.command == _RESOLVED_COMMAND[0]
    assert got.args == _RESOLVED_ARGS
    assert got.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"}


def test_get_server_falls_back_to_legacy(tmp_path: Path) -> None:
    cfg = tmp_path / ".claude" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": _RESOLVED_COMMAND[0],
                        "args": list(_RESOLVED_ARGS),
                    }
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli=None)
    got = reg.get_server("headroom")
    assert got is not None
    assert got.command == _RESOLVED_COMMAND[0]
    assert got.args == _RESOLVED_ARGS
    assert got.env == {}


def test_get_server_reads_claude_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": _RESOLVED_COMMAND[0],
                        "args": list(_RESOLVED_ARGS),
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    reg = ClaudeRegistrar(claude_cli=None)
    got = reg.get_server("headroom")
    assert got is not None
    assert got.command == _RESOLVED_COMMAND[0]
    assert got.args == _RESOLVED_ARGS


# ----------------------------------------------------------------------
# register_server() — happy paths
# ----------------------------------------------------------------------


def test_register_via_cli_calls_claude_mcp_add(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result) as run_mock:
        result = reg.register_server(_install_spec(monkeypatch))
    assert result.status == RegisterStatus.REGISTERED
    add_call = run_mock.call_args
    assert add_call is not None
    add_cmd = add_call.args[0]
    assert add_cmd[:6] == [
        "/usr/local/bin/claude",
        "mcp",
        "add",
        "headroom",
        "-s",
        "user",
    ]
    assert add_cmd[-(len(_RESOLVED_ARGS) + 2) :] == [
        "--",
        _RESOLVED_COMMAND[0],
        *_RESOLVED_ARGS,
    ]
    assert add_call.kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path)


def test_register_via_cli_includes_env(tmp_path: Path) -> None:
    spec = ServerSpec(
        name="headroom",
        command=_RESOLVED_COMMAND[0],
        args=_RESOLVED_ARGS,
        env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"},
    )
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result) as run_mock:
        reg.register_server(spec)
    add_call = run_mock.call_args
    assert add_call is not None
    add_cmd = add_call.args[0]
    assert "-e" in add_cmd
    e_idx = add_cmd.index("-e")
    assert add_cmd[e_idx + 1] == "HEADROOM_PROXY_URL=http://127.0.0.1:9000"
    assert add_call.kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path)


def test_register_via_cli_without_overrides_keeps_ambient_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "ambient")
    reg = ClaudeRegistrar(claude_cli="/usr/local/bin/claude")
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result) as run_mock:
        reg.register_server(_spec())
    assert run_mock.call_args is not None
    assert run_mock.call_args.kwargs["env"] is None


def test_register_via_cli_prefers_explicit_config_dir_over_ambient_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = tmp_path / "explicit-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "ambient")
    reg = ClaudeRegistrar(claude_cli="/usr/local/bin/claude", config_dir=config_dir)
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result) as run_mock:
        reg.register_server(_spec())
    assert run_mock.call_args is not None
    assert run_mock.call_args.kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(config_dir)


def test_register_writes_file_when_no_cli(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli=None)
    result = reg.register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    cfg = tmp_path / ".claude.json"
    data = json.loads(cfg.read_text())
    assert "headroom" in data["mcpServers"]
    assert data["mcpServers"]["headroom"]["command"] == _RESOLVED_COMMAND[0]
    assert data["mcpServers"]["headroom"]["args"] == list(_RESOLVED_ARGS)


def test_register_writes_to_legacy_when_only_legacy_exists(tmp_path: Path) -> None:
    legacy = tmp_path / ".claude" / "mcp.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({"mcpServers": {}}))
    reg = _make_registrar(tmp_path, cli=None)
    result = reg.register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    data = json.loads(legacy.read_text())
    assert "headroom" in data["mcpServers"]
    # Modern config should NOT have been created.
    assert not (tmp_path / ".claude.json").exists()


def test_register_writes_to_claude_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    reg = ClaudeRegistrar(claude_cli=None)
    result = reg.register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    cfg = tmp_path / ".claude.json"
    data = json.loads(cfg.read_text())
    assert data["mcpServers"]["headroom"]["command"] == _RESOLVED_COMMAND[0]
    assert not (tmp_path / ".claude" / ".claude.json").exists()


# ----------------------------------------------------------------------
# register_server() — already / mismatch / force
# ----------------------------------------------------------------------


def test_register_already_when_spec_matches(tmp_path: Path) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": _RESOLVED_COMMAND[0],
                        "args": list(_RESOLVED_ARGS),
                    }
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    with patch("subprocess.run") as run_mock:
        result = reg.register_server(_spec())
    assert result.status == RegisterStatus.ALREADY
    run_mock.assert_not_called()  # should not touch CLI when already matching


def test_register_mismatch_when_spec_differs_no_force(tmp_path: Path) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": _RESOLVED_COMMAND[0],
                        "args": list(_RESOLVED_ARGS),
                        "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
                    }
                }
            }
        )
    )
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    with patch("subprocess.run") as run_mock:
        result = reg.register_server(_spec())  # default proxy = no env
    assert result.status == RegisterStatus.MISMATCH
    assert "env" in (result.detail or "")
    run_mock.assert_not_called()  # do NOT overwrite without force


def test_register_force_overwrites_mismatch(tmp_path: Path) -> None:
    cfg = tmp_path / ".claude.json"
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
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
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
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    fail = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="claude: error")
    with patch("subprocess.run", return_value=fail):
        result = reg.register_server(_spec())
    # Even though CLI failed, we wrote the config file as a fallback.
    assert result.status == RegisterStatus.REGISTERED
    cfg = tmp_path / ".claude.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert "headroom" in data["mcpServers"]


# ----------------------------------------------------------------------
# unregister
# ----------------------------------------------------------------------


def test_unregister_via_cli(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=ok) as run_mock:
        assert reg.unregister_server("headroom") is True
    assert run_mock.call_args is not None
    cmd = run_mock.call_args.args[0]
    assert cmd[:5] == ["/usr/local/bin/claude", "mcp", "remove", "headroom", "-s"]
    assert cmd[5] == "user"
    assert run_mock.call_args.kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path)


def test_unregister_via_file_when_no_cli(tmp_path: Path) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {"command": _RESOLVED_COMMAND[0], "args": list(_RESOLVED_ARGS)},
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


def test_unregister_via_cli_also_removes_stale_legacy_entry(tmp_path: Path) -> None:
    legacy = tmp_path / ".claude" / "mcp.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({"mcpServers": {"headroom": {"command": "old"}}}))
    reg = _make_registrar(tmp_path, cli="/usr/local/bin/claude")
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=ok):
        assert reg.unregister_server("headroom") is True
    data = json.loads(legacy.read_text())
    assert "headroom" not in data["mcpServers"]


def test_unregister_returns_false_when_absent(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.unregister_server("headroom") is False


def test_unregister_removes_from_claude_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {"command": _RESOLVED_COMMAND[0], "args": list(_RESOLVED_ARGS)},
                    "other": {"command": "other"},
                }
            }
        )
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    reg = ClaudeRegistrar(claude_cli=None)
    assert reg.unregister_server("headroom") is True
    data = json.loads(cfg.read_text())
    assert "headroom" not in data["mcpServers"]
    assert "other" in data["mcpServers"]


# ----------------------------------------------------------------------
# Robustness: bad JSON should not crash
# ----------------------------------------------------------------------


@pytest.mark.parametrize("contents", ["", "not json", "{", "[]"])
def test_get_server_robust_to_bad_json(tmp_path: Path, contents: str) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(contents)
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.get_server("headroom") is None


@pytest.mark.parametrize("mcp_servers", ["null", "[]", '"oops"'])
def test_get_server_robust_to_non_dict_mcp_servers(tmp_path: Path, mcp_servers: str) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(f'{{"mcpServers": {mcp_servers}}}')
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.get_server("headroom") is None


@pytest.mark.parametrize("mcp_servers", ["null", "[]", '"oops"'])
def test_unregister_robust_to_non_dict_mcp_servers(tmp_path: Path, mcp_servers: str) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(f'{{"mcpServers": {mcp_servers}}}')
    reg = _make_registrar(tmp_path, cli=None)
    assert reg.unregister_server("headroom") is False


@pytest.mark.parametrize("mcp_servers", ["null", "[]", '"oops"'])
def test_register_robust_to_non_dict_mcp_servers(tmp_path: Path, mcp_servers: str) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(f'{{"mcpServers": {mcp_servers}}}')
    reg = _make_registrar(tmp_path, cli=None)
    result = reg.register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    data = json.loads(cfg.read_text())
    assert data["mcpServers"]["headroom"]["command"] == _RESOLVED_COMMAND[0]


@pytest.mark.parametrize("contents", ["not json", "{", '{"projects": }', "[]"])
def test_register_via_file_preserves_malformed_config(tmp_path: Path, contents: str) -> None:
    """Registering must NOT clobber an existing but unparseable config.

    ~/.claude.json holds unrelated Claude state (projects, oauthAccount,
    session history). Before the fix a malformed file was read as {} and then
    overwritten with only {"mcpServers": ...}, destroying everything else."""
    cfg = tmp_path / ".claude.json"
    cfg.write_text(contents, encoding="utf-8")
    reg = _make_registrar(tmp_path, cli=None)

    result = reg.register_server(_spec())

    assert result.status == RegisterStatus.FAILED
    assert "not valid JSON" in result.detail
    # The original bytes are untouched — nothing was overwritten.
    assert cfg.read_text(encoding="utf-8") == contents


def test_register_via_file_merges_into_existing_valid_config(tmp_path: Path) -> None:
    """The happy path still merges: unrelated keys are preserved and mcpServers
    gains the headroom entry."""
    cfg = tmp_path / ".claude.json"
    cfg.write_text(
        json.dumps({"projects": {"/x": {"y": 1}}, "oauthAccount": {"id": "abc"}}),
        encoding="utf-8",
    )
    reg = _make_registrar(tmp_path, cli=None)

    result = reg.register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["projects"] == {"/x": {"y": 1}}
    assert data["oauthAccount"] == {"id": "abc"}
    assert "headroom" in data["mcpServers"]
