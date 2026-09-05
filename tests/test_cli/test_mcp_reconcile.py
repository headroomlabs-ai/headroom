from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.mcp_registry import ClaudeRegistrar, build_serena_spec
from headroom.mcp_registry.ledger import headroom_installed_matching

FIXTURE = Path(__file__).parents[1] / "fixtures" / "headroom-issue-3054.json"


def _setup(monkeypatch, tmp_path: Path):
    config = tmp_path / ".claude.json"
    config.write_text(
        json.dumps(
            {
                "oauthAccount": {"email": "user@example.com"},
                "mcpServers": {
                    "serena": {
                        "command": "uvx",
                        "args": json.loads(FIXTURE.read_text())["old_serena_args"],
                    },
                    "other": {"command": "other", "args": []},
                },
                "projects": {"/repo": {"trust": True}},
            }
        )
    )
    registrar = ClaudeRegistrar(claude_cli=None, home_dir=tmp_path)
    monkeypatch.setattr("headroom.mcp_registry.ClaudeRegistrar", lambda: registrar)
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr("headroom.mcp_registry.ledger.ledger_path", lambda: ledger)
    return config, ledger


def test_issue_fixture_reconcile_is_base_fail_head_pass(monkeypatch, tmp_path: Path):
    config, _ = _setup(monkeypatch, tmp_path)
    fixture = json.loads(FIXTURE.read_text())
    recommended = build_serena_spec("claude-code")
    assert list(recommended.args) == fixture["recommended_serena_args"]
    assert CliRunner().invoke(main, ["mcp", "reconcile"]).exit_code == 0
    adopted = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])
    assert adopted.exit_code == 0, adopted.output
    assert json.loads(config.read_text())["mcpServers"]["serena"]["args"] == list(recommended.args)


def test_read_only_preserves_config_and_ledger_bytes_and_mtimes(monkeypatch, tmp_path: Path):
    config, ledger = _setup(monkeypatch, tmp_path)
    ledger.write_text("not json")
    before = (
        config.read_bytes(),
        ledger.read_bytes(),
        os.stat(config).st_mtime_ns,
        os.stat(ledger).st_mtime_ns,
    )
    result = CliRunner().invoke(main, ["mcp", "reconcile"])
    assert result.exit_code == 0, result.output
    after = (
        config.read_bytes(),
        ledger.read_bytes(),
        os.stat(config).st_mtime_ns,
        os.stat(ledger).st_mtime_ns,
    )
    assert after == before
    assert "--adopt" in result.output


def test_adopt_preserves_unrelated_config_and_records_ownership(monkeypatch, tmp_path: Path):
    config, ledger = _setup(monkeypatch, tmp_path)
    result = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])
    assert result.exit_code == 0, result.output
    data = json.loads(config.read_text())
    assert data["oauthAccount"] == {"email": "user@example.com"}
    assert data["projects"] == {"/repo": {"trust": True}}
    assert data["mcpServers"]["other"] == {"command": "other", "args": []}
    assert data["mcpServers"]["serena"]["args"] == list(build_serena_spec("claude-code").args)
    assert json.loads(ledger.read_text())["agents"]["claude"]["serena"]["fingerprint"]


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        "[]",
        '{"agents": null}',
        '{"agents": []}',
        '{"agents": {"claude": null}}',
        '{"agents": {"claude": []}}',
        '{"agents": {"claude": {"serena": null}}}',
    ],
)
def test_malformed_ledger_blocks_adopt_before_config_write(
    monkeypatch, tmp_path: Path, contents: str
):
    config, ledger = _setup(monkeypatch, tmp_path)
    before = config.read_bytes()
    ledger.write_text(contents)
    result = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])
    assert result.exit_code != 0
    assert "ledger" in result.output.lower()
    assert config.read_bytes() == before


def test_corrupt_ledger_is_tolerated_by_read_only(monkeypatch, tmp_path: Path):
    _, ledger = _setup(monkeypatch, tmp_path)
    ledger.write_text('{"agents": []}')
    result = CliRunner().invoke(main, ["mcp", "reconcile"])
    assert result.exit_code == 0, result.output


def test_reconcile_rejects_absent_claude(monkeypatch, tmp_path: Path):
    _, _ = _setup(monkeypatch, tmp_path)
    registrar = ClaudeRegistrar(claude_cli=None, home_dir=tmp_path)
    monkeypatch.setattr(registrar, "detect", lambda: False)
    monkeypatch.setattr("headroom.mcp_registry.ClaudeRegistrar", lambda: registrar)

    result = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])

    assert result.exit_code != 0
    assert "claude is not detected" in result.output


def test_reconcile_adopt_preserves_malformed_config(monkeypatch, tmp_path: Path):
    config, _ = _setup(monkeypatch, tmp_path)
    config.write_text("not json")
    before = config.read_bytes()

    result = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])

    assert result.exit_code != 0
    assert config.read_bytes() == before


def test_adopt_rejects_malformed_modern_before_touching_valid_legacy(monkeypatch, tmp_path: Path):
    modern = tmp_path / ".claude.json"
    legacy = tmp_path / ".claude" / "mcp.json"
    legacy.parent.mkdir()
    modern.write_text("not json")
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "serena": {"command": "uvx", "args": ["--from", "user"]},
                    "other": {"command": "other"},
                }
            }
        )
    )
    registrar = ClaudeRegistrar(claude_cli=None, home_dir=tmp_path)
    monkeypatch.setattr("headroom.mcp_registry.ClaudeRegistrar", lambda: registrar)
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr("headroom.mcp_registry.ledger.ledger_path", lambda: ledger)
    before = (modern.read_bytes(), legacy.read_bytes())

    result = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])

    assert result.exit_code != 0
    assert "not valid JSON" in result.output
    assert (modern.read_bytes(), legacy.read_bytes()) == before


def test_adopt_rejects_non_dict_mcp_servers_in_legacy_root(monkeypatch, tmp_path: Path):
    modern, _ = _setup(monkeypatch, tmp_path)
    legacy = tmp_path / ".claude" / "mcp.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({"mcpServers": []}))
    before = (modern.read_bytes(), legacy.read_bytes())

    result = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])

    assert result.exit_code != 0
    assert "non-object mcpServers" in result.output
    assert (modern.read_bytes(), legacy.read_bytes()) == before


def test_unreadable_ledger_blocks_adopt_without_partial_mutation(monkeypatch, tmp_path: Path):
    config, ledger = _setup(monkeypatch, tmp_path)
    ledger.write_text(json.dumps({"agents": {}}))
    before = (config.read_bytes(), ledger.read_bytes())
    original_read_text = Path.read_text

    def unreadable(path: Path, *args, **kwargs):
        if path == ledger:
            raise PermissionError("test unreadable ledger")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)

    result = CliRunner().invoke(main, ["mcp", "reconcile", "--adopt"])

    assert result.exit_code != 0
    assert "unreadable" in result.output
    assert (config.read_bytes(), ledger.read_bytes()) == before


@pytest.mark.parametrize("state", ["absent", "matching", "user-drift", "headroom-drift"])
@pytest.mark.parametrize("adopt", [False, True])
def test_reconcile_state_matrix(monkeypatch, tmp_path: Path, state: str, adopt: bool):
    config, ledger = _setup(monkeypatch, tmp_path)
    data = json.loads(config.read_text())
    recommended = build_serena_spec("claude-code")
    owned_spec = None
    if state == "absent":
        del data["mcpServers"]["serena"]
    elif state == "matching":
        data["mcpServers"]["serena"] = {
            "command": recommended.command,
            "args": list(recommended.args),
        }
    elif state == "user-drift":
        data["mcpServers"]["serena"]["args"] = ["--from", "user-managed"]
    elif state == "headroom-drift":
        from headroom.mcp_registry.ledger import record_install

        stale = build_serena_spec("claude-code")
        stale.args = ("--from", "headroom-installed-old")
        owned_spec = stale
        data["mcpServers"]["serena"] = {
            "command": stale.command,
            "args": list(stale.args),
        }
        record_install("claude", stale, path=ledger)
    config.write_text(json.dumps(data))
    if owned_spec is not None:
        assert headroom_installed_matching("claude", owned_spec, path=ledger)
    result = CliRunner().invoke(main, ["mcp", "reconcile"] + (["--adopt"] if adopt else []))
    assert result.exit_code == 0, result.output
    observed = json.loads(config.read_text())["mcpServers"].get("serena")
    ownership = observed is not None and headroom_installed_matching(
        "claude",
        build_serena_spec("claude-code") if observed["args"] == list(recommended.args) else None,
        path=ledger,
    )
    if adopt:
        assert observed == {
            "command": recommended.command,
            "args": list(recommended.args),
        }
        assert ownership
        assert "Adopted Headroom" in result.output
    elif state == "headroom-drift":
        assert observed["args"] == ["--from", "headroom-installed-old"]
        assert headroom_installed_matching("claude", owned_spec, path=ledger)
        assert ownership is False
        assert "observed: present" in result.output
    else:
        assert not ownership
        assert "Serena reconciliation for Claude" in result.output


def test_only_adopt_is_a_reconcile_mutation(monkeypatch, tmp_path: Path):
    _setup(monkeypatch, tmp_path)
    result = CliRunner().invoke(main, ["mcp", "reconcile", "--help"])
    assert result.exit_code == 0
    assert "--adopt" in result.output
    for option in ("--acknowledge", "--clear", "--agent", "--server"):
        assert option not in result.output


def test_ordinary_install_does_not_adopt_serena(monkeypatch, tmp_path: Path):
    config, _ = _setup(monkeypatch, tmp_path)
    before = config.read_bytes()
    monkeypatch.setitem(sys.modules, "mcp", object())
    registrar = ClaudeRegistrar(claude_cli=None, home_dir=tmp_path)
    monkeypatch.setattr("headroom.mcp_registry.install.get_all_registrars", lambda: [registrar])
    result = CliRunner().invoke(main, ["mcp", "install", "--agent", "claude"])
    assert result.exit_code == 0, result.output
    after = json.loads(config.read_text())
    before_data = json.loads(before)
    assert after["mcpServers"]["serena"] == before_data["mcpServers"]["serena"]
    assert after["mcpServers"]["headroom"]["args"] == ["mcp", "serve"]
    assert "mcp reconcile --adopt" not in result.output


def test_mcp_install_force_preserves_user_managed_serena(monkeypatch, tmp_path: Path):
    config, _ = _setup(monkeypatch, tmp_path)
    before = json.loads(config.read_text())["mcpServers"]["serena"]
    monkeypatch.setitem(sys.modules, "mcp", object())
    registrar = ClaudeRegistrar(claude_cli=None, home_dir=tmp_path)
    monkeypatch.setattr("headroom.mcp_registry.install.get_all_registrars", lambda: [registrar])

    result = CliRunner().invoke(main, ["mcp", "install", "--agent", "claude", "--force"])

    assert result.exit_code == 0, result.output
    assert json.loads(config.read_text())["mcpServers"]["serena"] == before
