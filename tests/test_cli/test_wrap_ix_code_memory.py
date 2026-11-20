"""``--code-memory ix`` registers the Ix code-graph MCP (``ix-memory``).

Ix (https://github.com/ix-infrastructure/Ix) serves its code graph over stdio
with ``ix mcp``. Headroom only registers that server; it never indexes. The
entry must match what Ix's own ``ix mcp install`` writes, and the ledger rules
Serena follows apply unchanged: an entry Headroom installed can be migrated
and removed, a user-managed one is never touched.

Most tests drive the helpers with a real ``ClaudeRegistrar`` writing into
``tmp_path``, so the assertions are about the JSON Claude Code actually reads.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main
from headroom.mcp_registry import IX_MCP_SERVER_NAME, build_ix_spec
from headroom.mcp_registry.base import ServerSpec
from headroom.mcp_registry.claude import SCOPE_LOCAL, SCOPE_USER, ClaudeRegistrar
from headroom.mcp_registry.ledger import headroom_installed_matching, record_install


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.delenv("HEADROOM_CODE_MEMORY", raising=False)
    monkeypatch.delenv("HEADROOM_CODE_MEMORY_SCOPE", raising=False)
    # Pretend both launchers are installed so no test depends on the runner.
    real_which = shutil.which
    fake = {"ix": "/usr/local/bin/ix", "uvx": "/usr/bin/uvx"}
    monkeypatch.setattr(
        wrap_cli.shutil,
        "which",
        lambda name, *a, **k: fake.get(name) or real_which(name, *a, **k),
    )
    # Serena's post-registration steps touch the real home or shell out.
    monkeypatch.setattr(wrap_cli, "_ensure_serena_dashboard_disabled", lambda *a, **k: None)
    monkeypatch.setattr(wrap_cli, "_inject_serena_instructions", lambda *a, **k: True)
    monkeypatch.setattr(wrap_cli, "_index_serena_project", lambda *a, **k: None)


def _registrar(tmp_path: Path, *, scope: str = SCOPE_LOCAL) -> ClaudeRegistrar:
    """Real registrar writing into ``tmp_path``; never touches the real CLI."""
    (tmp_path / ".claude").mkdir(exist_ok=True)
    return ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, scope=scope, project_dir=tmp_path / "proj"
    )


def _config(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / ".claude.json").read_text(encoding="utf-8"))


def _project_servers(tmp_path: Path) -> dict[str, Any]:
    return _config(tmp_path)["projects"][(tmp_path / "proj").as_posix()]["mcpServers"]


def _write_config(tmp_path: Path, config: dict[str, Any]) -> None:
    (tmp_path / ".claude.json").write_text(json.dumps(config), encoding="utf-8")


def _entry(spec: ServerSpec) -> dict[str, Any]:
    return {"command": spec.command, "args": list(spec.args)}


# --- spec and selection -------------------------------------------------------


def test_ix_spec_matches_ix_mcp_install() -> None:
    """Same name and command as ``ix mcp install`` (bare ``ix`` off Windows)."""
    spec = build_ix_spec()
    assert spec.name == IX_MCP_SERVER_NAME == "ix-memory"
    assert (spec.args, spec.env) == (("mcp",), {})
    if os.name != "nt":
        assert spec.command == "ix"


def test_resolver_accepts_ix_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "ix")
    assert wrap_cli._resolve_code_memory({"serena": True}) == wrap_cli._CODE_MEMORY_IX


def test_code_memory_flag_accepts_ix() -> None:
    """``--code-memory ix`` passes click validation and reaches the resolver."""
    result = CliRunner().invoke(wrap_cli.wrap, ["claude", "--code-memory", "ix", "--prepare-only"])

    assert result.exit_code == 0, result.output
    assert wrap_cli._resolve_code_memory({}) == wrap_cli._CODE_MEMORY_IX


def test_ix_selection_replaces_serena_and_sets_up_ix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "ix")
    calls: list[str] = []
    with (
        patch.object(wrap_cli, "_disable_tokensave_mcp", lambda *a, **k: None),
        patch.object(wrap_cli, "_disable_serena_mcp", lambda *a, **k: calls.append("no_serena")),
        patch.object(wrap_cli, "_disable_ix_mcp", lambda *a, **k: calls.append("no_ix")),
        patch.object(wrap_cli, "_setup_serena_mcp", lambda *a, **k: calls.append("serena")),
        patch.object(wrap_cli, "_setup_ix_mcp", lambda *a, **k: calls.append("ix")),
    ):
        wrap_cli._setup_coding_compressor(object(), serena_context="claude-code")
    assert calls == ["no_serena", "ix"]


# --- registration -------------------------------------------------------------


def test_ix_registers_the_ix_memory_spec_for_this_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registrar = _registrar(tmp_path)

    wrap_cli._setup_ix_mcp(registrar)

    assert _project_servers(tmp_path)["ix-memory"] == {"command": "ix", "args": ["mcp"]}
    assert "mcpServers" not in _config(tmp_path)  # stays out of the machine-wide map
    assert headroom_installed_matching(
        "claude", build_ix_spec(), ownership_key=registrar.ownership_key("ix-memory")
    )
    out = capsys.readouterr().out
    assert "Ix MCP: registered" in out
    assert "ix map" in out


def test_ix_skipped_when_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(wrap_cli.shutil, "which", lambda *_a, **_k: None)

    wrap_cli._setup_ix_mcp(_registrar(tmp_path))

    assert not (tmp_path / ".claude.json").exists()
    out = capsys.readouterr().out
    assert "`ix` not found on PATH" in out
    assert "https://github.com/ix-infrastructure/Ix" in out


def test_user_managed_ix_memory_is_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A differing ``ix-memory`` the user registered is reported, never overwritten."""
    mine = {"command": "/opt/ix/bin/ix", "args": ["mcp", "--debug"]}
    _write_config(
        tmp_path,
        {"projects": {(tmp_path / "proj").as_posix(): {"mcpServers": {"ix-memory": mine}}}},
    )

    # force=True (what Codex/Grok pass) must not override a user's entry either.
    wrap_cli._setup_ix_mcp(_registrar(tmp_path), force=True)

    assert _project_servers(tmp_path)["ix-memory"] == mine
    assert "existing config differs" in capsys.readouterr().out


def test_ix_mcp_install_entry_at_user_scope_is_reused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ix mcp install`` writes user scope; no duplicate project entry is added."""
    _write_config(tmp_path, {"mcpServers": {"ix-memory": _entry(build_ix_spec())}})

    wrap_cli._setup_ix_mcp(_registrar(tmp_path))

    config = _config(tmp_path)
    assert config["mcpServers"]["ix-memory"] == {"command": "ix", "args": ["mcp"]}
    assert "projects" not in config
    assert "already registered for every Claude Code session" in capsys.readouterr().out


def test_stale_headroom_installed_ix_memory_is_migrated(tmp_path: Path) -> None:
    registrar = _registrar(tmp_path)
    old = ServerSpec(name="ix-memory", command="ix", args=("mcp", "--old"))
    _write_config(
        tmp_path,
        {"projects": {(tmp_path / "proj").as_posix(): {"mcpServers": {"ix-memory": _entry(old)}}}},
    )
    record_install("claude", old, ownership_key=registrar.ownership_key("ix-memory"))

    wrap_cli._setup_ix_mcp(registrar)

    assert _project_servers(tmp_path)["ix-memory"] == {"command": "ix", "args": ["mcp"]}


# --- switching away and unwrap -------------------------------------------------


def test_switching_ix_to_serena_removes_headroom_installed_ix_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registrar = _registrar(tmp_path)
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "ix")
    wrap_cli._setup_coding_compressor(registrar, serena_context="claude-code")
    assert "ix-memory" in _project_servers(tmp_path)

    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "serena")
    wrap_cli._setup_coding_compressor(registrar, serena_context="claude-code")

    servers = _project_servers(tmp_path)
    assert "ix-memory" not in servers
    assert "serena" in servers


def test_switching_to_none_keeps_user_managed_ix_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user's own ``ix-memory`` (not in the ledger) survives a switch away."""
    _write_config(tmp_path, {"mcpServers": {"ix-memory": _entry(build_ix_spec())}})
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "none")

    wrap_cli._setup_coding_compressor(_registrar(tmp_path), serena_context="claude-code")

    assert _config(tmp_path)["mcpServers"]["ix-memory"] == {"command": "ix", "args": ["mcp"]}


def test_ix_selection_removes_headroom_installed_serena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registrar = _registrar(tmp_path)
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "serena")
    wrap_cli._setup_coding_compressor(registrar, serena_context="claude-code")
    assert "serena" in _project_servers(tmp_path)

    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "ix")
    wrap_cli._setup_coding_compressor(registrar, serena_context="claude-code")

    servers = _project_servers(tmp_path)
    assert "serena" not in servers
    assert servers["ix-memory"] == {"command": "ix", "args": ["mcp"]}


def test_unwrap_claude_removes_headroom_installed_ix_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _port: None)
    registrar = _registrar(tmp_path)
    wrap_cli._setup_ix_mcp(registrar)
    assert "ix-memory" in _project_servers(tmp_path)

    with (
        patch("headroom.mcp_registry.ClaudeRegistrar", return_value=registrar),
        patch("headroom.cli.wrap._remove_claude_managed_hooks", return_value=False),
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap"),
    ):
        result = CliRunner().invoke(main, ["unwrap", "claude"])

    assert result.exit_code == 0, result.output
    assert "Removed Headroom-installed Ix MCP server from Claude." in result.output
    assert "ix-memory" not in _project_servers(tmp_path)


def test_unwrap_leaves_user_managed_ix_memory(tmp_path: Path) -> None:
    _write_config(tmp_path, {"mcpServers": {"ix-memory": _entry(build_ix_spec())}})
    registrar = _registrar(tmp_path, scope=SCOPE_USER)

    status = wrap_cli._remove_headroom_installed_mcp(registrar, "ix-memory")
    assert status == "not_headroom_owned"
    assert "ix-memory" in _config(tmp_path)["mcpServers"]
