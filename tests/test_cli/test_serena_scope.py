"""Serena is registered for the wrapped project, not the whole machine (#2787).

Headroom used to register Serena at Claude Code's ``user`` scope, so every
Claude Code session on the machine launched it — including sessions in
projects that were never wrapped. When Serena could not start, that took down
Claude Code everywhere, and the only recovery was ``headroom unwrap claude``
(#2783).

``headroom wrap claude`` now registers Serena at ``local`` scope: it loads
only in the directory that was wrapped. ``--code-memory-scope user`` restores
the old machine-wide behaviour, and an existing global entry that Headroom
installed is retired on the next wrap.

These tests drive ``_setup_serena_mcp`` with a real ``ClaudeRegistrar`` whose
home directory is a tmp_path, so the assertions are about the JSON Claude Code
actually reads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from headroom.cli import wrap as wrap_cli
from headroom.mcp_registry import build_serena_spec
from headroom.mcp_registry.base import ServerSpec
from headroom.mcp_registry.claude import SCOPE_LOCAL, SCOPE_USER, ClaudeRegistrar
from headroom.mcp_registry.ledger import record_install


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.delenv("HEADROOM_CODE_MEMORY_SCOPE", raising=False)
    # ``_setup_serena_mcp`` bails out early when uvx is missing, which would
    # skip every path under test on runners without uv installed.
    real_which = shutil.which
    monkeypatch.setattr(
        wrap_cli.shutil,
        "which",
        lambda name, *a, **k: "/usr/bin/uvx" if name == "uvx" else real_which(name, *a, **k),
    )
    # Everything past registration touches the real home directory or shells
    # out to uvx. Covered elsewhere; neutralise it so these stay hermetic.
    monkeypatch.setattr(wrap_cli, "_ensure_serena_dashboard_disabled", lambda *a, **k: None)
    monkeypatch.setattr(wrap_cli, "_inject_serena_instructions", lambda *a, **k: True)
    monkeypatch.setattr(wrap_cli, "_index_serena_project", lambda *a, **k: None)


def _registrar(tmp_path: Path, *, scope: str = SCOPE_LOCAL) -> ClaudeRegistrar:
    """Real registrar writing into ``tmp_path`` instead of the user's home."""
    # With no CLI, ``detect`` looks for a ~/.claude directory or config file.
    (tmp_path / ".claude").mkdir(exist_ok=True)
    return ClaudeRegistrar(
        claude_cli=None,  # force the file path; never touch the real CLI
        home_dir=tmp_path,
        scope=scope,
        project_dir=tmp_path / "proj",
    )


def _config(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / ".claude.json").read_text(encoding="utf-8"))


def _project_servers(tmp_path: Path) -> dict[str, Any]:
    project_key = (tmp_path / "proj").as_posix()
    return _config(tmp_path)["projects"][project_key]["mcpServers"]


def _seed_global_serena(tmp_path: Path, spec: ServerSpec) -> None:
    """Write ``spec`` into the machine-wide map, as a pre-#2787 wrap did."""
    entry: dict[str, Any] = {"command": spec.command, "args": list(spec.args)}
    if spec.env:
        entry["env"] = dict(spec.env)
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"serena": entry}}), encoding="utf-8"
    )


def test_wrap_registers_serena_for_this_project_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registrar = _registrar(tmp_path)

    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    assert "serena" in _project_servers(tmp_path)
    # The machine-wide map is what broke unrelated sessions — stay out of it.
    assert "mcpServers" not in _config(tmp_path)
    assert "scoped to this project" in capsys.readouterr().out


def test_rewrap_retires_the_global_entry_headroom_installed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An upgrading user gets the machine-wide entry cleaned up automatically."""
    old_spec = build_serena_spec("claude-code")
    _seed_global_serena(tmp_path, old_spec)
    record_install("claude", old_spec)  # ledger proves Headroom installed it

    wrap_cli._setup_serena_mcp(_registrar(tmp_path), context="claude-code", verbose=True)

    config = _config(tmp_path)
    assert config["mcpServers"] == {}
    assert "serena" in _project_servers(tmp_path)
    assert "removed the machine-wide entry" in capsys.readouterr().out


def test_rewrap_retires_a_stale_global_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The global entry can predate the current spec and still be ours.

    Its fingerprint is the one recorded when it was installed, so the ledger
    check has to run before the new registration overwrites that record.
    """
    stale = ServerSpec(
        name="serena",
        command="uvx",
        args=("--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server"),
    )
    _seed_global_serena(tmp_path, stale)
    record_install("claude", stale)

    wrap_cli._setup_serena_mcp(_registrar(tmp_path), context="claude-code", verbose=True)

    assert _config(tmp_path)["mcpServers"] == {}
    assert "serena" in _project_servers(tmp_path)
    assert "removed the machine-wide entry" in capsys.readouterr().out


def test_rewrap_migration_clears_the_ledger_so_a_reinstall_is_never_deleted(
    tmp_path: Path,
) -> None:
    """The migration must retire its own ownership claim, not just the entry.

    If the ledger record survived, a user who later installs the *same*
    Serena command globally themselves would collide with the old
    fingerprint, and a subsequent wrap would mistake their entry for one
    Headroom still owns and delete it out from under them.
    """
    from headroom.mcp_registry.ledger import headroom_installed_matching

    old_spec = build_serena_spec("claude-code")
    _seed_global_serena(tmp_path, old_spec)
    record_install("claude", old_spec)

    (tmp_path / ".claude").mkdir(exist_ok=True)
    project_a = tmp_path / "project-a"
    project_a.mkdir()
    registrar_a = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, scope=SCOPE_LOCAL, project_dir=project_a
    )
    wrap_cli._setup_serena_mcp(registrar_a, context="claude-code")

    # The migration removed the config entry - the ownership record backing
    # it must be gone too, not just the entry it pointed at.
    assert not headroom_installed_matching("claude", old_spec)

    # The user reinstalls the identical command globally themselves.
    _seed_global_serena(tmp_path, old_spec)

    project_b = tmp_path / "project-b"
    project_b.mkdir()
    registrar_b = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, scope=SCOPE_LOCAL, project_dir=project_b
    )
    wrap_cli._setup_serena_mcp(registrar_b, context="claude-code")

    assert _config(tmp_path)["mcpServers"]["serena"]["command"] == old_spec.command


def test_remove_serena_clears_a_stale_ownership_record_with_no_live_entry(
    tmp_path: Path,
) -> None:
    """An ownership record must not outlive the config entry it authorized."""
    from headroom.mcp_registry.ledger import headroom_installed_matching

    registrar = _registrar(tmp_path)
    ownership_key = registrar.ownership_key("serena", scope=SCOPE_LOCAL)
    stale_spec = ServerSpec(name="serena", command="uvx", args=("start-mcp-server",))
    record_install("claude", stale_spec, ownership_key=ownership_key)
    # Deliberately no live entry: simulates one removed by another path
    # (e.g. the user's own agent config edit) without going through us.

    status = wrap_cli._remove_headroom_installed_serena_mcp(registrar)

    assert status == "not_headroom_owned"
    assert not headroom_installed_matching("claude", stale_spec, ownership_key=ownership_key)


def test_rewrap_leaves_a_user_managed_global_entry_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Serena the user registered globally is theirs, not ours to remove."""
    user_spec = ServerSpec(name="serena", command="/usr/local/bin/custom-serena")
    _seed_global_serena(tmp_path, user_spec)
    # Deliberately no record_install: the ledger has never seen this entry.

    wrap_cli._setup_serena_mcp(_registrar(tmp_path), context="claude-code", verbose=True)

    config = _config(tmp_path)
    assert config["mcpServers"]["serena"]["command"] == "/usr/local/bin/custom-serena"
    # The project entry is still written: local scope wins for this directory,
    # so the user's global Serena keeps working everywhere else.
    assert "serena" in _project_servers(tmp_path)
    assert "removed the machine-wide entry" not in capsys.readouterr().out


def test_user_scope_reproduces_the_previous_behaviour(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registrar = _registrar(tmp_path, scope=SCOPE_USER)

    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    config = _config(tmp_path)
    assert "serena" in config["mcpServers"]
    assert "projects" not in config
    out = capsys.readouterr().out
    assert "every Claude Code session on this machine" in out
    assert "removed the machine-wide entry" not in out


def test_project_ownership_records_do_not_clobber_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrapping and unwrapping one project must not orphan another project."""
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    (tmp_path / ".claude").mkdir()
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    registrar_a = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, scope=SCOPE_LOCAL, project_dir=project_a
    )
    registrar_b = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, scope=SCOPE_LOCAL, project_dir=project_b
    )

    wrap_cli._setup_serena_mcp(registrar_a, context="claude-code")
    wrap_cli._setup_serena_mcp(registrar_b, context="claude-code")

    from headroom.mcp_registry.ledger import headroom_installed_matching

    assert set(_config(tmp_path)["projects"]) == {project_a.as_posix(), project_b.as_posix()}
    assert headroom_installed_matching(
        "claude",
        registrar_a.get_server("serena", scope=SCOPE_LOCAL),
        ownership_key=registrar_a.ownership_key("serena", scope=SCOPE_LOCAL),
    )
    assert wrap_cli._remove_headroom_installed_serena_mcp(registrar_a) == "removed"
    assert registrar_a.get_server("serena", scope=SCOPE_LOCAL) is None
    assert registrar_b.get_server("serena", scope=SCOPE_LOCAL) is not None
    assert wrap_cli._remove_headroom_installed_serena_mcp(registrar_b) == "removed"


def test_project_unwrap_does_not_remove_unowned_global_serena(tmp_path: Path) -> None:
    """Scope-specific cleanup must not turn project ownership into global authority."""
    user_spec = ServerSpec(name="serena", command="custom-serena")
    _seed_global_serena(tmp_path, user_spec)
    registrar = _registrar(tmp_path)
    wrap_cli._setup_serena_mcp(registrar, context="claude-code")

    assert wrap_cli._remove_headroom_installed_serena_mcp(registrar) == "removed"
    assert registrar.get_server("serena", scope=SCOPE_LOCAL) is None
    assert registrar.get_server("serena", scope=SCOPE_USER) == user_spec


def test_scope_migration_is_skipped_for_registrars_without_scopes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Codex/Grok/OpenCode registrars have no scope and no scoped reads."""

    class _FakeRegistrar:
        name = "codex"
        display_name = "Codex"

        def __init__(self) -> None:
            self.server: ServerSpec | None = None

        def detect(self) -> bool:
            return True

        def get_server(self, server_name: str) -> ServerSpec | None:
            return self.server if server_name == "serena" else None

        def register_server(self, spec: ServerSpec, *, force: bool = False) -> Any:
            from headroom.mcp_registry.base import RegisterResult, RegisterStatus

            self.server = spec
            return RegisterResult(RegisterStatus.REGISTERED, "registered")

    registrar = _FakeRegistrar()
    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    assert registrar.server is not None
    out = capsys.readouterr().out
    assert "removed the machine-wide entry" not in out
    assert "scoped to this project" not in out


# ----------------------------------------------------------------------
# Flag / environment resolution
# ----------------------------------------------------------------------


def test_default_scope_is_project(monkeypatch: pytest.MonkeyPatch) -> None:
    assert wrap_cli._resolve_code_memory_scope() == "project"
    assert wrap_cli._claude_code_memory_scope() == SCOPE_LOCAL


def test_env_var_selects_user_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CODE_MEMORY_SCOPE", "user")
    assert wrap_cli._resolve_code_memory_scope() == "user"
    assert wrap_cli._claude_code_memory_scope() == SCOPE_USER


def test_invalid_env_var_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CODE_MEMORY_SCOPE", "global")
    with pytest.raises(Exception) as excinfo:
        wrap_cli._resolve_code_memory_scope()
    assert "HEADROOM_CODE_MEMORY_SCOPE" in str(excinfo.value)
