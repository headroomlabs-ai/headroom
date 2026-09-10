from __future__ import annotations

from pathlib import Path

from headroom.cli import wrap as wrap_cli
from headroom.mcp_registry import build_serena_spec
from headroom.mcp_registry.base import RegisterResult, RegisterStatus, ServerSpec
from headroom.mcp_registry.ledger import headroom_installed_matching, record_install


class _Registrar:
    display_name = "Claude Code"

    def __init__(self, current: ServerSpec | None, *, name: str = "claude"):
        self.name = name
        self.current = current
        self.force_calls: list[bool] = []

    def detect(self) -> bool:
        return True

    def get_server(self, name: str) -> ServerSpec | None:
        return self.current if name == "serena" else None

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        self.force_calls.append(force)
        if self.current == spec:
            return RegisterResult(RegisterStatus.ALREADY, "matches")
        if self.current is not None and not force:
            return RegisterResult(RegisterStatus.MISMATCH, "different")
        self.current = spec
        return RegisterResult(RegisterStatus.REGISTERED, "updated")


def _quiet(monkeypatch):
    monkeypatch.setattr(wrap_cli, "_ensure_serena_dashboard_disabled", lambda **kwargs: None)
    monkeypatch.setattr(wrap_cli, "_inject_serena_instructions", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrap_cli, "_serena_project_skip_reason", lambda root: "test")
    monkeypatch.setattr(wrap_cli, "_index_serena_project", lambda **kwargs: None)
    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: "uvx" if name == "uvx" else None)


def test_automatic_wrap_migrates_owned_drift_and_recurs_to_noop(
    monkeypatch, tmp_path: Path, capsys
):
    _quiet(monkeypatch)
    monkeypatch.setattr(
        "headroom.mcp_registry.ledger.ledger_path", lambda: tmp_path / "ledger.json"
    )
    stale = ServerSpec("serena", "uvx", ("--from", "old"))
    record_install("claude", stale)
    registrar = _Registrar(stale)
    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)
    assert registrar.current == build_serena_spec("claude-code")
    assert registrar.force_calls == [False, True]
    assert headroom_installed_matching("claude", registrar.current)
    capsys.readouterr()
    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)
    assert registrar.force_calls == [False, True, False]


def test_automatic_wrap_owned_drift_suggests_rerun_wrap(monkeypatch, tmp_path: Path, capsys):
    _quiet(monkeypatch)
    monkeypatch.setattr(
        "headroom.mcp_registry.ledger.ledger_path", lambda: tmp_path / "ledger.json"
    )
    stale = ServerSpec("serena", "uvx", ("--from", "old"))
    record_install("claude", stale)

    class _FailedMigrationRegistrar(_Registrar):
        def register_server(self, spec, *, force=False):
            if force:
                self.force_calls.append(force)
                return RegisterResult(RegisterStatus.MISMATCH, "still different")
            return super().register_server(spec, force=force)

    wrap_cli._setup_serena_mcp(
        _FailedMigrationRegistrar(stale), context="claude-code", verbose=True
    )

    output = capsys.readouterr().out
    assert "run headroom wrap again" in output
    assert "mcp reconcile --adopt" not in output


def test_automatic_wrap_preserves_user_managed_warning(monkeypatch, tmp_path: Path, capsys):
    _quiet(monkeypatch)
    monkeypatch.setattr(
        "headroom.mcp_registry.ledger.ledger_path", lambda: tmp_path / "ledger.json"
    )
    user = ServerSpec("serena", "uvx", ("--from", "user"))
    registrar = _Registrar(user)
    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)
    assert registrar.current == user
    assert registrar.force_calls == [False]
    assert "existing config differs" in capsys.readouterr().out


def test_automatic_wrap_recovers_from_malformed_ledger(monkeypatch, tmp_path: Path):
    _quiet(monkeypatch)
    ledger = tmp_path / "ledger.json"
    ledger.write_text("not json")
    monkeypatch.setattr("headroom.mcp_registry.ledger.ledger_path", lambda: ledger)
    registrar = _Registrar(None)

    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    current = registrar.get_server("serena")
    assert current == build_serena_spec("claude-code")
    assert headroom_installed_matching("claude", current)


def test_non_claude_wrap_keeps_usable_remediation_hint(monkeypatch, tmp_path: Path, capsys):
    _quiet(monkeypatch)
    monkeypatch.setattr(
        "headroom.mcp_registry.ledger.ledger_path", lambda: tmp_path / "ledger.json"
    )
    registrar = _Registrar(ServerSpec("serena", "uvx", ("--from", "user")), name="codex")

    wrap_cli._setup_serena_mcp(registrar, context="codex", verbose=True)

    output = capsys.readouterr().out
    assert "update or remove the existing serena MCP entry" in output
    assert "mcp reconcile --adopt" not in output
