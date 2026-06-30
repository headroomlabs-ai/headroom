"""Re-wrap must migrate a stale Headroom-installed Serena entry.

The dashboard-popup fix (#1003) added ``--open-web-dashboard False`` to the
Serena spec, but ``register_server`` refuses to overwrite a differing entry
without ``force``. So an already-wrapped user whose ``serena`` entry predates
the flag would keep the old spec — and the popup — on every re-wrap.

``_setup_serena_mcp`` closes that gap: when the ledger proves Headroom
installed the entry currently on disk, it force-updates to the current spec.
A user-managed Serena (absent from the ledger) is left untouched and the
mismatch is reported exactly as before.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from headroom.cli import wrap as wrap_cli
from headroom.mcp_registry import build_serena_spec
from headroom.mcp_registry.base import RegisterResult, RegisterStatus, ServerSpec
from headroom.mcp_registry.ledger import headroom_installed_matching, record_install

# The Serena spec Headroom wrote before the dashboard flag existed.
_STALE_SERENA_SPEC = ServerSpec(
    name="serena",
    command="uvx",
    args=(
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server",
        "--project-from-cwd",
        "--context",
        "claude-code",
    ),
)


def _equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    return (a.command, tuple(a.args), dict(a.env)) == (b.command, tuple(b.args), dict(b.env))


class _FakeRegistrar:
    """Registrar mirroring real ``register_server`` overwrite semantics."""

    def __init__(self, name: str, *, server: ServerSpec | None = None):
        self.name = name
        self.display_name = name.capitalize()
        self._server = server
        self.force_calls: list[bool] = []

    def detect(self) -> bool:
        return True

    def get_server(self, server_name: str) -> ServerSpec | None:
        return self._server if server_name == "serena" else None

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        self.force_calls.append(force)
        if self._server is not None:
            if _equivalent(self._server, spec):
                return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")
            if not force:
                return RegisterResult(RegisterStatus.MISMATCH, "args differ")
        self._server = spec
        return RegisterResult(RegisterStatus.REGISTERED, "registered")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    # These tests drive ``_setup_serena_mcp`` with a fake registrar, so the real
    # PATH is irrelevant — but the function bails early when ``uvx`` is absent.
    # CI test shards run on runners without uvx, which would skip every code
    # path under test. Stub uvx discovery so behaviour is PATH-independent.
    real_which = shutil.which
    monkeypatch.setattr(
        wrap_cli.shutil,
        "which",
        lambda name, *a, **k: "/usr/bin/uvx" if name == "uvx" else real_which(name, *a, **k),
    )


def test_rewrap_migrates_stale_headroom_serena(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Ledger proves Headroom installed the stale entry that's on disk.
    record_install("claude", _STALE_SERENA_SPEC)
    registrar = _FakeRegistrar("claude", server=_STALE_SERENA_SPEC)

    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    fresh = build_serena_spec("claude-code")
    assert _equivalent(registrar.get_server("serena"), fresh)  # entry replaced
    assert "--open-web-dashboard" in registrar.get_server("serena").args
    assert registrar.force_calls == [False, True]  # tried gentle, then forced
    out = capsys.readouterr().out
    assert "migrated previously-installed entry" in out
    # Ledger now tracks the new spec, so a subsequent re-wrap is a no-op match.
    assert headroom_installed_matching("claude", fresh)


def test_rewrap_leaves_user_managed_serena(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Differs from the current spec but is NOT in Headroom's ledger.
    user_spec = ServerSpec(name="serena", command="/usr/local/bin/custom-serena")
    registrar = _FakeRegistrar("claude", server=user_spec)

    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    assert registrar.get_server("serena") is user_spec  # never overwritten
    assert registrar.force_calls == [False]  # no forced retry
    out = capsys.readouterr().out
    assert "existing config differs" in out
    assert "migrated" not in out


def test_rewrap_fresh_install_records_dashboard_off_spec(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registrar = _FakeRegistrar("claude", server=None)

    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    entry = registrar.get_server("serena")
    assert entry is not None
    assert ("--open-web-dashboard", "False") == tuple(entry.args[-2:])
    assert registrar.force_calls == [False]  # no entry → no forced retry needed
    assert headroom_installed_matching("claude", entry)


def test_rewrap_already_current_is_noop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    current = build_serena_spec("claude-code")
    registrar = _FakeRegistrar("claude", server=current)

    wrap_cli._setup_serena_mcp(registrar, context="claude-code", verbose=True)

    assert registrar.force_calls == [False]  # ALREADY → no migration, no force
    assert "migrated" not in capsys.readouterr().out
