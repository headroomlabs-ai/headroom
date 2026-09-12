"""Retiring the legacy tokensave entry stays at user scope (#2992 review).

tokensave was replaced by Serena and is only ever cleaned up now: ``wrap``
retires the entry a prior release installed so the agent stops launching it,
and ``unwrap`` does the same. Headroom only ever installed tokensave
machine-wide — it was retired before #2787 gave Claude registrations a project
scope — so the ledger record authorizing the removal is the pre-scope,
user-scope one.

``ClaudeRegistrar.unregister_server()`` with no scope now deletes from the
project *and* the user scope, so an unscoped retirement would take a
project-local tokensave the user registered themselves, which no ledger record
authorizes. An unscoped read has the mirror problem: a project entry is found
first and shadows the machine-wide one Headroom owns, leaving ours behind
forever.

These tests drive the real helpers with a real ``ClaudeRegistrar`` whose home
directory is a tmp_path, so the assertions are about the JSON Claude Code
actually reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from headroom.cli import wrap as wrap_cli
from headroom.mcp_registry.base import ServerSpec
from headroom.mcp_registry.claude import SCOPE_LOCAL, SCOPE_USER, ClaudeRegistrar
from headroom.mcp_registry.ledger import record_install

HEADROOM_SPEC = ServerSpec(name="tokensave", command="tokensave", args=("mcp",))
USER_SPEC = ServerSpec(name="tokensave", command="/usr/local/bin/my-own-tokensave")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))


def _registrar(tmp_path: Path, *, scope: str = SCOPE_LOCAL) -> ClaudeRegistrar:
    """A real registrar pointed at ``tmp_path`` instead of the user's home."""
    (tmp_path / ".claude").mkdir(exist_ok=True)
    return ClaudeRegistrar(
        claude_cli=None,  # no CLI path; never touch the real one
        home_dir=tmp_path,
        scope=scope,
        project_dir=tmp_path / "proj",
    )


def _entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {"command": spec.command, "args": list(spec.args)}
    if spec.env:
        entry["env"] = dict(spec.env)
    return entry


def _seed(tmp_path: Path, *, global_spec: ServerSpec, project_spec: ServerSpec) -> None:
    """A machine-wide tokensave plus a project-scoped one, as Claude Code stores them."""
    config = {
        "mcpServers": {"tokensave": _entry(global_spec)},
        "projects": {
            (tmp_path / "proj").as_posix(): {"mcpServers": {"tokensave": _entry(project_spec)}}
        },
    }
    (tmp_path / ".claude.json").write_text(json.dumps(config), encoding="utf-8")


def test_retirement_spares_an_identical_user_managed_project_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same command in both scopes: only the one the ledger authorized goes."""
    _seed(tmp_path, global_spec=HEADROOM_SPEC, project_spec=HEADROOM_SPEC)
    record_install("claude", HEADROOM_SPEC)  # pre-scope ledger key: "tokensave"
    registrar = _registrar(tmp_path)

    wrap_cli._disable_tokensave_mcp(registrar)

    assert registrar.get_server("tokensave", scope=SCOPE_USER) is None
    assert registrar.get_server("tokensave", scope=SCOPE_LOCAL) == HEADROOM_SPEC
    assert "Removed retired tokensave MCP" in capsys.readouterr().out


def test_a_user_managed_project_entry_does_not_shadow_the_global_one(
    tmp_path: Path,
) -> None:
    """A project entry we do not own must not block retiring the one we do."""
    _seed(tmp_path, global_spec=HEADROOM_SPEC, project_spec=USER_SPEC)
    record_install("claude", HEADROOM_SPEC)
    registrar = _registrar(tmp_path)

    assert wrap_cli._remove_headroom_installed_tokensave_mcp(registrar) == "removed"
    assert registrar.get_server("tokensave", scope=SCOPE_USER) is None
    assert registrar.get_server("tokensave", scope=SCOPE_LOCAL) == USER_SPEC


def test_unwrap_leaves_a_user_managed_project_entry_alone(tmp_path: Path) -> None:
    """unwrap builds a user-scope registrar; it still must not reach into the project."""
    _seed(tmp_path, global_spec=HEADROOM_SPEC, project_spec=HEADROOM_SPEC)
    record_install("claude", HEADROOM_SPEC)
    registrar = _registrar(tmp_path, scope=SCOPE_USER)

    assert wrap_cli._remove_headroom_installed_tokensave_mcp(registrar) == "removed"
    assert registrar.get_server("tokensave", scope=SCOPE_USER) is None
    assert registrar.get_server("tokensave", scope=SCOPE_LOCAL) == HEADROOM_SPEC


def test_a_user_managed_global_entry_is_left_in_place(tmp_path: Path) -> None:
    """Unchanged behaviour: no ledger record, no removal."""
    _seed(tmp_path, global_spec=USER_SPEC, project_spec=USER_SPEC)
    registrar = _registrar(tmp_path)

    assert wrap_cli._remove_headroom_installed_tokensave_mcp(registrar) == "not_headroom_owned"
    assert registrar.get_server("tokensave", scope=SCOPE_USER) == USER_SPEC
    assert registrar.get_server("tokensave", scope=SCOPE_LOCAL) == USER_SPEC


def test_registrars_without_scopes_still_retire_tokensave(tmp_path: Path) -> None:
    """Codex/Grok/OpenCode registrars have no scope and reject a scoped read."""

    class _FakeRegistrar:
        name = "codex"
        display_name = "Codex"

        def __init__(self) -> None:
            self.server: ServerSpec | None = HEADROOM_SPEC

        def detect(self) -> bool:
            return True

        def get_server(self, server_name: str) -> ServerSpec | None:
            return self.server if server_name == "tokensave" else None

        def unregister_server(self, server_name: str) -> bool:
            if server_name != "tokensave" or self.server is None:
                return False
            self.server = None
            return True

    record_install("codex", HEADROOM_SPEC)
    registrar = _FakeRegistrar()

    assert wrap_cli._remove_headroom_installed_tokensave_mcp(registrar) == "removed"
    assert registrar.server is None
