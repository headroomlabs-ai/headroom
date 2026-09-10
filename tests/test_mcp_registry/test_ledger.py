from __future__ import annotations

import json

import pytest

import headroom.mcp_registry.ledger as ledger_module
from headroom.mcp_registry.base import ServerSpec
from headroom.mcp_registry.ledger import (
    LedgerMutationError,
    clear_install,
    headroom_installed_matching,
    record_install,
    spec_fingerprint,
    validate_ledger_for_mutation,
)


def _spec(command: str = "uvx") -> ServerSpec:
    return ServerSpec("serena", command, ("--from", "serena-agent", "serena"))


def test_ledger_records_and_clears_matching_install(tmp_path):
    ledger = tmp_path / "mcp_installs.json"
    spec = _spec()
    record_install("claude", spec, path=ledger)
    assert headroom_installed_matching("claude", spec, path=ledger)
    clear_install("claude", "serena", path=ledger)
    assert not headroom_installed_matching("claude", spec, path=ledger)


def test_spec_fingerprint_is_stable_for_env_order():
    a = ServerSpec("serena", "uvx", env={"B": "2", "A": "1"})
    b = ServerSpec("serena", "uvx", env={"A": "1", "B": "2"})
    assert spec_fingerprint(a) == spec_fingerprint(b)


@pytest.mark.parametrize(
    "value",
    [
        "not json",
        [],
        {"agents": None},
        {"agents": []},
        {"agents": {"claude": None}},
        {"agents": {"claude": []}},
        {"agents": {"claude": {"serena": None}}},
        {"agents": {"claude": {"serena": {"fingerprint": "only"}}}},
    ],
)
def test_mutation_preflight_rejects_unsafe_shapes(tmp_path, value):
    ledger = tmp_path / "mcp_installs.json"
    ledger.write_text(value if isinstance(value, str) else json.dumps(value))
    with pytest.raises(LedgerMutationError):
        validate_ledger_for_mutation(ledger)


def test_mutation_preflight_rejects_unreadable_ledger(monkeypatch, tmp_path):
    ledger = tmp_path / "mcp_installs.json"
    ledger.write_text('{"agents": {}}')
    original_read_text = ledger_module.Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == ledger:
            raise PermissionError("test unreadable ledger")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(ledger_module.Path, "read_text", unreadable)

    with pytest.raises(LedgerMutationError, match="unreadable"):
        validate_ledger_for_mutation(ledger)


def test_read_matching_tolerates_corrupt_ledger(tmp_path):
    ledger = tmp_path / "mcp_installs.json"
    ledger.write_text("not json")
    assert not headroom_installed_matching("claude", _spec(), path=ledger)


def test_record_install_recovers_from_corrupt_ledger(tmp_path):
    ledger = tmp_path / "mcp_installs.json"
    ledger.write_text("not json")
    spec = _spec()

    record_install("claude", spec, path=ledger)

    assert headroom_installed_matching("claude", spec, path=ledger)


@pytest.mark.parametrize("contents", ['{"agents": null}', '{"agents": {"claude": null}}'])
def test_record_install_recovers_from_unsafe_ledger_shape(tmp_path, contents):
    ledger = tmp_path / "mcp_installs.json"
    ledger.write_text(contents)

    record_install("claude", _spec(), path=ledger)

    assert headroom_installed_matching("claude", _spec(), path=ledger)
