from __future__ import annotations

from headroom.mcp_registry.base import ServerSpec
from headroom.mcp_registry.ledger import (
    clear_install,
    headroom_installed_matching,
    record_install,
    spec_fingerprint,
)


def _spec(command: str = "uvx") -> ServerSpec:
    return ServerSpec(
        name="serena",
        command=command,
        args=("--from", "git+https://github.com/oraios/serena", "serena"),
    )


def test_ledger_records_matching_install(tmp_path):
    ledger = tmp_path / "mcp_installs.json"
    spec = _spec()

    record_install("claude", spec, path=ledger)

    assert headroom_installed_matching("claude", spec, path=ledger) is True


def test_ledger_rejects_changed_spec(tmp_path):
    ledger = tmp_path / "mcp_installs.json"

    record_install("claude", _spec(), path=ledger)

    assert (
        headroom_installed_matching("claude", _spec(command="/custom/serena"), path=ledger) is False
    )


def test_clear_install_removes_entry(tmp_path):
    ledger = tmp_path / "mcp_installs.json"
    spec = _spec()
    record_install("claude", spec, path=ledger)

    clear_install("claude", "serena", path=ledger)

    assert headroom_installed_matching("claude", spec, path=ledger) is False


def test_spec_fingerprint_stable_for_env_order():
    a = ServerSpec(name="serena", command="uvx", env={"B": "2", "A": "1"})
    b = ServerSpec(name="serena", command="uvx", env={"A": "1", "B": "2"})

    assert spec_fingerprint(a) == spec_fingerprint(b)
