"""Persistent, local-store-backed headroom_retrieve registration (headroom-h76.6).

The retrieve MCP is registered with agy PERSISTENTLY and recorded in the install
ledger (mirroring codebase-memory-mcp / Serena) so agy caches and exposes the
tool across sessions. These tests pin: a stable port-independent spec, ledger
recording on REGISTERED and ledger-owned ALREADY, ledger-cleared handshake failure, and a
ledger-gated cooperative uninstall.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.ccr.mcp_server import CCR_TOOL_NAME
from headroom.cli.wrap import (
    _remove_headroom_installed_retrieve_mcp,
    _setup_headroom_retrieve_mcp_agy,
)
from headroom.mcp_registry import build_headroom_spec
from headroom.mcp_registry.agy import AgyRegistrar
from headroom.mcp_registry.ledger import headroom_installed_matching


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the install ledger to a tmp file (no global state touched)."""
    ledger_file = tmp_path / "install_ledger.json"
    monkeypatch.setattr("headroom.mcp_registry.ledger.ledger_path", lambda: ledger_file)
    return ledger_file


def _reg(tmp_path: Path) -> AgyRegistrar:
    return AgyRegistrar(home_dir=tmp_path / "home")


def _ledgered(reg: AgyRegistrar) -> bool:
    return headroom_installed_matching(reg.name, reg.get_server("headroom"))


class TestSpecShape:
    def test_stable_port_independent_local_store_spec(self) -> None:
        spec = build_headroom_spec()
        assert spec.name == "headroom"
        # No ephemeral proxy URL -> child resolves from the on-disk store.
        assert dict(spec.env) == {}
        assert tuple(spec.args[-2:]) == ("mcp", "serve")


class TestPersistentRegistration:
    def test_registers_and_records_ledger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: True)
        reg = _reg(tmp_path)
        assert _setup_headroom_retrieve_mcp_agy(reg) is True
        assert reg.get_server("headroom") is not None
        assert _ledgered(reg) is True  # persistent: recorded, not reverted

    def test_idempotent_already_still_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: True)
        reg = _reg(tmp_path)
        assert _setup_headroom_retrieve_mcp_agy(reg) is True
        # Second run hits ALREADY; record_install upserts, ledger stays valid.
        assert _setup_headroom_retrieve_mcp_agy(reg) is True
        assert _ledgered(reg) is True

    def test_matching_unowned_entry_is_not_claimed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: True)
        reg = _reg(tmp_path)
        # A matching spec alone cannot prove ownership after ledger loss.
        reg.register_server(build_headroom_spec(), force=True)
        assert _ledgered(reg) is False
        assert _setup_headroom_retrieve_mcp_agy(reg) is True
        assert _ledgered(reg) is False

    def test_handshake_failure_removes_entry_and_clears_ledger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = _reg(tmp_path)
        # First: succeed to seed a ledger record + entry.
        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: True)
        assert _setup_headroom_retrieve_mcp_agy(reg) is True
        assert _ledgered(reg) is True
        # Now a broken child: entry removed AND ledger cleared (no dead pointer,
        # no stale ownership claim).
        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: False)
        assert _setup_headroom_retrieve_mcp_agy(reg) is False
        assert reg.get_server("headroom") is None
        assert _ledgered(reg) is False


class TestLedgerGatedUninstall:
    def test_removes_ledgered_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: True)
        reg = _reg(tmp_path)
        _setup_headroom_retrieve_mcp_agy(reg)
        assert _remove_headroom_installed_retrieve_mcp(reg) == "removed"
        assert reg.get_server("headroom") is None
        assert _ledgered(reg) is False

    def test_leaves_non_ledgered_entry(self, tmp_path: Path) -> None:
        reg = _reg(tmp_path)
        # A user/fleet-managed "headroom" entry NOT recorded by wrap agy.
        reg.register_server(build_headroom_spec(), force=True)
        assert _remove_headroom_installed_retrieve_mcp(reg) == "not_headroom_owned"
        assert reg.get_server("headroom") is not None  # left untouched

    def test_absent_entry_is_not_owned(self, tmp_path: Path) -> None:
        reg = _reg(tmp_path)
        assert _remove_headroom_installed_retrieve_mcp(reg) == "not_headroom_owned"


class TestMarkerToolAlignment:
    def test_marker_names_exact_tool(self) -> None:
        from headroom.transforms.agy_fr_compressor import _FR_CCR_MARKER_PREFIX

        assert CCR_TOOL_NAME in _FR_CCR_MARKER_PREFIX

    def test_tool_description_claims_headroom_markers(self) -> None:
        import inspect

        from headroom.ccr import mcp_server

        src = inspect.getsource(mcp_server)
        # Description must steer the model to this tool for headroom markers,
        # disambiguating from any other expand/retrieve tool in the session.
        assert "ONLY" in src and "Headroom compression markers" in src
        assert "functionResponse compressed. Call headroom_retrieve" in src


class TestFirstRunToolCachePriming:
    """wrap agy pre-seeds agy's per-tool cache so retrieve is exposed run 1."""

    def _cache_file(self, reg: AgyRegistrar) -> Path:
        return reg.cache_dir / "headroom" / f"{CCR_TOOL_NAME}.json"

    def test_setup_primes_retrieve_tool_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        from headroom.ccr.mcp_server import (
            CCR_RETRIEVE_TOOL_DESCRIPTION,
            CCR_RETRIEVE_TOOL_INPUT_SCHEMA,
        )

        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: True)
        reg = _reg(tmp_path)
        cache = self._cache_file(reg)
        assert not cache.exists()  # clean install: no cache yet

        assert _setup_headroom_retrieve_mcp_agy(reg) is True

        assert cache.is_file()  # primed on the first setup, not after a relaunch
        payload = json.loads(cache.read_text(encoding="utf-8"))
        # Schema must mirror the live list_tools() entry (single source), with
        # MCP inputSchema serialised under agy's ``parameters`` key.
        assert payload == {
            "name": CCR_TOOL_NAME,
            "description": CCR_RETRIEVE_TOOL_DESCRIPTION,
            "parameters": CCR_RETRIEVE_TOOL_INPUT_SCHEMA,
        }

    def test_priming_does_not_clobber_existing_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("headroom.cli.wrap._smoke_verify_mcp_handshake", lambda *a, **k: True)
        reg = _reg(tmp_path)
        cache = self._cache_file(reg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text('{"name": "headroom_retrieve", "stale": true}', encoding="utf-8")

        assert _setup_headroom_retrieve_mcp_agy(reg) is True

        # An agy-written cache is authoritative; priming must not overwrite it.
        assert cache.read_text(encoding="utf-8") == '{"name": "headroom_retrieve", "stale": true}'
