"""Tests for stateless TOIN behavior.

In stateless mode the proxy must keep TOIN's in-memory learning but never read
or write toin.json — part of the "stateless writes nothing to disk" guarantee.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from headroom.telemetry.toin import (
    TOINConfig,
    ToolIntelligenceNetwork,
    get_toin,
    reset_toin,
)


def test_empty_storage_path_means_in_memory(tmp_path):
    """Mechanism: an empty storage_path yields a None backend that never writes."""
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=""))
    assert toin._backend is None
    toin.save()  # must be a no-op
    assert not list(tmp_path.rglob("toin.json"))


def test_filesystem_backend_persists(tmp_path):
    """Control: a real storage_path uses a filesystem backend that does write."""
    path = tmp_path / "toin.json"
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path)))
    assert toin._backend is not None
    toin.save()
    assert path.exists()


def test_apply_stateless_persistence_forces_toin_in_memory(tmp_path, monkeypatch):
    """The proxy wiring (_apply_stateless_persistence) reconfigures the global
    TOIN singleton to in-memory when the config is stateless."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    from headroom.proxy.server import _apply_stateless_persistence

    reset_toin()
    try:
        _apply_stateless_persistence(SimpleNamespace(stateless=True))
        toin = get_toin()
        # No backend -> no toin.json read or written, ever.
        assert toin._backend is None
        toin.save()  # no-op
        assert not list(tmp_path.rglob("toin.json"))
    finally:
        reset_toin()


def test_apply_stateless_persistence_noop_when_not_stateless(tmp_path, monkeypatch):
    """When not stateless, the helper does nothing (default persistence kept)."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    from headroom.proxy.server import _apply_stateless_persistence

    reset_toin()
    try:
        # Helper is a no-op; the default singleton keeps a filesystem backend.
        _apply_stateless_persistence(SimpleNamespace(stateless=False))
        toin = get_toin()
        assert toin._backend is not None
    finally:
        reset_toin()
