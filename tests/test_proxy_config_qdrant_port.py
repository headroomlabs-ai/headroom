"""ProxyConfig construction must survive a bad HEADROOM_QDRANT_PORT.

The port is resolved by a field default_factory that runs on every ProxyConfig()
construction, so a stray/typo'd value must not crash proxy startup for an
off-by-default subsystem."""

from __future__ import annotations

import pytest

from headroom.memory import qdrant_env
from headroom.proxy.models import ProxyConfig, _qdrant_env_port_or_default


@pytest.mark.parametrize("bad", ["not-a-port", "70000", "0"])
def test_bad_qdrant_port_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv("HEADROOM_QDRANT_PORT", bad)
    assert _qdrant_env_port_or_default() == qdrant_env.DEFAULT_QDRANT_PORT


def test_valid_qdrant_port_is_honored(monkeypatch):
    monkeypatch.setenv("HEADROOM_QDRANT_PORT", "6444")
    assert _qdrant_env_port_or_default() == 6444


def test_proxyconfig_construction_survives_bad_qdrant_port(monkeypatch):
    monkeypatch.setenv("HEADROOM_QDRANT_PORT", "not-a-port")

    # Must not raise even though the port is unparseable (memory is off by
    # default and unrelated to core proxying).
    config = ProxyConfig()

    assert config.memory_qdrant_port == qdrant_env.DEFAULT_QDRANT_PORT
