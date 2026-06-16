"""Unit tests for headroom.proxy lazy __getattr__ (PEP 562)."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def test_proxy_getattr_resolves_create_app_and_caches_it(monkeypatch) -> None:
    sentinel = object()
    fake_server = types.SimpleNamespace(create_app=sentinel, run_server=object())
    monkeypatch.setitem(sys.modules, "headroom.proxy.server", fake_server)

    import headroom.proxy as proxy

    module = importlib.reload(proxy)
    module.__dict__.pop("create_app", None)
    module.__dict__.pop("run_server", None)
    # Register teardown: monkeypatch notes these keys are absent now and will
    # remove them after the test, preventing sentinel leakage to later tests.
    monkeypatch.delitem(module.__dict__, "create_app", raising=False)
    monkeypatch.delitem(module.__dict__, "run_server", raising=False)

    result = module.__getattr__("create_app")
    assert result is sentinel
    assert module.__dict__["create_app"] is sentinel


def test_proxy_getattr_resolves_run_server(monkeypatch) -> None:
    sentinel = object()
    fake_server = types.SimpleNamespace(create_app=object(), run_server=sentinel)
    monkeypatch.setitem(sys.modules, "headroom.proxy.server", fake_server)

    import headroom.proxy as proxy

    module = importlib.reload(proxy)
    module.__dict__.pop("create_app", None)
    module.__dict__.pop("run_server", None)
    monkeypatch.delitem(module.__dict__, "create_app", raising=False)
    monkeypatch.delitem(module.__dict__, "run_server", raising=False)

    result = module.__getattr__("run_server")
    assert result is sentinel
    assert module.__dict__["run_server"] is sentinel


def test_proxy_getattr_raises_for_unknown_attribute() -> None:
    import headroom.proxy as proxy

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        proxy.__getattr__("nonexistent")
