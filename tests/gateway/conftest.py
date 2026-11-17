"""Fixtures for the gateway turn contract tests.

* ``make_headroom_client`` / ``headroom_client`` - the proxy app built like
  ``tests/test_compress_session_mode.py::_make_client`` and served through a
  loopback ``TestClient`` (``/v1/compress`` and ``/v1/compress/response`` are
  loopback-gated).
* ``fake_provider`` / ``provider_client`` - a :class:`FakeProvider` and a
  ``TestClient`` mounted on its ASGI app.
* ``make_gateway`` - a :class:`FakeGateway` factory bound to the two clients.
* ``outcome_spy`` - captures every ``RequestOutcome`` the proxy records.
* Autouse: turn hooks cleared before/after each test; ``HEADROOM_GATEWAY_*``
  scrubbed (the repo conftest already scrubs every ``HEADROOM_*`` var, this one
  makes the dependency explicit for the knobs these tests set).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.proxy.turn_hooks import clear_turn_hooks  # noqa: E402
from tests.gateway.fake_gateway import FakeGateway  # noqa: E402
from tests.gateway.fake_provider import FakeProvider  # noqa: E402

LOOPBACK = ("127.0.0.1", 12345)


@pytest.fixture(autouse=True)
def _clear_hooks() -> Iterator[None]:
    clear_turn_hooks()
    yield
    clear_turn_hooks()


@pytest.fixture(autouse=True)
def _scrub_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("HEADROOM_GATEWAY_") or key.startswith("HEADROOM_TEST_REDRIVE_"):
            monkeypatch.delenv(key, raising=False)


def proxy_config(**overrides: Any) -> ProxyConfig:
    """The same posture as the session-mode tests: optimize on, everything that
    would reach the network or add per-request noise off."""
    kwargs: dict[str, Any] = {
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "image_optimize": False,
    }
    kwargs.update(overrides)
    return ProxyConfig(**kwargs)


@pytest.fixture
def make_headroom_client() -> Iterator[Callable[..., TestClient]]:
    """Factory: ``make_headroom_client(**ProxyConfig overrides) -> TestClient``.

    Each client is entered (lifespan started) and closed at teardown. Use the
    factory when a test must set env vars BEFORE the app is created."""
    opened: list[TestClient] = []

    def _make(**overrides: Any) -> TestClient:
        app = create_app(proxy_config(**overrides))
        client = TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK)
        client.__enter__()
        opened.append(client)
        return client

    yield _make
    for client in reversed(opened):
        client.__exit__(None, None, None)


@pytest.fixture
def headroom_client(make_headroom_client: Callable[..., TestClient]) -> TestClient:
    return make_headroom_client()


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def provider_client(fake_provider: FakeProvider) -> Iterator[TestClient]:
    with TestClient(fake_provider.app) as client:
        yield client


@pytest.fixture
def make_gateway(
    headroom_client: TestClient, provider_client: TestClient
) -> Callable[..., FakeGateway]:
    def _make(**kwargs: Any) -> FakeGateway:
        kwargs.setdefault("can_redrive", True)
        kwargs.setdefault("can_relay_response", True)
        return FakeGateway(headroom_client, provider_client, **kwargs)

    return _make


@pytest.fixture
def outcome_spy(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], list[Any]]:
    """``outcome_spy(client_or_app_or_proxy) -> list`` that fills with every
    ``RequestOutcome`` recorded by that proxy instance (patched on the instance,
    so two apps in one test get separate lists)."""

    def _attach(target: Any) -> list[Any]:
        proxy = target
        if hasattr(target, "app"):
            proxy = target.app.state.proxy
        elif hasattr(target, "state"):
            proxy = target.state.proxy
        outcomes: list[Any] = []

        async def _spy(outcome: Any, *args: Any, **kwargs: Any) -> None:
            outcomes.append(outcome)

        monkeypatch.setattr(proxy, "_record_request_outcome", _spy)
        return outcomes

    return _attach


# --------------------------------------------------------------------------- #
# Helpers (plain functions, importable from the tests)                         #
# --------------------------------------------------------------------------- #


def compress(
    client: TestClient, body: dict[str, Any], *, headers: dict[str, str] | None = None
) -> Any:
    return client.post("/v1/compress", json=body, headers=headers)


def gateway_block(**flags: Any) -> dict[str, Any]:
    return dict(flags)


def count_loop_tasks(client: TestClient) -> int:
    """Number of asyncio tasks alive on the app's event loop (the TestClient
    portal thread). Used to prove the response half leaks no suspended hook
    tasks."""
    assert client.portal is not None, "client must be entered"
    return client.portal.call(lambda: len(asyncio.all_tasks()))


def registry_of(client: TestClient) -> Any:
    """The proxy's ``PendingTurnRegistry`` (``None`` until the core lands)."""
    return getattr(client.app.state.proxy, "gateway_turns", None)
