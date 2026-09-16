"""Per-request backend selection published by an extension.

The property under test throughout is the one that makes this safe to merge:
with nothing published, every path is exactly what it was before.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from headroom.proxy.route_advice import (
    BackendResolver,
    RouteAdvice,
    advice_from,
)


class _Req:
    def __init__(self, **state):
        self.state = SimpleNamespace(**state)


DEFAULT = object()  # stands in for the configured backend


# --- reading what an extension published ------------------------------------


def test_no_extension_means_no_advice():
    assert advice_from(_Req()) is None
    assert advice_from(object()) is None  # no .state at all
    assert advice_from(None) is None


def test_advice_is_duck_typed_so_extensions_need_not_import_us():
    a = advice_from(
        _Req(
            headroom_route=SimpleNamespace(
                model="moonshot/kimi-k2", provider="moonshot", reason="cheaper"
            )
        )
    )
    assert a == RouteAdvice("moonshot/kimi-k2", "moonshot", "cheaper")


def test_an_extension_may_omit_everything_but_the_model():
    a = advice_from(_Req(headroom_route=SimpleNamespace(model="gpt-5-mini")))
    assert a.model == "gpt-5-mini" and a.provider == ""


def test_malformed_advice_is_ignored_rather_than_raised():
    for bad in (
        SimpleNamespace(),
        SimpleNamespace(model=""),
        SimpleNamespace(model=123),
        "not an object",
    ):
        assert advice_from(_Req(headroom_route=bad)) is None


def test_advice_needs_a_model():
    with pytest.raises(ValueError):
        RouteAdvice("")


# --- absent means unchanged, which is the whole safety argument -------------


def test_no_advice_returns_the_configured_backend():
    r = BackendResolver(DEFAULT)
    assert r.for_request(_Req()) is DEFAULT


def test_no_advice_and_no_configured_backend_stays_none():
    """None is not "no backend", it is the direct-API path. It must survive."""
    assert BackendResolver(None).for_request(_Req()) is None


def test_a_native_provider_does_not_switch_backends():
    """Anthropic is the shape the proxy already holds, so a model rewrite is
    enough and the extension has already done it."""
    r = BackendResolver(DEFAULT)
    req = _Req(headroom_route=SimpleNamespace(model="claude-haiku-4-5", provider="anthropic"))
    assert r.for_request(req) is DEFAULT


def test_a_bare_anthropic_model_resolves_its_provider_and_stays_put():
    r = BackendResolver(DEFAULT)
    req = _Req(headroom_route=SimpleNamespace(model="claude-haiku-4-5"))
    assert r.for_request(req) is DEFAULT


# --- switching, and refusing to switch --------------------------------------


def test_a_foreign_provider_gets_its_own_backend(monkeypatch):
    built = []

    class FakeBackend:
        def __init__(self, provider):
            built.append(provider)
            self.provider = provider

    monkeypatch.setattr(BackendResolver, "_build", lambda self, p: FakeBackend(p))
    r = BackendResolver(DEFAULT)
    body = {"model": "claude-opus-5"}
    req = _Req(headroom_route=SimpleNamespace(model="moonshot/kimi-k2", provider="moonshot"))
    got = r.for_request(req, body=body)
    assert isinstance(got, FakeBackend) and got.provider == "moonshot"
    # The extension could not safely write a foreign model id; we do it.
    assert body["model"] == "moonshot/kimi-k2"


def test_backends_are_built_once_per_provider(monkeypatch):
    built = []
    monkeypatch.setattr(BackendResolver, "_build", lambda self, p: built.append(p) or object())
    r = BackendResolver(DEFAULT)
    req = _Req(headroom_route=SimpleNamespace(model="x", provider="moonshot"))
    for _ in range(5):
        r.for_request(req)
    assert built == ["moonshot"], "construction is expensive; cache it"


def test_a_backend_that_will_not_build_falls_back_and_stops_retrying(monkeypatch):
    calls = []
    monkeypatch.setattr(BackendResolver, "_build", lambda self, p: calls.append(p) or None)
    r = BackendResolver(DEFAULT)
    req = _Req(headroom_route=SimpleNamespace(model="x", provider="nope"))
    for _ in range(5):
        assert r.for_request(req) is DEFAULT
    assert calls == ["nope"], "a broken provider must not be retried per request"


def test_a_routing_preference_can_never_take_traffic_down(monkeypatch):
    """Missing credentials, missing optional dependency -- whatever the reason,
    the request still has to be served."""

    def boom(self, provider):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(BackendResolver, "_build", boom)
    r = BackendResolver(DEFAULT)
    req = _Req(headroom_route=SimpleNamespace(model="x", provider="moonshot"))
    assert r.for_request(req) is DEFAULT


def test_an_unknown_provider_is_rejected_at_resolve_time():
    """LiteLLMBackend accepts ANY provider string -- the registry falls through
    to a generic pass-through -- so a typo silently builds a backend that only
    fails later, at request time, with an error pointing nowhere near the typo.
    Validate the name up front instead."""
    r = BackendResolver(DEFAULT)
    assert r._build("definitely-not-a-provider-name") is None
    req = _Req(headroom_route=SimpleNamespace(model="x", provider="definitely-not-a-provider-name"))
    assert r.for_request(req) is DEFAULT


def test_a_real_provider_name_is_accepted():
    assert BackendResolver(DEFAULT)._build("moonshot") is not None


# --- streaming, which is the path agents actually take ----------------------


class _Backend:
    """Records that it, and not some other backend, served the request."""

    def __init__(self, name):
        self.name = name
        self.served = False

    async def stream_message(self, body, headers):
        self.served = True
        return
        yield  # pragma: no cover -- makes this an async generator

    async def stream_openai_message(self, body, headers):
        self.served = True
        return
        yield  # pragma: no cover


async def _drive(handler, **kw):
    from headroom.proxy.handlers.streaming import StreamingMixin

    resp = await StreamingMixin._stream_response_bedrock(
        handler,
        {"messages": []},
        {},
        "anthropic",
        "m",
        "rid",
        0,
        0,
        0,
        [],
        {},
        0.0,
        **kw,
    )
    async for _ in resp.body_iterator:
        pass


class _Config:
    """Every proxy flag off. Named individually the list would rot; the test
    is about which backend served the request, not about config."""

    def __getattr__(self, name):
        return False


def _handler(default):
    return SimpleNamespace(
        anthropic_backend=default,
        config=_Config(),
        _record_request_outcome=lambda outcome: _noop(),
    )


async def _noop():
    return None


def test_streaming_honours_the_routed_backend():
    """The non-streaming branch was the easy half. body["model"] has already
    been rewritten to a foreign id by the time we get here, so streaming to
    the configured backend would send e.g. moonshot/kimi-k2 to Anthropic."""
    configured, routed = _Backend("anthropic"), _Backend("moonshot")
    asyncio.run(_drive(_handler(configured), backend=routed))
    assert routed.served and not configured.served


def test_streaming_without_a_route_uses_the_configured_backend():
    configured = _Backend("anthropic")
    asyncio.run(_drive(_handler(configured)))
    assert configured.served


async def _drive_openai(handler, **kw):
    from headroom.proxy.handlers.streaming import StreamingMixin

    resp = await StreamingMixin._stream_openai_via_backend(
        handler,
        {"messages": []},
        {},
        "m",
        "rid",
        0.0,
        0,
        0,
        0,
        [],
        {},
        0.0,
        **kw,
    )
    async for _ in resp.body_iterator:
        pass


def test_openai_streaming_honours_the_routed_backend():
    """opencode and pi can speak either protocol, so the OpenAI chat path
    needs the same treatment as the Anthropic one."""
    configured, routed = _Backend("openai"), _Backend("moonshot")
    asyncio.run(_drive_openai(_handler(configured), backend=routed))
    assert routed.served and not configured.served


def test_openai_streaming_without_a_route_uses_the_configured_backend():
    configured = _Backend("openai")
    asyncio.run(_drive_openai(_handler(configured)))
    assert configured.served


def test_the_resolver_follows_a_reassigned_default():
    class H:
        anthropic_backend = None

    h = H()
    from headroom.proxy.route_advice import BackendResolver as BR

    first = BR(h.anthropic_backend)
    assert first.default is None
    h.anthropic_backend = DEFAULT
    assert BR(h.anthropic_backend).default is DEFAULT
