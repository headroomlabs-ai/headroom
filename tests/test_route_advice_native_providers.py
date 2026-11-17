"""BackendResolver.for_request native-provider gating.

One proxy instance serves both the Anthropic and OpenAI wire from a single
shared resolver, so "which provider needs no translation" cannot be a global
constant — it depends on the inbound protocol. `native_providers` supplies it
per dispatch site. Regression for the `_NATIVE = ("anthropic",)` hardcode that
made an OpenAI-in request to a Claude model short-circuit to the OpenAI backend
(never translating, never rewriting the model).
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom import copilot_auth
from headroom.proxy.route_advice import BackendResolver, RouteAdvice
from headroom.proxy.server import ProxyConfig, create_app


class _Sentinel:
    """Stands in for a translating backend built by the resolver."""


def _request(model: str, provider: str):
    state = SimpleNamespace(headroom_route=RouteAdvice(model=model, provider=provider))
    return SimpleNamespace(state=state)


def _resolver() -> BackendResolver:
    resolver = BackendResolver(default="DEFAULT_BACKEND")
    # Avoid constructing a real litellm backend; prove routing, not litellm.
    resolver._build = lambda provider: _Sentinel()  # type: ignore[method-assign]
    return resolver


def test_no_advice_returns_default() -> None:
    resolver = _resolver()
    assert resolver.for_request(SimpleNamespace(state=SimpleNamespace())) == "DEFAULT_BACKEND"


def test_anthropic_target_is_native_for_anthropic_handler() -> None:
    # Default native set = anthropic (the Anthropic handler). A Claude target
    # needs no translation: the body is already Anthropic-shaped.
    resolver = _resolver()
    result = resolver.for_request(_request("claude-opus-5", "anthropic"))
    assert result == "DEFAULT_BACKEND"


def test_anthropic_target_is_foreign_for_openai_handler() -> None:
    # THE FIX: for an OpenAI-shape request, anthropic is NOT native — it must be
    # translated. A Claude target now resolves to a built backend, not default.
    resolver = _resolver()
    body: dict = {"model": "gpt-4o"}
    result = resolver.for_request(
        _request("claude-opus-5", "anthropic"), body=body, native_providers=("openai",)
    )
    assert isinstance(result, _Sentinel)
    assert body["model"] == "claude-opus-5", "body model must be rewritten to the advised model"


def test_openai_target_is_native_for_openai_handler() -> None:
    resolver = _resolver()
    result = resolver.for_request(_request("gpt-4o", "openai"), native_providers=("openai",))
    assert result == "DEFAULT_BACKEND"


def test_openai_target_is_foreign_for_anthropic_handler() -> None:
    # The already-working forward direction (Anthropic-in -> OpenAI-out).
    resolver = _resolver()
    body: dict = {"model": "claude-opus-5"}
    result = resolver.for_request(_request("gpt-4o", "openai"), body=body)
    assert isinstance(result, _Sentinel)
    assert body["model"] == "gpt-4o"


def test_failed_provider_falls_back_to_default() -> None:
    resolver = BackendResolver(default="DEFAULT_BACKEND")
    resolver._build = lambda provider: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
    # First call fails to build -> default; provider remembered as failed.
    r1 = resolver.for_request(_request("claude-opus-5", "anthropic"), native_providers=("openai",))
    assert r1 == "DEFAULT_BACKEND"
    r2 = resolver.for_request(_request("claude-opus-5", "anthropic"), native_providers=("openai",))
    assert r2 == "DEFAULT_BACKEND"


def test_copilot_upstream_ignores_cross_provider_advice() -> None:
    # A Copilot proxy holds only a Copilot token, so advice naming another
    # provider must not move the request or rewrite its model. A GHE host
    # counts as Copilot too.
    resolver = _resolver()
    body: dict = {"model": "gpt-5-mini"}
    result = resolver.for_request(
        _request("claude-opus-5", "anthropic"),
        body=body,
        native_providers=("openai",),
        upstream_url="https://copilot-api.acme.ghe.com",
    )
    assert result == "DEFAULT_BACKEND"
    assert body["model"] == "gpt-5-mini"


def test_copilot_chat_stays_on_copilot_despite_cross_provider_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A proxy in front of GitHub Copilot holds a Copilot token only, and Copilot
    # serves every model family on its own /chat/completions. Advice naming
    # another provider used to send the request to Anthropic through a
    # translating backend, where it failed for want of a key.
    class _Provider:
        async def get_api_token(self, *, integration_id=None):  # noqa: ANN001, ANN202
            return SimpleNamespace(token="test-copilot-token")

    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _Provider())
    built: list[str] = []
    monkeypatch.setattr(BackendResolver, "_build", lambda self, provider: built.append(provider))
    sent: list[tuple[str, str]] = []

    async def fake_retry(method, url, headers, body, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        sent.append((url, body["model"]))
        return httpx.Response(200, json={"choices": [], "usage": {}})

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            openai_api_url="https://api.business.githubcopilot.com",
        )
    )

    @app.middleware("http")
    async def publish_advice(request, call_next):  # noqa: ANN001, ANN202
        request.state.headroom_route = RouteAdvice(model="claude-sonnet-5-5", provider="anthropic")
        return await call_next(request)

    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 200, response.text
    assert built == [], "no translating backend may be built for a Copilot upstream"
    assert sent == [("https://api.business.githubcopilot.com/chat/completions", "gpt-5-mini")]
