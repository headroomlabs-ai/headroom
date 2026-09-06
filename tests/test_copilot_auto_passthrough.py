"""Copilot "Auto" must survive the proxy untouched.

Auto is not a model on the wire. Every Copilot client resolves it to a concrete
model before the chat request leaves the machine, either through
``POST /models/session`` (plus the optional ``/models/session/intent`` router) or
the single-call ``POST /auto``. The chat request then carries the concrete model
id and a ``Copilot-Session-Token`` header; GitHub grants the Auto discount
against that token. Three things therefore have to hold through Headroom:

1. the session header (and Copilot's other routing headers) reach upstream;
2. the resolution calls pass through to Copilot with their paths intact, and
   are not mistaken for chat turns;
3. a Copilot-authenticated request never lands on a stock provider host.

Nothing here was pinned before, so a header-policy or routing change could have
silently cost every Auto user their discount.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from headroom import copilot_auth
from headroom.copilot_auth import (
    CopilotAPIToken,
    apply_copilot_api_auth,
    build_copilot_upstream_url,
    copilot_bearer_upstream,
)
from headroom.providers.openai_responses import OPENAI_RESPONSES_ROOT_PATHS
from headroom.providers.proxy_targets import select_passthrough_base_url
from headroom.providers.route_specs import OPENAI_HANDLER_ROUTES
from headroom.proxy.internal_header_policy import strip_internal_headers

COPILOT_API = "https://api.githubcopilot.com"

#: Headers the VS Code extension and the Copilot CLI add to a chat request. The
#: session token is the one Auto depends on; the rest tell CAPI which feature
#: and interaction flighted the request.
COPILOT_ROUTING_HEADERS = {
    "Copilot-Session-Token": "sess_auto_abc",
    "X-Initiator": "user",
    "OpenAI-Intent": "conversation-agent",
    "X-Interaction-Type": "conversation-agent",
    "X-GitHub-Api-Version": "2026-08-01",
}

#: CAPI paths Auto resolution uses, plus the ancillary calls the same clients
#: make. None of them carries a `/v1` prefix, and none may be rewritten.
AUTO_RESOLUTION_PATHS = ["/models/session", "/models/session/intent", "/auto"]
ANCILLARY_PATHS = ["/models", "/models/gpt-5.5/policy", "/agents/sessions", "/embeddings"]


@pytest.fixture(autouse=True)
def _clean_copilot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_ENTERPRISE_URL",
        "GITHUB_COPILOT_ENTERPRISE_DOMAIN",
        "GITHUB_COPILOT_API_TOKEN",
        "GITHUB_COPILOT_REFRESH_OAUTH_TOKEN",
        "HEADROOM_STRIP_INTERNAL_HEADERS",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# 1. Header survival
# --------------------------------------------------------------------------- #
def test_internal_header_strip_leaves_copilot_routing_headers_alone() -> None:
    headers = {**COPILOT_ROUTING_HEADERS, "x-headroom-bypass": "1", "Authorization": "Bearer tid_x"}
    stripped = strip_internal_headers(headers, mode="enabled")
    assert "x-headroom-bypass" not in stripped
    for name, value in COPILOT_ROUTING_HEADERS.items():
        assert stripped[name] == value


def test_session_token_survives_when_the_client_token_passes_through() -> None:
    """VS Code and the CLI send their own `tid_` token: Headroom forwards it as-is."""
    headers = {**COPILOT_ROUTING_HEADERS, "Authorization": "Bearer tid_client_token"}
    resolved = asyncio.run(apply_copilot_api_auth(headers, url=f"{COPILOT_API}/chat/completions"))
    assert resolved["Authorization"] == "Bearer tid_client_token"
    for name, value in COPILOT_ROUTING_HEADERS.items():
        assert resolved[name] == value


def test_session_token_survives_when_headroom_substitutes_its_own_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replacement branch rewrites Authorization and the integration id only."""

    class Provider:
        async def get_api_token(self, *, integration_id=None):  # noqa: ANN001
            return CopilotAPIToken(token="tid_headroom", expires_at=time.time() + 3600)

    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: Provider())
    headers = {**COPILOT_ROUTING_HEADERS, "Authorization": "Bearer not-a-copilot-token"}
    resolved = asyncio.run(apply_copilot_api_auth(headers, url=f"{COPILOT_API}/responses"))
    assert resolved["Authorization"] == "Bearer tid_headroom"
    for name, value in COPILOT_ROUTING_HEADERS.items():
        assert resolved[name] == value


# --------------------------------------------------------------------------- #
# 2. Resolution calls keep their paths and are not chat routes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", AUTO_RESOLUTION_PATHS + ANCILLARY_PATHS)
def test_copilot_native_paths_are_forwarded_verbatim(path: str) -> None:
    assert build_copilot_upstream_url(COPILOT_API, path) == f"{COPILOT_API}{path}"


def test_generic_openai_paths_still_lose_their_v1_prefix_at_copilot() -> None:
    """The `/v1` strip exists for OpenAI-SDK clients; Copilot's own paths never had one."""
    assert build_copilot_upstream_url(COPILOT_API, "/v1/chat/completions") == (
        f"{COPILOT_API}/chat/completions"
    )
    assert build_copilot_upstream_url(COPILOT_API, "/v1/responses") == f"{COPILOT_API}/responses"


@pytest.mark.parametrize("path", AUTO_RESOLUTION_PATHS)
def test_resolution_paths_are_not_registered_as_chat_handlers(path: str) -> None:
    """The router and `/auto` bodies carry the raw prompt; they must never be compressed."""
    handler_paths = {route.path for route in OPENAI_HANDLER_ROUTES}
    assert path not in handler_paths
    assert path not in OPENAI_RESPONSES_ROOT_PATHS


def test_the_chat_request_that_follows_auto_is_a_compressed_route() -> None:
    """Both unprefixed CAPI chat paths must hit real handlers, not the passthrough."""
    handler_paths = {route.path for route in OPENAI_HANDLER_ROUTES}
    assert "/chat/completions" in handler_paths
    assert "/responses" in OPENAI_RESPONSES_ROOT_PATHS


# --------------------------------------------------------------------------- #
# 3. A Copilot credential never lands on a stock provider host
# --------------------------------------------------------------------------- #
def _proxy(openai_target: str):
    class Runtime:
        @staticmethod
        def api_target(provider: str) -> str:
            return f"https://runtime.{provider}.test"

        @staticmethod
        def model_metadata_provider(headers) -> str:  # type: ignore[no-untyped-def]
            return "anthropic" if headers.get("x-api-key") else "openai"

    return type("Proxy", (), {"OPENAI_API_URL": openai_target, "provider_runtime": Runtime()})()


#: The bearer GitHub actually mints: a semicolon claim string, not a prefix.
REAL_COPILOT_API_TOKEN = "tid=0123456789abcdef0123456789abcdef;exp=1893456000;sku=copilot_for_business_seat;st=dotcom:9f8e7d6c"


@pytest.mark.parametrize("token", [REAL_COPILOT_API_TOKEN, "tid_session", "gho_oauth"])
def test_copilot_bearer_redirects_away_from_stock_hosts(token: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    assert copilot_bearer_upstream(headers, "https://api.openai.com") == COPILOT_API
    assert copilot_bearer_upstream(headers, "https://api.anthropic.com") == COPILOT_API


@pytest.mark.parametrize(
    "token",
    [
        "ghu_user_to_server",
        "ghp_personal",
        "github_pat_11A",
        "ghs_installation",
        "tid=notahex;exp=1",
    ],
)
def test_copilot_bearer_redirect_ignores_tokens_that_do_not_name_copilot(token: str) -> None:
    """Every routed token must also be forwardable, or the redirect would make
    ``apply_copilot_api_auth`` swap in the operator's own seat for any caller."""
    assert (
        copilot_bearer_upstream({"authorization": f"Bearer {token}"}, "https://api.openai.com")
        is None
    )


@pytest.mark.parametrize("token", [REAL_COPILOT_API_TOKEN, "tid_session", "gho_oauth"])
def test_every_routed_token_is_also_forwarded_unchanged(token: str) -> None:
    assert copilot_auth._is_copilot_routing_bearer(token) is True
    assert copilot_auth._is_forwardable_copilot_bearer_token(token) is True


def test_stock_host_is_recognised_with_a_port_and_path() -> None:
    assert (
        copilot_bearer_upstream(
            {"authorization": f"Bearer {REAL_COPILOT_API_TOKEN}"}, "https://api.openai.com:443/v1"
        )
        == COPILOT_API
    )


@pytest.mark.parametrize(
    ("headers", "target"),
    [
        # A personal access token is a valid GitHub credential for other products.
        ({"Authorization": "Bearer ghp_personal"}, "https://api.openai.com"),
        ({"Authorization": "Bearer github_pat_personal"}, "https://api.openai.com"),
        # An OpenAI key is not Copilot's.
        ({"Authorization": "Bearer sk-openai"}, "https://api.openai.com"),
        # The operator pinned a gateway: respect it.
        ({"Authorization": "Bearer tid_session"}, "https://litellm.corp.example/v1"),
        # The client named its own upstream for this request.
        (
            {"Authorization": "Bearer tid_session", "x-headroom-base-url": "https://gw.example"},
            "https://api.openai.com",
        ),
        ({}, "https://api.openai.com"),
    ],
)
def test_copilot_bearer_redirect_stays_out_of_everything_else(
    headers: dict[str, str], target: str
) -> None:
    assert copilot_bearer_upstream(headers, target) is None


def test_copilot_bearer_redirect_honours_the_configured_copilot_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://copilot-api.acme.ghe.com")
    assert (
        copilot_bearer_upstream({"Authorization": "Bearer tid_session"}, "https://api.openai.com")
        == "https://copilot-api.acme.ghe.com"
    )


@pytest.mark.parametrize("path", AUTO_RESOLUTION_PATHS + ANCILLARY_PATHS)
def test_passthrough_sends_copilot_authenticated_calls_to_copilot(path: str) -> None:
    """On a shared proxy with the stock OpenAI target, Auto resolution must reach GitHub."""
    proxy = _proxy("https://api.openai.com")
    headers = {"Authorization": "Bearer tid_session", **COPILOT_ROUTING_HEADERS}
    assert select_passthrough_base_url(proxy, headers, path) == COPILOT_API


def test_passthrough_keeps_a_pinned_copilot_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wrap copilot` pins the resolved host; the redirect must not override it."""
    pinned = "https://api.business.githubcopilot.com"
    proxy = _proxy(pinned)
    headers = {"Authorization": "Bearer tid_session"}
    assert select_passthrough_base_url(proxy, headers, "/models/session") == pinned


def test_passthrough_leaves_non_copilot_clients_on_the_openai_target() -> None:
    proxy = _proxy("https://api.openai.com")
    assert select_passthrough_base_url(proxy, {"Authorization": "Bearer sk-x"}, "/models") == (
        "https://api.openai.com"
    )


# --------------------------------------------------------------------------- #
# 4. The dedicated handlers honour the redirect, and gateway secrets stay home
# --------------------------------------------------------------------------- #
def _openai_handler(openai_target: str):
    from headroom.proxy.handlers.openai import OpenAIHandlerMixin

    return type("Handler", (OpenAIHandlerMixin,), {"OPENAI_API_URL": openai_target})()


def _request_with(headers: dict[str, str]):
    return type("Request", (), {"headers": headers})()


def test_openai_handler_sends_a_copilot_credential_to_copilot() -> None:
    handler = _openai_handler("https://api.openai.com")
    upstream = handler._resolve_openai_upstream(
        _request_with({"authorization": "Bearer tid_session", "copilot-session-token": "s"})
    )
    assert upstream == COPILOT_API


def test_openai_handler_keeps_a_stock_key_on_the_operator_target() -> None:
    handler = _openai_handler("https://api.openai.com")
    upstream = handler._resolve_openai_upstream(_request_with({"authorization": "Bearer sk-abc"}))
    assert upstream == "https://api.openai.com"


def test_openai_handler_keeps_a_copilot_credential_on_a_pinned_gateway() -> None:
    """An operator who pointed the proxy at a gateway chose where tokens go."""
    handler = _openai_handler("https://litellm.corp.example/v1")
    upstream = handler._resolve_openai_upstream(
        _request_with({"authorization": "Bearer tid_session"})
    )
    assert upstream == "https://litellm.corp.example/v1"


def test_openai_handler_lets_an_explicit_base_url_header_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.example")
    handler = _openai_handler("https://api.openai.com")
    upstream = handler._resolve_openai_upstream(
        _request_with(
            {
                "authorization": "Bearer tid_session",
                "x-headroom-base-url": "https://gateway.example",
            }
        )
    )
    assert upstream == "https://gateway.example"


def test_gateway_extra_headers_are_withheld_on_a_copilot_redirect() -> None:
    from headroom.proxy.handlers.openai import _extra_headers_unless_redirected

    secrets = {"X-Gateway-Key": "operator-secret"}
    assert _extra_headers_unless_redirected(secrets, redirected_to=COPILOT_API) is None
    assert _extra_headers_unless_redirected(secrets, redirected_to=None) is secrets
    assert _extra_headers_unless_redirected(None, redirected_to=None) is None


# --------------------------------------------------------------------------- #
# 5. Route level: /v1/messages with a Copilot bearer reaches the Copilot base
# --------------------------------------------------------------------------- #
def _messages_app(anthropic_target: str):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from headroom.providers.proxy_routes import register_provider_routes

    class Runtime:
        @staticmethod
        def api_target(provider: str) -> str:
            return f"https://{provider}.example.test"

        @staticmethod
        def model_metadata_provider(headers) -> str:  # type: ignore[no-untyped-def]
            # Mirrors the runtime: only an Anthropic-style key marks the caller
            # as Anthropic; a bare bearer (Copilot's included) reads as OpenAI.
            return "anthropic" if headers.get("x-api-key") else "openai"

    class Proxy:
        ANTHROPIC_API_URL = anthropic_target
        OPENAI_API_URL = "https://api.openai.com"
        GEMINI_API_URL = "https://gemini.example.test"
        CLOUDCODE_API_URL = "https://cloudcode.example.test"
        VERTEX_API_URL = "https://vertex.example.test"

        def __init__(self) -> None:
            self.config = SimpleNamespace(bedrock_api_url=None)
            self.provider_runtime = Runtime()
            self.upstreams: list[str | None] = []
            self.redirect_flags: list[bool] = []
            self.passthrough_targets: list[tuple[str, str]] = []
            self.http_client = object()

        async def handle_anthropic_messages(  # type: ignore[no-untyped-def]
            self, request, upstream_base_url=None, copilot_redirect=False
        ):
            self.upstreams.append(upstream_base_url)
            self.redirect_flags.append(copilot_redirect)
            return JSONResponse({"ok": True})

        async def handle_passthrough(self, request, base_url, endpoint_name="", provider=""):  # type: ignore[no-untyped-def]
            self.passthrough_targets.append((request.url.path, base_url))
            return JSONResponse({"ok": True})

    app = FastAPI()
    proxy = Proxy()
    register_provider_routes(app, proxy)
    return TestClient(app), proxy


def test_messages_route_redirects_a_copilot_bearer_to_copilot() -> None:
    client, proxy = _messages_app("https://api.anthropic.com")
    response = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-5", "messages": []},
        headers={"Authorization": "Bearer tid_session", **COPILOT_ROUTING_HEADERS},
    )
    assert response.status_code == 200
    assert proxy.upstreams == [COPILOT_API]
    assert proxy.redirect_flags == [True], "the handler must know the proxy chose this host"


def test_messages_route_leaves_anthropic_keys_on_the_default_path() -> None:
    client, proxy = _messages_app("https://api.anthropic.com")
    client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-5", "messages": []},
        headers={"x-api-key": "sk-ant-abc"},
    )
    client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-5", "messages": []},
        headers={"Authorization": "Bearer sk-ant-oat-abc"},
    )
    assert proxy.upstreams == [None, None]
    assert proxy.redirect_flags == [False, False]


def test_messages_route_keeps_a_copilot_bearer_on_a_pinned_target() -> None:
    client, proxy = _messages_app("https://anthropic.gateway.corp.example")
    client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-5", "messages": []},
        headers={"Authorization": "Bearer tid_session"},
    )
    assert proxy.upstreams == [None]


@pytest.mark.parametrize("path", AUTO_RESOLUTION_PATHS)
def test_auto_resolution_calls_reach_copilot_through_the_catch_all_route(path: str) -> None:
    """A bare ``@app.post`` for one of these paths that skipped the passthrough
    selector would not show up in ``OPENAI_HANDLER_ROUTES``; drive the router."""
    client, proxy = _messages_app("https://api.anthropic.com")
    response = client.post(
        path,
        json={"auto_mode": {"model_hints": ["auto"]}},
        headers={"Authorization": f"Bearer {REAL_COPILOT_API_TOKEN}"},
    )
    assert response.status_code == 200
    assert proxy.passthrough_targets == [(path, COPILOT_API)]


def test_prefixed_models_route_follows_the_same_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    from headroom.providers import proxy_routes

    seen: list[str] = []

    async def fake_metadata(proxy, request, *, endpoint, provider_api_base_url, provider_name):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        seen.append(provider_api_base_url)
        return JSONResponse({"data": []})

    monkeypatch.setattr(proxy_routes, "handle_model_metadata_endpoint", fake_metadata)
    client, _proxy = _messages_app("https://api.anthropic.com")
    client.get("/v1/models", headers={"Authorization": f"Bearer {REAL_COPILOT_API_TOKEN}"})
    client.get("/v1/models/gpt-5.5", headers={"Authorization": f"Bearer {REAL_COPILOT_API_TOKEN}"})
    client.get("/v1/models", headers={"Authorization": "Bearer sk-openai"})
    assert seen == [COPILOT_API, COPILOT_API, "https://api.openai.com"]
