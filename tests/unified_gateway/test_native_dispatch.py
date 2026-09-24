from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.credentials import CredentialLease, SecretHandle
from headroom.proxy.gateway.dispatch import dispatch_native_http
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)
CLOUD_EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.cloud-identities.json"
)


@pytest.mark.parametrize(
    ("path", "model", "expected_url"),
    [
        (
            "/v1/chat/completions",
            "public-openai",
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "/v1/messages",
            "public-anthropic",
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "/v1beta/models/public-gemini:generateContent",
            None,
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "REPLACE_WITH_ENABLED_GEMINI_MODEL:generateContent",
        ),
    ],
)
def test_native_provider_routes_share_gateway_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    model: str | None,
    expected_url: str,
) -> None:
    """A legacy-handler branch must not bypass gateway credential isolation."""

    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-provider-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-provider-secret")
    captured: list[httpx.Request] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=b'{"ok":true}')

    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    aliases = {
        "openai-native": "public-openai",
        "anthropic-native": "public-anthropic",
        "gemini-native": "public-gemini",
    }
    snapshot = snapshot.model_copy(
        update={
            "routes": tuple(
                route.model_copy(
                    update={
                        "public_model": aliases[route.id],
                        "body_contract": "routed-native",
                    }
                )
                for route in snapshot.routes
            )
        }
    )
    app = create_app(ProxyConfig(gateway=snapshot))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    body = b'{"contents":[{"role":"user"}]}'
    if model is not None:
        body = ('{"model":"' + model + '","messages":[]}').encode()

    response = TestClient(app).post(
        path,
        headers={
            "host": "127.0.0.1:8787",
            "authorization": "Bearer client-secret",
            "content-type": "application/json",
        },
        content=body,
    )

    assert response.status_code == 200
    assert len(captured) == 1
    assert str(captured[0].url) == expected_url
    assert "client-secret" not in captured[0].headers.values()
    assert b"public-" not in captured[0].content


def test_unqualified_batch_route_is_rejected_before_legacy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-provider-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-provider-secret")
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))

    response = TestClient(app, raise_server_exceptions=False).request(
        "GET",
        "/v1/messages/batches/missing",
        headers={
            "host": "127.0.0.1:8787",
            "authorization": "Bearer client-secret",
        },
        json={"model": "public-anthropic"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "gateway_unsupported_capability"


@pytest.mark.asyncio
async def test_native_sse_response_releases_first_chunk_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buffering the upstream entity would deadlock before the first SSE event."""

    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    completion_released = False

    class BarrierStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"type":"response.created"}\n\n'
            assert completion_released
            yield b'data: {"type":"response.completed"}\n\n'

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=BarrierStream(),
        )

    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    body = b'{"model":"REPLACE_WITH_ENABLED_OPENAI_MODEL","input":"hello","stream":true}'
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8787),
            "app": app,
            "state": {
                "gateway_generation": app.state.gateway_runtime.capture(),
                "gateway_principal": app.state.gateway_runtime.authenticator.authenticate(
                    {"authorization": "Bearer client-secret"}
                ),
            },
        },
        receive,
    )

    response = await dispatch_native_http(request, app.state.proxy, "openai-responses")

    assert isinstance(response, StreamingResponse)
    iterator = response.body_iterator.__aiter__()
    assert await anext(iterator) == b'data: {"type":"response.created"}\n\n'
    completion_released = True
    assert await anext(iterator) == b'data: {"type":"response.completed"}\n\n'
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.parametrize(
    ("path", "expected_url", "expected_auth_prefix"),
    [
        (
            "/v1/projects/REPLACE_WITH_AUTHORIZED_PROJECT/locations/us-central1/"
            "publishers/google/models/REPLACE_WITH_ENABLED_VERTEX_MODEL:generateContent",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/"
            "REPLACE_WITH_AUTHORIZED_PROJECT/locations/us-central1/publishers/google/models/"
            "REPLACE_WITH_ENABLED_VERTEX_MODEL:generateContent",
            "Bearer vertex-token",
        ),
        (
            "/model/REPLACE_WITH_ENABLED_BEDROCK_MODEL/invoke",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
            "REPLACE_WITH_ENABLED_BEDROCK_MODEL/invoke",
            "AWS4-HMAC-SHA256 ",
        ),
    ],
)
def test_cloud_native_routes_use_gateway_identity_and_exact_target(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    expected_url: str,
    expected_auth_prefix: str,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    captured: list[httpx.Request] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=b'{"ok":true}')

    class Broker:
        async def acquire(self, route: Any, account_ref: str | None = None) -> CredentialLease:
            assert account_ref == route.credentials[0]
            secret: object = "vertex-token"
            if route.provider == "bedrock":
                secret = type(
                    "AwsCredentials",
                    (),
                    {
                        "access_key": "access-key",
                        "secret_key": "secret-key",
                        "token": "session-token",
                    },
                )()
            return CredentialLease(
                credential_id=route.credentials[0],
                provider=route.provider,
                account_ref="test-account",
                allowed_origins=(route.upstream_origin,),
                allowed_path_prefixes=(route.upstream_path_prefix,),
                expires_at=None,
                generation=1,
                secret=SecretHandle(secret),
                source_kind="aws-chain" if route.provider == "bedrock" else "gcp-adc",
                project="REPLACE_WITH_AUTHORIZED_PROJECT" if route.provider == "vertex" else None,
                region="us-east-1" if route.provider == "bedrock" else None,
            )

    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(CLOUD_EXAMPLE)))
    app.state.gateway_runtime.dependencies.broker = Broker()
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )

    response = TestClient(app).post(
        path,
        headers={
            "host": "127.0.0.1:8787",
            "authorization": "Bearer client-secret",
            "content-type": "application/json",
        },
        content=b'{ "messages" : [], "unknown" : 1.00 }',
    )

    assert response.status_code == 200
    assert len(captured) == 1
    assert str(captured[0].url) == expected_url
    assert captured[0].headers["authorization"].startswith(expected_auth_prefix)


def test_openai_responses_uses_provider_key_and_preserves_entity_bytes(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    captured: list[httpx.Request] = []
    response_bytes = b'{  "id" : "response-fixture", "output": [] }'

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-upstream": "preserved"},
            content=response_bytes,
        )

    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    request_bytes = (
        '{ "model" : "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input" : "héllo", "unknown" : 1.00 }'
    ).encode()

    response = TestClient(app).post(
        "/v1/responses?api_key=client-secret&trace=kept",
        headers={
            "host": "127.0.0.1:8787",
            "authorization": "Bearer client-secret",
            "content-type": "application/json",
        },
        content=request_bytes,
    )

    assert response.status_code == 200
    assert response.content == response_bytes
    assert "x-upstream" not in response.headers
    assert len(captured) == 1
    assert captured[0].content == request_bytes
    assert captured[0].headers["authorization"] == "Bearer provider-secret"
    assert "client-secret" not in captured[0].headers.values()
    assert str(captured[0].url) == "https://api.openai.com/v1/responses?trace=kept"
