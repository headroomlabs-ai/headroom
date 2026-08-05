import base64
import json

import httpx
import pytest
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def _jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


def test_codex_responses_aliases_delegate_to_openai_handler(monkeypatch):
    async def fake_handle(self, request):  # type: ignore[no-untyped-def]
        return JSONResponse({"ok": True, "path": request.url.path})

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses", fake_handle)

    with TestClient(create_app(ProxyConfig())) as client:
        for path in (
            "/v1/codex/responses",
            "/backend-api/responses",
            "/backend-api/codex/responses",
        ):
            response = client.post(path, json={"model": "gpt-5.3-codex"})
            assert response.status_code == 200
            assert response.json() == {"ok": True, "path": path}


def test_codex_responses_websocket_aliases_delegate_to_openai_handler(monkeypatch):
    seen_paths: list[str] = []

    async def fake_handle_ws(self, websocket: WebSocket):  # type: ignore[no-untyped-def]
        seen_paths.append(websocket.url.path)
        await websocket.accept()
        await websocket.send_json({"ok": True, "path": websocket.url.path})
        await websocket.close()

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses_ws", fake_handle_ws)

    with TestClient(create_app(ProxyConfig())) as client:
        for path in (
            "/v1/codex/responses",
            "/backend-api/responses",
            "/backend-api/codex/responses",
        ):
            with client.websocket_connect(path) as websocket:
                assert websocket.receive_json() == {"ok": True, "path": path}

    assert seen_paths == [
        "/v1/codex/responses",
        "/backend-api/responses",
        "/backend-api/codex/responses",
    ]


def test_codex_responses_subpath_aliases_delegate_to_passthrough():
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def request(self, method, url, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((method, url))
            return httpx.Response(200, json={"method": method, "url": url})

        async def aclose(self) -> None:
            return None

    with TestClient(create_app(ProxyConfig())) as client:
        fake_http_client = FakeAsyncClient()
        client.app.state.proxy.http_client = fake_http_client
        client.app.state.proxy.OPENAI_API_URL = "https://api.openai.test"

        pi_response = client.post(
            "/v1/codex/responses/compact?trace=0",
            json={"model": "gpt-5.3-codex"},
        )
        api_key_response = client.post(
            "/backend-api/responses/compact?trace=1",
            json={"model": "gpt-5.3-codex"},
        )
        chatgpt_response = client.post(
            "/backend-api/codex/responses/compact?trace=2",
            headers={"chatgpt-account-id": "acct_123"},
            json={"model": "gpt-5.3-codex"},
        )

    assert pi_response.status_code == 200
    assert api_key_response.status_code == 200
    assert chatgpt_response.status_code == 200
    assert fake_http_client.calls == [
        ("POST", "https://api.openai.test/v1/responses/compact?trace=0"),
        ("POST", "https://api.openai.test/v1/responses/compact?trace=1"),
        ("POST", "https://chatgpt.com/backend-api/codex/responses/compact?trace=2"),
    ]


@pytest.mark.parametrize(
    ("path", "expected_url"),
    [
        (
            "/v1/codex/responses/compact?trace=jwt",
            "https://chatgpt.com/backend-api/codex/responses/compact?trace=jwt",
        ),
        (
            "/v1/responses/compact?trace=jwt-old",
            "https://chatgpt.com/backend-api/codex/responses/compact?trace=jwt-old",
        ),
    ],
)
def test_codex_responses_subpath_passthrough_derives_chatgpt_routing_from_jwt(path, expected_url):
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, str]]] = []

        async def request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((method, url, dict(kwargs.get("headers", {}))))
            return httpx.Response(200, json={"method": method, "url": url})

        async def aclose(self) -> None:
            return None

    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )

    with TestClient(create_app(ProxyConfig())) as client:
        fake_http_client = FakeAsyncClient()
        client.app.state.proxy.http_client = fake_http_client
        client.app.state.proxy.OPENAI_API_URL = "https://api.openai.test"

        response = client.post(
            path,
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "gpt-5.4"},
        )

    assert response.status_code == 200
    assert len(fake_http_client.calls) == 1

    method, url, headers = fake_http_client.calls[0]
    assert method == "POST"
    assert url == expected_url
    assert headers["authorization"] == f"Bearer {token}"
    assert headers["ChatGPT-Account-ID"] == "acct-from-jwt"


def test_codex_model_metadata_fetches_codex_registry_for_chatgpt_auth(monkeypatch):
    """Issue #478: under Codex ChatGPT-subscription OAuth, the proxy
    must NOT forward `/v1/models[/{id}]` to chatgpt.com/backend-api —
    that endpoint returns 403 to OAuth tokens. Instead, Headroom should
    fetch the Codex-specific model registry and synthesize an
    OpenAI-compatible payload from its slugs.
    """

    class FakeAsyncClient:
        def __init__(self):
            self.calls: list[tuple[str, str, dict[str, str]]] = []

        async def get(self, url, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(("GET", url, dict(kwargs.get("headers", {}))))
            return httpx.Response(
                200,
                json={"models": [{"slug": "gpt-5.5"}, {"slug": "gpt-5.3-codex-spark"}]},
            )

        async def aclose(self):
            return None

    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )

    with TestClient(create_app(ProxyConfig())) as client:
        fake_http_client = FakeAsyncClient()
        client.app.state.proxy.http_client = fake_http_client
        client.app.state.proxy.OPENAI_API_URL = "https://api.openai.test"

        list_response = client.get(
            "/v1/models?client_version=0.130.0",
            headers={"Authorization": f"Bearer {token}"},
        )
        known_response = client.get(
            "/v1/models/gpt-5.5",
            headers={"Authorization": f"Bearer {token}"},
        )
        unknown_response = client.get(
            "/v1/models/gpt-99-future?client_version=0.130.0",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert len(fake_http_client.calls) == 3
    for method, url, headers in fake_http_client.calls:
        assert method == "GET"
        assert url == "https://chatgpt.com/backend-api/codex/models?client_version=0.130.0"
        assert headers["authorization"] == f"Bearer {token}"
        assert headers["chatgpt-account-id"] == "acct-from-jwt"
        assert headers["accept"] == "application/json"
        assert "ChatGPT-Account-ID" not in headers
        assert "Accept" not in headers

    # List endpoint returns a non-empty OpenAI-compatible list.
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["object"] == "list"
    assert isinstance(list_payload["data"], list)
    assert len(list_payload["data"]) > 0
    assert {entry["id"] for entry in list_payload["data"]} == {
        "gpt-5.5",
        "gpt-5.3-codex-spark",
    }
    assert {entry["slug"] for entry in list_payload["models"]} == {
        "gpt-5.5",
        "gpt-5.3-codex-spark",
    }
    for entry in list_payload["models"]:
        assert entry["display_name"]
        assert entry["default_reasoning_level"] == "medium"
        assert entry["supports_parallel_tool_calls"] is True

    # Single-model GET returns a model object when known.
    assert known_response.status_code == 200
    known_payload = known_response.json()
    assert known_payload == {
        "id": "gpt-5.5",
        "object": "model",
        "created": 0,
        "owned_by": "openai",
    }

    # Unknown model variants 404 against the dynamic registry.
    assert unknown_response.status_code == 404


_CODEX_DESKTOP_UA = (
    "Codex Desktop/0.140.0-alpha.2 (Mac OS 15.7.7; arm64) unknown (Codex Desktop; 26.609.71450)"
)


def test_responses_middleware_stamps_x_client_codex_for_unidentified_caller(monkeypatch):
    # Codex Desktop's User-Agent isn't a known codex UA, so the HTTP middleware
    # must stamp X-Client: codex on /v1/responses before the handler classifies
    # the caller — otherwise a compression timeout is refused with a 413 that
    # Codex treats as a hard connection failure.
    seen: dict[str, str | None] = {}

    async def fake_handle(self, request):  # type: ignore[no-untyped-def]
        seen["x-client"] = request.headers.get("x-client")
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses", fake_handle)

    with TestClient(create_app(ProxyConfig())) as client:
        response = client.post(
            "/v1/responses",
            headers={"user-agent": _CODEX_DESKTOP_UA},
            json={"model": "gpt-5.3-codex"},
        )

    assert response.status_code == 200
    assert seen["x-client"] == "codex"


def test_responses_middleware_preserves_explicit_x_client(monkeypatch):
    # A caller that already self-identifies is left untouched by the stamp.
    seen: dict[str, str | None] = {}

    async def fake_handle(self, request):  # type: ignore[no-untyped-def]
        seen["x-client"] = request.headers.get("x-client")
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses", fake_handle)

    with TestClient(create_app(ProxyConfig())) as client:
        response = client.post(
            "/v1/responses",
            headers={"x-client": "aider", "user-agent": _CODEX_DESKTOP_UA},
            json={"model": "gpt-5.3-codex"},
        )

    assert response.status_code == 200
    assert seen["x-client"] == "aider"
