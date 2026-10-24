"""HTTP OpenAI requests inherit the process API key when clients omit auth."""

from __future__ import annotations

from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.helpers import apply_openai_api_key_fallback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.proxy.upstream_trust import ALLOWED_HOSTS_ENV  # noqa: E402

UPSTREAM = "https://api.commandcode.ai/provider/v1"
BODY = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hello"}],
}
UPSTREAM_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture(autouse=True)
def _allow_python_only_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")


def _make_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        openai_api_url=UPSTREAM,
    )


def _install_retry_capture(client: TestClient) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def _retry(method: str, url: str, headers: dict[str, str], body: dict, **kwargs: Any):
        captured.update(method=method, url=url, headers=dict(headers), body=body, kwargs=kwargs)
        return httpx.Response(
            200,
            json=UPSTREAM_RESPONSE,
            request=httpx.Request(method, url),
        )

    client.app.state.proxy._retry_request = _retry
    return captured


def test_buffered_openai_request_uses_process_api_key_when_client_is_keyless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    app = create_app(_make_config())

    with TestClient(app) as client:
        captured = _install_retry_capture(client)
        response = client.post("/v1/chat/completions", json={**BODY, "stream": False})

    assert response.status_code == 200, response.text
    assert captured["headers"]["Authorization"] == "Bearer env-key"


def test_streaming_openai_request_uses_process_api_key_when_client_is_keyless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    app = create_app(_make_config())
    captured: dict[str, Any] = {}

    async def _upstream(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
            request=request,
        )

    with TestClient(app) as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(_upstream)
        )
        response = client.post("/v1/chat/completions", json={**BODY, "stream": True})

    assert response.status_code == 200
    lower_headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert lower_headers["authorization"] == "Bearer env-key"


def test_buffered_openai_responses_uses_process_api_key_when_client_is_keyless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    app = create_app(_make_config())

    with TestClient(app) as client:
        captured = _install_retry_capture(client)
        response = client.post(
            "/v1/responses",
            json={"model": "gpt-4o-mini", "input": "hello", "stream": False},
        )

    assert response.status_code == 200, response.text
    lower_headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert lower_headers["authorization"] == "Bearer env-key"


def test_buffered_grok_responses_does_not_receive_openai_process_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        openai_api_url="https://api.openai.com/v1",
    )
    app = create_app(config)

    with TestClient(app) as client:
        captured = _install_retry_capture(client)
        response = client.post(
            "/v1/responses",
            headers={"x-xai-token-auth": "xai-grok-cli"},
            json={"model": "gpt-4o-mini", "input": "hello", "stream": False},
        )

    assert response.status_code == 200, response.text
    assert captured["url"].startswith("https://api.x.ai/")
    assert "authorization" not in {key.lower() for key in captured["headers"]}


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    [("Authorization", "Bearer client-key"), ("api-key", "client-key")],
)
def test_openai_http_request_preserves_client_credentials(
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
    header_value: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    app = create_app(_make_config())

    with TestClient(app) as client:
        captured = _install_retry_capture(client)
        response = client.post(
            "/v1/chat/completions",
            json={**BODY, "stream": False},
            headers={header_name: header_value},
        )

    assert response.status_code == 200, response.text
    lower_headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert lower_headers[header_name.lower()] == header_value
    if header_name.lower() == "api-key":
        assert "authorization" not in lower_headers
    else:
        assert lower_headers["authorization"] == header_value


def test_openai_api_key_fallback_rejects_untrusted_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    headers = apply_openai_api_key_fallback(
        {"content-type": "application/json"},
        upstream_url="https://attacker.example/v1",
        config=_make_config(),
    )

    assert "authorization" not in {key.lower() for key in headers}


def test_openai_api_key_fallback_is_scoped_to_openai_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "api.xai.example")
    config = ProxyConfig(
        openai_api_url=UPSTREAM,
        anthropic_api_url="https://api.xai.example/v1",
    )

    headers = apply_openai_api_key_fallback(
        {"content-type": "application/json"},
        upstream_url="https://api.xai.example/v1",
        config=config,
    )

    assert "authorization" not in {key.lower() for key in headers}


def test_openai_api_key_fallback_refuses_plaintext_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    http_upstream = "http://api.commandcode.ai/provider/v1"
    config = ProxyConfig(openai_api_url=http_upstream)

    headers = apply_openai_api_key_fallback(
        {"content-type": "application/json"},
        upstream_url=http_upstream,
        config=config,
    )

    assert "authorization" not in {key.lower() for key in headers}
