"""Transport and internal exception text must never reach a client (01-F4).

``httpx``/``ssl``/OS exception text names the resolved upstream host, its IP
and the failing library. Every response path maps such failures to the fixed
vocabulary in :mod:`headroom.proxy.public_errors` and keeps the detail in the
server log under the request id.

Companion: the shared proxy token must never land in request tags (10-F4).
"""

from __future__ import annotations

import asyncio
import json
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy import public_errors  # noqa: E402
from headroom.proxy.handlers.openai import OpenAIHandlerMixin  # noqa: E402
from headroom.proxy.helpers import extract_tags  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# A string that only ever appears in exception text; every assertion below is
# "this never reaches the wire".
SECRET = "gateway-int.corp.example:443 (10.42.0.7)"


# --- Vocabulary --------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (httpx.ConnectError(SECRET), public_errors.UPSTREAM_UNREACHABLE),
        (httpx.ConnectTimeout(SECRET), public_errors.UPSTREAM_UNREACHABLE),
        (httpx.PoolTimeout(SECRET), public_errors.UPSTREAM_UNREACHABLE),
        (httpx.ReadTimeout(SECRET), public_errors.UPSTREAM_TIMEOUT),
        (httpx.WriteTimeout(SECRET), public_errors.UPSTREAM_TIMEOUT),
        (httpx.RemoteProtocolError(SECRET), public_errors.UPSTREAM_PROTOCOL_ERROR),
        (httpx.ReadError(SECRET), public_errors.UPSTREAM_PROTOCOL_ERROR),
        (httpx.DecodingError(SECRET), public_errors.UPSTREAM_PROTOCOL_ERROR),
        (ConnectionResetError(104, SECRET), public_errors.UPSTREAM_UNREACHABLE),
        (OSError(113, SECRET), public_errors.UPSTREAM_UNREACHABLE),
        (asyncio.TimeoutError(), public_errors.UPSTREAM_TIMEOUT),
        (
            ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED"),
            public_errors.UPSTREAM_TLS_ERROR,
        ),
    ],
)
def test_transport_exceptions_classify_to_fixed_codes(exc, code) -> None:
    assert public_errors.classify_exception(exc) == code
    assert SECRET not in public_errors.public_message(code, request_id="req-1")


def test_tls_failure_wrapped_in_connect_error_is_tls() -> None:
    inner = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] " + SECRET)
    outer = httpx.ConnectError(SECRET)
    outer.__cause__ = inner
    assert public_errors.classify_exception(outer) == public_errors.UPSTREAM_TLS_ERROR


def test_non_transport_exceptions_are_not_classified() -> None:
    """Provider API errors (litellm/anyllm exceptions) keep their own message."""
    assert public_errors.classify_exception(RuntimeError("model not found: gpt-x")) is None
    assert public_errors.classify_exception(ValueError("bad")) is None
    assert public_errors.classify_or_internal(ValueError("bad")) == public_errors.INTERNAL_ERROR


def test_public_message_rejects_unknown_codes() -> None:
    with pytest.raises(ValueError):
        public_errors.public_message("stack trace here")


def test_wire_shapes_carry_code_and_request_id() -> None:
    openai = public_errors.openai_error_body(
        public_errors.UPSTREAM_UNREACHABLE, request_id="req-9", error_type="connection_error"
    )
    assert openai["error"]["type"] == "connection_error"
    assert openai["error"]["code"] == "upstream_unreachable"
    assert openai["error"]["request_id"] == "req-9"
    assert "req-9" in openai["error"]["message"]

    anthropic = public_errors.anthropic_error_body(public_errors.INTERNAL_ERROR, request_id="req-9")
    assert anthropic["type"] == "error"
    assert anthropic["error"]["type"] == "api_error"
    assert anthropic["error"]["code"] == "internal_error"
    assert anthropic["request_id"] == "req-9"


# --- Handler paths -------------------------------------------------------------


def _config(**overrides) -> ProxyConfig:
    base: dict[str, object] = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
        "retry_enabled": False,
    }
    base.update(overrides)
    return ProxyConfig(**base)


class _SecretConnectTimeoutClient:
    async def request(self, **kwargs):  # noqa: ANN001, ANN201
        raise httpx.ConnectTimeout(f"connect to {SECRET} timed out")


class _PassthroughRequest:
    method = "GET"
    headers = {}
    url = SimpleNamespace(path="/some/other/path", query="")

    async def body(self) -> bytes:
        return b""


def test_passthrough_connect_error_body_has_no_upstream_detail() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    handler.http_client = _SecretConnectTimeoutClient()

    response = asyncio.run(
        handler.handle_passthrough(_PassthroughRequest(), "https://api.openai.com")
    )

    assert response.status_code == 502
    payload = json.loads(response.body)
    assert payload["error"]["type"] == "connection_error"
    assert payload["error"]["code"] == public_errors.UPSTREAM_UNREACHABLE
    assert SECRET not in response.body.decode()


def test_direct_streaming_connect_error_event_has_no_upstream_detail(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    with TestClient(create_app(_config()), raise_server_exceptions=False) as client:
        proxy = client.app.state.proxy
        proxy.http_client.send = AsyncMock(side_effect=httpx.ConnectError(f"refused by {SECRET}"))

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 502
    text = response.text
    assert SECRET not in text
    assert "event: error" in text
    assert public_errors.UPSTREAM_UNREACHABLE in text


def test_compress_endpoint_error_body_has_no_exception_detail(monkeypatch) -> None:
    app = create_app(_config(optimize=True))
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345), raise_server_exceptions=False
    ) as client:
        proxy = client.app.state.proxy
        monkeypatch.setattr(
            proxy,
            "_run_compression_in_executor",
            AsyncMock(side_effect=RuntimeError(f"compressor exploded reading {SECRET}")),
        )

        response = client.post(
            "/v1/compress",
            json={"messages": [{"role": "user", "content": "compress me"}], "model": "gpt-4"},
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "compression_error"
    assert body["error"]["code"] == public_errors.INTERNAL_ERROR
    assert SECRET not in response.text


@pytest.mark.asyncio
async def test_memory_tool_result_hides_transport_and_internal_text(tmp_path) -> None:
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

    handler = MemoryHandler(
        MemoryConfig(enabled=False, use_native_tool=True, native_memory_dir=str(tmp_path / "n")),
        agent_type="codex",
    )

    class _Backend:
        exc: BaseException = OSError(5, f"sqlite at {SECRET} is locked")

        async def save_memory(self, **kwargs):  # noqa: ANN003
            raise self.exc

    backend = _Backend()
    handler._backend = backend

    transport = json.loads(
        await handler._execute_memory_tool("memory_save", {"content": "x"}, "u1")
    )
    assert transport["status"] == "error"
    assert SECRET not in json.dumps(transport)
    assert transport["error"] == public_errors.UPSTREAM_UNREACHABLE

    backend.exc = RuntimeError(f"save failed at {LEAKY}")
    internal = json.loads(await handler._execute_memory_tool("memory_save", {"content": "x"}, "u1"))
    assert internal["error"] == public_errors.INTERNAL_ERROR
    assert LEAKY not in json.dumps(internal)


# --- Unclassified exceptions fail closed ---------------------------------------

# Exception text that names a local path, an internal host and a token.
LEAKY = r"sqlite at C:\service\tenant-a.db via db-int.corp.example failed with sk-live-secret"

UNCLASSIFIED = [
    RuntimeError(LEAKY),
    ValueError(LEAKY),
    KeyError(LEAKY),
    json.JSONDecodeError(LEAKY, "{}", 0),
]


@pytest.mark.parametrize("exc", UNCLASSIFIED, ids=lambda e: type(e).__name__)
def test_client_message_hides_unclassified_exception_text(exc) -> None:
    message = public_errors.client_message(exc, LEAKY)
    assert message == public_errors.public_message(public_errors.INTERNAL_ERROR)
    assert "sk-live-secret" not in message


@pytest.mark.parametrize("exc", UNCLASSIFIED, ids=lambda e: type(e).__name__)
def test_tool_result_error_hides_unclassified_exception_text(exc) -> None:
    result = public_errors.tool_result_error(exc)
    assert result["error"] == public_errors.INTERNAL_ERROR
    assert "sk-live-secret" not in json.dumps(result)


def test_client_message_forwards_provider_http_error_response() -> None:
    import openai

    response = httpx.Response(401, request=httpx.Request("POST", "https://api.example/v1"))
    exc = openai.AuthenticationError("invalid x-api-key", response=response, body=None)
    assert public_errors.client_message(exc, "invalid x-api-key") == "invalid x-api-key"


@pytest.mark.asyncio
async def test_litellm_backend_hides_unclassified_exception_text(monkeypatch) -> None:
    from headroom.backends import litellm as litellm_backend

    monkeypatch.setattr(litellm_backend, "acompletion", AsyncMock(side_effect=RuntimeError(LEAKY)))
    backend = litellm_backend.LiteLLMBackend(provider="openai")
    result = await backend.send_openai_message(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}, {}
    )
    assert "sk-live-secret" not in json.dumps(result.body)
    assert result.body["error"]["message"] == public_errors.public_message(
        public_errors.INTERNAL_ERROR
    )


# --- Tags (10-F4) -------------------------------------------------------------


def test_extract_tags_never_carries_credentials() -> None:
    tags = extract_tags(
        {
            "X-Headroom-Proxy-Token": "hlk_secret_token_value",
            "x-headroom-project": "alpha",
            "x-headroom-session-id": "s-1",
            "x-headroom-license-token": "lic",
            "x-headroom-api-key": "k",
            "X-Headroom-Key": "hr_cloud_key",
            "authorization": "Bearer sk-live",
        }
    )
    assert tags == {"project": "alpha", "session-id": "s-1"}


def test_public_tags_drop_credential_like_keys_defensively() -> None:
    from headroom.proxy.savings_attribution import public_tags

    assert public_tags({"proxy-token": "x", "project": "alpha", "some_secret": "y"}) == {
        "project": "alpha"
    }
