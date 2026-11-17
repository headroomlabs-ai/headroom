import pytest

from headroom.providers.codex.responses import (
    codex_responses_http_url,
    codex_responses_subpath_url,
    codex_responses_websocket_url,
    handle_chatgpt_codex_responses_subpath,
    has_chatgpt_account_header,
    normalize_codex_responses_headers,
    sanitize_codex_responses_response_headers,
)


def test_codex_responses_subpath_url_includes_optional_query() -> None:
    assert (
        codex_responses_subpath_url("items/resp_1", "trace=2")
        == "https://chatgpt.com/backend-api/codex/responses/items/resp_1?trace=2"
    )
    assert (
        codex_responses_subpath_url("compact")
        == "https://chatgpt.com/backend-api/codex/responses/compact"
    )


def test_codex_responses_endpoint_urls_are_provider_owned() -> None:
    assert codex_responses_http_url() == "https://chatgpt.com/backend-api/codex/responses"
    assert (
        codex_responses_http_url("stream=true")
        == "https://chatgpt.com/backend-api/codex/responses?stream=true"
    )
    assert codex_responses_websocket_url() == "wss://chatgpt.com/backend-api/codex/responses"


def test_codex_responses_headers_drop_host_and_resolve_explicit_chatgpt_auth() -> None:
    headers, is_chatgpt_auth = normalize_codex_responses_headers(
        {
            "Host": "localhost:8787",
            "authorization": "Bearer token",
            "chatgpt-account-id": "acct",
            "originator": "Codex Desktop",
        }
    )

    assert is_chatgpt_auth is True
    assert headers == {
        "authorization": "Bearer token",
        "chatgpt-account-id": "acct",
        "originator": "Codex Desktop",
    }
    assert has_chatgpt_account_header(headers) is True


def test_codex_responses_headers_return_false_for_regular_openai_auth() -> None:
    headers, is_chatgpt_auth = normalize_codex_responses_headers(
        {
            "Host": "localhost:8787",
            "authorization": "Bearer sk-proj-test",
        }
    )

    assert is_chatgpt_auth is False
    assert headers == {"authorization": "Bearer sk-proj-test"}
    assert has_chatgpt_account_header(headers) is False


def test_codex_responses_headers_drop_accept_encoding() -> None:
    """accept-encoding must be dropped so the upstream returns plaintext; httpx
    decodes the body before we read it, so a gzip upstream would corrupt framing."""
    headers, _ = normalize_codex_responses_headers(
        {
            "Host": "localhost:8787",
            "authorization": "Bearer token",
            "chatgpt-account-id": "acct",
            "Accept-Encoding": "gzip, deflate, br",
        }
    )
    assert "accept-encoding" not in {k.lower() for k in headers}


def test_codex_responses_response_headers_drop_stale_framing_case_insensitive() -> None:
    assert sanitize_codex_responses_response_headers(
        {
            "Content-Encoding": "gzip",
            "Content-Length": "9999",
            "Server": "cloudflare",
            "Content-Type": "application/json",
            "x-request-id": "kept",
        }
    ) == {
        "Content-Type": "application/json",
        "x-request-id": "kept",
    }


class _FakeResp:
    def __init__(self, content: bytes, status_code: int, headers: dict[str, str]):
        self.content = content
        self.status_code = status_code
        self.headers = headers


class _FakeClient:
    def __init__(self, resp: _FakeResp):
        self._resp = resp
        self.last_headers: dict[str, str] | None = None

    async def request(self, method, url, *, headers, content, timeout):
        self.last_headers = headers
        return self._resp


def _request(method: str, path: str, headers: dict[str, str], body: bytes):
    from starlette.requests import Request

    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": raw,
    }
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


@pytest.mark.asyncio
async def test_subpath_passthrough_strips_upstream_compression_framing() -> None:
    """The upstream body is already-decoded plaintext, so the forwarded response
    must not replay content-encoding / a compressed content-length (#3019)."""
    plaintext = b'{"id":"resp_1","status":"completed"}'
    upstream = _FakeResp(
        content=plaintext,
        status_code=200,
        headers={
            "content-encoding": "gzip",  # stale: content is already decoded
            "content-length": "17",  # stale: compressed length
            "content-type": "application/json",
            "x-request-id": "req-123",
        },
    )
    client = _FakeClient(upstream)
    request = _request(
        "GET",
        "/v1/responses/resp_1",
        {
            "host": "localhost:8787",
            "authorization": "Bearer token",
            "chatgpt-account-id": "acct",
            "accept-encoding": "gzip",
        },
        body=b"",
    )

    response = await handle_chatgpt_codex_responses_subpath(client, request, "resp_1")

    assert response is not None
    assert response.status_code == 200
    assert response.body == plaintext
    out = {k.lower() for k, _ in response.raw_headers}
    assert b"content-encoding" not in out
    # accept-encoding was stripped before the upstream call.
    assert client.last_headers is not None
    assert "accept-encoding" not in {k.lower() for k in client.last_headers}
