"""Credential-scoped rate limiting (#3364) under the shared identity rule (01-F3).

#3364 made distinct OpenAI credentials use distinct buckets. VAPT finding 01-F3
showed the other side of that: a caller the proxy cannot authenticate could
rotate the credential header to mint a fresh bucket per request. Both hold now:

* an **authenticated** caller (proxy token) or a direct loopback caller gets
  one bucket per provider credential — #3364's behaviour;
* an **unauthenticated remote** caller is charged per peer, whatever credential
  header it sends — including one relayed by a trusted gateway, which is
  charged per forwarded client address.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

TOKEN = "hrpt_rate_limit_test_token_0123456789"

CHAT = (
    "/v1/chat/completions",
    {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "stream": False},
    {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    },
)
RESPONSES = (
    "/v1/responses",
    {"model": "gpt-4o-mini", "input": "hello", "stream": False},
    {
        "id": "resp-test",
        "object": "response",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    },
)


def _config(**kwargs) -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=1,
        rate_limit_tokens_per_minute=100_000,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        retry_enabled=False,
        **kwargs,
    )


def _two_requests(client, path, body, first_headers, second_headers):
    proxy = client.app.state.proxy
    return (
        proxy,
        client.post(path, headers=first_headers, json=body),
        client.post(path, headers=second_headers, json=body),
    )


@pytest.mark.parametrize("endpoint", [CHAT, RESPONSES], ids=["chat", "responses"])
def test_authenticated_callers_get_one_bucket_per_credential(monkeypatch, endpoint) -> None:
    """#3364: distinct provider credentials from an authenticated caller do not share."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    path, body, upstream_body = endpoint
    with TestClient(create_app(_config(proxy_token=TOKEN)), client=("203.0.113.9", 1)) as client:
        client.app.state.proxy._retry_request = AsyncMock(
            side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
        )
        proxy, first, second = _two_requests(
            client,
            path,
            body,
            {"x-headroom-proxy-token": TOKEN, "api-key": "gateway-key-A"},
            {"x-headroom-proxy-token": TOKEN, "api-key": "gateway-key-B"},
        )
    assert first.status_code == 200
    assert second.status_code == 200
    assert proxy._retry_request.await_count == 2


@pytest.mark.parametrize("endpoint", [CHAT, RESPONSES], ids=["chat", "responses"])
def test_loopback_callers_get_one_bucket_per_credential(monkeypatch, endpoint) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    path, body, upstream_body = endpoint
    with TestClient(create_app(_config()), client=("127.0.0.1", 1)) as client:
        client.app.state.proxy._retry_request = AsyncMock(
            side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
        )
        proxy, first, second = _two_requests(
            client, path, body, {"api-key": "key-A"}, {"api-key": "key-B"}
        )
    assert (first.status_code, second.status_code) == (200, 200)
    assert proxy._retry_request.await_count == 2


@pytest.mark.parametrize("endpoint", [CHAT, RESPONSES], ids=["chat", "responses"])
@pytest.mark.parametrize(
    "rotating",
    [
        ({"api-key": "rotated-1"}, {"api-key": "rotated-2"}),
        ({"authorization": "Bearer rotated-1"}, {"authorization": "Bearer rotated-2"}),
        ({"api-key": "rotated-1"}, {}),
    ],
    ids=["api-key", "authorization", "credential-then-none"],
)
def test_unauthenticated_remote_caller_cannot_rotate_credentials_into_new_buckets(
    monkeypatch, endpoint, rotating
) -> None:
    """01-F3: with no proxy token, a remote caller is charged per peer."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    path, body, upstream_body = endpoint
    with TestClient(create_app(_config()), client=("203.0.113.9", 1)) as client:
        client.app.state.proxy._retry_request = AsyncMock(
            side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
        )
        proxy, first, second = _two_requests(client, path, body, *rotating)
    assert first.status_code == 200
    assert second.status_code == 429
    assert proxy._retry_request.await_count == 1


def test_unauthenticated_remote_callers_on_different_peers_do_not_share(monkeypatch) -> None:
    """The fix is per peer, not one global bucket for everyone unauthenticated."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    path, body, upstream_body = CHAT
    app = create_app(_config())
    statuses = []
    for peer in ("203.0.113.9", "198.51.100.4"):
        with TestClient(app, client=(peer, 1)) as client:
            client.app.state.proxy._retry_request = AsyncMock(
                side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
            )
            statuses.append(client.post(path, json=body).status_code)
    assert statuses == [200, 200]


@pytest.mark.parametrize("endpoint", [CHAT, RESPONSES], ids=["chat", "responses"])
def test_unauthenticated_caller_behind_trusted_gateway_cannot_rotate_credentials(
    monkeypatch, endpoint
) -> None:
    """A trusted gateway vouches for the client address, not for the caller."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "10.0.0.0/8")
    path, body, upstream_body = endpoint
    forwarded = {"x-forwarded-for": "203.0.113.9"}
    with TestClient(create_app(_config()), client=("10.0.0.2", 1)) as client:
        client.app.state.proxy._retry_request = AsyncMock(
            side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
        )
        proxy, first, second = _two_requests(
            client,
            path,
            body,
            {**forwarded, "api-key": "rotated-A"},
            {**forwarded, "api-key": "rotated-B"},
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert proxy._retry_request.await_count == 1


def test_trusted_gateway_keys_unauthenticated_callers_by_forwarded_address(monkeypatch) -> None:
    """Distinct forwarded clients behind one gateway keep distinct buckets."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "10.0.0.0/8")
    path, body, upstream_body = CHAT
    with TestClient(create_app(_config()), client=("10.0.0.2", 1)) as client:
        client.app.state.proxy._retry_request = AsyncMock(
            side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
        )
        statuses = [
            client.post(path, headers={"x-forwarded-for": ip}, json=body).status_code
            for ip in ("203.0.113.9", "198.51.100.4")
        ]
    assert statuses == [200, 200]


def test_authenticated_caller_behind_trusted_gateway_gets_one_bucket_per_credential(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "10.0.0.0/8")
    path, body, upstream_body = CHAT
    headers = {"x-forwarded-for": "203.0.113.9", "x-headroom-proxy-token": TOKEN}
    with TestClient(create_app(_config(proxy_token=TOKEN)), client=("10.0.0.2", 1)) as client:
        client.app.state.proxy._retry_request = AsyncMock(
            side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
        )
        _, first, second = _two_requests(
            client, path, body, {**headers, "api-key": "key-A"}, {**headers, "api-key": "key-B"}
        )
    assert (first.status_code, second.status_code) == (200, 200)


@pytest.mark.parametrize("endpoint", [CHAT, RESPONSES], ids=["chat", "responses"])
def test_loopback_gateway_config_direct_and_forwarded_callers(monkeypatch, endpoint) -> None:
    """One loopback gateway CIDR: direct callers stay per credential, relayed ones per client.

    Direct loopback callers (no ``X-Forwarded-For``) with distinct keys keep
    distinct buckets. A caller relayed through the same loopback gateway with a
    fixed forwarded address cannot rotate keys into a fresh bucket.
    """
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "127.0.0.0/8")
    path, body, upstream_body = endpoint
    forwarded = {"x-forwarded-for": "203.0.113.9"}
    with TestClient(create_app(_config()), client=("127.0.0.1", 1)) as client:
        client.app.state.proxy._retry_request = AsyncMock(
            side_effect=lambda *a, **k: httpx.Response(200, json=upstream_body)
        )
        statuses = [
            client.post(path, headers=headers, json=body).status_code
            for headers in (
                {"api-key": "direct-A"},
                {"api-key": "direct-B"},
                {**forwarded, "api-key": "rotated-A"},
                {**forwarded, "api-key": "rotated-B"},
            )
        ]
    assert statuses == [200, 200, 200, 429]
