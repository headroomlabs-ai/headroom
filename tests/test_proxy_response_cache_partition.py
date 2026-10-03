"""01-F15: the response cache must never answer one caller from another's entry.

On a shared proxy (several principals behind one Headroom), two callers sending
the identical request under different provider credentials used to share one
semantic-cache entry: caller B was served the completion generated for caller A,
under A's key. The partition is now a keyed HMAC of the caller's credential
headers (and authenticated principal, when an identity resolver is installed),
threaded through every cache lookup and store.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy import identity
from headroom.proxy.semantic_cache import SemanticCache
from headroom.proxy.semantic_cache_key_policy import (
    ANONYMOUS_PARTITION,
    compute_cache_partition,
    compute_request_cache_partition,
)
from headroom.proxy.server import ProxyConfig, create_app

# ---------------------------------------------------------------------------
# Partition function
# ---------------------------------------------------------------------------


def test_partition_differs_per_credential_and_is_stable_per_credential() -> None:
    a = compute_cache_partition({"x-api-key": "sk-ant-A"})
    b = compute_cache_partition({"x-api-key": "sk-ant-B"})
    assert a != b
    assert a == compute_cache_partition({"X-Api-Key": "sk-ant-A"})


@pytest.mark.parametrize(
    "header",
    [
        "authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "chatgpt-account-id",
        "openai-organization",
        "openai-project",
    ],
)
def test_every_credential_or_account_header_partitions(header: str) -> None:
    assert compute_cache_partition({header: "one"}) != compute_cache_partition({header: "two"})


def test_non_credential_headers_do_not_fragment_the_cache() -> None:
    base = compute_cache_partition({"x-api-key": "k"})
    noisy = {
        "x-api-key": "k",
        "idempotency-key": "per-request-nonce",
        "x-headroom-proxy-token": "proxy-secret",
        "user-agent": "sdk/1.0",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    assert compute_cache_partition(noisy) == base


def test_no_credential_and_no_principal_is_the_shared_operator_partition() -> None:
    assert compute_cache_partition({"user-agent": "x"}) == ANONYMOUS_PARTITION


def test_partition_is_keyed_not_a_plain_hash_of_the_credential() -> None:
    part = compute_cache_partition({"x-api-key": "sk-ant-A"})
    for plain in (
        hashlib.sha256(b"sk-ant-A").hexdigest()[:32],
        hashlib.sha256(b"x-api-key:sk-ant-A").hexdigest()[:32],
    ):
        assert plain not in part


def test_authenticated_principal_partitions_callers_sharing_a_credential() -> None:
    same_key = {"authorization": "Bearer operator-key"}
    assert compute_cache_partition(same_key, principal="alice") != compute_cache_partition(
        same_key, principal="bob"
    )


def test_semantic_cache_refuses_calls_without_a_partition() -> None:
    cache = SemanticCache()
    with pytest.raises(TypeError):
        cache._compute_key([{"role": "user", "content": "x"}], "m")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_semantic_cache_isolates_partitions() -> None:
    cache = SemanticCache()
    msgs = [{"role": "user", "content": "Say hi."}]
    await cache.set(msgs, "m", b"A-body", {}, partition="p_a")
    assert await cache.get(msgs, "m", partition="p_b") is None
    hit = await cache.get(msgs, "m", partition="p_a")
    assert hit is not None and hit.response_body == b"A-body"


# ---------------------------------------------------------------------------
# End to end through the real handlers
# ---------------------------------------------------------------------------


def _client(*, cache_enabled: bool = True) -> TestClient:
    return TestClient(
        create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=cache_enabled,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                log_requests=False,
                ccr_inject_tool=False,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                image_optimize=False,
            )
        )
    )


def _anthropic_reply(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )


def _openai_reply(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        },
    )


ANTHROPIC_BODY = {
    "model": "claude-haiku-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "What is my account's secret plan?"}],
    "stream": False,
}
OPENAI_BODY = {
    "model": "gpt-4o-mini",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "What is my account's secret plan?"}],
    "stream": False,
}


def _drive(path: str, body: dict, reply, header: str, fmt: str):  # noqa: ANN001
    """Send A, B (different credential), A again; return bodies and upstream owners."""
    seen: list[str] = []
    out: list[str] = []
    with _client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            lowered = {k.lower(): v for k, v in headers.items()}
            owner = "A" if "key-A" in lowered.get(header, "") else "B"
            seen.append(owner)
            return reply(f"private answer for {owner}")

        proxy._retry_request = _fake_retry
        extra = {"anthropic-version": "2023-06-01"} if path == "/v1/messages" else {}
        for who in ("A", "B", "A"):
            r = client.post(path, headers={header: fmt.format(who), **extra}, json=body)
            assert r.status_code == 200, r.text
            out.append(r.text)
    return out, seen


@pytest.mark.parametrize(
    "path,body,reply,header,fmt",
    [
        ("/v1/messages", ANTHROPIC_BODY, _anthropic_reply, "x-api-key", "key-{}"),
        ("/v1/chat/completions", OPENAI_BODY, _openai_reply, "authorization", "Bearer key-{}"),
    ],
    ids=["anthropic", "openai"],
)
def test_second_caller_is_never_served_first_callers_cached_response(
    path, body, reply, header, fmt
) -> None:
    (a, b, a_again), upstream = _drive(path, body, reply, header, fmt)
    # B must reach the upstream under its own key and never see A's answer.
    assert "private answer for A" not in b
    assert "private answer for B" in b
    # A's repeat is still a cache hit: single-credential hit rate is unchanged.
    assert "private answer for A" in a_again
    assert upstream == ["A", "B"]


def test_installed_identity_resolver_partitions_callers_on_one_operator_key() -> None:
    identity.set_identity_resolver(
        lambda request, *, default: request.headers.get("x-test-user", "")
    )
    try:
        with _client() as client:
            proxy = client.app.state.proxy
            calls: list[str] = []

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                calls.append("x")
                return _anthropic_reply(f"answer {len(calls)}")

            proxy._retry_request = _fake_retry
            common = {"x-api-key": "shared-operator-key", "anthropic-version": "2023-06-01"}
            r1 = client.post(
                "/v1/messages", headers={**common, "x-test-user": "alice"}, json=ANTHROPIC_BODY
            )
            r2 = client.post(
                "/v1/messages", headers={**common, "x-test-user": "bob"}, json=ANTHROPIC_BODY
            )
            assert "answer 1" in r1.text
            assert "answer 2" in r2.text
            assert len(calls) == 2
    finally:
        identity.set_identity_resolver(None)


def _failing_resolver(request, *, default):  # noqa: ANN001
    raise RuntimeError("identity backend unavailable")


def _empty_resolver(request, *, default):  # noqa: ANN001
    # Installed, but cannot establish a principal for this caller.
    return ""


UNRESOLVED_RESOLVERS = pytest.mark.parametrize(
    "resolver", [_failing_resolver, _empty_resolver], ids=["raises", "empty"]
)


@UNRESOLVED_RESOLVERS
def test_failing_identity_resolver_yields_no_partition(resolver) -> None:  # noqa: ANN001
    identity.set_identity_resolver(resolver)
    try:
        request = SimpleNamespace(headers={"x-api-key": "shared-operator-key"})
        assert compute_request_cache_partition(request) is None
    finally:
        identity.set_identity_resolver(None)


def test_no_installed_resolver_keeps_the_credential_partition() -> None:
    request = SimpleNamespace(headers={"x-api-key": "shared-operator-key"})
    assert compute_request_cache_partition(request) == compute_cache_partition(request.headers)


@pytest.mark.parametrize(
    "path,body,reply,headers",
    [
        (
            "/v1/messages",
            ANTHROPIC_BODY,
            _anthropic_reply,
            {"x-api-key": "shared-operator-key", "anthropic-version": "2023-06-01"},
        ),
        (
            "/v1/chat/completions",
            OPENAI_BODY,
            _openai_reply,
            {"authorization": "Bearer shared-operator-key"},
        ),
    ],
    ids=["anthropic", "openai"],
)
@UNRESOLVED_RESOLVERS
def test_failing_identity_resolver_bypasses_the_response_cache(
    resolver,  # noqa: ANN001
    path,  # noqa: ANN001
    body,  # noqa: ANN001
    reply,  # noqa: ANN001
    headers,  # noqa: ANN001
) -> None:
    # A resolver that raises or returns no principal must not collapse tenants
    # on one operator key into the credential-only partition: the cache is
    # neither read nor written.
    identity.set_identity_resolver(resolver)
    try:
        with _client() as client:
            proxy = client.app.state.proxy
            calls: list[str] = []

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                calls.append("x")
                return reply(f"answer {len(calls)}")

            proxy._retry_request = _fake_retry
            texts = [
                client.post(path, headers={**headers, "x-test-user": who}, json=body).text
                for who in ("alice", "bob", "alice")
            ]
            assert "answer 1" in texts[0]
            assert "answer 1" not in texts[1] and "answer 2" in texts[1]
            assert "answer 3" in texts[2]
            assert len(calls) == 3
    finally:
        identity.set_identity_resolver(None)


@pytest.mark.parametrize(
    "path,body,reply,headers",
    [
        ("/v1/messages", ANTHROPIC_BODY, _anthropic_reply, {"x-api-key": "k"}),
        ("/v1/chat/completions", OPENAI_BODY, _openai_reply, {"authorization": "Bearer k"}),
    ],
    ids=["anthropic", "openai"],
)
def test_cache_disabled_requests_skip_identity_resolution(path, body, reply, headers) -> None:
    # No response cache means no partition is needed, so the (possibly
    # remote) identity resolver must not be consulted for it.
    resolved: list[str] = []

    def _counting_resolver(request, *, default):  # noqa: ANN001
        resolved.append("x")
        return "alice"

    identity.set_identity_resolver(_counting_resolver)
    try:
        with _client(cache_enabled=False) as client:
            proxy = client.app.state.proxy

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                return reply("ok")

            proxy._retry_request = _fake_retry
            extra = {"anthropic-version": "2023-06-01"} if path == "/v1/messages" else {}
            assert client.post(path, headers={**headers, **extra}, json=body).status_code == 200
        assert resolved == []
    finally:
        identity.set_identity_resolver(None)
