"""Repeated history markers must not renew a compression event's turn age."""

import copy

import httpx
from fastapi.testclient import TestClient

from headroom.cache.backends import InMemoryBackend
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.proxy.server import ProxyConfig, create_app


def test_replayed_history_marker_expires_by_original_compressing_turn(tmp_path):
    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    original = '{"authentication_middleware":"STALE_CONTEXT_SENTINEL"}'
    key = store.store(
        original, "authentication middleware", query_context="authentication middleware"
    )
    config = ProxyConfig(
        optimize=False,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
        ccr_context_tracking=True,
        ccr_proactive_expansion=True,
        ccr_max_turn_distance=3,
        ccr_inject_tool=True,
        ccr_inject_system_instructions=False,
    )
    with TestClient(
        create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 1234)
    ) as client:
        proxy = client.app.state.proxy
        forwarded = []

        async def upstream(method, url, headers, body, **kwargs):
            forwarded.append(copy.deepcopy(body))
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        proxy._retry_request = upstream

        def request(query):
            return client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 16,
                    "system": f"cwd: {tmp_path}",
                    "messages": [
                        {"role": "user", "content": f"Earlier tool result: <<ccr:{key},json,1KB>>"},
                        {"role": "user", "content": query},
                    ],
                },
            )

        for _ in range(4):
            result = request("Describe unrelated deployment work")
            assert result.status_code == 200, result.text
        result = request("authentication middleware")
        assert result.status_code == 200, result.text
        assert proxy._turn_counter == 5
        assert "STALE_CONTEXT_SENTINEL" not in str(forwarded[-1])
        assert store.retrieve(key).original_content == original
        # A new compression of identical data has the same hash but is a fresh
        # event; it must remain eligible rather than inheriting the expired age.
        assert (
            store.store(
                original, "authentication middleware", query_context="authentication middleware"
            )
            == key
        )
        result = request("authentication middleware")
        assert result.status_code == 200, result.text
        assert "STALE_CONTEXT_SENTINEL" in str(forwarded[-1])
