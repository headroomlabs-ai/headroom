"""Session-affinity header: sent to providers that route on it, and nowhere else."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

REPLY = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


def forwarded_headers(upstream: str, extra: dict[str, str] | None = None) -> list[dict]:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        anthropic_api_url=upstream,
    )
    sent: list[dict] = []

    async def fake_upstream(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003
        sent.append({k.lower(): v for k, v in headers.items()})
        return httpx.Response(200, json=REPLY)

    with TestClient(create_app(config)) as client:
        client.app.state.proxy._retry_request = fake_upstream
        for _ in range(2):
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "k", "anthropic-version": "2023-06-01", **(extra or {})},
                json={
                    "model": "kimi",
                    "max_tokens": 5,
                    "system": "You are a coding agent.",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200
    return sent


def test_fireworks_gets_one_stable_affinity_per_conversation() -> None:
    first, second = forwarded_headers("https://api.fireworks.ai/inference")
    assert first["x-session-affinity"]
    assert first["x-session-affinity"] == second["x-session-affinity"]


@pytest.mark.parametrize("upstream", ["https://api.anthropic.com", "http://vllm.internal:8000"])
def test_other_upstreams_get_no_affinity_header(upstream: str) -> None:
    assert all("x-session-affinity" not in h for h in forwarded_headers(upstream))


def test_client_affinity_header_is_kept() -> None:
    sent = forwarded_headers("https://api.fireworks.ai/inference", {"X-Session-Affinity": "mine"})
    assert all(h["x-session-affinity"] == "mine" for h in sent)
