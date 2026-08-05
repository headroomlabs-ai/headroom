from __future__ import annotations

from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def test_anthropic_messages_strips_local_1m_model_suffix_before_forwarding() -> None:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    client = TestClient(app)

    captured: dict[str, Any] = {}

    async def _fake_retry(
        method: str,  # noqa: ARG001
        url: str,  # noqa: ARG001
        headers: dict[str, str],  # noqa: ARG001
        body: dict[str, Any],
        body_mutated: bool,
        mutation_reasons: list[str],
        **kwargs: Any,
    ) -> httpx.Response:
        captured["body"] = dict(body)
        captured["body_mutated"] = body_mutated
        captured["mutation_reasons"] = list(mutation_reasons)
        return httpx.Response(
            200,
            json={
                "id": "msg_glm_1m",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    app.state.proxy._retry_request = _fake_retry  # type: ignore[assignment]

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "glm-5.2[1m]",
            "max_tokens": 10,
            "stream": False,
            "messages": [{"role": "user", "content": "2+2"}],
        },
    )

    assert response.status_code == 200, response.text
    assert captured["body"]["model"] == "glm-5.2"
    assert captured["body_mutated"] is True
    assert captured["mutation_reasons"] == ["sanitize_model_id"]
