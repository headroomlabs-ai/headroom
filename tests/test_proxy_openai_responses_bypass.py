from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


class _MemoryHandler:
    def __init__(self) -> None:
        self.search_calls = 0
        self.tool_calls = 0
        self.config = SimpleNamespace(inject_context=True, inject_tools=True)

    async def search_and_format_context(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> str:
        self.search_calls += 1
        return "memory context that must not be injected"

    def compute_memory_tool_definitions(self, provider: str) -> list[dict[str, Any]]:
        self.tool_calls += 1
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "search memory",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def has_memory_tool_calls(self, response: dict[str, Any], provider: str) -> bool:
        return False


def test_responses_bypass_skips_memory_and_compression_mutation() -> None:
    app = create_app(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None

    original_input = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "summarize"}],
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "large tool output " * 200,
        },
    ]
    captured: dict[str, Any] = {}

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        memory_handler = _MemoryHandler()
        proxy.memory_handler = memory_handler

        async def _fake_retry(
            method: str,
            url: str,
            headers: dict[str, str],
            body: dict[str, Any],
            stream: bool = False,
            **kwargs: Any,
        ) -> httpx.Response:
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/responses",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-bypass": "true",
                "x-headroom-user-id": "user-1",
            },
            json={"model": "gpt-4o-mini", "input": original_input},
        )

    assert response.status_code == 200
    assert captured["body"]["input"] == original_input
    assert "tools" not in captured["body"]
    assert memory_handler.search_calls == 0
    assert memory_handler.tool_calls == 0
