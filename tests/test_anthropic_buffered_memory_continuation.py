"""Buffered memory continuations answer every tool_use they replay (GH #4009).

On a buffered Anthropic turn the proxy runs its injected memory tools and sends
the model a continuation. That continuation replays the whole assistant turn,
so a client tool called alongside a memory tool used to go upstream with no
``tool_result`` and the provider rejected it with a 400.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

MEMORY_TOOLS = ("memory_save", "memory_search")


class _MemoryHandler:
    def __init__(self) -> None:
        self.config = type(
            "MemoryConfig",
            (),
            {"inject_context": False, "inject_tools": True, "project_root_override": ""},
        )()
        self.initialized = False
        self.backend = None
        self.executed: list[str] = []

    def compute_memory_tool_definitions(self, provider: str) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": name, "input_schema": {"type": "object"}}
            for name in MEMORY_TOOLS
        ]

    def get_beta_headers(self) -> dict[str, str]:
        return {}

    def has_memory_tool_calls(self, response: dict[str, Any], provider: str) -> bool:
        return bool(self._memory_calls(response))

    async def handle_memory_tool_calls(
        self, response: dict[str, Any], user_id: str, provider: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        # Like the real handler: results only for memory tools.
        results = []
        for block in self._memory_calls(response):
            self.executed.append(block["name"])
            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": "ok"})
        return results

    @staticmethod
    def _memory_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            block
            for block in response.get("content") or []
            if block.get("type") == "tool_use" and block.get("name") in MEMORY_TOOLS
        ]


def _message(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


MEMORY_CALL = {
    "type": "tool_use",
    "id": "toolu_mem",
    "name": "memory_search",
    "input": {"query": "deploy region"},
}
CLIENT_CALL = {"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {"command": "ls"}}


def _run(upstream: list[httpx.Response]) -> tuple[httpx.Response, list[dict], _MemoryHandler]:
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
    sent: list[dict] = []
    handler = _MemoryHandler()
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        proxy.memory_handler = handler

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent.append(json.loads(json.dumps(body)))
            return upstream[len(sent) - 1]

        proxy._retry_request = _fake_retry  # type: ignore[assignment]
        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "u1",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
                "messages": [{"role": "user", "content": "check the deploy region"}],
            },
        )
    return response, sent, handler


def test_mixed_turn_returns_client_tool_without_continuation() -> None:
    first = _message([MEMORY_CALL, CLIENT_CALL], "tool_use")
    response, sent, handler = _run([httpx.Response(200, json=first)])

    assert response.status_code == 200, response.text
    # No continuation: it could not answer the Bash call.
    assert len(sent) == 1
    assert handler.executed == ["memory_search"]
    # The client gets its own tool call back, not the proxy's.
    assert [block["name"] for block in response.json()["content"]] == ["Bash"]


def test_memory_only_turn_continues_with_a_result_for_every_call() -> None:
    first = _message([MEMORY_CALL], "tool_use")
    final = _message([{"type": "text", "text": "us-east-1"}], "end_turn")
    response, sent, _ = _run([httpx.Response(200, json=first), httpx.Response(200, json=final)])

    assert len(sent) == 2
    assistant, user = sent[1]["messages"][-2:]
    assert assistant == {"role": "assistant", "content": [MEMORY_CALL]}
    assert [r["tool_use_id"] for r in user["content"]] == ["toolu_mem"]
    assert sent[1]["messages"][:-2] == sent[0]["messages"]
    assert response.json()["content"] == [{"type": "text", "text": "us-east-1"}]


def test_failed_continuation_is_not_logged_as_complete(caplog: pytest.LogCaptureFixture) -> None:
    first = _message([MEMORY_CALL], "tool_use")
    rejected = httpx.Response(
        400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
    )
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        _run([httpx.Response(200, json=first), rejected])

    assert "Memory: Continuation failed with upstream status 400" in caplog.text
    assert "continuation complete" not in caplog.text
