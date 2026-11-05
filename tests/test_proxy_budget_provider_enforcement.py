"""Provider-level regression coverage for proxy spend-budget enforcement."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _budget_exhausted_app():  # noqa: ANN202
    return create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=True,
            budget_limit_usd=0.0,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
    )


@pytest.mark.parametrize(
    ("path", "headers", "body"),
    [
        (
            "/v1/messages",
            {"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
        (
            "/v1/chat/completions",
            {"authorization": "Bearer test-key"},
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        ),
        (
            "/v1/responses",
            {"authorization": "Bearer test-key"},
            {"model": "gpt-4o-mini", "input": "hello"},
        ),
        (
            "/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            {},
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        ),
        (
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent?key=test-key",
            {},
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        ),
    ],
)
def test_exhausted_budget_rejects_every_generation_http_route_before_upstream(
    path: str,
    headers: dict[str, str],
    body: dict[str, object],
) -> None:
    upstream_calls: list[str] = []

    with TestClient(_budget_exhausted_app(), client=("127.0.0.1", 50000)) as client:
        proxy = client.app.state.proxy

        async def _unexpected_retry(method, url, request_headers, request_body, **kwargs):  # noqa: ANN001, ANN202
            upstream_calls.append(str(url))
            raise AssertionError("budget-exhausted request reached the upstream")

        async def _unexpected_stream(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            upstream_calls.append("stream")
            raise AssertionError("budget-exhausted stream reached the upstream")

        proxy._retry_request = _unexpected_retry
        proxy._stream_response = _unexpected_stream

        response = client.post(path, headers=headers, json=body)

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert upstream_calls == []


class _BudgetDeniedWebSocket:
    def __init__(self) -> None:
        self.headers = {"authorization": "Bearer test-key"}
        self.url = SimpleNamespace(path="/v1/responses")
        self.client = SimpleNamespace(host="127.0.0.1", port=50000)
        self.closed: tuple[int, str] | None = None

    async def close(self, *, code: int, reason: str) -> None:
        self.closed = (code, reason)


def test_exhausted_budget_rejects_responses_websocket_before_upstream() -> None:
    app = _budget_exhausted_app()
    websocket = _BudgetDeniedWebSocket()

    asyncio.run(app.state.proxy.handle_openai_responses_ws(websocket))

    assert websocket.closed == (1008, "Budget exceeded for daily period")


@pytest.mark.asyncio
async def test_responses_websocket_rechecks_budget_between_turns() -> None:
    from tests.test_openai_codex_ws_lifecycle import (
        _DummyOpenAIHandler,
        _FakeUpstream,
        _FakeWebSocket,
        _first_frame,
        _make_fake_websockets_module,
    )

    class _TurnBudget:
        def __init__(self) -> None:
            self.first_turn_recorded = asyncio.Event()

        def check_budget(self) -> tuple[bool, float]:
            return (not self.first_turn_recorded.is_set(), 1.0)

        def budget_denial_detail(self) -> str:
            return "Budget exceeded for daily period"

        def record_tokens(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.first_turn_recorded.set()

    class _TwoTurnWebSocket(_FakeWebSocket):
        def __init__(self, budget: _TurnBudget) -> None:
            super().__init__(frames=[_first_frame(), _first_frame()])
            self._budget = budget
            self._received = 0

        async def receive_text(self) -> str:
            self._received += 1
            if self._received == 2:
                await self._budget.first_turn_recorded.wait()
            return await super().receive_text()

    completed = json.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "model": "gpt-5.4",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        }
    )
    budget = _TurnBudget()
    upstream = _FakeUpstream([completed], hold_after_events=True)
    handler = _DummyOpenAIHandler()
    handler.cost_tracker = budget
    websocket = _TwoTurnWebSocket(budget)
    fake_websockets = _make_fake_websockets_module(upstream)

    with patch.dict(sys.modules, {"websockets": fake_websockets}):
        await handler.handle_openai_responses_ws(websocket)

    assert len(upstream.sent) == 1
    assert websocket.close_code == 1008
    assert websocket.close_reason == "Budget exceeded for daily period"
