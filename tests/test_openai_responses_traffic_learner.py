from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi.testclient import TestClient

from headroom.memory.traffic_learner import TrafficLearner
from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _responses_input_to_learner_messages,
)
from headroom.proxy.server import ProxyConfig, create_app


class _CompletedResponseTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "model": "gpt-5",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )


class _RecordingLearner:
    def __init__(self) -> None:
        self._backend = None
        self._extractor = TrafficLearner(backend=None)
        self.message_batches: list[list[dict[str, Any]]] = []
        self.tool_results: list[dict[str, Any]] = []

    def extract_tool_results_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._extractor.extract_tool_results_from_messages(messages)

    async def on_tool_result(self, **tool_result: Any) -> None:
        self.tool_results.append(tool_result)

    async def on_messages(self, messages: list[dict[str, Any]]) -> None:
        self.message_batches.append(messages)


def _responses_input() -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Always return compact JSON."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "shell",
            "arguments": '{"cmd":"missing-command"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "command not found",
            "status": "failed",
        },
    ]


def test_responses_input_normalizes_messages_and_tool_results() -> None:
    messages = _responses_input_to_learner_messages("Follow repository rules.", _responses_input())
    learner = TrafficLearner(backend=None)

    assert messages[0] == {"role": "system", "content": "Follow repository rules."}
    assert messages[1] == {"role": "user", "content": "Always return compact JSON."}
    assert learner.extract_tool_results_from_messages(messages) == [
        {
            "tool_name": "shell",
            "input": {"cmd": "missing-command"},
            "output": "command not found",
            "is_error": True,
            "call_id": "call_1",
        }
    ]


def test_responses_input_does_not_promote_unknown_role_to_user() -> None:
    messages = _responses_input_to_learner_messages(
        None,
        [
            {
                "type": "message",
                "content": [{"type": "input_text", "text": "Never expose ambient UI."}],
            },
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Always follow runtime policy."}],
            },
        ],
    )

    assert messages == [
        {"role": "unknown", "content": "Never expose ambient UI."},
        {"role": "developer", "content": "Always follow runtime policy."},
    ]


def test_unresolved_project_skips_openai_learner_entry_points() -> None:
    learner = _RecordingLearner()
    backend_bindings: list[object] = []
    setattr(learner, "set_backend", backend_bindings.append)
    memory_handler = SimpleNamespace(
        is_project_unresolved=lambda _ctx: True,
        initialized=True,
        backend=object(),
    )
    proxy = SimpleNamespace(traffic_learner=learner, memory_handler=memory_handler)

    async def run() -> None:
        await OpenAIHandlerMixin._observe_openai_responses_traffic(
            proxy,
            {"input": _responses_input()},
            request_id="responses-unresolved",
            request_context=object(),
        )
        await OpenAIHandlerMixin._observe_openai_chat_traffic(
            proxy,
            [{"role": "user", "content": "remember this"}],
            request_id="chat-unresolved",
            request_context=object(),
        )
        seen_call_ids: set[str] = set()
        await OpenAIHandlerMixin._observe_openai_ws_response_create(
            proxy,
            _ws_frame(["baseline"]),
            seen_call_ids=seen_call_ids,
            baseline=True,
            request_id="ws-baseline-unresolved",
            request_context=object(),
        )
        await OpenAIHandlerMixin._observe_openai_ws_response_create(
            proxy,
            _ws_frame(["later"]),
            seen_call_ids=seen_call_ids,
            baseline=False,
            request_id="ws-later-unresolved",
            request_context=object(),
        )
        assert seen_call_ids == set()

    asyncio.run(run())
    assert backend_bindings == []
    assert learner.message_batches == []
    assert learner.tool_results == []


def test_responses_http_request_reaches_traffic_learner() -> None:
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
    learner = _RecordingLearner()
    proxy = app.state.proxy
    proxy.traffic_learner = learner
    proxy.http_client = httpx.AsyncClient(transport=_CompletedResponseTransport())
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"authorization": "Bearer test-token"},
        json={"model": "gpt-5", "input": _responses_input(), "stream": False},
    )

    assert response.status_code == 200, response.text
    assert len(learner.message_batches) == 1
    assert learner.tool_results == [
        {
            "tool_name": "shell",
            "tool_input": {"cmd": "missing-command"},
            "tool_output": "command not found",
            "is_error": True,
        }
    ]


def _ws_frame(call_ids: list[str]) -> dict[str, Any]:
    """A response.create inner payload whose input carries one shell tool
    round-trip per call id."""
    input_items: list[dict[str, Any]] = []
    for cid in call_ids:
        input_items.append(
            {"type": "function_call", "call_id": cid, "name": "shell", "arguments": "{}"}
        )
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": cid,
                "output": "ok",
                "status": "completed",
            }
        )
    return {"input": input_items}


def test_ws_response_create_baselines_and_dedups_replayed_transcript() -> None:
    handler = OpenAIHandlerMixin()
    learner = _RecordingLearner()
    handler.traffic_learner = learner
    seen: set[str] = set()

    # First frame is the baseline: A and B are recorded as seen but NOT learned,
    # and preference extraction is skipped.
    asyncio.run(
        handler._observe_openai_ws_response_create(
            _ws_frame(["A", "B"]), seen_call_ids=seen, baseline=True, request_id="r"
        )
    )
    assert learner.tool_results == []
    assert learner.message_batches == []
    assert seen == {"A", "B"}

    # Second frame replays A, B and appends C -> only C is learned.
    asyncio.run(
        handler._observe_openai_ws_response_create(
            _ws_frame(["A", "B", "C"]), seen_call_ids=seen, baseline=False, request_id="r"
        )
    )
    assert len(learner.tool_results) == 1
    assert seen == {"A", "B", "C"}
    assert len(learner.message_batches) == 1

    # Third frame replays A, B, C and appends D -> only D is learned.
    asyncio.run(
        handler._observe_openai_ws_response_create(
            _ws_frame(["A", "B", "C", "D"]), seen_call_ids=seen, baseline=False, request_id="r"
        )
    )
    assert len(learner.tool_results) == 2  # C then D, never A/B again
    assert seen == {"A", "B", "C", "D"}


def test_ws_reconnect_replay_adds_no_evidence() -> None:
    # A reconnect is a fresh connection: its first frame replays the whole
    # transcript, which is baselined, so nothing is re-learned.
    handler = OpenAIHandlerMixin()
    learner = _RecordingLearner()
    handler.traffic_learner = learner
    seen: set[str] = set()

    asyncio.run(
        handler._observe_openai_ws_response_create(
            _ws_frame(["A", "B", "C", "D"]), seen_call_ids=seen, baseline=True, request_id="r"
        )
    )
    assert learner.tool_results == []
    assert seen == {"A", "B", "C", "D"}
