from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.test_openai_codex_ws_lifecycle import (
    _DummyOpenAIHandler,
    _FakeUpstream,
    _FakeWebSocket,
    _make_fake_websockets_module,
)


class _MemoryHandler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            inject_context=True,
            inject_tools=True,
            project_root_override="",
        )
        self.queries: list[str] = []

    async def search_and_format_context(self, _user_id, messages, **_kwargs):
        current_turn = messages[-1]["content"] if messages else ""
        self.queries.append(current_turn)
        return f"current memory: {current_turn}"

    def compute_memory_tool_definitions(self, _provider):
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "search",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_save",
                    "description": "save",
                    "parameters": {"type": "object"},
                },
            },
        ]


def _expected_memory_response_tools() -> list[dict[str, object]]:
    expected: list[dict[str, object]] = []
    for tool in _MemoryHandler().compute_memory_tool_definitions("openai"):
        function = tool["function"]
        expected.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"],
            }
        )
    return expected


def _turn(text: str) -> str:
    return json.dumps({"type": "response.create", "response": {"input": text}})


def _direct_turn(text: str) -> str:
    return json.dumps({"input": text})


def _issue_2059_artifact_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "issues" / "headroom_issue_2059.json"


def _issue_2059_turns() -> tuple[str, str]:
    issue_path = _issue_2059_artifact_path()
    issue = json.loads(issue_path.read_text(encoding="utf-8"))
    match = re.search(r"```json\s*(.*?)```", issue["body"], re.DOTALL)
    assert match is not None, "issue 2059 artifact must contain a JSON code sample"
    frames = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    assert len(frames) == 2, "issue 2059 artifact must contain exactly two frames"
    return frames[0], frames[1]


def _issue_2059_inputs() -> tuple[str, str]:
    first, later = _issue_2059_turns()
    return (
        json.loads(first)["response"]["input"],
        json.loads(later)["response"]["input"],
    )


def _list_turn(text: str, *, instructions: str) -> str:
    return json.dumps(
        {
            "type": "response.create",
            "response": {
                "instructions": instructions,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                ],
            },
        }
    )


class _FlakyMemoryHandler(_MemoryHandler):
    def __init__(self, *, fail_on: set[str]) -> None:
        super().__init__()
        self.fail_on = set(fail_on)

    async def search_and_format_context(self, _user_id, messages, **_kwargs):
        current_turn = messages[-1]["content"] if messages else ""
        self.queries.append(current_turn)
        if current_turn in self.fail_on:
            raise RuntimeError(f"memory failed for {current_turn}")
        return f"current memory: {current_turn}"


class _ToolFailingMemoryHandler(_MemoryHandler):
    def __init__(self) -> None:
        super().__init__()
        self._fail_next_tools = True

    def compute_memory_tool_definitions(self, _provider):
        if self._fail_next_tools:
            self._fail_next_tools = False
            raise RuntimeError("memory tool preparation failed")
        return super().compute_memory_tool_definitions(_provider)


@pytest.mark.asyncio
async def test_memory_lookup_runs_for_each_issue_artifact_frame_and_preserves_non_create_frames():
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    first_turn, later_turn = _issue_2059_turns()
    first_input, later_input = _issue_2059_inputs()
    client_frames = [
        first_turn,
        json.dumps({"type": "response.cancel"}),
        later_turn,
    ]
    client_ws = _FakeWebSocket(frames=client_frames)
    handler = _DummyOpenAIHandler()
    memory = _MemoryHandler()
    handler.memory_handler = memory

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await handler.handle_openai_responses_ws(client_ws)

    assert memory.queries == [first_input, later_input]
    assert upstream.sent[1] == client_frames[1]
    forwarded_turns = [
        json.loads(frame) for frame in upstream.sent if "response" in json.loads(frame)
    ]
    assert f"current memory: {first_input}" in forwarded_turns[0]["response"]["input"]
    assert f"current memory: {later_input}" in forwarded_turns[1]["response"]["input"]
    expected_tools = _expected_memory_response_tools()
    for frame in forwarded_turns:
        assert frame["response"]["tools"] == expected_tools
    assert forwarded_turns[0]["response"]["tools"] == forwarded_turns[1]["response"]["tools"]


@pytest.mark.asyncio
async def test_memory_lookup_skips_input_bearing_non_create_first_frame():
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    _first_input, later_input = _issue_2059_inputs()
    cancel_frame = json.dumps(
        {
            "type": "response.cancel",
            "response_id": "r_1",
            "input": "must not query",
        }
    )
    later_turn = _issue_2059_turns()[1]
    client_ws = _FakeWebSocket(frames=[cancel_frame, later_turn])
    handler = _DummyOpenAIHandler()
    memory = _MemoryHandler()
    handler.memory_handler = memory

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await handler.handle_openai_responses_ws(client_ws)

    assert memory.queries == [later_input]
    assert upstream.sent[0] == cancel_frame
    forwarded_later = json.loads(upstream.sent[1])
    assert f"current memory: {later_input}" in forwarded_later["response"]["input"]


@pytest.mark.asyncio
async def test_memory_lookup_skips_bypassed_frames():
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    first, later = _issue_2059_turns()
    client_ws = _FakeWebSocket(
        frames=[first, later],
        headers={"authorization": "Bearer test", "x-headroom-bypass": "true"},
    )
    handler = _DummyOpenAIHandler()
    memory = _MemoryHandler()
    handler.memory_handler = memory

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await handler.handle_openai_responses_ws(client_ws)

    assert memory.queries == []
    assert upstream.sent == [first, later]


@pytest.mark.asyncio
async def test_memory_lookup_keeps_legacy_direct_first_frame():
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    first_input, later_input = _issue_2059_inputs()
    first = _direct_turn(first_input)
    later = _issue_2059_turns()[1]
    client_ws = _FakeWebSocket(frames=[first, later])
    handler = _DummyOpenAIHandler()
    memory = _MemoryHandler()
    handler.memory_handler = memory

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await handler.handle_openai_responses_ws(client_ws)

    assert memory.queries == [first_input, later_input]
    forwarded_first = json.loads(upstream.sent[0])
    forwarded_later = json.loads(upstream.sent[1])
    assert f"current memory: {first_input}" in forwarded_first["input"]
    assert f"current memory: {later_input}" in forwarded_later["response"]["input"]


@pytest.mark.asyncio
async def test_memory_lookup_skips_disabled_memory(monkeypatch):
    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "disabled")
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    first, later = _issue_2059_turns()
    client_ws = _FakeWebSocket(frames=[first, later])
    handler = _DummyOpenAIHandler()
    memory = _MemoryHandler()
    handler.memory_handler = memory

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await handler.handle_openai_responses_ws(client_ws)

    assert memory.queries == []
    assert upstream.sent == [first, later]


@pytest.mark.asyncio
async def test_memory_lookup_fails_open_and_recovers_on_later_frame():
    first, later = _issue_2059_turns()
    first_input, later_input = _issue_2059_inputs()
    upstream = _FakeUpstream([], hold_after_events=True)
    client_ws = _FakeWebSocket(frames=[first, later], hold_after_initial=True)
    handler = _DummyOpenAIHandler()
    memory = _FlakyMemoryHandler(fail_on={first_input})
    handler.memory_handler = memory

    async def _trigger() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        trigger_task = asyncio.create_task(_trigger())
        try:
            await handler.handle_openai_responses_ws(client_ws)
        finally:
            trigger_task.cancel()
            try:
                await trigger_task
            except asyncio.CancelledError:
                pass

    assert memory.queries == [first_input, later_input]
    assert upstream.sent[0] == first
    assert f"current memory: {later_input}" in json.loads(upstream.sent[1])["response"]["input"]


@pytest.mark.asyncio
async def test_memory_lookup_fails_open_when_tool_preparation_raises():
    first, later = _issue_2059_turns()
    first_input, later_input = _issue_2059_inputs()
    upstream = _FakeUpstream([], hold_after_events=True)
    client_ws = _FakeWebSocket(frames=[first, later], hold_after_initial=True)
    handler = _DummyOpenAIHandler()
    memory = _ToolFailingMemoryHandler()
    handler.memory_handler = memory

    async def _trigger() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        trigger_task = asyncio.create_task(_trigger())
        try:
            await handler.handle_openai_responses_ws(client_ws)
        finally:
            trigger_task.cancel()
            try:
                await trigger_task
            except asyncio.CancelledError:
                pass

    assert memory.queries == [first_input, later_input]
    assert upstream.sent[0] == first
    assert f"current memory: {later_input}" in json.loads(upstream.sent[1])["response"]["input"]


@pytest.mark.asyncio
async def test_memory_lookup_preserves_list_shaped_later_frame_input():
    first, _later = _issue_2059_turns()
    list_frame = _list_turn(
        "later turn with list payload",
        instructions="list payload instructions",
    )
    expected_input = json.loads(list_frame)["response"]["input"]
    upstream = _FakeUpstream([], hold_after_events=True)
    client_ws = _FakeWebSocket(frames=[first, list_frame], hold_after_initial=True)
    handler = _DummyOpenAIHandler()
    memory = _MemoryHandler()
    handler.memory_handler = memory

    async def _trigger() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        trigger_task = asyncio.create_task(_trigger())
        try:
            await handler.handle_openai_responses_ws(client_ws)
        finally:
            trigger_task.cancel()
            try:
                await trigger_task
            except asyncio.CancelledError:
                pass

    forwarded_later = json.loads(upstream.sent[1])
    assert forwarded_later["response"]["input"] == expected_input
    assert memory.queries[-1] == "list payload instructions"


@pytest.mark.asyncio
async def test_later_frame_compression_receives_memory_prepared_input():
    first, later = _issue_2059_turns()
    _first_input, later_input = _issue_2059_inputs()
    upstream = _FakeUpstream([], hold_after_events=True)
    client_ws = _FakeWebSocket(frames=[first, later], hold_after_initial=True)
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    memory = _MemoryHandler()
    handler.memory_handler = memory
    seen_inputs: list[object] = []

    def _capture_compress(payload, *, model, request_id, timing=None):
        seen_inputs.append(payload["input"])
        return payload, False, 0, [], "test_noop", 10, 10, 0

    async def _trigger() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    handler._compress_openai_responses_payload = _capture_compress  # type: ignore[method-assign]

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        trigger_task = asyncio.create_task(_trigger())
        try:
            await handler.handle_openai_responses_ws(client_ws)
        finally:
            trigger_task.cancel()
            try:
                await trigger_task
            except asyncio.CancelledError:
                pass

    assert len(seen_inputs) == 2
    assert f"current memory: {later_input}" in str(seen_inputs[1])
