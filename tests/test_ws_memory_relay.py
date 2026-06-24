"""Tests for WebSocket memory tool interception in the Codex Responses API relay.

Verifies that:
1. Memory tool events are suppressed from reaching Codex
2. response.created is buffered and only flushed for non-memory responses
3. Tool execution happens and continuation is sent upstream
4. Non-memory responses pass through with normal streaming latency
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from headroom.proxy.memory_handler import MEMORY_TOOL_NAMES

# ---------------------------------------------------------------------------
# Minimal WS relay state machine (mirrors the logic in openai.py)
# ---------------------------------------------------------------------------


@dataclass
class WSMemoryRelayState:
    """State machine for WS event processing with memory tool interception.

    This mirrors the logic in ``_upstream_to_client`` but is decoupled from
    actual WebSocket I/O so it can be unit-tested.
    """

    memory_tool_names: set[str] = field(default_factory=lambda: set(MEMORY_TOOL_NAMES))

    # Per-response state (reset after each response.completed)
    event_buffer: list[str] = field(default_factory=list)
    decided: bool = False
    suppress_response: bool = False
    pending_function_calls: list[dict[str, Any]] = field(default_factory=list)
    last_response_id: str | None = None

    def process_event(self, msg_str: str) -> dict[str, Any]:
        """Process a single upstream WS event.

        Returns a dict with possible keys:
            relay: list[str]        — events to send to Codex
            execute_tools: list     — function_call items to execute
            send_continuation: dict — continuation payload to send upstream
        """
        result: dict[str, Any] = {"relay": [], "execute_tools": [], "send_continuation": None}

        try:
            event = json.loads(msg_str)
        except (json.JSONDecodeError, TypeError):
            # Not JSON — always relay
            result["relay"].append(msg_str)
            return result

        event_type = event.get("type", "")

        # ---- Phase 1: Buffering (before first output item) ----
        if not self.decided:
            self.event_buffer.append(msg_str)

            if event_type == "response.output_item.added":
                item = event.get("item", {})
                if (
                    item.get("type") == "function_call"
                    and item.get("name") in self.memory_tool_names
                ):
                    # Memory tool is first output → suppress entire response
                    self.suppress_response = True
                    self.decided = True
                    self.event_buffer.clear()
                else:
                    # Non-memory item → flush buffer and pass through
                    self.decided = True
                    result["relay"].extend(self.event_buffer)
                    self.event_buffer.clear()

            elif event_type == "response.completed":
                # Response completed with no output items — flush all
                self.decided = True
                result["relay"].extend(self.event_buffer)
                self.event_buffer.clear()

            return result

        # ---- Phase 2a: Suppress mode (memory tool response) ----
        if self.suppress_response:
            # Capture completed function_call items
            if event_type == "response.output_item.done":
                item = event.get("item", {})
                if (
                    item.get("type") == "function_call"
                    and item.get("name") in self.memory_tool_names
                ):
                    self.pending_function_calls.append(item)

            if event_type == "response.completed":
                resp = event.get("response", {})
                self.last_response_id = resp.get("id")

                if self.pending_function_calls:
                    result["execute_tools"] = list(self.pending_function_calls)
                    # Build continuation payload
                    # (actual tool execution + output building done by caller)
                    result["send_continuation"] = {
                        "response_id": self.last_response_id,
                        "function_calls": list(self.pending_function_calls),
                    }

                # Reset for next response (continuation)
                self._reset_response_state()

            return result  # Nothing relayed in suppress mode

        # ---- Phase 2b: Pass-through mode (normal response) ----
        result["relay"].append(msg_str)
        return result

    def _reset_response_state(self) -> None:
        """Reset per-response state for the next response."""
        self.event_buffer.clear()
        self.decided = False
        self.suppress_response = False
        self.pending_function_calls.clear()
        self.last_response_id = None


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: str, **kwargs: Any) -> str:
    data: dict[str, Any] = {"type": event_type}
    data.update(kwargs)
    return json.dumps(data)


def _response_created(response_id: str = "resp_A") -> str:
    return _make_event("response.created", response={"id": response_id})


def _output_item_added_text(index: int = 0) -> str:
    return _make_event(
        "response.output_item.added",
        output_index=index,
        item={"type": "message", "role": "assistant"},
    )


def _output_item_added_function_call(name: str, index: int = 0, call_id: str = "call_1") -> str:
    return _make_event(
        "response.output_item.added",
        output_index=index,
        item={"type": "function_call", "name": name, "call_id": call_id},
    )


def _function_call_args_delta(index: int = 0, delta: str = '{"qu') -> str:
    return _make_event(
        "response.function_call_arguments.delta",
        output_index=index,
        delta=delta,
    )


def _function_call_args_done(index: int = 0, arguments: str = '{"query": "codename"}') -> str:
    return _make_event(
        "response.function_call_arguments.done",
        output_index=index,
        arguments=arguments,
    )


def _output_item_done_function_call(
    name: str,
    index: int = 0,
    call_id: str = "call_1",
    arguments: str = '{"query": "codename"}',
) -> str:
    return _make_event(
        "response.output_item.done",
        output_index=index,
        item={
            "type": "function_call",
            "name": name,
            "call_id": call_id,
            "arguments": arguments,
        },
    )


def _output_text_delta(index: int = 0, text: str = "Hello") -> str:
    return _make_event(
        "response.output_text.delta",
        output_index=index,
        delta=text,
    )


def _output_item_done_text(index: int = 0) -> str:
    return _make_event(
        "response.output_item.done",
        output_index=index,
        item={"type": "message", "role": "assistant"},
    )


def _response_completed(response_id: str = "resp_A") -> str:
    return _make_event(
        "response.completed",
        response={"id": response_id, "status": "completed"},
    )


def _output_item_added_shell(index: int = 0) -> str:
    """Simulate a Codex built-in tool (shell) that should pass through."""
    return _make_event(
        "response.output_item.added",
        output_index=index,
        item={"type": "function_call", "name": "shell", "call_id": "call_shell"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWSMemoryRelayNonMemory:
    """Responses with no memory tools pass through normally."""

    def test_text_response_relayed_immediately(self):
        """Text-only response: all events relayed, no buffering after first item."""
        relay = WSMemoryRelayState()

        events = [
            _response_created(),
            _output_item_added_text(),
            _output_text_delta(text="The answer is 42"),
            _output_item_done_text(),
            _response_completed(),
        ]

        all_relayed: list[str] = []
        for ev in events:
            result = relay.process_event(ev)
            all_relayed.extend(result["relay"])
            assert result["execute_tools"] == []
            assert result["send_continuation"] is None

        # All 5 events should be relayed
        assert len(all_relayed) == 5

        # First event (response.created) should be buffered then flushed
        # with the second event (output_item_added_text)
        types = [json.loads(e)["type"] for e in all_relayed]
        assert types == [
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.output_item.done",
            "response.completed",
        ]

    def test_shell_tool_relayed(self):
        """Codex built-in tools (shell) pass through without interception."""
        relay = WSMemoryRelayState()

        events = [
            _response_created(),
            _output_item_added_shell(),
            _response_completed(),
        ]

        all_relayed: list[str] = []
        for ev in events:
            result = relay.process_event(ev)
            all_relayed.extend(result["relay"])
            assert result["execute_tools"] == []
            assert result["send_continuation"] is None

        assert len(all_relayed) == 3

    def test_empty_response_relayed(self):
        """Response with no output items still relays created + completed."""
        relay = WSMemoryRelayState()

        events = [
            _response_created(),
            _response_completed(),
        ]

        all_relayed: list[str] = []
        for ev in events:
            result = relay.process_event(ev)
            all_relayed.extend(result["relay"])

        assert len(all_relayed) == 2


class TestWSMemoryRelayMemoryTool:
    """Responses with memory tools are intercepted transparently."""

    def test_memory_search_fully_suppressed(self):
        """memory_search call: ALL events suppressed from Codex."""
        relay = WSMemoryRelayState()

        events = [
            _response_created("resp_A"),
            _output_item_added_function_call("memory_search", index=0),
            _function_call_args_delta(index=0),
            _function_call_args_done(index=0),
            _output_item_done_function_call("memory_search", index=0),
            _response_completed("resp_A"),
        ]

        all_relayed: list[str] = []
        tool_executions: list[Any] = []
        continuations: list[Any] = []

        for ev in events:
            result = relay.process_event(ev)
            all_relayed.extend(result["relay"])
            tool_executions.extend(result["execute_tools"])
            if result["send_continuation"]:
                continuations.append(result["send_continuation"])

        # ZERO events relayed to Codex
        assert len(all_relayed) == 0, (
            f"Expected 0 relayed events, got {len(all_relayed)}: "
            f"{[json.loads(e)['type'] for e in all_relayed]}"
        )

        # Tool execution triggered
        assert len(tool_executions) == 1
        assert tool_executions[0]["name"] == "memory_search"

        # Continuation requested
        assert len(continuations) == 1
        assert continuations[0]["response_id"] == "resp_A"

    def test_memory_save_also_suppressed(self):
        """memory_save call is also intercepted."""
        relay = WSMemoryRelayState()

        events = [
            _response_created("resp_B"),
            _output_item_added_function_call("memory_save", index=0, call_id="call_save"),
            _function_call_args_done(index=0, arguments='{"content": "user likes dark mode"}'),
            _output_item_done_function_call(
                "memory_save",
                index=0,
                call_id="call_save",
                arguments='{"content": "user likes dark mode"}',
            ),
            _response_completed("resp_B"),
        ]

        all_relayed: list[str] = []
        tool_executions: list[Any] = []

        for ev in events:
            result = relay.process_event(ev)
            all_relayed.extend(result["relay"])
            tool_executions.extend(result["execute_tools"])

        assert len(all_relayed) == 0
        assert len(tool_executions) == 1
        assert tool_executions[0]["name"] == "memory_save"

    def test_continuation_response_relayed_normally(self):
        """After memory tool handling, the continuation response passes through."""
        relay = WSMemoryRelayState()

        # --- First response: memory_search (suppressed) ---
        first_response_events = [
            _response_created("resp_A"),
            _output_item_added_function_call("memory_search", index=0),
            _function_call_args_done(index=0),
            _output_item_done_function_call("memory_search", index=0),
            _response_completed("resp_A"),
        ]

        for ev in first_response_events:
            relay.process_event(ev)

        # --- Second response: continuation text (relayed) ---
        continuation_events = [
            _response_created("resp_B"),
            _output_item_added_text(index=0),
            _output_text_delta(index=0, text="The codename is Pegasus-2"),
            _output_item_done_text(index=0),
            _response_completed("resp_B"),
        ]

        all_relayed: list[str] = []
        for ev in continuation_events:
            result = relay.process_event(ev)
            all_relayed.extend(result["relay"])
            assert result["execute_tools"] == []
            assert result["send_continuation"] is None

        # All continuation events relayed
        assert len(all_relayed) == 5
        types = [json.loads(e)["type"] for e in all_relayed]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"

        # Verify the text content
        text_events = [
            json.loads(e)
            for e in all_relayed
            if json.loads(e)["type"] == "response.output_text.delta"
        ]
        assert len(text_events) == 1
        assert text_events[0]["delta"] == "The codename is Pegasus-2"

    def test_non_json_message_always_relayed(self):
        """Binary or non-JSON messages pass through regardless."""
        relay = WSMemoryRelayState()

        result = relay.process_event("not valid json {{{")
        assert len(result["relay"]) == 1
        assert result["relay"][0] == "not valid json {{{"

    def test_multiple_memory_tools_in_one_response(self):
        """Multiple memory tools in one response — all suppressed."""
        relay = WSMemoryRelayState()

        events = [
            _response_created("resp_multi"),
            _output_item_added_function_call("memory_search", index=0, call_id="call_1"),
            _output_item_done_function_call("memory_search", index=0, call_id="call_1"),
            # The model decides to save something too
            _output_item_added_function_call("memory_save", index=1, call_id="call_2"),
            _output_item_done_function_call(
                "memory_save",
                index=1,
                call_id="call_2",
                arguments='{"content": "test"}',
            ),
            _response_completed("resp_multi"),
        ]

        all_relayed: list[str] = []
        tool_executions: list[Any] = []
        continuations: list[Any] = []

        for ev in events:
            result = relay.process_event(ev)
            all_relayed.extend(result["relay"])
            tool_executions.extend(result["execute_tools"])
            if result["send_continuation"]:
                continuations.append(result["send_continuation"])

        assert len(all_relayed) == 0
        assert len(tool_executions) == 2
        assert {t["name"] for t in tool_executions} == {"memory_search", "memory_save"}
        assert len(continuations) == 1


class TestWSMemoryRelayStateReset:
    """State resets properly between responses."""

    def test_state_resets_after_memory_response(self):
        """After a memory response, the relay is ready for a fresh response."""
        relay = WSMemoryRelayState()

        # Memory response
        for ev in [
            _response_created("resp_A"),
            _output_item_added_function_call("memory_search"),
            _output_item_done_function_call("memory_search"),
            _response_completed("resp_A"),
        ]:
            relay.process_event(ev)

        # State should be reset
        assert relay.decided is False
        assert relay.suppress_response is False
        assert len(relay.pending_function_calls) == 0
        assert len(relay.event_buffer) == 0

    def test_alternating_memory_and_normal(self):
        """Memory response → normal response → both work correctly."""
        relay = WSMemoryRelayState()

        # 1. Memory response (suppressed)
        for ev in [
            _response_created("resp_A"),
            _output_item_added_function_call("memory_search"),
            _output_item_done_function_call("memory_search"),
            _response_completed("resp_A"),
        ]:
            relay.process_event(ev)

        # 2. Continuation text response (relayed)
        relayed: list[str] = []
        for ev in [
            _response_created("resp_B"),
            _output_item_added_text(),
            _output_text_delta(text="Pegasus-2"),
            _output_item_done_text(),
            _response_completed("resp_B"),
        ]:
            result = relay.process_event(ev)
            relayed.extend(result["relay"])

        assert len(relayed) == 5

        # 3. Another normal response should also work
        relayed2: list[str] = []
        for ev in [
            _response_created("resp_C"),
            _output_item_added_shell(),
            _response_completed("resp_C"),
        ]:
            result = relay.process_event(ev)
            relayed2.extend(result["relay"])

        assert len(relayed2) == 3
