"""Tests for the x-headroom-keep-last-turns feature (issue #2858).

Covers the apply_keep_last_turns() helper that both anthropic.py and
openai.py call before the optimized_messages assignment, plus handler-level
regression coverage (PR #3059 review): the helper-only tests below all use
perfectly alternating user/assistant pairs, which cannot catch a turn-
boundary defect that only shows up once a turn spans more than two messages
(tool_calls/tool-result round trips).
"""

from __future__ import annotations

import httpx
import pytest

from headroom.proxy.helpers import apply_keep_last_turns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turns(n: int) -> list[dict]:
    """Build a plausible conversation: n user+assistant pairs + trailing user."""
    msgs: list[dict] = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    msgs.append({"role": "user", "content": "final question"})
    return msgs


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


def test_empty_messages_returns_unchanged():
    result, dropped = apply_keep_last_turns([], n=0)
    assert result == []
    assert dropped == 0


def test_n_negative_is_noop():
    msgs = _turns(3)
    result, dropped = apply_keep_last_turns(msgs, n=-1)
    assert result is msgs
    assert dropped == 0


def test_n_larger_than_history_is_noop():
    """When n >= existing turns no messages are dropped."""
    msgs = _turns(2)  # 5 messages total
    result, dropped = apply_keep_last_turns(msgs, n=10)
    assert result is msgs
    assert dropped == 0


def test_exact_turns_match_is_noop():
    """Asking for exactly the number of turns present keeps everything."""
    msgs = _turns(3)  # 3 turns → 7 messages
    result, dropped = apply_keep_last_turns(msgs, n=3)
    assert result is msgs
    assert dropped == 0


# ---------------------------------------------------------------------------
# Trimming cases
# ---------------------------------------------------------------------------


def test_n0_keeps_only_final_user_message():
    """n=0 means 'no prior turns' → only the trailing user message survives."""
    msgs = _turns(3)  # 7 messages
    result, dropped = apply_keep_last_turns(msgs, n=0)
    assert len(result) == 1
    assert result[0]["content"] == "final question"
    assert dropped == 6


def test_n1_keeps_one_prior_turn_plus_current():
    """n=1 keeps the most recent user+assistant pair plus the current user message."""
    msgs = _turns(3)  # q0 a0 q1 a1 q2 a2 q_final (7 msgs)
    result, dropped = apply_keep_last_turns(msgs, n=1)
    # Keeps: a2, q_final = indices 5, 6 → but formula gives tail = 7-1-1*2=4
    # messages[4:] = q2 a2 q_final  (3 messages)
    assert len(result) == 3
    assert result[-1]["content"] == "final question"
    assert dropped == 4


def test_n2_keeps_two_prior_turns_plus_current():
    msgs = _turns(4)  # 9 messages
    result, dropped = apply_keep_last_turns(msgs, n=2)
    # tail = 9-1-2*2 = 4; messages[4:] = 5 messages
    assert len(result) == 5
    assert result[-1]["content"] == "final question"
    assert dropped == 4


def test_trailing_user_message_always_present():
    """The current (final) user message is never dropped regardless of n."""
    for n in range(5):
        msgs = _turns(5)
        result, _ = apply_keep_last_turns(msgs, n=n)
        assert result[-1] == {"role": "user", "content": "final question"}, f"n={n}"


def test_total_dropped_plus_kept_equals_original():
    msgs = _turns(5)  # 11 messages
    for n in range(6):
        result, dropped = apply_keep_last_turns(msgs, n=n)
        assert len(result) + dropped == len(msgs), f"n={n}"


# ---------------------------------------------------------------------------
# Single-message conversation
# ---------------------------------------------------------------------------


def test_single_message_conversation_n0():
    """A single-message conversation (no history) is never trimmed."""
    msgs = [{"role": "user", "content": "hello"}]
    result, dropped = apply_keep_last_turns(msgs, n=0)
    assert result is msgs
    assert dropped == 0


def test_single_message_conversation_large_n():
    msgs = [{"role": "user", "content": "hello"}]
    result, dropped = apply_keep_last_turns(msgs, n=100)
    assert result is msgs
    assert dropped == 0


# ---------------------------------------------------------------------------
# Return-value contract
# ---------------------------------------------------------------------------


def test_no_trim_returns_original_object():
    """When nothing is dropped the original list is returned (identity, not copy)."""
    msgs = _turns(2)
    result, dropped = apply_keep_last_turns(msgs, n=100)
    assert result is msgs
    assert dropped == 0


def test_trim_returns_new_slice():
    """When trimming, a new slice (not the original object) is returned."""
    msgs = _turns(3)
    result, dropped = apply_keep_last_turns(msgs, n=0)
    assert result is not msgs
    assert dropped > 0


# ---------------------------------------------------------------------------
# Turn-boundary correctness with tool_calls/tool_result round trips
# (PR #3059 review: a fixed-size arithmetic slice assumes every turn is
# exactly two messages, so a turn with tool calls -- which spans more than
# two -- gets cut mid-turn, orphaning a tool result from its tool_calls
# entry.)
# ---------------------------------------------------------------------------


def _openai_tool_turn(q: str, tool_call_id: str, tool_result: str, a: str) -> list[dict]:
    """One OpenAI-shape turn with a tool round trip: 4 messages, not 2."""
    return [
        {"role": "user", "content": q},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": tool_call_id, "content": tool_result},
        {"role": "assistant", "content": a},
    ]


def test_tool_call_turn_kept_whole_not_split_at_arithmetic_midpoint():
    """A single tool-calling turn (4 messages) plus the trailing user message
    must be kept entirely when n=1 -- the old formula (len-1-n*2) would slice
    at index 2, orphaning the tool result.
    """
    msgs = [
        *_openai_tool_turn("q0", "call_1", "result0", "a0"),
        {"role": "user", "content": "final"},
    ]

    result, dropped = apply_keep_last_turns(msgs, n=1)

    assert result is msgs
    assert dropped == 0


def test_earlier_tool_call_turn_dropped_whole_not_split():
    """Dropping an earlier tool-calling turn must drop it atomically -- the
    tool result never survives without its tool_calls entry, and vice versa.
    """
    msgs = [
        *_openai_tool_turn("q0", "call_1", "result0", "a0"),
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "final"},
    ]

    result, dropped = apply_keep_last_turns(msgs, n=1)

    assert dropped == 4
    assert result == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "final"},
    ]
    # No orphaned tool message and no dangling tool_calls entry.
    assert not any(m.get("role") == "tool" for m in result)
    assert not any(m.get("tool_calls") for m in result)


def test_anthropic_tool_result_user_message_does_not_start_a_new_turn():
    """Anthropic represents a tool result as role="user" with a tool_result
    content block -- it must not be misread as the start of a fresh turn.
    """
    msgs = [
        {"role": "user", "content": "q0"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "result0"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "a0"}]},
        {"role": "user", "content": "final"},
    ]

    result, dropped = apply_keep_last_turns(msgs, n=1)

    assert result is msgs
    assert dropped == 0


# ---------------------------------------------------------------------------
# system/developer instruction preservation (PR #3059 review, round 2):
# OpenAI keeps system/developer messages inline in the same `messages` array
# a client sends, so without special handling a trim that reaches back far
# enough silently drops the application's own instructions along with old
# conversation turns.
# ---------------------------------------------------------------------------


def test_leading_system_and_developer_messages_survive_full_trim():
    """The reviewer's exact repro: n=0 must drop the old turn but keep both
    instruction messages, not just the trailing user message.
    """
    msgs = [
        {"role": "system", "content": "Always answer in JSON"},
        {"role": "developer", "content": "Use the required response schema"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]

    result, dropped = apply_keep_last_turns(msgs, n=0)

    assert result == [
        {"role": "system", "content": "Always answer in JSON"},
        {"role": "developer", "content": "Use the required response schema"},
        {"role": "user", "content": "new question"},
    ]
    # Only the two actually-removed conversational messages count as
    # dropped -- the two retained instruction messages do not, even though
    # the turn-boundary cutoff logically falls past them.
    assert dropped == 2


def test_system_and_developer_messages_keep_their_original_relative_order():
    """Instruction messages are preserved in place, not hoisted to the front."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
        {"role": "developer", "content": "dev"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "final"},
    ]

    result, dropped = apply_keep_last_turns(msgs, n=0)

    assert result == [
        {"role": "system", "content": "sys"},
        {"role": "developer", "content": "dev"},
        {"role": "user", "content": "final"},
    ]
    assert dropped == 4


def test_no_instruction_messages_behaves_exactly_as_before():
    """Backward compatibility: with no system/developer messages present,
    behavior is identical to the plain turn-boundary trim.
    """
    msgs = _turns(3)

    result, dropped = apply_keep_last_turns(msgs, n=1)

    assert len(result) == 3
    assert dropped == 4


# ---------------------------------------------------------------------------
# Handler-level regression coverage (PR #3059 review): exercise the real
# /v1/chat/completions and /v1/messages handlers so this proves the message
# list actually forwarded upstream is coherent, not just the helper in
# isolation.
# ---------------------------------------------------------------------------


def test_handler_openai_keep_last_turns_preserves_tool_call_pairing():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

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
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        captured: dict[str, object] = {}

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["messages"] = body["messages"]
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                },
            )

        proxy._retry_request = _fake_retry

        messages = [
            *_openai_tool_turn("q0", "call_1", "result0", "a0"),
            {"role": "user", "content": "final question"},
        ]
        response = client.post(
            "/v1/chat/completions",
            headers={"x-headroom-keep-last-turns": "1"},
            json={"model": "gpt-4o", "messages": messages},
        )

        assert response.status_code == 200, response.text
        sent = captured["messages"]
        # n=1 with exactly one prior (tool-calling) turn: nothing dropped.
        assert sent == messages
        tool_call_ids = {
            tc["id"]
            for msg in sent
            if msg.get("role") == "assistant"
            for tc in (msg.get("tool_calls") or [])
        }
        tool_message_ids = {msg["tool_call_id"] for msg in sent if msg.get("role") == "tool"}
        assert tool_message_ids <= tool_call_ids, (
            "a tool result was orphaned from its tool_calls entry"
        )


def test_handler_openai_keep_last_turns_preserves_instructions_and_retained_tool_turn():
    """Regression for PR #3059 review, round 2: system/developer instructions
    must survive a trim that drops an older turn, and a tool-using turn that
    IS retained must still forward intact (no orphaned tool result) --
    exercised through the real handler, not the helper in isolation.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

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
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        captured: dict[str, object] = {}

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["messages"] = body["messages"]
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                },
            )

        proxy._retry_request = _fake_retry

        system_msg = {"role": "system", "content": "Always answer in JSON"}
        developer_msg = {"role": "developer", "content": "Use the required response schema"}
        old_turn = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        retained_tool_turn = _openai_tool_turn("q0", "call_1", "result0", "a0")
        messages = [
            system_msg,
            developer_msg,
            *old_turn,
            *retained_tool_turn,
            {"role": "user", "content": "final question"},
        ]
        response = client.post(
            "/v1/chat/completions",
            headers={"x-headroom-keep-last-turns": "1"},
            json={"model": "gpt-4o", "messages": messages},
        )

        assert response.status_code == 200, response.text
        sent = captured["messages"]

        # Both instructions survive, in their original relative position.
        assert sent[0] == system_msg
        assert sent[1] == developer_msg
        # The old plain turn is gone; the retained tool-using turn and the
        # trailing user message follow immediately after the instructions.
        assert sent[2:] == [*retained_tool_turn, {"role": "user", "content": "final question"}]

        tool_call_ids = {
            tc["id"]
            for msg in sent
            if msg.get("role") == "assistant"
            for tc in (msg.get("tool_calls") or [])
        }
        tool_message_ids = {msg["tool_call_id"] for msg in sent if msg.get("role") == "tool"}
        assert tool_message_ids <= tool_call_ids, (
            "a tool result was orphaned from its tool_calls entry"
        )


def test_handler_anthropic_keep_last_turns_ordinary_history():
    """Ordinary (non-tool) Anthropic history must still trim correctly."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_inject_system_instructions=False,
        image_optimize=False,
    )
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        captured: dict[str, object] = {}

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["messages"] = body["messages"]
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        messages = [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": [{"type": "text", "text": "a0"}]},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "user", "content": "final question"},
        ]
        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-keep-last-turns": "1",
            },
            json={"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": messages},
        )

        assert response.status_code == 200, response.text
        sent = captured["messages"]
        # n=1 with two prior turns: keep the last prior turn + trailing user.
        # The handler injects a cache_control breakpoint on the last
        # assistant text block (unrelated Anthropic prompt-caching feature),
        # so compare content by stripping that marker rather than byte-exact.
        expected = messages[2:]
        assert len(sent) == len(expected)
        assert [m["role"] for m in sent] == [m["role"] for m in expected]
        for sent_msg, expected_msg in zip(sent, expected, strict=True):
            if isinstance(sent_msg.get("content"), list):
                for block in sent_msg["content"]:
                    block.pop("cache_control", None)
            assert sent_msg == expected_msg


def test_handler_anthropic_keep_last_turns_preserves_tool_use_pairing():
    """Anthropic tool_use/tool_result round trip must not be split -- the
    tool_result is a role="user" message and must not be mistaken for a new
    turn boundary.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_inject_system_instructions=False,
        image_optimize=False,
    )
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        captured: dict[str, object] = {}

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["messages"] = body["messages"]
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        messages = [
            {"role": "user", "content": "q0"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result0"}
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "a0"}]},
            {"role": "user", "content": "final question"},
        ]
        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-keep-last-turns": "1",
            },
            json={"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": messages},
        )

        assert response.status_code == 200, response.text
        # n=1 with exactly one prior (tool-using) turn: nothing dropped, and
        # the tool_use/tool_result pairing survives intact. The handler
        # injects a cache_control breakpoint on the last assistant text
        # block (unrelated Anthropic prompt-caching feature), so strip that
        # marker before comparing rather than asserting byte-exact equality.
        sent = captured["messages"]
        assert len(sent) == len(messages)
        for sent_msg in sent:
            if isinstance(sent_msg.get("content"), list):
                for block in sent_msg["content"]:
                    block.pop("cache_control", None)
        assert sent == messages
        tool_use_ids = {
            block["id"]
            for msg in sent
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
            for block in msg["content"]
            if block.get("type") == "tool_use"
        }
        tool_result_ids = {
            block["tool_use_id"]
            for msg in sent
            if msg.get("role") == "user" and isinstance(msg.get("content"), list)
            for block in msg["content"]
            if block.get("type") == "tool_result"
        }
        assert tool_result_ids <= tool_use_ids, "a tool_result was orphaned from its tool_use block"
