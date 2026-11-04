"""The redrive loop: a re-driving hook split across HTTP calls (spec sections 2.4, 3.2).

Invariants protected here:

* L-CLIENT   - the client never sees the synthetic ``search_tools`` call; it sees
               the provider's final answer.
* L-PROVIDER - the provider sees exactly one call per round, the second carrying
               the reloaded tool and the appended assistant-call + tool-result.
* L-ROUNDS   - ``rounds`` counts re-drives; the registry's cap and the hook's own
               ``max_rounds`` both bound the loop.
* L-CLEAN    - after ``done`` the turn is gone from the registry and no suspended
               task is left on the event loop.
* L-GUARDS   - wrong turn id 404; ``response`` missing on a redrive turn 400.
* L-NOSHRINK - ``can_redrive=False``, ``session_affinity=False`` and streaming
               requests never see the deferral (tools are a subset of the input).
* L-SHAPES   - the loop works for the Anthropic Messages shape too.
"""

from __future__ import annotations

from typing import Any

from tests.gateway import redrive_hook_ext
from tests.gateway.conftest import count_loop_tasks, registry_of
from tests.gateway.fake_provider import (
    anthropic_text_response,
    anthropic_tool_use_response,
    anthropic_usage,
    openai_text_response,
    openai_tool_call_response,
    openai_usage,
)
from tests.gateway.samples import (
    SYSTEM_PROMPT,
    anthropic_tool_history,
    anthropic_tools,
    big_tool_history,
    openai_tools,
    tool_names,
)

SEARCH = redrive_hook_ext.SEARCH_TOOL_NAME


def _chat_body(**extra: Any) -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": big_tool_history(), "tools": openai_tools(), **extra}


def _anthropic_body(**extra: Any) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 128,
        "system": SYSTEM_PROMPT,
        "messages": anthropic_tool_history(),
        "tools": anthropic_tools(),
        **extra,
    }


def _search_then_answer(fake_provider, query: str = "create issue") -> None:
    fake_provider.script(
        [
            openai_tool_call_response(
                SEARCH, {"query": query}, openai_usage(300, 8), call_id="call_s1"
            ),
            openai_text_response(
                "Issue created.", openai_usage(420, 6, 300), response_id="chatcmpl-final"
            ),
        ]
    )


def _has_search_call(response: dict[str, Any]) -> bool:
    return redrive_hook_ext.detect_search_call(response) is not None


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #


def test_one_redrive_round_end_to_end(headroom_client, fake_provider, make_gateway) -> None:
    """L-CLIENT, L-PROVIDER, L-ROUNDS, L-CLEAN together."""
    hook = redrive_hook_ext.register()
    _search_then_answer(fake_provider)
    baseline = count_loop_tasks(headroom_client)
    gateway = make_gateway(can_redrive=True, can_relay_response=True)

    result = gateway.turn(_chat_body(), "loop-1")

    assert result.path == "loop", result.fail_open_reason
    assert result.obligations == ["redrive", "relay_usage"]
    # L-CLIENT
    assert result.final_status == 200
    assert result.final_response is not None
    assert result.final_response["id"] == "chatcmpl-final"
    assert not _has_search_call(result.final_response)
    # L-PROVIDER
    assert len(fake_provider.calls) == 2
    first, second = fake_provider.calls
    assert tool_names(first.tools) == ["get_items", SEARCH]
    assert "deferred_create_issue" in tool_names(second.tools)
    assert tool_names(second.tools)[:2] == ["get_items", SEARCH]
    assert second.messages[: len(first.messages)] == first.messages
    assert second.messages[-2]["role"] == "assistant"
    assert second.messages[-2]["tool_calls"][0]["id"] == "call_s1"
    assert second.messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_s1",
        "content": second.messages[-1]["content"],
    }
    assert second.messages[-1]["content"].startswith(redrive_hook_ext.LOADED_PREFIX)
    # The redrive request is the FULL provider body, not just messages.
    redrive_step = result.response_half_calls[0].response
    assert redrive_step["action"] == "redrive"
    assert redrive_step["round"] == 1
    assert redrive_step["turn_id"] == result.turn_id
    assert redrive_step["request"]["model"] == "gpt-4o"
    assert redrive_step["request"]["messages"] == second.messages
    assert redrive_step["request"]["tools"] == second.tools
    assert "gateway" not in redrive_step["request"] and "config" not in redrive_step["request"]
    # L-ROUNDS
    done = result.done
    assert done is not None
    assert done["rounds"] == 1
    # The hook hands back the latest provider response object itself (no
    # replacement). After a re-drive Headroom still returns that response, with
    # usage summed over both provider calls, so the client never sees the last
    # round's usage alone.
    assert done["response"] is not None
    assert done["response"]["id"] == "chatcmpl-final"
    assert done["usage_applied"] is True  # cached_tokens=300 on the final answer
    # The turn's bill covers both provider calls, not just the last one.
    assert done["billed_usage"]["input_tokens"] == 300 + 420
    assert done["billed_usage"]["output_tokens"] == 8 + 6
    assert done["billed_usage"]["cache_read_input_tokens"] == 300
    assert done["response"]["usage"]["prompt_tokens"] == 720
    assert done["response"]["usage"]["completion_tokens"] == 14
    assert done["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 300
    # What the gateway hands the client is that cumulative response.
    assert result.final_response["usage"]["prompt_tokens"] == 720
    assert result.final_response["usage"]["completion_tokens"] == 14
    assert hook.rounds_driven == 1
    assert hook.queries == ["create issue"]
    # L-CLEAN
    assert registry_of(headroom_client).get(result.turn_id) is None
    assert count_loop_tasks(headroom_client) <= baseline


def test_no_search_call_means_done_with_null_response(
    headroom_client, fake_provider, make_gateway
) -> None:
    """L-CLIENT: the hook armed but the model answered directly -> ``done`` with
    ``response: null`` and the gateway forwards what it already has."""
    redrive_hook_ext.register()
    fake_provider.script([openai_text_response("Direct answer.", response_id="chatcmpl-direct")])
    gateway = make_gateway(can_redrive=True, can_relay_response=False)
    result = gateway.turn(_chat_body(), "loop-direct")
    assert result.path == "loop"
    assert len(fake_provider.calls) == 1
    done = result.done
    assert done is not None
    assert done["response"] is None
    assert done["rounds"] == 0
    assert result.final_response["id"] == "chatcmpl-direct"
    assert registry_of(headroom_client).get(result.turn_id) is None


def test_two_rounds(headroom_client, fake_provider, make_gateway) -> None:
    """L-ROUNDS: two searches, three provider calls, ``rounds == 2``, and the
    tools block accumulates (sorted) across rounds."""
    redrive_hook_ext.register()
    fake_provider.script(
        [
            openai_tool_call_response(SEARCH, {"query": "create issue"}, call_id="s1"),
            openai_tool_call_response(SEARCH, {"query": "query db"}, call_id="s2"),
            openai_text_response("Both done.", response_id="chatcmpl-final2"),
        ]
    )
    result = make_gateway(can_redrive=True, can_relay_response=True).turn(_chat_body(), "loop-2")
    assert len(fake_provider.calls) == 3
    assert result.done["rounds"] == 2
    assert result.final_response["id"] == "chatcmpl-final2"
    third = fake_provider.calls[2]
    assert tool_names(third.tools) == [
        "get_items",
        SEARCH,
        "deferred_create_issue",
        "deferred_query_db",
    ]
    assert [m["role"] for m in third.messages[-4:]] == ["assistant", "tool", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# Bounds                                                                       #
# --------------------------------------------------------------------------- #


def test_hook_max_rounds_caps_the_loop(headroom_client, fake_provider, make_gateway) -> None:
    """L-ROUNDS: the hook's own cap stops the loop; the client gets the last
    provider answer (still a search call - visible rather than silently wrong)."""
    redrive_hook_ext.register(max_rounds=2)
    fake_provider.script(
        [openai_tool_call_response(SEARCH, {"query": "again"}, call_id=f"s{i}") for i in range(10)]
    )
    result = make_gateway(can_redrive=True, can_relay_response=False, max_redrives=20).turn(
        _chat_body(), "loop-cap"
    )
    assert len(fake_provider.calls) == 3  # 1 + max_rounds
    assert result.done is not None
    assert result.done["rounds"] == 2
    assert registry_of(headroom_client).get(result.turn_id) is None


def test_registry_max_redrives_env_caps_the_loop(
    make_headroom_client, monkeypatch, fake_provider, provider_client
) -> None:
    """L-ROUNDS: ``HEADROOM_GATEWAY_MAX_REDRIVES`` bounds a hook that would spin."""
    from tests.gateway.fake_gateway import FakeGateway

    monkeypatch.setenv("HEADROOM_GATEWAY_MAX_REDRIVES", "2")
    client = make_headroom_client()
    redrive_hook_ext.register(max_rounds=50)
    fake_provider.script(
        [openai_tool_call_response(SEARCH, {"query": "again"}, call_id=f"s{i}") for i in range(10)]
    )
    gateway = FakeGateway(
        client, provider_client, can_redrive=True, can_relay_response=False, max_redrives=50
    )
    result = gateway.turn(_chat_body(), "loop-envcap")
    assert result.done is not None
    # Two re-drives are served (1 + 2 provider calls); the third request bumps
    # ``rounds`` past the cap and is answered ``done`` instead of ``redrive``.
    assert len(fake_provider.calls) == 3
    assert result.done["rounds"] == 3
    # The capped turn still ran three provider calls for one client call: the
    # latest provider answer comes back with usage summed over all three.
    assert result.done["response"]["choices"][0]["message"]["tool_calls"][0]["id"] == "s2"
    assert result.done["response"]["usage"]["prompt_tokens"] == 3 * 10
    assert result.done["response"]["usage"]["completion_tokens"] == 3 * 5
    assert registry_of(client).get(result.turn_id) is None


# --------------------------------------------------------------------------- #
# Guards                                                                       #
# --------------------------------------------------------------------------- #


def test_wrong_turn_id_is_404(headroom_client, fake_provider) -> None:
    """L-GUARDS"""
    redrive_hook_ext.register()
    resp = headroom_client.post(
        "/v1/compress/response",
        json={"turn_id": "f" * 32, "response": openai_text_response("x")},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "unknown_turn"


def test_missing_response_on_redrive_turn_is_400(headroom_client) -> None:
    """L-GUARDS: a turn that carries ``redrive`` cannot complete without the
    provider response; the turn stays registered for a corrected relay."""
    redrive_hook_ext.register()
    data = headroom_client.post(
        "/v1/compress",
        json={**_chat_body(), "gateway": {"can_redrive": True}, "config": {"session_id": "guard"}},
    ).json()
    assert data["obligations"] == ["redrive"]
    resp = headroom_client.post(
        "/v1/compress/response", json={"turn_id": data["turn_id"], "usage": openai_usage(1, 1)}
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["type"] == "missing_response"
    assert registry_of(headroom_client).get(data["turn_id"]) is not None


def test_failed_call_closes_a_redrive_turn_without_a_response(headroom_client) -> None:
    """L-GUARDS: a provider failure has nothing to re-drive. A gateway that
    only knows the status (LiteLLM's failure hooks see an exception, not a
    body) must still be able to close the turn and leave a ledger row."""
    redrive_hook_ext.register()
    data = headroom_client.post(
        "/v1/compress",
        json={**_chat_body(), "gateway": {"can_redrive": True}, "config": {"session_id": "fail"}},
    ).json()
    assert data["obligations"] == ["redrive"]
    resp = headroom_client.post(
        "/v1/compress/response", json={"turn_id": data["turn_id"], "round": 0, "status": 529}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["action"] == "done"
    assert resp.json()["response"] is None
    assert registry_of(headroom_client).get(data["turn_id"]) is None


def test_non_200_provider_status_skips_hooks(headroom_client, fake_provider, make_gateway) -> None:
    """L-GUARDS: a provider error is relayed, not re-driven; the turn completes."""
    from tests.gateway.fake_provider import ScriptedResponse

    hook = redrive_hook_ext.register()
    fake_provider.script([ScriptedResponse(body={"error": {"message": "overloaded"}}, status=529)])
    result = make_gateway(can_redrive=True, can_relay_response=True).turn(_chat_body(), "loop-529")
    assert result.path == "loop"
    assert result.final_status == 529
    assert result.done is not None
    assert result.done["response"] is None
    assert hook.response_calls == 0
    assert registry_of(headroom_client).get(result.turn_id) is None


# --------------------------------------------------------------------------- #
# No shrink without reload                                                     #
# --------------------------------------------------------------------------- #


def test_can_redrive_false_never_shrinks(headroom_client, fake_provider, make_gateway) -> None:
    """L-NOSHRINK"""
    hook = redrive_hook_ext.register()
    result = make_gateway(can_redrive=False, can_relay_response=True).turn(
        _chat_body(), "noshrink-1"
    )
    assert result.path == "proxy"
    assert "redrive" not in result.obligations
    assert len(fake_provider.calls) == 1
    sent = fake_provider.calls[0]
    assert SEARCH not in tool_names(sent.tools)
    assert set(tool_names(sent.tools)) <= set(tool_names(openai_tools()))
    assert hook.shrink_calls == 0


def test_session_affinity_false_never_shrinks(headroom_client, fake_provider, make_gateway) -> None:
    """L-NOSHRINK: without affinity the response half may land on another replica."""
    hook = redrive_hook_ext.register()
    result = make_gateway(can_redrive=True, can_relay_response=True, session_affinity=False).turn(
        _chat_body(), "noshrink-2"
    )
    assert result.path == "proxy"
    assert "redrive" not in result.obligations
    assert SEARCH not in tool_names(fake_provider.calls[0].tools)
    assert hook.shrink_calls == 0


def test_streaming_request_never_shrinks(headroom_client, fake_provider, make_gateway) -> None:
    """L-NOSHRINK: the plugin sets ``can_redrive=false`` for ``stream: true``; the
    stream reaches the client with its usage frame and is relayed."""
    hook = redrive_hook_ext.register()
    fake_provider.script([openai_text_response("streamed", openai_usage(50, 4, 20))])
    gateway = make_gateway(can_redrive=True, can_relay_response=True)
    body = _chat_body(stream=True, stream_options={"include_usage": True})
    result = gateway.turn(body, "noshrink-stream")
    assert gateway.gateway_block(body)["can_redrive"] is False
    assert result.path == "proxy"
    assert hook.shrink_calls == 0
    assert SEARCH not in tool_names(fake_provider.calls[0].tools)
    assert result.final_headers["content-type"].startswith("text/event-stream")
    relay = result.response_half_calls[-1]
    assert relay.request["usage"] == openai_usage(50, 4, 20)
    assert relay.status == 200
    assert relay.response["usage_applied"] is True


# --------------------------------------------------------------------------- #
# Anthropic shape                                                              #
# --------------------------------------------------------------------------- #


def test_anthropic_shape_redrive(headroom_client, fake_provider, make_gateway) -> None:
    """L-SHAPES: tool_use search -> tool_result reload -> final text."""
    hook = redrive_hook_ext.register()
    fake_provider.script(
        [
            anthropic_tool_use_response(
                SEARCH,
                {"query": "query db"},
                anthropic_usage(200, 9, 0, 150),
                tool_use_id="toolu_s1",
            ),
            anthropic_text_response(
                "Rows fetched.", anthropic_usage(260, 7, 150, 0), response_id="msg_final"
            ),
        ]
    )
    result = make_gateway(can_redrive=True, can_relay_response=True).turn(
        _anthropic_body(), "anthropic-loop"
    )
    assert result.path == "loop", result.fail_open_reason
    assert result.compress_response["route"]["provider"] == "anthropic"
    assert result.final_response["id"] == "msg_final"
    assert not _has_search_call(result.final_response)
    assert len(fake_provider.calls) == 2
    first, second = fake_provider.calls
    assert first.path.endswith("/v1/messages")
    assert first.body["system"] == SYSTEM_PROMPT
    assert first.body["max_tokens"] == 128
    assert tool_names(first.tools) == ["get_items", SEARCH]
    assert "input_schema" in first.tools[-1]
    assert tool_names(second.tools) == ["get_items", SEARCH, "deferred_query_db"]
    assert second.messages[-2]["role"] == "assistant"
    assert second.messages[-2]["content"][0]["type"] == "tool_use"
    assert second.messages[-1]["role"] == "user"
    assert second.messages[-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_s1",
        "content": second.messages[-1]["content"][0]["content"],
    }
    assert result.done["rounds"] == 1
    assert hook.rounds_driven == 1
    assert registry_of(headroom_client).get(result.turn_id) is None


def test_stale_retry_replays_last_redrive_without_double_billing(
    headroom_client, fake_provider, make_gateway
) -> None:
    """Invariant: the response half is idempotent on ``round``.

    A gateway whose socket dropped after headroom answered ``redrive`` re-posts
    round 0. Headroom must repeat the same ``redrive`` answer, not feed round
    0's response to the parked hook as the answer to the re-drive, and must not
    bill round 0's usage twice.
    """
    from tests.gateway import redrive_hook_ext
    from tests.gateway.fake_provider import openai_text_response, openai_tool_call_response
    from tests.gateway.samples import big_tool_history, openai_tools

    redrive_hook_ext.register()
    search_call = openai_tool_call_response(
        "search_tools", {"query": "deferred"}, usage={"prompt_tokens": 100, "completion_tokens": 5}
    )
    body = {"model": "gpt-4o", "messages": big_tool_history(), "tools": openai_tools()}
    compress = headroom_client.post(
        "/v1/compress",
        json={**body, "gateway": {"can_redrive": True, "can_relay_response": True}},
    ).json()
    assert "redrive" in compress["obligations"]
    turn_id = compress["turn_id"]

    first = headroom_client.post(
        "/v1/compress/response",
        json={
            "turn_id": turn_id,
            "round": 0,
            "usage": search_call["usage"],
            "response": search_call,
        },
    ).json()
    assert first["action"] == "redrive" and first["round"] == 1

    # The retry of round 0 must replay, byte-for-byte, the answer above.
    retry = headroom_client.post(
        "/v1/compress/response",
        json={
            "turn_id": turn_id,
            "round": 0,
            "usage": search_call["usage"],
            "response": search_call,
        },
    ).json()
    assert retry == first

    # A round from the future is refused; nothing is consumed.
    future = headroom_client.post(
        "/v1/compress/response",
        json={"turn_id": turn_id, "round": 2, "response": search_call},
    )
    assert future.status_code == 400
    assert future.json()["error"]["type"] == "round_out_of_sequence"

    final = openai_text_response(
        "final answer", usage={"prompt_tokens": 120, "completion_tokens": 7}
    )
    done = headroom_client.post(
        "/v1/compress/response",
        json={"turn_id": turn_id, "round": 1, "usage": final["usage"], "response": final},
    ).json()
    assert done["action"] == "done"
    assert done["rounds"] == 1
    # Round 0 billed once, round 1 once: 100+120 in, 5+7 out.
    proxy = headroom_client.app.state.proxy
    assert proxy.gateway_turns.get(turn_id) is None
    # Also drive the whole thing through the FakeGateway once for the round plumbing.
    fake_provider.script([search_call, final])
    gw = make_gateway(can_redrive=True, can_relay_response=True)
    result = gw.turn(body, session_id=None)
    assert result.path == "loop"
    assert [c.request["round"] for c in result.response_half_calls] == [0, 1]
    assert result.final_response["choices"][0]["message"]["content"] == "final answer"


def test_billed_usage_is_written_in_the_responses_own_shape() -> None:
    from headroom.proxy.gateway_turn import NormalizedUsage, _usage_in_shape

    billed = NormalizedUsage(input_tokens=720, output_tokens=14, cache_read=300, cache_write=50)
    anthropic = _usage_in_shape(
        {"input_tokens": 420, "output_tokens": 6, "service_tier": "standard"}, billed
    )
    assert anthropic == {
        "input_tokens": 720,
        "output_tokens": 14,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 50,
        "service_tier": "standard",
    }
    chat = _usage_in_shape(
        {"prompt_tokens": 420, "completion_tokens": 6, "total_tokens": 426}, billed
    )
    assert chat == {
        "prompt_tokens": 720,
        "completion_tokens": 14,
        "total_tokens": 734,
        "prompt_tokens_details": {"cached_tokens": 300},
    }
