"""Tool definitions a turn hook shrinks on ``/v1/responses`` are booked as tool savings.

Kiro CLI 3.x and Codex speak OpenAI Responses. Tool search condenses their tool
descriptions in ``on_request``; the chat-completions path booked that on
``turn_hook_tools_saved_tokens``, the tag every savings surface (PERF ``tool_saved``,
``/stats``, the control-plane heartbeat's ``toolTokensSaved``) sums, but the
Responses path dropped it, so the dashboard showed 0 tool savings for Kiro.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


@pytest.fixture(autouse=True)
def _forget_response_arms(monkeypatch):
    """Each test starts with no remembered response chains."""
    from collections import OrderedDict

    from headroom.proxy import output_arm_key

    monkeypatch.setattr(output_arm_key, "_keys_by_response", OrderedDict())


class _CondenseDescriptions:
    """Shortens every tool description in place, as tool search's condense pass does."""

    name = "condense"
    stream_safe = True

    def on_request(self, ctx: Any) -> None:
        for tool in ctx.tools or []:
            tool["description"] = tool["description"][:20]


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": f"{name} reads things carefully and reports them back " * 12,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }


def _post(body: dict) -> list:
    return _post_turns([body])


def _sse(response_id: str) -> bytes:
    response = {
        "id": response_id,
        "object": "response",
        "model": "gpt-5.5",
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    return "".join(
        f"event: {kind}\ndata: {json.dumps({'type': kind, 'response': {**response, 'status': status}})}\n\n"
        for kind, status in (
            ("response.created", "in_progress"),
            ("response.completed", "completed"),
        )
    ).encode()


def _post_turns(
    bodies: list[dict],
    *,
    response_ids: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> list:
    """POST each body to one app in order; the provider answers turn ``i`` with
    ``response_ids[i]`` (buffered or streamed, as the body asks)."""
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    ids = iter(response_ids or ["resp_1"] * len(bodies))
    app = create_app(ProxyConfig(optimize=True, cache_enabled=False, rate_limit_enabled=False))
    app.dependency_overrides[require_loopback] = lambda: None
    outcomes: list = []
    with TestClient(app) as client:
        server = app.state.proxy

        async def fake_retry(method, url, headers, body, stream=False, **kwargs):
            return httpx.Response(
                200,
                json={
                    "id": next(ids),
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5.5",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
                request=httpx.Request("POST", url),
            )

        def stream_upstream(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=_sse(next(ids)), headers={"content-type": "text/event-stream"}
            )

        record = server._record_request_outcome

        async def capture(outcome):
            outcomes.append(outcome)
            await record(outcome)

        server._retry_request = fake_retry
        server.http_client = httpx.AsyncClient(transport=httpx.MockTransport(stream_upstream))
        server.http_client_h1 = server.http_client
        server._record_request_outcome = capture
        for body in bodies:
            resp = client.post(
                "/v1/responses",
                content=json.dumps(body).encode(),
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                    **(headers or {}),
                },
            )
            assert resp.status_code == 200, resp.text
    return outcomes


def _arms(outcome: Any) -> set[str]:
    from headroom.proxy.output_savings import parse_stratum_label

    parsed = map(parse_stratum_label, outcome.transforms_applied)
    return {p[0] for p in parsed if p is not None}


def test_hook_shrunk_tool_descriptions_are_booked_as_tool_savings() -> None:
    register_turn_hook(_CondenseDescriptions())

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "list the files"}],
            "tools": [_tool("read_file"), _tool("list_dir")],
            "stream": False,
        }
    )

    assert outcomes, "no request outcome recorded"
    saved = int((outcomes[-1].tags or {}).get("turn_hook_tools_saved_tokens", 0) or 0)
    assert saved > 100, outcomes[-1].tags


def test_no_hook_change_books_no_tool_savings() -> None:
    class _Noop:
        name = "noop"
        stream_safe = True

        def on_request(self, ctx: Any) -> None:
            return None

    register_turn_hook(_Noop())

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "list the files"}],
            "tools": [_tool("read_file")],
            "stream": False,
        }
    )

    assert outcomes
    assert not (outcomes[-1].tags or {}).get("turn_hook_tools_saved_tokens")


def test_http_turn_steered_in_its_input_is_booked_as_shaped(monkeypatch) -> None:
    """Kiro's extension steers the user turn because Kiro rejects ``instructions``;
    the HTTP outcome must still carry the verbosity label the output ledger reads."""
    from headroom.proxy.output_verbosity_policy import steering_text

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0")

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": f"fix it\n\n{steering_text(3)}"}],
            "stream": False,
        }
    )

    assert outcomes
    assert "output_shaper:verbosity:L3" in outcomes[-1].transforms_applied


def _text_in_arm(model: str, arm: str) -> str:
    """A steered first user turn whose conversation lands in ``arm`` at a 30% holdout.

    Picked on the opener as the client sent it, before any steering: that is
    the body a create hook decides its steering on.
    """
    from headroom.proxy.output_savings import assign_arm, conversation_key_from_body
    from headroom.proxy.output_verbosity_policy import steering_text

    for i in range(200):
        body = {"model": model, "input": [{"role": "user", "content": f"task {i}"}]}
        if assign_arm(conversation_key_from_body(body), 0.3) == arm:
            return f"task {i}\n\n{steering_text(3)}"
    raise AssertionError(f"no conversation landed in {arm}")


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_both_shaping_passes_put_a_request_in_one_holdout_arm(monkeypatch, arm: str) -> None:
    """Optimized HTTP Responses shapes twice (in compression, then before the
    send). With a fractional holdout, two arm keys could label one request both
    treatment and control; the ledger reads the first stratum and drops a
    steered treatment turn that lacks a verbosity label."""
    from headroom.proxy.output_savings import parse_stratum_label

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0.3")
    text = _text_in_arm("gpt-5.5", arm)

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": text}],
            "stream": False,
        }
    )

    assert outcomes
    labels = list(outcomes[-1].transforms_applied)
    arms = {parsed[0] for parsed in map(parse_stratum_label, labels) if parsed is not None}
    assert arms == {arm}, labels
    # The ledger drops a treatment sample that lacks a verbosity label.
    assert ("output_shaper:verbosity:L3" in labels) is (arm == "treatment"), labels


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _body_arm(body: dict) -> str:
    from headroom.proxy.output_savings import assign_arm, conversation_key_from_body

    return assign_arm(conversation_key_from_body(body), 0.3)


@pytest.mark.parametrize("stream", [False, True])
def test_a_response_chain_keeps_its_openers_holdout_arm(monkeypatch, stream: bool) -> None:
    """A turn sent with ``previous_response_id`` carries only its own prompt, so
    keying it on its body moved the conversation to another arm mid-life."""
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0.3")
    opener = {"model": "gpt-5.5", "input": _user("fix the failing test"), "stream": stream}
    first, second = f"resp_chain_{stream}_1", f"resp_chain_{stream}_2"
    follow_up = {
        "model": "gpt-5.5",
        "previous_response_id": first,
        "input": _user("and the lint"),
        "stream": stream,
    }
    # On their own bodies the two turns land in different arms.
    assert _body_arm(opener) != _body_arm(follow_up)

    outcomes = _post_turns([opener, follow_up], response_ids=[first, second])

    assert len(outcomes) == 2
    assert [_arms(o) for o in outcomes] == [{_body_arm(opener)}] * 2


@pytest.mark.parametrize("header_name", ["conversation_id", "session_id"])
def test_generic_session_headers_do_not_change_the_body_holdout_arm(
    monkeypatch, header_name: str
) -> None:
    """Generic session identifiers must not change which requests receive treatment."""
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0.3")
    from headroom.proxy.output_savings import assign_arm

    opener = {"model": "gpt-5.5", "input": _user("fix the failing test"), "stream": False}
    # A header value whose key lands in the other arm, so honouring the header
    # would show up as a different arm.
    session = next(
        f"client-session-{i}"
        for i in range(200)
        if assign_arm(f"header:{header_name}:client-session-{i}", 0.3) != _body_arm(opener)
    )

    outcomes = _post_turns([opener], headers={header_name: session})

    assert len(outcomes) == 1
    assert _arms(outcomes[0]) == {_body_arm(opener)}


class _KiroLikeSteering:
    """An ``on_responses_create`` hook that steers as headroom-kiro does: it
    decides the arm on the body it is handed (``x-headroom-session-id``, else
    ``conversation_key_from_body``), drops ``instructions``, and in treatment
    appends the steering block to every user prompt."""

    name = "kiro-like"

    def __init__(self) -> None:
        self.steered: list[bool] = []

    def on_responses_create(self, request: Any) -> dict:
        from headroom.proxy.output_savings import assign_arm, conversation_key_from_body
        from headroom.proxy.output_shaper import output_holdout_fraction
        from headroom.proxy.output_verbosity_policy import (
            replace_or_append_steering_block,
            steering_text,
        )

        session = request.headers.get("x-headroom-session-id")
        key = (
            f"header:x-headroom-session-id:{session}"
            if session
            else conversation_key_from_body(dict(request.body))
        )
        treatment = assign_arm(key, output_holdout_fraction()) == "treatment"
        self.steered.append(treatment)
        body = {k: v for k, v in request.body.items() if k != "instructions"}
        if treatment:
            block = steering_text(3)
            assert block is not None
            body["input"] = [
                {**item, "content": replace_or_append_steering_block(item["content"], block)[0]}
                if item.get("role") == "user"
                else item
                for item in body["input"]
            ]
        return body


@pytest.mark.parametrize("arm", ["control", "treatment"])
@pytest.mark.parametrize("instructions", [False, True])
def test_create_hook_steering_and_core_agree_on_the_arm(
    monkeypatch, arm: str, instructions: bool
) -> None:
    """The hook steers before core assigns its arm. Core keyed the steered body
    (and, with ``instructions`` dropped, a different field), so a conversation
    the hook steered could be booked as control and the reverse."""
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0.3")
    hook = _KiroLikeSteering()
    register_turn_hook(hook)
    bodies = []
    # Thirty conversations, enough that some (``task 17``) changed arm when
    # their steered body was keyed; the arm is picked on the unsteered body.
    for i in range(30):
        body: dict = {"model": "gpt-5.5", "input": _user(f"task {i}"), "stream": False}
        if instructions:
            body["instructions"] = f"You are agent {i}."
        if _body_arm(body) == arm:
            bodies.append(body)

    outcomes = _post_turns(bodies)

    assert hook.steered == [arm == "treatment"] * len(bodies)
    assert [_arms(o) for o in outcomes] == [{arm}] * len(bodies)
    for outcome in outcomes:
        labels = list(outcome.transforms_applied)
        assert ("output_shaper:verbosity:L3" in labels) is (arm == "treatment"), labels


_SHARED_INSTRUCTIONS = "You are a coding assistant."


def _shared_instructions_body(i: int) -> dict:
    return {
        "model": "gpt-5.5",
        "instructions": _SHARED_INSTRUCTIONS,
        "input": _user(f"task {i}"),
        "stream": False,
    }


def test_conversations_sharing_instructions_keep_their_own_holdout_keys() -> None:
    """The key used ``instructions`` as a conversation id, so every conversation
    of a client that sends one system prompt shared one key, and one arm."""
    from headroom.proxy.output_arm_key import output_arm_key
    from headroom.proxy.output_savings import assign_arm

    keys = [output_arm_key(_shared_instructions_body(i), {}) for i in range(200)]

    assert len(set(keys)) == 200
    assert {assign_arm(key, 0.3) for key in keys} == {"control", "treatment"}


def test_create_hook_and_core_agree_when_conversations_share_instructions(monkeypatch) -> None:
    """Same model, same instructions: the hook and core must still split the
    conversations across both arms and agree on each one."""
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0.3")
    hook = _KiroLikeSteering()
    register_turn_hook(hook)
    bodies = [_shared_instructions_body(i) for i in range(30)]
    expected = [_body_arm(body) for body in bodies]
    assert set(expected) == {"control", "treatment"}

    outcomes = _post_turns(bodies)

    assert hook.steered == [arm == "treatment" for arm in expected]
    assert [_arms(o) for o in outcomes] == [{arm} for arm in expected]
    for arm, outcome in zip(expected, outcomes, strict=True):
        labels = list(outcome.transforms_applied)
        assert ("output_shaper:verbosity:L3" in labels) is (arm == "treatment"), labels


def test_responses_conversations_without_an_id_get_their_own_holdout_keys() -> None:
    """Without a response id the key used a constant, so every Responses
    conversation of a model shared one holdout arm."""
    from headroom.proxy.output_savings import conversation_key_from_body

    def key(text: str) -> str:
        return conversation_key_from_body(
            {"model": "gpt-5.5", "input": [{"role": "user", "content": text}]}
        )

    assert key("list the files") != key("fix the failing test")
    assert key("list the files") == key("list the files")


def test_a_steering_block_in_the_opener_does_not_move_its_holdout_key() -> None:
    """headroom-kiro appends the block to the user turn; the key, and so the
    arm, must read the same before and after."""
    from headroom.proxy.output_savings import conversation_key_from_body
    from headroom.proxy.output_verbosity_policy import steering_text

    def key(text: str) -> str:
        return conversation_key_from_body(
            {"model": "gpt-5.5", "input": [{"role": "user", "content": text}]}
        )

    assert key(f"task 17\n\n{steering_text(3)}") == key("task 17")


def test_bypassed_request_with_null_messages_reaches_the_provider() -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(ProxyConfig(optimize=True, cache_enabled=False, rate_limit_enabled=False))
    app.dependency_overrides[require_loopback] = lambda: None
    with TestClient(app) as client:

        async def fake_retry(method, url, headers, body, stream=False, **kwargs):
            return httpx.Response(
                200,
                json={"id": "r", "object": "response", "status": "completed", "output": []},
                request=httpx.Request("POST", url),
            )

        app.state.proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/responses",
            content=json.dumps(
                {"model": "gpt-5.5", "input": "hi", "messages": None, "stream": False}
            ).encode(),
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
                "x-headroom-bypass": "true",
            },
        )
    assert resp.status_code == 200, resp.text
