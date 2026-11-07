"""Gemini ``generateContent`` and Bedrock Converse bodies on ``/v1/compress``.

What matters is what must NOT change: every part the provider needs back
(``thoughtSignature``, ``functionCall``, ``toolUse``, ``cachePoint``) comes back
byte-identical, and the body comes back in the shape it arrived in.
"""

from __future__ import annotations

import copy
import json
from unittest.mock import AsyncMock

import pytest

from headroom.proxy.gateway_native import apply_view, attach_view, native_shape
from tests.gateway.samples import anthropic_tool_history

GREP = "\n".join(
    f"headroom/proxy/handlers/openai.py:{i}:    value_{i} = compute(value_{i})"
    for i in range(1, 400)
)
SIGNATURE = "CtYBAdHtim9vDGb1wQ5opaque"


def _compress(client, body, gateway=None, **config):
    sent = {
        **copy.deepcopy(body),
        "gateway": gateway or {},
        "config": {"mode": "lossless", **config},
    }
    resp = client.post("/v1/compress", json=sent)
    assert resp.status_code == 200, resp.text
    return resp.json()


def gemini_body() -> dict:
    return {
        "model": "gemini-2.5-pro",
        "systemInstruction": {"parts": [{"text": "You are a coding agent."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Find the values."}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "grep", "args": {"pattern": "value_"}},
                        "thoughtSignature": SIGNATURE,
                    }
                ],
            },
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": "grep", "response": {"output": GREP}}}],
            },
        ],
        "tools": [{"functionDeclarations": [{"name": "grep", "parameters": {"type": "object"}}]}],
        "generationConfig": {"temperature": 0},
    }


def converse_body() -> dict:
    rows = [{"id": i, "name": f"item {i}", "status": "active"} for i in range(50)]
    return {
        "model": "bedrock/anthropic.claude-sonnet-4-5",
        "system": [{"text": "You are a coding agent."}, {"cachePoint": {"type": "default"}}],
        "messages": [
            {"role": "user", "content": [{"text": "Find the values."}]},
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "tu_1", "name": "grep", "input": {"p": "value_"}}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tu_1",
                            "content": [{"text": GREP}, {"json": {"rows": rows}}],
                            "status": "success",
                        }
                    },
                    {"cachePoint": {"type": "default"}},
                ],
            },
        ],
        "inferenceConfig": {"maxTokens": 1024},
        "toolConfig": {"tools": [{"toolSpec": {"name": "grep", "inputSchema": {"json": {}}}}]},
    }


def test_a_gemini_body_comes_back_as_gemini_with_its_tool_output_folded(headroom_client):
    sent = gemini_body()
    data = _compress(headroom_client, sent)
    body = data["body"]

    assert "messages" not in body
    assert {k: v for k, v in body.items() if k != "contents"} == {
        k: v for k, v in sent.items() if k != "contents"
    }
    assert body["contents"][:2] == sent["contents"][:2]
    folded = body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"]
    assert len(folded) < len(GREP)
    assert all(f"value_{i} = compute(value_{i})" in folded for i in range(1, 400))


def test_a_converse_body_comes_back_as_converse_with_its_tool_output_folded(headroom_client):
    sent = converse_body()
    data = _compress(headroom_client, sent)
    body = data["body"]

    assert {k: v for k, v in body.items() if k != "messages"} == {
        k: v for k, v in sent.items() if k != "messages"
    }
    assert body["messages"][:2] == sent["messages"][:2]
    result_block, cache_point = body["messages"][2]["content"]
    assert cache_point == {"cachePoint": {"type": "default"}}
    result = result_block["toolResult"]
    assert result["toolUseId"] == "tu_1" and result["status"] == "success"
    folded, rows = result["content"]
    assert len(folded["text"]) < len(GREP)
    assert all(f"value_{i} = compute(value_{i})" in folded["text"] for i in range(1, 400))
    assert isinstance(rows["json"], dict)
    json.dumps(rows["json"])
    assert data["obligations"] == []


@pytest.mark.parametrize(
    "messages",
    [
        anthropic_tool_history(),
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        # Untyped parts, but `system` is not a Converse role.
        [
            {"role": "system", "content": [{"text": "You are terse."}]},
            {"role": "user", "content": [{"text": "hi"}]},
        ],
    ],
    ids=["anthropic", "openai-typed", "openai-untyped-system"],
)
def test_a_messages_body_is_not_mistaken_for_converse(headroom_client, messages):
    sent = {"model": "claude-sonnet-4-5", "messages": messages}
    assert native_shape(sent) is None
    data = _compress(headroom_client, sent)

    assert [m["role"] for m in data["body"]["messages"]] == [m["role"] for m in messages]
    assert data["messages"] == data["body"]["messages"]


def test_a_client_supplied_view_marker_is_ignored(headroom_client):
    sent = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "_headroom_native_view": {"shape": "gemini"},
    }
    data = _compress(headroom_client, sent)

    assert data["body"]["messages"] == sent["messages"]


def test_without_a_gateway_block_native_bodies_are_handled_as_before(headroom_client):
    converse = converse_body()
    resp = headroom_client.post("/v1/compress", json=converse)
    assert resp.status_code == 200, resp.text
    returned = resp.json()["messages"]
    assert [m["role"] for m in returned] == ["user", "assistant", "user"]
    assert returned[1] == converse["messages"][1]

    resp = headroom_client.post("/v1/compress", json=gemini_body())
    assert resp.status_code == 400


def test_counts_the_caller_sends_are_in_native_messages(headroom_client):
    sent = converse_body()
    sent["system"] = [{"text": f"Rule {i}."} for i in range(3)]
    data = _compress(headroom_client, sent, session_id="native-frozen", frozen_message_count=3)

    assert data["body"]["messages"] == sent["messages"]
    assert data["session"]["frozen_message_count"] == 3


def test_native_turns_are_never_asked_to_redrive(headroom_client):
    gateway = {"can_redrive": True, "session_affinity": True, "can_relay_response": True}
    data = _compress(headroom_client, converse_body(), gateway=gateway, mode="ccr")

    assert "redrive" not in data["obligations"]
    assert "relay_usage" in data["obligations"]


def test_a_converse_fail_open_answer_keeps_its_own_messages(headroom_client, monkeypatch):
    proxy = headroom_client.app.state.proxy
    monkeypatch.setattr(proxy, "_run_compression_in_executor", AsyncMock(side_effect=TimeoutError))
    sent = converse_body()
    data = _compress(headroom_client, sent)

    assert data["body"]["messages"] == sent["messages"]
    assert data["messages"] == sent["messages"]


def _view_of(body):
    body = {**copy.deepcopy(body), "gateway": {}}
    assert attach_view(body)
    marker = body.pop("_headroom_native_view")
    view = body.pop("messages")
    body.pop("gateway")
    return body, marker, view


@pytest.mark.parametrize("rewritten", ["not json", "[1, 2]"])
def test_a_json_slot_is_only_written_back_as_the_same_json_type(rewritten):
    body, marker, view = _view_of(converse_body())
    json_slot = next(
        i for i, m in enumerate(view) if m["role"] == "tool" and m["content"].startswith("{")
    )
    view[json_slot] = {**view[json_slot], "content": rewritten}

    out, aligned = apply_view(body, marker, view)

    assert aligned
    assert out["messages"] == converse_body()["messages"]


def test_a_view_that_does_not_line_up_forwards_the_original():
    body, marker, view = _view_of(converse_body())

    out, aligned = apply_view(body, marker, [*view, {"role": "user", "content": "extra"}])

    assert not aligned
    assert out["messages"] == converse_body()["messages"]


def _gemini_read(response_id=None):
    body = gemini_body()
    body["contents"][1]["parts"][0]["functionCall"]["name"] = "Read"
    response = body["contents"][2]["parts"][0]["functionResponse"]
    response["name"] = "Read"
    if response_id:
        response["id"] = response_id
    return body


def _converse_read():
    body = converse_body()
    body["messages"][1]["content"][0]["toolUse"]["name"] = "Read"
    return body


@pytest.mark.parametrize(
    "sent",
    [_gemini_read(), _gemini_read(response_id="resp_1"), _converse_read()],
    ids=["gemini", "gemini-response-id-only", "converse"],
)
def test_a_read_result_keeps_its_bytes_because_the_tool_name_is_seen(headroom_client, sent):
    data = _compress(headroom_client, sent)

    native = "contents" if "contents" in sent else "messages"
    assert data["body"][native] == sent[native]
