"""Gemini ``generateContent`` and Bedrock Converse bodies on ``/v1/compress``.

What matters is what must NOT change: every part the provider needs back
(``thoughtSignature``, ``functionCall``, ``toolUse``, ``cachePoint``) comes back
byte-identical, and the body comes back in the shape it arrived in.
"""

from __future__ import annotations

import copy
import json

from tests.gateway.samples import anthropic_tool_history

GREP = "\n".join(
    f"headroom/proxy/handlers/openai.py:{i}:    value_{i} = compute(value_{i})"
    for i in range(1, 400)
)
SIGNATURE = "CtYBAdHtim9vDGb1wQ5opaque"


def _compress(client, body):
    sent = {**copy.deepcopy(body), "gateway": {}, "config": {"mode": "lossless"}}
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


def test_an_anthropic_body_is_not_mistaken_for_converse(headroom_client):
    sent = {"model": "claude-sonnet-4-5", "messages": anthropic_tool_history()}
    data = _compress(headroom_client, sent)

    assert all(
        isinstance(block, dict) and "type" in block
        for message in data["body"]["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
    )
    assert len(data["body"]["messages"]) == len(sent["messages"])


def test_a_gemini_read_result_keeps_its_bytes_because_the_tool_name_is_seen(headroom_client):
    sent = gemini_body()
    sent["contents"][1]["parts"][0]["functionCall"]["name"] = "Read"
    sent["contents"][2]["parts"][0]["functionResponse"]["name"] = "Read"
    data = _compress(headroom_client, sent)

    assert data["body"]["contents"] == sent["contents"]
