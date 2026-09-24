"""Protocol-specific semantic features are rejected before identity acquisition."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.capabilities import requested_features
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLE = (
    Path(__file__).parents[2] / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
)


@pytest.mark.parametrize(
    ("protocol", "payload", "features"),
    [
        (
            "openai-responses",
            {"input": [{"type": "function_call_output", "call_id": "call-a", "output": "value"}]},
            ["text"],
        ),
        (
            "openai-responses",
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call-a",
                        "name": "tool",
                        "arguments": "{}",
                    }
                ]
            },
            ["text"],
        ),
        (
            "openai-chat",
            {"messages": [{"role": "tool", "tool_call_id": "call-a", "content": "value"}]},
            ["text"],
        ),
        (
            "anthropic-messages",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call-a", "content": "value"}
                        ],
                    }
                ]
            },
            ["text"],
        ),
        ("gemini-generate", {"generationConfig": {"responseSchema": {"type": "OBJECT"}}}, ["text"]),
        (
            "gemini-generate",
            {"generationConfig": {"responseJsonSchema": {"type": "object"}}},
            ["text"],
        ),
        (
            "gemini-generate",
            {
                "contents": [
                    {"parts": [{"functionResponse": {"name": "tool", "response": {"ok": True}}}]}
                ]
            },
            ["text"],
        ),
        (
            "gemini-generate",
            {"contents": [{"parts": [{"inlineData": {"mimeType": "audio/wav", "data": "AAAA"}}]}]},
            ["text", "inline_images"],
        ),
        ("gemini-generate", {"tools": [{"googleSearch": {}}]}, ["text", "tools"]),
        ("gemini-generate", {"tools": [{"codeExecution": {}}]}, ["text", "tools"]),
        pytest.param(
            "openai-responses",
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call-a",
                        "output": [
                            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                        ],
                    }
                ]
            },
            ["text", "tools"],
            id="response-tool-output-image",
        ),
        pytest.param(
            "openai-responses",
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call-a",
                        "output": [{"type": "input_file", "file_id": "file-a"}],
                    }
                ]
            },
            ["text", "tools"],
            id="response-tool-output-file",
        ),
    ],
)
def test_protocol_feature_rejected_before_identity_and_dispatch(
    monkeypatch, protocol, payload, features
):
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    raw = json.loads(EXAMPLE.read_text())
    route = next(r for r in raw["routes"] if protocol in r["ingress_protocols"])
    route["capabilities"][protocol] = {"http-json": {"features": features}}
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(raw)))
    calls = {"identity": 0, "upstream": 0}

    async def acquire(*args, **kwargs):
        calls["identity"] += 1
        raise AssertionError("capability denial reached identity")

    async def upstream(request):
        calls["upstream"] += 1
        return httpx.Response(200, json={})

    app.state.gateway_runtime.broker.acquire = acquire
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    path = {
        "openai-chat": "/v1/chat/completions",
        "openai-responses": "/v1/responses",
        "anthropic-messages": "/v1/messages",
        "gemini-generate": f"/v1beta/models/{route['public_model']}:generateContent",
    }[protocol]
    client = TestClient(app)
    headers = {"host": "127.0.0.1", "authorization": "Bearer client-secret"}
    assert client.get(f"/v1/models/{route['public_model']}", headers=headers).status_code == 200
    response = client.post(path, headers=headers, json={"model": route["public_model"], **payload})
    assert response.status_code == 400
    error = response.json()["error"]
    if protocol == "gemini-generate":
        assert error["code"] == 400
        assert error["status"] == "INVALID_ARGUMENT"
        assert error["details"] == [{"reason": "gateway_unsupported_capability"}]
    else:
        assert error["code"] == "gateway_unsupported_capability"
    assert calls == {"identity": 0, "upstream": 0}


@pytest.mark.parametrize(
    ("output", "feature"),
    [
        ({"type": "input_image", "image_url": "data:image/png;base64,AAAA"}, "inline_images"),
        ({"type": "input_file", "file_id": "file-a"}, "unsupported_media"),
    ],
    ids=["image", "file"],
)
def test_response_tool_output_content_classification(output, feature):
    payload = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call-a",
                "output": [{"type": "input_text", "text": "Tool result"}, output],
            }
        ]
    }
    assert requested_features("openai-responses", payload) == {"text", "tools", feature}


def test_response_tool_arguments_and_schema_are_not_content():
    arbitrary_data = {"output": [{"type": "input_file", "file_id": "not-a-real-file"}]}
    payload = {
        "tools": [
            {
                "type": "function",
                "name": "tool",
                "parameters": {"type": "object", "examples": [arbitrary_data]},
            }
        ],
        "input": [
            {
                "type": "function_call",
                "call_id": "call-a",
                "name": "tool",
                "arguments": json.dumps(arbitrary_data),
            },
            {
                "type": "function_call_output",
                "call_id": "call-a",
                "output": json.dumps(arbitrary_data),
            },
        ],
        "metadata": arbitrary_data,
    }
    assert requested_features("openai-responses", payload) == {"text", "tools"}
