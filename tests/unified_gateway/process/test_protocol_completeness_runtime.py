"""Local TLS process, SDK parser and byte-oracle protocol qualification."""

import json
import threading

import pytest

from tests.unified_gateway.process.http_harness import (
    frame,
    http_process,
    json_response,
    stream_headers,
)


@pytest.mark.parametrize(
    "protocol,path,body,wire",
    [
        (
            "openai-chat",
            "/v1/chat/completions",
            {"model": "fixture-model", "messages": []},
            b'{ "choices":[{"message":{"role":"assistant","content":"partial"},"finish_reason":"length"}] }',
        ),
        (
            "anthropic-messages",
            "/v1/messages",
            {"model": "fixture-model", "messages": [], "max_tokens": 8},
            b'{ "type":"message","content":[{"type":"text","text":"partial"}],"stop_reason":"max_tokens" }',
        ),
        (
            "gemini-generate",
            "/v1beta/models/fixture-model:generateContent",
            {"contents": []},
            b'{ "candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}] }',
        ),
    ],
)
def test_native_valid_finish_and_refusal_preserve_success_bytes(
    local_pki, tmp_path, protocol, path, body, wire
):
    def handler(upstream):
        json_response(upstream, wire)

    def configure(raw):
        if protocol == "gemini-generate":
            raw["routes"][0]["upstream_path_prefix"] = "/v1beta/"
            raw["credentials"][0]["allowed_path_prefixes"] = ["/v1beta/"]

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        response = client.post(path, json=body)
        assert response.status_code == 200
        assert response.content == wire
        assert len(calls) == 1


@pytest.mark.parametrize("ingress", ["anthropic-messages", "gemini-generate"])
def test_translated_reverse_stream_reaches_sdk_before_upstream_completion(
    local_pki, tmp_path, ingress
):
    import anthropic
    from google.genai import _api_client, types

    release = threading.Event()
    completed = threading.Event()
    first = 'data: {"id":"chat-1","choices":[{"index":0,"delta":{"role":"assistant","content":"雪"},"finish_reason":null}]}\n\n'.encode()
    tail = b'data: {"id":"chat-1","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\ndata: [DONE]\n\n'

    def handler(upstream):
        assert upstream.path == "/v1/chat/completions"
        sent = json.loads(upstream.body)
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}
        stream_headers(upstream)
        for byte in first:
            frame(upstream, bytes([byte]))
        assert release.wait(8)
        frame(upstream, tail)
        completed.set()

    def configure(raw):
        raw["routes"][0].update(
            native_protocols=["openai-chat"],
            ingress_protocols=[ingress],
            translation="qualified",
            capabilities={ingress: {"http-stream": {"features": ["text", "inline_images"]}}},
        )

    path, body = (
        (
            "/v1/messages",
            {
                "model": "fixture-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 8,
            },
        )
        if ingress == "anthropic-messages"
        else (
            "/v1beta/models/fixture-model:streamGenerateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        )
    )
    try:
        with http_process(local_pki, tmp_path, handler, configure=configure) as (
            client,
            calls,
            _,
            _,
        ):
            with client.stream("POST", path, json=body) as response:
                assert response.status_code == 200
                if ingress == "gemini-generate":
                    events = (
                        types.GenerateContentResponse.model_validate(item)
                        for item in _api_client.HttpResponse(
                            response.headers, response_stream=response
                        ).segments()
                    )
                    assert next(events).candidates[0].content.parts[0].text == "雪"
                    assert not completed.is_set()
                    release.set()
                    rest = list(events)
                    assert rest[0].candidates[0].finish_reason == "MAX_TOKENS"
                    assert rest[-1].usage_metadata.total_token_count == 7
                else:
                    with anthropic.Anthropic(api_key="parser-only") as sdk:
                        events = iter(
                            anthropic.Stream(
                                cast_to=anthropic.types.RawMessageStreamEvent,
                                response=response,
                                client=sdk,
                            )
                        )
                        assert next(events).type == "message_start"
                        assert next(events).type == "content_block_start"
                        assert next(events).delta.text == "雪"
                        assert not completed.is_set()
                        release.set()
                        rest = list(events)
                        assert rest[-2].delta.stop_reason == "max_tokens"
                        assert rest[-2].usage.output_tokens == 2
                        assert rest[-1].type == "message_stop"
            assert len(calls) == 1
    finally:
        release.set()


def test_gemini_native_stream_preserves_method_and_configured_prefix(local_pki, tmp_path):
    wire = b'data: {"candidates":[{"index":0,"finishReason":"STOP"}]}\n\n'

    def configure(raw):
        raw["routes"][0]["upstream_path_prefix"] = "/private/google/"
        raw["credentials"][0]["allowed_path_prefixes"] = ["/private/google/"]

    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, wire)

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        response = client.post(
            "/v1beta/models/fixture-model:streamGenerateContent?alt=sse", json={"contents": []}
        )
        assert response.status_code == 200
        assert response.content == wire
        assert calls[0][0] == "/private/google/models/fixture-model:streamGenerateContent?alt=sse"


@pytest.mark.parametrize(
    "protocol,path,body,reply",
    [
        (
            "openai-chat",
            "/v1/chat/completions",
            b'{ "model":"fixture-model", "messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"lookup","arguments":"{}"}}]}], "unknown":1.00 }',
            b'{ "choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}], "opaque":1.00 }',
        ),
        (
            "openai-responses",
            "/v1/responses",
            b'{ "model":"fixture-model", "input":[{"type":"reasoning","encrypted_content":"SIGNED\\u0061","summary":[]}], "unknown":1.00 }',
            b'{ "id":"resp_native_signed", "status":"completed", "output":[], "opaque":1.00 }',
        ),
        (
            "anthropic-messages",
            "/v1/messages",
            b'{ "model":"fixture-model", "max_tokens":8, "messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"private","signature":"SIGNED\\u0061"},{"type":"text","text":"ok"}]}], "unknown":1.00 }',
            b'{ "type":"message", "content":[{"type":"thinking","thinking":"private","signature":"SIGNED\\u0061"},{"type":"text","text":"ok"}], "stop_reason":"end_turn", "opaque":1.00 }',
        ),
        (
            "gemini-generate",
            "/v1beta/models/fixture-model:generateContent",
            b'{ "contents":[{"role":"model","parts":[{"text":"private","thoughtSignature":"SIGNED\\u0061"}]}], "unknown":1.00 }',
            b'{ "candidates":[{"content":{"role":"model","parts":[{"text":"ok","thoughtSignature":"SIGNED\\u0061"}]},"finishReason":"STOP"}], "opaque":1.00 }',
        ),
    ],
    ids=[
        "chat-parallel-tools",
        "responses-encrypted-state",
        "anthropic-signed-thinking",
        "gemini-signed-thought",
    ],
)
def test_strict_native_signed_and_tool_entities_are_byte_exact_over_tls(
    local_pki, tmp_path, protocol, path, body, reply
):
    def configure(raw):
        raw["routes"][0]["capabilities"][protocol]["http-json"]["features"] = [
            "text",
            "tools",
            "parallel_tools",
            "signed_state",
        ]
        if protocol == "gemini-generate":
            raw["routes"][0]["upstream_path_prefix"] = "/v1beta/"
            raw["credentials"][0]["allowed_path_prefixes"] = ["/v1beta/"]

    def handler(upstream):
        json_response(upstream, reply)

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        response = client.post(path, content=body, headers={"content-type": "application/json"})
        assert response.status_code == 200
        assert response.content == reply
        assert calls[0][1] == body
