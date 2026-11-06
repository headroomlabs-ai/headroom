"""T036–T040: real process, verified TLS and client/upstream barriers."""

import threading
import time

import httpx
import pytest

from tests.unified_gateway.process.http_harness import frame, http_process, stream_headers

CHAT_FIRST = 'data: {"choices":[{"index":0,"delta":{"content":"雪","tool_calls":[{"function":{"arguments":"{}"}}]},"finish_reason":null}]}\r\n\r\n'.encode()
CHAT_END = b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\r\n\r\ndata: [DONE]\r\n\r\n'


@pytest.mark.parametrize("translated", [False, True])
def test_first_event_precedes_upstream_completion_barrier(local_pki, tmp_path, translated):  # noqa: F811
    release = threading.Event()
    finished = threading.Event()
    first = (
        b'data: {"type":"message_start","message":{"content":[]}}\r\n\r\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\r\n\r\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\r\n\r\n'
        if translated
        else CHAT_FIRST
    )
    final = (
        b'data: {"type":"content_block_stop","index":0}\r\n\r\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\r\n\r\ndata: {"type":"message_stop"}\r\n\r\n'
        if translated
        else CHAT_END
    )

    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, first)
        assert release.wait(10)
        frame(upstream, final)
        finished.set()

    def configure(raw):
        if translated:
            route = raw["routes"][0]
            route.update(
                native_protocols=["anthropic-messages"],
                ingress_protocols=["openai-chat"],
                translation="qualified",
                capabilities={"openai-chat": {"http-stream": {"features": ["text"]}}},
            )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        try:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "fixture-model", "messages": [], "stream": True, "max_tokens": 8},
            ) as response:
                assert response.status_code == 200
                iterator = response.iter_raw()
                initial = next(iterator)
                if translated:
                    while b"hello" not in initial:
                        initial += next(iterator)
                assert b"hello" in initial if translated else CHAT_FIRST == initial
                assert not finished.is_set()
                release.set()
                remainder = b"".join(iterator)
                assert b"[DONE]" in remainder
            probe = client.get("/__test/idle").json()
            assert probe["totals"]["logical_requests"] == 1
            assert probe["totals"]["attempts"] == 1
            assert len(calls) == 1
            assert probe["active"] == 0
            if translated:
                assert probe["ledger"]["known_micro_usd"] == 4
                assert probe["ledger"]["unknown_charge_count"] == 1
            else:
                assert probe["ledger"]["known_micro_usd"] == 7
        finally:
            release.set()


@pytest.mark.parametrize(
    "protocol",
    [
        "openai-chat",
        "openai-responses",
        "anthropic-messages",
        "gemini-generate",
        "vertex-generate",
        "anthropic-to-chat",
    ],
)
def test_fragmented_unicode_tool_events_cross_real_sockets(local_pki, tmp_path, protocol):  # noqa: F811
    import json

    import anthropic
    import openai
    from google.genai import _api_client, types

    from tests.unified_gateway.process import protocol_fixtures as fixtures

    translated = protocol == "anthropic-to-chat"
    wire = {
        "openai-chat": fixtures.CHAT,
        "openai-responses": fixtures.RESPONSES,
        "anthropic-messages": fixtures.ANTHROPIC,
        "gemini-generate": fixtures.GEMINI,
        "vertex-generate": fixtures.GEMINI,
        "anthropic-to-chat": fixtures.ANTHROPIC_TEXT,
    }[protocol]
    path = {
        "openai-chat": "/v1/chat/completions",
        "openai-responses": "/v1/responses",
        "anthropic-messages": "/v1/messages",
        "gemini-generate": "/v1beta/models/fixture-model:streamGenerateContent",
        "vertex-generate": "/v1/projects/p/locations/r/publishers/google/models/fixture-model:streamGenerateContent",
        "anthropic-to-chat": "/v1/chat/completions",
    }[protocol]
    body = {"model": "fixture-model", "messages": [], "stream": True, "max_tokens": 10}
    if protocol in {"gemini-generate", "vertex-generate"}:
        body = {"contents": [], "generationConfig": {"candidateCount": 2}}
    elif protocol == "openai-responses":
        body = {"model": "fixture-model", "input": "hello", "stream": True}

    sent = 0

    def handler(upstream):
        nonlocal sent
        sent += 1
        assert upstream.path == ("/v1/messages" if translated else path)
        stream_headers(upstream)
        current_wire = wire.replace(b"resp_1", f"resp_{sent}".encode())
        for byte in current_wire:
            frame(upstream, bytes([byte]))

    def configure(raw):
        ingress = "openai-chat" if translated else protocol
        if protocol == "gemini-generate":
            raw["routes"][0]["upstream_path_prefix"] = "/v1beta/"
            raw["credentials"][0]["allowed_path_prefixes"] = ["/v1beta/"]
        raw["routes"][0].update(
            ingress_protocols=[ingress],
            native_protocols=["anthropic-messages" if translated else protocol],
            translation="qualified" if translated else "disabled",
            capabilities={ingress: {"http-stream": {"features": ["text"]}}},
        )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, _, _, _):
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        if not translated:
            assert response.content == wire
        # A second real-socket request is decoded by each provider's maintained
        # SDK framing/model parser, independently from the gateway observer.
        with client.stream("POST", path, json=body) as response:
            if protocol in {"gemini-generate", "vertex-generate"}:
                events = [
                    types.GenerateContentResponse.model_validate(item)
                    for item in _api_client.HttpResponse(
                        response.headers, response_stream=response
                    ).segments()
                ]
                assert events[0].candidates[0].content.parts[0].text == "雪"
                assert events[0].candidates[1].content.parts[0].function_call.args == {"city": "雪"}
                assert events[2].candidates[0].content.parts[0].text == "tail"
                assert events[-1].usage_metadata.candidates_token_count == 2
            elif protocol == "anthropic-messages":
                with anthropic.Anthropic(api_key="parser-only") as sdk:
                    events = list(
                        anthropic.Stream(
                            cast_to=anthropic.types.RawMessageStreamEvent,
                            response=response,
                            client=sdk,
                        )
                    )
                assert events[2].delta.text == "雪"
                assert json.loads(events[5].delta.partial_json) == {"city": "雪"}
                assert events[-2].usage.output_tokens == 2
                assert events[-1].type == "message_stop"
            else:
                cast_to = (
                    openai.types.responses.ResponseStreamEvent
                    if protocol == "openai-responses"
                    else openai.types.chat.ChatCompletionChunk
                )
                with openai.OpenAI(api_key="parser-only") as sdk:
                    events = list(openai.Stream(cast_to=cast_to, response=response, client=sdk))
                if protocol == "openai-responses":
                    assert events[0].delta == "雪"
                    assert json.loads(events[1].delta) == {"city": "雪"}
                    assert events[-1].response.usage.output_tokens == 2
                else:
                    assert (
                        "".join(
                            choice.delta.content or ""
                            for event in events
                            for choice in event.choices
                        )
                        == "雪"
                    )
                    assert any(choice.finish_reason for event in events for choice in event.choices)
                    if not translated:
                        assert json.loads(
                            events[0].choices[0].delta.tool_calls[0].function.arguments
                        ) == {"city": "雪"}
                        assert events[-1].usage.completion_tokens == 2
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 14
        assert probe["ledger"]["unknown_charge_count"] == 0
        assert all(operation["terminal"] == "success" for operation in probe["operations"])


@pytest.mark.parametrize(
    "tail", [b"", b"data: {", b"data: \xe9", b"data: {invalid}\n\n", b"data: " + b"x" * 1100]
)
def test_eof_without_terminal_is_failed(local_pki, tmp_path, tail):  # noqa: F811
    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, CHAT_FIRST + tail)

    with http_process(
        local_pki,
        tmp_path,
        handler,
        configure=lambda raw: raw["limits"].update(max_frame_bytes=1024),
    ) as (client, calls, output, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], "stream": True}
        )
        assert b'"code":"gateway_upstream_error"' in response.content
        assert b"[DONE]" not in response.content
        probe = client.get("/__test/idle").json()
        assert probe["active"] == 0
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30
        assert len(calls) == 1
        assert "fixture-key" not in "".join(output)


@pytest.mark.parametrize(
    "prefix,absolute", [(b": ping\n\n", False), (b"data: {", False), (b": ping\n\n", True)]
)
def test_heartbeat_and_partial_line_deadlines_fire_without_new_bytes(
    local_pki, tmp_path, prefix, absolute
):  # noqa: F811
    closed = threading.Event()

    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, prefix)
        upstream.connection.settimeout(4)
        assert upstream.connection.recv(1) == b""
        closed.set()

    def configure(raw):
        if absolute:
            raw["limits"].update(
                request_deadline_seconds=0.3,
                stream_content_idle_seconds=10,
                partial_frame_seconds=10,
            )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, _, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], "stream": True}
        )
        assert b'"code":"gateway_upstream_error"' in response.content
        assert closed.wait(5)
        # An aborted SSE retires its downstream keepalive connection. Probe the
        # lifecycle on an independent socket, not that just-aborted connection.
        with httpx.Client(base_url=client.base_url, headers=client.headers, timeout=5) as probe:
            assert probe.get("/__test/idle").json()["active"] == 0


@pytest.mark.parametrize("expose", [False, True])
def test_disconnect_cancels_upstream_and_finalizes_once(local_pki, tmp_path, expose):  # noqa: F811
    closed = threading.Event()

    def handler(upstream):
        stream_headers(upstream)
        if expose:
            frame(upstream, CHAT_FIRST)
        upstream.connection.settimeout(2)
        try:
            assert upstream.connection.recv(1) == b""
        except ConnectionResetError:
            pass
        closed.set()

    def configure(raw):
        raw["limits"].update(
            request_deadline_seconds=60,
            stream_content_idle_seconds=60,
            partial_frame_seconds=60,
            shutdown_cleanup_seconds=1,
        )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "fixture-model", "messages": [], "stream": True},
        ) as response:
            if expose:
                assert next(response.iter_raw()) == CHAT_FIRST
            disconnected_at = time.monotonic()
        assert closed.wait(1), "upstream did not close on client disconnect"
        assert time.monotonic() - disconnected_at < 1
        probe = client.get("/__test/idle").json()
        assert probe["active"] == 0
        assert probe["queued"] == 0
        assert probe["totals"]["attempts"] == len(calls) == 1
        assert probe["totals"]["logical_requests"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30
        assert probe["ledger"]["unknown_charge_count"] == 1
