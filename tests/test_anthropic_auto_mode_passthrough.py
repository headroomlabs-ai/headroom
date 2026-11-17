from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.anthropic_wire import (
    AnthropicSSEEnvelope,
    build_anthropic_upstream_url,
    has_dangerous_tool_use_beta,
    is_safeguard_capable_request,
    strip_safeguard_payload,
)
from headroom.proxy.handlers.streaming import StreamingMixin
from headroom.proxy.server import ProxyConfig, create_app

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_native_url_preserves_path_and_raw_query() -> None:
    assert (
        build_anthropic_upstream_url(
            "https://api.anthropic.com", "/v1/messages", "beta=true&beta=false"
        )
        == "https://api.anthropic.com/v1/messages?beta=true&beta=false"
    )


def test_classifier_request_detection_does_not_need_safeguard_schema() -> None:
    request = json.loads(_fixture("claude_code_auto_mode_request.json"))
    assert is_safeguard_capable_request(request, "prompt-caching-2024-07-31")
    assert has_dangerous_tool_use_beta("prompt-caching-2024-07-31,dangerous-tool-use-2026-09")
    assert not is_safeguard_capable_request({}, "prompt-caching-2024-07-31")


def test_sse_round_trip_preserves_opaque_result_order_and_ids() -> None:
    envelope = AnthropicSSEEnvelope.parse(_fixture("claude_code_auto_mode_stream.sse"))
    assert envelope.message["safeguard_results"]["decision"] == "allow"
    assert envelope.message["content"][0]["id"] == "toolu_auto_001"

    rendered = b"".join(envelope.render(envelope.message))
    safeguard_offset = rendered.index(b"event: safeguard_results")
    delta_offset = rendered.index(b"event: message_delta")
    stop_offset = rendered.index(b"event: message_stop")
    assert safeguard_offset < delta_offset < stop_offset
    assert b'"id": "toolu_auto_001"' in rendered


def test_sse_round_trip_keeps_results_across_a_ccr_continuation() -> None:
    envelope = AnthropicSSEEnvelope.parse(_fixture("claude_code_auto_mode_stream.sse"))
    continuation = {
        "id": "msg_continuation_001",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-20260924",
        "content": [{"type": "text", "text": "retrieval complete"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 4},
    }

    rendered = b"".join(envelope.render(continuation))

    assert b"safeguard_results" in rendered
    assert b"toolu_auto_001" in rendered
    assert b"msg_continuation_001" in rendered


def _rendered_events(chunks: list[bytes]) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in b"".join(chunks).decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


def test_nested_extensions_on_known_frames_survive_round_trip() -> None:
    envelope = AnthropicSSEEnvelope.parse(_fixture("claude_code_auto_mode_nested_extensions.sse"))
    events = {event["type"]: event for event in _rendered_events(envelope.render())}

    assert envelope.is_message_reconstructable()
    assert events["message_start"]["x_message_start"] == {"sentinel": "message_start"}
    assert events["content_block_start"]["x_block_start"] == {"sentinel": "block_start"}
    assert events["content_block_delta"]["x_delta_event"] == {"sentinel": "delta_event"}
    assert events["content_block_delta"]["delta"] == {
        "type": "text_delta",
        "text": "Checking.",
        "x_text_delta": {"sentinel": "text_delta"},
    }
    assert events["content_block_stop"]["x_block_stop"] == {"sentinel": "block_stop"}
    assert events["message_delta"]["delta"]["x_message_delta_delta"] == {
        "sentinel": "message_delta_delta"
    }
    assert events["message_delta"]["usage"]["cache_read_input_tokens"] == 7
    assert events["message_delta"]["usage"]["x_usage"] == {"sentinel": "usage"}
    assert events["message_stop"]["x_message_stop"] == {"sentinel": "message_stop"}


def test_block_extensions_do_not_attach_to_replacement_blocks() -> None:
    envelope = AnthropicSSEEnvelope.parse(_fixture("claude_code_auto_mode_nested_extensions.sse"))
    continuation = {
        "id": "msg_continuation_002",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-20260924",
        "content": [{"type": "text", "text": "retrieval complete"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 4},
    }

    rendered = b"".join(envelope.render(continuation))
    events = {event["type"]: event for event in _rendered_events([rendered])}

    # Message-level members belong to the protocol and survive the swap.
    assert events["message_start"]["x_message_start"] == {"sentinel": "message_start"}
    assert "x_message_delta_delta" in events["message_delta"]["delta"]
    assert events["message_stop"]["x_message_stop"] == {"sentinel": "message_stop"}
    # Members of the replaced block's frames do not describe the new block.
    for sentinel in (b"x_block_start", b"x_delta_event", b"x_text_delta", b"x_block_stop"):
        assert sentinel not in rendered
    assert b"retrieval complete" in rendered


def test_unknown_content_delta_is_complete_and_replayed_verbatim() -> None:
    raw = b"""event: message_start
data: {"type":"message_start","message":{"id":"msg_future","type":"message","role":"assistant","content":[]}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"future_delta","value":"opaque"}}

event: message_stop
data: {"type":"message_stop"}

"""

    envelope = AnthropicSSEEnvelope.parse(raw)

    assert envelope.is_complete()
    assert not envelope.is_message_reconstructable()
    assert b'"type":"future_delta"' in b"".join(envelope.render())


def test_malformed_utf8_delta_is_replayed_without_replacement() -> None:
    raw = b"""event: message_start
data: {"type":"message_start","message":{"id":"msg_bad_utf8","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\xff"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}

"""

    envelope = AnthropicSSEEnvelope.parse(raw)
    rendered = b"".join(envelope.render())

    assert envelope.is_complete()
    assert not envelope.is_message_reconstructable()
    assert b"\xff" in rendered
    assert b"\xef\xbf\xbd" not in rendered


_MALFORMED_KNOWN_FRAMES = {
    "invalid-json-message-delta": b"event: message_delta\ndata: {not-json\n\n",
    "invalid-utf8-block-start": (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":"\xff"}}\n\n'
    ),
    "non-object-content-block": (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,"content_block":"future"}\n\n'
    ),
    "delta-for-unopened-block": (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":9,"delta":{"type":"text_delta","text":"x"}}\n\n'
    ),
    "stop-for-unopened-block": (
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":9}\n\n'
    ),
    # ``index`` and ``delta.type`` are provider JSON and may be any JSON value;
    # a non-scalar one must not reach a dictionary lookup.
    "object-index-block-start": (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":{"i":1},"content_block":{"type":"text","text":""}}\n\n'
    ),
    "object-index-delta": (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":{"i":0},"delta":{"type":"text_delta","text":"x"}}\n\n'
    ),
    "array-index-delta": (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":[0],"delta":{"type":"text_delta","text":"x"}}\n\n'
    ),
    "array-delta-type": (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":["text_delta"],"text":"x"}}\n\n'
    ),
    "non-string-delta-text": (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":{"t":"x"}}}\n\n'
    ),
    "object-index-stop": (
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":{"i":0}}\n\n'
    ),
    "array-index-stop": (
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":[0]}\n\n'
    ),
    "array-usage-message-delta": (
        b'event: message_delta\ndata: {"type":"message_delta","delta":{},"usage":[1]}\n\n'
    ),
}


@pytest.mark.parametrize("bad_frame", _MALFORMED_KNOWN_FRAMES.values(), ids=_MALFORMED_KNOWN_FRAMES)
def test_malformed_known_frame_is_replayed_verbatim(bad_frame: bytes) -> None:
    raw = (
        b"""event: message_start
data: {"type":"message_start","message":{"id":"msg_bad_known","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

"""
        + bad_frame
        + b"""event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

"""
    )

    envelope = AnthropicSSEEnvelope.parse(raw)
    rendered = b"".join(envelope.render())

    assert envelope.is_complete()
    assert not envelope.is_message_reconstructable()
    assert rendered.count(bad_frame) == 1
    assert rendered.index(b"event: content_block_stop") < rendered.index(bad_frame)
    assert rendered.index(bad_frame) < rendered.rindex(b"event: message_delta")


@pytest.mark.parametrize(
    "bad_frames",
    [
        pytest.param(
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":{"i":0},"delta":{"type":"text_delta","text":"x"}}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":["text_delta"],"text":"x"}}\n\n',
            id="delta",
        ),
        pytest.param(
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":{"i":0}}\n\n'
            b'event: content_block_stop\ndata: {"type":"content_block_stop","index":[0]}\n\n',
            id="stop",
        ),
    ],
)
def test_invalid_discriminators_on_an_open_block_do_not_break_the_stream(
    bad_frames: bytes,
) -> None:
    raw = (
        b"""event: message_start
data: {"type":"message_start","message":{"id":"msg_open","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}

"""
        + bad_frames
        + b"""event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}

"""
    )

    envelope = AnthropicSSEEnvelope.parse(raw)
    rendered = b"".join(envelope.render())

    # The valid frames around the invalid ones still reconstruct the block.
    assert envelope.message["content"] == [{"type": "text", "text": "hello"}]
    assert envelope.is_complete()
    assert not envelope.is_message_reconstructable()
    assert rendered.count(bad_frames) == 1


_DELTA_PAYLOAD_CASES = {
    "text_delta": ("text", {"type": "text", "text": ""}),
    "thinking_delta": ("thinking", {"type": "thinking", "thinking": ""}),
    "input_json_delta": (
        "partial_json",
        {"type": "tool_use", "id": "toolu_1", "name": "run", "input": {}},
    ),
    "signature_delta": ("signature", {"type": "thinking", "thinking": ""}),
    "citations_delta": ("citation", {"type": "text", "text": ""}),
}
_MISSING = object()


def _single_delta_stream(block: dict, delta: dict) -> tuple[bytes, bytes]:
    def event(payload: dict) -> bytes:
        return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()

    frame = event({"type": "content_block_delta", "index": 0, "delta": delta})
    raw = (
        event(
            {
                "type": "message_start",
                "message": {"id": "msg_d", "type": "message", "role": "assistant", "content": []},
            }
        )
        + event({"type": "content_block_start", "index": 0, "content_block": block})
        + frame
        + event({"type": "content_block_stop", "index": 0})
        + event({"type": "message_stop"})
    )
    return raw, frame


@pytest.mark.parametrize(
    "value", [0, False, None, _MISSING], ids=["zero", "false", "null", "missing"]
)
@pytest.mark.parametrize("delta_type", _DELTA_PAYLOAD_CASES)
def test_falsy_non_string_delta_payload_is_replayed_not_consumed(
    delta_type: str, value: object
) -> None:
    member, block = _DELTA_PAYLOAD_CASES[delta_type]
    delta = {"type": delta_type} if value is _MISSING else {"type": delta_type, member: value}
    raw, frame = _single_delta_stream(block, delta)

    envelope = AnthropicSSEEnvelope.parse(raw)

    assert envelope.is_complete()
    assert not envelope.is_message_reconstructable()
    assert b"".join(envelope.render()).count(frame) == 1


@pytest.mark.parametrize("delta_type", ["text_delta", "thinking_delta", "input_json_delta"])
def test_empty_string_delta_payload_is_still_reconstructed(delta_type: str) -> None:
    member, block = _DELTA_PAYLOAD_CASES[delta_type]
    raw, frame = _single_delta_stream(block, {"type": delta_type, member: ""})

    envelope = AnthropicSSEEnvelope.parse(raw)

    assert envelope.is_message_reconstructable()
    assert frame not in b"".join(envelope.render())


def test_non_object_message_usage_is_replayed_not_merged() -> None:
    bad_start = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_usage","type":"message",'
        b'"role":"assistant","content":[],"usage":[12]}}\n\n'
    )
    raw = (
        bad_start
        + b"""event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

"""
    )

    envelope = AnthropicSSEEnvelope.parse(raw)
    rendered = b"".join(envelope.render())

    assert not envelope.is_message_reconstructable()
    assert rendered.count(bad_start) == 1
    assert envelope.message["usage"] == {"output_tokens": 1}


def test_non_stream_response_keeps_unknown_top_level_fields() -> None:
    response = json.loads(_fixture("claude_code_auto_mode_response.json"))
    rendered = b"".join(StreamingMixin()._response_to_sse(response, "anthropic"))
    assert b"safeguard_results" in rendered
    assert b"toolu_auto_001" in rendered


def test_observation_copy_removes_classifier_payloads() -> None:
    value = {
        "safeguards": {"sentinel": "do-not-log"},
        "nested": [{"safeguard_results": {"sentinel": "do-not-log"}}],
        "safe": "kept",
    }
    observed = strip_safeguard_payload(value)
    assert observed == {"nested": [{}], "safe": "kept"}
    assert value["safeguards"]["sentinel"] == "do-not-log"


class _AutoModeFixtureStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _fixture("claude_code_auto_mode_stream.sse")


class _CapturingAnthropicTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.url: str | None = None
        self.headers: dict[str, str] | None = None
        self.body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.url = str(request.url)
        self.headers = dict(request.headers.items())
        self.body = b"".join([chunk async for chunk in request.stream])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_AutoModeFixtureStream(),
        )


def test_native_handler_preserves_classifier_request_envelope() -> None:
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
    app = create_app(config)
    proxy = app.state.proxy
    transport = _CapturingAnthropicTransport()
    proxy.http_client = httpx.AsyncClient(transport=transport)
    request = json.loads(_fixture("claude_code_auto_mode_request.json"))
    request["model"] = "claude-sonnet-4-6"

    with TestClient(app).stream(
        "POST",
        "/v1/messages?beta=true",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,dangerous-tool-use-2026-09-03",
            "content-type": "application/json",
        },
        content=json.dumps(request).encode(),
    ) as response:
        assert response.status_code == 200
        assert b"safeguard_results" in b"".join(response.iter_bytes())

    assert transport.url == "https://api.anthropic.com/v1/messages?beta=true"
    assert transport.headers is not None
    assert (
        transport.headers["anthropic-beta"] == "claude-code-20250219,dangerous-tool-use-2026-09-03"
    )
    assert transport.body is not None
    assert json.loads(transport.body)["safeguards"] == request["safeguards"]
