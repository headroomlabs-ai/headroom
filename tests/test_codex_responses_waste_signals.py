"""Codex (OpenAI Responses API) waste-signal visibility (issue #820).

The /v1/responses path never ran ``parse_messages``: compression goes through
CompressionUnits (not TransformPipeline), and the minimal ``messages`` list it
synthesises drops list-typed ``input`` entirely — so tool output never reached
waste detection and the dashboard "What Headroom Removed" stayed empty for
Codex traffic.

The fix is telemetry-only:

1. ``_responses_input_to_waste_messages`` converts a Responses payload into
   OpenAI-style messages — tool output items (``function_call_output`` etc.)
   become ``role="tool"`` messages, ``message`` items keep their role and
   joined part text.
2. ``handle_openai_responses`` parses that list (behind the same >100
   saved-token gate as ``TransformPipeline.apply``) and threads the result
   into both the non-streaming ``RequestOutcome`` and
   ``_stream_response(waste_signals=...)``.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from headroom import OpenAIProvider, Tokenizer
from headroom.parser import parse_messages
from headroom.proxy.handlers.openai import (
    _RESPONSES_OUTPUT_ITEM_TYPES,
    OpenAIHandlerMixin,
    _responses_input_to_waste_messages,
    _responses_part_text,
)

_provider = OpenAIProvider()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(_provider.get_token_counter("gpt-4o"), "gpt-4o")


def _big_output(rows: int = 200) -> str:
    return json.dumps(
        [{"id": i, "name": f"item_{i}", "status": "ok", "score": i * 3.14} for i in range(rows)]
    )


def _fco(output: object, call_id: str = "call_1") -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


class TestResponsesPartText:
    def test_string_passthrough(self):
        assert _responses_part_text("plain") == "plain"

    def test_part_list_joined(self):
        parts = [
            {"type": "output_text", "text": "first"},
            "second",
            {"type": "input_text", "text": "third"},
            {"type": "input_image", "image_url": "ignored"},
        ]
        assert _responses_part_text(parts) == "first\nsecond\nthird"

    def test_non_text_returns_empty(self):
        assert _responses_part_text(None) == ""
        assert _responses_part_text({"text": "not a list"}) == ""


class TestResponsesWasteConversion:
    def test_string_input_and_instructions(self):
        messages = _responses_input_to_waste_messages("be terse", "hello")
        assert messages == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ]

    def test_message_items_keep_role(self):
        items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ]
        messages = _responses_input_to_waste_messages(None, items)
        assert messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_function_call_output_becomes_tool_message(self):
        output = _big_output()
        messages = _responses_input_to_waste_messages(None, [_fco(output)])
        assert messages == [{"role": "tool", "content": output, "tool_call_id": "call_1"}]

    def test_output_part_list_joined(self):
        messages = _responses_input_to_waste_messages(
            None,
            [_fco([{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}])],
        )
        assert messages[0]["content"] == "a\nb"

    def test_all_output_item_types_covered(self):
        for item_type in _RESPONSES_OUTPUT_ITEM_TYPES:
            messages = _responses_input_to_waste_messages(
                None, [{"type": item_type, "output": "tool output text"}]
            )
            assert messages == [{"role": "tool", "content": "tool output text"}], item_type

    def test_skips_unusable_items(self):
        items = [
            "not a dict",
            {"type": "function_call", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": ""},
            {"type": "message", "role": "user", "content": []},
        ]
        assert _responses_input_to_waste_messages(None, items) == []

    def test_non_list_non_string_input(self):
        assert _responses_input_to_waste_messages(None, {"weird": True}) == []

    def test_class_attr_aliases_module_constant(self):
        assert OpenAIHandlerMixin.OPENAI_RESPONSES_OUTPUT_TYPES is _RESPONSES_OUTPUT_ITEM_TYPES


class TestResponsesWasteParsing:
    def test_tool_output_reaches_waste_signals(self, tokenizer):
        items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            _fco(_big_output()),
        ]
        messages = _responses_input_to_waste_messages("be terse", items)
        blocks, _, waste = parse_messages(messages, tokenizer)
        assert any(b.kind == "tool_result" for b in blocks)
        assert waste.json_bloat_tokens > 0

    def test_repeated_tool_output_counts_as_reread(self, tokenizer):
        output = _big_output()
        filler = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"step {i}"}],
            }
            for i in range(5)
        ]
        items = [_fco(output, "call_1"), *filler, _fco(output, "call_2")]
        messages = _responses_input_to_waste_messages(None, items)
        _, _, waste = parse_messages(messages, tokenizer)
        assert waste.reread_tokens > 0


class TestResponsesWasteParsingOffLoop:
    """The handler's waste-signal parse must not run on the event loop.

    ``parse_messages`` is CPU-bound over the whole re-sent conversation, and
    Codex re-sends multi-MB inputs every turn. Run inline, it stalled every
    other in-flight request on the proxy for the duration of a telemetry-only
    statistic. The parse must run on a worker thread, and its result must
    still reach the ``RequestOutcome``.
    """

    def test_parse_runs_off_loop_and_still_reports(self, monkeypatch):
        import threading

        import httpx
        from fastapi.testclient import TestClient

        import headroom.parser as parser_module
        from headroom.proxy.loopback_guard import require_loopback
        from headroom.proxy.server import ProxyConfig, create_app

        app = create_app(
            ProxyConfig(
                optimize=True,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
            )
        )
        app.dependency_overrides[require_loopback] = lambda: None
        server = app.state.proxy

        loop_threads: list[int] = []
        parse_threads: list[int] = []
        real_parse = parser_module.parse_messages

        def _recording_parse(*args, **kwargs):
            parse_threads.append(threading.get_ident())
            return real_parse(*args, **kwargs)

        monkeypatch.setattr(parser_module, "parse_messages", _recording_parse)

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):
            loop_threads.append(threading.get_ident())
            return httpx.Response(
                200,
                json={
                    "id": "resp_test",
                    "object": "response",
                    "output": [],
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
                request=httpx.Request("POST", url),
            )

        server._retry_request = _fake_retry

        async def _fake_compress(payload, **kwargs):
            # Clear the >100 saved-token gate so the waste parse runs.
            return (
                dict(payload),
                True,
                5_000,
                ["openai:responses:trim"],
                None,
                20_000,
                15_000,
                5_000,
                {},
            )

        server._compress_openai_responses_payload_in_executor = _fake_compress

        outcomes = []

        async def _capture_outcome(outcome):
            outcomes.append(outcome)

        server._record_request_outcome = _capture_outcome

        payload = {
            "model": "gpt-5-codex",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                },
                {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
                _fco(_big_output()),
            ],
        }

        with TestClient(app) as client:
            resp = client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
                json=payload,
            )

        assert resp.status_code == 200
        assert loop_threads, "upstream call never ran"
        assert parse_threads, "waste-signal parse never ran"
        assert parse_threads[0] != loop_threads[0], "waste-signal parse ran on the event loop"
        assert outcomes, "no RequestOutcome recorded"
        assert outcomes[0].waste_signals
        assert outcomes[0].waste_signals["json_bloat"] > 0
