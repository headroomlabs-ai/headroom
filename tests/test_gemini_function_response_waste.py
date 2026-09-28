"""Gemini functionResponse waste-signal visibility (issue #819).

Gemini ``functionResponse`` parts are preserved verbatim on the wire (never
compressed), but their payloads previously never reached ``parse_messages``,
so tool output — where most waste lives — contributed nothing to waste
detection on the Gemini paths.

The fix is telemetry-only:

1. ``_gemini_contents_to_messages(..., include_function_responses=True)``
   additionally emits each functionResponse payload as a ``role="tool"``
   message.
2. ``TransformPipeline.apply(..., waste_messages=...)`` parses that richer
   list for waste signals instead of the transform input. The transform path
   and token accounting are untouched.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from headroom import OpenAIProvider, Tokenizer
from headroom.config import HeadroomConfig
from headroom.parser import parse_messages
from headroom.proxy.server import HeadroomProxy, ProxyConfig
from headroom.transforms.pipeline import TransformPipeline

_provider = OpenAIProvider()


@pytest.fixture
def proxy() -> HeadroomProxy:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    return HeadroomProxy(config)


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(_provider.get_token_counter("gpt-4o"), "gpt-4o")


def _big_payload(rows: int = 200) -> dict:
    return {
        "result": [
            {"id": i, "name": f"item_{i}", "status": "ok", "score": i * 3.14} for i in range(rows)
        ]
    }


def _function_response_content(payload: object, name: str = "fetch_data") -> dict:
    return {
        "role": "user",
        "parts": [{"functionResponse": {"name": name, "response": payload}}],
    }


class TestFunctionResponseConversion:
    def test_default_conversion_emits_no_tool_messages(self, proxy):
        contents = [
            {"role": "user", "parts": [{"text": "fetch the data"}]},
            _function_response_content(_big_payload()),
        ]
        messages, preserved = proxy._gemini_contents_to_messages(contents)
        assert [m["role"] for m in messages] == ["user"]
        assert preserved == {1}

    def test_flag_emits_tool_message_for_dict_response(self, proxy):
        payload = _big_payload()
        contents = [
            {"role": "user", "parts": [{"text": "fetch the data"}]},
            _function_response_content(payload),
        ]
        messages, preserved = proxy._gemini_contents_to_messages(
            contents, include_function_responses=True
        )
        assert [m["role"] for m in messages] == ["user", "tool"]
        assert json.loads(messages[1]["content"]) == payload
        # preserved_indices semantics unchanged: the entry is still restored
        # verbatim on the wire regardless of the telemetry conversion.
        assert preserved == {1}

    def test_flag_passes_string_response_through(self, proxy):
        contents = [_function_response_content("plain text tool output")]
        messages, _ = proxy._gemini_contents_to_messages(contents, include_function_responses=True)
        assert messages == [{"role": "tool", "content": "plain text tool output"}]

    def test_flag_skips_missing_response(self, proxy):
        contents = [
            {"role": "user", "parts": [{"functionResponse": {"name": "noop"}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "none", "response": None}}]},
        ]
        messages, _ = proxy._gemini_contents_to_messages(contents, include_function_responses=True)
        assert messages == []

    def test_flag_emits_text_before_tool_within_entry(self, proxy):
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": "tool said:"},
                    {"functionResponse": {"name": "f", "response": "output"}},
                ],
            }
        ]
        messages, _ = proxy._gemini_contents_to_messages(contents, include_function_responses=True)
        assert [m["role"] for m in messages] == ["user", "tool"]
        assert messages[0]["content"] == "tool said:"
        assert messages[1]["content"] == "output"

    def test_unserializable_response_falls_back_to_str(self, proxy):
        circular: dict = {"name": "loop"}
        circular["self"] = circular
        text = proxy._function_response_text({"response": circular})
        assert "loop" in text


class TestMalformedPartsToleration:
    """A request-controlled `parts` that is null or carries non-dict elements
    must not crash the compression-path conversion helpers."""

    def test_null_parts_does_not_crash(self, proxy):
        contents = [
            {"role": "user", "parts": None},
            {"role": "user", "parts": [{"text": "real"}]},
        ]
        messages, preserved = proxy._gemini_contents_to_messages(contents)
        assert messages == [{"role": "user", "content": "real"}]
        assert preserved == set()

    def test_string_part_elements_do_not_crash(self, proxy):
        # A client that treats `parts` as a string array sends bare strings;
        # they carry no `text` key, so they contribute nothing but must not
        # crash `.get`.
        contents = [{"role": "user", "parts": ["bare string", {"text": "kept"}]}]
        messages, _ = proxy._gemini_contents_to_messages(contents)
        assert messages == [{"role": "user", "content": "kept"}]

    def test_null_part_element_is_skipped(self, proxy):
        contents = [{"role": "user", "parts": [None, {"text": "kept"}]}]
        messages, _ = proxy._gemini_contents_to_messages(contents)
        assert messages == [{"role": "user", "content": "kept"}]

    def test_has_non_text_parts_tolerates_null_parts(self, proxy):
        assert proxy._has_non_text_parts({"role": "user", "parts": None}) is False
        assert proxy._has_non_text_parts({"role": "user", "parts": ["str"]}) is False
        assert proxy._has_non_text_parts({"parts": [{"inlineData": {"data": "x"}}]}) is True

    def test_non_dict_content_entry_is_tolerated(self, proxy):
        # A non-dict entry in contents[] is treated as an empty user turn rather
        # than crashing content.get / the parts iteration.
        contents = ["not a dict", {"role": "user", "parts": [{"text": "kept"}]}]
        messages, _ = proxy._gemini_contents_to_messages(contents)
        assert messages == [{"role": "user", "content": "kept"}]


class TestFunctionResponseWasteParsing:
    def test_function_response_payload_reaches_waste_signals(self, proxy, tokenizer):
        contents = [
            {"role": "user", "parts": [{"text": "fetch the data"}]},
            _function_response_content(_big_payload()),
        ]
        messages, _ = proxy._gemini_contents_to_messages(contents, include_function_responses=True)
        blocks, _, waste = parse_messages(messages, tokenizer)
        assert any(b.kind == "tool_result" for b in blocks)
        assert waste.json_bloat_tokens > 0

    def test_repeated_function_response_counts_as_reread(self, proxy, tokenizer):
        payload = _big_payload()
        filler = [{"role": "user", "parts": [{"text": f"working on step {i}"}]} for i in range(5)]
        contents = [
            _function_response_content(payload),
            *filler,
            _function_response_content(payload),
        ]
        messages, _ = proxy._gemini_contents_to_messages(contents, include_function_responses=True)
        _, _, waste = parse_messages(messages, tokenizer)
        assert waste.reread_tokens > 0


class TestPipelineWasteMessages:
    @staticmethod
    def _base_messages() -> list[dict]:
        # Compressible enough that the pipeline clears the >100 saved-token
        # gate that guards waste-signal detection.
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Inspect the data set."},
            {"role": "tool", "content": json.dumps(_big_payload(400)["result"])},
        ]

    def test_waste_messages_override_waste_source(self, tokenizer):
        messages = self._base_messages()
        extra_tool = {"role": "tool", "content": json.dumps(_big_payload(300))}

        baseline = TransformPipeline(HeadroomConfig()).apply(
            [dict(m) for m in messages], model="gpt-4o", model_limit=128000
        )
        enriched = TransformPipeline(HeadroomConfig()).apply(
            [dict(m) for m in messages],
            model="gpt-4o",
            model_limit=128000,
            waste_messages=[*messages, extra_tool],
        )

        assert baseline.waste_signals is not None
        assert enriched.waste_signals is not None
        assert enriched.waste_signals.json_bloat_tokens > baseline.waste_signals.json_bloat_tokens

    def test_waste_messages_do_not_affect_transform_output(self, tokenizer):
        messages = self._base_messages()
        extra_tool = {"role": "tool", "content": json.dumps(_big_payload(300))}

        baseline = TransformPipeline(HeadroomConfig()).apply(
            [dict(m) for m in messages], model="gpt-4o", model_limit=128000
        )
        enriched = TransformPipeline(HeadroomConfig()).apply(
            [dict(m) for m in messages],
            model="gpt-4o",
            model_limit=128000,
            waste_messages=[*messages, extra_tool],
        )

        assert enriched.messages == baseline.messages
        assert enriched.tokens_before == baseline.tokens_before
        assert enriched.tokens_after == baseline.tokens_after

    def test_no_waste_messages_falls_back_to_transform_input(self, tokenizer):
        result = TransformPipeline(HeadroomConfig()).apply(
            [dict(m) for m in self._base_messages()], model="gpt-4o", model_limit=128000
        )
        assert result.waste_signals is not None
        assert result.waste_signals.json_bloat_tokens > 0
