"""LiteLLMBackend thinking-block fidelity.

Regression suite for the bug where the Anthropic<->LiteLLM converter dropped
extended-thinking blocks in both directions. Dropping them is not a quality
loss but a hard Anthropic 400 on a tool-use continuation
(`Expected thinking or redacted_thinking, but found tool_use`), and the
signature is validated cryptographically, so blocks must round-trip verbatim.

Contract pinned here:
- REQUEST (Anthropic-family target): assistant thinking/redacted_thinking blocks
  are carried on the outgoing message's `thinking_blocks` field, which litellm's
  own Anthropic/Bedrock transforms forward with signature + lead position.
- REQUEST (cross-vendor target): thinking is stripped, so litellm cannot ship an
  unknown field to a non-Anthropic provider.
- RESPONSE: a signed thinking block from litellm (message.thinking_blocks) is
  rebuilt as a LEADING Anthropic thinking block; unsigned reasoning is dropped.
- STREAMING: thinking is emitted as a leading thinking block with thinking_delta
  + signature_delta before any text/tool_use.
- OpenAI-in: reasoning survives into the rebuilt OpenAI response (Codex->Claude).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import (  # noqa: E402
    LiteLLMBackend,
    _extract_thinking_content_blocks,
    _is_anthropic_family_model,
)

SIG = "ErUBCkYIBRgCKkB_signed_opaque_blob=="


def _backend(provider: str = "anthropic") -> LiteLLMBackend:
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        return LiteLLMBackend(provider=provider)


THINKING_TOOL_TURN = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "The user wants the weather.", "signature": SIG},
        {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
    ],
}

THINKING_TEXT_TURN = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "Simple greeting.", "signature": SIG},
        {"type": "text", "text": "Hello!"},
    ],
}


# --------------------------------------------------------------------------- #
# helper unit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model,expected",
    [
        ("anthropic/claude-opus-4-20250514", True),
        ("bedrock/anthropic.claude-sonnet-4-20250514-v1:0", True),
        ("vertex_ai/claude-3-5-sonnet", True),
        ("claude-opus-5", True),
        ("gpt-4o", False),
        ("openai/gpt-5", False),
        ("deepseek/deepseek-chat", False),
        ("", False),
    ],
)
def test_is_anthropic_family_model(model: str, expected: bool) -> None:
    assert _is_anthropic_family_model(model) is expected


# --------------------------------------------------------------------------- #
# request-direction conversion
# --------------------------------------------------------------------------- #
def test_thinking_survives_conversion_for_anthropic_target() -> None:
    backend = _backend()
    converted = backend._convert_messages_for_litellm([THINKING_TOOL_TURN], preserve_thinking=True)
    assistant = next(m for m in converted if m.get("tool_calls"))
    assert "thinking_blocks" in assistant, "signed thinking dropped for Anthropic target"
    tb = assistant["thinking_blocks"][0]
    assert tb["type"] == "thinking"
    assert tb["signature"] == SIG, "signature must round-trip byte-for-byte"


def test_thinking_survives_on_text_only_assistant_turn() -> None:
    backend = _backend()
    converted = backend._convert_messages_for_litellm([THINKING_TEXT_TURN], preserve_thinking=True)
    assistant = next(m for m in converted if m["role"] == "assistant")
    assert assistant["thinking_blocks"][0]["signature"] == SIG
    assert assistant["content"] == "Hello!"


def test_thinking_stripped_for_cross_vendor_target() -> None:
    backend = _backend(provider="openai")
    converted = backend._convert_messages_for_litellm([THINKING_TOOL_TURN], preserve_thinking=False)
    assistant = next(m for m in converted if m.get("tool_calls"))
    assert "thinking_blocks" not in assistant, (
        "thinking must NOT be shipped to a non-Anthropic provider"
    )


def test_redacted_thinking_survives_conversion() -> None:
    backend = _backend()
    turn = {
        "role": "assistant",
        "content": [
            {"type": "redacted_thinking", "data": "ENCRYPTED_BLOB=="},
            {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
        ],
    }
    converted = backend._convert_messages_for_litellm([turn], preserve_thinking=True)
    assistant = next(m for m in converted if m.get("tool_calls"))
    assert assistant["thinking_blocks"][0] == {
        "type": "redacted_thinking",
        "data": "ENCRYPTED_BLOB==",
    }


# --------------------------------------------------------------------------- #
# response-direction reconstruction (pure helper)
# --------------------------------------------------------------------------- #
def test_extract_thinking_keeps_signed_block_first() -> None:
    message = SimpleNamespace(
        content="Answer.",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoning", "signature": SIG}],
    )
    blocks = _extract_thinking_content_blocks(message)
    assert blocks == [{"type": "thinking", "thinking": "reasoning", "signature": SIG}]


def test_extract_thinking_drops_unsigned_reasoning() -> None:
    # An unsigned block would be rejected by Anthropic on replay; never emit it.
    message = SimpleNamespace(
        content="Answer.",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoning", "signature": ""}],
        reasoning_content="reasoning",
    )
    assert _extract_thinking_content_blocks(message) == []


def test_extract_thinking_keeps_redacted() -> None:
    message = SimpleNamespace(thinking_blocks=[{"type": "redacted_thinking", "data": "BLOB"}])
    assert _extract_thinking_content_blocks(message) == [
        {"type": "redacted_thinking", "data": "BLOB"}
    ]


# --------------------------------------------------------------------------- #
# send_message end-to-end (mocked acompletion)
# --------------------------------------------------------------------------- #
def _mock_response(*, content="Done.", thinking_blocks=None, reasoning=None, tool_calls=None):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        thinking_blocks=thinking_blocks,
        reasoning_content=reasoning,
    )
    return SimpleNamespace(
        id="resp_1",
        created=1,
        choices=[SimpleNamespace(index=0, finish_reason="stop", message=message)],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


@pytest.mark.asyncio
async def test_send_message_forwards_thinking_and_reconstructs_response() -> None:
    with (
        patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
        patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
    ):
        mock_acomp.return_value = _mock_response(
            thinking_blocks=[{"type": "thinking", "thinking": "resp reasoning", "signature": SIG}]
        )
        backend = LiteLLMBackend(provider="anthropic")
        body = {
            "model": "claude-opus-4-20250514",
            "messages": [THINKING_TOOL_TURN, {"role": "user", "content": "and tomorrow?"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
        result = await backend.send_message(body, {})

    # request: thinking preserved into outgoing history + thinking param forwarded
    sent = mock_acomp.call_args.kwargs
    hist_assistant = next(m for m in sent["messages"] if m.get("tool_calls"))
    assert hist_assistant["thinking_blocks"][0]["signature"] == SIG
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2000}

    # response: signed thinking rebuilt as the LEADING content block
    content = result.body["content"]
    assert content[0]["type"] == "thinking"
    assert content[0]["signature"] == SIG
    assert content[1]["type"] == "text"


@pytest.mark.asyncio
async def test_send_message_strips_thinking_for_cross_vendor() -> None:
    with (
        patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
        patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
    ):
        mock_acomp.return_value = _mock_response()
        backend = LiteLLMBackend(provider="openai")
        body = {
            "model": "gpt-4o",
            "messages": [THINKING_TOOL_TURN, {"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
        await backend.send_message(body, {})

    sent = mock_acomp.call_args.kwargs
    hist_assistant = next(m for m in sent["messages"] if m.get("tool_calls"))
    assert "thinking_blocks" not in hist_assistant
    assert "thinking" not in sent, "thinking config must not be forwarded cross-vendor"


# --------------------------------------------------------------------------- #
# streaming response reconstruction
# --------------------------------------------------------------------------- #
class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _delta(**kw):
    base = {"reasoning_content": None, "thinking_blocks": None, "content": None, "tool_calls": None}
    base.update(kw)
    return SimpleNamespace(**base)


def _chunk(delta, finish_reason=None, usage=None):
    ns = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, delta=delta)])
    if usage is not None:
        ns.usage = usage
    return ns


@pytest.mark.asyncio
async def test_streaming_emits_leading_thinking_block_with_signature() -> None:
    chunks = [
        _chunk(_delta(reasoning_content="Let me ")),
        _chunk(_delta(reasoning_content="think.")),
        _chunk(_delta(thinking_blocks=[{"type": "thinking", "signature": SIG}])),
        _chunk(_delta(content="The answer.")),
        _chunk(
            _delta(),
            finish_reason="stop",
            usage=SimpleNamespace(
                prompt_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
        ),
    ]
    with (
        patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
        patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
    ):
        mock_acomp.return_value = _FakeAsyncStream(chunks)
        backend = LiteLLMBackend(provider="anthropic")
        body = {"model": "claude-opus-4-20250514", "messages": [{"role": "user", "content": "hi"}]}
        events = [ev.data async for ev in backend.stream_message(body, {})]

    types = [
        (
            e.get("type"),
            e.get("delta", {}).get("type")
            if e.get("type") == "content_block_delta"
            else e.get("content_block", {}).get("type")
            if e.get("type") == "content_block_start"
            else None,
        )
        for e in events
    ]
    # thinking block opens first, carries thinking_delta then signature_delta,
    # closes, then the text block opens.
    assert ("content_block_start", "thinking") in types
    assert ("content_block_delta", "thinking_delta") in types
    assert ("content_block_delta", "signature_delta") in types
    assert ("content_block_start", "text") in types
    # ordering: the thinking block_start precedes the text block_start
    start_types = [t for t in types if t[0] == "content_block_start"]
    assert start_types[0] == ("content_block_start", "thinking")
    assert start_types[1] == ("content_block_start", "text")
    # the signed thinking text was accumulated
    thinking_text = "".join(
        e["delta"]["thinking"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    )
    assert thinking_text == "Let me think."
    sig = [
        e["delta"]["signature"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "signature_delta"
    ]
    assert sig == [SIG]


# --------------------------------------------------------------------------- #
# OpenAI-in reasoning passthrough (Codex -> Claude direction)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_send_openai_message_carries_reasoning_through() -> None:
    with (
        patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
        patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
    ):
        mock_acomp.return_value = _mock_response(
            reasoning="claude reasoning",
            thinking_blocks=[
                {"type": "thinking", "thinking": "claude reasoning", "signature": SIG}
            ],
        )
        backend = LiteLLMBackend(provider="anthropic")
        body = {"model": "claude-opus-4-20250514", "messages": [{"role": "user", "content": "hi"}]}
        result = await backend.send_openai_message(body, {})

    msg = result.body["choices"][0]["message"]
    assert msg["reasoning_content"] == "claude reasoning"
    assert msg["thinking_blocks"][0]["signature"] == SIG


# --------------------------------------------------------------------------- #
# streaming edge cases (adversarial-review regressions)
# --------------------------------------------------------------------------- #
def _block_starts(events):
    return [e["content_block"]["type"] for e in events if e.get("type") == "content_block_start"]


def _delta_types(events):
    return [e["delta"]["type"] for e in events if e.get("type") == "content_block_delta"]


async def _run_stream(chunks):
    with (
        patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
        patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
    ):
        mock_acomp.return_value = _FakeAsyncStream(chunks)
        backend = LiteLLMBackend(provider="anthropic")
        body = {"model": "claude-opus-4-20250514", "messages": [{"role": "user", "content": "hi"}]}
        return [ev.data async for ev in backend.stream_message(body, {})]


@pytest.mark.asyncio
async def test_streaming_drops_unsigned_thinking() -> None:
    # Reasoning streams but a signature never arrives (e.g. truncation): the
    # thinking block must be dropped, never emitted unsigned (rule 3).
    events = await _run_stream(
        [
            _chunk(_delta(reasoning_content="Let me ")),
            _chunk(_delta(reasoning_content="think.")),
            _chunk(_delta(content="answer")),
            _chunk(_delta(), finish_reason="stop"),
        ]
    )
    assert "thinking" not in _block_starts(events), "unsigned thinking must not be emitted"
    assert "thinking_delta" not in _delta_types(events)
    assert "signature_delta" not in _delta_types(events)
    assert _block_starts(events) == ["text"]


@pytest.mark.asyncio
async def test_streaming_late_signature_after_text_is_dropped_not_reordered() -> None:
    # Reasoning + signature arrive AFTER text already started. A thinking block
    # can only lead, so it must be dropped — never reopened after text (rule 2),
    # and no stray signature_delta emitted against a closed/foreign block.
    events = await _run_stream(
        [
            _chunk(_delta(content="Hello")),
            _chunk(_delta(reasoning_content="late thought")),
            _chunk(_delta(thinking_blocks=[{"type": "thinking", "signature": SIG}])),
            _chunk(_delta(content=" world")),
            _chunk(_delta(), finish_reason="stop"),
        ]
    )
    assert _block_starts(events) == ["text"], "no thinking block may follow text"
    assert "thinking_delta" not in _delta_types(events)
    assert "signature_delta" not in _delta_types(events)


@pytest.mark.asyncio
async def test_streaming_thinking_only_turn_flushes_signed() -> None:
    # Signed reasoning with no subsequent content: still emitted as a leading
    # (and only) thinking block at end of stream.
    events = await _run_stream(
        [
            _chunk(_delta(reasoning_content="just thinking")),
            _chunk(_delta(thinking_blocks=[{"type": "thinking", "signature": SIG}])),
            _chunk(_delta(), finish_reason="stop"),
        ]
    )
    assert _block_starts(events) == ["thinking"]
    assert "signature_delta" in _delta_types(events)
    sig = [
        e["delta"]["signature"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "signature_delta"
    ]
    assert sig == [SIG]


def test_request_drops_unsigned_thinking_block() -> None:
    # Symmetric with the response path: an unsigned thinking block in history is
    # NOT forwarded (litellm would ship it and Anthropic would 400 the turn).
    backend = _backend()
    turn = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "unsigned"},  # no signature
            {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
        ],
    }
    converted = backend._convert_messages_for_litellm([turn], preserve_thinking=True)
    assistant = next(m for m in converted if m.get("tool_calls"))
    assert "thinking_blocks" not in assistant, "unsigned thinking must not be forwarded"
