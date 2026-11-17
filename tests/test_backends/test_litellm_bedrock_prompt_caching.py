"""Opt-in Bedrock prompt caching on `LiteLLMBackend`'s OpenAI-format path (#3554).

The OpenAI-compatible route forwards `messages` verbatim, so a client that never
sends `cache_control` (OpenAI-compat gateways such as Bifrost) gets no Bedrock
`cachePoint` and therefore never sees `cache_read_input_tokens`. With the
`bedrock_openai_prompt_caching` rollout feature enabled, the backend marks the
first system message when litellm reports the *mapped* model supports prompt
caching. Default behaviour (feature off) is untouched.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

import litellm  # noqa: E402

from headroom.backends.litellm import (  # noqa: E402  (must follow importorskip)
    LiteLLMBackend,
    _place_system_cache_control,
)

FEATURE = "bedrock_openai_prompt_caching"
CACHING_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
EPHEMERAL = {"type": "ephemeral"}


class _FakeAsyncStream:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> _FakeAsyncStream:
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _make_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_123",
        created=123456,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="ok", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )


def _make_backend(provider: str = "bedrock") -> LiteLLMBackend:
    # Patch the inference-profile fetch so `__init__` doesn't try to talk to AWS.
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        return LiteLLMBackend(provider=provider, region="us-east-1")


def _request_body(model: str = CACHING_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 32,
    }


async def _send(backend: LiteLLMBackend, body: dict[str, Any]) -> dict[str, Any]:
    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = _make_response()
        await backend.send_openai_message(body, {})
    return mock_acomp.await_args.kwargs


async def _stream(backend: LiteLLMBackend, body: dict[str, Any]) -> dict[str, Any]:
    stream = _FakeAsyncStream(
        [SimpleNamespace(model_dump=lambda **kwargs: {"id": "chunk1", "choices": []})]
    )
    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = stream
        chunks = [chunk async for chunk in backend.stream_openai_message(body, {})]
    assert chunks[-1] == "data: [DONE]\n\n"
    return mock_acomp.await_args.kwargs


@pytest.fixture
def feature_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_FEATURES", FEATURE)
    monkeypatch.delenv("HEADROOM_DISABLE_FEATURES", raising=False)


@pytest.fixture
def feature_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_FEATURES", raising=False)
    monkeypatch.delenv("HEADROOM_DISABLE_FEATURES", raising=False)


# =============================================================================
# _place_system_cache_control (pure)
# =============================================================================


def test_marks_first_system_message_string_content() -> None:
    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "system", "content": "c"},
    ]
    before = copy.deepcopy(messages)

    out = _place_system_cache_control(messages)

    assert out[0] == {"role": "system", "content": "a", "cache_control": EPHEMERAL}
    assert out[1] is messages[1]
    assert out[2] is messages[2]  # only the first system message is marked
    assert messages == before  # caller-owned input never mutated


def test_marks_last_text_block_of_list_content() -> None:
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
                {"type": "text", "text": ""},
            ],
        },
        {"role": "user", "content": "hi"},
    ]
    before = copy.deepcopy(messages)

    out = _place_system_cache_control(messages)

    blocks = out[0]["content"]
    assert blocks[0] is messages[0]["content"][0]
    assert blocks[1] == {"type": "text", "text": "b", "cache_control": EPHEMERAL}
    assert blocks[2] is messages[0]["content"][2]  # empty text is not a breakpoint
    assert messages == before


def test_system_message_anywhere_in_the_list() -> None:
    messages = [{"role": "user", "content": "hi"}, {"role": "system", "content": "s"}]

    out = _place_system_cache_control(messages)

    assert out[0] is messages[0]
    assert out[1]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize(
    "content",
    ["", [], None, 7, [{"type": "image_url", "image_url": {"url": "x"}}], ["plain string"]],
)
def test_unmarkable_system_content_is_skipped(content: Any) -> None:
    messages = [{"role": "system", "content": content}, {"role": "system", "content": "s"}]

    out = _place_system_cache_control(messages)

    assert out[0] is messages[0]
    assert out[1]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "hi"}],
        [{"role": "system", "content": ""}],
        ["not a dict", {"role": "user", "content": "hi"}],
        # The client already owns breakpoint placement -- anywhere, any shape.
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u", "cache_control": EPHEMERAL},
        ],
        [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "u", "cache_control": EPHEMERAL}],
            },
        ],
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "s", "cache_control": EPHEMERAL}],
            }
        ],
    ],
)
def test_returns_input_unchanged_when_nothing_to_do(messages: list[Any]) -> None:
    before = copy.deepcopy(messages)

    assert _place_system_cache_control(messages) is messages
    assert messages == before


# =============================================================================
# Backend gate
# =============================================================================


@pytest.mark.usefixtures("feature_default")
@pytest.mark.parametrize("call", [_send, _stream])
async def test_feature_off_by_default_forwards_messages_verbatim(call: Any) -> None:
    body = _request_body()

    kwargs = await call(_make_backend(), body)

    assert kwargs["messages"] is body["messages"]
    assert "cache_control" not in kwargs["messages"][0]


@pytest.mark.usefixtures("feature_enabled")
@pytest.mark.parametrize("call", [_send, _stream])
async def test_feature_on_marks_system_prompt_for_caching_model(call: Any) -> None:
    body = _request_body()
    client_messages = copy.deepcopy(body["messages"])

    kwargs = await call(_make_backend(), body)

    assert kwargs["messages"][0]["cache_control"] == EPHEMERAL
    assert kwargs["messages"][1] is body["messages"][1]
    assert body["messages"] == client_messages  # proxy re-reads body["messages"] after send


@pytest.mark.usefixtures("feature_enabled")
async def test_feature_on_skips_models_without_prompt_caching() -> None:
    body = _request_body(model="meta.llama3-1-70b-instruct-v1:0")

    with patch("headroom.backends.litellm.supports_prompt_caching", return_value=False):
        kwargs = await _send(_make_backend(), body)

    assert kwargs["messages"] is body["messages"]


@pytest.mark.usefixtures("feature_enabled")
async def test_gate_checks_the_mapped_litellm_model() -> None:
    # A bare alias such as `claude-sonnet-4-20250514` is not a Bedrock entry in
    # litellm's registry; the resolved `bedrock/<region>.anthropic...` id is.
    body = _request_body(model="claude-sonnet-4-20250514")
    backend = _make_backend()

    with patch(
        "headroom.backends.litellm.supports_prompt_caching", return_value=True
    ) as mock_supports:
        kwargs = await _send(backend, body)

    mock_supports.assert_called_once_with(model=backend.map_model_id("claude-sonnet-4-20250514"))
    assert kwargs["messages"][0]["cache_control"] == EPHEMERAL


@pytest.mark.usefixtures("feature_enabled")
async def test_feature_on_is_a_no_op_for_non_bedrock_providers() -> None:
    body = _request_body(model="anthropic/claude-3-5-sonnet-20241022")

    with patch("headroom.backends.litellm.supports_prompt_caching") as mock_supports:
        kwargs = await _send(_make_backend(provider="openrouter"), body)

    mock_supports.assert_not_called()
    assert kwargs["messages"] is body["messages"]


@pytest.mark.usefixtures("feature_enabled")
async def test_feature_on_respects_client_placed_markers() -> None:
    body = _request_body()
    body["messages"][1]["cache_control"] = EPHEMERAL

    kwargs = await _send(_make_backend(), body)

    assert kwargs["messages"] is body["messages"]


def test_disable_features_is_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_FEATURES", FEATURE)
    monkeypatch.setenv("HEADROOM_DISABLE_FEATURES", FEATURE)

    assert _make_backend()._openai_prompt_caching is False


@pytest.mark.usefixtures("feature_enabled")
async def test_marker_becomes_a_bedrock_converse_cache_point() -> None:
    """The marker shape must be the one litellm's Converse transform consumes."""
    body = _request_body()

    kwargs = await _send(_make_backend(), body)
    request = litellm.AmazonConverseConfig().transform_request(
        model=CACHING_MODEL,
        messages=kwargs["messages"],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["system"] == [{"text": "You are terse."}, {"cachePoint": {"type": "default"}}]
