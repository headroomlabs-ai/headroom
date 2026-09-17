"""Bedrock prompt caching + max_completion_tokens on the OpenAI path (#3554).

On ``--backend bedrock`` the OpenAI-compatible route (``send_openai_message`` /
``stream_openai_message``) used to forward the client's ``messages`` verbatim.
Two gaps follow:

1. A gateway that renames ``max_tokens`` to ``max_completion_tokens`` (e.g.
   Bifrost) must not see that field fall into ``extra_body``, which Bedrock
   rejects with ``extra_body: Extra inputs are not permitted``.
2. A client that never sends ``cache_control`` (OpenAI-compat gateways strip
   it) never gets a Bedrock ``cachePoint``. Injection is gated on
   ``litellm.utils.supports_prompt_caching`` for the *mapped* litellm model —
   ``get_supported_openai_params`` never reports ``cache_control`` for Bedrock
   and must not block.
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


def _make_backend(provider: str = "bedrock") -> LiteLLMBackend:
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        return LiteLLMBackend(provider=provider, region="us-east-1")


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


def _request_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 32,
    }
    body.update(overrides)
    return body


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


@pytest.mark.parametrize("call", [_send, _stream])
@pytest.mark.asyncio
async def test_max_completion_tokens_not_dumped_into_extra_body(call: Any) -> None:
    """Gateways that rename max_tokens → max_completion_tokens must not 400 Bedrock."""
    body = _request_body(max_completion_tokens=64)
    body.pop("max_tokens")

    kwargs = await call(_make_backend(), body)

    assert kwargs["max_completion_tokens"] == 64
    assert "extra_body" not in kwargs
    assert "max_completion_tokens" not in kwargs.get("extra_body", {})


@pytest.mark.asyncio
async def test_real_registry_caching_model_gets_breakpoint() -> None:
    """End-to-end against litellm's registry: Claude on Bedrock is cacheable."""
    body = _request_body(model="global.anthropic.claude-sonnet-5")
    kwargs = await _send(_make_backend(), body)
    assert kwargs["messages"][0]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize("call", [_send, _stream])
@pytest.mark.asyncio
async def test_supports_prompt_caching_true_injects_cache_control(call: Any) -> None:
    body = _request_body()
    client_messages = copy.deepcopy(body["messages"])

    with patch("headroom.backends.litellm.supports_prompt_caching", return_value=True):
        kwargs = await call(_make_backend(), body)

    assert kwargs["messages"][0]["cache_control"] == EPHEMERAL
    assert kwargs["messages"][1] is body["messages"][1]
    assert body["messages"] == client_messages


@pytest.mark.parametrize("call", [_send, _stream])
@pytest.mark.asyncio
async def test_get_supported_openai_params_lacking_cache_control_does_not_block(
    call: Any,
) -> None:
    """litellm 1.96.2 never returns cache_control for Bedrock; that API must not gate."""
    body = _request_body()

    with (
        patch("headroom.backends.litellm.supports_prompt_caching", return_value=True),
        patch.object(
            litellm,
            "get_supported_openai_params",
            return_value=["max_tokens", "temperature", "top_p", "stream"],
        ) as mock_params,
    ):
        kwargs = await call(_make_backend(), body)

    assert kwargs["messages"][0]["cache_control"] == EPHEMERAL
    mock_params.assert_not_called()


@pytest.mark.asyncio
async def test_gate_checks_the_mapped_litellm_model() -> None:
    backend = _make_backend()
    litellm_model = backend.map_model_id("claude-sonnet-4-20250514")
    assert litellm_model.startswith("bedrock/")

    with patch(
        "headroom.backends.litellm.supports_prompt_caching", return_value=True
    ) as mock_supports:
        kwargs = await _send(backend, _request_body())

    mock_supports.assert_called_once_with(
        model=litellm_model,
        custom_llm_provider="bedrock",
    )
    assert kwargs["messages"][0]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize("call", [_send, _stream])
@pytest.mark.asyncio
async def test_non_supporting_model_unchanged(call: Any) -> None:
    body = _request_body()

    with patch("headroom.backends.litellm.supports_prompt_caching", return_value=False):
        kwargs = await call(_make_backend(), body)

    assert kwargs["messages"] is body["messages"]
    assert "cache_control" not in kwargs["messages"][0]


@pytest.mark.parametrize("call", [_send, _stream])
@pytest.mark.asyncio
async def test_non_bedrock_provider_unchanged(call: Any) -> None:
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        backend = LiteLLMBackend(provider="openrouter", region=None)
    body = {"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]}

    with patch("headroom.backends.litellm.supports_prompt_caching") as mock_supports:
        kwargs = await call(backend, body)

    mock_supports.assert_not_called()
    assert kwargs["messages"] is body["messages"]


@pytest.mark.asyncio
async def test_client_placed_markers_are_respected() -> None:
    body = _request_body()
    body["messages"][1] = {
        "role": "user",
        "content": [{"type": "text", "text": "hi", "cache_control": EPHEMERAL}],
    }

    with patch("headroom.backends.litellm.supports_prompt_caching", return_value=True):
        kwargs = await _send(_make_backend(), body)

    assert kwargs["messages"] is body["messages"]


@pytest.mark.asyncio
async def test_supports_check_failure_forwards_verbatim() -> None:
    body = _request_body()

    with patch(
        "headroom.backends.litellm.supports_prompt_caching", side_effect=RuntimeError("boom")
    ):
        kwargs = await _send(_make_backend(), body)

    assert kwargs["messages"] is body["messages"]


@pytest.mark.asyncio
async def test_injected_marker_becomes_bedrock_converse_cache_point() -> None:
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    with patch("headroom.backends.litellm.supports_prompt_caching", return_value=True):
        kwargs = await _send(_make_backend(), _request_body())

    _, system_blocks = AmazonConverseConfig()._transform_system_message(
        kwargs["messages"],
        model="global.anthropic.claude-sonnet-5",
    )
    assert any("cachePoint" in block for block in system_blocks)


def test_helper_marks_first_system_message_only() -> None:
    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "system", "content": "c"},
    ]
    before = copy.deepcopy(messages)

    out = _place_system_cache_control(messages)

    assert out[0] == {"role": "system", "content": "a", "cache_control": EPHEMERAL}
    assert out[1] is messages[1]
    assert out[2] is messages[2]
    assert messages == before


def test_helper_marks_last_text_block_of_list_content() -> None:
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        },
    ]
    before = copy.deepcopy(messages)

    out = _place_system_cache_control(messages)

    assert out[0]["content"][1] == {"type": "text", "text": "b", "cache_control": EPHEMERAL}
    assert out[0]["content"][0] is messages[0]["content"][0]
    assert messages == before
