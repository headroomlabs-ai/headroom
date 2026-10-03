from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402


class FakeAsyncStream:
    def __init__(self, items) -> None:  # noqa: ANN001
        self._items = list(items)

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def make_backend() -> LiteLLMBackend:
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        return LiteLLMBackend(provider="openrouter")


def make_response() -> SimpleNamespace:
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


def request_body(**overrides):
    body = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_chat_template_kwargs_forwarded_buffered() -> None:
    backend = make_backend()

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = make_response()

        await backend.send_openai_message(
            request_body(chat_template_kwargs={"enable_thinking": False}),
            {},
        )

    kwargs = mock_acomp.await_args.kwargs
    assert kwargs["max_tokens"] == 32
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_chat_template_kwargs_forwarded_streaming() -> None:
    backend = make_backend()

    stream = FakeAsyncStream(
        [
            SimpleNamespace(model_dump=lambda **kwargs: {"id": "chunk1", "choices": []}),
        ]
    )

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = stream

        chunks = [
            chunk
            async for chunk in backend.stream_openai_message(
                request_body(
                    chat_template_kwargs={"enable_thinking": False},
                    stream_options={"include_usage": True},
                ),
                {},
            )
        ]

    kwargs = mock_acomp.await_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_streaming_chunks_report_requested_model_not_mapped_slug() -> None:
    """Streamed chunks must carry the client's requested model, not the
    LiteLLM-mapped provider slug (matches the non-streaming path)."""
    import json

    backend = make_backend()  # provider="openrouter"

    # LiteLLM tags each chunk with the mapped model it was called with
    # ("openrouter/qwen3"); the client asked for "qwen3".
    stream = FakeAsyncStream(
        [
            SimpleNamespace(
                model_dump=lambda **kwargs: {
                    "id": "chunk1",
                    "model": "openrouter/qwen3",
                    "choices": [{"index": 0, "delta": {"content": "hi"}}],
                }
            ),
        ]
    )

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = stream

        chunks = [
            chunk async for chunk in backend.stream_openai_message(request_body(model="qwen3"), {})
        ]

    data_chunks = [c for c in chunks if c.startswith("data: ") and "[DONE]" not in c]
    payload = json.loads(data_chunks[0][len("data: ") :])
    assert payload["model"] == "qwen3"


@pytest.mark.asyncio
async def test_standard_only_body_has_no_extra_body() -> None:
    backend = make_backend()

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = make_response()

        await backend.send_openai_message(
            request_body(temperature=0.1, top_p=0.9),
            {},
        )

    kwargs = mock_acomp.await_args.kwargs
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_standard_params_still_forwarded() -> None:
    backend = make_backend()

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = make_response()

        await backend.send_openai_message(
            request_body(
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
                chat_template_kwargs={"enable_thinking": False},
            ),
            {},
        )

    kwargs = mock_acomp.await_args.kwargs
    assert kwargs["temperature"] == 0.1
    assert kwargs["top_p"] == 0.9
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_max_completion_tokens_forwarded_as_standard_param() -> None:
    backend = make_backend()
    body = request_body(max_completion_tokens=64)
    body.pop("max_tokens")

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = make_response()

        await backend.send_openai_message(body, {})

    kwargs = mock_acomp.await_args.kwargs
    assert kwargs["max_completion_tokens"] == 64
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_max_completion_tokens_forwarded_as_standard_param_streaming() -> None:
    backend = make_backend()
    body = request_body(max_completion_tokens=64)
    body.pop("max_tokens")
    stream = FakeAsyncStream(
        [SimpleNamespace(model_dump=lambda **kwargs: {"id": "chunk1", "choices": []})]
    )

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = stream

        chunks = [chunk async for chunk in backend.stream_openai_message(body, {})]

    kwargs = mock_acomp.await_args.kwargs
    assert kwargs["max_completion_tokens"] == 64
    assert "extra_body" not in kwargs
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_reasoning_params_forwarded_top_level_not_extra_body() -> None:
    # reasoning_effort / max_completion_tokens are litellm-mapped params: they
    # must go as top-level kwargs (litellm maps reasoning_effort -> Anthropic
    # thinking), NOT swept into extra_body, which a Claude target 400s. Regression
    # for the OpenAI-in -> Claude-out cross-protocol reasoning path.
    backend = make_backend()

    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = make_response()

        await backend.send_openai_message(
            request_body(reasoning_effort="low", max_completion_tokens=256),
            {},
        )

    kwargs = mock_acomp.await_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"] == 256
    # Must NOT leak into extra_body (which would reach a non-OpenAI provider raw).
    assert "reasoning_effort" not in kwargs.get("extra_body", {})
    assert "max_completion_tokens" not in kwargs.get("extra_body", {})
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_reasoning_params_forwarded_top_level_streaming() -> None:
    backend = make_backend()
    stream = FakeAsyncStream(
        [SimpleNamespace(model_dump=lambda **kwargs: {"id": "c1", "choices": []})]
    )
    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = stream
        _ = [
            c
            async for c in backend.stream_openai_message(
                request_body(reasoning_effort="high"),
                {},
            )
        ]

    kwargs = mock_acomp.await_args.kwargs
    assert kwargs["reasoning_effort"] == "high"
    assert "reasoning_effort" not in kwargs.get("extra_body", {})
