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
