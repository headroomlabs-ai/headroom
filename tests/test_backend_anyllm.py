from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.backends import anyllm
from headroom.backends.base import BackendResponse, StreamEvent


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


class FakeAnyLLMInstance:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.response = None
        self.raise_error: Exception | None = None

    async def acompletion(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        return self.response


def make_backend(
    monkeypatch: pytest.MonkeyPatch, provider: str = "groq"
) -> tuple[anyllm.AnyLLMBackend, FakeAnyLLMInstance]:
    fake_instance = FakeAnyLLMInstance()

    class FakeAnyLLM:
        @staticmethod
        def create(requested_provider: str, **kwargs):  # noqa: ANN003
            assert requested_provider == provider
            return fake_instance

    monkeypatch.setattr(anyllm, "ANYLLM_AVAILABLE", True)
    monkeypatch.setattr(anyllm, "AnyLLM", FakeAnyLLM)
    return anyllm.AnyLLMBackend(provider=provider.upper()), fake_instance


def test_init_forwards_api_base_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for #942: custom api_base/api_key must reach AnyLLM.create."""
    fake_instance = FakeAnyLLMInstance()
    create_calls: list[dict[str, object]] = []

    class FakeAnyLLM:
        @staticmethod
        def create(requested_provider: str, **kwargs):  # noqa: ANN003
            create_calls.append({"provider": requested_provider, **kwargs})
            return fake_instance

    monkeypatch.setattr(anyllm, "ANYLLM_AVAILABLE", True)
    monkeypatch.setattr(anyllm, "AnyLLM", FakeAnyLLM)

    backend = anyllm.AnyLLMBackend(
        provider="openai",
        api_key="sk-custom",
        api_base="https://custom-provider.example/v1",
    )

    assert backend.api_base == "https://custom-provider.example/v1"
    assert create_calls == [
        {
            "provider": "openai",
            "api_key": "sk-custom",
            "api_base": "https://custom-provider.example/v1",
        }
    ]


def test_init_omits_unset_api_base_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset overrides must not be forwarded, preserving provider env defaults."""
    fake_instance = FakeAnyLLMInstance()
    create_calls: list[dict[str, object]] = []

    class FakeAnyLLM:
        @staticmethod
        def create(requested_provider: str, **kwargs):  # noqa: ANN003
            create_calls.append({"provider": requested_provider, **kwargs})
            return fake_instance

    monkeypatch.setattr(anyllm, "ANYLLM_AVAILABLE", True)
    monkeypatch.setattr(anyllm, "AnyLLM", FakeAnyLLM)

    anyllm.AnyLLMBackend(provider="openai")

    assert create_calls == [{"provider": "openai"}]


def test_init_treats_empty_overrides_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty-string api_base/api_key must not be forwarded (env var set to "")."""
    fake_instance = FakeAnyLLMInstance()
    create_calls: list[dict[str, object]] = []

    class FakeAnyLLM:
        @staticmethod
        def create(requested_provider: str, **kwargs):  # noqa: ANN003
            create_calls.append({"provider": requested_provider, **kwargs})
            return fake_instance

    monkeypatch.setattr(anyllm, "ANYLLM_AVAILABLE", True)
    monkeypatch.setattr(anyllm, "AnyLLM", FakeAnyLLM)

    backend = anyllm.AnyLLMBackend(provider="openai", api_key="", api_base="")

    assert backend.api_base is None
    assert backend.api_key is None
    assert create_calls == [{"provider": "openai"}]


def make_choice(
    content: str = "hello", finish_reason: str = "stop", tool_calls=None, index: int = 0
):
    return SimpleNamespace(
        index=index,
        finish_reason=finish_reason,
        message=SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls),
    )


def make_response(*choices, usage=None):
    return SimpleNamespace(
        id="resp_123",
        created=123456,
        choices=list(choices),
        usage=usage,
    )


def make_tool_call(tool_id: str, name: str, arguments):
    return SimpleNamespace(id=tool_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_init_raises_without_anyllm() -> None:
    original_available = anyllm.ANYLLM_AVAILABLE
    try:
        anyllm.ANYLLM_AVAILABLE = False
        with pytest.raises(ImportError):
            anyllm.AnyLLMBackend()
    finally:
        anyllm.ANYLLM_AVAILABLE = original_available


def test_init_name_and_basic_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, instance = make_backend(monkeypatch, provider="groq")

    assert backend.provider == "groq"
    assert backend.name == "anyllm-groq"
    assert backend.map_model_id("claude-3-5") == "claude-3-5"
    assert backend.supports_model("anything") is True
    assert backend.llm is instance


def test_convert_content_blocks_and_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _instance = make_backend(monkeypatch)

    assert backend._convert_content_blocks([{"type": "text", "text": "hello"}]) == "hello"
    assert backend._convert_content_blocks(
        [
            {"type": "text", "text": "caption"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"},
            },
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
        ]
    ) == [
        {"type": "text", "text": "caption"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
    ]
    assert backend._convert_content_blocks([{"type": "tool_use", "id": "ignored"}]) == ""

    converted = backend._convert_messages(
        [
            {"role": "user", "content": "plain text"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com"}},
                ],
            },
            {"role": "user", "content": 123},
        ]
    )

    assert converted == [
        {"role": "user", "content": "plain text"},
        {"role": "assistant", "content": "a\nb"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://example.com"}},
            ],
        },
    ]


def test_to_anthropic_response_maps_tool_calls_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _instance = make_backend(monkeypatch)
    response = make_response(
        make_choice(
            content="hello",
            finish_reason="tool_calls",
            tool_calls=[
                make_tool_call("tc1", "memory_save", '{"content":"python"}'),
                make_tool_call("tc2", "memory_search", {"query": "python"}),
            ],
        ),
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
    )

    converted = backend._to_anthropic_response(response, "claude-sonnet")

    assert converted["type"] == "message"
    assert converted["role"] == "assistant"
    assert converted["model"] == "claude-sonnet"
    assert converted["stop_reason"] == "tool_use"
    assert converted["usage"] == {"input_tokens": 12, "output_tokens": 7}
    assert converted["content"][0] == {"type": "text", "text": "hello"}
    assert converted["content"][1]["input"] == {"content": "python"}
    assert converted["content"][2]["input"] == {"query": "python"}


@pytest.mark.asyncio
async def test_send_message_builds_anthropic_response(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, instance = make_backend(monkeypatch)
    instance.response = make_response(
        make_choice("done", "stop"),
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6),
    )

    result = await backend.send_message(
        {
            "model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "system": [{"text": "system rule"}, "extra"],
            "max_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.8,
            "stop_sequences": ["END"],
            "tools": [{"name": "t"}],
            "tool_choice": {"type": "auto"},
        },
        {},
    )

    assert isinstance(result, BackendResponse)
    assert result.status_code == 200
    assert result.headers == {"content-type": "application/json"}
    assert result.body["content"][0]["text"] == "done"
    assert instance.calls[0]["messages"][0] == {"role": "system", "content": "system rule extra"}
    assert instance.calls[0]["stop"] == ["END"]


@pytest.mark.asyncio
async def test_send_message_returns_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, instance = make_backend(monkeypatch)
    instance.raise_error = RuntimeError("authentication api_key missing")

    result = await backend.send_message({"messages": []}, {})

    assert result.status_code == 401
    assert result.body["error"]["type"] == "authentication_error"
    assert result.error == "authentication api_key missing"


@pytest.mark.asyncio
async def test_stream_message_yields_events_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, instance = make_backend(monkeypatch)
    instance.response = FakeAsyncStream(
        [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
            SimpleNamespace(choices=[]),
        ]
    )

    events = [
        event
        async for event in backend.stream_message(
            {"model": "claude", "messages": [], "system": "sys"}, {}
        )
    ]

    assert [event.event_type for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0].data["message"]["model"] == "claude"
    assert events[5].data["usage"] == {"output_tokens": 2}
    assert instance.calls[0]["stream"] is True
    assert instance.calls[0]["messages"][0] == {"role": "system", "content": "sys"}

    backend_error, instance_error = make_backend(monkeypatch, provider="openai")
    instance_error.raise_error = RuntimeError("stream broke")
    error_events = [event async for event in backend_error.stream_message({"messages": []}, {})]
    assert error_events[-1].event_type == "error"
    assert error_events[-1].data["error"]["message"] == "stream broke"


@pytest.mark.asyncio
async def test_send_openai_message_maps_choices_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, instance = make_backend(monkeypatch)
    instance.response = make_response(
        make_choice(
            content="answer",
            finish_reason="stop",
            tool_calls=[
                make_tool_call("tc1", "memory_search", '{"query":"python"}'),
                SimpleNamespace(id="tc2", function=None),
            ],
            index=0,
        ),
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )

    result = await backend.send_openai_message(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["END"],
            "tools": [{"name": "memory"}],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
            "seed": 1,
            "n": 2,
        },
        {},
    )

    assert result.status_code == 200
    assert result.body["object"] == "chat.completion"
    assert (
        result.body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "memory_search"
    )
    assert result.body["choices"][0]["message"]["tool_calls"][1] == {
        "id": "tc2",
        "type": "function",
    }
    assert result.body["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


@pytest.mark.asyncio
async def test_send_openai_message_returns_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, instance = make_backend(monkeypatch)
    instance.raise_error = RuntimeError("model not found")

    result = await backend.send_openai_message({"messages": []}, {})

    assert result.status_code == 404
    assert result.body["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
async def test_stream_openai_message_yields_sse_chunks_and_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, instance = make_backend(monkeypatch)
    instance.response = FakeAsyncStream(
        [
            SimpleNamespace(
                model_dump=lambda **kwargs: {
                    "id": "chunk1",
                    "choices": [{"delta": {"content": "a"}}],
                }
            ),
            SimpleNamespace(
                model_dump=lambda **kwargs: {
                    "id": "chunk2",
                    "choices": [{"delta": {"content": "b"}}],
                }
            ),
        ]
    )

    chunks = [
        chunk
        async for chunk in backend.stream_openai_message(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream_options": {"include_usage": True},
            },
            {},
        )
    ]

    assert chunks[0].startswith("data: {")
    assert chunks[-1] == "data: [DONE]\n\n"
    assert instance.calls[0]["stream"] is True
    assert instance.calls[0]["stream_options"] == {"include_usage": True}

    backend_error, instance_error = make_backend(monkeypatch, provider="anthropic")
    instance_error.raise_error = RuntimeError("rate limit hit")
    error_chunks = [
        chunk async for chunk in backend_error.stream_openai_message({"messages": []}, {})
    ]
    assert '"backend_error"' in error_chunks[0]
    assert error_chunks[-1] == "data: [DONE]\n\n"


def test_error_response_classifies_common_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _instance = make_backend(monkeypatch)

    auth = backend._error_response(RuntimeError("authentication api key missing"))
    rate = backend._error_response(RuntimeError("rate limit exceeded"), openai_format=True)
    model = backend._error_response(RuntimeError("model not found"), openai_format=True)
    generic = backend._error_response(RuntimeError("other error"))

    assert auth.status_code == 401
    assert auth.body["error"]["type"] == "authentication_error"
    assert rate.status_code == 429
    assert rate.body["error"]["type"] == "rate_limit_exceeded"
    assert model.status_code == 404
    assert model.body["error"]["type"] == "model_not_found"
    assert generic.status_code == 500
    assert generic.body["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_close_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _instance = make_backend(monkeypatch)
    assert await backend.close() is None
    assert isinstance(StreamEvent(event_type="message_start", data={}), StreamEvent)
