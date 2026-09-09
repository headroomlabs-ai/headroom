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


def test_to_anthropic_response_empty_choices_returns_empty_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A content-filtered / usage-only upstream response can be 200 with an empty
    # choices list (e.g. Azure OpenAI content filtering). Indexing choices[0]
    # would raise IndexError; the converter must return a valid empty turn, the
    # way the streaming path already skips empty-choice chunks.
    backend, _instance = make_backend(monkeypatch)
    response = make_response(usage=SimpleNamespace(prompt_tokens=9, completion_tokens=0))

    converted = backend._to_anthropic_response(response, "claude-sonnet")

    assert converted["type"] == "message"
    assert converted["role"] == "assistant"
    assert converted["model"] == "claude-sonnet"
    assert converted["content"] == []
    assert converted["stop_reason"] == "end_turn"
    assert converted["usage"] == {"input_tokens": 9, "output_tokens": 0}


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
async def test_send_message_converts_anthropic_tools_and_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic tools/tool_choice must reach any-llm in the OpenAI shape.

    any-llm speaks OpenAI; forwarding the raw Anthropic ``input_schema`` tool and
    the ``{"type": ...}`` tool_choice makes the provider ignore or reject them,
    so the model never calls a tool. Regression for tool use silently not
    working on the any-llm backend.
    """
    backend, instance = make_backend(monkeypatch)
    instance.response = make_response(make_choice("ok", "stop"))

    await backend.send_message(
        {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "look up weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "any"},
        },
        {},
    )

    sent = instance.calls[0]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "look up weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert sent["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_stream_message_converts_anthropic_tools_and_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming request path converts tools/tool_choice the same way."""
    backend, instance = make_backend(monkeypatch)
    instance.response = FakeAsyncStream([])

    _events = [
        event
        async for event in backend.stream_message(
            {
                "model": "claude",
                "messages": [],
                "tools": [{"name": "t", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "tool", "name": "t"},
            },
            {},
        )
    ]

    sent = instance.calls[0]
    assert sent["tools"] == [
        {"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}
    ]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "t"}}


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


def _tool_call_delta(*, index, tc_id=None, name=None, arguments=None):  # noqa: ANN001, ANN202
    """Build an OpenAI-style streaming tool_call delta chunk."""
    func = SimpleNamespace(name=name, arguments=arguments)
    tc = SimpleNamespace(index=index, id=tc_id, function=func)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=[tc]), finish_reason=None)]
    )


@pytest.mark.asyncio
async def test_stream_message_emits_tool_use_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool call streamed over any-llm must surface as an Anthropic tool_use block.

    Regression: the streamer only handled text deltas, so ``tools`` were
    forwarded upstream but any tool call the model streamed back was dropped and
    the client saw an empty turn with stop_reason=end_turn. The block must open,
    stream its arguments as input_json_delta, and the turn must end tool_use.
    """
    backend, instance = make_backend(monkeypatch)
    instance.response = FakeAsyncStream(
        [
            _tool_call_delta(index=0, tc_id="call_abc", name="get_weather"),
            _tool_call_delta(index=0, arguments='{"city":'),
            _tool_call_delta(index=0, arguments='"paris"}'),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(), finish_reason="tool_calls")]
            ),
        ]
    )

    events = [
        event async for event in backend.stream_message({"model": "claude", "messages": []}, {})
    ]
    types = [e.event_type for e in events]

    # The tool call is buffered and flushed as one complete block: start, a
    # single input_json_delta with the reassembled arguments, then stop.
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    start = next(e for e in events if e.event_type == "content_block_start")
    assert start.data["content_block"]["type"] == "tool_use"
    assert start.data["content_block"]["id"] == "call_abc"
    assert start.data["content_block"]["name"] == "get_weather"

    arg_deltas = [e for e in events if e.event_type == "content_block_delta"]
    assert [d.data["delta"]["type"] for d in arg_deltas] == ["input_json_delta"]
    joined = "".join(d.data["delta"]["partial_json"] for d in arg_deltas)
    assert joined == '{"city":"paris"}'

    message_delta = next(e for e in events if e.event_type == "message_delta")
    assert message_delta.data["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_stream_message_handles_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interleaved parallel tool calls must produce valid, disjoint Anthropic blocks.

    OpenAI can introduce two tool indices in one chunk and then stream argument
    fragments for each across later chunks. Each Anthropic tool_use block must be
    fully framed (exactly one start and stop, arguments reassembled) with no
    delta emitted after that block's stop.
    """
    backend, instance = make_backend(monkeypatch)
    instance.response = FakeAsyncStream(
        [
            # One chunk introduces BOTH tool indices at once.
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_0",
                                    function=SimpleNamespace(name="alpha", arguments='{"a":'),
                                ),
                                SimpleNamespace(
                                    index=1,
                                    id="call_1",
                                    function=SimpleNamespace(name="beta", arguments='{"b":'),
                                ),
                            ]
                        ),
                        finish_reason=None,
                    )
                ]
            ),
            # Interleaved argument fragments: index 0, then index 1.
            _tool_call_delta(index=0, arguments="1}"),
            _tool_call_delta(index=1, arguments="2}"),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(), finish_reason="tool_calls")]
            ),
        ]
    )

    events = [
        event async for event in backend.stream_message({"model": "claude", "messages": []}, {})
    ]

    # Each tool block index must have exactly one start and one stop, and no
    # delta may appear after that index's stop.
    stopped: set[int] = set()
    starts: dict[int, int] = {}
    stops: dict[int, int] = {}
    args: dict[int, str] = {}
    for e in events:
        if e.event_type == "content_block_start":
            idx = e.data["index"]
            starts[idx] = starts.get(idx, 0) + 1
            assert e.data["content_block"]["type"] == "tool_use"
        elif e.event_type == "content_block_delta":
            idx = e.data["index"]
            assert idx not in stopped, f"delta for block {idx} after its stop"
            args[idx] = args.get(idx, "") + e.data["delta"]["partial_json"]
        elif e.event_type == "content_block_stop":
            idx = e.data["index"]
            stops[idx] = stops.get(idx, 0) + 1
            stopped.add(idx)

    assert starts == {0: 1, 1: 1}
    assert stops == {0: 1, 1: 1}
    assert args == {0: '{"a":1}', 1: '{"b":2}'}

    block0 = next(
        e for e in events if e.event_type == "content_block_start" and e.data["index"] == 0
    )
    block1 = next(
        e for e in events if e.event_type == "content_block_start" and e.data["index"] == 1
    )
    assert block0.data["content_block"]["name"] == "alpha"
    assert block0.data["content_block"]["id"] == "call_0"
    assert block1.data["content_block"]["name"] == "beta"
    assert block1.data["content_block"]["id"] == "call_1"


@pytest.mark.asyncio
async def test_stream_message_maps_length_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated (length) text stream must report stop_reason=max_tokens."""
    backend, instance = make_backend(monkeypatch)
    instance.response = FakeAsyncStream(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason=None)]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(), finish_reason="length")]
            ),
        ]
    )

    events = [
        event async for event in backend.stream_message({"model": "claude", "messages": []}, {})
    ]
    message_delta = next(e for e in events if e.event_type == "message_delta")
    assert message_delta.data["delta"]["stop_reason"] == "max_tokens"


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
