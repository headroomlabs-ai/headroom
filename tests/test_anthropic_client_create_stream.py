"""The SDK wrapper preserves Anthropic create/stream entrypoint contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from headroom.client import Messages
from headroom.providers.registry import call_client_transport


@pytest.mark.parametrize("stream", [False, True])
def test_messages_create_consumes_explicit_stream_flag(stream):
    client = SimpleNamespace(_create=Mock(return_value="response"))
    assert Messages(client).create(model="claude", messages=[], stream=stream) == "response"
    assert client._create.call_args.kwargs["stream"] is stream
    assert client._create.call_args.kwargs["api_style"] == "anthropic_create"


def test_create_stream_uses_native_create_iterator_not_stream_manager():
    events = iter([{"type": "message_start"}, {"type": "message_stop"}])
    original = SimpleNamespace(
        messages=SimpleNamespace(create=Mock(return_value=events), stream=Mock())
    )
    storage = SimpleNamespace(save=Mock())
    client = SimpleNamespace(
        _original=original,
        _storage=storage,
        _wrap_stream=Mock(side_effect=lambda value, metrics: value),
    )
    metrics = SimpleNamespace(tokens_output=0, cached_tokens=0)

    result = call_client_transport(
        "anthropic_create",
        client,
        model="claude",
        messages=[],
        stream=True,
        metrics=metrics,
        max_tokens=8,
    )

    assert list(result) == [{"type": "message_start"}, {"type": "message_stop"}]
    original.messages.create.assert_called_once_with(
        model="claude", messages=[], stream=True, max_tokens=8
    )
    original.messages.stream.assert_not_called()
    client._wrap_stream.assert_called_once_with(events, metrics)


def test_messages_stream_keeps_native_context_manager():
    manager = object()
    original = SimpleNamespace(
        messages=SimpleNamespace(create=Mock(), stream=Mock(return_value=manager))
    )
    storage = SimpleNamespace(save=Mock())
    client = SimpleNamespace(_original=original, _storage=storage)
    metrics = SimpleNamespace(tokens_output=0, cached_tokens=0)

    result = call_client_transport(
        "anthropic", client, model="claude", messages=[], stream=True, metrics=metrics, max_tokens=8
    )

    assert result is manager
    original.messages.stream.assert_called_once_with(model="claude", messages=[], max_tokens=8)
    original.messages.create.assert_not_called()
    storage.save.assert_called_once_with(metrics)
