"""Usage must cover every completed Anthropic CCR round, including partial exits."""

import asyncio
import copy

import pytest

from headroom.cache.backends import InMemoryBackend
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.response_handler import (
    CCRResponseHandler,
    ResponseHandlerConfig,
    _combine_anthropic_usage,
)


def message(n, retrieve=None):
    return {
        "type": "message",
        "content": (
            [
                {
                    "type": "tool_use",
                    "id": str(n),
                    "name": "headroom_retrieve",
                    "input": {"hash": retrieve},
                }
            ]
            if retrieve
            else [{"type": "text", "text": "done"}]
        ),
        "usage": {
            "input_tokens": 11 * n,
            "output_tokens": 7 * n,
            "cache_read_input_tokens": 13 * n,
            "cache_creation_input_tokens": 17 * n,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 8 * n,
                "ephemeral_1h_input_tokens": 9 * n,
            },
        },
    }


@pytest.mark.parametrize("calls", [1, 2, 3])
def test_counts_every_completed_call_without_mutation(calls):
    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    key = store.store(original="synthetic original", compressed="{}")
    responses = [message(n, key if n < calls else None) for n in range(1, calls + 1)]
    original = copy.deepcopy(responses)
    pending = iter(responses[1:])

    async def continuation(messages, tools):
        assert "synthetic original" in messages[-1]["content"][0]["content"]
        return next(pending)

    result = asyncio.run(
        CCRResponseHandler().handle_response(responses[0], [], [], continuation, "anthropic")
    )
    expected = message(sum(range(1, calls + 1)))["usage"]
    assert result["usage"] == expected
    assert responses == original
    reset_compression_store()


@pytest.mark.parametrize("bad", [None, True, -1, "7"])
def test_unknown_counter_stays_unknown_across_rounds(bad):
    initial = message(1)
    initial["usage"]["output_tokens"] = bad
    result = _combine_anthropic_usage(initial, message(2))
    assert result["usage"]["output_tokens"] is None
    assert _combine_anthropic_usage(result, message(3))["usage"]["output_tokens"] is None


def test_missing_usage_is_not_a_complete_final_call_total():
    for missing in ({}, {"usage": None}, {"usage": {}}):
        result = _combine_anthropic_usage(missing, message(2))
        assert result["usage"] is None or result["usage"]["input_tokens"] is None


@pytest.mark.parametrize("limited", [False, True])
def test_partial_exit_retains_only_completed_call_usage(limited):
    reset_compression_store()
    key = get_compression_store(backend=InMemoryBackend()).store(
        original="synthetic", compressed="{}"
    )
    calls = 0

    async def continuation(messages, tools):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic failure, no provider response")
        return message(2, key)

    handler = CCRResponseHandler(ResponseHandlerConfig(max_retrieval_rounds=1 if limited else 3))
    result = asyncio.run(
        handler.handle_response(message(1, key), [], [], continuation, "anthropic")
    )
    assert result["usage"] == message(3)["usage"]
    assert result["content"][0]["type"] == "tool_use"
    reset_compression_store()
