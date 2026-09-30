"""Non-streaming LiteLLM responses must surface Bedrock cache token usage (GH #1345).

LiteLLM reports ``prompt_tokens`` as the total prompt size including cached
tokens, while the Anthropic response shape expects ``input_tokens`` to exclude
cache reads/writes and to carry ``cache_read_input_tokens`` /
``cache_creation_input_tokens`` alongside. The streaming and OpenAI paths
already map these fields; the non-streaming ``complete_message`` path dropped
them, so a working Bedrock prompt cache was indistinguishable from a broken
one for non-streaming clients.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

litellm_backend = pytest.importorskip("headroom.backends.litellm")
_anthropic_usage_from_litellm = litellm_backend._anthropic_usage_from_litellm


def test_plain_usage_without_cache_fields() -> None:
    usage = _anthropic_usage_from_litellm(SimpleNamespace(prompt_tokens=100, completion_tokens=7))
    assert usage == {"input_tokens": 100, "output_tokens": 7}


def test_cache_read_surfaced_and_input_excludes_cached() -> None:
    usage = _anthropic_usage_from_litellm(
        SimpleNamespace(
            prompt_tokens=1213,
            completion_tokens=4,
            cache_read_input_tokens=1202,
            cache_creation_input_tokens=0,
        )
    )
    assert usage["input_tokens"] == 11
    assert usage["cache_read_input_tokens"] == 1202
    assert usage["cache_creation_input_tokens"] == 0


def test_cache_write_on_first_call() -> None:
    usage = _anthropic_usage_from_litellm(
        SimpleNamespace(
            prompt_tokens=1237,
            completion_tokens=4,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1226,
        )
    )
    assert usage["input_tokens"] == 11
    assert usage["cache_creation_input_tokens"] == 1226


def test_prompt_tokens_details_fallback() -> None:
    usage = _anthropic_usage_from_litellm(
        SimpleNamespace(
            prompt_tokens=1213,
            completion_tokens=4,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1202, cache_creation_tokens=0),
        )
    )
    assert usage["input_tokens"] == 11
    assert usage["cache_read_input_tokens"] == 1202


def test_input_tokens_never_negative() -> None:
    usage = _anthropic_usage_from_litellm(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1,
            cache_read_input_tokens=15,
        )
    )
    assert usage["input_tokens"] == 0


def test_output_tokens_none_coerced_to_zero() -> None:
    # A provider can carry the completion_tokens attribute but leave it None.
    # The mapping must emit an int (0), not None, so RequestOutcome's int
    # contract holds downstream (prometheus does tokens_output_total +=
    # output_tokens, which would raise TypeError on None).
    usage = _anthropic_usage_from_litellm(
        SimpleNamespace(prompt_tokens=100, completion_tokens=None)
    )
    assert usage["output_tokens"] == 0
    assert isinstance(usage["output_tokens"], int)


def test_to_anthropic_response_empty_choices_returns_empty_turn() -> None:
    # A content-filtered / usage-only upstream response can be HTTP 200 with an
    # empty choices list (e.g. Azure OpenAI content filtering). Indexing
    # choices[0] would raise IndexError and 500 the request; the converter must
    # return a valid empty assistant turn, the way the streaming path already
    # `continue`s on an empty-choice chunk. _to_anthropic_response uses no
    # instance state, so exercise it on a bare instance.
    backend = object.__new__(litellm_backend.LiteLLMBackend)
    response = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=0),
    )

    converted = backend._to_anthropic_response(response, "claude-sonnet")

    assert converted["type"] == "message"
    assert converted["role"] == "assistant"
    assert converted["model"] == "claude-sonnet"
    assert converted["content"] == []
    assert converted["stop_reason"] == "end_turn"
    assert converted["usage"]["input_tokens"] == 42
    assert converted["usage"]["output_tokens"] == 0
