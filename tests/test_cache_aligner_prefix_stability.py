from __future__ import annotations

from copy import deepcopy

from headroom import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.transforms.cache_aligner import CacheAligner
from headroom.utils import compute_short_hash

_provider = OpenAIProvider()


def _tokenizer() -> Tokenizer:
    counter = _provider.get_token_counter("gpt-4o")
    return Tokenizer(counter, "gpt-4o")


def _claude_code_messages(
    *,
    cached_tool_output: str = "cached tool output v1",
    live_tail: str = "latest live turn",
) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "You are Headroom. Keep the cached prefix stable."},
        {"role": "user", "content": "Summarize the repo state."},
        {"role": "assistant", "content": cached_tool_output},
        {"role": "user", "content": live_tail},
    ]


def test_frozen_prefix_change_flags_prefix_changed() -> None:
    aligner = CacheAligner()
    tokenizer = _tokenizer()

    first = _claude_code_messages(cached_tool_output="cached tool output v1")
    second = _claude_code_messages(cached_tool_output="cached tool output v2")

    result1 = aligner.apply(first, tokenizer, frozen_message_count=3)
    result2 = aligner.apply(second, tokenizer, frozen_message_count=3)

    assert result1.cache_metrics.prefix_changed is False
    assert result2.cache_metrics.prefix_changed is True
    assert result2.cache_metrics.previous_hash == result1.cache_metrics.stable_prefix_hash
    assert result2.cache_metrics.stable_prefix_hash != result1.cache_metrics.stable_prefix_hash


def test_identical_frozen_prefix_is_stable() -> None:
    aligner = CacheAligner()
    tokenizer = _tokenizer()
    messages = _claude_code_messages()

    result1 = aligner.apply(messages, tokenizer, frozen_message_count=3)
    result2 = aligner.apply(deepcopy(messages), tokenizer, frozen_message_count=3)

    assert result1.cache_metrics.prefix_changed is False
    assert result2.cache_metrics.prefix_changed is False
    assert result2.cache_metrics.stable_prefix_hash == result1.cache_metrics.stable_prefix_hash


def test_live_tail_change_does_not_flag() -> None:
    aligner = CacheAligner()
    tokenizer = _tokenizer()

    first = _claude_code_messages(live_tail="latest live turn")
    second = _claude_code_messages(live_tail="different live turn")

    aligner.apply(first, tokenizer, frozen_message_count=3)
    result2 = aligner.apply(second, tokenizer, frozen_message_count=3)

    assert result2.cache_metrics.prefix_changed is False


def test_apply_is_byte_equal_deepcopy() -> None:
    aligner = CacheAligner()
    tokenizer = _tokenizer()
    messages = [
        {
            "role": "system",
            "content": "Keep the transcript stable.",
            "meta": {"source": "test"},
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        },
    ]

    result = aligner.apply(messages, tokenizer, frozen_message_count=1)

    assert result.messages == messages
    assert result.messages is not messages
    assert result.messages[0] is not messages[0]
    assert result.messages[1] is not messages[1]


def test_first_turn_scope_unchanged() -> None:
    aligner = CacheAligner()
    tokenizer = _tokenizer()
    messages = _claude_code_messages()
    system_text = messages[0]["content"]

    result = aligner.apply(messages, tokenizer, frozen_message_count=0)

    assert result.cache_metrics.prefix_changed is False
    assert result.cache_metrics.stable_prefix_hash == compute_short_hash(system_text)
    assert result.cache_metrics.stable_prefix_bytes == len(str(system_text).encode("utf-8"))
    assert result.cache_metrics.stable_prefix_tokens_est == tokenizer.count_text(str(system_text))
