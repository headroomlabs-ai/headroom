from __future__ import annotations

from typing import Any

from headroom.tokenizer import Tokenizer, count_tokens_messages, count_tokens_text


class FakeTokenCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def count_text(self, text: str) -> int:
        self.calls.append(("text", text))
        return len(text.split())

    def count_message(self, message: dict[str, Any]) -> int:
        self.calls.append(("message", message))
        return len(str(message.get("content", "")).split())

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        self.calls.append(("messages", messages))
        return sum(len(str(msg.get("content", "")).split()) for msg in messages)


def test_claude_priced_with_real_bpe_not_char_estimate() -> None:
    """Claude has no public tokenizer, so we price it against a real BPE
    (tiktoken o200k_base) instead of a content-adaptive character estimate —
    otherwise before/after counts drift between components and compressing text
    can appear to *increase* tokens. A tool_result fold must always register as
    a reduction; and when the vocab is available the count is the exact o200k
    count (proving it is a real BPE, not a chars/token ratio)."""
    from headroom.tokenizers import get_tokenizer

    tok = get_tokenizer("claude-opus-4-8")

    long_msg = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": "alpha " * 300}],
        }
    ]
    short_msg = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": "alpha " * 3}],
        }
    ]
    assert tok.count_messages(long_msg) > tok.count_messages(short_msg)  # fold visible

    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
    except Exception:  # vocab unavailable → estimator fallback; monotonicity above still holds
        return
    sample = "The quick brown fox jumps over the lazy dog. " * 10
    assert tok.count_text(sample) == len(enc.encode(sample))


def test_tokenizer_delegates_to_counter() -> None:
    counter = FakeTokenCounter()
    tokenizer = Tokenizer(counter, model="gpt-4o")

    assert tokenizer.model == "gpt-4o"
    assert tokenizer.available is True
    assert tokenizer.count_text("hello world") == 2
    assert tokenizer.count_message({"role": "user", "content": "three word text"}) == 3
    assert tokenizer.count_messages([{"content": "one two"}, {"content": "three"}]) == 3
    assert counter.calls == [
        ("text", "hello world"),
        ("message", {"role": "user", "content": "three word text"}),
        ("messages", [{"content": "one two"}, {"content": "three"}]),
    ]


def test_tokenizer_convenience_functions() -> None:
    counter = FakeTokenCounter()
    messages = [{"content": "one"}, {"content": "two three"}]

    assert count_tokens_text("alpha beta gamma", counter) == 3
    assert count_tokens_messages(messages, counter) == 3
    assert counter.calls == [
        ("text", "alpha beta gamma"),
        ("messages", messages),
    ]
