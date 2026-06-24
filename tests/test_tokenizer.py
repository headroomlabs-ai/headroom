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
