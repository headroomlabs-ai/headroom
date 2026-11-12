"""HuggingFaceTokenizer.count_message must never return a negative count (GH #4274).

The base ``count_message`` subtracts a fixed three-token reply overhead from
``count_messages([message])``. For an HF tokenizer with a chat template,
``count_messages`` renders the template with ``add_generation_prompt=True``, so
its overhead is whatever the template defines, not three tokens. A template
that adds a one-token generation prompt made an empty user message count as
``2 - 3 = -1``, which then fed a negative cost into message budgets.

These tests use a real in-memory ``PreTrainedTokenizerFast`` so the chat
template is rendered by transformers itself rather than by a test double.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from headroom.tokenizers.huggingface import HuggingFaceTokenizer

# One EOS per message, plus a one-token generation prompt when requested.
_EOS_PER_MESSAGE_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}<eos>{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


def _real_tokenizer(chat_template: str | None) -> Any:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    engine = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "<eos>": 1, "<assistant>": 2, "hello": 3, "world": 4},
            unk_token="[UNK]",
        )
    )
    engine.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=engine,
        unk_token="[UNK]",
        eos_token="<eos>",
        additional_special_tokens=["<assistant>"],
    )
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    return tokenizer


def _counter(chat_template: str | None = _EOS_PER_MESSAGE_TEMPLATE) -> HuggingFaceTokenizer:
    counter = HuggingFaceTokenizer("llama-3-8b")
    # Inject the loaded tokenizer so the test never touches the HF hub.
    counter._tokenizer = _real_tokenizer(chat_template)
    return counter


def test_empty_message_counts_template_tokens_not_negative() -> None:
    counter = _counter()
    message = {"role": "user", "content": ""}

    reference = counter.tokenizer.apply_chat_template(
        [message], tokenize=True, add_generation_prompt=False, return_dict=False
    )
    assert reference == [1]  # the template emits a single EOS for this message

    assert counter.count_message(message) == 1


def test_count_message_matches_single_message_template_render() -> None:
    counter = _counter()
    message = {"role": "user", "content": "hello world"}

    # "hello", "world", "<eos>" -- the generation prompt is not part of a message.
    assert counter.count_message(message) == 3


def test_count_messages_still_includes_generation_prompt() -> None:
    counter = _counter()
    messages = [{"role": "user", "content": ""}, {"role": "user", "content": "hello"}]

    # "<eos>", "hello", "<eos>", "<assistant>"
    assert counter.count_messages(messages) == 4


def test_count_message_is_clamped_when_template_render_fails() -> None:
    # A template that only renders with a generation prompt forces the base
    # count-minus-reply-overhead path, which gives 2 - 3 for this message.
    template = (
        "{% if not add_generation_prompt %}"
        "{{ raise_exception('generation prompt required') }}"
        "{% endif %}" + _EOS_PER_MESSAGE_TEMPLATE
    )
    counter = _counter(template)

    assert counter.count_message({"role": "user", "content": ""}) >= 0


def test_count_message_without_chat_template_uses_base_estimate() -> None:
    counter = _counter(chat_template=None)
    message = {"role": "user", "content": "hello world"}

    # No template: the per-message estimate (overhead + role + content) applies.
    count = counter.count_message(message)
    assert count == (
        HuggingFaceTokenizer.MESSAGE_OVERHEAD
        + counter.count_text("user")
        + counter.count_text("hello world")
    )
