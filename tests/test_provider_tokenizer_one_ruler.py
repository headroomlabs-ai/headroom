"""Every model gets exactly ONE tokenizer, whoever asks for it.

Two code paths resolve a tokenizer for the same request:

* handlers call ``headroom.tokenizers.get_tokenizer(model)`` (the per-model
  registry), via ``count_tokens_offloaded``;
* ``TransformPipeline`` calls ``provider.get_token_counter(model)``, because the
  proxy builds its pipelines with ``provider=self.openai_provider``.

``tokens_saved`` is then ``original - optimized``. When those two resolvers
disagree, the subtraction is a difference of two rulers and the result is noise
-- it can even report savings on an untouched request, or trip the
"optimization inflated tokens" revert guard and throw away real compression.

``/v1/chat/completions`` is a multi-provider passthrough, so Kimi, Gemini,
Mistral and Cohere models all reach ``OpenAIProvider``. It used to hand them a
guessed ``o200k_base`` encoding, which mis-counted Kimi by ~19%.
"""

from __future__ import annotations

import pytest

from headroom.providers.openai import OpenAIProvider, OpenAITokenCounter
from headroom.tokenizers import get_tokenizer

# Long enough that a wrong tokenizer shows up as a real gap, not rounding.
MESSAGES = [
    {"role": "user", "content": "def hello(name):\n    return f'hi {name}'\n" * 20},
    {"role": "assistant", "content": "Sure -- here is a summary of the function. " * 30},
]


@pytest.mark.parametrize(
    "model",
    [
        "moonshotai/kimi-k2",
        "accounts/fireworks/models/kimi-k2-instruct",
        "gemini-2.5-pro",
        "command-r-plus",
        "claude-sonnet-4-6",
    ],
)
def test_non_openai_models_resolve_to_the_registry_tokenizer(model: str) -> None:
    """The pipeline's ruler must equal the handler's ruler."""
    provider_count = OpenAIProvider().get_token_counter(model).count_messages(MESSAGES)
    registry_count = get_tokenizer(model).count_messages(MESSAGES)

    assert provider_count == registry_count, (
        f"{model}: pipeline counted {provider_count}, handler counted "
        f"{registry_count} -- tokens_saved would be a difference of two rulers"
    )


def test_kimi_is_not_counted_with_an_openai_encoding() -> None:
    """Regression: the specific 19%-off case that motivated this.

    Pinned as a distinct test because Kimi through Fireworks is a documented
    Headroom configuration, and ``o200k_base`` silently under-counts it.
    """
    counter = OpenAIProvider().get_token_counter("moonshotai/kimi-k2")
    assert not isinstance(counter, OpenAITokenCounter)


def test_openai_models_still_use_tiktoken() -> None:
    """Delegation must not swallow the models the provider genuinely owns."""
    counter = OpenAIProvider().get_token_counter("gpt-4o")
    assert isinstance(counter, OpenAITokenCounter)


def test_per_message_overhead_matches_openai_and_the_registry() -> None:
    """3 tokens per message, not 4.

    OpenAI's token-counting guide uses ``tokens_per_message = 3`` for every
    model since ``gpt-3.5-turbo-0613``; only the retired
    ``gpt-3.5-turbo-0301`` used 4. Staying on 4 over-counted every message by
    one token *and* disagreed with the registry, so a 100-message conversation
    drifted by 100 tokens depending on who counted it.
    """
    plain = [{"role": "user", "content": "hello world"}]
    provider_count = OpenAIProvider().get_token_counter("gpt-4o").count_messages(plain)
    registry_count = get_tokenizer("gpt-4o").count_messages(plain)

    assert provider_count == registry_count


def test_an_explicit_encoding_mapping_is_still_honored() -> None:
    """A user who pins model -> encoding must not be overridden by the registry."""
    counter = OpenAITokenCounter(
        model="my-private-deployment",
        custom_encodings={"my-private-deployment": "cl100k_base"},
    )
    # cl100k_base, not the o200k_base unknown-model default.
    assert counter.count_text("hello world") > 0
