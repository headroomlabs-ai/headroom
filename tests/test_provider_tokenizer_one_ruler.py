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

``/v1/messages`` builds its pipeline with ``AnthropicProvider``, which counted
Claude with ``cl100k_base`` x 1.1 while the registry prices Claude against
``o200k_base``: 403 vs 367 tokens on the same messages (#5037).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from headroom.providers.anthropic import AnthropicProvider, AnthropicTokenCounter
from headroom.providers.openai import OpenAIProvider, OpenAITokenCounter
from headroom.tokenizers import (
    EstimatingTokenCounter,
    TiktokenCounter,
    TokenizerRegistry,
    get_tokenizer,
)

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


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-3-5-haiku-20241022",
    ],
)
def test_claude_pipeline_ruler_equals_handler_ruler(model: str) -> None:
    """``/v1/messages``: the pipeline counts via ``AnthropicProvider``, the
    handler via the registry. Both must be one ruler, as for ``OpenAIProvider``.
    """
    provider_count = AnthropicProvider(warn=False).get_token_counter(model).count_messages(MESSAGES)
    registry_count = get_tokenizer(model).count_messages(MESSAGES)

    assert provider_count == registry_count, (
        f"{model}: pipeline counted {provider_count}, handler counted "
        f"{registry_count} -- tokens_saved would be a difference of two rulers"
    )


def test_issue_5037_repro_claude_counts_match() -> None:
    """The exact repro from #5037 (367 vs 403 on main)."""
    messages = [{"role": "user", "content": "def hello(name):\n    return name.upper()\n" * 40}]
    handler = get_tokenizer("claude-sonnet-4-5")
    pipeline = AnthropicProvider(warn=False).get_token_counter("claude-sonnet-4-5")

    assert pipeline.count_messages(messages) == handler.count_messages(messages)


def test_claude_with_an_api_client_keeps_the_token_count_api() -> None:
    """A client means Claude's real tokenizer: that counter must stay.

    The Token Count API is the accurate ruler, not a second approximation, so
    it is not folded into the registry. Callers that compare against it pass
    the same counter through ``TransformPipeline.apply(token_counter=...)``.
    """

    class _Usage:
        input_tokens = 42

    class _Messages:
        def count_tokens(self, **_: object) -> _Usage:
            return _Usage()

    class _Client:
        messages = _Messages()

    counter = AnthropicProvider(client=_Client()).get_token_counter("claude-sonnet-4-5")

    assert isinstance(counter, AnthropicTokenCounter)
    assert counter.count_messages(MESSAGES) == 42


def test_no_client_warning_still_points_at_the_same_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deferring to the registry must not move the advice warning's location.

    It used to be raised from ``AnthropicTokenCounter.__init__`` with
    ``stacklevel=4``: the caller of whoever asked for the counter.
    """
    import inspect
    import warnings

    import headroom.providers.anthropic as anthropic_mod

    monkeypatch.setattr(anthropic_mod, "_FALLBACK_WARNING_SHOWN", False)

    def asks_for_a_counter() -> None:
        AnthropicProvider(warn=True).get_token_counter("claude-sonnet-4-5")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
        asks_for_a_counter()  # <- the frame the warning must point at

    (warning,) = [w for w in caught if "No client provided" in str(w.message)]
    assert warning.filename == __file__
    assert warning.lineno == expected_line


# ---------------------------------------------------------------------------
# Operator encoding mappings (``openai.encodings`` in HEADROOM_MODEL_LIMITS /
# models.json) must reach every caller of the one ruler, not only
# OpenAIProvider.
# ---------------------------------------------------------------------------

ALIAS = "my-private-deployment"


@pytest.fixture
def _isolated_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fresh registry state and no developer ``models.json``; restored afterwards."""
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty_home))
    registry = TokenizerRegistry()
    for attr in ("_tokenizers", "_model_factories"):
        monkeypatch.setattr(registry, attr, getattr(registry, attr).copy())
    monkeypatch.setattr(registry, "_cache", {})
    # raising=False: the attribute does not exist before the registry learned
    # about configured encodings.
    monkeypatch.setattr(TokenizerRegistry, "_configured_encodings", None, raising=False)


@pytest.fixture
def _alias_mapped_to_cl100k(_isolated_registry: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS",
        json.dumps({"openai": {"encodings": {ALIAS: "cl100k_base"}}}),
    )


def test_registry_honors_an_operator_encoding_mapping(_alias_mapped_to_cl100k: None) -> None:
    """A gateway alias mapped to ``cl100k_base`` is counted with ``cl100k_base``.

    The proxy hands the registry's counter to the pipeline, so the mapping must
    live in the registry to govern the compression gates as well as savings.
    """
    counter = get_tokenizer(ALIAS)
    assert isinstance(counter, TiktokenCounter)
    assert counter.encoding_name == "cl100k_base"
    assert get_tokenizer(ALIAS.upper()) is counter

    # Models without a mapping resolve exactly as before.
    assert get_tokenizer("gpt-4o").encoding_name == "o200k_base"
    assert isinstance(get_tokenizer("unknown-model-xyz"), EstimatingTokenCounter)

    # An explicitly registered tokenizer still beats the mapping.
    custom = EstimatingTokenCounter(chars_per_token=3.0)
    TokenizerRegistry.register(ALIAS, tokenizer=custom)
    assert get_tokenizer(ALIAS) is custom


def test_a_mapped_alias_is_one_ruler_for_provider_and_registry(
    _alias_mapped_to_cl100k: None,
) -> None:
    """Before, OpenAIProvider used the mapping and the registry estimated."""
    provider_count = OpenAIProvider().get_token_counter(ALIAS).count_messages(MESSAGES)
    registry_count = get_tokenizer(ALIAS).count_messages(MESSAGES)

    assert provider_count == registry_count


# ---------------------------------------------------------------------------
# The registry reads only ``openai.encodings``. Every model's counter is
# resolved through it, so a malformed pricing or context-limit section (or a
# malformed encodings section) must leave Claude, Gemini and OpenAI models on
# the counters they get without any configuration.
# ---------------------------------------------------------------------------

CLAUDE = "claude-sonnet-4-6"
GEMINI = "gemini-2.5-pro"


def _assert_ordinary_counters() -> None:
    """The counters these models get with no operator configuration at all."""
    claude = get_tokenizer(CLAUDE)
    assert isinstance(claude, TiktokenCounter), repr(claude)
    assert claude.encoding_name == "o200k_base"
    gemini = get_tokenizer(GEMINI)
    assert repr(gemini) == "EstimatingTokenCounter(chars_per_token=4.0)"
    assert get_tokenizer("gpt-4o").encoding_name == "o200k_base"


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"openai": {"pricing": None}}, id="pricing-null"),
        pytest.param({"openai": {"context_limits": [1]}}, id="context-limits-list"),
        pytest.param({"openai": {"encodings": None}}, id="encodings-not-an-object"),
        pytest.param({"openai": 5}, id="openai-not-an-object"),
        pytest.param("{not json", id="not-json"),
    ],
)
def test_malformed_settings_leave_every_model_on_its_own_counter(
    _isolated_registry: None, monkeypatch: pytest.MonkeyPatch, config: object
) -> None:
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS", config if isinstance(config, str) else json.dumps(config)
    )
    _assert_ordinary_counters()


def test_malformed_settings_in_models_json_leave_every_model_on_its_own_counter(
    _isolated_registry: None,
) -> None:
    from headroom import paths

    models_json = paths.models_config_path()
    models_json.parent.mkdir(parents=True, exist_ok=True)
    models_json.write_text(
        json.dumps({"openai": {"pricing": None, "encodings": {ALIAS: "cl100k_base"}}})
    )

    _assert_ordinary_counters()
    assert get_tokenizer(ALIAS).encoding_name == "cl100k_base"


def test_valid_encodings_beside_malformed_pricing_are_still_honored(
    _isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS",
        json.dumps({"openai": {"pricing": None, "encodings": {ALIAS: "cl100k_base"}}}),
    )

    counter = get_tokenizer(ALIAS)
    assert isinstance(counter, TiktokenCounter), repr(counter)
    assert counter.encoding_name == "cl100k_base"
    _assert_ordinary_counters()


def test_a_bad_encoding_entry_only_affects_its_own_alias(
    _isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS",
        json.dumps(
            {
                "openai": {
                    "encodings": {
                        ALIAS: "cl100k_base",
                        "alias-with-a-number": 5,
                        "alias-with-an-unknown-encoding": "no_such_encoding",
                    }
                }
            }
        ),
    )

    assert get_tokenizer(ALIAS).encoding_name == "cl100k_base"
    assert isinstance(get_tokenizer("alias-with-a-number"), EstimatingTokenCounter)
    assert isinstance(get_tokenizer("alias-with-an-unknown-encoding"), EstimatingTokenCounter)
    _assert_ordinary_counters()


def test_the_configuration_is_read_once_even_when_malformed(
    _isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed read used to stay uncached, so every cache miss re-read it."""
    raw = json.dumps({"openai": {"pricing": None}})
    monkeypatch.setenv("HEADROOM_MODEL_LIMITS", raw)
    reads: list[str] = []
    real_loads = json.loads

    def _counting_loads(s: Any, *args: Any, **kwargs: Any) -> Any:
        if s == raw:
            reads.append(s)
        return real_loads(s, *args, **kwargs)

    monkeypatch.setattr(json, "loads", _counting_loads)

    for model in (CLAUDE, GEMINI, "gpt-4o", "unknown-model-xyz"):
        get_tokenizer(model)
    assert len(reads) == 1

    TokenizerRegistry.clear_cache()
    get_tokenizer(CLAUDE)
    assert len(reads) == 2
