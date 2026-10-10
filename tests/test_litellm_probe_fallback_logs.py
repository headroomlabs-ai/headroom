"""A failed LiteLLM probe falls back as before and leaves a DEBUG record naming the model."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import headroom.pricing.counterfactual as counterfactual
import headroom.providers.cohere as cohere
import headroom.providers.openai as openai_provider


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def records() -> Iterator[_Records]:
    # Attach directly: headroom.* loggers may not propagate to caplog's root handler.
    handler = _Records()
    loggers = [logging.getLogger(m.__name__) for m in (openai_provider, cohere, counterfactual)]
    saved = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    yield handler
    for lg, level in zip(loggers, saved):
        lg.removeHandler(handler)
        lg.setLevel(level)


# A dated variant is not in the built-in table, so get_context_limit reaches the LiteLLM probe.
_DATED_GPT4O = "gpt-4o-2099-01-01"


def _failing(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise RuntimeError("model not mapped")


def test_openai_context_limit_probe_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, records: _Records
) -> None:
    provider = openai_provider.OpenAIProvider()
    expected = provider._get_context_limit_manual(_DATED_GPT4O)

    monkeypatch.setattr(
        openai_provider, "_get_litellm_module", lambda: SimpleNamespace(get_model_info=_failing)
    )
    assert provider.get_context_limit(_DATED_GPT4O) == expected
    assert (
        "LiteLLM get_model_info failed for gpt-4o-2099-01-01: model not mapped" in records.messages
    )

    records.messages.clear()
    monkeypatch.setattr(
        openai_provider,
        "_get_litellm_module",
        lambda: SimpleNamespace(get_model_info=lambda m: {"max_input_tokens": 1234}),
    )
    assert provider.get_context_limit(_DATED_GPT4O) == 1234
    assert not [m for m in records.messages if "get_model_info failed" in m]


def test_cohere_context_limit_probe_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, records: _Records
) -> None:
    monkeypatch.setattr(cohere, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(cohere, "litellm", SimpleNamespace(get_model_info=_failing), raising=False)

    assert cohere.CohereProvider().get_context_limit("command-r") == 128000
    assert (
        "LiteLLM get_model_info failed for cohere/command-r: model not mapped" in records.messages
    )
    assert "LiteLLM get_model_info failed for command-r: model not mapped" in records.messages


def test_counterfactual_catalog_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, records: _Records
) -> None:
    broken_catalog = SimpleNamespace(model_cost=SimpleNamespace(get=_failing))
    monkeypatch.setattr(counterfactual, "_litellm", lambda: broken_catalog)
    counterfactual.resolve_rates.cache_clear()
    try:
        assert counterfactual.resolve_rates("probe-fail-model") is None
    finally:
        counterfactual.resolve_rates.cache_clear()
    assert "counterfactual: LiteLLM catalog lookup failed for probe-fail-model" in records.messages
