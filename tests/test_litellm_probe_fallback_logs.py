"""A failed LiteLLM probe falls back as before and logs the model and failure type only.

The probe's exception text comes from LiteLLM and can carry backend configuration or
credentials, so no log record may contain it, at any level, with or without a traceback.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import headroom.pricing.counterfactual as counterfactual
import headroom.providers.cohere as cohere
import headroom.providers.openai as openai_provider

CREDENTIAL_CANARY = "sk-canary-0123456789abcdef"
CONTENT_CANARY = "user-prompt-canary-text"

# A dated variant is not in the built-in table, so get_context_limit reaches the LiteLLM probe.
_DATED_GPT4O = "gpt-4o-2099-01-01"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def formatted(self) -> list[str]:
        # A real Formatter renders exc_info and stack_info, which getMessage() leaves out.
        formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
        return [formatter.format(record) for record in self.records]


@pytest.fixture
def records(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Records]:
    # The content opt-in would put exception text back on purpose; test the default.
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
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


def _failing(*_args: object, **_kwargs: object) -> dict[str, object]:
    try:
        raise KeyError(f"api_key={CREDENTIAL_CANARY}")
    except KeyError as cause:
        raise RuntimeError(
            f"catalog error for {CONTENT_CANARY} at https://u:{CREDENTIAL_CANARY}@x.test"
        ) from cause


def _assert_safe(records: _Records, model: str) -> None:
    lines = records.formatted()
    for line in lines:
        assert CREDENTIAL_CANARY not in line
        assert CONTENT_CANARY not in line
        assert "Traceback" not in line
    failure_lines = [line for line in lines if "failed for" in line and repr(model) in line]
    assert failure_lines, lines
    assert all("RuntimeError" in line for line in failure_lines)


def test_openai_context_limit_probe_failure_is_logged_safely(
    monkeypatch: pytest.MonkeyPatch, records: _Records
) -> None:
    provider = openai_provider.OpenAIProvider()
    expected = provider._get_context_limit_manual(_DATED_GPT4O)

    monkeypatch.setattr(
        openai_provider, "_get_litellm_module", lambda: SimpleNamespace(get_model_info=_failing)
    )
    assert provider.get_context_limit(_DATED_GPT4O) == expected
    _assert_safe(records, _DATED_GPT4O)

    records.records.clear()
    monkeypatch.setattr(
        openai_provider,
        "_get_litellm_module",
        lambda: SimpleNamespace(get_model_info=lambda m: {"max_input_tokens": 1234}),
    )
    assert provider.get_context_limit(_DATED_GPT4O) == 1234
    assert not [line for line in records.formatted() if "get_model_info failed" in line]


def test_cohere_context_limit_probe_failure_is_logged_safely(
    monkeypatch: pytest.MonkeyPatch, records: _Records
) -> None:
    monkeypatch.setattr(cohere, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(cohere, "litellm", SimpleNamespace(get_model_info=_failing), raising=False)

    assert cohere.CohereProvider().get_context_limit("command-r") == 128000
    _assert_safe(records, "cohere/command-r")
    _assert_safe(records, "command-r")


def test_counterfactual_catalog_failure_is_logged_safely(
    monkeypatch: pytest.MonkeyPatch, records: _Records
) -> None:
    broken_catalog = SimpleNamespace(model_cost=SimpleNamespace(get=_failing))
    monkeypatch.setattr(counterfactual, "_litellm", lambda: broken_catalog)
    counterfactual.resolve_rates.cache_clear()
    try:
        assert counterfactual.resolve_rates("probe-fail-model") is None
    finally:
        counterfactual.resolve_rates.cache_clear()
    _assert_safe(records, "probe-fail-model")
