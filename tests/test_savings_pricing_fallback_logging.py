"""Savings pricing fallbacks are debuggable without flooding or leaking into the log.

An unpriced model is expected (local models, gateway aliases) and the estimators
run on every request, so it is logged once per model. Unexpected pricing errors
are logged by exception type and code location only: their text can carry a
credential (a provider error quoting an API key) and must not reach any level.
"""

from __future__ import annotations

import logging

import pytest

import headroom.proxy.savings_tracker as st
from headroom.log_safety import WarnOnce

CREDENTIAL_CANARY = "sk-ant-CREDENTIAL-CANARY-91c2"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _CredentialEchoingPrice:
    """A price value whose conversion fails with a message quoting a key."""

    def __float__(self) -> float:
        raise ValueError(f"bad price from provider config api_key={CREDENTIAL_CANARY}")


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch):
    class _FakeLiteLLM:
        model_cost = {"weird-priced-model": {"input_cost_per_token": _CredentialEchoingPrice()}}

        @staticmethod
        def cost_per_token(**_kwargs):
            raise Exception(f"This model isn't mapped yet. key={CREDENTIAL_CANARY}")

    monkeypatch.setattr(st, "_get_litellm_module", lambda: _FakeLiteLLM())
    monkeypatch.setattr(st, "_unpriced_models_logged", WarnOnce(256, "test unpriced models"))
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    st._resolve_litellm_model.cache_clear()
    handler = _Capture()
    old_level = st.logger.level
    st.logger.addHandler(handler)
    st.logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        st.logger.removeHandler(handler)
        st.logger.setLevel(old_level)
        st._resolve_litellm_model.cache_clear()


def _assert_no_canary(records: list[logging.LogRecord]) -> None:
    formatter = logging.Formatter()
    for record in records:
        assert CREDENTIAL_CANARY not in formatter.format(record)


def test_unpriced_model_logs_once(captured) -> None:
    for _ in range(3):
        assert st._estimate_compression_savings_usd("local-llama-7b", 1000) == pytest.approx(
            1000 * st.DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
        )
        st._estimate_output_savings_usd("local-llama-7b", 1000)
        st._estimate_input_cost_usd("local-llama-7b", 1000)

    unpriced = [r for r in captured if "No litellm" in r.getMessage()]
    assert len(unpriced) == 1
    assert "model='local-llama-7b'" in unpriced[0].getMessage()
    assert all(r.exc_info is None for r in captured)
    # The litellm probe error quoted a key; it never reaches the log.
    _assert_no_canary(captured)


def test_credential_bearing_pricing_error_is_described_not_quoted(captured) -> None:
    assert st._estimate_compression_savings_usd("weird-priced-model", 1000) == pytest.approx(
        1000 * st.DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
    )

    failures = [
        r for r in captured if "fallback rate for model='weird-priced-model'" in r.getMessage()
    ]
    assert len(failures) == 1
    assert "ValueError" in failures[0].getMessage()
    assert not any("No litellm" in r.getMessage() for r in captured)
    _assert_no_canary(captured)


def test_unpriced_model_not_marked_when_debug_disabled(captured) -> None:
    st.logger.setLevel(logging.INFO)
    st._estimate_compression_savings_usd("quiet-model", 10)
    st.logger.setLevel(logging.DEBUG)
    st._estimate_compression_savings_usd("quiet-model", 10)
    assert [
        r for r in captured if "quiet-model" in r.getMessage() and "No litellm" in r.getMessage()
    ]
