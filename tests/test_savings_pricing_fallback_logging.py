"""Savings pricing fallbacks must be debuggable without flooding the debug log.

An unpriced model is expected (local models, gateway aliases) and the estimators
run on every request, so it is logged once per model and without a traceback.
A genuinely broken price entry still logs with its traceback.
"""

from __future__ import annotations

import logging

import pytest

import headroom.proxy.savings_tracker as st


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch):
    class _FakeLiteLLM:
        model_cost = {"weird-priced-model": {"input_cost_per_token": "not-a-number"}}

        @staticmethod
        def cost_per_token(**_kwargs):
            raise Exception("This model isn't mapped yet.")

    monkeypatch.setattr(st, "_get_litellm_module", lambda: _FakeLiteLLM())
    st._unpriced_models_logged.clear()
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
        st._unpriced_models_logged.clear()


def test_unpriced_model_logs_once_without_traceback(captured) -> None:
    for _ in range(3):
        assert st._estimate_compression_savings_usd("local-llama-7b", 1000) == pytest.approx(
            1000 * st.DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
        )
        st._estimate_output_savings_usd("local-llama-7b", 1000)
        st._estimate_input_cost_usd("local-llama-7b", 1000)

    unpriced = [r for r in captured if "No litellm" in r.getMessage()]
    assert len(unpriced) == 1
    assert "model=local-llama-7b" in unpriced[0].getMessage()
    assert all(r.exc_info is None for r in captured)


def test_broken_price_entry_logs_with_traceback(captured) -> None:
    assert st._estimate_compression_savings_usd("weird-priced-model", 1000) == pytest.approx(
        1000 * st.DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
    )

    failures = [
        r for r in captured if "fallback rate for model=weird-priced-model" in r.getMessage()
    ]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert not any("No litellm" in r.getMessage() for r in captured)


def test_unpriced_model_set_is_bounded(captured) -> None:
    for i in range(st._UNPRICED_MODELS_LOGGED_MAX + 10):
        st._estimate_compression_savings_usd(f"alias-{i}", 10)
    assert len(st._unpriced_models_logged) <= st._UNPRICED_MODELS_LOGGED_MAX


def test_unpriced_model_not_marked_when_debug_disabled(captured) -> None:
    st.logger.setLevel(logging.INFO)
    st._estimate_compression_savings_usd("quiet-model", 10)
    assert "quiet-model" not in st._unpriced_models_logged
