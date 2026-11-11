"""The unknown-model warning must log the client-supplied model id as one bounded token."""

from __future__ import annotations

import logging

from headroom.providers import openai as openai_provider


def test_unknown_model_warning_is_one_line(caplog) -> None:
    provider = openai_provider.OpenAIProvider()
    openai_provider._UNKNOWN_MODEL_WARNINGS.discard("odd\nmodel")
    with caplog.at_level(logging.WARNING, logger=openai_provider.logger.name):
        provider._warn_unknown_model("odd\nmodel", 128000, "pattern match")
    messages = [r.getMessage() for r in caplog.records if "Unknown OpenAI model" in r.getMessage()]
    assert messages
    assert all("\n" not in m for m in messages)
