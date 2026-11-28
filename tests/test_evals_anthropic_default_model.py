"""The Anthropic evals default to a served model and parse its responses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.evals.batch_compression_eval import BatchCompressionEvaluator, BatchRequest
from headroom.evals.runners.before_after import BeforeAfterRunner, LLMConfig


class _Messages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        # A thinking block can precede the answer on Claude 5 models.
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="42"),
            ]
        )


def _before_after():
    runner = BeforeAfterRunner.__new__(BeforeAfterRunner)
    runner.llm_config = LLMConfig()
    runner._llm_client = SimpleNamespace(messages=_Messages())
    return runner, lambda: runner._call_llm("context", "question")


def _batch():
    evaluator = BatchCompressionEvaluator.__new__(BatchCompressionEvaluator)
    evaluator.provider = "anthropic"
    evaluator.model = evaluator._get_default_model("anthropic")
    evaluator._llm_client = SimpleNamespace(messages=_Messages())
    return evaluator, lambda: evaluator._call_llm([{"role": "user", "content": "q"}])


@pytest.mark.parametrize("build", [_before_after, _batch])
def test_default_model_call_reads_text_after_thinking(build):
    owner, call = build()

    assert call() == "42"
    sent = owner._llm_client.messages.kwargs
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["thinking"] == {"type": "between_tools"}
    assert "temperature" not in sent


def test_batch_request_default_omits_sampling_parameters():
    params = BatchRequest(custom_id="a", messages=[]).to_anthropic_format()["params"]

    assert params["model"] == "claude-sonnet-5-5"
    assert "temperature" not in params


def test_explicit_older_model_request_is_unchanged():
    request = BatchRequest(custom_id="a", messages=[], model="claude-sonnet-4-6")
    params = request.to_anthropic_format()["params"]

    assert params["temperature"] == 0.0
    assert "thinking" not in params
