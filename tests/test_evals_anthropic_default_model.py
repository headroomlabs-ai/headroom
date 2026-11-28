"""The Anthropic evals default to a served model and parse its responses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from headroom.cli.evals import evals
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
    # No top-level ``thinking``: anthropic 0.18.0 does not accept it.
    assert sent["extra_body"] == {"thinking": {"type": "between_tools"}}
    assert "thinking" not in sent
    assert "temperature" not in sent


def test_batch_request_default_omits_sampling_parameters():
    params = BatchRequest(custom_id="a", messages=[]).to_anthropic_format()["params"]

    assert params["model"] == "claude-sonnet-5-5"
    assert params["thinking"] == {"type": "between_tools"}
    assert "temperature" not in params


def test_explicit_older_model_request_is_unchanged():
    request = BatchRequest(custom_id="a", messages=[], model="claude-sonnet-4-6")
    params = request.to_anthropic_format()["params"]

    assert params["temperature"] == 0.0
    assert "thinking" not in params


def test_proxy_arm_sends_the_same_thinking_setting_as_the_direct_arm():
    runner, direct = _before_after()
    proxy = _Messages()
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="42"))])
    proxy.create = lambda **kwargs: setattr(proxy, "kwargs", kwargs) or reply
    runner._proxy_client = SimpleNamespace(chat=SimpleNamespace(completions=proxy))

    direct()
    runner._call_llm_via_proxy("context", "question")

    assert proxy.kwargs["extra_body"] == runner._llm_client.messages.kwargs["extra_body"]
    assert "temperature" not in proxy.kwargs


@pytest.mark.parametrize(
    ("answer_model", "expected"),
    [
        (
            "claude-sonnet-5-5",
            {"model": "anthropic/claude-sonnet-5-5", "thinking": {"type": "between_tools"}},
        ),
        ("gpt-4o-mini", {"model": "gpt-4o-mini", "temperature": 0.0}),
    ],
)
def test_memory_answer_and_default_judge_requests(monkeypatch, answer_model, expected):
    litellm = pytest.importorskip("litellm")
    memory = pytest.importorskip("headroom.evals.memory")
    sent = []

    def completion(**kwargs):
        sent.append(kwargs)
        message = SimpleNamespace(content="Score: 3")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Evaluator:
        def __init__(self, answer_fn, llm_judge_fn, config):
            self.answer_fn, self.judge_fn = answer_fn, llm_judge_fn

        async def run(self):
            self.judge_fn("q", "truth", self.answer_fn("q", ["a memory"]))
            return SimpleNamespace(summary=lambda: "done")

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(memory, "LoCoMoEvaluator", Evaluator)

    result = CliRunner().invoke(evals, ["memory", "--answer-model", answer_model, "--llm-judge"])

    assert result.exit_code == 0, result.output
    assert len(sent) == 2
    for kwargs in sent:
        settings = {k: v for k, v in kwargs.items() if k not in ("messages", "max_tokens")}
        assert settings == expected
