"""Compression safety rails (issue #847).

Three rails, each of which only ever makes compression LESS aggressive:

1. Error-output protection — failed tool calls / error outputs pass
   through ``ContentRouter`` verbatim (string path and content-block path,
   including Anthropic ``is_error: true``), capped by
   ``error_protection_max_chars`` so big error-laden logs still reach
   ``LogCompressor`` (which preserves error lines).
2. Pipeline circuit breaker — after N consecutive pipeline failures,
   ``TransformPipeline.apply`` passes messages through untouched for a
   cooldown window instead of re-running failing transforms.
3. Library inflation guard — ``headroom.compress()`` reverts to the
   original messages when "optimization" inflated tokens, mirroring the
   proxy handlers.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import pytest

from headroom import OpenAIProvider, Tokenizer
from headroom.compress import compress
from headroom.config import HeadroomConfig, TransformResult
from headroom.tokenizer import Tokenizer as TokenizerType
from headroom.transforms.base import Transform
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.pipeline import TransformPipeline

# ``headroom.compress`` the submodule is shadowed by the function of the
# same name re-exported in ``headroom/__init__.py``.
compress_module = importlib.import_module("headroom.compress")

_provider = OpenAIProvider()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(_provider.get_token_counter("gpt-4o"), "gpt-4o")


# A realistic failed-tool-call output: error indicators, > min_tokens (50),
# well under the 8000-char protection cap.
_TRACEBACK = (
    "Traceback (most recent call last):\n"
    + "".join(
        f'  File "/app/services/worker_{i}.py", line {i * 17}, in handle_request\n'
        f"    result = downstream.dispatch(payload, retries={i})\n"
        for i in range(12)
    )
    + "ValueError: connection refused while dispatching payload to upstream "
    "service after 3 retries; check that the worker pool is initialized "
    "before the scheduler starts accepting jobs\n"
)

# Error text with no error-indicator keywords — only the explicit
# Anthropic ``is_error`` flag marks it as a failure.
_NEUTRAL_TOOL_OUTPUT = (
    "The operation finished without producing the expected artifact. "
    "Output directory listing follows.\n"
    + "\n".join(f"entry_{i}.txt  4096 bytes" for i in range(80))
)


# Benign outputs that merely MENTION errors — exactly one distinct
# indicator keyword ("error"). A lax substring gate would exempt these
# from compression (savings regression); the strong gate must not.
_BENIGN_GREP_OUTPUT = (
    "src/error_handler.py:12:def handle_error(code):\n"
    "src/error_handler.py:48:    log_error(code, context)\n"
    + "\n".join(
        f"src/module_{i}.py:{i * 3}:    error_count = metrics.get('error', 0)" for i in range(20)
    )
)

_BENIGN_JSON_OUTPUT = (
    '{"status": "completed", "errors": [], "warnings": [], "items": ['
    + ", ".join(f'{{"id": {i}, "name": "artifact_{i}", "size": {i * 1024}}}' for i in range(30))
    + "]}"
)


def _filler_messages(n: int = 2) -> list[dict[str, Any]]:
    return [{"role": "user", "content": f"step {i}: please continue the task"} for i in range(n)]


class TestErrorOutputProtection:
    def test_string_tool_message_with_error_protected(self, tokenizer: Tokenizer) -> None:
        router = ContentRouter()
        messages = _filler_messages() + [
            {"role": "tool", "tool_call_id": "call_1", "content": _TRACEBACK},
            {"role": "user", "content": "what went wrong?"},
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" in result.transforms_applied
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == _TRACEBACK

    def test_tool_result_block_with_is_error_flag_protected(self, tokenizer: Tokenizer) -> None:
        router = ContentRouter()
        messages = _filler_messages() + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": True,
                        "content": _NEUTRAL_TOOL_OUTPUT,
                    }
                ],
            },
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" in result.transforms_applied
        block = result.messages[-1]["content"][0]
        assert block["content"] == _NEUTRAL_TOOL_OUTPUT

    def test_tool_result_block_with_error_indicators_protected(self, tokenizer: Tokenizer) -> None:
        router = ContentRouter()
        messages = _filler_messages() + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": _TRACEBACK,
                    }
                ],
            },
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" in result.transforms_applied
        block = result.messages[-1]["content"][0]
        assert block["content"] == _TRACEBACK

    def test_single_indicator_string_output_not_protected(self, tokenizer: Tokenizer) -> None:
        """Grep-style output mentioning "error" must not skip compression."""
        router = ContentRouter()
        messages = _filler_messages() + [
            {"role": "tool", "tool_call_id": "call_1", "content": _BENIGN_GREP_OUTPUT},
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" not in result.transforms_applied

    def test_single_indicator_block_not_protected_without_flag(self, tokenizer: Tokenizer) -> None:
        """`"errors": []` JSON without `is_error` must not skip compression."""
        router = ContentRouter()
        messages = _filler_messages() + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_3",
                        "content": _BENIGN_JSON_OUTPUT,
                    }
                ],
            },
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" not in result.transforms_applied

    def test_is_error_flag_alone_protects_single_indicator_block(
        self, tokenizer: Tokenizer
    ) -> None:
        """The explicit `is_error` flag needs no indicator corroboration."""
        router = ContentRouter()
        messages = _filler_messages() + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_4",
                        "is_error": True,
                        "content": _BENIGN_JSON_OUTPUT,
                    }
                ],
            },
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" in result.transforms_applied
        block = result.messages[-1]["content"][0]
        assert block["content"] == _BENIGN_JSON_OUTPUT

    def test_oversized_error_output_falls_through(self, tokenizer: Tokenizer) -> None:
        config = ContentRouterConfig(error_protection_max_chars=100)
        router = ContentRouter(config=config)
        messages = _filler_messages() + [
            {"role": "tool", "tool_call_id": "call_1", "content": _TRACEBACK},
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" not in result.transforms_applied

    def test_protection_disabled_via_config(self, tokenizer: Tokenizer) -> None:
        config = ContentRouterConfig(protect_error_outputs=False)
        router = ContentRouter(config=config)
        messages = _filler_messages() + [
            {"role": "tool", "tool_call_id": "call_1", "content": _TRACEBACK},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": True,
                        "content": _TRACEBACK,
                    }
                ],
            },
        ]
        result = router.apply(messages, tokenizer)
        assert "router:protected:error_output" not in result.transforms_applied


class _FailingTransform(Transform):
    name = "always_fails"

    def apply(
        self, messages: list[dict[str, Any]], tokenizer: TokenizerType, **kwargs: Any
    ) -> TransformResult:
        raise RuntimeError("boom")


class _FlakyTransform(Transform):
    """Fails for the first ``fail_times`` calls, then succeeds."""

    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def apply(
        self, messages: list[dict[str, Any]], tokenizer: TokenizerType, **kwargs: Any
    ) -> TransformResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        tokens = tokenizer.count_messages(messages)
        return TransformResult(
            messages=messages,
            tokens_before=tokens,
            tokens_after=tokens,
            transforms_applied=[],
        )


_MESSAGES = [{"role": "user", "content": "hello there, please summarize the build log"}]


class TestPipelineCircuitBreaker:
    def test_opens_after_threshold_and_passes_through(self) -> None:
        pipeline = TransformPipeline(HeadroomConfig(), transforms=[_FailingTransform()])
        for _ in range(3):
            with pytest.raises(RuntimeError):
                pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        result = pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        assert result.transforms_applied == ["pipeline:circuit_open"]
        assert result.messages == _MESSAGES
        assert result.tokens_before == result.tokens_after

    def test_success_resets_consecutive_failures(self) -> None:
        flaky = _FlakyTransform(fail_times=2)
        pipeline = TransformPipeline(HeadroomConfig(), transforms=[flaky])
        for _ in range(2):
            with pytest.raises(RuntimeError):
                pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        # Third call succeeds — resets the consecutive-failure count.
        pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        # Two more failures still don't reach the threshold of 3.
        flaky.fail_times = flaky.calls + 2
        for _ in range(2):
            with pytest.raises(RuntimeError):
                pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        result = pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        assert result.transforms_applied != ["pipeline:circuit_open"]

    def test_cooldown_expiry_closes_breaker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S", "0.05")
        flaky = _FlakyTransform(fail_times=3)
        pipeline = TransformPipeline(HeadroomConfig(), transforms=[flaky])
        for _ in range(3):
            with pytest.raises(RuntimeError):
                pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        assert pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024).transforms_applied == [
            "pipeline:circuit_open"
        ]
        time.sleep(0.1)
        result = pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        assert result.transforms_applied != ["pipeline:circuit_open"]

    def test_invalid_env_values_fall_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typo'd breaker env vars must not crash proxy startup."""
        monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_THRESHOLD", "three")
        monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S", "1m")
        pipeline = TransformPipeline(HeadroomConfig(), transforms=[_FailingTransform()])
        assert pipeline._breaker_threshold == 3
        assert pipeline._breaker_cooldown_s == 60.0

    def test_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_THRESHOLD", "0")
        pipeline = TransformPipeline(HeadroomConfig(), transforms=[_FailingTransform()])
        for _ in range(5):
            with pytest.raises(RuntimeError):
                pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)
        # Breaker never opens — failures keep propagating.
        with pytest.raises(RuntimeError):
            pipeline.apply(_MESSAGES, model="gpt-4o", model_limit=1024)


class _InflatingPipeline:
    """Fake pipeline whose 'optimization' makes messages bigger."""

    def apply(self, messages: list[dict[str, Any]], **kwargs: Any) -> TransformResult:
        bloated = [{**m, "content": str(m.get("content", "")) + " PADDING" * 50} for m in messages]
        return TransformResult(
            messages=bloated,
            tokens_before=100,
            tokens_after=250,
            transforms_applied=["fake:inflate"],
        )


class TestLibraryInflationGuard:
    def test_inflated_result_reverts_to_originals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(compress_module, "_pipeline", _InflatingPipeline())
        messages = [{"role": "user", "content": "compress this message please"}]
        result = compress(messages, model="gpt-4o")
        assert result.messages == messages
        assert result.transforms_applied == ["inflation_guard:reverted"]
        assert result.tokens_saved == 0
        assert result.compression_ratio == 0.0
