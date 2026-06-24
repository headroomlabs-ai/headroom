"""Waste-signal detection must not discard a finished compression (#296).

On very large Claude Code transcripts the telemetry-only waste-signal re-parse
of the *original* messages can take tens of seconds and blow the Anthropic
compression timeout, making the proxy fail open and forward the original
request even though compression already succeeded. The pipeline now skips that
diagnostic above ``MAX_WASTE_SIGNAL_DETECTION_TOKENS`` so the compression
result stays on the critical path.
"""

from __future__ import annotations

from typing import Any

from headroom.config import HeadroomConfig, TransformResult
from headroom.transforms.base import Transform
from headroom.transforms.pipeline import TransformPipeline


class _FakeTokenizer:
    """Reports a fixed token count for the original messages so the test can
    drive ``tokens_before`` above or below the waste-signal limit."""

    def __init__(self, before: int, after: int) -> None:
        self._before = before
        self._after = after

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        # The compressed message carries the marker "compressed".
        if any(m.get("content") == "compressed" for m in messages):
            return self._after
        return self._before

    def count_text(self, text: Any) -> int:
        return len(str(text))


class _ShrinkTransform(Transform):
    name = "test_shrink"

    def apply(
        self, messages: list[dict[str, Any]], tokenizer: Any, **kwargs: Any
    ) -> TransformResult:
        optimized = [dict(m) for m in messages]
        optimized[-1] = {**optimized[-1], "content": "compressed"}
        return TransformResult(
            messages=optimized,
            tokens_before=tokenizer.count_messages(messages),
            tokens_after=tokenizer.count_messages(optimized),
            transforms_applied=["test:shrink"],
        )


def _run(monkeypatch, *, before: int, after: int, limit: int):
    """Run the pipeline with a stub transform; return (result, parse_called)."""
    pipeline = TransformPipeline(HeadroomConfig())
    pipeline.transforms = [_ShrinkTransform()]
    monkeypatch.setattr(pipeline, "_get_tokenizer", lambda _model: _FakeTokenizer(before, after))

    parse_called = False

    def _tracked_parse_messages(*args: Any, **kwargs: Any):
        nonlocal parse_called
        parse_called = True
        return [], {}, None

    monkeypatch.setattr("headroom.parser.parse_messages", _tracked_parse_messages)

    messages = [{"role": "user", "content": "x" * 1000}]
    result = pipeline.apply(
        messages,
        model="claude-3-5-sonnet",
        model_limit=1_000_000,
        record_metrics=False,
        waste_signal_token_limit=limit,
    )
    return result, parse_called


def test_large_request_skips_waste_signal_and_keeps_compression(monkeypatch):
    """Above the limit, waste-signal detection is skipped but the compression
    result is preserved (the bug discarded it via the timeout)."""
    result, parse_called = _run(monkeypatch, before=200_000, after=180_000, limit=100_000)

    assert parse_called is False, "waste-signal parse must be skipped above the limit"
    assert "test:shrink" in result.transforms_applied
    assert result.tokens_after < result.tokens_before
    assert result.messages[-1]["content"] == "compressed"


def test_small_request_still_runs_waste_signal_detection(monkeypatch):
    """Below the limit, the diagnostic still runs (no behavior change)."""
    _result, parse_called = _run(monkeypatch, before=10_000, after=5_000, limit=100_000)

    assert parse_called is True, "waste-signal parse must still run below the limit"
