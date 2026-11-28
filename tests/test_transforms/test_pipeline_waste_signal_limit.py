"""Waste-signal detection must not discard a finished compression (#296).

On very large Claude Code transcripts the telemetry-only waste-signal re-parse
of the *original* messages can take tens of seconds and blow the Anthropic
compression timeout, making the proxy fail open and forward the original
request even though compression already succeeded. The pipeline now skips that
diagnostic above ``MAX_WASTE_SIGNAL_DETECTION_TOKENS`` so the compression
result stays on the critical path.
"""

from __future__ import annotations

from types import SimpleNamespace
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


def _run(monkeypatch, *, before: int, after: int, limit: int, defer_waste_signals: bool = False):
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
        defer_waste_signals=defer_waste_signals,
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


def test_defer_waste_signals_skips_inline_parse_even_when_small(monkeypatch):
    """perf review F5: proxy callers set defer_waste_signals=True, which skips
    the inline re-parse regardless of size — it re-derives off-path instead.
    SDK callers never pass this flag (default False), so the two tests above
    already guard that their inline behavior is unchanged."""
    result, parse_called = _run(
        monkeypatch, before=10_000, after=5_000, limit=100_000, defer_waste_signals=True
    )

    assert parse_called is False, "defer_waste_signals must skip the inline parse entirely"
    assert result.waste_signals is None
    assert "test:shrink" in result.transforms_applied
    assert result.messages[-1]["content"] == "compressed"
    # The gate would have parsed inline, so the deferred provider is attached
    # for the caller to run off-path.
    assert result.waste_signals_provider is not None


def test_defer_provider_runs_the_same_parse_lazily(monkeypatch):
    """Calling the deferred provider performs the parse the inline path would
    have done (same messages, same tokenizer), returning the WasteSignals."""
    fake_ws = SimpleNamespace(total=lambda: 2, to_dict=lambda: {"duplicate_tool_result": 2})

    result, _ = _run(
        monkeypatch, before=10_000, after=5_000, limit=100_000, defer_waste_signals=True
    )
    monkeypatch.setattr("headroom.parser.parse_messages", lambda *a, **k: ([], {}, fake_ws))

    assert result.waste_signals_provider() is fake_ws


def test_defer_provider_uses_pipeline_time_snapshots(monkeypatch):
    """Later handler mutations cannot change deferred telemetry inputs."""
    pipeline = TransformPipeline(HeadroomConfig())
    pipeline.transforms = [_ShrinkTransform()]
    monkeypatch.setattr(pipeline, "_get_tokenizer", lambda _model: _FakeTokenizer(10_000, 5_000))
    waste_messages = [{"role": "user", "content": "original waste input"}]
    captured = {}
    fake_ws = SimpleNamespace(total=lambda: 1, to_dict=lambda: {"duplicate_tool_result": 1})

    def _capture(messages, tokenizer, *, compressed_messages):  # noqa: ANN001
        captured["messages"] = messages
        captured["compressed"] = compressed_messages
        return [], {}, fake_ws

    monkeypatch.setattr("headroom.parser.parse_messages", _capture)
    result = pipeline.apply(
        [{"role": "user", "content": "request input"}],
        model="claude-3-5-sonnet",
        model_limit=1_000_000,
        record_metrics=False,
        waste_messages=waste_messages,
        defer_waste_signals=True,
    )
    waste_messages[0]["content"] = "MUTATED WASTE"
    result.messages[0]["content"] = "MUTATED RESULT"

    assert result.waste_signals_provider is not None
    assert result.waste_signals_provider() is fake_ws
    assert captured["messages"][0]["content"] == "original waste input"
    assert captured["compressed"][0]["content"] == "compressed"


def test_defer_provider_absent_when_savings_too_small(monkeypatch):
    """Gate parity: below _MIN_TOKENS_SAVED_FOR_WASTE_SIGNALS the inline path
    would not have parsed, so deferral must not attach a provider either."""
    result, parse_called = _run(
        monkeypatch, before=100, after=99, limit=100_000, defer_waste_signals=True
    )

    assert parse_called is False
    assert result.waste_signals_provider is None


def test_defer_provider_absent_above_size_limit(monkeypatch):
    """Gate parity: the waste_signal_token_limit override is honored under
    deferral exactly as inline (#296 size cap)."""
    result, parse_called = _run(
        monkeypatch, before=200_000, after=180_000, limit=100_000, defer_waste_signals=True
    )

    assert parse_called is False
    assert result.waste_signals_provider is None


def test_pipeline_injects_tokens_before_hint_into_transform_kwargs(monkeypatch):
    """perf review F2/F3 plumbing: apply() must pass the already-computed
    tokens_before to every transform via kwargs, so ContentRouter (Phase 6)
    can reuse it instead of recounting messages itself."""
    pipeline = TransformPipeline(HeadroomConfig())
    tokenizer = _FakeTokenizer(before=12_345, after=12_345)
    monkeypatch.setattr(pipeline, "_get_tokenizer", lambda _model: tokenizer)

    captured_kwargs: dict[str, Any] = {}

    class _CaptureTransform(Transform):
        name = "test_capture"

        def apply(self, messages, tokenizer, **kwargs: Any):  # noqa: ANN001
            captured_kwargs.update(kwargs)
            return TransformResult(
                messages=messages,
                tokens_before=tokenizer.count_messages(messages),
                tokens_after=tokenizer.count_messages(messages),
                transforms_applied=["test:capture"],
            )

    pipeline.transforms = [_CaptureTransform()]

    pipeline.apply(
        [{"role": "user", "content": "x" * 100}],
        model="claude-3-5-sonnet",
        model_limit=1_000_000,
        record_metrics=False,
    )

    assert captured_kwargs.get("tokens_before_hint") == 12_345


def test_tokens_before_hint_refreshes_after_each_transform(monkeypatch):
    """An earlier transform that changes the messages (e.g. the
    HEADROOM_INTERCEPT_ENABLED interceptor before the ContentRouter) must not
    leave later transforms holding the stale pipeline-entry count: the hint is
    refreshed from each transform's reported tokens_after."""
    pipeline = TransformPipeline(HeadroomConfig())
    tokenizer = _FakeTokenizer(before=10_000, after=10_000)
    monkeypatch.setattr(pipeline, "_get_tokenizer", lambda _model: tokenizer)

    hints_seen: list[int] = []

    class _ShrinkReporting(Transform):
        name = "test_shrink_reporting"

        def apply(self, messages, tokenizer, **kwargs: Any):  # noqa: ANN001
            hints_seen.append(kwargs.get("tokens_before_hint"))
            return TransformResult(
                messages=messages,
                tokens_before=10_000,
                tokens_after=4_000,  # removed content
                transforms_applied=["test:shrink_reporting"],
            )

    class _Capture(Transform):
        name = "test_capture_second"

        def apply(self, messages, tokenizer, **kwargs: Any):  # noqa: ANN001
            hints_seen.append(kwargs.get("tokens_before_hint"))
            return TransformResult(
                messages=messages,
                tokens_before=kwargs.get("tokens_before_hint", 0),
                tokens_after=kwargs.get("tokens_before_hint", 0),
                transforms_applied=["test:capture_second"],
            )

    pipeline.transforms = [_ShrinkReporting(), _Capture()]
    pipeline.apply(
        [{"role": "user", "content": "x" * 100}],
        model="claude-3-5-sonnet",
        model_limit=1_000_000,
        record_metrics=False,
    )

    assert hints_seen == [10_000, 4_000]
