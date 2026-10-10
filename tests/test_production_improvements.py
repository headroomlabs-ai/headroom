"""Focused coverage for opt-in production compression controls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from headroom.compress import compress
from headroom.config import HeadroomConfig, TransformResult
from headroom.transforms.kompress_compressor import _get_diversity_lambda
from headroom.transforms.pipeline import TransformPipeline


def test_semantic_scoring_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_SEMANTIC_SCORE_ENABLED", raising=False)
    messages = [{"role": "user", "content": "word " * 1000}]
    with patch(
        "headroom.transforms.semantic_scorer.score_messages",
        side_effect=AssertionError("semantic scorer called"),
    ):
        result = compress(messages, optimize=False)
    assert result.semantic_score is None


def test_semantic_scoring_failure_is_fail_open(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SEMANTIC_SCORE_ENABLED", "true")
    messages = [{"role": "user", "content": "word " * 1000}]
    compressed = [{"role": "user", "content": "short"}]
    pipeline_result = TransformResult(
        messages=compressed,
        tokens_before=1000,
        tokens_after=10,
        transforms_applied=["test:compressed"],
    )
    with (
        patch("headroom.compress._get_pipeline") as get_pipeline,
        patch("headroom.transforms.semantic_scorer.score_messages", side_effect=RuntimeError),
    ):
        get_pipeline.return_value.apply.return_value = pipeline_result
        result = compress(messages)
    assert result.messages == compressed
    assert result.semantic_score is None


def test_short_input_guard_reports_skip() -> None:
    pipeline = TransformPipeline(HeadroomConfig(min_input_tokens=100))
    messages = [{"role": "user", "content": "short"}]
    result = pipeline.apply(messages, "gpt-4o", model_limit=128_000)
    assert result.messages == messages
    assert result.transforms_applied == ["pipeline:min_input_tokens_skip"]


def test_negative_min_input_tokens_clamps_to_disabled() -> None:
    config = HeadroomConfig(min_input_tokens=-1)
    assert config.min_input_tokens == 0
    assert TransformPipeline(config)._min_input_tokens == 0


def test_min_input_guard_is_disabled_by_default() -> None:
    config = HeadroomConfig()
    assert config.min_input_tokens == 0
    assert TransformPipeline(config)._min_input_tokens == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("-1", 0.0), ("0", 0.0), ("0.4", 0.4), ("2", 1.0), ("invalid", 0.7)],
)
def test_diversity_lambda_is_validated(monkeypatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("HEADROOM_DIVERSITY_LAMBDA", raw)
    assert _get_diversity_lambda() == expected
