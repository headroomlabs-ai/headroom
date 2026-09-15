"""Tests for per-message compression diagnostics (issue #2855).

Verifies that CompressConfig(diagnostics=True) — and the HEADROOM_DIAGNOSTICS=1
env var — populate CompressResult.diagnostics with per-message MessageDecision
objects, and that the compress() function correctly threads collect_diagnostics
through to the pipeline and collects the results.

ContentRouter integration tests (which need the compiled headroom._core extension)
run in CI; this file tests the compress() ↔ pipeline contract via a mock pipeline.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from unittest.mock import patch

from headroom.compress import CompressConfig, compress
from headroom.config import MessageDecision, TransformResult

# ---------------------------------------------------------------------------
# Mock pipeline factory
# ---------------------------------------------------------------------------


def _noop_result(messages: list[dict]) -> TransformResult:
    """TransformResult that passes messages through unchanged, no decisions."""
    tokens = sum(len(str(m.get("content", ""))) for m in messages)
    return TransformResult(
        messages=messages,
        tokens_before=tokens,
        tokens_after=tokens,
        transforms_applied=["router:noop"],
    )


def _result_with_decisions(
    messages: list[dict], decisions: list[MessageDecision]
) -> TransformResult:
    """TransformResult that includes per-message decisions."""
    tokens = sum(len(str(m.get("content", ""))) for m in messages)
    return TransformResult(
        messages=messages,
        tokens_before=tokens,
        tokens_after=tokens,
        transforms_applied=["router:noop"],
        message_decisions=decisions,
    )


class _MockPipeline:
    """Pipeline stub that captures kwargs and returns a configurable result."""

    def __init__(self, decisions: list[MessageDecision] | None = None):
        self.last_kwargs: dict = {}
        self._decisions = decisions or []

    def apply(self, messages, model, **kwargs):
        self.last_kwargs = kwargs
        return _result_with_decisions(messages, self._decisions)


def _fake_otel():
    return SimpleNamespace(record_compression_failure=lambda **kw: None)


# ---------------------------------------------------------------------------
# MessageDecision dataclass
# ---------------------------------------------------------------------------


def test_message_decision_dataclass():
    """MessageDecision is a dataclass with the expected fields."""
    dec = MessageDecision(
        message_index=0,
        role="user",
        tokens_before=100,
        tokens_after=60,
        action="compressed:kompress:0.60",
    )
    assert dec.message_index == 0
    assert dec.role == "user"
    assert dec.tokens_before == 100
    assert dec.tokens_after == 60
    assert dec.action == "compressed:kompress:0.60"


def test_message_decision_importable_from_headroom():
    """MessageDecision is accessible via the headroom top-level package."""
    import headroom

    assert hasattr(headroom, "MessageDecision")
    cls = headroom.MessageDecision
    dec = cls(
        message_index=1,
        role="assistant",
        tokens_before=50,
        tokens_after=50,
        action="passthrough:small",
    )
    assert dec.message_index == 1


# ---------------------------------------------------------------------------
# TransformResult carries message_decisions
# ---------------------------------------------------------------------------


def test_transform_result_has_message_decisions_field():
    """TransformResult has a message_decisions field defaulting to []."""
    tr = TransformResult(
        messages=[],
        tokens_before=0,
        tokens_after=0,
        transforms_applied=[],
    )
    assert hasattr(tr, "message_decisions")
    assert tr.message_decisions == []


def test_transform_result_message_decisions_populated():
    """TransformResult stores MessageDecision objects when provided."""
    dec = MessageDecision(
        message_index=0, role="user", tokens_before=10, tokens_after=10, action="passthrough:small"
    )
    tr = TransformResult(
        messages=[],
        tokens_before=0,
        tokens_after=0,
        transforms_applied=[],
        message_decisions=[dec],
    )
    assert len(tr.message_decisions) == 1
    assert tr.message_decisions[0] is dec


# ---------------------------------------------------------------------------
# CompressConfig.diagnostics field
# ---------------------------------------------------------------------------


def test_compress_config_diagnostics_defaults_false():
    """CompressConfig.diagnostics is False by default."""
    cfg = CompressConfig()
    assert cfg.diagnostics is False


def test_compress_config_diagnostics_can_be_set():
    """CompressConfig(diagnostics=True) stores True."""
    cfg = CompressConfig(diagnostics=True)
    assert cfg.diagnostics is True


# ---------------------------------------------------------------------------
# compress() does NOT collect diagnostics by default
# ---------------------------------------------------------------------------


def test_compress_result_diagnostics_none_by_default(monkeypatch):
    """CompressResult.diagnostics is None when diagnostics is not requested."""
    compress_module = importlib.import_module("headroom.compress")
    mock_pipeline = _MockPipeline()
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr(compress_module, "get_otel_metrics", _fake_otel)

    messages = [{"role": "user", "content": "hello"}]
    result = compress(messages, model="gpt-4o")

    assert result.diagnostics is None
    assert mock_pipeline.last_kwargs.get("collect_diagnostics") is False


# ---------------------------------------------------------------------------
# compress() passes collect_diagnostics when CompressConfig.diagnostics=True
# ---------------------------------------------------------------------------


def test_compress_passes_collect_diagnostics_when_config_true(monkeypatch):
    """compress() passes collect_diagnostics=True to pipeline.apply() when config.diagnostics=True."""
    compress_module = importlib.import_module("headroom.compress")
    mock_pipeline = _MockPipeline()
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr(compress_module, "get_otel_metrics", _fake_otel)

    messages = [{"role": "user", "content": "hello"}]
    cfg = CompressConfig(diagnostics=True)
    compress(messages, model="gpt-4o", config=cfg)

    assert mock_pipeline.last_kwargs.get("collect_diagnostics") is True


def test_compress_result_diagnostics_list_when_config_true(monkeypatch):
    """CompressResult.diagnostics is a list (possibly empty) when config.diagnostics=True."""
    compress_module = importlib.import_module("headroom.compress")
    mock_pipeline = _MockPipeline(decisions=[])
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr(compress_module, "get_otel_metrics", _fake_otel)

    messages = [{"role": "user", "content": "hello"}]
    cfg = CompressConfig(diagnostics=True)
    result = compress(messages, model="gpt-4o", config=cfg)

    assert result.diagnostics is not None
    assert isinstance(result.diagnostics, list)


def test_compress_result_diagnostics_contains_decisions(monkeypatch):
    """CompressResult.diagnostics contains the MessageDecision objects from the pipeline."""
    compress_module = importlib.import_module("headroom.compress")
    expected = [
        MessageDecision(
            message_index=0,
            role="user",
            tokens_before=10,
            tokens_after=10,
            action="protected:user_message",
        ),
        MessageDecision(
            message_index=1,
            role="tool",
            tokens_before=200,
            tokens_after=80,
            action="compressed:kompress:0.40",
        ),
    ]
    mock_pipeline = _MockPipeline(decisions=expected)
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr(compress_module, "get_otel_metrics", _fake_otel)

    messages = [
        {"role": "user", "content": "x" * 40},
        {"role": "tool", "tool_call_id": "t1", "content": "y" * 800},
    ]
    cfg = CompressConfig(diagnostics=True)
    result = compress(messages, model="gpt-4o", config=cfg)

    assert result.diagnostics == expected


# ---------------------------------------------------------------------------
# HEADROOM_DIAGNOSTICS=1 env var
# ---------------------------------------------------------------------------


def test_env_var_enables_diagnostics(monkeypatch):
    """HEADROOM_DIAGNOSTICS=1 enables collect_diagnostics without config flag."""
    compress_module = importlib.import_module("headroom.compress")
    mock_pipeline = _MockPipeline()
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr(compress_module, "get_otel_metrics", _fake_otel)

    messages = [{"role": "user", "content": "hello"}]
    with patch.dict(os.environ, {"HEADROOM_DIAGNOSTICS": "1"}):
        result = compress(messages, model="gpt-4o")

    assert mock_pipeline.last_kwargs.get("collect_diagnostics") is True
    assert result.diagnostics is not None


def test_env_var_absent_keeps_diagnostics_none(monkeypatch):
    """When HEADROOM_DIAGNOSTICS is unset, result.diagnostics stays None."""
    compress_module = importlib.import_module("headroom.compress")
    mock_pipeline = _MockPipeline()
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr(compress_module, "get_otel_metrics", _fake_otel)

    messages = [{"role": "user", "content": "hello"}]
    env = {k: v for k, v in os.environ.items() if k != "HEADROOM_DIAGNOSTICS"}
    with patch.dict(os.environ, env, clear=True):
        result = compress(messages, model="gpt-4o")

    assert result.diagnostics is None


def test_env_var_overrides_config_false(monkeypatch):
    """HEADROOM_DIAGNOSTICS=1 enables diagnostics even when CompressConfig.diagnostics=False."""
    compress_module = importlib.import_module("headroom.compress")
    mock_pipeline = _MockPipeline()
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr(compress_module, "get_otel_metrics", _fake_otel)

    messages = [{"role": "user", "content": "hello"}]
    cfg = CompressConfig(diagnostics=False)
    with patch.dict(os.environ, {"HEADROOM_DIAGNOSTICS": "1"}):
        result = compress(messages, model="gpt-4o", config=cfg)

    assert result.diagnostics is not None
    assert mock_pipeline.last_kwargs.get("collect_diagnostics") is True


# ---------------------------------------------------------------------------
# diagnostics=None when optimization is disabled
# ---------------------------------------------------------------------------


def test_no_diagnostics_when_optimize_false():
    """compress(optimize=False) returns empty CompressResult without calling pipeline."""
    cfg = CompressConfig(diagnostics=True)
    messages = [{"role": "user", "content": "hello"}]
    result = compress(messages, model="gpt-4o", optimize=False, config=cfg)
    # optimize=False early-returns before calling the pipeline
    assert result.diagnostics is None
