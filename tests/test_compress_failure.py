from __future__ import annotations

import importlib
from types import SimpleNamespace

from headroom.compress import compress


class _FailingPipeline:
    def apply(self, **kwargs):  # noqa: ANN003, ANN201
        raise RuntimeError("boom")


def test_compress_returns_original_messages_when_pipeline_fails(monkeypatch) -> None:
    metrics: list[dict[str, str]] = []
    compress_module = importlib.import_module("headroom.compress")
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: _FailingPipeline())
    monkeypatch.setattr(
        compress_module,
        "get_otel_metrics",
        lambda: SimpleNamespace(
            record_compression_failure=lambda **kwargs: metrics.append(kwargs),
        ),
    )

    messages = [{"role": "user", "content": "hello world " * 100}]
    result = compress(messages, model="gpt-4o")

    assert result.messages == messages
    assert result.tokens_before == 0
    assert result.tokens_after == 0
    assert result.tokens_saved == 0
    assert metrics == [
        {
            "model": "gpt-4o",
            "operation": "compress",
            "error_type": "RuntimeError",
        }
    ]
