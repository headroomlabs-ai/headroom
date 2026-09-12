"""Tests for CCR pipeline selection on OpenAI chat completions."""

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.handlers.openai import OpenAIHandlerMixin  # noqa: E402


def test_streaming_openai_chat_uses_marker_free_pipeline():
    """Streaming chat cannot redeem CCR markers through a tool continuation."""
    live_pipeline = object()
    marker_free_pipeline = object()

    class Handler(OpenAIHandlerMixin):
        def _no_ccr_pipeline(self):
            return marker_free_pipeline

    handler = Handler.__new__(Handler)
    handler.openai_pipeline = live_pipeline

    assert handler._chat_compression_pipeline(stream=True) is marker_free_pipeline
    assert handler._chat_compression_pipeline(stream=False) is live_pipeline
