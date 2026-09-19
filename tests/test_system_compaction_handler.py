"""The Anthropic handler's system-prompt compaction must tolerate a pipeline
with no ContentRouter instead of logging a spurious failure."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def test_system_compaction_without_content_router_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
    monkeypatch.setattr(
        "headroom.transforms.compression_units.find_content_router", lambda _pipeline: None
    )
    app = create_app(
        ProxyConfig(
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            anthropic_api_url="http://127.0.0.1:9",
        )
    )
    # The headroom logger does not propagate to root, so capture it directly.
    proxy_logger = logging.getLogger("headroom.proxy")
    proxy_logger.addHandler(caplog.handler)
    try:
        with TestClient(app) as client:
            client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-ant-test"},
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 8,
                    "system": "You are terse.",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
    finally:
        proxy_logger.removeHandler(caplog.handler)
    assert "Request failed" in caplog.text  # reached the (dead) upstream, so compaction ran
    assert "system prompt compaction FAILED" not in caplog.text
