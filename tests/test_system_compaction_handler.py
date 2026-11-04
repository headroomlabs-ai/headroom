"""The Anthropic handler's system-prompt compaction must tolerate a pipeline
with no ContentRouter instead of logging a spurious failure, and must record
the transform when a router does rewrite the prompt."""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _post_with_system_compaction(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> str:
    """POST one /v1/messages request with HEADROOM_SYSTEM_COMPACT on; return the log."""
    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
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
    caplog.set_level(logging.DEBUG, logger="headroom.proxy")
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
    return caplog.text


def test_system_compaction_without_content_router_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "headroom.transforms.compression_units.find_content_router", lambda _pipeline: None
    )
    log = _post_with_system_compaction(monkeypatch, caplog)
    assert "system prompt compaction FAILED" not in log


def test_system_compaction_with_content_router_records_the_transform(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    router = object()
    monkeypatch.setattr(
        "headroom.transforms.compression_units.find_content_router", lambda _pipeline: router
    )

    def fake_compact(
        payload: dict[str, Any], **kwargs: Any
    ) -> tuple[dict[str, Any], bool, int, int]:
        assert kwargs["router"] is router
        return payload, True, 100, 40

    monkeypatch.setattr("headroom.proxy.system_compaction.compact_system_prompt", fake_compact)
    log = _post_with_system_compaction(monkeypatch, caplog)
    assert "system prompt compaction: 100 -> 40 bytes (60% saved)" in log
    assert "system prompt compaction FAILED" not in log
