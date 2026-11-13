"""Credentials stay out of the proxy log and the upstream-error dumps."""

from __future__ import annotations

import json
import logging
import stat
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.proxy.server import create_app  # noqa: E402
from headroom.proxy.wire_debug_redaction_policy import should_redact_key  # noqa: E402

FAKE_KEY = "AIzaSyFAKEKEY0123456789"


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )


def test_inbound_request_log_drops_query_values_and_key_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO), TestClient(create_app(_config())) as client:
        client.get(
            f"/stats?key={FAKE_KEY}&alt=sse",
            headers={"Ocp-Apim-Subscription-Key": "FAKE-OCP", "apikey": "FAKE-APIKEY"},
        )
    lines = [r.getMessage() for r in caplog.records if "proxy_inbound_request" in r.getMessage()]
    assert lines
    for line in lines:
        assert FAKE_KEY not in line
        assert "FAKE-OCP" not in line
        assert "FAKE-APIKEY" not in line
    assert any("query=key=<redacted>&alt=<redacted>" in line for line in lines)


def test_streaming_upstream_error_log_drops_query(caplog: pytest.LogCaptureFixture) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad"}})

    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    with caplog.at_level(logging.WARNING), TestClient(create_app(_config())) as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        client.post(
            f"/v1beta/models/gemini-2.5-flash:streamGenerateContent?key={FAKE_KEY}&alt=sse",
            json=body,
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("Upstream provider returned an error" in m for m in messages), messages
    assert not any(FAKE_KEY in m for m in messages)


@pytest.mark.parametrize(
    "name",
    ["Ocp-Apim-Subscription-Key", "X-Functions-Key", "apikey", "x-goog-api-key", "x-api-key"],
)
def test_wire_debug_redacts_key_headers(name: str) -> None:
    assert should_redact_key(name)


@pytest.mark.parametrize("name", ["max_tokens", "input_tokens", "keyboard", "monkey", "key_id"])
def test_wire_debug_key_rule_does_not_overreach(name: str) -> None:
    assert not should_redact_key(name)


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        (
            "/v1/messages",
            {"x-api-key": "sk-ant-FAKESECRET-0123456789", "anthropic-version": "2023-06-01"},
        ),
        ("/v1/chat/completions", {"authorization": "Bearer sk-FAKESECRET-0123456789"}),
    ],
)
def test_upstream_error_dump_masks_credentials_and_is_owner_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch, path: str, headers: dict[str, str]
) -> None:
    from headroom import paths

    dump_dir = tmp_path / "debug_400"
    monkeypatch.setattr(paths, "debug_400_dir", lambda: dump_dir)
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "1")
    error = httpx.Response(
        400,
        content=json.dumps({"error": {"type": "invalid_request_error", "message": "bad"}}).encode(),
        headers={"content-type": "application/json"},
    )
    body = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    with patch("headroom.proxy.server.AnyLLMBackend"):
        with TestClient(create_app(_config())) as client:
            client.app.state.proxy._retry_request = AsyncMock(return_value=error)
            client.post(
                path, headers={**headers, "Ocp-Apim-Subscription-Key": "FAKE-OCP"}, json=body
            )

    (dump,) = dump_dir.glob("*.json")
    text = dump.read_text(encoding="utf-8")
    assert "FAKESECRET" not in text
    assert "sk-ant-FAKE" not in text and "Bearer sk-F" not in text
    assert "FAKE-OCP" not in text
    assert stat.S_IMODE(dump.stat().st_mode) == 0o600
