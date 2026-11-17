from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# Whitespace the canonical serializer would normalise away.
RAW_BODY = (
    b'{ "requests": [ { "custom_id": "req-1", "params": { "model": "claude-sonnet-4-6",'
    b' "max_tokens": 64, "messages": [ { "role": "user", "content": "hello" } ] } } ] }'
)


def _post(*, headers: dict[str, str], license_ok: bool = True):
    config = ProxyConfig(
        optimize=True,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    captured: dict[str, object] = {"pipeline_calls": 0}
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        real_reporter = proxy.usage_reporter
        if not license_ok:
            proxy.usage_reporter = SimpleNamespace(should_compress=False)

        def _apply(**kwargs):  # noqa: ANN003
            captured["pipeline_calls"] += 1
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=100,
                tokens_after=100,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["kwargs"] = kwargs
            captured["body"] = body
            return httpx.Response(200, json={"id": "msgbatch_1", "type": "message_batch"})

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/messages/batches",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                **headers,
            },
            content=RAW_BODY,
        )
        proxy.usage_reporter = real_reporter  # shutdown calls the real reporter's stop()
    assert response.status_code == 200, response.text
    return captured


@pytest.mark.parametrize(
    "bypass_headers",
    [{"x-headroom-bypass": "true"}, {"x-headroom-mode": "passthrough"}],
)
def test_batch_create_bypass_skips_compression_and_forwards_original_bytes(bypass_headers):
    captured = _post(headers=bypass_headers)
    assert captured["pipeline_calls"] == 0
    assert captured["kwargs"]["body_mutated"] is False
    assert captured["kwargs"]["original_body_bytes"] == RAW_BODY
    assert captured["body"]["requests"][0]["params"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_batch_create_license_denied_skips_compression():
    captured = _post(headers={}, license_ok=False)
    assert captured["pipeline_calls"] == 0
    assert captured["body"]["requests"][0]["params"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_batch_create_without_bypass_still_compresses():
    captured = _post(headers={})
    assert captured["pipeline_calls"] == 1
    assert captured["kwargs"]["body_mutated"] is True
    assert json.loads(RAW_BODY)["requests"][0]["custom_id"] == "req-1"
