from __future__ import annotations

import json
import time
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("headroom._core")

from fastapi.testclient import TestClient

from headroom.config import TransformResult
from headroom.proxy.server import ProxyConfig, create_app


def _proxy_config(**overrides: Any) -> ProxyConfig:
    defaults: dict[str, Any] = {
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
        "disable_kompress": True,
        "compression_max_workers": 1,
    }
    defaults.update(overrides)
    return ProxyConfig(**defaults)


def test_proxy_health_surfaces_compression_runtime_metrics(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(_proxy_config(optimize=False))

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        live = client.get("/livez")
        health = client.get("/health")

    assert live.status_code == 200
    assert live.json()["alive"] is True
    assert health.status_code == 200
    runtime = health.json()["runtime"]
    assert runtime["compression_executor"]["max_workers"] == 1
    assert runtime["compression_executor"]["queued"] == 0
    assert runtime["compression_executor"]["queue_timeouts_total"] == 0


def test_v1_compress_success_reports_actual_metrics(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(_proxy_config())
    proxy = app.state.proxy
    request_messages = [{"role": "user", "content": "summarize this repeated payload"}]
    compressed_messages = [{"role": "user", "content": "summary payload"}]
    ccr_hash = "abc123def4567890abc123de"

    def fake_apply(**kwargs):
        assert kwargs["messages"] == request_messages
        assert kwargs["model"] == "gpt-4o"
        return TransformResult(
            messages=compressed_messages,
            tokens_before=100,
            tokens_after=40,
            transforms_applied=["test:compress"],
            markers_inserted=[ccr_hash],
        )

    # The default /v1/compress mode runs a marker-free pipeline derived from
    # `openai_pipeline`, not `openai_pipeline` itself, so patch the one the
    # route actually uses. It is built eagerly at create_app() time.
    monkeypatch.setattr(proxy._compress_pipeline_cache["no_ccr"], "apply", fake_apply)

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.post(
            "/v1/compress",
            json={"model": "gpt-4o", "messages": request_messages},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["messages"] == compressed_messages
    assert body["tokens_before"] == 100
    assert body["tokens_after"] == 40
    assert body["tokens_saved"] == 60
    assert body["compression_ratio"] == 0.4
    assert body["transforms_applied"] == ["test:compress"]
    assert body["transforms_summary"] == {"test:compress": 1}
    assert body["ccr_hashes"] == [ccr_hash]


def test_v1_compress_timeout_fails_open_quickly(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(_proxy_config())
    proxy = app.state.proxy
    request_messages = [{"role": "user", "content": "do not mutate me"}]

    async def timeout_executor(fn, *, timeout):  # noqa: ANN001
        raise TimeoutError("compression deadline exceeded")

    monkeypatch.setattr(proxy, "_run_compression_in_executor", timeout_executor)

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        started = time.perf_counter()
        response = client.post(
            "/v1/compress",
            json={"model": "gpt-4o", "messages": request_messages},
        )
        elapsed = time.perf_counter() - started

    body = response.json()
    assert response.status_code == 200
    assert elapsed < 0.5
    assert body["messages"] == request_messages
    assert body["tokens_saved"] == 0
    assert body["compression_ratio"] == 1.0
    assert body["transforms_applied"] == []
    assert body["compression_skipped"] is True
    assert body["skip_reason"] == "compression_timeout"


def test_v1_compress_real_json_tool_payload_reduces_tokens(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(
        _proxy_config(
            ccr_inject_marker=False,
            min_tokens_to_crush=20,
        )
    )
    items = [
        {
            "id": i,
            "status": "ok",
            "score": i % 5,
            "message": "same repeated value " * 20,
        }
        for i in range(80)
    ]
    request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "summarize rows"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "list_rows", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": json.dumps(items)},
        ],
    }

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.post("/v1/compress", json=request)

    body = response.json()
    assert response.status_code == 200, response.text
    assert body["tokens_before"] > body["tokens_after"], body
    assert body["tokens_saved"] > 0
    assert body["compression_ratio"] < 1.0
    assert body["transforms_applied"], body


def test_compression_quarantine_releases_after_time_cap(monkeypatch) -> None:
    """A leaked/hung timed-out worker must not pin the quarantine open forever:
    once the time cap lapses, compression resumes and the release is counted
    once (#2360)."""
    import asyncio

    from headroom.proxy.server import CompressionQuarantinedError

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(_proxy_config())
    proxy = app.state.proxy

    # Simulate a timed-out worker that is still running (debt standing).
    proxy._compression_timed_out_in_flight = 1

    # Within the cap: new compression is quarantined.
    proxy._compression_quarantine_deadline = time.monotonic() + 1000.0
    with pytest.raises(CompressionQuarantinedError):
        asyncio.run(proxy._run_compression_in_executor(lambda: "unused", timeout=5.0))
    assert proxy._compression_quarantine_releases == 0

    # Past the cap: the worker is presumed leaked and compression runs again;
    # the release is recorded once.
    proxy._compression_quarantine_deadline = time.monotonic() - 1.0
    assert asyncio.run(proxy._run_compression_in_executor(lambda: "ran", timeout=5.0)) == "ran"
    assert proxy._compression_quarantine_releases == 1

    # A subsequent request is not re-counted and still runs.
    assert asyncio.run(proxy._run_compression_in_executor(lambda: "ok", timeout=5.0)) == "ok"
    assert proxy._compression_quarantine_releases == 1
