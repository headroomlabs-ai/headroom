"""Byte-faithful passthrough for Codex Desktop /v1/responses posts (issue #1542).

Codex Desktop sends ``POST /v1/responses`` with ``content-encoding: zstd``. The
handler decodes the body to parse it, but when nothing mutates the request it
must forward the *original decoded bytes* verbatim and must not re-advertise the
stale ``content-encoding`` header. Otherwise the upstream ChatGPT Codex endpoint
either re-canonicalizes a body it rejects, or tries to zstd-decode already-decoded
JSON — both surface to the client as ``400 {"detail":"Bad Request"}``.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx
from fastapi.testclient import TestClient

from headroom.proxy.loopback_guard import require_loopback
from headroom.proxy.server import ProxyConfig, create_app


def _make_client(optimize: bool = False):
    config = ProxyConfig(
        optimize=optimize,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _fake_upstream_response(url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "object": "response",
            "output": [],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        },
        request=httpx.Request("POST", url),
    )


def _patch_capture(app):
    """Replace the server's upstream forwarder with a capturing stub."""
    captured: dict = {}
    server = app.state.proxy

    async def fake_retry(method, url, headers, body, stream=False, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = dict(headers)
        captured["body"] = body
        captured["kwargs"] = kwargs
        return _fake_upstream_response(url)

    server._retry_request = fake_retry
    return captured


def test_unmutated_zstd_post_forwards_decoded_bytes_and_strips_content_encoding():
    zstandard = pytest.importorskip("zstandard")
    app = _make_client(optimize=False)

    payload = {
        "model": "gpt-5-codex",
        "input": "list the files in this repo",
        "instructions": "be terse",
    }
    raw = json.dumps(payload).encode("utf-8")
    compressed = zstandard.ZstdCompressor().compress(raw)

    with TestClient(app) as client:
        captured = _patch_capture(app)
        resp = client.post(
            "/v1/responses",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
                "Content-Encoding": "zstd",
                "originator": "codex_desktop",
            },
            content=compressed,
        )

    assert resp.status_code == 200
    # Nothing mutated the request -> byte-faithful passthrough engages.
    assert captured["kwargs"].get("body_mutated") is False
    assert captured["kwargs"].get("original_body_bytes") == raw
    # The stale content-encoding must not ride along with already-decoded bytes.
    fwd_headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert "content-encoding" not in fwd_headers


def test_unmutated_plain_post_passes_original_bytes_through():
    app = _make_client(optimize=False)
    raw = json.dumps({"model": "gpt-5-codex", "input": "hi"}).encode("utf-8")

    with TestClient(app) as client:
        captured = _patch_capture(app)
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            content=raw,
        )

    assert resp.status_code == 200
    assert captured["kwargs"].get("body_mutated") is False
    assert captured["kwargs"].get("original_body_bytes") == raw
