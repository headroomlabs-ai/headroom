"""Tests for the AWS Bedrock InvokeModel passthrough handler.

Covers the routes registered by ``register_provider_routes`` when
``--bedrock-api-url`` is set, and the ``handle_bedrock_invoke`` behavior:

1. Routes register ONLY when ``bedrock_api_url`` is configured.
2. A large request body is compressed via ``anthropic_pipeline.apply`` and the
   compressed messages are what gets forwarded upstream.
3. Inference-profile model ids (dots/colons) are captured whole and re-encoded
   into the upstream URL.
4. The streaming route forwards upstream bytes byte-faithfully and closes the
   upstream connection.
5. Fail-open: a malformed JSON body is forwarded verbatim, never a 500.
6. Bypass (``optimize=False``) forwards verbatim — no compression.
7. The request outcome is recorded with ``provider="bedrock"``.
8. ``BEDROCK_TARGET_API_URL`` feeds the env config path.

All forwarding is mocked at ``proxy.http_client`` so no real upstream is needed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

UPSTREAM = "http://127.0.0.1:4000"
SONNET_BEDROCK = "anthropic.claude-3-5-sonnet-20241022-v2:0"
INVOKE = f"/model/{SONNET_BEDROCK}/invoke"
INVOKE_STREAM = f"/model/{SONNET_BEDROCK}/invoke-with-response-stream"


class _FakeUpstream:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(
        self,
        status_code: int = 200,
        headers: dict | None = None,
        chunks: tuple[bytes, ...] = (b'{"ok":true}',),
    ) -> None:
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {"content-type": "application/json"})
        self._chunks = list(chunks)
        self.closed = False

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class _FakeResult:
    """Stand-in for TransformPipeline.apply's TransformResult."""

    def __init__(
        self,
        messages: list[dict],
        tokens_before: int,
        tokens_after: int,
        transforms: tuple[str, ...] = ("smartcrush",),
    ) -> None:
        self.messages = messages
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.transforms_applied = list(transforms)
        self.timing = {"total": 1.0}


def _make_config(**overrides) -> ProxyConfig:
    base = {
        "bedrock_api_url": UPSTREAM,
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _install_fake_client(proxy, upstream: _FakeUpstream) -> MagicMock:
    """Replace proxy.http_client so forwarding never touches the network."""
    client = MagicMock()
    client.build_request = MagicMock(return_value=MagicMock(name="upstream_request"))
    client.send = AsyncMock(return_value=upstream)
    client.aclose = AsyncMock()  # awaited by proxy.shutdown() on lifespan exit
    proxy.http_client = client
    return client


def _forwarded(client: MagicMock) -> tuple[str, dict]:
    """Return (url, parsed_json_body_or_raw) handed to build_request."""
    call = client.build_request.call_args
    url = call.args[1] if len(call.args) > 1 else call.kwargs["url"]
    content = call.kwargs["content"]
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        parsed = content
    return url, parsed


# ── route gating ──────────────────────────────────────────────────────


def _paths(cfg: ProxyConfig) -> set[str]:
    app = create_app(cfg)
    return {r.path for r in app.routes if hasattr(r, "path")}


def test_routes_absent_when_bedrock_api_url_unset():
    paths = _paths(ProxyConfig())
    assert "/model/{model_id:path}/invoke" not in paths


def test_routes_present_when_bedrock_api_url_set():
    paths = _paths(_make_config())
    assert "/model/{model_id:path}/invoke" in paths
    assert "/model/{model_id:path}/invoke-with-response-stream" in paths


# ── compression ───────────────────────────────────────────────────────


def test_invoke_forwards_compressed_messages():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        compressed = [{"role": "user", "content": "short"}]
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult(compressed, tokens_before=5000, tokens_after=200)
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": "x" * 5000}],
            "max_tokens": 100,
        }
        resp = client.post(INVOKE, json=body)

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == compressed


def test_compressed_body_drops_stale_content_length():
    """Regression: a shrunk body must not carry the inbound content-length, or
    httpx raises 'Too little data for declared Content-Length' (caught in
    live testing). content-encoding is dropped on the same path so a stale
    gzip claim can't mislabel the re-serialized JSON."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "tiny"}], 5000, 100)
        )
        resp = client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "x" * 8000}], "max_tokens": 8},
        )

    assert resp.status_code == 200
    sent_headers = http.build_request.call_args.kwargs["headers"]
    lower = {k.lower() for k in sent_headers}
    assert "content-length" not in lower
    assert "content-encoding" not in lower


def test_invoke_preserves_non_message_body_fields():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult(
                [{"role": "user", "content": "c"}], tokens_before=900, tokens_after=100
            )
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": "y" * 3000}],
            "max_tokens": 256,
        }
        client.post(INVOKE, json=body)

    _, forwarded = _forwarded(http)
    assert forwarded["anthropic_version"] == "bedrock-2023-05-31"
    assert forwarded["max_tokens"] == 256


# ── model id encoding ─────────────────────────────────────────────────


def test_inference_profile_model_id_is_captured_and_reencoded():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 900, 100)
        )
        profile = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        client.post(
            f"/model/{profile}/invoke",
            json={"messages": [{"role": "user", "content": "z" * 3000}], "max_tokens": 8},
        )

    url, _ = _forwarded(http)
    # Colon is percent-encoded; the whole profile id survives in the path.
    assert url == f"{UPSTREAM}/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke"


# ── streaming ─────────────────────────────────────────────────────────


def test_invoke_with_response_stream_is_byte_faithful():
    app = create_app(_make_config())
    upstream = _FakeUpstream(chunks=(b"event-stream-chunk-1", b"event-stream-chunk-2"))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_fake_client(proxy, upstream)
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 900, 100)
        )
        resp = client.post(
            INVOKE_STREAM,
            json={"messages": [{"role": "user", "content": "w" * 3000}], "max_tokens": 8},
        )

    assert resp.status_code == 200
    assert resp.content == b"event-stream-chunk-1event-stream-chunk-2"
    assert upstream.closed is True


# ── fail-open + bypass ────────────────────────────────────────────────


def test_malformed_body_is_forwarded_verbatim():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        # Pipeline must NOT be invoked on an unparseable body.
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=AssertionError("should not run"))
        resp = client.post(
            INVOKE,
            content=b"not-json-at-all",
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded == b"not-json-at-all"


def test_optimize_disabled_forwards_verbatim():
    app = create_app(_make_config(optimize=False))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=AssertionError("should not run"))
        body = {"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8}
        resp = client.post(INVOKE, json=body)

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == body["messages"]


def test_compression_failure_forwards_verbatim():
    """Fail-open: if the pipeline raises, forward the ORIGINAL body untouched
    rather than 500ing the request."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=RuntimeError("boom"))
        body = {"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8}
        resp = client.post(INVOKE, json=body)

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == body["messages"]


def test_bypass_header_skips_compression():
    """`x-headroom-bypass: true` forwards verbatim — the pipeline never runs."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=AssertionError("should not run"))
        body = {"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8}
        resp = client.post(INVOKE, json=body, headers={"x-headroom-bypass": "true"})

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == body["messages"]


def test_upstream_connect_error_returns_502():
    """A transport failure to the gateway surfaces as a clean 502, not a crash."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        client_mock = _install_fake_client(proxy, _FakeUpstream())
        client_mock.send = AsyncMock(side_effect=httpx.ConnectError("no route"))
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 900, 100)
        )
        resp = client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "z" * 3000}], "max_tokens": 8},
        )

    assert resp.status_code == 502


# ── metrics ───────────────────────────────────────────────────────────


def test_outcome_recorded_with_bedrock_provider():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_fake_client(proxy, _FakeUpstream())
        proxy._record_request_outcome = AsyncMock()
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 1000, 250)
        )
        client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "q" * 3000}], "max_tokens": 8},
        )

    assert proxy._record_request_outcome.await_count == 1
    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.provider == "bedrock"
    assert outcome.tokens_saved == 750


# ── env config path ───────────────────────────────────────────────────


def test_env_var_feeds_config(monkeypatch):
    from headroom.proxy.server import _proxy_config_from_env

    monkeypatch.delenv("HEADROOM_PROXY_CONFIG_JSON", raising=False)
    monkeypatch.setenv("BEDROCK_TARGET_API_URL", UPSTREAM)
    cfg = _proxy_config_from_env()
    assert cfg.bedrock_api_url == UPSTREAM
