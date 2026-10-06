"""Extension middleware runs inside the proxy's security gate and body ceiling.

Regression tests for the ordering contract documented in
``headroom/proxy/extensions.py``. Before this contract existed, ``install_all``
ran after the gate was registered, and Starlette's prepend-on-register stack
made every extension middleware the OUTERMOST layer: an extension that answered
``POST /`` itself served unauthenticated callers, one that buffered the body
held an unbounded body before the 401 was ever issued, and one that rewrote
``Authorization`` did so before the gate compared it.

The extension used here is a stand-in for the shapes the private extensions
actually take (a middleware that claims a request and answers it, and one that
buffers the whole body), installed through the real ``install_all`` seam by
monkeypatching discovery, so the tests exercise the app exactly as
``headroom proxy --proxy-extension <name>`` would.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from headroom.cache.compression_store import reset_compression_store
from headroom.proxy import extensions
from headroom.proxy import helpers as proxy_helpers
from headroom.proxy.request_body_limit import RequestBodyLimitMiddleware
from headroom.proxy.server import ProxyConfig, WebSocketAuthMiddleware, create_app

NONLOOPBACK = ("203.0.113.5", 44444)  # TEST-NET-3, never loopback
LOOPBACK = ("127.0.0.1", 12345)
TOKEN = "s3cr3t-token"


class _RecordingExtension:
    """Raw ASGI middleware that records every scope it sees and answers ``POST /``.

    Mirrors the shape of an extension that intercepts a vendor protocol on the
    root path and relays it upstream itself, never calling the wrapped app.
    """

    seen: list[dict[str, Any]] = []

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] in {"http", "websocket"}:
            type(self).seen.append({"type": scope["type"], "path": scope.get("path")})
        if scope["type"] == "http" and scope.get("path") == "/" and scope.get("method") == "POST":
            body = b'{"claimed_by":"extension"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


class _BufferingExtension:
    """Raw ASGI middleware that buffers the full request body before passing it on.

    Mirrors an extension that must read the JSON to decide how to route or
    rewrite the request. ``max_seen`` records the largest body it ever held.
    """

    max_seen = 0

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunks.extend(message.get("body") or b"")
            type(self).max_seen = max(type(self).max_seen, len(chunks))
            if not message.get("more_body"):
                break
        replay = {"type": "http.request", "body": bytes(chunks), "more_body": False}
        sent = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return replay
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


def _install_recording(app: Any, config: Any) -> None:
    app.add_middleware(_RecordingExtension)


def _install_buffering(app: Any, config: Any) -> None:
    app.add_middleware(_BufferingExtension)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    _RecordingExtension.seen = []
    _BufferingExtension.max_seen = 0


def _make_app(monkeypatch, install, **overrides) -> Any:
    reset_compression_store()
    monkeypatch.setattr(extensions, "discover", lambda: iter([("test_ext", install)]))
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        proxy_extensions=["test_ext"],
        **overrides,
    )
    return create_app(config)


class TestExtensionMiddlewareIsInsideTheSecurityGate:
    def test_unauthenticated_request_never_reaches_extension_middleware(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post("/", json={}, headers={"x-amz-target": "Vendor.Generate"})
        assert resp.status_code == 401
        assert resp.json() == {"error": "unauthorized"}
        assert _RecordingExtension.seen == [], "extension ran before the proxy token was checked"

    def test_authenticated_request_reaches_extension_middleware(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post("/", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200
        assert resp.json() == {"claimed_by": "extension"}
        assert [s["path"] for s in _RecordingExtension.seen] == ["/"]

    def test_wrong_token_never_reaches_extension_middleware(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post("/", json={}, headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401
        assert _RecordingExtension.seen == []

    def test_loopback_caller_is_exempt_and_reaches_extension(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as c:
            resp = c.post("/", json={})
        assert resp.status_code == 200
        assert [s["path"] for s in _RecordingExtension.seen] == ["/"]

    def test_health_probe_stays_exempt_with_extension_installed(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.get("/health")
        assert resp.status_code == 200
        # Exempt from auth, so the extension legitimately sees it — the gate
        # passes it through rather than the extension pre-empting the gate.
        assert [s["path"] for s in _RecordingExtension.seen] == ["/health"]

    def test_gate_security_headers_apply_to_extension_responses(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post("/", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_unauthenticated_websocket_never_reaches_extension_middleware(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            with pytest.raises(Exception):  # noqa: B017 — TestClient surfaces the refused upgrade
                with c.websocket_connect("/v1/responses"):
                    pass
        assert _RecordingExtension.seen == [], "extension saw a refused WebSocket handshake"


class TestBodyCeilingAppliesBeforeExtensionMiddleware:
    def test_declared_oversize_body_is_refused_before_extension_buffers_it(self, monkeypatch):
        monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 1024)
        app = _make_app(monkeypatch, _install_buffering, proxy_token=TOKEN)
        payload = b"x" * 4096
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post(
                "/v1/chat/completions",
                content=payload,
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "content-type": "application/json",
                },
            )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "request_too_large"
        assert _BufferingExtension.max_seen == 0

    def test_chunked_oversize_body_is_cut_off_before_extension_holds_it_all(self, monkeypatch):
        monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 1024)
        app = _make_app(monkeypatch, _install_buffering, proxy_token=TOKEN)
        chunk = b"x" * 256

        def body():
            for _ in range(64):  # 16 KiB, no Content-Length
                yield chunk

        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post(
                "/v1/chat/completions",
                content=body(),
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "content-type": "application/json",
                },
            )
        assert resp.status_code == 413
        # The ceiling plus at most one delivered message; never the whole 16 KiB.
        # (TestClient delivers the generator as one message, so the extension
        # sees nothing at all here; the multi-message case is covered by
        # TestRequestBodyLimitMiddlewareDirectly below.)
        assert _BufferingExtension.max_seen <= 1024 + len(chunk)

    def test_anthropic_dialect_payload_on_messages_path(self, monkeypatch):
        monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 1024)
        app = _make_app(monkeypatch, _install_buffering, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post(
                "/v1/messages",
                content=b"x" * 4096,
                headers={"Authorization": f"Bearer {TOKEN}", "content-type": "application/json"},
            )
        assert resp.status_code == 413
        assert resp.json() == {
            "type": "error",
            "error": {
                "type": "request_too_large",
                "message": "Request body too large. Maximum size is 0MB",
            },
        }

    @pytest.mark.parametrize(
        "route",
        [
            "/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke",
            "/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke-with-response-stream",
        ],
        ids=["invoke", "invoke-with-response-stream"],
    )
    def test_bedrock_invoke_dialect_payload_on_model_paths(self, monkeypatch, route):
        """The two Bedrock InvokeModel routes keep handle_bedrock_invoke's wire
        dialect on the body ceiling: exactly
        ``{"error": {"type": "request_too_large", "message": ...}}`` — not the
        OpenAI-style payload — and the extension never buffers the body.
        """
        monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 1024)
        app = _make_app(
            monkeypatch,
            _install_buffering,
            proxy_token=TOKEN,
            bedrock_api_url="http://127.0.0.1:4000",
        )
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post(
                route,
                content=b"x" * 4096,
                headers={"Authorization": f"Bearer {TOKEN}", "content-type": "application/json"},
            )
        assert resp.status_code == 413
        assert resp.json() == {
            "error": {
                "type": "request_too_large",
                "message": "Request body too large. Maximum size is 0MB",
            }
        }
        assert _BufferingExtension.max_seen == 0

    def test_body_within_ceiling_reaches_extension_intact(self, monkeypatch):
        monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 1024)
        app = _make_app(monkeypatch, _install_buffering, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            # /stats is a local route, so the request completes without an upstream.
            resp = c.post(
                "/stats",
                content=b"y" * 512,
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        assert resp.status_code != 413
        assert _BufferingExtension.max_seen == 512

    def test_unauthenticated_oversize_body_is_refused_by_the_gate_first(self, monkeypatch):
        monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 1024)
        app = _make_app(monkeypatch, _install_buffering, proxy_token=TOKEN)
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.post("/v1/chat/completions", content=b"x" * 4096)
        assert resp.status_code == 401
        assert _BufferingExtension.max_seen == 0


class TestStackOrderIsPinned:
    """The contract is structural, so check the stack itself, not only its effects."""

    @staticmethod
    def _index_of(app, predicate) -> int:
        for index, entry in enumerate(app.user_middleware):
            if predicate(entry):
                return index
        raise AssertionError("middleware not found in stack")

    def test_extension_middleware_is_inner_to_gate_and_body_limit(self, monkeypatch):
        app = _make_app(monkeypatch, _install_recording, proxy_token=TOKEN)
        added = app.state.extension_middleware
        assert len(added) == 1
        assert added[0].cls is _RecordingExtension

        # Lower index == outer layer (Starlette prepends on add_middleware).
        ext_index = self._index_of(app, lambda e: e.cls is _RecordingExtension)
        limit_index = self._index_of(app, lambda e: e.cls is RequestBodyLimitMiddleware)
        ws_auth_index = self._index_of(app, lambda e: e.cls is WebSocketAuthMiddleware)
        gate_index = self._index_of(
            app,
            lambda e: getattr(e.kwargs.get("dispatch"), "__name__", "") == "_security_gate",
        )
        assert ws_auth_index < gate_index < limit_index < ext_index

    def test_install_all_records_nothing_for_routes_only_extensions(self, monkeypatch):
        def routes_only(app: Any, config: Any) -> None:
            @app.get("/ext/ping")
            async def ping():
                return {"ok": True}

        app = _make_app(monkeypatch, routes_only, proxy_token=TOKEN)
        assert app.state.extension_middleware == []
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            assert c.get("/ext/ping").status_code == 401
            assert c.get("/ext/ping", headers={"Authorization": f"Bearer {TOKEN}"}).json() == {
                "ok": True
            }


class TestRequestBodyLimitMiddlewareDirectly:
    """Drive the layer with a hand-rolled ASGI conversation to pin the metering."""

    @staticmethod
    def _http_scope(path="/v1/chat/completions", headers=()):
        return {"type": "http", "method": "POST", "path": path, "headers": list(headers)}

    @staticmethod
    def _chunked_receive(chunks):
        messages = [{"type": "http.request", "body": c, "more_body": True} for c in chunks]
        messages[-1]["more_body"] = False
        messages.append({"type": "http.disconnect"})
        it = iter(messages)

        async def receive():
            return next(it)

        return receive

    async def test_stream_over_ceiling_answers_413_once_and_stops_the_reader(self):
        held = bytearray()
        downstream_sent: list[dict] = []
        downstream_error: list[BaseException] = []

        async def downstream(scope, receive, send):
            try:
                while True:
                    m = await receive()
                    if m["type"] != "http.request":
                        break
                    held.extend(m["body"])
                    if not m.get("more_body"):
                        break
            except proxy_helpers.RequestBodyTooLarge as exc:
                downstream_error.append(exc)
                # A real handler that swallowed this would now try to answer;
                # that answer must be discarded because the 413 already went out.
                await send({"type": "http.response.start", "status": 400, "headers": []})
                await send({"type": "http.response.body", "body": b"{}"})
                downstream_sent.append({"attempted": True})
                return
            raise AssertionError("reader should have been stopped")

        sent: list[dict] = []

        async def send(m):
            sent.append(m)

        mw = RequestBodyLimitMiddleware(downstream, max_bytes=1000)
        await mw(self._http_scope(), self._chunked_receive([b"a" * 400] * 5), send)

        assert downstream_error, "the reader was not told to stop"
        # The message that trips the ceiling is never delivered, so the reader
        # holds at most the ceiling itself.
        assert 0 < len(held) <= 1000
        statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
        assert statuses == [413]  # exactly one refusal; the downstream 400 was dropped
        assert downstream_sent == [{"attempted": True}]

    @pytest.mark.parametrize(
        "path",
        [
            "/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke",
            "/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke-with-response-stream",
        ],
        ids=["invoke", "invoke-with-response-stream"],
    )
    async def test_stream_trip_answers_once_in_bedrock_dialect(self, path):
        """On a metered-stream trip over a Bedrock InvokeModel route the single
        refusal carries handle_bedrock_invoke's dialect, and the handler's own
        413 answer (its RequestBodyTooLarge response) is discarded by the
        response-once state: exactly one response reaches the wire.
        """
        downstream_sent: list[dict] = []

        async def downstream(scope, receive, send):
            try:
                while True:
                    m = await receive()
                    if m["type"] != "http.request":
                        break
                    if not m.get("more_body"):
                        break
            except proxy_helpers.RequestBodyTooLarge:
                # handle_bedrock_invoke answers RequestBodyTooLarge with its own
                # 413; response-once must drop it because the ceiling already answered.
                body = b'{"error": {"type": "request_too_large", "message": "from handler"}}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                downstream_sent.append({"attempted": True})
                return
            raise AssertionError("reader should have been stopped")

        sent: list[dict] = []

        async def send(m):
            sent.append(m)

        mw = RequestBodyLimitMiddleware(downstream, max_bytes=1000)
        await mw(self._http_scope(path=path), self._chunked_receive([b"a" * 400] * 5), send)

        statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
        assert statuses == [413]  # exactly one response; the handler's 413 was dropped
        body = b"".join(m.get("body") or b"" for m in sent if m["type"] == "http.response.body")
        assert json.loads(body) == {
            "error": {
                "type": "request_too_large",
                "message": "Request body too large. Maximum size is 0MB",
            }
        }
        assert downstream_sent == [{"attempted": True}]

    async def test_body_under_ceiling_is_delivered_untouched(self):
        held = bytearray()

        async def downstream(scope, receive, send):
            while True:
                m = await receive()
                held.extend(m["body"])
                if not m.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        sent: list[dict] = []

        async def send(m):
            sent.append(m)

        mw = RequestBodyLimitMiddleware(downstream, max_bytes=1000)
        await mw(self._http_scope(), self._chunked_receive([b"a" * 300] * 3), send)
        assert bytes(held) == b"a" * 900
        assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]

    async def test_response_already_started_is_not_clobbered(self):
        """A streaming handler that answered before finishing the body keeps its response."""

        async def downstream(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
            try:
                while True:
                    m = await receive()
                    if not m.get("more_body"):
                        break
            except proxy_helpers.RequestBodyTooLarge:
                await send({"type": "http.response.body", "body": b"", "more_body": False})

        sent: list[dict] = []

        async def send(m):
            sent.append(m)

        mw = RequestBodyLimitMiddleware(downstream, max_bytes=1000)
        await mw(self._http_scope(), self._chunked_receive([b"a" * 600] * 3), send)
        assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]
        assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}

    async def test_unparseable_content_length_falls_back_to_metering(self):
        async def downstream(scope, receive, send):
            while True:
                m = await receive()
                if not m.get("more_body"):
                    break

        sent: list[dict] = []

        async def send(m):
            sent.append(m)

        mw = RequestBodyLimitMiddleware(downstream, max_bytes=100)
        scope = self._http_scope(headers=[(b"content-length", b"not-a-number")])
        await mw(scope, self._chunked_receive([b"a" * 80, b"a" * 80]), send)
        assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [413]

    async def test_non_http_scope_passes_through(self):
        seen = []

        async def downstream(scope, receive, send):
            seen.append(scope["type"])

        mw = RequestBodyLimitMiddleware(downstream, max_bytes=1)
        await mw({"type": "websocket", "path": "/v1/live"}, None, None)
        await mw({"type": "lifespan"}, None, None)
        assert seen == ["websocket", "lifespan"]
