"""Vertex onboarding diagnostics: hints a user sees when Vertex rejects a request.

Every hint must be fixed text (public_errors contract): request data and raw
exception text may *select* a hint but must never reach the client.
"""

import json
import logging
from unittest.mock import patch

import pytest

from headroom.providers.registry import BackendUnavailableError, create_proxy_backend
from headroom.providers.vertex import (
    HINT_HEADER,
    annotate_vertex_error,
    public_backend_error_message,
    vertex_error_hint,
)
from headroom.proxy.public_errors import INTERNAL_ERROR, public_message

_HOSTILE = "m\r\nX-Injected: 1<script>caf\u00e9-secret-proj-123"
_SECRET_RAW = (
    "Your default credentials were not found. project=secret-proj-123 "
    "GOOGLE_APPLICATION_CREDENTIALS=/etc/keys/sa.json host=10.1.2.3"
)
_LEAKS = ("secret-proj-123", "/etc/keys/sa.json", "10.1.2.3")


class _FakeResponse:
    """Minimal stand-in for a Starlette Response."""

    def __init__(self, status_code: int, body: bytes | None = None):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        if body is not None:
            self.body = body


def _error_body(status: int, message: str | None = "upstream said no") -> bytes:
    error: dict = {"code": status}
    if message is not None:
        error["message"] = message
    return json.dumps({"error": error}).encode()


class TestVertexErrorHint:
    @pytest.mark.parametrize("status", [200, 400, 500, 503])
    def test_no_hint_for_unexplainable_statuses(self, status):
        """Inventing a hint for an unknown status would mislead, not help."""
        assert vertex_error_hint(status, publisher="google") is None

    def test_401_points_at_adc_not_user_token(self):
        hint = vertex_error_hint(401)
        assert "gcloud auth application-default login" in hint
        assert "gcloud auth print-access-token` user token is NOT accepted" in hint

    def test_429_names_quota(self):
        assert vertex_error_hint(429, publisher="anthropic").startswith("Quota exhausted")

    @pytest.mark.parametrize("status", [403, 404])
    def test_partner_vs_gemini_remedies(self, status):
        claude = vertex_error_hint(status, publisher="anthropic")
        gemini = vertex_error_hint(status, publisher="google")
        assert "Model Garden" in claude and "Claude 4.7+" in claude
        assert "no US regional endpoint" in gemini and "Model Garden" not in gemini

    @pytest.mark.parametrize("status", [401, 403, 404, 429])
    @pytest.mark.parametrize("publisher", ["google", "anthropic", "", _HOSTILE])
    def test_every_hint_is_a_legal_header_value(self, status, publisher):
        """h11 rejects control / non-latin-1 bytes at send time, after we return."""
        hint = vertex_error_hint(status, publisher=publisher)
        hint.encode("latin-1")
        assert all(" " <= ch <= "~" for ch in hint)


class TestAnnotateVertexError:
    def test_success_untouched(self):
        body = json.dumps({"candidates": []}).encode()
        resp = annotate_vertex_error(_FakeResponse(200, body))
        assert resp.body == body and HINT_HEADER not in resp.headers

    def test_hint_in_header_and_error_message_with_consistent_length(self):
        resp = annotate_vertex_error(_FakeResponse(404, _error_body(404)), publisher="google")
        message = json.loads(resp.body)["error"]["message"]
        assert message == f"upstream said no\n[headroom] hint: {resp.headers[HINT_HEADER]}"
        assert resp.headers["content-length"] == str(len(resp.body))

    def test_annotation_is_idempotent(self):
        resp = annotate_vertex_error(_FakeResponse(401, _error_body(401)))
        once = resp.body
        assert annotate_vertex_error(resp).body == once

    def test_error_without_message_gets_namespaced_key(self):
        resp = annotate_vertex_error(_FakeResponse(404, _error_body(404, message=None)))
        assert json.loads(resp.body)["headroom_hint"] == resp.headers[HINT_HEADER]

    @pytest.mark.parametrize("body", [b"<html>nope</html>", b"[1, 2]", None])
    def test_unrewritable_bodies_survive_with_header(self, body):
        """Non-JSON, non-object JSON and streaming (no body) keep the header only."""
        resp = annotate_vertex_error(_FakeResponse(403, body), publisher="anthropic")
        assert getattr(resp, "body", None) == body
        assert HINT_HEADER in resp.headers

    def test_hostile_path_params_reach_log_escaped_but_never_client(self, caplog):
        with caplog.at_level(logging.WARNING):
            resp = annotate_vertex_error(
                _FakeResponse(404, _error_body(404)),
                location=_HOSTILE,
                publisher=_HOSTILE,
                model=_HOSTILE,
            )
        client_visible = resp.headers[HINT_HEADER] + resp.body.decode()
        assert "X-Injected" not in client_visible and "secret-proj" not in client_visible
        # Operators keep the context, with CR/LF escaped by %r.
        assert "secret-proj-123" in caplog.text and "\r\n" not in caplog.text


class TestNativeRoutesWiring:
    """The decorator must actually be applied on the real FastAPI routes."""

    @pytest.fixture
    def client(self, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app

        async def upstream_404(self, request, *args):
            return JSONResponse({"error": {"code": 404, "message": "not found"}}, 404)

        for name in ("handle_gemini_generate_content", "handle_passthrough"):
            monkeypatch.setattr(HeadroomProxy, name, upstream_404)
        app = create_app(ProxyConfig(optimize=False, cache_enabled=False, rate_limit_enabled=False))
        with TestClient(app) as client:
            yield client

    @pytest.mark.parametrize(
        "publisher,action,expect",
        [
            ("google", "generateContent", "no US regional endpoint"),
            ("anthropic", "generateContent", "Model Garden"),
        ],
    )
    def test_upstream_404_carries_hint(self, client, publisher, action, expect):
        resp = client.post(
            f"/v1/projects/p/locations/us-central1/publishers/{publisher}/models/m:{action}"
        )
        assert resp.status_code == 404
        assert expect in resp.headers[HINT_HEADER]
        assert "[headroom] hint:" in resp.json()["error"]["message"]

    def test_encoded_crlf_in_path_is_not_reflected(self, client):
        """%0d%0a decodes into the `model` path param; the response must stay sane."""
        resp = client.post(
            "/v1/projects/p/locations/us-central1/publishers/google/models/"
            "m%0d%0aX-Injected:%201:generateContent"
        )
        assert resp.status_code == 404
        assert "no US regional endpoint" in resp.headers[HINT_HEADER]
        assert "x-injected" not in resp.headers
        assert "X-Injected" not in resp.text


class TestPublicBackendErrorMessage:
    def test_internal_error_gets_fixed_sentence_plus_hint_only(self):
        out = public_backend_error_message(RuntimeError(_SECRET_RAW), _SECRET_RAW)
        prefix = f"{public_message(INTERNAL_ERROR)}\n[headroom] hint: "
        assert out.startswith(prefix)
        assert "gcloud auth application-default login" in out
        assert not any(leak in out for leak in _LEAKS)

    def test_missing_sdk_hint(self):
        raw = "vertexai import failed. Got error: No module named 'vertexai' at /opt/venv"
        out = public_backend_error_message(ImportError(raw), raw)
        assert "headroom-ai[proxy,vertex]" in out
        assert "do NOT need `--backend vertex`" in out
        assert "/opt/venv" not in out

    def test_unrecognized_error_is_plain_public_message(self):
        raw = "upstream timed out talking to 10.9.9.9"
        assert public_backend_error_message(RuntimeError(raw), raw) == public_message(
            INTERNAL_ERROR
        )


class TestLiteLLMVertexBackendErrors:
    """Through the real backend: raw text stays in the log, hint reaches client."""

    @pytest.fixture
    def litellm_mod(self, monkeypatch):
        mod = pytest.importorskip("headroom.backends.litellm")
        if not mod.LITELLM_AVAILABLE:
            pytest.skip("litellm not installed")

        async def boom(**_kw):
            raise RuntimeError(_SECRET_RAW)

        monkeypatch.setattr(mod, "acompletion", boom)
        return mod

    _BODY = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}
    _OPENAI_BODY = {"model": "gemini-3.5-flash", "messages": [{"role": "user", "content": "hi"}]}

    @pytest.mark.asyncio
    async def test_send_message(self, litellm_mod, caplog):
        backend = litellm_mod.LiteLLMBackend(provider="vertex_ai")
        with caplog.at_level(logging.ERROR):
            resp = await backend.send_message(dict(self._BODY), {})
        message = resp.body["error"]["message"]
        assert "[headroom] hint:" in message
        assert not any(leak in message for leak in _LEAKS)
        assert "secret-proj-123" in caplog.text

    @pytest.mark.asyncio
    async def test_send_openai_message(self, litellm_mod):
        backend = litellm_mod.LiteLLMBackend(provider="vertex_ai")
        resp = await backend.send_openai_message(dict(self._OPENAI_BODY), {})
        message = resp.body["error"]["message"]
        assert "[headroom] hint:" in message
        assert not any(leak in message for leak in _LEAKS)

    @pytest.mark.asyncio
    async def test_stream_message_error_event(self, litellm_mod):
        backend = litellm_mod.LiteLLMBackend(provider="vertex_ai")
        events = [e async for e in backend.stream_message(dict(self._BODY, stream=True), {})]
        errors = [e.data["error"]["message"] for e in events if e.event_type == "error"]
        assert len(errors) == 1
        assert "[headroom] hint:" in errors[0]
        assert not any(leak in errors[0] for leak in _LEAKS)

    @pytest.mark.asyncio
    async def test_stream_openai_message_error_chunk(self, litellm_mod):
        backend = litellm_mod.LiteLLMBackend(provider="vertex_ai")
        chunks = [c async for c in backend.stream_openai_message(dict(self._OPENAI_BODY), {})]
        assert chunks[-1] == "data: [DONE]\n\n"
        message = json.loads(chunks[-2].removeprefix("data: "))["error"]["message"]
        assert "[headroom] hint:" in message
        assert not any(leak in message for leak in _LEAKS)

    @pytest.mark.asyncio
    async def test_non_vertex_backend_gets_no_vertex_hint(self, litellm_mod):
        backend = litellm_mod.LiteLLMBackend(provider="openrouter")
        resp = await backend.send_message(dict(self._BODY), {})
        assert "[headroom] hint:" not in resp.body["error"]["message"]


class TestVertexSdkPreflight:
    """`--backend vertex` without the SDK must fail at startup, not per-request."""

    @pytest.fixture
    def no_sdk(self, monkeypatch):
        monkeypatch.setattr(
            "headroom.providers.vertex.diagnostics.importlib.util.find_spec",
            lambda name: None if name == "vertexai" else object(),
        )

    def _create(self, backend, **kwargs):
        return create_proxy_backend(
            backend=backend,
            anyllm_provider="",
            bedrock_region=None,
            logger=logging.getLogger("test"),
            **kwargs,
        )

    @pytest.mark.parametrize(
        "backend", ["vertex", "vertex_ai", "litellm-vertex", "google-vertex", "googlevertex"]
    )
    def test_every_vertex_alias_is_preflighted(self, no_sdk, backend):
        with pytest.raises(BackendUnavailableError) as excinfo:
            self._create(backend)
        assert "headroom-ai[proxy,vertex]" in str(excinfo.value)

    def test_other_providers_are_unaffected(self, no_sdk):
        """The preflight must not become a general-purpose backend gate."""
        self._create("litellm-openrouter")

    def test_injected_backend_class_bypasses_preflight(self, no_sdk):
        sentinel = object()
        assert self._create("vertex", litellm_backend_cls=lambda **_: sentinel) is sentinel

    def test_present_sdk_passes(self, monkeypatch):
        monkeypatch.setattr(
            "headroom.providers.vertex.diagnostics.importlib.util.find_spec",
            lambda name: object(),
        )
        assert self._create("vertex") is not None

    def test_cli_prints_remedy_and_exits_2(self):
        click_testing = pytest.importorskip("click.testing")
        from headroom.cli.main import main

        def unavailable(config, **kwargs):
            raise BackendUnavailableError("Vertex SDK is missing. remedy-text")

        with patch("headroom.proxy.server.run_server", unavailable):
            result = click_testing.CliRunner().invoke(main, ["proxy"])
        assert result.exit_code == 2
        assert "Cannot start proxy: Vertex SDK is missing. remedy-text" in result.output
        assert "Traceback" not in result.output
