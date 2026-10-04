"""Unit tests for Vertex onboarding diagnostics.

These are the hints end users see in situ when Vertex rejects a request, so they
need to be correct without a live GCP project -- everything here is pure.
"""

import json
import logging

import pytest

from headroom.providers.registry import BackendUnavailableError, create_proxy_backend
from headroom.providers.vertex import (
    HINT_HEADER,
    annotate_vertex_error,
    backend_error_hint,
    ensure_vertex_sdk_available,
    public_backend_error_message,
    vertex_error_hint,
    vertex_sdk_available,
)
from headroom.proxy.public_errors import INTERNAL_ERROR, public_message


class _FakeResponse:
    """Minimal stand-in for a Starlette Response."""

    def __init__(self, status_code: int, body: bytes | None = None):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        if body is not None:
            self.body = body


class TestVertexErrorHint:
    @pytest.mark.parametrize("status", [200, 201, 400, 500, 503])
    def test_no_hint_for_unexplainable_statuses(self, status):
        """Inventing a hint for an unknown status would mislead, not help."""
        assert vertex_error_hint(status, location="global", publisher="google") is None

    def test_401_points_at_adc_not_user_token(self):
        hint = vertex_error_hint(401, location="global", publisher="google", model="m")
        assert "application-default" in hint
        # The trap this exists for: the plain user token is silently rejected.
        assert "gcloud auth print-access-token" in hint

    def test_429_names_quota(self):
        assert "Quota" in vertex_error_hint(429, location="us-east5", publisher="anthropic")

    def test_404_on_partner_model_mentions_model_garden(self):
        hint = vertex_error_hint(404, location="us-east5", publisher="anthropic", model="claude-x")
        assert "Model Garden" in hint
        assert "Claude 4.7+" in hint

    def test_404_on_gemini_mentions_regional_gap_not_model_garden(self):
        hint = vertex_error_hint(404, location="us-central1", publisher="google", model="g")
        assert "no US regional endpoint" in hint
        assert "Model Garden" not in hint

    def test_403_and_404_share_remedies(self):
        kwargs = {"location": "global", "publisher": "google", "model": "m"}
        assert vertex_error_hint(403, **kwargs) == vertex_error_hint(404, **kwargs)

    @pytest.mark.parametrize(
        "hostile",
        ["c\r\nX-Injected: 1", "caf\u00e9", "secret-proj-123", "<script>alert(1)</script>"],
    )
    @pytest.mark.parametrize("status", [401, 403, 404, 429])
    def test_hint_never_echoes_client_path_params(self, hostile, status):
        """Path params are client-supplied; the hint is fixed text by construction."""
        hint = vertex_error_hint(status, location=hostile, publisher="google", model=hostile)
        assert hostile not in hint
        assert hint.isascii()
        assert "\r" not in hint and "\n" not in hint


class TestAnnotateVertexErrorHostileParams:
    def test_crlf_model_cannot_reach_header_or_body(self):
        body = json.dumps({"error": {"code": 404, "message": "not found"}}).encode()
        evil = "m\r\nX-Injected: 1"
        resp = annotate_vertex_error(
            _FakeResponse(404, body), location=evil, publisher="google", model=evil
        )
        header = resp.headers[HINT_HEADER]
        assert "X-Injected" not in header
        assert header.isascii() and "\r" not in header and "\n" not in header
        assert "X-Injected" not in json.loads(resp.body)["error"]["message"]

    def test_hint_header_is_latin1_encodable(self):
        """h11 rejects non-latin-1 / control bytes at send time, after we return."""
        for status in (401, 403, 404, 429):
            for publisher in ("google", "anthropic", ""):
                hint = vertex_error_hint(status, publisher=publisher)
                hint.encode("latin-1")
                assert all(ch >= " " for ch in hint)


class TestAnnotateVertexError:
    def test_success_untouched(self):
        body = json.dumps({"candidates": []}).encode()
        resp = annotate_vertex_error(_FakeResponse(200, body), location="global")
        assert resp.body == body
        assert HINT_HEADER not in resp.headers

    def test_hint_appended_to_error_message_and_header(self):
        body = json.dumps({"error": {"code": 404, "message": "not found"}}).encode()
        resp = annotate_vertex_error(
            _FakeResponse(404, body), location="us-central1", publisher="google", model="g"
        )
        payload = json.loads(resp.body)
        assert payload["error"]["message"].startswith("not found")
        assert "[headroom] hint:" in payload["error"]["message"]
        assert HINT_HEADER in resp.headers

    def test_content_length_stays_consistent(self):
        body = json.dumps({"error": {"code": 401, "message": "nope"}}).encode()
        resp = annotate_vertex_error(_FakeResponse(401, body), location="global")
        assert resp.headers["content-length"] == str(len(resp.body))

    def test_annotation_is_idempotent(self):
        body = json.dumps({"error": {"code": 401, "message": "nope"}}).encode()
        resp = annotate_vertex_error(_FakeResponse(401, body), location="global")
        once = resp.body
        again = annotate_vertex_error(resp, location="global").body
        assert once == again

    def test_non_json_body_survives(self):
        resp = annotate_vertex_error(_FakeResponse(404, b"<html>nope</html>"), location="global")
        assert resp.body == b"<html>nope</html>"
        assert HINT_HEADER in resp.headers

    def test_streaming_response_without_body_still_gets_header(self):
        """No body to rewrite; the header and log line carry the hint."""
        resp = annotate_vertex_error(_FakeResponse(403), location="global", publisher="anthropic")
        assert HINT_HEADER in resp.headers

    def test_error_payload_without_message_gets_namespaced_key(self):
        body = json.dumps({"error": {"code": 404}}).encode()
        resp = annotate_vertex_error(_FakeResponse(404, body), location="global")
        assert "headroom_hint" in json.loads(resp.body)


class TestBackendErrorHint:
    def test_missing_vertex_sdk_is_recognized(self):
        msg = (
            "litellm.BadRequestError: Vertex_aiException - vertexai import failed please run "
            "`pip install -U \"google-cloud-aiplatform>=1.38\"`. Got error: No module named 'vertexai'"
        )
        hint = backend_error_hint(msg)
        assert "google-cloud-aiplatform" in hint
        # The cheaper fix is usually dropping the flag entirely.
        assert "do NOT need `--backend vertex`" in hint

    def test_missing_adc_is_recognized(self):
        assert "application-default" in backend_error_hint(
            "Could not automatically determine credentials"
        )

    def test_unrelated_error_gets_no_hint(self):
        assert backend_error_hint("upstream timed out after 30s") is None


_SECRET_RAW = (
    "Your default credentials were not found. project=secret-proj-123 "
    "GOOGLE_APPLICATION_CREDENTIALS=/etc/keys/sa.json host=10.1.2.3"
)


class TestPublicBackendErrorMessage:
    """Hints must live inside the sanitized public_errors contract."""

    def test_internal_error_gets_fixed_sentence_plus_hint(self):
        out = public_backend_error_message(RuntimeError(_SECRET_RAW), _SECRET_RAW)
        assert out.startswith(public_message(INTERNAL_ERROR))
        assert "[headroom] hint:" in out
        assert "application-default" in out

    @pytest.mark.parametrize("leak", ["secret-proj-123", "/etc/keys/sa.json", "10.1.2.3"])
    def test_raw_exception_text_never_reaches_client(self, leak):
        out = public_backend_error_message(RuntimeError(_SECRET_RAW), _SECRET_RAW)
        assert leak not in out

    def test_missing_sdk_hint_without_exception_text(self):
        raw = "No module named 'vertexai' at /opt/venv/lib/site-packages"
        out = public_backend_error_message(ImportError(raw), raw)
        assert "google-cloud-aiplatform" in out
        assert "/opt/venv" not in out

    def test_unrecognized_error_is_plain_public_message(self):
        raw = "upstream timed out talking to 10.9.9.9"
        assert public_backend_error_message(RuntimeError(raw), raw) == public_message(
            INTERNAL_ERROR
        )

    def test_hints_are_fixed_constants(self):
        """Every possible hint is static text: nothing interpolated from input."""
        for raw in (_SECRET_RAW, "No module named 'vertexai' xyz-123"):
            hint = backend_error_hint(raw)
            assert hint is not None
            assert "secret-proj-123" not in hint and "xyz-123" not in hint


class TestLiteLLMVertexBackendErrors:
    """End to end through the real backend: raw text stays in the log."""

    @pytest.mark.asyncio
    async def test_vertex_backend_error_is_sanitized_with_hint(self, monkeypatch, caplog):
        litellm_mod = pytest.importorskip("headroom.backends.litellm")
        if not litellm_mod.LITELLM_AVAILABLE:
            pytest.skip("litellm not installed")

        async def boom(**_kw):
            raise RuntimeError(_SECRET_RAW)

        monkeypatch.setattr(litellm_mod, "acompletion", boom)
        backend = litellm_mod.LiteLLMBackend(provider="vertex_ai")
        with caplog.at_level(logging.ERROR):
            resp = await backend.send_message(
                {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
                {},
            )
        message = resp.body["error"]["message"]
        assert "[headroom] hint:" in message
        assert "secret-proj-123" not in message
        assert "/etc/keys/sa.json" not in message
        # Operators still get the full detail server-side.
        assert "secret-proj-123" in caplog.text

    @pytest.mark.asyncio
    async def test_non_vertex_backend_gets_no_vertex_hint(self, monkeypatch):
        litellm_mod = pytest.importorskip("headroom.backends.litellm")
        if not litellm_mod.LITELLM_AVAILABLE:
            pytest.skip("litellm not installed")

        async def boom(**_kw):
            raise RuntimeError(_SECRET_RAW)

        monkeypatch.setattr(litellm_mod, "acompletion", boom)
        backend = litellm_mod.LiteLLMBackend(provider="bedrock")
        resp = await backend.send_message(
            {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
            {},
        )
        assert "[headroom] hint:" not in resp.body["error"]["message"]


class TestVertexSdkPreflight:
    """`--backend vertex` without the SDK must fail at startup, not per-request."""

    def _no_sdk(self, monkeypatch):
        monkeypatch.setattr(
            "headroom.providers.vertex.diagnostics.importlib.util.find_spec",
            lambda name: None if name == "vertexai" else object(),
        )

    def test_available_reflects_import_spec(self, monkeypatch):
        self._no_sdk(monkeypatch)
        assert vertex_sdk_available() is False

    def test_ensure_raises_with_both_remedies(self, monkeypatch):
        self._no_sdk(monkeypatch)
        with pytest.raises(BackendUnavailableError) as excinfo:
            ensure_vertex_sdk_available()
        message = str(excinfo.value)
        assert "headroom-ai[proxy,vertex]" in message
        # The cheaper remedy: native passthrough routes never needed the flag.
        assert "do NOT need `--backend vertex`" in message

    def test_ensure_is_a_noop_when_present(self, monkeypatch):
        monkeypatch.setattr(
            "headroom.providers.vertex.diagnostics.importlib.util.find_spec",
            lambda name: object(),
        )
        ensure_vertex_sdk_available()

    @pytest.mark.parametrize(
        "backend", ["vertex", "vertex_ai", "litellm-vertex", "google-vertex", "googlevertex"]
    )
    def test_every_vertex_alias_is_preflighted(self, monkeypatch, backend):
        """registry aliases several spellings onto vertex_ai; all must be caught."""
        self._no_sdk(monkeypatch)
        with pytest.raises(BackendUnavailableError):
            create_proxy_backend(
                backend=backend,
                anyllm_provider="",
                bedrock_region=None,
                logger=logging.getLogger("test"),
            )

    def test_other_providers_are_unaffected(self, monkeypatch):
        """The preflight must not become a general-purpose backend gate."""
        self._no_sdk(monkeypatch)
        create_proxy_backend(
            backend="litellm-openrouter",
            anyllm_provider="",
            bedrock_region=None,
            logger=logging.getLogger("test"),
        )

    def test_injected_backend_class_bypasses_preflight(self, monkeypatch):
        """Tests supplying a fake backend should not need the real SDK."""
        self._no_sdk(monkeypatch)
        sentinel = object()
        result = create_proxy_backend(
            backend="vertex",
            anyllm_provider="",
            bedrock_region="us-east5",
            logger=logging.getLogger("test"),
            litellm_backend_cls=lambda **kwargs: sentinel,
        )
        assert result is sentinel
