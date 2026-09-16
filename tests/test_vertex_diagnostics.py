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
    annotate_backend_error_body,
    annotate_vertex_error,
    backend_error_hint,
    ensure_vertex_sdk_available,
    vertex_error_hint,
    vertex_sdk_available,
)


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

    def test_hint_names_the_failing_triple(self):
        hint = vertex_error_hint(404, location="europe-west2", publisher="google", model="gem")
        assert "google/gem @ europe-west2" in hint


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


class TestAnnotateBackendErrorBody:
    def test_success_body_untouched(self):
        body = {"content": [{"text": "hi"}]}
        assert annotate_backend_error_body(body, 200) == body

    def test_hint_appended_to_nested_message(self):
        body = {
            "type": "error",
            "error": {"type": "api_error", "message": "No module named 'vertexai'"},
        }
        out = annotate_backend_error_body(body, 500)
        assert "[headroom] hint:" in out["error"]["message"]

    def test_idempotent(self):
        body = {"error": {"message": "No module named 'vertexai'"}}
        first = annotate_backend_error_body(body, 500)["error"]["message"]
        second = annotate_backend_error_body(body, 500)["error"]["message"]
        assert first == second

    def test_unrecognized_error_untouched(self):
        body = {"error": {"message": "upstream timed out"}}
        assert annotate_backend_error_body(body, 500) == body

    def test_non_dict_body_survives(self):
        assert annotate_backend_error_body("plain text", 500) == "plain text"


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
