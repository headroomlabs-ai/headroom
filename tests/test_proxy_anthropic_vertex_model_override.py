"""Regression coverage for #2363: litellm-vertex silently defaulting to
claude-3-5-sonnet-20241022 because the URL-derived Vertex model never made it
into ``body["model"]`` before reaching the backend.

Vertex's rawPredict/streamRawPredict wire format carries the model as a URL
path segment, never as a body field, so ``handle_anthropic_messages``'s
mutation guard (which only fired when an existing ``body["model"]`` string
needed sanitizing) silently dropped the resolved model whenever the body had
no ``model`` key at all -- which is every Vertex request.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.backends.base import BackendResponse
from headroom.proxy.server import ProxyConfig, create_app


def _vertex_body() -> dict[str, Any]:
    # Real Vertex rawPredict shape: no top-level "model" key.
    return {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 10,
        "stream": False,
        "messages": [{"role": "user", "content": "2+2"}],
    }


class _FakeAnthropicBackend:
    """Stand-in for LiteLLMBackend: only cares what body it's handed."""

    name = "fake-litellm-vertex"

    def __init__(self) -> None:
        self.captured_body: dict[str, Any] | None = None

    async def send_message(self, body: dict[str, Any], headers: dict[str, str]) -> BackendResponse:
        del headers
        self.captured_body = dict(body)
        return BackendResponse(
            body={
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": body.get("model", "unknown"),
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
            status_code=200,
        )


def _app() -> Any:
    return create_app(
        ProxyConfig(
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
    )


def test_vertex_rawpredict_propagates_url_model_to_backend_when_body_has_none() -> None:
    """The primary bug: with a backend configured (e.g. litellm-vertex), the
    model resolved from the Vertex URL path must reach body["model"] even
    though the request body never had a "model" key."""
    app = _app()
    fake_backend = _FakeAnthropicBackend()
    app.state.proxy.anthropic_backend = fake_backend  # type: ignore[assignment]

    client = TestClient(app)
    response = client.post(
        "/v1/projects/p/locations/europe-west1/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict",
        json=_vertex_body(),
    )

    assert response.status_code == 200, response.text
    assert fake_backend.captured_body is not None
    assert fake_backend.captured_body["model"] == "claude-sonnet-4-6"


def test_vertex_native_passthrough_body_unaffected_when_no_backend_configured() -> None:
    """Guardrail for the scoping decision: when no backend is configured
    (self.anthropic_backend is None, e.g. plain `--backend vertex`), the body
    forwarded to the real Vertex endpoint must stay byte-for-byte as before --
    no "model" key should be injected into a shape Vertex has never received
    from Headroom."""
    app = _app()
    assert app.state.proxy.anthropic_backend is None

    captured: dict[str, Any] = {}

    async def _fake_retry(
        method: str,  # noqa: ARG001
        url: str,  # noqa: ARG001
        headers: dict[str, str],  # noqa: ARG001
        body: dict[str, Any],
        body_mutated: bool,
        mutation_reasons: list[str],
        **kwargs: Any,
    ) -> httpx.Response:
        captured["body"] = dict(body)
        captured["body_mutated"] = body_mutated
        captured["mutation_reasons"] = list(mutation_reasons)
        return httpx.Response(
            200,
            json={
                "id": "msg_native",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    app.state.proxy._retry_request = _fake_retry  # type: ignore[assignment]

    client = TestClient(app)
    response = client.post(
        "/v1/projects/p/locations/europe-west1/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict",
        json=_vertex_body(),
    )

    assert response.status_code == 200, response.text
    assert "model" not in captured["body"]


def test_missing_model_and_no_override_does_not_write_unknown_into_body() -> None:
    """Third guard clause: even with a backend configured, a request that
    supplies neither a body model nor a model_override (e.g. a malformed
    direct /v1/messages call) must not have body["model"] set to the
    "unknown" fallback string -- that would be a new failure mode, not a fix."""
    app = _app()
    fake_backend = _FakeAnthropicBackend()
    app.state.proxy.anthropic_backend = fake_backend  # type: ignore[assignment]

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "max_tokens": 10,
            "stream": False,
            "messages": [{"role": "user", "content": "2+2"}],
        },
    )

    assert response.status_code == 200, response.text
    assert fake_backend.captured_body is not None
    assert "model" not in fake_backend.captured_body
