"""A Claude turn routed to GitHub Copilot must reach it authenticated, and be
attributed to Copilot — on the buffered (non-streaming) arm, not just streaming.

Copilot serves Claude models from its Anthropic surface (``/v1/messages``) on
the same host as its OpenAI surface, so the resolved Anthropic target can be a
Copilot host with no per-request ``x-headroom-base-url`` in play. Two things
used to be true only on the streaming path:

- **Auth.** ``apply_copilot_api_auth`` is keyed on the upstream URL and was
  applied only by ``_stream_response``. The buffered arm sends through
  ``_retry_request``, which forwards headers untouched, so the request carried
  no minted token and no ``Copilot-Integration-Id``.
- **Attribution.** ``build_copilot_upstream_url`` is the only place the
  routed-to-Copilot flag is set, and the buffered arm built its URL by
  f-string — so ``emit_request_outcome`` never relabeled the provider and the
  turn showed as "anthropic".

Both are pinned here at the ``_retry_request`` seam: the URL that was built, the
headers as they went on the wire, and the flag as it stood at send time.
"""

from __future__ import annotations

import contextvars

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom import copilot_auth  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

MESSAGES = "/v1/messages"
COPILOT = "https://api.githubcopilot.com"
ANTHROPIC = "https://api.anthropic.com"
BODY = {
    "model": "claude-sonnet-5",
    "max_tokens": 16,
    "stream": False,
    "messages": [{"role": "user", "content": "hi"}],
}
MINTED = "tid_minted_for_test"


def _make_config(**overrides) -> ProxyConfig:
    base = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _stub_token_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mint a deterministic Copilot API token instead of calling GitHub."""

    class _Token:
        token = MINTED

    class _Provider:
        async def get_api_token(self, integration_id: str | None = None):
            return _Token()

    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _Provider())


class _Send:
    """Capture what the buffered arm was about to put on the wire."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.headers: dict[str, str] = {}
        self.routed_to_copilot: bool | None = None

    async def __call__(self, method, url, headers, body, **kwargs):
        self.url = url
        self.headers = dict(headers)
        # Read the flag where it matters: at send time, before the outcome
        # funnel consumes it.
        self.routed_to_copilot = copilot_auth.request_routed_to_copilot()
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": BODY["model"],
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            request=httpx.Request(method, url),
        )


def _post(anthropic_api_url: str, monkeypatch: pytest.MonkeyPatch) -> _Send:
    _stub_token_provider(monkeypatch)
    send = _Send()
    app = create_app(_make_config(anthropic_api_url=anthropic_api_url))
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = send
        resp = client.post(MESSAGES, json=BODY)
    assert resp.status_code == 200
    return send


def _headers_lower(send: _Send) -> dict[str, str]:
    return {k.lower(): v for k, v in send.headers.items()}


def _emitted_providers(anthropic_api_url: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Provider labels on the outcomes this request emitted.

    The relabel happens inside ``emit_request_outcome``, which runs in a task
    created by ``asyncio.shield`` — so this also pins that the flag survives the
    context copy into that task, which asserting on the flag alone would not.
    """
    import headroom.telemetry.session as telemetry_session

    seen: list[str] = []
    monkeypatch.setattr(
        telemetry_session, "record_outcome", lambda outcome: seen.append(outcome.provider)
    )
    _post(anthropic_api_url, monkeypatch)
    return seen


# --- Copilot target ---------------------------------------------------------


def test_buffered_turn_to_copilot_is_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    send = contextvars.Context().run(lambda: _post(COPILOT, monkeypatch))

    headers = _headers_lower(send)
    assert headers["authorization"] == f"Bearer {MINTED}"
    # The credential and the integration id have to leave together, or GitHub
    # cannot HMAC-validate the pair.
    assert headers.get("copilot-integration-id")
    assert headers.get("editor-version")


def test_buffered_turn_to_copilot_keeps_the_v1_messages_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copilot's Anthropic surface keeps ``/v1``; stripping it 404s (#2409)."""
    send = contextvars.Context().run(lambda: _post(COPILOT, monkeypatch))

    assert send.url == f"{COPILOT}/v1/messages"


def test_buffered_turn_to_copilot_is_flagged_for_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = contextvars.Context().run(lambda: _post(COPILOT, monkeypatch))

    assert send.routed_to_copilot is True


def test_buffered_turn_to_copilot_is_labeled_copilot(monkeypatch: pytest.MonkeyPatch) -> None:
    """End of the chain: the outcome that reaches the dashboard says "copilot"."""
    providers = contextvars.Context().run(lambda: _emitted_providers(COPILOT, monkeypatch))

    assert providers == ["copilot"]


# --- non-Copilot target (control) -------------------------------------------


def test_anthropic_target_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off the Copilot path both changes must be inert."""
    send = contextvars.Context().run(lambda: _post(ANTHROPIC, monkeypatch))

    headers = _headers_lower(send)
    assert send.url == f"{ANTHROPIC}/v1/messages"
    assert send.routed_to_copilot is False
    # No Copilot credential or handshake headers invented for a non-Copilot host.
    assert headers.get("authorization") != f"Bearer {MINTED}"
    assert "copilot-integration-id" not in headers


def test_anthropic_target_is_not_relabeled(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = contextvars.Context().run(lambda: _emitted_providers(ANTHROPIC, monkeypatch))

    assert providers == ["anthropic"]
