"""Tests for ``OpenAIHandlerMixin._resolve_openai_upstream``.

The dedicated OpenAI handlers (``/v1/chat/completions``,
``/v1/responses``) must honor the ``x-headroom-base-url`` request header
so OpenAI-compatible gateways (LiteLLM, CPA, self-hosted vLLM, Azure
OpenAI) route correctly — consistent with the generic passthrough route
that already honors it (see ``providers/proxy_routes.py``).

These tests pin the resolution contract:
- header present  → its value wins
- header absent   → configured ``OPENAI_API_URL`` fallback
- header empty or whitespace-only → fallback (no blanking)
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.datastructures import Headers  # noqa: E402

from headroom.proxy.handlers.openai import OpenAIHandlerMixin  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_reserved_test_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permit the reserved, intentionally unresolvable test origin."""
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.example")


class _FakeRequest:
    """Minimal stand-in exposing ``headers`` like a real Starlette request.

    Uses ``starlette.datastructures.Headers`` so header lookup is
    case-insensitive, matching the production ``request.headers`` — a
    plain ``dict`` would let case-folding regressions pass silently.
    """

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = Headers(headers=headers)


def _stub_proxy(fallback_url: str) -> OpenAIHandlerMixin:
    """A bare mixin instance with only ``OPENAI_API_URL`` configured."""
    return type(  # type: ignore[return-value]
        "_S",
        (OpenAIHandlerMixin,),
        {"OPENAI_API_URL": fallback_url},
    )()


def test_header_overrides_configured_url() -> None:
    proxy = _stub_proxy("https://api.openai.test")
    # The transport sends the upstream origin (no /v1 path).
    request = _FakeRequest({"x-headroom-base-url": "https://gateway.example"})

    assert proxy._resolve_openai_upstream(request) == "https://gateway.example"


def test_missing_header_falls_back_to_configured_url() -> None:
    proxy = _stub_proxy("https://api.openai.test")
    request = _FakeRequest({})

    assert proxy._resolve_openai_upstream(request) == "https://api.openai.test"


def test_empty_header_falls_back_to_configured_url() -> None:
    """An explicitly empty or whitespace-only header must not blank the upstream."""
    proxy = _stub_proxy("https://api.openai.test")

    empty = _FakeRequest({"x-headroom-base-url": ""})
    assert proxy._resolve_openai_upstream(empty) == "https://api.openai.test"

    whitespace = _FakeRequest({"x-headroom-base-url": "   "})
    assert proxy._resolve_openai_upstream(whitespace) == "https://api.openai.test"


def test_header_lookup_is_case_insensitive() -> None:
    """Transports may send mixed-case header names; lookup must still resolve."""
    proxy = _stub_proxy("https://api.openai.test")
    # Real transports routinely send Title-Case header names.
    request = _FakeRequest({"X-Headroom-Base-Url": "https://gateway.example"})

    assert proxy._resolve_openai_upstream(request) == "https://gateway.example"


def test_header_with_subpath_preserves_path() -> None:
    """A custom upstream served from a sub-path (e.g. /api/v1) must keep the path,
    not be collapsed to the bare origin (#2047)."""
    proxy = _stub_proxy("https://api.openai.test")
    request = _FakeRequest({"x-headroom-base-url": "https://gateway.example/api/v1"})

    assert proxy._resolve_openai_upstream(request) == "https://gateway.example/api/v1"

    # Trailing slash is normalized away, not doubled.
    trailing = _FakeRequest({"x-headroom-base-url": "https://gateway.example/api/v1/"})
    assert proxy._resolve_openai_upstream(trailing) == "https://gateway.example/api/v1"


def test_grok_session_login_request_routes_to_session_host_without_header() -> None:
    """The Grok CLI cannot send x-headroom-base-url; a `grok login` session token
    must still reach the host that accepts it instead of the api.x.ai target
    `headroom wrap grok` configures (which answers it 401)."""
    proxy = _stub_proxy("https://api.x.ai")
    session = _FakeRequest(
        {
            "Authorization": "Bearer eyJ0eXAiOiJhdCtqd3QifQ.x.y",
            "X-Xai-Token-Auth": "xai-grok-cli",
            "User-Agent": "grok-shell/0.2.112 (macos; aarch64)",
        }
    )
    assert proxy._resolve_openai_upstream(session) == "https://cli-chat-proxy.grok.com"

    # An xai- API key stays on the configured target.
    keyed = _FakeRequest(
        {"Authorization": "Bearer xai-abc", "User-Agent": "grok-shell/0.2.112 (macos; aarch64)"}
    )
    assert proxy._resolve_openai_upstream(keyed) == "https://api.x.ai"

    # An explicit header still wins over the session default.
    explicit = _FakeRequest(
        {
            "Authorization": "Bearer eyJ0eXAiOiJhdCtqd3QifQ.x.y",
            "X-Xai-Token-Auth": "xai-grok-cli",
            "x-headroom-base-url": "https://gateway.example",
        }
    )
    assert proxy._resolve_openai_upstream(explicit) == "https://gateway.example"


def test_grok_session_routing_never_bypasses_a_configured_internal_target() -> None:
    """A JWT-shaped bearer plus spoofed Grok headers must not pull a request off
    an operator's internal gateway and send its credential to grok.com."""
    proxy = _stub_proxy("https://gateway.internal")
    spoofed = _FakeRequest(
        {
            "Authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJodHRwczovL3Nzby5jb3JwIn0.sig",
            "X-Xai-Token-Auth": "xai-grok-cli",
            "User-Agent": "grok-shell/0.2.112 (macos; aarch64)",
        }
    )
    assert proxy._resolve_openai_upstream(spoofed) == "https://gateway.internal"
