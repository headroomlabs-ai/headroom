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
- Grok CLI wire signals → ``api.x.ai``, but only while the OpenAI target is
  still the default (a configured gateway is never bypassed)

The chat/responses call sites are covered end to end in
``tests/test_grok_shared_proxy_call_sites.py``.
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


def test_grok_cli_routes_to_xai_when_process_default_is_openai() -> None:
    """Shared proxy started by Claude/Codex keeps OPENAI_API_URL=api.openai.com.

    Grok CLI cannot stamp x-headroom-base-url; wire signals must still route
    chat completions to api.x.ai so xAI session tokens are not sent to OpenAI.
    """
    proxy = _stub_proxy("https://api.openai.com")
    request = _FakeRequest(
        {
            "authorization": "Bearer redacted",
            "x-xai-token-auth": "xai-grok-cli",
            "user-agent": "grok-pager/0.2.117 grok-shell/0.2.117 (macos; aarch64)",
        }
    )

    assert proxy._resolve_openai_upstream(request) == "https://api.x.ai"


def test_grok_cli_ua_alone_routes_to_xai() -> None:
    proxy = _stub_proxy("https://api.openai.com")
    request = _FakeRequest({"user-agent": "grok-shell/0.2.112 (macos; aarch64)"})

    assert proxy._resolve_openai_upstream(request) == "https://api.x.ai"


def test_custom_base_url_still_wins_over_grok_signals() -> None:
    """Explicit x-headroom-base-url remains highest priority (gateway override)."""
    proxy = _stub_proxy("https://api.openai.com")
    request = _FakeRequest(
        {
            "x-headroom-base-url": "https://gateway.example",
            "x-xai-token-auth": "xai-grok-cli",
            "user-agent": "grok-shell/0.2.112",
        }
    )

    assert proxy._resolve_openai_upstream(request) == "https://gateway.example"


def test_non_grok_clients_keep_process_openai_default() -> None:
    proxy = _stub_proxy("https://api.openai.com")
    request = _FakeRequest(
        {
            "authorization": "Bearer sk-test",
            "user-agent": "codex-tui/0.146.0",
        }
    )

    assert proxy._resolve_openai_upstream(request) == "https://api.openai.com"


def test_grok_signals_do_not_override_a_configured_gateway() -> None:
    """Grok routing is a fallback for the default target, not an override."""
    proxy = _stub_proxy("https://gateway.internal")
    request = _FakeRequest(
        {
            "x-xai-token-auth": "xai-grok-cli",
            "user-agent": "grok-shell/0.2.117",
        }
    )

    assert proxy._resolve_openai_upstream(request) == "https://gateway.internal"
