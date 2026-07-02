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
