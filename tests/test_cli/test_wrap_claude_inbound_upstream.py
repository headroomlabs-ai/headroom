"""Tests for inbound ANTHROPIC_BASE_URL → upstream forwarding (issue #1353).

Verifies that `headroom wrap claude` preserves a pre-existing custom
ANTHROPIC_BASE_URL (e.g. a LiteLLM gateway) as the proxy's Anthropic upstream
instead of silently reverting routing to api.anthropic.com.
"""

from __future__ import annotations

import pytest

from headroom.cli import wrap as wrap_cli

_PROXY_PORT = 8787
_LITELLM_URL = "https://litellm.example.internal/anthropic"
_MALFORMED_LOCAL_PORT_URL = "http://localhost:notaport"


def test_detects_custom_litellm_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _LITELLM_URL)
    assert wrap_cli._detect_inbound_anthropic_upstream(_PROXY_PORT) == _LITELLM_URL


def test_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert wrap_cli._detect_inbound_anthropic_upstream(_PROXY_PORT) is None


def test_returns_none_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "   ")
    assert wrap_cli._detect_inbound_anthropic_upstream(_PROXY_PORT) is None


@pytest.mark.parametrize(
    "url",
    [
        f"http://127.0.0.1:{_PROXY_PORT}",
        f"http://localhost:{_PROXY_PORT}",
        f"http://127.0.0.1:{_PROXY_PORT}/",
        f"http://LOCALHOST:{_PROXY_PORT}/v1",
    ],
)
def test_returns_none_for_self_pointing_url(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    assert wrap_cli._detect_inbound_anthropic_upstream(_PROXY_PORT) is None


def test_returns_localhost_on_different_port(monkeypatch: pytest.MonkeyPatch) -> None:
    other_port_url = f"http://127.0.0.1:{_PROXY_PORT + 1}"
    monkeypatch.setenv("ANTHROPIC_BASE_URL", other_port_url)
    assert wrap_cli._detect_inbound_anthropic_upstream(_PROXY_PORT) == other_port_url


def test_returns_none_for_malformed_local_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _MALFORMED_LOCAL_PORT_URL)
    assert wrap_cli._detect_inbound_anthropic_upstream(_PROXY_PORT) is None
