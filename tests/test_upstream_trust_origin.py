"""Operator secret headers go to a designated origin, not just a designated host.

``x-headroom-base-url`` is chosen by the client. When trust compared only the
hostname, ``http://`` or another port of a designated host also received the
operator's extra headers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.proxy.helpers import merge_extra_headers
from headroom.proxy.upstream_trust import ALLOWED_HOSTS_ENV, is_trusted_upstream


@pytest.fixture(autouse=True)
def _no_env_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOWED_HOSTS_ENV, raising=False)


def test_secret_extra_headers_are_not_merged_for_another_scheme_or_port() -> None:
    extra = {"api-key": "corp-gateway-secret"}

    def sent(url: str) -> dict[str, str]:
        return merge_extra_headers({}, extra, upstream_url=url)

    assert sent("https://api.openai.com/v1/chat/completions") == extra
    assert sent("https://API.OpenAI.com:443/v1/chat/completions") == extra
    assert sent("http://api.openai.com/v1/chat/completions") == {}
    assert sent("https://api.openai.com:8443/v1/chat/completions") == {}


def test_a_configured_api_url_designates_its_own_origin() -> None:
    cfg = SimpleNamespace(openai_api_url="http://llm.internal:8080/v1")

    assert is_trusted_upstream("http://llm.internal:8080/v1/chat/completions", cfg)
    assert not is_trusted_upstream("https://llm.internal:8080/v1/chat/completions", cfg)
    assert not is_trusted_upstream("http://llm.internal:9090/v1/chat/completions", cfg)
    assert not is_trusted_upstream("http://llm.internal/v1/chat/completions", cfg)


def test_a_scheme_less_configured_api_url_means_https() -> None:
    cfg = SimpleNamespace(anthropic_api_url="gateway.example.com/v1")

    assert is_trusted_upstream("https://gateway.example.com/v1/messages", cfg)
    assert not is_trusted_upstream("http://gateway.example.com/v1/messages", cfg)


@pytest.mark.parametrize(
    ("entry", "trusted", "untrusted"),
    [
        ("llm.corp", "https://llm.corp/v1", ["http://llm.corp/v1", "https://llm.corp:8443/v1"]),
        ("llm.corp:8443", "https://llm.corp:8443/v1", ["https://llm.corp/v1"]),
        ("http://llm.corp:8080", "http://llm.corp:8080/v1", ["https://llm.corp:8080/v1"]),
    ],
)
def test_allowed_hosts_entries_designate_one_origin(
    monkeypatch: pytest.MonkeyPatch, entry: str, trusted: str, untrusted: list[str]
) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, f"other.example, {entry}")

    assert is_trusted_upstream(trusted)
    for url in untrusted:
        assert not is_trusted_upstream(url), url


@pytest.mark.parametrize(
    "url",
    [
        "https://[api.openai.com]/v1",
        "https://api.openai.com:notaport/v1",
    ],
)
def test_malformed_destinations_are_not_trusted(url: str) -> None:
    assert not is_trusted_upstream(url)


def test_each_provider_extras_stay_with_that_provider() -> None:
    """OpenAI extras never go to Anthropic's API, and Anthropic extras never go to OpenAI's."""
    openai_extra = {"x-operator-oai": "oai-secret"}
    anthropic_extra = {"x-operator-ant": "ant-secret"}
    cfg = SimpleNamespace(openai_api_url="https://gateway.example/v1")

    def sent(extra: dict[str, str], url: str, provider: str) -> dict[str, str]:
        return merge_extra_headers({}, extra, upstream_url=url, config=cfg, provider=provider)

    assert sent(openai_extra, "https://api.openai.com/v1", "openai") == openai_extra
    assert sent(openai_extra, "https://gateway.example/v1", "openai") == openai_extra
    assert sent(openai_extra, "https://api.anthropic.com/v1", "openai") == {}
    assert sent(anthropic_extra, "https://api.anthropic.com/v1", "anthropic") == anthropic_extra
    assert sent(anthropic_extra, "https://api.openai.com/v1", "anthropic") == {}
    assert sent(anthropic_extra, "https://gateway.example/v1", "anthropic") == {}


def test_an_allowed_host_is_designated_for_both_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "shared-gateway.example")
    for provider in ("openai", "anthropic"):
        assert is_trusted_upstream("https://shared-gateway.example/v1", None, provider)


def test_the_handlers_scope_extras_to_their_own_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wiring check: chat sends only OpenAI extras, messages only Anthropic extras."""
    from headroom.proxy.handlers.anthropic import _build_outbound_anthropic_headers
    from tests.test_grok_shared_proxy_call_sites import _OPENAI_CLIENT_HEADERS, _run_chat

    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "api.anthropic.com")
    outbound = _run_chat(
        {**_OPENAI_CLIENT_HEADERS, "x-headroom-base-url": "https://api.anthropic.com"},
        model="gpt-4o-mini",
        extras={"X-Operator-Sentinel": "must-not-leak"},
    )
    assert outbound.url.startswith("https://api.anthropic.com")
    assert "x-operator-sentinel" not in outbound.headers

    headers = _build_outbound_anthropic_headers(
        {},
        {"x-operator-ant": "ant-secret"},
        routed_to_omlx=False,
        upstream_url="https://api.openai.com",
        config=SimpleNamespace(),
    )
    assert "x-operator-ant" not in headers
