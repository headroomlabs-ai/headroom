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
