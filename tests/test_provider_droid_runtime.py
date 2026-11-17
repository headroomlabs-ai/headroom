"""Tests for the Factory Droid runtime helpers behind ``headroom wrap droid``."""

from __future__ import annotations

import pytest

from headroom.providers.droid import (
    DEFAULT_FACTORY_API_URL,
    canonical_factory_api_url,
    proxy_base_url,
    resolve_factory_upstream,
)


@pytest.fixture(autouse=True)
def _no_inherited_factory_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACTORY_API_BASE_URL", raising=False)


def test_default_upstream_is_public_factory() -> None:
    assert DEFAULT_FACTORY_API_URL == "https://api.factory.ai"
    assert resolve_factory_upstream(None) == DEFAULT_FACTORY_API_URL


def test_env_beats_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_API_BASE_URL", "https://eu.factory.example/")
    assert resolve_factory_upstream(None) == "https://eu.factory.example"


def test_explicit_flag_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_API_BASE_URL", "https://eu.factory.example")
    assert resolve_factory_upstream("https://flag.factory.example") == (
        "https://flag.factory.example"
    )


def test_proxy_base_url_is_loopback_without_v1() -> None:
    assert proxy_base_url(8799) == "http://127.0.0.1:8799"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://API.Factory.AI", "https://api.factory.ai"),
        ("https://api.factory.ai/", "https://api.factory.ai"),
        ("https://api.factory.ai:443", "https://api.factory.ai"),
        ("http://gw.example:80/", "http://gw.example"),
        ("https://gw.example:8443/factory/", "https://gw.example:8443/factory"),
        ("  https://api.factory.ai  ", "https://api.factory.ai"),
        ("https://[2001:db8::1]:8443", "https://[2001:db8::1]:8443"),
    ],
)
def test_canonicalisation(raw: str, expected: str) -> None:
    assert canonical_factory_api_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # loopback, in every spelling that reaches this host
        "http://127.0.0.1:8787",
        "http://localhost:8787",
        "http://LOCALHOST",
        "http://x.localhost",
        "http://[::1]:8787",
        "http://127.1:8787",
        "http://0.0.0.0:8787",
        # credentials, query, fragment
        "https://user:pw@api.factory.ai",
        "https://user@api.factory.ai",
        "https://api.factory.ai?x=1",
        "https://api.factory.ai#frag",
        # schemes and malformed values
        "ftp://api.factory.ai",
        "file:///etc/passwd",
        "api.factory.ai",
        "https://",
        "https://api.factory.ai:99999",
        "https://api.factory.ai:notaport",
        "https://bad_host!.example",
        "",
        None,
        42,
    ],
)
def test_rejections(raw: object) -> None:
    assert canonical_factory_api_url(raw) is None
