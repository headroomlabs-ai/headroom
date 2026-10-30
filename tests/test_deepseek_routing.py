"""Tests for DeepSeek traffic detection and upstream routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from headroom.proxy.handlers.openai import OpenAIHandlerMixin, _is_deepseek_request


@pytest.fixture(autouse=True)
def _allow_test_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep routing-precedence tests independent of DNS and SSRF policy."""
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.example")


def _headers(*pairs: tuple[str, str]) -> dict[str, str]:
    return dict(pairs)


def test_is_deepseek_request_header() -> None:
    assert _is_deepseek_request(_headers(("x-deepseek-harness-user-id", "anon")), model=None)


def test_is_deepseek_request_model_prefix() -> None:
    assert _is_deepseek_request(_headers(), model="deepseek-v4-flash")


def test_is_deepseek_request_neither() -> None:
    assert not _is_deepseek_request(_headers(), model="gpt-4o")


def test_is_deepseek_request_none_model() -> None:
    assert not _is_deepseek_request(_headers(), model=None)


def _mix() -> Any:
    mix = OpenAIHandlerMixin.__new__(OpenAIHandlerMixin)
    mix.OPENAI_API_URL = "https://api.openai.com"
    mix.DEEPSEEK_API_URL = "https://api.deepseek.com"
    return mix


def test_resolve_openai_upstream_deepseek_header() -> None:
    mix = _mix()
    req = SimpleNamespace(headers={"x-deepseek-harness-user-id": "anon"})
    assert mix._resolve_openai_upstream(req, model=None) == "https://api.deepseek.com"


def test_resolve_openai_upstream_deepseek_model() -> None:
    mix = _mix()
    req = SimpleNamespace(headers={})
    assert mix._resolve_openai_upstream(req, model="deepseek-v4-pro") == "https://api.deepseek.com"


def test_resolve_openai_upstream_openai_default() -> None:
    mix = _mix()
    req = SimpleNamespace(headers={})
    assert mix._resolve_openai_upstream(req, model="gpt-4o") == "https://api.openai.com"


def test_resolve_openai_upstream_custom_base_url_wins_when_not_deepseek() -> None:
    mix = _mix()
    req = SimpleNamespace(headers={"x-headroom-base-url": "https://gateway.example/v1"})
    assert mix._resolve_openai_upstream(req, model="gpt-4o") == "https://gateway.example/v1"


def test_resolve_openai_upstream_base_url_wins_over_deepseek_model() -> None:
    mix = _mix()
    req = SimpleNamespace(headers={"x-headroom-base-url": "https://gateway.example/v1"})
    assert (
        mix._resolve_openai_upstream(req, model="deepseek-v4-flash") == "https://gateway.example/v1"
    )


def test_resolve_openai_upstream_base_url_wins_over_deepseek_header() -> None:
    mix = _mix()
    req = SimpleNamespace(
        headers={
            "x-headroom-base-url": "https://gateway.example/v1",
            "x-deepseek-harness-user-id": "anon",
        }
    )
    assert mix._resolve_openai_upstream(req, model=None) == "https://gateway.example/v1"
