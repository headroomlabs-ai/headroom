"""Tests for pure proxy mode normalization policy."""

from __future__ import annotations

from headroom.proxy.proxy_mode_policy import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    normalize_proxy_mode_decision,
    normalize_proxy_mode_value,
)


def test_decision_normalizes_canonical_modes() -> None:
    token = normalize_proxy_mode_decision("token")
    assert token.normalized == PROXY_MODE_TOKEN
    assert token.alias_used is False
    assert token.unknown is False

    cache = normalize_proxy_mode_decision("cache")
    assert cache.normalized == PROXY_MODE_CACHE
    assert cache.alias_used is False
    assert cache.unknown is False


def test_decision_normalizes_aliases_and_marks_alias_used() -> None:
    token = normalize_proxy_mode_decision(" token_headroom ")
    assert token.key == "token_headroom"
    assert token.normalized == PROXY_MODE_TOKEN
    assert token.alias_used is True

    cache = normalize_proxy_mode_decision("cost_savings")
    assert cache.normalized == PROXY_MODE_CACHE
    assert cache.alias_used is True


def test_decision_uses_default_for_blank_mode() -> None:
    decision = normalize_proxy_mode_decision(" ", default=PROXY_MODE_CACHE)
    assert decision.normalized == PROXY_MODE_CACHE
    assert decision.used_default is True
    assert decision.unknown is False


def test_decision_uses_default_and_marks_unknown_for_invalid_mode() -> None:
    decision = normalize_proxy_mode_decision("wat", default=PROXY_MODE_CACHE)
    assert decision.normalized == PROXY_MODE_CACHE
    assert decision.used_default is True
    assert decision.unknown is True


def test_value_helper_returns_only_canonical_mode() -> None:
    assert normalize_proxy_mode_value("token_savings") == PROXY_MODE_TOKEN
    assert normalize_proxy_mode_value("cache_mode") == PROXY_MODE_CACHE
