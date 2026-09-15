"""CORS guardrails for local proxy content endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from headroom.proxy.cors import (
    CORS_ORIGINS_ENV,
    DEFAULT_LOOPBACK_ORIGIN_REGEX,
    cors_origin_regex_for_config,
    cors_origins_for_config,
)


@dataclass(frozen=True)
class _Config:
    port: int


def test_cors_defaults_to_loopback_regex(monkeypatch) -> None:
    monkeypatch.delenv(CORS_ORIGINS_ENV, raising=False)

    assert cors_origins_for_config(_Config(port=9901)) == []
    assert cors_origin_regex_for_config(_Config(port=9901)) == DEFAULT_LOOPBACK_ORIGIN_REGEX


def test_cors_env_override_allows_explicit_wildcard(monkeypatch) -> None:
    monkeypatch.setenv(CORS_ORIGINS_ENV, "*")

    assert cors_origins_for_config(_Config(port=9901)) == ["*"]
    assert cors_origin_regex_for_config(_Config(port=9901)) is None


def test_cors_env_override_trims_custom_origins(monkeypatch) -> None:
    monkeypatch.setenv(
        CORS_ORIGINS_ENV,
        " https://dashboard.example.test, http://127.0.0.1:7777 ,,",
    )

    origins = cors_origins_for_config(_Config(port=9901))

    assert origins == ["https://dashboard.example.test", "http://127.0.0.1:7777"]
    assert cors_origin_regex_for_config(_Config(port=9901)) is None
