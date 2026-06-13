"""CORS guardrails for local proxy content endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from headroom.proxy.cors import CORS_ORIGINS_ENV, cors_origins_for_config


@dataclass(frozen=True)
class _Config:
    port: int


def test_cors_defaults_to_configured_localhost_port(monkeypatch) -> None:
    monkeypatch.delenv(CORS_ORIGINS_ENV, raising=False)

    origins = cors_origins_for_config(_Config(port=9901))

    assert origins == ["http://127.0.0.1:9901", "http://localhost:9901"]


def test_cors_env_override_allows_explicit_wildcard(monkeypatch) -> None:
    monkeypatch.setenv(CORS_ORIGINS_ENV, "*")

    assert cors_origins_for_config(_Config(port=9901)) == ["*"]


def test_cors_env_override_trims_custom_origins(monkeypatch) -> None:
    monkeypatch.setenv(
        CORS_ORIGINS_ENV,
        " https://dashboard.example.test, http://127.0.0.1:7777 ,,",
    )

    origins = cors_origins_for_config(_Config(port=9901))

    assert origins == ["https://dashboard.example.test", "http://127.0.0.1:7777"]
