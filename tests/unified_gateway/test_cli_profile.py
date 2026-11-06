from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.proxy import proxy

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


def test_proxy_help_advertises_gateway_profile() -> None:
    result = CliRunner().invoke(proxy, ["--help"])

    assert result.exit_code == 0
    assert "--gateway" in result.output
    assert "--gateway-config" in result.output
    assert "--check-config" in result.output


def test_check_config_exits_before_server_start(monkeypatch) -> None:
    monkeypatch.setattr("headroom.cli.proxy._reexec_with_malloc_tuning", lambda: None)
    monkeypatch.setattr("headroom.cli.proxy.ensure_proxy_dependencies", lambda **_kwargs: None)
    monkeypatch.setattr(
        "headroom.proxy.server.run_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("server started")),
    )

    result = CliRunner().invoke(
        proxy,
        ["--gateway", "--gateway-config", str(EXAMPLE), "--check-config"],
    )

    assert result.exit_code == 0, result.output
    assert "gateway configuration valid" in result.output.lower()
    assert "OPENAI_API_KEY" not in result.output


def test_gateway_requires_config() -> None:
    result = CliRunner().invoke(proxy, ["--gateway", "--check-config"])

    assert result.exit_code != 0
    assert "--gateway-config" in result.output


@pytest.mark.parametrize("flag", ["--memory", "--code-graph", "--lossless"])
def test_gateway_rejects_transforming_flags(monkeypatch, flag: str) -> None:
    monkeypatch.setattr("headroom.cli.proxy._reexec_with_malloc_tuning", lambda: None)
    monkeypatch.setattr("headroom.cli.proxy.ensure_proxy_dependencies", lambda **_kwargs: None)

    result = CliRunner().invoke(
        proxy,
        ["--gateway", "--gateway-config", str(EXAMPLE), "--check-config", flag],
    )

    assert result.exit_code != 0
    assert "incompatible with gateway pure mode" in result.output
