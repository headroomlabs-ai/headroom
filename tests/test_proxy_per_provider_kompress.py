"""Per-provider Kompress enable/disable (disable_kompress_{anthropic,openai}).

The global ``disable_kompress`` is the baseline for both providers; a per-provider
override wins when set. Only ``enable_kompress`` differs between the two pipelines,
so when both resolve identically they reuse ONE ContentRouter instance (keeping the
single Kompress model load).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli.main import main
from headroom.proxy.server import (
    HeadroomProxy,
    ProxyConfig,
    _get_env_optional_bool,
    _proxy_config_from_env,
)


def _build(**overrides: object) -> HeadroomProxy:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        code_aware_enabled=False,
        **overrides,
    )
    return HeadroomProxy(config)


def _routers(proxy: HeadroomProxy):
    # ContentRouter is the last transform in each pipeline.
    return (
        proxy.anthropic_pipeline.transforms[-1],
        proxy.openai_pipeline.transforms[-1],
    )


def test_default_enables_kompress_and_shares_one_router() -> None:
    anthropic, openai = _routers(_build())
    assert anthropic.config.enable_kompress is True
    assert openai.config.enable_kompress is True
    # Identical resolution -> one shared instance (Kompress model loads once).
    assert anthropic is openai


def test_global_disable_respected_by_both() -> None:
    anthropic, openai = _routers(_build(disable_kompress=True))
    assert anthropic.config.enable_kompress is False
    assert openai.config.enable_kompress is False
    assert anthropic is openai


def test_disable_for_anthropic_only() -> None:
    anthropic, openai = _routers(_build(disable_kompress_anthropic=True))
    assert anthropic.config.enable_kompress is False
    assert openai.config.enable_kompress is True
    assert anthropic is not openai


def test_disable_for_openai_only() -> None:
    anthropic, openai = _routers(_build(disable_kompress_openai=True))
    assert anthropic.config.enable_kompress is True
    assert openai.config.enable_kompress is False
    assert anthropic is not openai


def test_per_provider_override_beats_global() -> None:
    # Global disables Kompress; Anthropic override force-enables it, OpenAI inherits.
    anthropic, openai = _routers(_build(disable_kompress=True, disable_kompress_anthropic=False))
    assert anthropic.config.enable_kompress is True
    assert openai.config.enable_kompress is False
    assert anthropic is not openai


def test_get_env_optional_bool_tristate() -> None:
    os.environ.pop("HRD_KOMPRESS_TEST", None)
    assert _get_env_optional_bool("HRD_KOMPRESS_TEST") is None  # unset
    with patch.dict(os.environ, {"HRD_KOMPRESS_TEST": ""}):
        assert _get_env_optional_bool("HRD_KOMPRESS_TEST") is None  # empty
    for truthy in ("1", "true", "yes", "on"):
        with patch.dict(os.environ, {"HRD_KOMPRESS_TEST": truthy}):
            assert _get_env_optional_bool("HRD_KOMPRESS_TEST") is True
    for falsy in ("0", "false", "no", "off"):
        with patch.dict(os.environ, {"HRD_KOMPRESS_TEST": falsy}):
            assert _get_env_optional_bool("HRD_KOMPRESS_TEST") is False


def test_proxy_config_from_env_reads_per_provider_kompress() -> None:
    with patch.dict(
        os.environ,
        {
            "HEADROOM_DISABLE_KOMPRESS_ANTHROPIC": "1",
            "HEADROOM_DISABLE_KOMPRESS_OPENAI": "0",
        },
    ):
        config = _proxy_config_from_env()
    assert config.disable_kompress_anthropic is True
    assert config.disable_kompress_openai is False


def test_cli_disable_kompress_anthropic_only() -> None:
    captured: dict = {}

    def mock_run_server(config, **kwargs):
        captured["config"] = config

    with patch("headroom.proxy.server.run_server", mock_run_server):
        result = CliRunner().invoke(
            main,
            ["proxy", "--disable-kompress-anthropic"],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert captured["config"].disable_kompress_anthropic is True
    assert captured["config"].disable_kompress_openai is None


def test_cli_enable_kompress_openai_from_env() -> None:
    captured: dict = {}

    def mock_run_server(config, **kwargs):
        captured["config"] = config

    with patch("headroom.proxy.server.run_server", mock_run_server):
        result = CliRunner().invoke(
            main,
            ["proxy"],
            env={"HEADROOM_DISABLE_KOMPRESS_OPENAI": "0"},
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert captured["config"].disable_kompress_openai is False
