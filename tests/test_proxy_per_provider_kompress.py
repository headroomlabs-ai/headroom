"""Per-provider Kompress enable/disable (disable_kompress_{anthropic,openai}).

The global ``disable_kompress`` is the baseline for both providers; a per-provider
override wins when set. Only ``enable_kompress`` differs between the two pipelines,
so when both resolve identically they reuse ONE ContentRouter instance (keeping the
single Kompress model load).
"""

from __future__ import annotations

from headroom.proxy.server import HeadroomProxy, ProxyConfig


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
