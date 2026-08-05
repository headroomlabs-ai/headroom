"""Ports of the LiteLLM/Kong sidecar patches (see the sidecar PATCHES.md #1/#4/#5/#6).

- #1  HEADROOM_COMPRESS_ALLOW_REMOTE opt-in drops the loopback guard on
      /v1/compress so an authorized in-network gateway (Kong, LiteLLM) can reach it.
- #4/#5  HEADROOM_MODEL_ALIAS_MAP reduces a gateway-aliased model name (e.g.
      "claude-opus") to a priced litellm.model_cost key — one shared resolver for
      the live (cost.py) and persisted (savings_tracker) price paths.
- #6  an operator-configured context limit (HEADROOM_MODEL_LIMITS) wins over the
      128K default for an aliased name.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _clear_pricing_cache() -> None:
    from headroom.pricing import litellm_pricing as lp

    lp._resolved_model_cache.clear()


def _first_priced_opus_key() -> str:
    litellm = pytest.importorskip("litellm")
    for key, val in litellm.model_cost.items():
        if "opus" in key.lower() and val.get("input_cost_per_token") is not None:
            return key
    pytest.skip("no priced opus key in this litellm build")


# ----- #4/#5 pricing: HEADROOM_MODEL_ALIAS_MAP -> priced key -----


def test_alias_map_resolves_gateway_name_to_priced_key(monkeypatch):
    litellm = pytest.importorskip("litellm")
    from headroom.pricing.litellm_pricing import resolve_litellm_model

    key = _first_priced_opus_key()
    monkeypatch.setenv("HEADROOM_MODEL_ALIAS_MAP", json.dumps({"claude-opus": key}))
    _clear_pricing_cache()

    resolved = resolve_litellm_model("claude-opus")
    info = litellm.model_cost.get(resolved)
    assert info and info.get("input_cost_per_token") is not None


def test_alias_map_strips_bedrock_prefix(monkeypatch):
    pytest.importorskip("litellm")
    from headroom.pricing.litellm_pricing import resolve_litellm_model

    key = _first_priced_opus_key()
    monkeypatch.setenv("HEADROOM_MODEL_ALIAS_MAP", json.dumps({"claude-opus": f"bedrock/{key}"}))
    _clear_pricing_cache()
    assert resolve_litellm_model("claude-opus") == key


def test_unpriced_alias_falls_through_soft(monkeypatch):
    from headroom.pricing.litellm_pricing import resolve_litellm_model

    monkeypatch.setenv("HEADROOM_MODEL_ALIAS_MAP", json.dumps({"claude-opus": "not-a-real-model"}))
    _clear_pricing_cache()
    # No crash; falls back to bare-prefix resolution (never returns the bogus target).
    assert resolve_litellm_model("claude-opus") != "not-a-real-model"


def test_unset_env_is_unchanged(monkeypatch):
    from headroom.pricing.litellm_pricing import resolve_litellm_model

    monkeypatch.delenv("HEADROOM_MODEL_ALIAS_MAP", raising=False)
    _clear_pricing_cache()
    assert isinstance(resolve_litellm_model("gpt-4o"), str)


def test_savings_tracker_delegates_to_shared_resolver(monkeypatch):
    pytest.importorskip("litellm")
    from headroom.proxy.savings_tracker import _resolve_litellm_model

    key = _first_priced_opus_key()
    monkeypatch.setenv("HEADROOM_MODEL_ALIAS_MAP", json.dumps({"claude-opus": key}))
    _clear_pricing_cache()
    # Persisted funnel prices the alias identically to the live path.
    assert _resolve_litellm_model("claude-opus") == key


# ----- #6 context limit: configured alias wins over the 128K default -----


def test_configured_context_limit_wins_over_default(monkeypatch):
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS",
        json.dumps({"openai": {"context_limits": {"claude-opus": 200000}}}),
    )
    from headroom.providers.openai import OpenAIProvider

    provider = OpenAIProvider()
    assert provider.get_context_limit("claude-opus") == 200000  # not the 128000 default


# ----- #1 loopback opt-in on /v1/compress -----


def _fast_app():
    return create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )


_BODY = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"}


def test_compress_blocks_non_loopback_by_default(monkeypatch):
    monkeypatch.delenv("HEADROOM_COMPRESS_ALLOW_REMOTE", raising=False)
    # A vanilla TestClient presents client.host="testclient" (non-loopback).
    client = TestClient(_fast_app())
    assert client.post("/v1/compress", json=_BODY).status_code == 404


def test_compress_allows_non_loopback_with_flag(monkeypatch):
    monkeypatch.setenv("HEADROOM_COMPRESS_ALLOW_REMOTE", "1")
    client = TestClient(_fast_app())
    resp = client.post("/v1/compress", json=_BODY)
    assert resp.status_code == 200, resp.text
