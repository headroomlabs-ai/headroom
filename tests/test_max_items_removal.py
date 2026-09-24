"""Regression tests for removing max_items_after_crush.

The setting was forwarded into the Rust crusher and ignored. Removing it is
behaviour-preserving, but it is still a public name that scripts may set, so the
entry points that used to read it have to say it no longer does anything, and a
worker started by a parent of a different generation must not lose its config.
"""

import json
import logging

import pytest

from headroom.config import SmartCrusherConfig as ConfigSmartCrusherConfig
from headroom.proxy.models import ProxyConfig
from headroom.transforms.smart_crusher import SmartCrusherConfig as TransformSmartCrusherConfig


class TestSettingIsGone:
    @pytest.mark.parametrize(
        "cls", [ConfigSmartCrusherConfig, TransformSmartCrusherConfig, ProxyConfig]
    )
    def test_constructor_rejects_the_removed_kwarg(self, cls):
        """Both same-named SmartCrusherConfig dataclasses, and ProxyConfig."""
        with pytest.raises(TypeError):
            cls(max_items_after_crush=15)

    def test_not_forwarded_to_the_rust_crusher(self):
        from headroom.transforms.smart_crusher import SmartCrusher

        crusher = SmartCrusher(config=TransformSmartCrusherConfig())
        assert "max_items_after_crush" not in crusher._rust_cfg_kwargs

    def test_mcp_profiles_carry_no_item_count(self):
        from headroom.integrations.mcp.server import DEFAULT_MCP_PROFILES

        for profile in DEFAULT_MCP_PROFILES:
            assert not hasattr(profile, "max_items")


class TestDeprecationIsAnnounced:
    def test_warns_when_the_env_var_is_set(self, monkeypatch, caplog):
        import headroom.proxy.models as models

        monkeypatch.setattr(models, "_MAX_ITEMS_WARNED", False)
        monkeypatch.setenv("HEADROOM_MAX_ITEMS", "5")
        with caplog.at_level(logging.WARNING):
            models.warn_if_max_items_configured()
        assert "HEADROOM_MAX_ITEMS is deprecated" in caplog.text

    def test_silent_when_unset(self, monkeypatch, caplog):
        import headroom.proxy.models as models

        monkeypatch.setattr(models, "_MAX_ITEMS_WARNED", False)
        monkeypatch.delenv("HEADROOM_MAX_ITEMS", raising=False)
        with caplog.at_level(logging.WARNING):
            models.warn_if_max_items_configured()
        assert "deprecated" not in caplog.text

    def test_warns_only_once(self, monkeypatch, caplog):
        import headroom.proxy.models as models

        monkeypatch.setattr(models, "_MAX_ITEMS_WARNED", False)
        monkeypatch.setenv("HEADROOM_MAX_ITEMS", "5")
        with caplog.at_level(logging.WARNING):
            models.warn_if_max_items_configured()
            models.warn_if_max_items_configured()
        assert caplog.text.count("HEADROOM_MAX_ITEMS is deprecated") == 1


class TestWorkerConfigSurvivesUnknownKeys:
    def test_unknown_key_does_not_discard_the_parent_config(self, monkeypatch, caplog):
        from headroom.proxy.server import _MULTI_WORKER_CONFIG_ENV, _proxy_config_from_env

        monkeypatch.setenv(
            _MULTI_WORKER_CONFIG_ENV,
            json.dumps({"port": 9999, "min_tokens_to_crush": 123, "max_items_after_crush": 50}),
        )
        with caplog.at_level(logging.WARNING):
            config = _proxy_config_from_env()

        # Without the filter this falls back to env-var defaults and loses both.
        assert config.port == 9999
        assert config.min_tokens_to_crush == 123
        assert "Ignoring unknown" in caplog.text
