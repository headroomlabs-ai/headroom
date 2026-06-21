"""Tests for OpenCode provider runtime: provider discovery, overlay, launch env."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.providers.opencode.runtime import (
    _KNOWN_UPSTREAMS,
    _strip_jsonc_comments,
    build_launch_env,
    build_overlay,
    build_provider_upstream_map,
    discover_user_providers,
    has_zen_auth,
    proxy_base_url,
)


# ============================================================================
# JSONC comment stripping
# ============================================================================


class TestStripJsoncComments:
    def test_plain_json_unchanged(self):
        text = '{"key": "value"}'
        assert _strip_jsonc_comments(text).strip() == text

    def test_line_comment_removed(self):
        text = '{\n  // this is a comment\n  "key": "value"\n}'
        result = _strip_jsonc_comments(text)
        assert "// this is a comment" not in result
        assert '"key": "value"' in result

    def test_block_comment_removed(self):
        text = '/* header */\n{"key": "value"}'
        result = _strip_jsonc_comments(text)
        assert "header" not in result
        assert '"key": "value"' in result

    def test_string_with_comment_like_content_preserved(self):
        text = '{"url": "https://api.deepseek.com/v1"}'
        result = _strip_jsonc_comments(text)
        assert "https://api.deepseek.com/v1" in result

    def test_comment_inside_string_preserved(self):
        text = '{"msg": "hello // world"}'
        result = _strip_jsonc_comments(text)
        assert "hello // world" in result

    def test_escaped_quote_in_string(self):
        text = r'{"msg": "hello \"world\""}'
        result = _strip_jsonc_comments(text)
        assert r'hello \"world\"' in result

    def test_empty_string(self):
        assert _strip_jsonc_comments("").strip() == ""

    def test_only_comments(self):
        assert _strip_jsonc_comments("// just a comment\n").strip() == ""

    def test_multiline_block_comment(self):
        text = '{\n  /* line1\n     line2 */\n  "key": 1\n}'
        result = _strip_jsonc_comments(text)
        assert "line1" not in result
        assert "line2" not in result
        assert '"key": 1' in result


# ============================================================================
# Known upstreams
# ============================================================================


class TestKnownUpstreams:
    def test_all_entries_have_urls(self):
        for name, url in _KNOWN_UPSTREAMS.items():
            assert url.startswith("https://"), f"non-https URL for {name}"

    def test_opencode_go_is_mapped(self):
        assert "opencode-go" in _KNOWN_UPSTREAMS
        assert _KNOWN_UPSTREAMS["opencode-go"] == "https://opencode.ai/zen/go/v1"

    def test_opencode_zen_is_mapped(self):
        assert "opencode" in _KNOWN_UPSTREAMS
        assert _KNOWN_UPSTREAMS["opencode"] == "https://opencode.ai/zen/v1"

    def test_common_providers_are_mapped(self):
        for name in ("anthropic", "openai", "deepseek", "google", "groq", "xai"):
            assert name in _KNOWN_UPSTREAMS


# ============================================================================
# Provider discovery
# ============================================================================


class TestDiscoverUserProviders:
    def test_empty_when_no_config(self):
        providers = discover_user_providers()
        assert isinstance(providers, dict)

    def test_parses_mimo_custom_provider(self, tmp_path: Path):
        config = tmp_path / "opencode.json"
        config.write_text(
            json.dumps(
                {
                    "provider": {
                        "mimo": {
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {"baseURL": "https://custom.api.com/v1"},
                        }
                    }
                }
            )
        )
        with patch(
            "headroom.providers.opencode.runtime._user_config_path",
            return_value=config,
        ):
            providers = discover_user_providers()
            assert "mimo" in providers
            assert providers["mimo"]["options"]["baseURL"] == "https://custom.api.com/v1"


class TestBuildProviderUpstreamMap:
    def test_returns_dict(self):
        result = build_provider_upstream_map()
        assert isinstance(result, dict)

    def test_known_providers_in_map(self):
        result = build_provider_upstream_map()
        # These are in the user's auth.json + have known upstreams
        for name in ("deepseek", "openai", "opencode", "opencode-go", "google"):
            assert name in result, f"{name} not in upstream map"

    def test_oauth_provider_excluded(self):
        result = build_provider_upstream_map()
        assert "github-copilot" not in result

    def test_localhost_urls_filtered(self, tmp_path: Path):
        config = tmp_path / "opencode.json"
        config.write_text(
            json.dumps(
                {
                    "provider": {
                        "bad_provider": {
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {
                                "baseURL": "http://127.0.0.1:8787/v1"
                            },
                        }
                    }
                }
            )
        )
        auth = tmp_path / "auth.json"
        auth.write_text(
            json.dumps({"bad_provider": {"type": "api", "key": "sk-test"}})
        )
        with patch(
            "headroom.providers.opencode.runtime._user_config_path",
            return_value=config,
        ), patch(
            "headroom.providers.opencode.runtime._AUTH_PATH", auth
        ):
            result = build_provider_upstream_map()
            assert "bad_provider" not in result

    def test_empty_when_no_auth(self, tmp_path: Path):
        auth = tmp_path / "auth.json"
        auth.write_text("{}")
        with patch(
            "headroom.providers.opencode.runtime._AUTH_PATH", auth
        ), patch(
            "headroom.providers.opencode.runtime._user_config_path",
            return_value=None,
        ):
            result = build_provider_upstream_map()
            for k, v in result.items():
                assert "127.0.0.1" not in v
                assert "localhost" not in v


class TestHasZenAuth:
    def test_returns_bool(self):
        result = has_zen_auth()
        assert isinstance(result, bool)

    def test_false_when_missing_file(self, tmp_path: Path):
        auth = tmp_path / "auth.json"
        with patch(
            "headroom.providers.opencode.runtime._AUTH_PATH", auth
        ):
            assert has_zen_auth() is False

    def test_true_with_opencode_api(self, tmp_path: Path):
        auth = tmp_path / "auth.json"
        auth.write_text(
            json.dumps({"opencode": {"type": "api", "key": "sk-test"}})
        )
        with patch(
            "headroom.providers.opencode.runtime._AUTH_PATH", auth
        ):
            assert has_zen_auth() is True

    def test_true_with_opencode_go_api(self, tmp_path: Path):
        auth = tmp_path / "auth.json"
        auth.write_text(
            json.dumps({"opencode-go": {"type": "api", "key": "sk-test"}})
        )
        with patch(
            "headroom.providers.opencode.runtime._AUTH_PATH", auth
        ):
            assert has_zen_auth() is True


# ============================================================================
# URL construction
# ============================================================================


class TestProxyBaseUrl:
    def test_default(self):
        assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"

    def test_with_project(self):
        url = proxy_base_url(8787, "myproject")
        assert url.startswith("http://127.0.0.1:8787/p/myproject/v1")


# ============================================================================
# Overlay generation
# ============================================================================


class TestBuildOverlay:
    def test_multi_mode_returns_provider_key(self):
        pm = {"deepseek": "https://api.deepseek.com"}
        overlay = build_overlay(pm, 8787, routing_mode="multi")
        assert "provider" in overlay
        assert "deepseek" in overlay["provider"]

    def test_single_mode_returns_provider_key(self):
        pm = {"deepseek": "https://api.deepseek.com"}
        overlay = build_overlay(pm, 8787, routing_mode="single")
        assert "provider" in overlay
        assert "deepseek" in overlay["provider"]

    def test_single_mode_includes_headers_for_passthrough(self):
        pm = {"deepseek": "https://api.deepseek.com"}
        overlay = build_overlay(pm, 8787, routing_mode="single")
        entry = overlay["provider"]["deepseek"]
        assert "headers" in entry["options"]
        assert (
            entry["options"]["headers"]["x-headroom-base-url"]
            == "https://api.deepseek.com"
        )

    def test_single_mode_excludes_headers_for_openai(self):
        pm = {"openai": "https://api.openai.com/v1"}
        overlay = build_overlay(pm, 8787, routing_mode="single")
        entry = overlay["provider"]["openai"]
        assert "headers" not in entry["options"]

    def test_single_mode_openai_gets_dedicated_port(self):
        pm = {
            "deepseek": "https://api.deepseek.com",
            "openai": "https://api.openai.com/v1",
        }
        overlay = build_overlay(pm, 8787, routing_mode="single")
        assert overlay["provider"]["deepseek"]["options"]["baseURL"].endswith("8787/v1")
        assert overlay["provider"]["openai"]["options"]["baseURL"].endswith("8788/v1")

    def test_multi_mode_uses_port_map(self):
        pm = {"deepseek": "https://api.deepseek.com", "groq": "https://api.groq.com"}
        port_map = {"deepseek": 9000, "groq": 9001}
        overlay = build_overlay(pm, 8787, routing_mode="multi", port_map=port_map)
        assert overlay["provider"]["deepseek"]["options"]["baseURL"].endswith("9000/v1")
        assert overlay["provider"]["groq"]["options"]["baseURL"].endswith("9001/v1")

    def test_empty_provider_map(self):
        pm: dict[str, str] = {}
        overlay = build_overlay(pm, 8787)
        assert overlay == {"provider": {}}


# ============================================================================
# Launch environment
# ============================================================================


class TestBuildLaunchEnv:
    def test_sets_opencode_config_content(self):
        env, display = build_launch_env(8787)
        assert "OPENCODE_CONFIG_CONTENT" in env
        content = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert "provider" in content

    def test_preserves_existing_base_urls(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.openai.com")
        env, display = build_launch_env(8787)
        assert "OPENAI_BASE_URL" in env
        assert env["OPENAI_BASE_URL"] == "https://custom.openai.com"

    def test_includes_project_name_in_env(self):
        env, display = build_launch_env(8787, project="myproject")
        assert env.get("HEADROOM_PROJECT") == "myproject"

    def test_does_not_clobber_openai_base_url(self):
        environ = {"OPENAI_BASE_URL": "https://original.openai.com"}
        env, display = build_launch_env(8787, environ=environ)
        assert env["OPENAI_BASE_URL"] == "https://original.openai.com"
