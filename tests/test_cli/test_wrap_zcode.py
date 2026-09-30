"""Tests for `headroom wrap zcode` and `headroom unwrap zcode` commands.

ZCode is a desktop Electron app (zcode.z.ai) with no CLI binary. The wrap
command follows the Pattern-B (proxy-only watcher) approach: it starts the
proxy and prints the ZCode settings the user should configure in the app's
settings UI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Wrap: setup instructions output
# ---------------------------------------------------------------------------


def test_wrap_prints_proxy_urls(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrap command must print the proxy URLs for ZCode configuration."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    def fake_watcher(**kwargs):  # noqa: ANN003
        print_fn = kwargs.get("print_setup_lines")
        if callable(print_fn):
            print_fn(kwargs["port"])

    with patch.object(wrap_mod, "_run_proxy_only_watcher", side_effect=fake_watcher):
        result = runner.invoke(main, ["wrap", "zcode", "--port", "9000"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9000/v1" in result.output
    assert "http://127.0.0.1:9000" in result.output
    assert "Settings > Model Settings" in result.output


# ---------------------------------------------------------------------------
# Runtime: proxy targets
# ---------------------------------------------------------------------------


def test_build_proxy_targets() -> None:
    """build_proxy_targets returns correct OpenAI and Anthropic URLs."""
    from headroom.providers.zcode.runtime import build_proxy_targets

    targets = build_proxy_targets(8787)
    assert targets.openai_base_url == "http://127.0.0.1:8787/v1"
    assert targets.anthropic_base_url == "http://127.0.0.1:8787"


def test_build_proxy_targets_custom_port() -> None:
    """build_proxy_targets respects custom port."""
    from headroom.providers.zcode.runtime import build_proxy_targets

    targets = build_proxy_targets(9999)
    assert targets.openai_base_url == "http://127.0.0.1:9999/v1"
    assert targets.anthropic_base_url == "http://127.0.0.1:9999"


def test_render_setup_lines_includes_mcp_instruction() -> None:
    """render_setup_lines includes the MCP paste JSON for user convenience."""
    from headroom.providers.zcode.runtime import render_setup_lines

    lines = render_setup_lines(8787)
    joined = "\n".join(lines)
    assert "headroom" in joined.lower()
    assert "MCP" in joined
    assert '"stdio"' in joined


# ---------------------------------------------------------------------------
# Runtime: upstream detection
# ---------------------------------------------------------------------------


def test_detect_upstream_from_config(tmp_path: Path) -> None:
    """detect_upstream reads config.json and returns the enabled provider."""
    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "config.json"
    config.write_text(
        '{"provider": {"zai": {"name": "Z.ai", "kind": "anthropic", '
        '"enabled": true, "options": {"baseURL": "https://api.z.ai/api/anthropic"}}}}',
        encoding="utf-8",
    )
    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"
    assert upstream.kind == "anthropic"


def test_detect_upstream_openai_compatible(tmp_path: Path) -> None:
    """detect_upstream handles OpenAI-compatible providers."""
    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "config.json"
    config.write_text(
        '{"provider": {"custom": {"name": "Custom", "kind": "openai", '
        '"enabled": true, "options": {"baseURL": "https://my-api.example.com/v1"}}}}',
        encoding="utf-8",
    )
    upstream = detect_upstream(config)
    assert upstream.kind == "openai"
    assert upstream.base_url == "https://my-api.example.com/v1"


def test_detect_upstream_disabled_provider_ignored(tmp_path: Path) -> None:
    """detect_upstream skips disabled providers."""
    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "config.json"
    config.write_text(
        '{"provider": {"zai": {"name": "Z.ai", "kind": "anthropic", '
        '"enabled": false, "options": {"baseURL": "https://api.z.ai/api/anthropic"}}}}',
        encoding="utf-8",
    )
    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai (default)"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"


def test_detect_upstream_no_baseurl_skips(tmp_path: Path) -> None:
    """detect_upstream skips providers with empty or missing baseURL."""
    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "config.json"
    config.write_text(
        '{"provider": {"zai": {"name": "Z.ai", "kind": "anthropic", '
        '"enabled": true, "options": {"baseURL": ""}}}}',
        encoding="utf-8",
    )
    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai (default)"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"


@pytest.mark.parametrize(
    "bad_options",
    [
        None,
        [],
        42,
        "just a string",
    ],
    ids=["null", "list", "int", "string"],
)
def test_detect_upstream_malformed_options_skips(tmp_path: Path, bad_options: object) -> None:
    """detect_upstream falls back when provider options is not a dict."""
    import json as _json

    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "config.json"
    config.write_text(
        _json.dumps(
            {
                "provider": {
                    "bad": {
                        "name": "Bad Provider",
                        "kind": "anthropic",
                        "enabled": True,
                        "options": bad_options,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai (default)"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"


def test_detect_upstream_missing_file_fallback(tmp_path: Path) -> None:
    """detect_upstream falls back to default when config file is missing."""
    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "nonexistent.json"
    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai (default)"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"
    assert upstream.kind == "anthropic"


def test_detect_upstream_invalid_json_fallback(tmp_path: Path) -> None:
    """detect_upstream falls back to default on malformed JSON."""
    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "config.json"
    config.write_text("not json at all", encoding="utf-8")
    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai (default)"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"


def test_detect_upstream_no_providers_fallback(tmp_path: Path) -> None:
    """detect_upstream falls back when config has no provider section."""
    from headroom.providers.zcode.runtime import detect_upstream

    config = tmp_path / "config.json"
    config.write_text('{"settings": {}}', encoding="utf-8")
    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai (default)"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"


# ---------------------------------------------------------------------------
# Runtime: upstream to proxy URLs
# ---------------------------------------------------------------------------


def test_upstream_to_proxy_urls_anthropic() -> None:
    """upstream_to_proxy_urls returns (url, None) for anthropic upstream."""
    from headroom.providers.zcode.runtime import ZCodeUpstream, upstream_to_proxy_urls

    upstream = ZCodeUpstream(
        provider_name="Z.ai", base_url="https://api.z.ai/api/anthropic", kind="anthropic"
    )
    anthropic_url, openai_url = upstream_to_proxy_urls(upstream)
    assert anthropic_url == "https://api.z.ai/api/anthropic"
    assert openai_url is None


def test_upstream_to_proxy_urls_openai() -> None:
    """upstream_to_proxy_urls returns (None, url) for openai-compatible upstream."""
    from headroom.providers.zcode.runtime import ZCodeUpstream, upstream_to_proxy_urls

    upstream = ZCodeUpstream(
        provider_name="Custom",
        base_url="https://my-api.example.com/v1",
        kind="openai",
    )
    anthropic_url, openai_url = upstream_to_proxy_urls(upstream)
    assert anthropic_url is None
    assert openai_url == "https://my-api.example.com/v1"


# ---------------------------------------------------------------------------
# Wrap: upstream detection integration
# ---------------------------------------------------------------------------


def test_wrap_zcode_detects_upstream(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wrap zcode detects upstream and prints detected provider in setup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    config = tmp_path / "config.json"
    config.write_text(
        '{"provider": {"zai": {"name": "Z.ai Coding", "kind": "anthropic", '
        '"enabled": true, "options": {"baseURL": "https://api.z.ai/api/anthropic"}}}}',
        encoding="utf-8",
    )

    from headroom.providers.zcode.runtime import detect_upstream, upstream_to_proxy_urls

    upstream = detect_upstream(config)
    assert upstream.provider_name == "Z.ai Coding"
    assert upstream.base_url == "https://api.z.ai/api/anthropic"

    anthropic_url, openai_url = upstream_to_proxy_urls(upstream)
    assert anthropic_url == "https://api.z.ai/api/anthropic"
    assert openai_url is None


def test_wrap_zcode_passes_upstream_to_watcher(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wrap zcode forwards detected upstream URLs to _run_proxy_only_watcher."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    from headroom.providers.zcode.runtime import ZCodeUpstream

    captured: dict[str, object] = {}

    def fake_watcher(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    fake_upstream = ZCodeUpstream(
        provider_name="Z.ai", base_url="https://api.z.ai/api/anthropic", kind="anthropic"
    )

    with patch.object(wrap_mod, "_detect_zcode_upstream", return_value=fake_upstream):
        with patch.object(wrap_mod, "_run_proxy_only_watcher", side_effect=fake_watcher):
            runner.invoke(main, ["wrap", "zcode", "--port", "9000"])

    assert captured.get("anthropic_api_url") == "https://api.z.ai/api/anthropic"
    assert captured.get("openai_api_url") is None
    assert captured.get("port") == 9000


def test_wrap_zcode_passes_openai_upstream(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wrap zcode forwards OpenAI-compatible upstream URLs correctly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    from headroom.providers.zcode.runtime import ZCodeUpstream

    captured: dict[str, object] = {}

    def fake_watcher(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    fake_upstream = ZCodeUpstream(
        provider_name="Custom", base_url="https://my-api.example.com/v1", kind="openai"
    )

    with patch.object(wrap_mod, "_detect_zcode_upstream", return_value=fake_upstream):
        with patch.object(wrap_mod, "_run_proxy_only_watcher", side_effect=fake_watcher):
            runner.invoke(main, ["wrap", "zcode", "--port", "9000"])

    assert captured.get("anthropic_api_url") is None
    assert captured.get("openai_api_url") == "https://my-api.example.com/v1"
