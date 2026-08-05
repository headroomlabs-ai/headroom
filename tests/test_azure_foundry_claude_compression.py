"""Azure AI Foundry + Claude Code compression wiring.

Covers the gap fixed by feat(azure-foundry): when only ANTHROPIC_FOUNDRY_RESOURCE
is set (no explicit ANTHROPIC_FOUNDRY_BASE_URL), wrap claude must derive the
upstream URL and route Claude Code's Foundry requests through the proxy.

No real Azure endpoint is contacted — helpers are unit-tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.cli import wrap as wrap_cli
from headroom.providers.claude import proxy_base_url as _claude_proxy_base_url
from headroom.providers.registry import resolve_api_overrides

# --------------------------------------------------------------------------
# Upstream URL derivation from ANTHROPIC_FOUNDRY_RESOURCE
# --------------------------------------------------------------------------


def test_foundry_upstream_url_builds_services_endpoint() -> None:
    assert (
        wrap_cli._foundry_upstream_url("my-org-claude")
        == "https://my-org-claude.services.ai.azure.com/anthropic"
    )


def test_foundry_upstream_url_strips_whitespace() -> None:
    assert (
        wrap_cli._foundry_upstream_url("  my-resource  ")
        == "https://my-resource.services.ai.azure.com/anthropic"
    )


def test_foundry_upstream_url_preserves_hyphens_and_digits() -> None:
    assert (
        wrap_cli._foundry_upstream_url("avanade-claude-42")
        == "https://avanade-claude-42.services.ai.azure.com/anthropic"
    )


# --------------------------------------------------------------------------
# Local proxy URL for Foundry mode includes /anthropic path component
# --------------------------------------------------------------------------


def test_foundry_proxy_url_appends_anthropic_path() -> None:
    proxy_url = _claude_proxy_base_url(8787)  # http://127.0.0.1:8787
    assert wrap_cli._foundry_proxy_url(proxy_url) == "http://127.0.0.1:8787/anthropic"


def test_foundry_proxy_url_strips_trailing_slash() -> None:
    assert (
        wrap_cli._foundry_proxy_url("http://127.0.0.1:8787/") == "http://127.0.0.1:8787/anthropic"
    )


# --------------------------------------------------------------------------
# resolve_api_overrides picks up ANTHROPIC_FOUNDRY_BASE_URL as anthropic target
# --------------------------------------------------------------------------


def test_resolve_api_overrides_uses_foundry_base_url_as_anthropic_target() -> None:
    overrides = resolve_api_overrides(
        anthropic_api_url=None,
        openai_api_url=None,
        gemini_api_url=None,
        cloudcode_api_url=None,
        environ={
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://my-resource.services.ai.azure.com/anthropic"
        },
    )
    assert overrides.anthropic == "https://my-resource.services.ai.azure.com/anthropic"


def test_resolve_api_overrides_explicit_target_beats_foundry_base_url() -> None:
    # ANTHROPIC_TARGET_API_URL takes precedence; FOUNDRY_BASE_URL is the fallback.
    overrides = resolve_api_overrides(
        anthropic_api_url=None,
        openai_api_url=None,
        gemini_api_url=None,
        cloudcode_api_url=None,
        environ={
            "ANTHROPIC_TARGET_API_URL": "https://explicit-override.example.com",
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://my-resource.services.ai.azure.com/anthropic",
        },
    )
    assert overrides.anthropic == "https://explicit-override.example.com"


# --------------------------------------------------------------------------
# settings.json written with ANTHROPIC_FOUNDRY_BASE_URL in Foundry mode
# Uses _claude_proxy_base_url + _foundry_proxy_url to cover the real wrap path
# --------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def test_write_foundry_mode_sets_foundry_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    # Mirror the real production path: derive proxy_url then apply _foundry_proxy_url
    proxy_url = _claude_proxy_base_url(8787)
    foundry_url = wrap_cli._foundry_proxy_url(proxy_url)
    wrap_cli._write_claude_wrap_base_url(foundry_url, foundry_mode=True, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_FOUNDRY_BASE_URL"] == "http://127.0.0.1:8787/anthropic"
    assert "ANTHROPIC_BASE_URL" not in payload["env"]


def test_write_non_foundry_mode_does_not_set_foundry_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    proxy_url = _claude_proxy_base_url(8787)
    wrap_cli._write_claude_wrap_base_url(proxy_url, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert "ANTHROPIC_FOUNDRY_BASE_URL" not in payload["env"]


def test_restore_foundry_mode_removes_foundry_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_FOUNDRY_BASE_URL": "http://127.0.0.1:8787/anthropic"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, foundry_mode=True, settings_path=path)
    # The restore may delete the file entirely when the env dict becomes empty,
    # or leave a file with the key absent — both indicate correct removal.
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "ANTHROPIC_FOUNDRY_BASE_URL" not in payload.get("env", {})
    # else: file deleted — key is gone, which is also correct
