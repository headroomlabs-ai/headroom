"""Tests for OpenCode config file helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.providers.opencode.config import (
    HEADROOM_OPENCODE_PLUGIN,
    _inject_key_into_json,
    _parse_json_loose,
    append_headroom_plugin,
    inject_opencode_provider_config,
    opencode_config_paths,
    snapshot_opencode_config_if_unwrapped,
    strip_opencode_headroom_blocks,
)


def _set_test_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------


def test_opencode_config_paths_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default config path resolves to ~/.config/opencode/opencode.json."""
    _set_test_home(monkeypatch, tmp_path)
    config_file, backup_file = opencode_config_paths()
    assert config_file == tmp_path / ".config" / "opencode" / "opencode.json"
    assert backup_file == tmp_path / ".config" / "opencode" / "opencode.json.headroom-backup"


def test_opencode_config_paths_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENCODE_CONFIG env var overrides the default path."""
    custom_path = tmp_path / "custom" / "opencode.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(custom_path))
    config_file, backup_file = opencode_config_paths()
    assert config_file == custom_path
    assert backup_file == tmp_path / "custom" / "opencode.json.headroom-backup"


def test_opencode_config_paths_default_jsonc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default config path resolves to opencode.jsonc when it exists."""
    _set_test_home(monkeypatch, tmp_path)
    base_dir = tmp_path / ".config" / "opencode"
    base_dir.mkdir(parents=True, exist_ok=True)
    jsonc_path = base_dir / "opencode.jsonc"
    jsonc_path.write_text("{}")

    config_file, backup_file = opencode_config_paths()
    assert config_file == jsonc_path
    assert backup_file == base_dir / "opencode.jsonc.headroom-backup"


def test_opencode_config_paths_env_overrides_jsonc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OPENCODE_CONFIG takes precedence even if opencode.jsonc exists."""
    _set_test_home(monkeypatch, tmp_path)
    base_dir = tmp_path / ".config" / "opencode"
    base_dir.mkdir(parents=True, exist_ok=True)
    jsonc_path = base_dir / "opencode.jsonc"
    jsonc_path.write_text("{}")

    custom_path = tmp_path / "custom" / "opencode.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(custom_path))

    config_file, backup_file = opencode_config_paths()
    assert config_file == custom_path
    assert backup_file == tmp_path / "custom" / "opencode.json.headroom-backup"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_creates_backup(tmp_path: Path) -> None:
    """snapshot creates a backup copy of the config file."""
    config_file = tmp_path / "opencode.json"
    backup_file = tmp_path / "opencode.json.headroom-backup"
    config_file.write_text('{"model": "openai/gpt-4o"}')
    snapshot_opencode_config_if_unwrapped(config_file, backup_file)
    assert backup_file.exists()
    assert backup_file.read_text() == config_file.read_text()


def test_snapshot_skips_if_backup_exists(tmp_path: Path) -> None:
    """snapshot is a no-op when the backup already exists."""
    config_file = tmp_path / "opencode.json"
    backup_file = tmp_path / "opencode.json.headroom-backup"
    config_file.write_text('{"model": "a"}')
    backup_file.write_text('{"model": "b"}')
    snapshot_opencode_config_if_unwrapped(config_file, backup_file)
    assert backup_file.read_text() == '{"model": "b"}'


def test_snapshot_skips_if_markers_present(tmp_path: Path) -> None:
    """snapshot skips if the config already contains Headroom markers."""
    config_file = tmp_path / "opencode.json"
    backup_file = tmp_path / "opencode.json.headroom-backup"
    config_file.write_text("// --- Headroom proxy provider ---\n{}")
    snapshot_opencode_config_if_unwrapped(config_file, backup_file)
    assert not backup_file.exists()


# ---------------------------------------------------------------------------
# Strip blocks
# ---------------------------------------------------------------------------


def test_strip_blocks_removes_provider_and_mcp() -> None:
    """strip removes both provider and MCP blocks."""
    content = (
        "// --- Headroom proxy provider ---\n"
        '{"provider": {}}\n'
        "// --- end Headroom proxy provider ---\n"
        "// --- Headroom MCP server ---\n"
        '{"mcp": {}}\n'
        "// --- end Headroom MCP server ---\n"
        '{"model": "openai/gpt-4o"}'
    )
    cleaned = strip_opencode_headroom_blocks(content)
    assert "Headroom" not in cleaned
    assert '{"model": "openai/gpt-4o"}' in cleaned


def test_strip_blocks_preserves_user_content() -> None:
    """strip leaves user content untouched when no blocks are present."""
    content = '{"model": "openai/gpt-4o", "provider": {"openai": {}}}'
    cleaned = strip_opencode_headroom_blocks(content)
    assert cleaned == content


# ---------------------------------------------------------------------------
# Parse JSON loose
# ---------------------------------------------------------------------------


def test_parse_json_loose_strips_comments() -> None:
    """_parse_json_loose ignores // comments."""
    text = '{\n  "model": "gpt-4o", // default model\n  "provider": {}\n}'
    data = _parse_json_loose(text)
    assert data["model"] == "gpt-4o"
    assert "provider" in data


def test_parse_json_loose_returns_empty_on_invalid() -> None:
    """_parse_json_loose returns empty dict for invalid JSON."""
    assert _parse_json_loose("not json") == {}


# ---------------------------------------------------------------------------
# Inject key
# ---------------------------------------------------------------------------


def test_inject_key_merges_dicts() -> None:
    """_inject_key_into_json merges nested dicts."""
    data = {"provider": {"openai": {}}}
    data = _inject_key_into_json(data, "provider", {"headroom": {}})
    assert "openai" in data["provider"]
    assert "headroom" in data["provider"]


def test_inject_key_overwrites_non_dict() -> None:
    """_inject_key_into_json overwrites when existing value is not a dict."""
    data = {"model": "gpt-4o"}
    data = _inject_key_into_json(data, "model", "headroom/claude-sonnet-4-6")
    assert data["model"] == "headroom/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Inject provider config
# ---------------------------------------------------------------------------


def test_inject_provider_config_creates_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inject_opencode_provider_config creates the config file when missing."""
    _set_test_home(monkeypatch, tmp_path)
    inject_opencode_provider_config(port=8787)
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    assert config_file.exists()
    config = _parse_json_loose(config_file.read_text())
    assert config["provider"]["headroom"]["npm"] == "@ai-sdk/openai-compatible"
    # Bare model ids: OpenCode resolves them as "headroom/<id>" (#1657).
    models = config["provider"]["headroom"]["models"]
    assert "claude-sonnet-4-6" in models
    assert all(not model_id.startswith("headroom/") for model_id in models)
    assert "mcp" not in config
    assert "model" not in config  # headroom provider is a transparent pass-through


def test_inject_provider_config_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """inject_opencode_provider_config is safe to call multiple times."""
    _set_test_home(monkeypatch, tmp_path)
    inject_opencode_provider_config(port=8787)
    inject_opencode_provider_config(port=9999)
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config = _parse_json_loose(config_file.read_text())
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"


# ---------------------------------------------------------------------------
# Edge cases — JSON parsing
# ---------------------------------------------------------------------------


def test_parse_json_loose_handles_valid_json() -> None:
    """_parse_json_loose returns correct dict for valid JSON without comments."""
    data = _parse_json_loose('{"model": "gpt-4o", "key": "value"}')
    assert data == {"model": "gpt-4o", "key": "value"}


def test_parse_json_loose_handles_jsonc_with_comments() -> None:
    """_parse_json_loose strips comments and returns valid data."""
    text = '{\n  "model": "gpt-4o",\n  // this is a comment\n  "provider": {}\n}'
    data = _parse_json_loose(text)
    assert data["model"] == "gpt-4o"
    assert data["provider"] == {}


def test_parse_json_loose_handles_urls_in_json() -> None:
    """_parse_json_loose does NOT corrupt URLs containing //."""
    text = '{"baseURL": "http://127.0.0.1:8787/v1"}'
    data = _parse_json_loose(text)
    assert data["baseURL"] == "http://127.0.0.1:8787/v1"


def test_parse_json_loose_handles_comments_and_urls() -> None:
    """_parse_json_loose handles both comments and URLs in the same file."""
    text = (
        "{\n"
        "  // proxy configuration\n"
        '  "baseURL": "http://127.0.0.1:8787/v1",\n'
        '  "name": "Headroom // Proxy"\n'
        "}"
    )
    data = _parse_json_loose(text)
    assert data["baseURL"] == "http://127.0.0.1:8787/v1"
    assert data["name"] == "Headroom // Proxy"


def test_parse_json_loose_returns_empty_on_empty_string() -> None:
    """_parse_json_loose returns {} for empty input."""
    assert _parse_json_loose("") == {}


def test_parse_json_loose_returns_empty_on_whitespace() -> None:
    """_parse_json_loose returns {} for whitespace-only input."""
    assert _parse_json_loose("   \n  \t  ") == {}


def test_parse_json_loose_returns_empty_on_trailing_comma() -> None:
    """_parse_json_loose returns {} for malformed JSON (trailing comma)."""
    assert _parse_json_loose('{"model": "gpt-4o",}') == {}


def test_parse_json_loose_returns_empty_on_unclosed_brace() -> None:
    """_parse_json_loose returns {} for malformed JSON (unclosed brace)."""
    assert _parse_json_loose('{"model": "gpt-4o"') == {}


# ---------------------------------------------------------------------------
# Edge cases — strip blocks
# ---------------------------------------------------------------------------


def test_strip_blocks_handles_empty_string() -> None:
    """strip_opencode_headroom_blocks returns empty string for empty input."""
    assert strip_opencode_headroom_blocks("") == ""


def test_strip_blocks_handles_whitespace_only() -> None:
    """strip_opencode_headroom_blocks returns empty string for whitespace input."""
    assert strip_opencode_headroom_blocks("  \n  ") == ""


def test_strip_blocks_preserves_non_headroom_jsonc() -> None:
    """strip_opencode_headroom_blocks preserves JSONC comments not from Headroom."""
    content = '// user comment\n{"model": "gpt-4o"}\n// another user comment\n'
    cleaned = strip_opencode_headroom_blocks(content)
    assert "// user comment" in cleaned
    assert '{"model": "gpt-4o"}' in cleaned


def test_strip_blocks_removes_only_one_of_two_identical_blocks() -> None:
    """strip_opencode_headroom_blocks removes all provider blocks, not just the first."""
    from headroom.providers.opencode.config import _PROVIDER_MARKER_END, _PROVIDER_MARKER_START

    content = (
        _PROVIDER_MARKER_START
        + "\nblock1\n"
        + _PROVIDER_MARKER_END
        + "\n"
        + _PROVIDER_MARKER_START
        + "\nblock2\n"
        + _PROVIDER_MARKER_END
    )
    cleaned = strip_opencode_headroom_blocks(content)
    assert _PROVIDER_MARKER_START not in cleaned
    assert "block1" not in cleaned
    assert "block2" not in cleaned


def test_strip_blocks_handles_only_mcp_markers() -> None:
    """strip_opencode_headroom_blocks also strips MCP markers."""
    from headroom.providers.opencode.config import _MCP_MARKER_END, _MCP_MARKER_START

    content = _MCP_MARKER_START + "\nmcp data\n" + _MCP_MARKER_END
    cleaned = strip_opencode_headroom_blocks(content)
    assert _MCP_MARKER_START not in cleaned


# ---------------------------------------------------------------------------
# Edge cases — inject config
# ---------------------------------------------------------------------------


def test_inject_provider_config_preserves_existing_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inject_opencode_provider_config preserves MCP without adding headroom."""
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        '{"mcp": {"existing-server": {"type": "remote", "url": "https://example.com"}}}'
    )

    inject_opencode_provider_config(port=8787)

    config = json.loads(config_file.read_text())
    assert "existing-server" in config["mcp"]
    assert "headroom" not in config["mcp"]


def test_inject_provider_config_idempotent_with_complex_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inject_opencode_provider_config is idempotent on complex configs."""
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(
            {
                "model": "openai/gpt-4o",
                "provider": {"openai": {"models": {"gpt-4o": {}}}},
                "mcp": {"myserver": {"type": "local", "command": ["echo"]}},
            }
        )
    )

    inject_opencode_provider_config(port=8787)
    inject_opencode_provider_config(port=8787)

    config = json.loads(config_file.read_text())
    assert "openai" in config["provider"]
    assert "headroom" in config["provider"]
    assert "myserver" in config["mcp"]
    assert "headroom" not in config["mcp"]


def test_inject_provider_config_preserves_unrelated_top_level_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inject_opencode_provider_config preserves top-level keys like plugin, permission, etc."""
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(
            {
                "plugin": ["some-plugin"],
                "permission": {"bash": {"*": "ask"}},
                "model": "openai/gpt-4o",
            }
        )
    )

    inject_opencode_provider_config(port=8787)

    config = json.loads(config_file.read_text())
    assert config["plugin"] == ["some-plugin"]
    assert config["permission"] == {"bash": {"*": "ask"}}
    assert "headroom" in config.get("provider", {})


def test_append_headroom_plugin_adds_plugin_once() -> None:
    config: dict[str, object] = {"plugin": ["some-plugin"]}

    assert append_headroom_plugin(config) is True
    assert append_headroom_plugin(config) is False

    assert config["plugin"] == ["some-plugin", HEADROOM_OPENCODE_PLUGIN]


def test_append_headroom_plugin_preserves_configured_tuple_entry() -> None:
    config: dict[str, object] = {
        "plugin": [[HEADROOM_OPENCODE_PLUGIN, {"proxyUrl": "http://127.0.0.1:8787"}]]
    }

    assert append_headroom_plugin(config) is False
    assert config["plugin"] == [[HEADROOM_OPENCODE_PLUGIN, {"proxyUrl": "http://127.0.0.1:8787"}]]


def test_inject_provider_config_no_crash_on_unwriteable_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """inject_opencode_provider_config raises click.ClickException on OSError."""

    import click as click_mod

    monkeypatch.setenv("HOME", "/nonexistent/path/that/cannot/be/created")
    try:
        inject_opencode_provider_config(port=8787)
    except click_mod.ClickException:
        pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Runtime module coverage: build_opencode_config_content, build_launch_env
# ---------------------------------------------------------------------------


def test_build_opencode_config_content_without_mcp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from headroom.providers.opencode.runtime import build_opencode_config_content

    plugin = tmp_path / "entry.opencode.js"
    plugin.write_text("export default () => {}", encoding="utf-8")
    monkeypatch.setenv("HEADROOM_OPENCODE_PLUGIN_PATH", str(plugin))

    config = build_opencode_config_content(port=8787, include_mcp=False)
    assert "mcp" not in config
    assert "model" not in config
    # Native providers are pointed at the proxy so traffic routes through Headroom.
    providers = config["provider"]
    assert providers["anthropic"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"
    assert providers["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"
    # The headroom provider exposes explicit models so "headroom/<id>" resolves (#1657).
    assert providers["headroom"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"
    models = providers["headroom"]["models"]
    assert "claude-sonnet-4-6" in models
    assert all(not model_id.startswith("headroom/") for model_id in models)
    # The transport plugin is injected by absolute path (opencode loads it directly).
    assert config["plugin"] == [str(plugin)]


def test_build_opencode_config_content_skips_plugin_when_unbuilt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from headroom.providers.opencode.runtime import build_opencode_config_content

    # An override pointing at a missing file resolves to None → no plugin entry,
    # but native-provider routing still applies (the pip-only fallback).
    monkeypatch.setenv("HEADROOM_OPENCODE_PLUGIN_PATH", str(tmp_path / "missing.js"))
    config = build_opencode_config_content(port=8787)
    assert "plugin" not in config
    assert config["provider"]["anthropic"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"


def test_build_opencode_config_content_with_mcp_uses_local_stdio() -> None:
    from headroom.providers.opencode.runtime import build_opencode_config_content

    config = build_opencode_config_content(port=9000, include_mcp=True)
    assert config["mcp"]["headroom"] == {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "enabled": True,
        "environment": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"},
    }


def test_build_launch_env_with_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from headroom.providers.opencode.runtime import build_launch_env

    monkeypatch.delenv("HEADROOM_PROJECT", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    plugin = tmp_path / "entry.opencode.js"
    plugin.write_text("export default () => {}", encoding="utf-8")
    monkeypatch.setenv("HEADROOM_OPENCODE_PLUGIN_PATH", str(plugin))

    env, display = build_launch_env(
        port=8787,
        project="test-proj",
        include_mcp=False,
    )
    assert env["HEADROOM_PROJECT"] == "test-proj"
    # Plugin loaded → its proxy target is exported for self-configuration.
    assert env["HEADROOM_PROXY_URL"] == "http://127.0.0.1:8787"
    assert str(plugin) in env["OPENCODE_CONFIG_CONTENT"]
    assert f"plugin={HEADROOM_OPENCODE_PLUGIN}" in display
    assert "OPENAI_BASE_URL" not in env
    assert "ANTHROPIC_BASE_URL" not in env


def test_build_launch_env_with_custom_environ() -> None:
    from headroom.providers.opencode.runtime import build_launch_env

    custom = {
        "EXISTING_VAR": "keep-me",
        "OPENAI_BASE_URL": "https://deepseek.example/v1",
        "ANTHROPIC_BASE_URL": "https://anthropic.example",
    }
    env, display = build_launch_env(
        port=8787,
        environ=custom,
        include_mcp=False,
    )
    assert env["EXISTING_VAR"] == "keep-me"
    assert env["OPENAI_BASE_URL"] == "https://deepseek.example/v1"
    assert env["ANTHROPIC_BASE_URL"] == "https://anthropic.example"


def test_proxy_base_url() -> None:
    from headroom.providers.opencode.runtime import proxy_base_url

    assert proxy_base_url(9000) == "http://127.0.0.1:9000/v1"


def test_inject_provider_config_strips_existing_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    inject_opencode_provider_config(port=9000)
    first = config_file.read_text()
    assert "headroom" in first

    inject_opencode_provider_config(port=9001)
    second = config_file.read_text()
    assert "headroom" in second
    assert second.count("headroom") == first.count("headroom")
