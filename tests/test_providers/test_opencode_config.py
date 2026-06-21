"""Tests for OpenCode config helpers: backup/restore, marker injection, strip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.providers.opencode.config import (
    _MCP_MARKER_END,
    _MCP_MARKER_START,
    _PROVIDER_MARKER_END,
    _PROVIDER_MARKER_START,
    _inject_key_into_json,
    _opencode_config_path,
    _parse_json_loose,
    snapshot_opencode_config_if_unwrapped,
    strip_opencode_headroom_blocks,
)


# ============================================================================
# JSONC loose parsing
# ============================================================================


class TestParseJsonLoose:
    def test_valid_json(self):
        data = _parse_json_loose('{"key": "value"}')
        assert data == {"key": "value"}

    def test_json_with_comments(self):
        text = '// comment\n{"key": "value"}'
        data = _parse_json_loose(text)
        assert data == {"key": "value"}

    def test_json_with_trailing_comma_comment(self):
        text = '{"a": 1, // inline\n"b": 2}'
        data = _parse_json_loose(text)
        assert data == {"a": 1, "b": 2}

    def test_empty_string(self):
        assert _parse_json_loose("") == {}

    def test_malformed_json(self):
        assert _parse_json_loose("not json at all") == {}

    def test_malformed_json_with_comments(self):
        assert _parse_json_loose("// comment\n{not json}") == {}


# ============================================================================
# Key injection
# ============================================================================


class TestInjectKeyIntoJson:
    def test_adds_new_key(self):
        data: dict = {}
        result = _inject_key_into_json(data, "provider", {"mimo": {}})
        assert result["provider"] == {"mimo": {}}

    def test_merges_existing_dict(self):
        data = {"provider": {"mimo": {"npm": "@ai-sdk/openai-compatible"}}}
        result = _inject_key_into_json(
            data, "provider", {"deepseek": {"npm": "@ai-sdk/openai-compatible"}}
        )
        assert "mimo" in result["provider"]
        assert "deepseek" in result["provider"]

    def test_overwrites_non_dict_with_dict(self):
        data = {"provider": "old_value"}
        result = _inject_key_into_json(data, "provider", {"new": {}})
        assert result["provider"] == {"new": {}}

    def test_preserves_other_keys(self):
        data = {"model": "claude", "autoupdate": True}
        result = _inject_key_into_json(data, "provider", {"mimo": {}})
        assert result["model"] == "claude"
        assert result["autoupdate"] is True


# ============================================================================
# Marker stripping
# ============================================================================


class TestStripHeadroomBlocks:
    def test_strips_provider_block(self):
        content = (
            '{"model": "test"}\n'
            f'{_PROVIDER_MARKER_START}\n'
            '  "provider": {"headroom": {"baseURL": "http://127.0.0.1:8787/v1"}}\n'
            f'{_PROVIDER_MARKER_END}\n'
            '{"other": "data"}\n'
        )
        result = strip_opencode_headroom_blocks(content)
        assert "headroom" not in result
        assert "other" in result

    def test_strips_mcp_block(self):
        content = (
            f'{_MCP_MARKER_START}\n'
            '  "mcp": {"headroom": {"type": "remote", "url": "http://127.0.0.1:8787/mcp"}}\n'
            f'{_MCP_MARKER_END}\n'
            '{"other": "data"}\n'
        )
        result = strip_opencode_headroom_blocks(content, remove_mcp=True)
        assert "headroom" not in result
        assert "other" in result

    def test_preserves_mcp_when_remove_mcp_false(self):
        content = (
            f'{_MCP_MARKER_START}\n'
            '  "mcp": {"headroom": {"type": "remote"}}\n'
            f'{_MCP_MARKER_END}\n'
            '{"other": "data"}\n'
        )
        result = strip_opencode_headroom_blocks(content, remove_mcp=False)
        assert "headroom" in result

    def test_no_markers_returns_original(self):
        content = '{"model": "test"}\n'
        result = strip_opencode_headroom_blocks(content)
        assert '"model"' in result

    def test_empty_content(self):
        result = strip_opencode_headroom_blocks("")
        assert result == ""

    def test_collapse_extra_blank_lines(self):
        content = (
            f'{_PROVIDER_MARKER_START}\n'
            f'{_PROVIDER_MARKER_END}\n'
            '\n\n\n'
            '{"key": "value"}\n'
        )
        result = strip_opencode_headroom_blocks(content)
        assert "\n\n\n" not in result


# ============================================================================
# Snapshot / backup
# ============================================================================


class TestSnapshotBackup:
    def test_creates_backup(self, tmp_path: Path):
        config = tmp_path / "opencode.json"
        config.write_text('{"model": "test"}')
        backup = tmp_path / "opencode.json.headroom-backup"

        snapshot_opencode_config_if_unwrapped(config, backup)
        assert backup.exists()
        assert backup.read_text() == '{"model": "test"}'

    def test_skips_if_backup_exists(self, tmp_path: Path):
        config = tmp_path / "opencode.json"
        config.write_text('{"new": true}')
        backup = tmp_path / "opencode.json.headroom-backup"
        backup.write_text('{"old": true}')

        snapshot_opencode_config_if_unwrapped(config, backup)
        assert backup.read_text() == '{"old": true}'

    def test_skips_if_config_missing(self, tmp_path: Path):
        config = tmp_path / "nope.json"
        backup = tmp_path / "nope.json.headroom-backup"

        snapshot_opencode_config_if_unwrapped(config, backup)
        assert not backup.exists()

    def test_skips_if_headroom_marker_present(self, tmp_path: Path):
        config = tmp_path / "opencode.json"
        config.write_text(
            f'{{"model": "test"}}\n{_PROVIDER_MARKER_START}\nheadroom stuff\n{_PROVIDER_MARKER_END}\n'
        )
        backup = tmp_path / "opencode.json.headroom-backup"

        snapshot_opencode_config_if_unwrapped(config, backup)
        assert not backup.exists()


# ============================================================================
# Config path
# ============================================================================


class TestConfigPath:
    def test_default_path(self):
        path = _opencode_config_path()
        assert path.name == "opencode.json"
        assert ".config" in str(path) or "opencode" in str(path)

    def test_respects_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENCODE_CONFIG", "/tmp/custom-opencode.json")
        path = _opencode_config_path()
        assert str(path) == "/tmp/custom-opencode.json"
