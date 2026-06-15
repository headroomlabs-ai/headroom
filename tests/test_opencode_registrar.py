"""Tests for OpenCode MCP registrar."""

from __future__ import annotations

import json

from headroom.mcp_registry.base import ServerSpec
from headroom.mcp_registry.opencode import (
    OpenCodeRegistrar,
    _parse_jsonc,
    _strip_jsonc_comments,
)


class TestStripJsoncComments:
    def test_single_line_comment(self):
        assert _strip_jsonc_comments('{// comment\n"a": 1}') == '{\n"a": 1}'

    def test_block_comment(self):
        assert _strip_jsonc_comments('{/* comment */"a": 1}') == '{"a": 1}'

    def test_preserves_strings(self):
        text = '{"url": "http://example.com//not-a-comment"}'
        assert _strip_jsonc_comments(text) == text

    def test_preserves_strings_with_colon(self):
        text = '{"url": "http://example.com"}'
        assert _strip_jsonc_comments(text) == text

    def test_mixed_comments(self):
        text = '{\n  // line comment\n  "a": 1, /* block */ "b": 2\n}'
        result = _strip_jsonc_comments(text)
        assert "//" not in result
        assert "/*" not in result
        assert '"a": 1' in result
        assert '"b": 2' in result


class TestParseJsonc:
    def test_empty(self):
        assert _parse_jsonc("") == {}

    def test_valid_json(self):
        assert _parse_jsonc('{"a": 1}') == {"a": 1}

    def test_jsonc_with_comments(self):
        text = '{\n  // comment\n  "a": 1\n}'
        assert _parse_jsonc(text) == {"a": 1}

    def test_invalid_json(self):
        assert _parse_jsonc("not json") == {}


class TestOpenCodeRegistrar:
    def test_detect(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        assert registrar.detect() is True

    def test_detect_not_installed(self, tmp_path):
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        assert registrar.detect() is False

    def test_get_server_empty(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        assert registrar.get_server("headroom") is None

    def test_get_server(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.jsonc"
        config_file.write_text(json.dumps({
            "mcp": {
                "headroom": {
                    "type": "local",
                    "command": ["headroom", "mcp", "serve"],
                    "enabled": True,
                }
            }
        }))
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        spec = registrar.get_server("headroom")
        assert spec is not None
        assert spec.name == "headroom"
        assert spec.command == "headroom"
        assert spec.args == ("mcp", "serve")

    def test_register_server(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        result = registrar.register_server(spec)
        assert result.ok is True

        # Verify it was written
        spec2 = registrar.get_server("headroom")
        assert spec2 is not None
        assert spec2.command == "headroom"

    def test_unregister_server(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        registrar.register_server(spec)
        assert registrar.unregister_server("headroom") is True
        assert registrar.get_server("headroom") is None

    def test_add_provider(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        registrar.add_provider("headroom", base_url="http://127.0.0.1:8787/v1")
        config = registrar._read_config()
        assert "headroom" in config.get("provider", {})
        assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"

    def test_remove_provider(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        registrar.add_provider("headroom", base_url="http://127.0.0.1:8787/v1")
        registrar.remove_provider("headroom")
        config = registrar._read_config()
        assert "headroom" not in config.get("provider", {})

    def test_override_provider_base_url(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        registrar.override_provider_base_url("deepseek", "http://127.0.0.1:8787/v1")
        config = registrar._read_config()
        assert config["provider"]["deepseek"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"

    def test_is_wrapped_with_override(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        assert registrar.is_wrapped() is False
        registrar.override_provider_base_url("deepseek", "http://127.0.0.1:8787/v1")
        assert registrar.is_wrapped() is True

    def test_add_instruction(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        registrar.add_instruction("~/.config/opencode/instructions/rtk.md")
        config = registrar._read_config()
        assert "~/.config/opencode/instructions/rtk.md" in config.get("instructions", [])

    def test_remove_instruction(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        registrar.add_instruction("~/.config/opencode/instructions/rtk.md")
        registrar.remove_instruction("~/.config/opencode/instructions/rtk.md")
        config = registrar._read_config()
        assert "~/.config/opencode/instructions/rtk.md" not in config.get("instructions", [])

    def test_is_wrapped(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        assert registrar.is_wrapped() is False
        registrar.add_provider("headroom", base_url="http://127.0.0.1:8787/v1")
        assert registrar.is_wrapped() is True

    def test_snapshot_and_restore(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.jsonc"
        original = {"provider": {"openai": {}}}
        config_file.write_text(json.dumps(original))

        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        registrar.snapshot_config()

        # Modify config
        registrar.add_provider("headroom", base_url="http://127.0.0.1:8787/v1")
        assert registrar.is_wrapped() is True

        # Restore
        assert registrar.restore_config() is True
        config = registrar._read_config()
        assert "headroom" not in config.get("provider", {})
        assert "openai" in config.get("provider", {})

    def test_cleanup_after_unwrap(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.jsonc"
        original = {"provider": {"openai": {}}}
        config_file.write_text(json.dumps(original))

        registrar = OpenCodeRegistrar(home_dir=tmp_path)
        registrar.snapshot_config()
        registrar.add_provider("headroom", base_url="http://127.0.0.1:8787/v1")

        # Simulate unwrap
        restored = registrar.restore_config()
        assert restored is True

        # Verify backup is removed
        backup = config_file.with_suffix(".jsonc.headroom-backup")
        assert backup.exists() is False
