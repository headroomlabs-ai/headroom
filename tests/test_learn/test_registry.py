"""Tests for the learn plugin registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from headroom.learn.base import LearnPlugin
from headroom.learn.registry import (
    auto_detect_plugins,
    available_agent_names,
    get_plugin,
    get_registry,
    reset_registry,
)
from headroom.learn.scanner import ConversationScanner


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before/after each test."""
    reset_registry()
    yield
    reset_registry()


class TestBuiltinDiscovery:
    def test_discovers_three_builtin_plugins(self):
        reg = get_registry()
        assert "claude" in reg
        assert "codex" in reg
        assert "gemini" in reg
        assert len(reg) >= 3

    def test_all_plugins_are_learn_plugins(self):
        for name, plugin in get_registry().items():
            assert isinstance(plugin, LearnPlugin), f"{name} is not a LearnPlugin"

    def test_all_plugins_are_conversation_scanners(self):
        """Backwards compat: plugins must also be ConversationScanners."""
        for name, plugin in get_registry().items():
            assert isinstance(plugin, ConversationScanner), f"{name} is not a ConversationScanner"

    def test_plugins_have_identity(self):
        for name, plugin in get_registry().items():
            assert plugin.name == name
            assert plugin.display_name  # non-empty
            assert plugin.description  # non-empty


class TestGetPlugin:
    def test_get_existing_plugin(self):
        plugin = get_plugin("claude")
        assert plugin.name == "claude"
        assert plugin.display_name == "Claude Code"

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown agent.*cursor"):
            get_plugin("cursor")

    def test_error_message_lists_available(self):
        with pytest.raises(KeyError, match="claude"):
            get_plugin("nonexistent")


class TestAutoDetect:
    def test_filters_to_detected_only(self):
        """Only plugins where detect() returns True are included."""
        detected = auto_detect_plugins()
        for plugin in detected:
            assert plugin.detect()

    def test_returns_empty_when_nothing_detected(self):
        """All plugins returning False → empty list."""
        with (
            patch.object(get_registry()["claude"], "detect", return_value=False),
            patch.object(get_registry()["codex"], "detect", return_value=False),
            patch.object(get_registry()["gemini"], "detect", return_value=False),
        ):
            assert auto_detect_plugins() == []


class TestAvailableNames:
    def test_returns_sorted_list(self):
        names = available_agent_names()
        assert names == sorted(names)
        assert "claude" in names
        assert "codex" in names
        assert "gemini" in names


class TestExternalPlugin:
    def test_external_plugin_via_entry_point(self):
        """Mock an external plugin registered via entry_points."""
        mock_plugin = MagicMock(spec=LearnPlugin)
        mock_plugin.name = "cursor"
        mock_plugin.display_name = "Cursor"
        mock_plugin.description = "Cursor IDE (~/.cursor/)"

        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_plugin
        mock_ep.name = "cursor"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            reset_registry()
            reg = get_registry()
            assert "cursor" in reg
            assert reg["cursor"].name == "cursor"


class TestResetRegistry:
    def test_reset_clears_cache(self):
        reg1 = get_registry()
        assert reg1 is get_registry()  # Same object (cached)
        reset_registry()
        reg2 = get_registry()
        assert reg2 is not reg1  # New object (cache cleared)


class TestPluginCreateWriter:
    def test_all_plugins_create_valid_writers(self):
        from headroom.learn.writer import ContextWriter

        for name, plugin in get_registry().items():
            writer = plugin.create_writer()
            assert isinstance(writer, ContextWriter), f"{name} writer is not a ContextWriter"
