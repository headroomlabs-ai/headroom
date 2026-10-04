from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(relative_path: str) -> object:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


# Plugins built on Claude Code's function hooks, which Copilot CLI cannot load.
CLAUDE_CODE_ONLY = {"headroom-snip"}


def test_marketplace_manifests_match() -> None:
    claude = _load_json(".claude-plugin/marketplace.json")
    assert isinstance(claude, dict)
    shared = {
        **claude,
        "plugins": [p for p in claude["plugins"] if p["name"] not in CLAUDE_CODE_ONLY],
    }
    assert shared == _load_json(".github/plugin/marketplace.json")


def test_claude_code_only_plugins_are_listed_and_versioned() -> None:
    marketplace = _load_json(".claude-plugin/marketplace.json")
    assert isinstance(marketplace, dict)
    for entry in marketplace["plugins"]:
        if entry["name"] not in CLAUDE_CODE_ONLY:
            continue
        plugin_root = (REPO_ROOT / entry["source"]).resolve()
        manifest = _load_json(f"{entry['source']}/.claude-plugin/plugin.json")
        assert isinstance(manifest, dict)
        assert manifest["name"] == entry["name"]
        assert manifest["version"] == entry["version"] == marketplace["metadata"]["version"]
        assert (plugin_root / "hooks" / "hooks.json").is_file()


def test_plugin_manifests_share_core_metadata() -> None:
    claude = _load_json("plugins/headroom-agent-hooks/.claude-plugin/plugin.json")
    copilot = _load_json("plugins/headroom-agent-hooks/.github/plugin/plugin.json")
    assert isinstance(claude, dict)
    assert isinstance(copilot, dict)
    for key in ("name", "version", "description", "author", "homepage", "repository", "keywords"):
        assert claude[key] == copilot[key]
    assert "hooks" not in claude
    assert copilot["hooks"] == "./hooks"


def test_marketplace_entry_points_to_plugin_root() -> None:
    marketplace = _load_json(".claude-plugin/marketplace.json")
    assert isinstance(marketplace, dict)
    plugins = marketplace["plugins"]
    assert isinstance(plugins, list)
    plugin = plugins[0]
    assert plugin["name"] == "headroom"
    plugin_root = (REPO_ROOT / plugin["source"]).resolve()
    assert plugin_root.is_dir()
    assert (plugin_root / ".claude-plugin" / "plugin.json").is_file()
    assert (plugin_root / "hooks" / "hooks.json").is_file()


def test_plugin_metadata_points_to_upstream_repo() -> None:
    expected_repo = "https://github.com/chopratejas/headroom"
    marketplace = _load_json(".claude-plugin/marketplace.json")
    claude = _load_json("plugins/headroom-agent-hooks/.claude-plugin/plugin.json")
    assert isinstance(marketplace, dict)
    assert isinstance(claude, dict)
    plugin = marketplace["plugins"][0]
    assert plugin["author"]["url"] == expected_repo
    assert plugin["homepage"] == expected_repo
    assert plugin["repository"] == expected_repo
    assert claude["author"]["url"] == expected_repo
    assert claude["homepage"] == expected_repo
    assert claude["repository"] == expected_repo
