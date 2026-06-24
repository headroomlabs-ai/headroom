from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(relative_path: str) -> object:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_marketplace_manifests_match() -> None:
    assert _load_json(".claude-plugin/marketplace.json") == _load_json(
        ".github/plugin/marketplace.json"
    )


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
