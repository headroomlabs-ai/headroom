"""Independently versioned local plugins must survive a repository release."""

import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def marketplace(tmp_path):
    manifest = tmp_path / "integrations/sidebar/.claude-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "headroom-sidebar", "version": "0.1.1"}))
    path = tmp_path / ".claude-plugin/marketplace.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "metadata": {"version": "0.40.0"},
                "plugins": [
                    {
                        "name": "headroom",
                        "source": "./plugins/headroom-agent-hooks",
                        "version": "0.40.0",
                    },
                    {
                        "name": "headroom-sidebar",
                        "source": "./integrations/sidebar",
                        "version": "0.1.1",
                    },
                ],
            }
        )
    )
    return path, manifest


def test_release_sync_preserves_independent_plugin_version(tmp_path):
    path, manifest = marketplace(tmp_path)
    load("version-sync").update_marketplace_manifest(path, "0.40.1")
    data = json.loads(path.read_text())
    assert data["metadata"]["version"] == "0.40.1"
    assert data["plugins"][0]["version"] == "0.40.1"
    assert data["plugins"][1]["version"] == json.loads(manifest.read_text())["version"]


def test_verifier_accepts_matching_independent_manifest(tmp_path, monkeypatch):
    path, _ = marketplace(tmp_path)
    module = load("verify-versions")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert set(module._read_marketplace_versions(path).values()) == {"0.40.0"}


def test_release_sync_keeps_snip_manifest_and_marketplace_aligned(tmp_path, monkeypatch):
    path, sidebar_manifest = marketplace(tmp_path)
    snip_manifest = tmp_path / "plugins/headroom-snip/.claude-plugin/plugin.json"
    snip_manifest.parent.mkdir(parents=True)
    snip_manifest.write_text(json.dumps({"name": "headroom-snip", "version": "0.40.0"}))
    data = json.loads(path.read_text())
    data["plugins"].append(
        {
            "name": "headroom-snip",
            "source": "./plugins/headroom-snip",
            "version": "0.40.0",
        }
    )
    path.write_text(json.dumps(data))

    sync = load("version-sync")
    sync.update_plugin_manifest(snip_manifest, "0.40.1")
    sync.update_marketplace_manifest(path, "0.40.1")
    verifier = load("verify-versions")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)

    assert set(verifier._read_marketplace_versions(path).values()) == {"0.40.1"}
    assert json.loads(sidebar_manifest.read_text())["version"] == "0.1.1"


def test_verifier_rejects_stale_independent_manifest(tmp_path, monkeypatch):
    path, manifest = marketplace(tmp_path)
    manifest.write_text(json.dumps({"name": "headroom-sidebar", "version": "0.1.2"}))
    module = load("verify-versions")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="headroom-sidebar"):
        module._read_marketplace_versions(path)


def test_verifier_rejects_source_outside_repository(tmp_path, monkeypatch):
    path, _ = marketplace(tmp_path)
    data = json.loads(path.read_text())
    data["plugins"][1]["source"] = "../../elsewhere"
    path.write_text(json.dumps(data))
    module = load("verify-versions")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="source"):
        module._read_marketplace_versions(path)
