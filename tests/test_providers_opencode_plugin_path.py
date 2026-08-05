"""Resolution of the OpenCode transport-plugin path across install layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

import headroom.providers.opencode.runtime as oc_runtime
from headroom.providers.opencode.runtime import headroom_opencode_plugin_path

_PACKAGED = Path(oc_runtime.__file__).resolve().parent / "_dist" / "entry.opencode.js"
_REPO_BUILD = (
    Path(oc_runtime.__file__).resolve().parents[3]
    / "plugins"
    / "opencode"
    / "dist"
    / "entry.opencode.js"
)


def test_packaged_bundle_is_committed_and_self_contained() -> None:
    # The wheel picks this file up via maturin's python-source packaging; if
    # it goes missing, pip installs silently lose all-provider routing again.
    assert _PACKAGED.is_file(), "committed wheel bundle missing - run npm run build:standalone"
    text = _PACKAGED.read_text(encoding="utf-8")
    assert len(text) > 10_000, "bundle suspiciously small - not the standalone build?"
    # Self-contained: no bare npm imports; node builtins are the only imports
    # allowed (site-packages has no node_modules to resolve anything else).
    assert 'from "headroom-ai"' not in text
    assert 'from "@opencode-ai/plugin"' not in text


def test_plugin_path_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "custom.js"
    override.write_text("// plugin")
    monkeypatch.setenv("HEADROOM_OPENCODE_PLUGIN_PATH", str(override))
    assert headroom_opencode_plugin_path() == str(override)


def test_plugin_path_falls_back_to_packaged_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_OPENCODE_PLUGIN_PATH", raising=False)
    resolved = headroom_opencode_plugin_path()
    assert resolved is not None
    # In a repo checkout with a built plugins/opencode/dist the repo build wins
    # (fresher during development); otherwise the packaged bundle must resolve.
    if _REPO_BUILD.is_file():
        assert resolved == str(_REPO_BUILD)
    else:
        assert resolved == str(_PACKAGED)
