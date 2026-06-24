"""Tests for sync-plugin-versions.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).parent.parent / "sync-plugin-versions.py"
    spec = importlib.util.spec_from_file_location("sync_plugin_versions", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_repo_semver_uses_release_helpers(monkeypatch) -> None:
    module = _load_module()
    calls: dict[str, object] = {}

    monkeypatch.setattr(module, "list_release_tags", lambda root: ["v0.9.0"])
    monkeypatch.setattr(module, "find_latest_release_tag", lambda tags: "v0.9.0")
    monkeypatch.setattr(module, "list_release_commits", lambda root, tag: ["feat: add init"])
    monkeypatch.setattr(module, "determine_bump_level", lambda commits: "minor")
    monkeypatch.setattr(module, "get_canonical_version", lambda root: "0.5.25")

    def fake_compute_release_version(*, canonical_version: str, level: str, tags: list[str]):
        calls["canonical_version"] = canonical_version
        calls["level"] = level
        calls["tags"] = tags
        return type("Info", (), {"npm_version": "0.10.0"})()

    monkeypatch.setattr(module, "compute_release_version", fake_compute_release_version)

    assert module.compute_repo_semver(Path("repo")) == "0.10.0"
    assert calls == {
        "canonical_version": "0.5.25",
        "level": "minor",
        "tags": ["v0.9.0"],
    }


def test_main_runs_plugin_only_version_sync(monkeypatch) -> None:
    """Locks the sync-execution path. ``_should_sync`` is forced True so
    we don't depend on the test machine's git branch context."""
    module = _load_module()
    commands: list[list[str]] = []

    monkeypatch.setattr(module, "compute_repo_semver", lambda root: "0.10.0")
    monkeypatch.setattr(module, "_should_sync", lambda root: True)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, cwd, check: commands.append(command),
    )

    module.main()

    assert commands == [
        [
            module.sys.executable,
            str(module.ROOT / "scripts" / "version-sync.py"),
            "--root",
            str(module.ROOT),
            "--version",
            "0.10.0",
            "--plugin-manifests-only",
        ]
    ]


def test_main_is_noop_on_feature_branch(monkeypatch, capsys) -> None:
    """Locks the branch-aware contract: when ``_should_sync`` returns
    False (feature branch, no HEADROOM_SYNC_VERSIONS opt-in), main()
    prints a skip line and returns without invoking the version-sync
    subprocess. Pre-this-fix the hook bumped manifests on every commit
    regardless of branch — leaking version-bump noise into every PR."""
    module = _load_module()
    commands: list[list[str]] = []

    monkeypatch.setattr(module, "_should_sync", lambda root: False)
    monkeypatch.setattr(module, "_current_branch", lambda root: "feature/foo")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: commands.append(args),
    )

    module.main()

    assert commands == [], "Subprocess must not run when _should_sync returns False"
    captured = capsys.readouterr()
    assert "skipping on branch 'feature/foo'" in captured.out


def test_should_sync_honours_env_override(monkeypatch) -> None:
    """``HEADROOM_SYNC_VERSIONS=1`` forces a sync even on feature
    branches — the release workflow uses this opt-in so the canonical
    manifest sync still happens at publish time."""
    module = _load_module()
    monkeypatch.setenv("HEADROOM_SYNC_VERSIONS", "1")
    # Even on a "feature" branch, env override wins.
    monkeypatch.setattr(module, "_current_branch", lambda root: "feature/foo")
    assert module._should_sync(Path("ignored")) is True


def test_should_sync_main_branch_runs(monkeypatch) -> None:
    """On ``main``, sync runs without needing the env var."""
    module = _load_module()
    monkeypatch.delenv("HEADROOM_SYNC_VERSIONS", raising=False)
    monkeypatch.setattr(module, "_current_branch", lambda root: "main")
    assert module._should_sync(Path("ignored")) is True


def test_should_sync_feature_branch_is_skip(monkeypatch) -> None:
    """On any non-main branch without the env var, sync is a no-op."""
    module = _load_module()
    monkeypatch.delenv("HEADROOM_SYNC_VERSIONS", raising=False)
    monkeypatch.setattr(module, "_current_branch", lambda root: "fix/some-bug")
    assert module._should_sync(Path("ignored")) is False


def test_should_sync_returns_false_when_git_unavailable(monkeypatch) -> None:
    """Defensive: if ``_current_branch`` returns None (git not on
    PATH, detached HEAD, etc.) the safe default is no-op."""
    module = _load_module()
    monkeypatch.delenv("HEADROOM_SYNC_VERSIONS", raising=False)
    monkeypatch.setattr(module, "_current_branch", lambda root: None)
    assert module._should_sync(Path("ignored")) is False
