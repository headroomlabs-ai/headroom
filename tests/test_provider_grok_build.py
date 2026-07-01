from __future__ import annotations

from pathlib import Path

from headroom.providers.grok_build import build_proxy_targets, render_setup_lines
from headroom.providers.grok_build.config import (
    inject_grok_provider_config,
    render_headroom_block,
    restore_grok_provider_config,
    strip_grok_headroom_blocks,
)
from headroom.providers.grok_build.install import build_install_env


def test_grok_build_proxy_targets_use_local_headroom_proxy() -> None:
    target = build_proxy_targets(9999)

    assert target.base_url == "http://127.0.0.1:9999/v1"


def test_grok_build_setup_lines_include_proxy_url() -> None:
    lines = render_setup_lines(8787)
    joined = "\n".join(lines)

    assert "http://127.0.0.1:8787/v1" in joined
    assert "[model.grok-build]" in joined


def test_grok_build_build_install_env_returns_proxy_url() -> None:
    env = build_install_env(port=7654, backend="ignored")

    assert env == {"GROK_MODEL_GROK_BUILD_BASE_URL": "http://127.0.0.1:7654/v1"}


def test_grok_build_proxy_targets_apply_project_path_prefix() -> None:
    target = build_proxy_targets(9999, project="frontend")

    assert target.base_url == "http://127.0.0.1:9999/p/frontend/v1"


def test_grok_build_setup_lines_mention_project_attribution() -> None:
    lines = render_setup_lines(8787, project="frontend")
    joined = "\n".join(lines)

    assert "http://127.0.0.1:8787/p/frontend/v1" in joined
    assert "attributed to project 'frontend'" in joined


def test_grok_build_config_inject_and_restore_round_trip(tmp_path: Path, monkeypatch) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    monkeypatch.setenv("GROK_HOME", str(grok_home))

    config_file = inject_grok_provider_config(8787, project="demo")
    content = config_file.read_text(encoding="utf-8")

    assert render_headroom_block(8787, project="demo").strip() in content
    assert 'base_url = "http://127.0.0.1:8787/p/demo/v1"' in content

    status, _ = restore_grok_provider_config()
    assert status == "removed"
    assert not config_file.exists()


def test_grok_build_config_strip_preserves_user_content() -> None:
    original = (
        "[models]\n"
        'default = "grok-build"\n\n'
        f"{render_headroom_block(8787)}"
    )
    cleaned = strip_grok_headroom_blocks(original)

    assert "[models]" in cleaned
    assert "headroom:grok-build" not in cleaned