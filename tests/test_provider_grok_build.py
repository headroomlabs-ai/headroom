from __future__ import annotations

import sys
from pathlib import Path

import pytest

from headroom.providers.grok_build import build_proxy_targets, render_setup_lines
from headroom.providers.grok_build.config import (
    inject_grok_provider_config,
    redirect_existing_grok_build_base_url,
    render_headroom_block,
    restore_grok_provider_config,
    strip_grok_headroom_blocks,
)
from headroom.providers.grok_build.install import build_install_env

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _assert_valid_toml(content: str) -> None:
    payload = content.encode("utf-8")
    try:
        tomllib.loads(payload)
    except TypeError:
        # Some environments expose a str-accepting TOML parser shim.
        tomllib.loads(content)  # type: ignore[arg-type]


def _count_grok_build_tables(content: str) -> int:
    return content.count("[model.grok-build]")


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
    original = f'[models]\ndefault = "grok-build"\n\n{render_headroom_block(8787)}'
    cleaned = strip_grok_headroom_blocks(original)

    assert "[models]" in cleaned
    assert "headroom:grok-build" not in cleaned


def test_grok_build_inject_updates_existing_user_table_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    monkeypatch.setenv("GROK_HOME", str(grok_home))

    original = (
        "[models]\n"
        'default = "grok-build"\n\n'
        "[model.grok-build]\n"
        'model = "grok-build"\n'
        'base_url = "https://api.x.ai/v1"\n'
        "temperature = 0.5\n"
    )
    config_file = grok_home / "config.toml"
    config_file.write_text(original, encoding="utf-8")

    inject_grok_provider_config(8787, project="demo")
    content = config_file.read_text(encoding="utf-8")

    assert _count_grok_build_tables(content) == 1
    assert 'base_url = "http://127.0.0.1:8787/p/demo/v1"  # was: https://api.x.ai/v1' in content
    assert "temperature = 0.5" in content
    assert "headroom:grok-build" not in content
    _assert_valid_toml(content)


def test_grok_build_redirect_existing_base_url_is_idempotent() -> None:
    original = '[model.grok-build]\nbase_url = "http://127.0.0.1:8787/v1"\ntemperature = 0.2\n'

    updated, changed = redirect_existing_grok_build_base_url(original, "http://127.0.0.1:8787/v1")

    assert changed is False
    assert updated == original
    _assert_valid_toml(updated)
