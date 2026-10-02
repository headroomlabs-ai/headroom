from __future__ import annotations

from headroom.providers.antigravity import (
    build_install_env,
    build_proxy_targets,
    render_setup_lines,
)


def test_antigravity_proxy_targets_openai_base_url() -> None:
    targets = build_proxy_targets(8787)
    assert targets.openai_base_url == "http://127.0.0.1:8787/v1"


def test_antigravity_proxy_targets_anthropic_base_url() -> None:
    targets = build_proxy_targets(8787)
    assert targets.anthropic_base_url == "http://127.0.0.1:8787"


def test_antigravity_proxy_targets_use_given_port() -> None:
    targets = build_proxy_targets(9999)
    assert targets.openai_base_url == "http://127.0.0.1:9999/v1"
    assert targets.anthropic_base_url == "http://127.0.0.1:9999"


def test_antigravity_proxy_targets_apply_project_prefix() -> None:
    targets = build_proxy_targets(8787, project="myrepo")
    assert targets.openai_base_url == "http://127.0.0.1:8787/p/myrepo/v1"
    assert targets.anthropic_base_url == "http://127.0.0.1:8787/p/myrepo"


def test_antigravity_proxy_targets_ignore_blank_project() -> None:
    targets = build_proxy_targets(8787, project="   ")
    assert targets.openai_base_url == "http://127.0.0.1:8787/v1"
    assert targets.anthropic_base_url == "http://127.0.0.1:8787"


def test_antigravity_build_install_env_sets_base_urls() -> None:
    env = build_install_env(port=8787, backend="ignored")
    assert env == {
        "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    }


def test_antigravity_build_install_env_uses_given_port() -> None:
    env = build_install_env(port=9999, backend="ignored")
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"


def test_antigravity_render_setup_lines_contains_proxy_url() -> None:
    lines = render_setup_lines(8787)
    joined = "\n".join(lines)
    assert "http://127.0.0.1:8787/v1" in joined
    assert "Antigravity" in joined


def test_antigravity_render_setup_lines_mentions_custom_provider() -> None:
    joined = "\n".join(render_setup_lines(8787))
    assert "OpenAI-compatible" in joined
    assert "/v1/models" in joined


def test_antigravity_render_setup_lines_project_attribution() -> None:
    lines = render_setup_lines(8787, project="my-sf-project")
    joined = "\n".join(lines)
    assert "my-sf-project" in joined
    plain = "\n".join(render_setup_lines(8787))
    assert "attributed" not in plain


def test_antigravity_install_registry_includes_antigravity() -> None:
    from headroom.providers.install_registry import build_install_target_envs

    result = build_install_target_envs(
        port=1234, backend="ignored", targets=["antigravity"]
    )
    assert result["antigravity"]["OPENAI_BASE_URL"] == "http://127.0.0.1:1234/v1"
    assert result["antigravity"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234"


def test_antigravity_install_registry_unknown_target_skipped() -> None:
    from headroom.providers.install_registry import build_install_target_envs

    result = build_install_target_envs(
        port=1234, backend="ignored", targets=["antigravity", "unknown-tool"]
    )
    assert "unknown-tool" not in result
    assert "antigravity" in result
