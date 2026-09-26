from __future__ import annotations

from headroom.providers.hermes import build_launch_env


def test_build_launch_env_sets_openai_base_url() -> None:
    env, _ = build_launch_env(9000, {})

    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"


def test_build_launch_env_sets_anthropic_base_url() -> None:
    env, _ = build_launch_env(9000, {})

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"


def test_build_launch_env_returns_display_strings() -> None:
    _, lines = build_launch_env(9000, {})

    assert lines == [
        "OPENAI_BASE_URL=http://127.0.0.1:9000/v1",
        "ANTHROPIC_BASE_URL=http://127.0.0.1:9000",
    ]


def test_build_launch_env_inherits_base_environ() -> None:
    env, _ = build_launch_env(9000, {"EXISTING_KEY": "existing-value"})

    assert env["EXISTING_KEY"] == "existing-value"
