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


def test_build_launch_env_empty_mapping_does_not_inherit_host(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_TEST_HOST_ONLY", "1")
    env, _ = build_launch_env(9000, {})
    assert "HEADROOM_TEST_HOST_ONLY" not in env


def test_unrouted_provider_warning_for_default_and_openrouter() -> None:
    from headroom.providers.hermes import unrouted_provider_warning

    assert unrouted_provider_warning(()) is not None
    assert unrouted_provider_warning(("--provider", "openrouter")) is not None
    assert unrouted_provider_warning(("--provider=nous",)) is not None


def test_unrouted_provider_warning_silent_for_routed_providers() -> None:
    from headroom.providers.hermes import unrouted_provider_warning

    assert unrouted_provider_warning(("-z", "hi", "--provider", "anthropic")) is None
    assert unrouted_provider_warning(("--provider=OpenAI",)) is None
    assert unrouted_provider_warning(("--provider", "custom")) is None
