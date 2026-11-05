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


def test_provider_error_for_default_and_unsupported_providers() -> None:
    from headroom.providers.hermes import unsupported_provider_error

    assert unsupported_provider_error(()) is not None
    assert unsupported_provider_error(("--provider",)) is not None
    assert unsupported_provider_error(("--provider", "openrouter")) is not None
    assert unsupported_provider_error(("--provider=nous",)) is not None
    assert unsupported_provider_error(("--provider", "custom")) is not None


def test_provider_error_silent_for_routed_providers() -> None:
    from headroom.providers.hermes import unsupported_provider_error

    assert unsupported_provider_error(("-z", "hi", "--provider", "anthropic")) is None
    assert unsupported_provider_error(("--provider=OpenAI",)) is None
    assert unsupported_provider_error(("--provider", "openrouter", "--provider", "openai")) is None
