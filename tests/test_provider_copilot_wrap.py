from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import click
import pytest

from headroom.providers.copilot.wrap import (
    build_launch_env,
    copilot_model_from_args,
    default_wire_api_for_model,
    detect_running_proxy_backend,
    is_auto_model,
    model_configured,
    model_prefers_responses_api,
    provider_key_source,
    query_proxy_config,
    resolve_provider_type,
    strip_auto_model_args,
    validate_configuration,
)


def test_query_proxy_config_handles_success_and_invalid_payload() -> None:
    payload = io.BytesIO(json.dumps({"config": {"backend": "anyllm"}}).encode("utf-8"))
    payload_missing = io.BytesIO(json.dumps({"status": "ok"}).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=payload):
        assert query_proxy_config(8787) == {"backend": "anyllm"}
    with patch("urllib.request.urlopen", return_value=payload_missing):
        assert query_proxy_config(8787) is None
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        assert query_proxy_config(8787) is None


def test_detect_running_proxy_backend_requires_string_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.providers.copilot.wrap.query_proxy_config",
        lambda port: {"backend": 123} if port == 8787 else None,
    )

    assert detect_running_proxy_backend(8787) is None
    assert detect_running_proxy_backend(9999) is None


def test_resolve_provider_type_prefers_explicit_and_env() -> None:
    assert resolve_provider_type("anthropic", "openai") == "openai"
    assert resolve_provider_type(None, "auto", {"HEADROOM_BACKEND": "anthropic"}) == "anthropic"
    assert resolve_provider_type(None, "auto", {"HEADROOM_BACKEND": "anyllm"}) == "openai"


def test_validate_configuration_accepts_supported_combinations() -> None:
    validate_configuration(provider_type="openai", wire_api="responses", backend=None)
    validate_configuration(provider_type="openai", wire_api="completions", backend="anyllm")


def test_validate_configuration_rejects_invalid_combinations() -> None:
    with pytest.raises(click.ClickException, match="--wire-api is only valid"):
        validate_configuration(provider_type="anthropic", wire_api="responses", backend=None)

    with pytest.raises(click.ClickException, match="not supported with translated backends"):
        validate_configuration(provider_type="openai", wire_api="responses", backend="anyllm")


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.5", True),
        ("gpt-5-codex", True),
        ("openai/gpt-5.4", True),
        ("o1", True),
        ("o3-mini", True),
        ("gpt-4.1", False),
        ("claude-sonnet-4.6", False),
        (None, False),
    ],
)
def test_model_prefers_responses_api_for_reasoning_models(
    model: str | None,
    expected: bool,
) -> None:
    assert model_prefers_responses_api(model) is expected
    assert default_wire_api_for_model(model) == ("responses" if expected else "completions")


def test_copilot_model_from_args_prefers_cli_over_environment() -> None:
    assert (
        copilot_model_from_args(
            ("--model", "gpt-5.5"),
            {"COPILOT_MODEL": "gpt-4.1"},
        )
        == "gpt-5.5"
    )
    assert (
        copilot_model_from_args(
            ("--model=gpt-5-codex",),
            {"COPILOT_PROVIDER_MODEL_ID": "gpt-4.1"},
        )
        == "gpt-5-codex"
    )
    assert copilot_model_from_args((), {"COPILOT_PROVIDER_MODEL_ID": "gpt-4.1"}) == "gpt-4.1"


def test_provider_key_source_and_build_launch_env_cover_anthropic_and_openai() -> None:
    assert provider_key_source("anthropic") == "ANTHROPIC_API_KEY"
    assert provider_key_source("openai") == "OPENAI_API_KEY"

    anthropic_env, anthropic_lines = build_launch_env(
        port=8787,
        provider_type="anthropic",
        wire_api="responses",
        environ={
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "COPILOT_PROVIDER_WIRE_API": "stale",
        },
    )
    openai_env, openai_lines = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api=None,
        environ={"OPENAI_API_KEY": "sk-proj-test"},
    )

    assert anthropic_env["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert anthropic_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787"
    assert anthropic_env["COPILOT_PROVIDER_API_KEY"] == "sk-ant-test"
    assert "COPILOT_PROVIDER_WIRE_API" not in anthropic_env
    assert anthropic_lines == [
        "COPILOT_PROVIDER_TYPE=anthropic",
        "COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:8787",
    ]

    assert openai_env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert openai_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert openai_env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert openai_env["COPILOT_PROVIDER_API_KEY"] == "sk-proj-test"
    assert openai_lines[-1] == "COPILOT_PROVIDER_WIRE_API=completions"


def test_build_launch_env_keeps_existing_provider_key_and_allows_missing_source_key() -> None:
    existing_env, _existing_lines = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api="responses",
        environ={
            "COPILOT_PROVIDER_API_KEY": "existing-provider-key",
            "OPENAI_API_KEY": "sk-proj-test",
        },
    )
    missing_env, _missing_lines = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api="responses",
        environ={},
    )

    assert existing_env["COPILOT_PROVIDER_API_KEY"] == "existing-provider-key"
    assert "COPILOT_PROVIDER_API_KEY" not in missing_env


def test_model_configured_detects_env_and_cli_variants() -> None:
    assert model_configured((), {"COPILOT_MODEL": "gpt-4o"}) is True
    assert model_configured(("--model", "gpt-4o"), {}) is True
    assert model_configured(("--model=gpt-4o",), {}) is True
    assert model_configured(("--other", "value"), {}) is False
    # ``auto`` is not a valid BYOK model — must be treated as unconfigured.
    assert model_configured(("--model", "auto"), {}) is False
    assert model_configured(("--model=auto",), {}) is False
    assert model_configured((), {"COPILOT_MODEL": "auto"}) is False
    assert model_configured((), {"COPILOT_PROVIDER_MODEL_ID": "auto"}) is False


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("auto", True),
        ("Auto", True),
        ("AUTO", True),
        ("  auto  ", True),
        ("gpt-4o", False),
        ("gpt-5", False),
        ("claude-sonnet-4.6", False),
        (None, False),
        ("", False),
    ],
)
def test_is_auto_model(model: str | None, expected: bool) -> None:
    assert is_auto_model(model) is expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # Strips --model auto (space-separated)
        (
            ("--model", "auto", "--port", "8788"),
            ("--port", "8788"),
        ),
        # Strips --model=auto (equals form)
        (
            ("--model=auto", "-p", "hello"),
            ("-p", "hello"),
        ),
        # Case-insensitive stripping
        (
            ("--model", "AUTO", "--allow-all-tools"),
            ("--allow-all-tools",),
        ),
        # Leaves concrete models untouched
        (
            ("--model", "gpt-4o", "--port", "8788"),
            ("--model", "gpt-4o", "--port", "8788"),
        ),
        # Leaves --model=gpt-4o untouched
        (
            ("--model=gpt-4o",),
            ("--model=gpt-4o",),
        ),
        # Empty args unchanged
        ((), ()),
        # --model at end with no value (malformed) — leave as-is, don't crash
        (("--model",), ("--model",)),
    ],
)
def test_strip_auto_model_args(
    args: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    assert strip_auto_model_args(args) == expected


def test_build_launch_env_applies_project_path_prefix() -> None:
    anthropic_env, _ = build_launch_env(
        port=8787,
        provider_type="anthropic",
        wire_api=None,
        environ={"ANTHROPIC_API_KEY": "sk-ant-test"},
        project="api server",
    )
    openai_env, openai_lines = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api=None,
        environ={"OPENAI_API_KEY": "sk-proj-test"},
        project="api server",
    )

    assert anthropic_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/p/api%20server"
    assert openai_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/p/api%20server/v1"
    assert "COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:8787/p/api%20server/v1" in openai_lines

    plain_env, _ = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api=None,
        environ={"OPENAI_API_KEY": "sk-proj-test"},
    )
    assert plain_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
