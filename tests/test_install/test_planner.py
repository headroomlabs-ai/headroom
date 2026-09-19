from __future__ import annotations

import click
import pytest

from headroom.install.models import ConfigScope, InstallPreset, ProviderSelectionMode, ToolTarget
from headroom.install.planner import PROVIDER_SCOPE_TARGETS, build_manifest, resolve_targets


def test_resolve_targets_auto_falls_back_when_detection_empty(monkeypatch) -> None:
    monkeypatch.setattr("headroom.install.planner.detect_targets", lambda: [])

    targets = resolve_targets(ProviderSelectionMode.AUTO.value, [])

    assert targets == [
        ToolTarget.CLAUDE.value,
        ToolTarget.CODEX.value,
        ToolTarget.COPILOT.value,
    ]


def test_build_manifest_for_persistent_docker_sets_expected_defaults() -> None:
    manifest = build_manifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_DOCKER.value,
        runtime_kind="docker",
        scope="user",
        provider_mode="manual",
        targets=["claude", "copilot"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=True,
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )

    assert manifest.supervisor_kind == "none"
    assert manifest.runtime_kind == "docker"
    assert manifest.health_url == "http://127.0.0.1:8787/readyz"
    assert manifest.base_env["HEADROOM_PORT"] == "8787"
    assert manifest.base_env["HEADROOM_TELEMETRY"] == "off"
    assert "--no-telemetry" in manifest.proxy_args
    assert manifest.tool_envs["claude"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    # Native lane by default: the CLI keeps GitHub's model picker and Auto mode.
    assert manifest.tool_envs["copilot"] == {"COPILOT_API_URL": "http://127.0.0.1:8787"}
    assert "--memory" in manifest.proxy_args
    # A container runtime must NOT carry the host memory DB path: it does not
    # exist inside the container and would keep /readyz at 503 (#2803). The proxy
    # resolves the DB under its own cwd, which is the bind-mounted ~/.headroom.
    assert "--memory-db-path" not in manifest.proxy_args


def test_build_manifest_python_runtime_keeps_explicit_memory_db_path() -> None:
    manifest = build_manifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_SERVICE.value,
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=True,
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )

    # On the host the resolved path is correct, so it is still passed explicitly.
    assert "--memory" in manifest.proxy_args
    assert "--memory-db-path" in manifest.proxy_args


def test_build_manifest_falls_back_from_windows_service_to_task(monkeypatch) -> None:
    monkeypatch.setattr("headroom.install.planner.sys.platform", "win32")

    manifest = build_manifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_SERVICE.value,
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=False,
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )

    assert manifest.preset == InstallPreset.PERSISTENT_TASK.value
    assert manifest.supervisor_kind == "task"


def test_build_manifest_uses_provider_slice_env_builders_for_all_supported_targets() -> None:
    manifest = build_manifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_SERVICE.value,
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=["claude", "copilot", "codex", "aider", "cursor"],
        port=9999,
        backend="anyllm",
        anyllm_provider="groq",
        region=None,
        proxy_mode="token",
        memory_enabled=False,
        telemetry_enabled=True,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )

    # telemetry_enabled=True must write the explicit opt-in value + flag.
    assert manifest.base_env["HEADROOM_TELEMETRY"] == "on"
    assert "--telemetry" in manifest.proxy_args
    assert manifest.tool_envs["claude"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
    assert manifest.tool_envs["codex"]["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert manifest.tool_envs["aider"] == {
        "OPENAI_API_BASE": "http://127.0.0.1:9999/v1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999",
    }
    assert manifest.tool_envs["cursor"] == {
        "OPENAI_BASE_URL": "http://127.0.0.1:9999/v1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999",
    }
    assert manifest.tool_envs["copilot"] == {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:9999/v1",
        "COPILOT_PROVIDER_WIRE_API": "completions",
    }


def test_resolve_targets_provider_scope_auto_excludes_copilot(monkeypatch) -> None:
    monkeypatch.setattr("headroom.install.planner.detect_targets", lambda: [])

    targets = resolve_targets(
        ProviderSelectionMode.AUTO.value,
        [],
        scope=ConfigScope.PROVIDER.value,
    )

    assert targets == [ToolTarget.CLAUDE.value, ToolTarget.CODEX.value]


def test_resolve_targets_manual_dedupes_and_filters_invalid() -> None:
    targets = resolve_targets(
        ProviderSelectionMode.MANUAL.value,
        ["claude", "copilot", "claude", "invalid"],
    )

    assert targets == [ToolTarget.CLAUDE.value, ToolTarget.COPILOT.value]


def test_build_manifest_omits_no_http2_by_default() -> None:
    manifest = build_manifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_SERVICE.value,
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=False,
        telemetry_enabled=True,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )

    assert "--no-http2" not in manifest.proxy_args


def test_build_manifest_persists_no_http2_override() -> None:
    manifest = build_manifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_SERVICE.value,
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=False,
        telemetry_enabled=True,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
        no_http2=True,
    )

    assert manifest.proxy_args.count("--no-http2") == 1
    assert "HEADROOM_HTTP2" not in manifest.base_env


def test_resolve_targets_provider_scope_all_ignores_unsupported_requested() -> None:
    """`all` mode never consults the requested list, so an unsupported entry
    like `cursor` must not make it raise — it should return the full provider
    target set (regression: this used to raise a ClickException)."""
    targets = resolve_targets(
        ProviderSelectionMode.ALL.value,
        ["cursor"],
        scope=ConfigScope.PROVIDER.value,
    )

    assert targets == [t.value for t in PROVIDER_SCOPE_TARGETS]


def test_resolve_targets_provider_scope_auto_ignores_unsupported_requested(monkeypatch) -> None:
    """`auto` mode also ignores the requested list, so an unsupported entry
    must not raise."""
    monkeypatch.setattr("headroom.install.planner.detect_targets", lambda: [])

    targets = resolve_targets(
        ProviderSelectionMode.AUTO.value,
        ["cursor"],
        scope=ConfigScope.PROVIDER.value,
    )

    assert targets == [ToolTarget.CLAUDE.value, ToolTarget.CODEX.value]


def test_resolve_targets_provider_scope_manual_rejects_unsupported() -> None:
    """The manual path DOES consult the requested list, so an unsupported
    target under provider scope must still be rejected."""
    with pytest.raises(click.ClickException, match="cursor"):
        resolve_targets(
            ProviderSelectionMode.MANUAL.value,
            ["cursor"],
            scope=ConfigScope.PROVIDER.value,
        )


def _base_manifest_kwargs(**overrides):
    kwargs = {
        "profile": "default",
        "preset": InstallPreset.PERSISTENT_SERVICE.value,
        "runtime_kind": "python",
        "scope": "user",
        "provider_mode": "manual",
        "targets": ["claude"],
        "port": 8787,
        "backend": "bedrock",
        "anyllm_provider": None,
        "region": "eu-west-1",
        "proxy_mode": "token",
        "memory_enabled": False,
        "telemetry_enabled": False,
        "image": "ghcr.io/chopratejas/headroom:latest",
    }
    kwargs.update(overrides)
    return kwargs


def test_build_manifest_omits_new_bedrock_flags_by_default() -> None:
    manifest = build_manifest(**_base_manifest_kwargs())

    assert "--code-aware" not in manifest.proxy_args
    assert "--no-code-aware" not in manifest.proxy_args
    assert "--intercept-tool-results" not in manifest.proxy_args
    assert "--protect-tool-results" not in manifest.proxy_args
    assert "--bedrock-profile" not in manifest.proxy_args


def test_build_manifest_persists_code_aware_true() -> None:
    manifest = build_manifest(**_base_manifest_kwargs(code_aware=True))

    assert "--code-aware" in manifest.proxy_args
    assert "--no-code-aware" not in manifest.proxy_args


def test_build_manifest_persists_code_aware_false() -> None:
    manifest = build_manifest(**_base_manifest_kwargs(code_aware=False))

    assert "--no-code-aware" in manifest.proxy_args
    assert "--code-aware" not in manifest.proxy_args


def test_build_manifest_persists_intercept_tool_results() -> None:
    manifest = build_manifest(**_base_manifest_kwargs(intercept_tool_results=True))

    assert "--intercept-tool-results" in manifest.proxy_args
    assert manifest.base_env["HEADROOM_ROLLOUT_CHANNEL"] == "canary"


def test_build_manifest_rejects_interceptor_below_required_rollout_channel() -> None:
    with pytest.raises(click.ClickException, match="requires HEADROOM_ROLLOUT_CHANNEL=canary"):
        build_manifest(
            **_base_manifest_kwargs(
                intercept_tool_results=True,
                extra_env={"HEADROOM_ROLLOUT_CHANNEL": "stable"},
            )
        )


def test_build_manifest_persists_protect_tool_results() -> None:
    manifest = build_manifest(**_base_manifest_kwargs(protect_tool_results="Bash,WebFetch"))

    idx = manifest.proxy_args.index("--protect-tool-results")
    assert manifest.proxy_args[idx + 1] == "Bash,WebFetch"


def test_build_manifest_persists_bedrock_profile() -> None:
    manifest = build_manifest(**_base_manifest_kwargs(bedrock_profile="sso-bedrock"))

    idx = manifest.proxy_args.index("--bedrock-profile")
    assert manifest.proxy_args[idx + 1] == "sso-bedrock"


def test_build_manifest_merges_extra_env_into_base_env() -> None:
    manifest = build_manifest(
        **_base_manifest_kwargs(extra_env={"HEADROOM_WORKSPACE_DIR": "/custom/workspace"})
    )

    assert manifest.base_env["HEADROOM_WORKSPACE_DIR"] == "/custom/workspace"


def test_build_manifest_extra_env_overrides_derived_defaults() -> None:
    manifest = build_manifest(**_base_manifest_kwargs(extra_env={"HEADROOM_TELEMETRY": "on"}))

    # telemetry_enabled=False in _base_manifest_kwargs would normally set "off";
    # an explicit --env must win.
    assert manifest.base_env["HEADROOM_TELEMETRY"] == "on"


def test_build_manifest_grok_build_only_sets_xai_upstream() -> None:
    """Persistent install for Grok Build alone must route proxy upstream to xAI."""
    from headroom.providers.grok import DEFAULT_API_URL

    manifest = build_manifest(**_base_manifest_kwargs(targets=["grok_build"], backend="openai"))

    assert manifest.base_env.get("OPENAI_TARGET_API_URL") == DEFAULT_API_URL
    idx = manifest.proxy_args.index("--openai-api-url")
    assert manifest.proxy_args[idx + 1] == DEFAULT_API_URL


def test_build_manifest_grok_with_codex_does_not_force_xai() -> None:
    """Do not override OpenAI upstream when OpenAI-native tools share the proxy."""
    manifest = build_manifest(
        **_base_manifest_kwargs(targets=["grok_build", "codex"], backend="openai")
    )

    assert "OPENAI_TARGET_API_URL" not in manifest.base_env
    assert "--openai-api-url" not in manifest.proxy_args


def test_build_manifest_extra_env_wins_over_grok_xai_default() -> None:
    manifest = build_manifest(
        **_base_manifest_kwargs(
            targets=["grok_build"],
            backend="openai",
            extra_env={"OPENAI_TARGET_API_URL": "https://gateway.example/v1"},
        )
    )

    assert manifest.base_env["OPENAI_TARGET_API_URL"] == "https://gateway.example/v1"
    idx = manifest.proxy_args.index("--openai-api-url")
    assert manifest.proxy_args[idx + 1] == "https://gateway.example/v1"
