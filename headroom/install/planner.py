"""Planner for persistent deployment manifests."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterable

import click

from headroom import paths as _paths
from headroom.providers.grok.runtime import DEFAULT_API_URL as _GROK_DEFAULT_API_URL
from headroom.providers.install_registry import build_install_target_envs
from headroom.rollout import RolloutChannel

from .models import (
    ConfigScope,
    DeploymentManifest,
    InstallPreset,
    ProviderSelectionMode,
    RuntimeKind,
    SupervisorKind,
    ToolTarget,
)
from .paths import validate_profile_name

SUPPORTED_TARGETS = [
    ToolTarget.CLAUDE,
    ToolTarget.COPILOT,
    ToolTarget.CODEX,
    ToolTarget.AIDER,
    ToolTarget.CURSOR,
    ToolTarget.GROK_BUILD,
    ToolTarget.GROK,
    ToolTarget.OPENCLAW,
    ToolTarget.OPENCODE,
]
PROVIDER_SCOPE_TARGETS = [
    ToolTarget.CLAUDE,
    ToolTarget.CODEX,
    ToolTarget.OPENCLAW,
    ToolTarget.OPENCODE,
]


def _binary_name(target: ToolTarget) -> str | None:
    if target == ToolTarget.CURSOR:
        return None
    return str(target.value)


def detect_targets() -> list[str]:
    """Auto-detect available tool targets on the current host."""

    detected: list[str] = []
    for target in SUPPORTED_TARGETS:
        binary = _binary_name(target)
        if binary and shutil.which(binary):
            detected.append(target.value)
            continue
        if target == ToolTarget.CURSOR and shutil.which("cursor"):
            detected.append(target.value)
            continue
        if target == ToolTarget.GROK_BUILD and shutil.which("grok"):
            detected.append(target.value)
    return detected


def resolve_targets(
    provider_mode: str, requested_targets: Iterable[str], *, scope: str = ConfigScope.USER.value
) -> list[str]:
    """Resolve target selection according to the requested provider mode."""

    valid_targets = SUPPORTED_TARGETS
    if scope == ConfigScope.PROVIDER.value:
        valid_targets = PROVIDER_SCOPE_TARGETS

    valid = {target.value for target in valid_targets}
    requested = [target.strip().lower() for target in requested_targets]

    if provider_mode == ProviderSelectionMode.ALL.value:
        return [target.value for target in valid_targets]

    if provider_mode == ProviderSelectionMode.AUTO.value:
        detected = [target for target in detect_targets() if target in valid]
        return detected or [
            ToolTarget.CLAUDE.value,
            ToolTarget.CODEX.value,
            *([] if scope == ConfigScope.PROVIDER.value else [ToolTarget.COPILOT.value]),
        ]

    # Manual selection is the only mode that consults `requested`, so the
    # provider-scope validation belongs here. Running it earlier rejected
    # unsupported entries that `all`/`auto` ignore entirely — e.g.
    # `install apply --scope provider --providers all --target cursor` raised
    # instead of returning the provider target set.
    if scope == ConfigScope.PROVIDER.value:
        unsupported = [target for target in requested if target and target not in valid]
        if unsupported:
            unsupported_list = ", ".join(sorted(set(unsupported)))
            raise click.ClickException(
                "Provider scope supports only claude, codex, openclaw, and opencode; "
                f"unsupported targets: {unsupported_list}"
            )

    normalized = []
    seen: set[str] = set()
    for value in requested:
        if value in valid and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def build_tool_envs(port: int, backend: str, targets: list[str]) -> dict[str, dict[str, str]]:
    """Build per-target environment variables for the selected tools."""
    return build_install_target_envs(port, backend, targets)


def build_manifest(
    *,
    profile: str,
    preset: str,
    runtime_kind: str,
    scope: str,
    provider_mode: str,
    targets: list[str],
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    proxy_mode: str | None,
    memory_enabled: bool,
    telemetry_enabled: bool,
    image: str,
    learn_enabled: bool | None = None,
    memory_storage_mode: str = "project",
    traffic_learning_min_evidence: int = 5,
    memory_project_root: str = "",
    no_http2: bool = False,
    code_aware: bool | None = None,
    intercept_tool_results: bool = False,
    protect_tool_results: str | None = None,
    bedrock_profile: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> DeploymentManifest:
    """Create a normalized deployment manifest."""

    normalized_profile = validate_profile_name(profile)

    # A Windows service must implement the Service Control Manager protocol.
    # The Python runner is an ordinary console process, so registering it with
    # ``sc.exe create`` always fails at start with SCM error 1053.  Task
    # Scheduler can run the same runner safely and already provides startup
    # plus periodic health recovery, so make it the effective preset on
    # Windows instead of creating a service that can never start (#2552).
    effective_preset = preset
    if sys.platform.startswith("win") and preset == InstallPreset.PERSISTENT_SERVICE.value:
        effective_preset = InstallPreset.PERSISTENT_TASK.value

    if effective_preset == InstallPreset.PERSISTENT_SERVICE.value:
        supervisor_kind = SupervisorKind.SERVICE.value
    elif effective_preset == InstallPreset.PERSISTENT_TASK.value:
        supervisor_kind = SupervisorKind.TASK.value
    else:
        supervisor_kind = SupervisorKind.NONE.value

    resolved_targets = resolve_targets(provider_mode, targets, scope=scope)
    tool_envs = build_tool_envs(port, backend, resolved_targets)
    base_env = {
        "HEADROOM_PORT": str(port),
        "HEADROOM_HOST": "127.0.0.1",
        "HEADROOM_BACKEND": backend,
    }
    if anyllm_provider:
        base_env["HEADROOM_ANYLLM_PROVIDER"] = anyllm_provider
    if region:
        base_env["HEADROOM_REGION"] = region
    # Telemetry is opt-in (off by default). Write the value explicitly so the
    # generated manifest is unambiguous and doesn't depend on the runtime default.
    base_env["HEADROOM_TELEMETRY"] = "on" if telemetry_enabled else "off"
    if memory_enabled:
        base_env["HEADROOM_MEMORY_ENABLED"] = "1"
    # Grok / Grok Build need proxy upstream = xAI. Only auto-set when no other
    # OpenAI-compatible tools share this proxy (those may need api.openai.com /
    # Copilot). Explicit OPENAI_TARGET_API_URL in extra_env still wins below.
    _openai_native = {
        ToolTarget.CODEX.value,
        ToolTarget.COPILOT.value,
        ToolTarget.AIDER.value,
        ToolTarget.OPENCODE.value,
    }
    _grok_targets = {ToolTarget.GROK.value, ToolTarget.GROK_BUILD.value}
    target_set = set(resolved_targets)
    if target_set & _grok_targets and not (target_set & _openai_native):
        base_env.setdefault("OPENAI_TARGET_API_URL", _GROK_DEFAULT_API_URL)
    # Applied last so explicit --env overrides win over the auto-derived
    # defaults above (e.g. a custom HEADROOM_WORKSPACE_DIR).
    if extra_env:
        base_env.update(extra_env)
    if intercept_tool_results:
        configured_channel = base_env.get("HEADROOM_ROLLOUT_CHANNEL")
        if configured_channel is None:
            # The flag is an explicit canary opt-in. Persist the matching
            # channel so the generated service can actually start.
            base_env["HEADROOM_ROLLOUT_CHANNEL"] = RolloutChannel.CANARY.value
        else:
            channel = RolloutChannel.parse(configured_channel)
            unsafe = base_env.get("HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
                "enabled",
            }
            if not channel.allows(RolloutChannel.CANARY) and not unsafe:
                raise click.ClickException(
                    "--intercept-tool-results requires HEADROOM_ROLLOUT_CHANNEL=canary "
                    "(or dev), unless the unsafe rollout override is explicitly enabled"
                )

    proxy_args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--backend",
        backend,
    ]
    if proxy_mode is not None:
        proxy_args.extend(["--mode", proxy_mode])
    proxy_args.append("--telemetry" if telemetry_enabled else "--no-telemetry")
    if memory_enabled:
        proxy_args.append("--memory")
        # `_paths.memory_db_path()` resolves against the HOST home. A container
        # runtime cannot use it: the container's HOME is /tmp/headroom-home and
        # the host's ~/.headroom is bind-mounted there, so a host path like
        # /home/<user>/.headroom/memory.db does not exist inside the container,
        # SQLite fails to open the DB, /readyz stays 503, and the deployment
        # times out and rolls back (#2803). Omit the flag for a container runtime:
        # the proxy then resolves the DB under its own cwd (.headroom/memory.db),
        # which is the container's workdir and therefore the bind mount, landing
        # in the same host file the explicit path intended. On the host (python)
        # runtime the resolved host path is correct, so keep passing it.
        if runtime_kind != RuntimeKind.DOCKER.value:
            proxy_args.extend(["--memory-db-path", str(_paths.memory_db_path())])
    if learn_enabled is not None:
        proxy_args.append("--learn" if learn_enabled else "--no-learn")
    proxy_args.extend(["--memory-storage", memory_storage_mode])
    if "HEADROOM_MIN_EVIDENCE" not in base_env:
        proxy_args.extend(["--min-evidence", str(traffic_learning_min_evidence)])
    if memory_project_root:
        proxy_args.extend(["--memory-project-root", memory_project_root])
    if anyllm_provider:
        proxy_args.extend(["--anyllm-provider", anyllm_provider])
    if region:
        proxy_args.extend(["--region", region])
    if no_http2:
        proxy_args.append("--no-http2")
    if code_aware is not None:
        proxy_args.append("--code-aware" if code_aware else "--no-code-aware")
    if intercept_tool_results:
        proxy_args.append("--intercept-tool-results")
    if protect_tool_results:
        proxy_args.extend(["--protect-tool-results", protect_tool_results])
    if bedrock_profile:
        proxy_args.extend(["--bedrock-profile", bedrock_profile])
    openai_target = base_env.get("OPENAI_TARGET_API_URL")
    if openai_target:
        proxy_args.extend(["--openai-api-url", openai_target])

    container_name = f"headroom-{normalized_profile}"
    return DeploymentManifest(
        profile=normalized_profile,
        preset=effective_preset,
        runtime_kind=runtime_kind,
        supervisor_kind=supervisor_kind,
        scope=scope,
        provider_mode=provider_mode,
        targets=resolved_targets,
        port=port,
        host="127.0.0.1",
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode=proxy_mode,
        memory_enabled=memory_enabled,
        memory_db_path=str(_paths.memory_db_path()),
        learn_enabled=learn_enabled,
        memory_storage_mode=memory_storage_mode,
        traffic_learning_min_evidence=traffic_learning_min_evidence,
        memory_project_root=memory_project_root,
        telemetry_enabled=telemetry_enabled,
        image=image,
        service_name=f"headroom-{normalized_profile}",
        container_name=container_name,
        health_url=f"http://127.0.0.1:{port}/readyz",
        base_env=base_env,
        tool_envs=tool_envs,
        proxy_args=proxy_args,
    )
