"""Persistent install / deployment CLI commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import click
from click.core import ParameterSource

from headroom._subprocess import run
from headroom.install.health import probe_json, probe_ready
from headroom.install.models import (
    ConfigScope,
    DeploymentManifest,
    InstallPreset,
    ProviderSelectionMode,
    RuntimeKind,
    SupervisorKind,
)
from headroom.install.planner import build_manifest
from headroom.install.providers import apply_mutations, revert_mutations
from headroom.install.runtime import (
    acquire_runtime_start_lock,
    run_foreground,
    runtime_status,
    start_detached_agent,
    start_persistent_docker,
    stop_runtime,
    wait_ready,
)
from headroom.install.state import (
    ManifestError,
    delete_manifest,
    list_manifests,
    load_manifest,
    save_manifest,
)
from headroom.install.supervisors import (
    install_supervisor,
    remove_supervisor,
    start_supervisor,
    stop_supervisor,
)

from .main import main


@dataclass(frozen=True)
class TurnkeyPlan:
    """Resolved runtime strategy for one-line deployments."""

    preset: str
    runtime: str
    supervisor_kind: str | None
    reason: str
    base_env: dict[str, str] | None = None


@main.group()
def install() -> None:
    """Install and manage persistent Headroom deployments."""


def _profile_selection_was_explicit() -> bool:
    """True when the current command received an explicit ``--profile``.

    An explicit selection must be honored verbatim or rejected, never redirected
    to ``HEADROOM_DEPLOYMENT_PROFILE`` or a lone installed deployment: silently
    operating ``stop``/``restart``/``remove`` on a different profile than the one
    the user typed is dangerous. Only a defaulted (omitted) ``--profile`` is
    eligible for the recovery fallback. Outside a Click command context (direct
    calls / unit tests) there is no explicit selection to protect.
    """
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    return bool(ctx.get_parameter_source("profile") == ParameterSource.COMMANDLINE)


def _missing_profile_error(
    name: str,
    installed: list[DeploymentManifest],
    *,
    source: str | None = None,
) -> click.ClickException:
    if installed:
        names = ", ".join(sorted(m.profile for m in installed))
        hint = f" Installed: {names}. Select one with --profile <name>."
    else:
        hint = " No deployments are installed; run `headroom init` or `headroom install apply`."
    origin = f" (from {source})" if source else ""
    return click.ClickException(f"No deployment profile named '{name}'{origin} is installed.{hint}")


def _require_manifest(profile: str) -> DeploymentManifest:
    try:
        manifest = load_manifest(profile)
    except ManifestError as e:
        raise click.ClickException(str(e)) from None
    if manifest is not None:
        return manifest

    # The requested profile isn't installed. `headroom init` installs under a
    # non-"default" profile name (e.g. init-user), while every lifecycle command
    # defaults --profile to "default" -- so on an init'd machine the documented
    # bare commands (`headroom install status`, etc.) would all dead-end (#2811).
    installed = list_manifests()

    # An EXPLICIT --profile is honored or rejected verbatim, never redirected: a
    # typo must not silently act on the env/lone profile (#2832 review).
    if _profile_selection_was_explicit():
        raise _missing_profile_error(profile, installed)

    # --profile was defaulted. A non-empty HEADROOM_DEPLOYMENT_PROFILE (which the
    # runtime exports) is itself an explicit selection: honor it when installed,
    # otherwise fail naming it. It must never fall through to the lone-manifest
    # fallback and silently operate on a different deployment (#2832 review).
    env_profile = os.environ.get("HEADROOM_DEPLOYMENT_PROFILE", "").strip()
    if env_profile:
        if env_profile != profile:
            try:
                resolved = load_manifest(env_profile)
            except ManifestError:
                resolved = None
            if resolved is not None:
                return resolved
        raise _missing_profile_error(env_profile, installed, source="HEADROOM_DEPLOYMENT_PROFILE")

    # Neither CLI nor environment named a profile. A single installed deployment
    # is unambiguous, so use it; otherwise report what is available.
    if len(installed) == 1:
        return installed[0]
    raise _missing_profile_error(profile, installed)


def _is_windows() -> bool:
    """Return whether this command is running on Windows."""

    return sys.platform.startswith("win")


def _start_deployment(manifest: DeploymentManifest, *, assume_start_lock: bool = False) -> None:
    if not assume_start_lock:
        with acquire_runtime_start_lock(manifest.profile) as acquired:
            if not acquired:
                click.echo(f"Deployment '{manifest.profile}' start is already in progress.")
                return
            _start_deployment(manifest, assume_start_lock=True)
            return

    if probe_ready(manifest.health_url):
        return
    if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value and shutil.which("docker") is None:
        raise click.ClickException(
            "Docker is required for this deployment but 'docker' was not found on PATH."
        )
    if runtime_status(manifest) == "running":
        if wait_ready(manifest, timeout_seconds=_STARTUP_READY_TIMEOUT_SECONDS):
            return
        stop_runtime(manifest)

    try:
        if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
            start_persistent_docker(manifest)
        elif manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            start_supervisor(manifest)
        else:
            start_detached_agent(manifest.profile)
    except FileNotFoundError as e:
        # A required external binary (docker, launchctl, systemctl) is missing.
        raise click.ClickException(f"Cannot start deployment '{manifest.profile}': {e}") from None
    except subprocess.CalledProcessError as e:
        raise click.ClickException(
            f"Cannot start deployment '{manifest.profile}': command failed "
            f"({' '.join(map(str, e.cmd)) if isinstance(e.cmd, list | tuple) else e.cmd})"
        ) from None

    if not wait_ready(manifest, timeout_seconds=45):
        raise click.ClickException(
            f"Deployment '{manifest.profile}' did not become ready after start."
        )


def _stop_deployment(manifest: DeploymentManifest) -> None:
    if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        stop_supervisor(manifest)
    stop_runtime(manifest)


def _deactivate_deployment_mutations(
    manifest: DeploymentManifest, *, persist_manifest: bool = True
) -> None:
    if not manifest.mutations:
        return
    revert_mutations(manifest)
    manifest.mutations = []
    if persist_manifest:
        save_manifest(manifest)


def _activate_deployment_mutations(manifest: DeploymentManifest) -> None:
    manifest.mutations = apply_mutations(manifest)
    save_manifest(manifest)


def _remove_deployment(manifest: DeploymentManifest) -> None:
    try:
        _deactivate_deployment_mutations(manifest, persist_manifest=False)
    except Exception:
        pass
    try:
        _stop_deployment(manifest)
    except Exception:
        pass
    try:
        remove_supervisor(manifest)
    except Exception:
        pass
    delete_manifest(manifest.profile)


def _restore_deployment(manifest: DeploymentManifest) -> None:
    restored = deepcopy(manifest)
    restored.artifacts = install_supervisor(restored)
    save_manifest(restored)
    _start_deployment(restored)
    _activate_deployment_mutations(restored)


def _reject_task_lifecycle(manifest: DeploymentManifest, action: str) -> None:
    if manifest.supervisor_kind == SupervisorKind.TASK.value:
        raise click.ClickException(
            f"Deployment '{manifest.profile}' uses persistent-task scheduling; "
            f"`headroom install {action}` is not supported for task deployments."
        )


def _command_available(command: str) -> bool:
    return shutil.which(command) is not None


def _probe_command(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return None


def _detect_nvidia_gpu_names() -> list[str]:
    if not _command_available("nvidia-smi"):
        return []
    result = _probe_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if result is None or result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _docker_supports_nvidia_gpus() -> bool:
    if not _command_available("docker"):
        return False
    result = _probe_command(["docker", "info", "--format", "{{json .Runtimes}}"])
    if result is None or result.returncode != 0:
        return False
    return "nvidia" in result.stdout.lower()


def _select_turnkey_plan(*, prefer_docker: bool = True) -> TurnkeyPlan:
    """Choose the most portable available deployment strategy for this host."""

    gpu_names = _detect_nvidia_gpu_names()
    if prefer_docker and gpu_names and _docker_supports_nvidia_gpus():
        gpu_summary = ", ".join(gpu_names[:2])
        if len(gpu_names) > 2:
            gpu_summary += f", +{len(gpu_names) - 2} more"
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_DOCKER.value,
            runtime=RuntimeKind.DOCKER.value,
            supervisor_kind=SupervisorKind.NONE.value,
            reason=f"NVIDIA GPU available ({gpu_summary}); using Docker GPU passthrough.",
            base_env={"HEADROOM_DOCKER_GPUS": "all"},
        )

    if prefer_docker and _command_available("docker"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_DOCKER.value,
            runtime=RuntimeKind.DOCKER.value,
            supervisor_kind=SupervisorKind.NONE.value,
            reason="Docker is available, so Headroom can run in a restartable container.",
        )

    if sys.platform == "darwin" and _command_available("launchctl"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_TASK.value,
            runtime=RuntimeKind.PYTHON.value,
            supervisor_kind=SupervisorKind.TASK.value,
            reason="launchd is available for scheduled health recovery.",
        )

    if sys.platform.startswith("win") and _command_available("schtasks"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_TASK.value,
            runtime=RuntimeKind.PYTHON.value,
            supervisor_kind=SupervisorKind.TASK.value,
            reason="Windows Task Scheduler is available for scheduled health recovery.",
        )

    if sys.platform.startswith("linux") and _command_available("crontab"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_TASK.value,
            runtime=RuntimeKind.PYTHON.value,
            supervisor_kind=SupervisorKind.TASK.value,
            reason="cron is available for scheduled health recovery.",
        )

    return TurnkeyPlan(
        preset=InstallPreset.PERSISTENT_TASK.value,
        runtime=RuntimeKind.PYTHON.value,
        supervisor_kind=SupervisorKind.NONE.value,
        reason="No supported supervisor was detected; Headroom will start a managed detached runtime.",
    )


def _build_deployment_manifest(
    *,
    preset: str,
    runtime: str,
    scope: str,
    provider_mode: str,
    targets: tuple[str, ...],
    profile: str,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    proxy_mode: str,
    memory: bool,
    telemetry: bool,
    no_telemetry: bool,
    image: str,
    no_http2: bool,
    code_aware: bool | None = None,
    intercept_tool_results: bool = False,
    protect_tool_results: str | None = None,
    bedrock_profile: str | None = None,
    extra_env: dict[str, str] | None = None,
    supervisor_kind: str | None = None,
    extra_base_env: dict[str, str] | None = None,
    no_rate_limit: bool = False,
) -> DeploymentManifest:
    manifest = build_manifest(
        profile=profile,
        preset=preset,
        runtime_kind=runtime,
        scope=scope,
        provider_mode=provider_mode,
        targets=list(targets),
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode=proxy_mode,
        memory_enabled=memory,
        telemetry_enabled=telemetry and not no_telemetry,
        no_rate_limit=no_rate_limit,
        image=image,
        no_http2=no_http2,
        code_aware=code_aware,
        intercept_tool_results=intercept_tool_results,
        protect_tool_results=protect_tool_results,
        bedrock_profile=bedrock_profile,
        extra_env=extra_env or {},
    )
    if supervisor_kind is not None:
        manifest.supervisor_kind = supervisor_kind
    if extra_base_env:
        manifest.base_env.update(extra_base_env)
    return manifest


# Upstream-routing overrides the interactive `headroom proxy` reads from the
# environment (via resolve_api_overrides), but a supervised runner starts from a
# bare environment, so these never reach the persistent proxy unless captured
# into the manifest. Without this, `install apply` with e.g.
# ANTHROPIC_TARGET_API_URL exported silently routes to the default provider
# endpoint instead of the user's gateway (#2240). Only URL overrides are
# captured; the *_TARGET_API_HEADERS vars can carry bearer tokens and are left
# to explicit `--env` so a secret is never persisted to the manifest implicitly.
_PASSTHROUGH_URL_ENV_VARS = (
    "ANTHROPIC_TARGET_API_URL",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "OPENAI_TARGET_API_URL",
    "GEMINI_TARGET_API_URL",
    "CLOUDCODE_TARGET_API_URL",
    "VERTEX_TARGET_API_URL",
    "BEDROCK_TARGET_API_URL",
)


def _capture_passthrough_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Return the upstream-routing overrides present in ``environ``.

    An empty or unset value is skipped so it cannot shadow an auto-derived
    default. Explicit ``--env`` values are meant to win over these, so callers
    should merge the returned dict *under* the parsed ``--env`` map.
    """
    captured: dict[str, str] = {}
    for name in _PASSTHROUGH_URL_ENV_VARS:
        value = environ.get(name)
        if value:
            captured[name] = value
    return captured


def _apply_manifest(manifest: DeploymentManifest) -> None:
    try:
        existing = load_manifest(manifest.profile)
    except ManifestError as e:
        # A corrupt existing manifest shouldn't block a fresh apply; overwrite it.
        click.echo(f"Warning: {e}; overwriting.")
        existing = None
    if existing is not None:
        click.echo(f"Updating existing deployment profile '{manifest.profile}'...")
        _remove_deployment(existing)

    try:
        manifest.artifacts = install_supervisor(manifest)
        save_manifest(manifest)
        _start_deployment(manifest)
        _activate_deployment_mutations(manifest)
    except Exception as exc:
        _remove_deployment(manifest)
        if existing is not None:
            click.echo(f"Restoring previous deployment '{manifest.profile}'...")
            _restore_deployment(existing)
        # Surface non-Click errors (OSError, CalledProcessError, ...) as a clean
        # message rather than a raw traceback; Click errors pass through as-is.
        if isinstance(exc, click.ClickException | click.Abort):
            raise
        raise click.ClickException(
            f"Failed to install deployment '{manifest.profile}': {exc}"
        ) from exc


def _echo_installed(manifest: DeploymentManifest, *, prefix: str = "Installed persistent") -> None:
    click.echo(
        f"{prefix} deployment '{manifest.profile}' "
        f"({manifest.preset}, runtime={manifest.runtime_kind}, scope={manifest.scope})."
    )
    click.echo(f"Health: {manifest.health_url}")
    if manifest.targets:
        click.echo(f"Targets: {', '.join(manifest.targets)}")


@install.command("apply")
@click.option(
    "--preset",
    type=click.Choice([preset.value for preset in InstallPreset]),
    default=InstallPreset.PERSISTENT_SERVICE.value,
    show_default=True,
    help="Persistent runtime preset to install.",
)
@click.option(
    "--runtime",
    type=click.Choice([runtime.value for runtime in RuntimeKind]),
    default=RuntimeKind.PYTHON.value,
    show_default=True,
    help="Runtime used to execute Headroom for service/task modes.",
)
@click.option(
    "--scope",
    type=click.Choice([scope.value for scope in ConfigScope]),
    default=ConfigScope.USER.value,
    show_default=True,
    help="Where to apply persistent configuration.",
)
@click.option(
    "--providers",
    "provider_mode",
    type=click.Choice([mode.value for mode in ProviderSelectionMode]),
    default=ProviderSelectionMode.AUTO.value,
    show_default=True,
    help="Target selection mode for direct tool configuration.",
)
@click.option(
    "--target",
    "targets",
    multiple=True,
    type=click.Choice(
        ["claude", "copilot", "codex", "aider", "cursor", "grok_build", "openclaw", "opencode"]
    ),
    help="Tool target to configure when --providers manual is used.",
)
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
@click.option(
    "--port",
    "-p",
    default=8787,
    envvar="HEADROOM_PORT",
    type=click.IntRange(1, 65535),
    show_default=True,
    help="Persistent proxy port (env: HEADROOM_PORT).",
)
@click.option(
    "--backend",
    default="anthropic",
    show_default=True,
    help="Proxy backend for the persistent runtime.",
)
@click.option(
    "--anyllm-provider",
    default=None,
    help="Provider for any-llm backends when --backend anyllm is used.",
)
@click.option("--region", default=None, help="Cloud region for Bedrock / Vertex style backends.")
@click.option(
    "--mode",
    "proxy_mode",
    default="cache",
    show_default=True,
    help="Proxy optimization mode. cache = delta-only compression at ~0 prefix-cache busts.",
)
@click.option("--memory", is_flag=True, help="Enable persistent memory in the proxy runtime.")
@click.option(
    "--telemetry",
    is_flag=True,
    help="Opt in to anonymous telemetry in the runtime (off by default).",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    help="Force anonymous telemetry off in the runtime (already the default).",
)
@click.option(
    "--no-rate-limit",
    "no_rate_limit",
    is_flag=True,
    default=False,
    help=(
        "Disable the proxy's built-in rate limiter (default: 60 req/min). "
        "Recommended for always-on agentic targets (Claude Code, Codex) that "
        "burst above the default threshold. The flag is persisted in the "
        "deployment manifest so reinstalls don't silently reintroduce throttling."
    ),
)
@click.option(
    "--image",
    default="ghcr.io/headroomlabs-ai/headroom:latest",
    show_default=True,
    help="Docker image to use when runtime=docker or preset=persistent-docker.",
)
@click.option(
    "--no-http2",
    is_flag=True,
    help="Disable HTTP/2 in the persistent runtime (enabled by default).",
)
@click.option(
    "--code-aware/--no-code-aware",
    "code_aware",
    default=None,
    help=(
        "Enable/disable AST-based code compression in the persistent runtime. "
        "Requires the optional tree-sitter dependency: pip install headroom-ai[code]. "
        "Default: disabled, matching `headroom proxy`."
    ),
)
@click.option(
    "--intercept-tool-results",
    is_flag=True,
    help=(
        "Opt in to tool_result interceptors (ast-grep Read outliner, etc.) in the "
        "persistent runtime. This also selects the required canary rollout channel "
        "unless --env HEADROOM_ROLLOUT_CHANNEL=... is supplied."
    ),
)
@click.option(
    "--protect-tool-results",
    default=None,
    help=(
        "Comma-separated tool names whose results are never lossy-compressed in "
        "the persistent runtime, merged with the built-in defaults (e.g. Bash,WebFetch)."
    ),
)
@click.option(
    "--bedrock-profile",
    default=None,
    help="AWS profile name for Bedrock in the persistent runtime (default: use default credentials).",
)
@click.option(
    "--env",
    "extra_env",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Extra environment variable for the supervised process, e.g. "
        "--env HEADROOM_WORKSPACE_DIR=/path. Supervisors (launchd, systemd, cron) "
        "start with a bare environment and do not inherit the interactive shell's "
        "exports, so anything the runtime needs beyond the flags above must be set "
        "here. Repeatable."
    ),
)
def install_apply(
    preset: str,
    runtime: str,
    scope: str,
    provider_mode: str,
    targets: tuple[str, ...],
    profile: str,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    proxy_mode: str,
    memory: bool,
    telemetry: bool,
    no_telemetry: bool,
    no_rate_limit: bool,
    image: str,
    no_http2: bool,
    code_aware: bool | None,
    intercept_tool_results: bool,
    protect_tool_results: str | None,
    bedrock_profile: str | None,
    extra_env: tuple[str, ...],
) -> None:
    """Install a persistent Headroom deployment."""

    if anyllm_provider and backend != "anyllm":
        click.echo(
            f"Warning: --anyllm-provider is ignored unless --backend anyllm "
            f"(got --backend {backend})."
        )

    if preset == InstallPreset.PERSISTENT_DOCKER.value:
        runtime = RuntimeKind.DOCKER.value

    parsed_env: dict[str, str] = {}
    for item in extra_env:
        if "=" not in item:
            raise click.ClickException(f"--env expects KEY=VALUE, got: {item!r}")
        key, _, value = item.partition("=")
        parsed_env[key] = value

    # Auto-carry upstream-routing overrides from the current environment so a
    # supervised runner forwards to the same gateway the interactive proxy would
    # (#2240). Explicit --env wins, so merge the captured vars underneath.
    combined_env = {**_capture_passthrough_env(os.environ), **parsed_env}

    manifest = _build_deployment_manifest(
        profile=profile,
        preset=preset,
        runtime=runtime,
        scope=scope,
        provider_mode=provider_mode,
        targets=targets,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode=proxy_mode,
        memory=memory,
        telemetry=telemetry,
        no_telemetry=no_telemetry,
        no_rate_limit=no_rate_limit,
        image=image,
        no_http2=no_http2,
        code_aware=code_aware,
        intercept_tool_results=intercept_tool_results,
        protect_tool_results=protect_tool_results,
        bedrock_profile=bedrock_profile,
        extra_env=combined_env,
    )
    if (
        preset == InstallPreset.PERSISTENT_SERVICE.value
        and manifest.preset == InstallPreset.PERSISTENT_TASK.value
        and _is_windows()
    ):
        click.echo(
            "Warning: persistent-service is not supported on Windows because the "
            "Python runner cannot act as a Windows service. Falling back to "
            "persistent-task with Task Scheduler."
        )

    _apply_manifest(manifest)
    _echo_installed(manifest)


@main.command("deploy")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
@click.option(
    "--port",
    "-p",
    default=8787,
    envvar="HEADROOM_PORT",
    type=int,
    show_default=True,
    help="Persistent proxy port (env: HEADROOM_PORT).",
)
@click.option(
    "--backend",
    default="anthropic",
    show_default=True,
    help="Proxy backend for the persistent runtime.",
)
@click.option(
    "--anyllm-provider",
    default=None,
    help="Provider for any-llm backends when --backend anyllm is used.",
)
@click.option("--region", default=None, help="Cloud region for Bedrock / Vertex style backends.")
@click.option(
    "--mode",
    "proxy_mode",
    default="cache",
    show_default=True,
    help="Proxy optimization mode. cache = delta-only compression at ~0 prefix-cache busts.",
)
@click.option(
    "--scope",
    type=click.Choice([scope.value for scope in ConfigScope]),
    default=ConfigScope.USER.value,
    show_default=True,
    help="Where to apply persistent configuration.",
)
@click.option(
    "--providers",
    "provider_mode",
    type=click.Choice([mode.value for mode in ProviderSelectionMode]),
    default=ProviderSelectionMode.AUTO.value,
    show_default=True,
    help="Target selection mode for direct tool configuration.",
)
@click.option(
    "--target",
    "targets",
    multiple=True,
    type=click.Choice(["claude", "copilot", "codex", "aider", "cursor", "openclaw", "opencode"]),
    help="Tool target to configure when --providers manual is used.",
)
@click.option("--memory", is_flag=True, help="Enable persistent memory in the proxy runtime.")
@click.option(
    "--telemetry",
    is_flag=True,
    help="Opt in to anonymous telemetry in the runtime (off by default).",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    help="Force anonymous telemetry off in the runtime (already the default).",
)
@click.option(
    "--image",
    default="ghcr.io/headroomlabs-ai/headroom:latest",
    show_default=True,
    help="Docker image to use when Docker is selected.",
)
@click.option(
    "--no-docker",
    is_flag=True,
    help="Use the native Python runtime even when Docker is installed.",
)
@click.option(
    "--no-http2",
    is_flag=True,
    help="Disable HTTP/2 in the persistent runtime (enabled by default).",
)
def deploy(
    profile: str,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    proxy_mode: str,
    scope: str,
    provider_mode: str,
    targets: tuple[str, ...],
    memory: bool,
    telemetry: bool,
    no_telemetry: bool,
    image: str,
    no_docker: bool,
    no_http2: bool,
) -> None:
    """Deploy a turnkey local Headroom proxy and configure detected tools."""

    plan = _select_turnkey_plan(prefer_docker=not no_docker)
    click.echo(f"Selected {plan.preset} ({plan.runtime}): {plan.reason}")
    manifest = _build_deployment_manifest(
        profile=profile,
        preset=plan.preset,
        runtime=plan.runtime,
        scope=scope,
        provider_mode=provider_mode,
        targets=targets,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode=proxy_mode,
        memory=memory,
        telemetry=telemetry,
        no_telemetry=no_telemetry,
        image=image,
        no_http2=no_http2,
        supervisor_kind=plan.supervisor_kind,
        extra_base_env=plan.base_env,
    )
    _apply_manifest(manifest)
    _echo_installed(manifest, prefix="Deployed turnkey")


@install.command("status")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_status(profile: str) -> None:
    """Show persistent deployment status."""

    manifest = _require_manifest(profile)
    payload = probe_json(manifest.health_url.replace("/readyz", "/health"))
    click.echo(f"Profile:    {manifest.profile}")
    click.echo(f"Preset:     {manifest.preset}")
    click.echo(f"Runtime:    {manifest.runtime_kind}")
    click.echo(f"Supervisor: {manifest.supervisor_kind}")
    click.echo(f"Scope:      {manifest.scope}")
    click.echo(f"Port:       {manifest.port}")
    click.echo(f"Status:     {runtime_status(manifest)}")
    click.echo(f"Healthy:    {'yes' if probe_ready(manifest.health_url) else 'no'}")
    if payload and isinstance(payload, dict):
        click.echo(f"Health URL: {manifest.health_url.replace('/readyz', '/health')}")
        # `config` may be a non-dict (null / string / list) if a different or
        # older service is answering on the port. `payload.get('config', {})`
        # only defaults on a MISSING key, so a present-but-non-dict value would
        # reach `.get('backend', ...)` and crash with AttributeError. Guard on
        # isinstance, mirroring wrap.py's _proxy_health_config.
        config = payload.get("config")
        if not isinstance(config, dict):
            config = {}
        click.echo(f"Backend:    {config.get('backend', manifest.backend)}")


@install.command("start")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_start(profile: str) -> None:
    """Start a persistent deployment."""

    manifest = _require_manifest(profile)
    _reject_task_lifecycle(manifest, "start")
    if not probe_ready(manifest.health_url):
        _deactivate_deployment_mutations(manifest)
    _start_deployment(manifest)
    if probe_ready(manifest.health_url) and not manifest.mutations:
        _activate_deployment_mutations(manifest)
    click.echo(f"Started deployment '{profile}'.")


@install.command("stop")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_stop(profile: str) -> None:
    """Stop a persistent deployment."""

    manifest = _require_manifest(profile)
    _reject_task_lifecycle(manifest, "stop")
    _deactivate_deployment_mutations(manifest)
    _stop_deployment(manifest)
    click.echo(f"Stopped deployment '{profile}'.")


@install.command("restart")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_restart(profile: str) -> None:
    """Restart a persistent deployment."""

    manifest = _require_manifest(profile)
    _reject_task_lifecycle(manifest, "restart")
    _deactivate_deployment_mutations(manifest)
    _stop_deployment(manifest)
    _start_deployment(manifest)
    _activate_deployment_mutations(manifest)
    click.echo(f"Restarted deployment '{profile}'.")


@install.command("remove")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_remove(profile: str) -> None:
    """Remove a persistent deployment and undo managed config."""

    manifest = _require_manifest(profile)
    _deactivate_deployment_mutations(manifest, persist_manifest=False)
    try:
        if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            stop_supervisor(manifest)
    except Exception:
        pass
    try:
        stop_runtime(manifest)
    except Exception:
        pass
    try:
        remove_supervisor(manifest)
    except Exception:
        pass
    delete_manifest(profile)
    click.echo(f"Removed deployment '{profile}'.")


@install.group("agent", hidden=True)
def install_agent() -> None:
    """Hidden runtime helpers used by persistent supervisors."""


@install_agent.command("run")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_agent_run(profile: str) -> None:
    """Run the persistent runtime in the foreground."""

    manifest = _require_manifest(profile)
    raise SystemExit(run_foreground(manifest))


_STARTUP_READY_TIMEOUT_SECONDS = 15


@install_agent.command("ensure")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_agent_ensure(profile: str) -> None:
    """Ensure a persistent deployment is healthy, starting it when needed."""

    manifest = _require_manifest(profile)
    if probe_ready(manifest.health_url):
        click.echo(f"Deployment '{profile}' is already healthy.")
        return
    with acquire_runtime_start_lock(manifest.profile) as acquired:
        if not acquired:
            click.echo(f"Deployment '{profile}' start is already in progress.")
            return
        # Double-check after acquiring the lock — another ensure may have
        # started the runtime while we waited for the lock.
        if probe_ready(manifest.health_url):
            click.echo(f"Deployment '{profile}' is already healthy.")
            return
        if runtime_status(manifest) == "running":
            # Runtime exists but isn't ready yet — give it a grace period
            # before deciding it's wedged and restarting.
            if wait_ready(manifest, timeout_seconds=_STARTUP_READY_TIMEOUT_SECONDS):
                click.echo(f"Deployment '{profile}' is healthy.")
                return
            _deactivate_deployment_mutations(manifest)
            stop_runtime(manifest)
        else:
            _deactivate_deployment_mutations(manifest)
        _start_deployment(manifest, assume_start_lock=True)
        _activate_deployment_mutations(manifest)
    click.echo(f"Deployment '{profile}' is healthy.")
