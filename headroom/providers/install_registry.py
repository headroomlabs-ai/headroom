"""Install-time provider registry helpers."""

from __future__ import annotations

from collections.abc import Callable

from headroom.install.models import DeploymentManifest, ManagedMutation
from headroom.providers.aider.install import build_install_env as _build_aider_install_env
from headroom.providers.claude.install import (
    apply_provider_scope as _apply_claude_provider_scope,
)
from headroom.providers.claude.install import (
    build_install_env as _build_claude_install_env,
)
from headroom.providers.claude.install import (
    revert_provider_scope as _revert_claude_provider_scope,
)
from headroom.providers.codex.install import (
    apply_provider_scope as _apply_codex_provider_scope,
)
from headroom.providers.codex.install import build_install_env as _build_codex_install_env
from headroom.providers.codex.install import (
    revert_provider_scope as _revert_codex_provider_scope,
)
from headroom.providers.copilot.install import (
    build_install_env as _build_copilot_install_env,
)
from headroom.providers.cortex_code.install import (
    build_install_env as _build_cortex_code_install_env,
)
from headroom.providers.cursor.install import build_install_env as _build_cursor_install_env
from headroom.providers.openclaw.install import (
    apply_provider_scope as _apply_openclaw_provider_scope,
)
from headroom.providers.openclaw.install import (
    revert_provider_scope as _revert_openclaw_provider_scope,
)
from headroom.providers.opencode.install import (
    apply_provider_scope as _apply_opencode_provider_scope,
)
from headroom.providers.opencode.install import build_install_env as _build_opencode_install_env
from headroom.providers.opencode.install import (
    revert_provider_scope as _revert_opencode_provider_scope,
)

_InstallEnvBuilder = Callable[..., dict[str, str]]
_ProviderScopeApplier = Callable[[DeploymentManifest], ManagedMutation | None]
_ProviderScopeReverter = Callable[[ManagedMutation, DeploymentManifest], None]

_ENV_BUILDERS: dict[str, _InstallEnvBuilder] = {
    "claude": _build_claude_install_env,
    "copilot": _build_copilot_install_env,
    "codex": _build_codex_install_env,
    "aider": _build_aider_install_env,
    "cortex-code": _build_cortex_code_install_env,
    "cursor": _build_cursor_install_env,
    "opencode": _build_opencode_install_env,
}

_PROVIDER_SCOPE_HANDLERS: dict[str, tuple[_ProviderScopeApplier, _ProviderScopeReverter]] = {
    "claude": (_apply_claude_provider_scope, _revert_claude_provider_scope),
    "codex": (_apply_codex_provider_scope, _revert_codex_provider_scope),
    "openclaw": (_apply_openclaw_provider_scope, _revert_openclaw_provider_scope),
    "opencode": (_apply_opencode_provider_scope, _revert_opencode_provider_scope),
}


def build_install_target_envs(
    port: int, backend: str, targets: list[str]
) -> dict[str, dict[str, str]]:
    """Build per-target install environment values via provider slices."""
    target_envs: dict[str, dict[str, str]] = {}
    for target in targets:
        builder = _ENV_BUILDERS.get(target)
        if builder is None:
            continue
        target_envs[target] = builder(port=port, backend=backend)
    return target_envs


def apply_provider_scope_mutations(manifest: DeploymentManifest) -> list[ManagedMutation]:
    """Apply provider-scope mutations owned by provider slices."""
    mutations: list[ManagedMutation] = []
    for target in manifest.targets:
        handlers = _PROVIDER_SCOPE_HANDLERS.get(target)
        if handlers is None:
            continue
        mutation = handlers[0](manifest)
        if mutation is not None:
            mutations.append(mutation)
    return mutations


def revert_provider_scope_mutation(manifest: DeploymentManifest, mutation: ManagedMutation) -> None:
    """Revert a provider-scope mutation via the owning provider slice."""
    handlers = _PROVIDER_SCOPE_HANDLERS.get(mutation.target)
    if handlers is None:
        return
    handlers[1](mutation, manifest)
